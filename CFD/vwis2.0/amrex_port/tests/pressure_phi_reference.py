#!/usr/bin/env python3
"""Observe pressure semantics and isolate Phi RHS/layout/solver differences.

Needs NumPy only for an independent serial array operator. Numerical BLOCKED
results are saved, never converted to algorithm PASS. --require-backend makes
any isolated backend failure fatal after writing evidence. No long time gate.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import tempfile

import numpy as np

LOCK = '9219ba416b7ba2073dd1b12bf19fdce27391f17b'


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def arrays(v):
    nx, ny, nz = v['n_cell']
    cells = nx*ny*nz
    result = {}
    for key, data in v['fields'].items():
        a = np.asarray(data, dtype=float)
        require(np.isfinite(a).all(), f'nonfinite {key}')
        shape = [nz, ny, nx]
        if key.startswith(('Q_', 'S_', 'beta_', 'step_Ucont_')):
            d = int(key.rsplit('_', 1)[1])
            shape[2-d] += 1
        size = math.prod(shape)
        require(a.size % size == 0, f'field shape {key}')
        comps = a.size//size
        expected = 9 if key == 'grad_xi' else 3 if key.startswith(('Q_', 'S_')) or key in ('initial_Ucat', 'reference_Ucat', 'predictor_Ucat', 'step_Ucat', 'perturbed_P_step_Ucat') else 1
        require(comps == expected, f'component count {key}')
        result[key] = a.reshape((comps, *shape)) if comps > 1 else a.reshape(shape)
    require(result['cell_volume'].size == cells and (result['cell_volume'] > 0).all(), 'volumes')
    return result


def face_extend(a, d):
    axis = 2-d
    return np.concatenate((a, np.take(a, [0], axis=axis)), axis=axis)


def divergence(fluxes, volume):
    return sum(np.diff(f, axis=2-d) for d, f in enumerate(fluxes))/volume


def operator(v, f, p, part='full', dic=False):
    """Independent canonical serial DG / DIC; no production calls or mean repair."""
    h = v['spacing']
    vol = f['cell_volume']
    if dic:
        logical = [(np.roll(p, -1, axis=2-d)-np.roll(p, 1, axis=2-d))/(2*h[d]) for d in range(3)]
        g = f['grad_xi']
        u = [sum(g[3*d+c]*logical[d] for d in range(3)) for c in range(3)]
        fluxes = [sum(f[f'S_{d}'][c]*face_extend((u[c]+np.roll(u[c], 1, axis=2-d))*0.5, d)
                      for c in range(3)) for d in range(3)]
    else:
        fluxes = []
        for d in range(3):
            axis = 2-d
            left = np.roll(p, 1, axis=axis)
            q = f[f'Q_{d}']
            flux = q[d]*face_extend((p-left)/h[d], d) if part != 'cross' else np.zeros_like(q[d])
            if part != 'diag':
                for t in range(3):
                    if t == d:
                        continue
                    tangent = (np.roll(p, -1, axis=2-t)-np.roll(p, 1, axis=2-t)
                               + np.roll(left, -1, axis=2-t)-np.roll(left, 1, axis=2-t))/(4*h[t])
                    flux += q[t]*face_extend(tangent, d)
            fluxes.append(f[f'beta_{d}']*flux)
    return divergence(fluxes, vol)


def difference(a, b, volumes=None):
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    require(a.size == b.size and np.isfinite(a).all() and np.isfinite(b).all(), 'difference data')
    delta = a-b
    denom = max(float(np.linalg.norm(a)), float(np.linalg.norm(b)))
    relative = float(np.linalg.norm(delta))/denom if denom else float(np.linalg.norm(delta))
    pointwise = bool((np.abs(delta) <= 2e-10+2e-11*np.maximum(np.abs(a), np.abs(b))).all())
    result = {'relative_l2': relative, 'max_absolute': float(np.max(np.abs(delta))),
              'rms': float(np.sqrt(np.mean(delta*delta))),
              'gate': 'PASS' if relative <= 1e-12 and pointwise else 'BLOCKED'}
    if volumes is not None:
        w = np.asarray(volumes).ravel()
        mean = float(np.dot(delta, w)/np.sum(w))
        result.update(weighted_mean_difference=mean,
                      centered_difference_rms=float(np.sqrt(np.mean((delta-mean)**2))))
    return result


def inspect(v):
    require(v['schema'] == 'avwis-pressure-reference-v1' and v['status'] == 'OBSERVATION', 'schema')
    require(v['locked_sha'] == LOCK and v['amrex_version'] == '26.04', 'lock')
    require(v['active_ranks'] == min(v['ranks'], v['boxes']), 'active ranks')
    require(not v.get('solve_failed', False) and v['trace'], 'missing successful solver trace')
    def finite_tree(x):
        if isinstance(x, dict):
            return all(finite_tree(y) for y in x.values())
        if isinstance(x, list):
            return all(finite_tree(y) for y in x)
        return not isinstance(x, (int, float)) or math.isfinite(x)
    require(finite_tree(v), 'nonfinite metadata or trace')
    f = arrays(v)
    required = {'probe', 'probe_DG', 'probe_DG_diag', 'probe_DG_cross', 'probe_DIC',
                'initial_Phi', 'raw_divergence', 'initial_Ucat', 'initial_DI_Ucat', 'initial_D_Ucont',
                'reference_Ucat', 'reference_DI_Ucat', 'reference_D_Ucont', 'predictor_divergence',
                'formed_rhs', 'solve_rhs', 'solve_Phi', 'step_Ucat', 'step_P', 'step_Phi',
                'perturbed_P_step_Ucat', 'perturbed_P_step_Phi', 'perturbed_P_step_P', 'old_P_perturbation'}
    require(required <= set(f), 'missing required diagnostic field')
    op_checks = {}
    for prefix, pkey in [('probe', 'probe'), ('initial_phi', 'initial_Phi'), ('solve_phi', 'solve_Phi')]:
        for suffix, part, dic in [('DG', 'full', False), ('DG_diag', 'diag', False), ('DG_cross', 'cross', False), ('DIC', 'full', True)]:
            key = f'{prefix}_{suffix}'
            op_checks[key] = difference(f[key], operator(v, f, f[pkey], part, dic))
            require(op_checks[key]['gate'] == 'PASS', f'independent operator mismatch {key}: {op_checks[key]}')
    alias_identity = difference(f['initial_DI_Ucat'], f['raw_divergence']-f['initial_phi_DIC'])
    require(alias_identity['gate'] == 'PASS', 'initial DI identity')
    require(np.array_equal(f['step_Ucat'], f['perturbed_P_step_Ucat']) and
            np.array_equal(f['step_Phi'], f['perturbed_P_step_Phi']), 'old P changed actual predictor/solve')
    pressure_error = float(np.max(np.abs(f['perturbed_P_step_P']-f['old_P_perturbation']-f['step_Phi'])))
    require(pressure_error <= 1e-12, 'P += Phi identity')
    for key in ('initial_D_Ucont', 'step_D_Ucont'):
        require(np.max(np.abs(f[key])) <= 1e-10, f'mass gate {key}')
    require(v['trace'][-1]['full_relative'] <= 1e-11, 'full residual gate')
    for n, t in enumerate(v['trace'], 1):
        require(t['outer'] == n and 0 < t['inner'] <= 1600, 'trace iteration sequence')
        require(t['inner_initial'] >= 0 and 0 <= t['inner_final'] <= t['inner_initial'], 'inner residuals')
    computed_residual = f['solve_phi_DG']+f['solve_rhs']/f['cell_volume']
    computed_relative = float(np.max(np.abs(computed_residual))/np.max(np.abs(f['solve_rhs']/f['cell_volume'])))
    require(computed_relative <= 1e-11, 'captured full physical residual gate')
    require(abs(float(np.max(np.abs(f['predictor_divergence'])))-v['actual_step']['before']) <= 1e-12,
            'retained predictor differs from actual advance')
    vol = f['cell_volume']
    return {'independent_operator_checks': op_checks, 'initial_DI_identity': alias_identity,
            'initial_D_Ucont_max': float(np.max(np.abs(f['initial_D_Ucont']))),
            'initial_DI_Ucat_max': float(np.max(np.abs(f['initial_DI_Ucat']))),
            'DG_minus_DIC_max': float(np.max(np.abs(f['initial_phi_DG']-f['initial_phi_DIC']))),
            'predictor_divergence_max': float(np.max(np.abs(f['predictor_divergence']))),
            'old_P_velocity_Phi_exact': True, 'P_accumulation_error_max': pressure_error,
            'step_Phi_max': float(np.max(np.abs(f['step_Phi']))),
            'dt_step_Phi_max': v['dt']*float(np.max(np.abs(f['step_Phi']))),
            'solve_Phi_weighted_mean': float(np.sum(f['solve_Phi']*vol)/np.sum(vol)),
            'captured_full_relative_residual': computed_relative,
            'actual_step': v['actual_step'], 'trace': v['trace'],
            'formed_replay_Phi': difference(f['step_Phi'], f['solve_Phi'], vol) if not v['imported_rhs'] else None}


def run(exe, input_path, work, name, dt, grid=8, warmup=0, rhs=None, ranks=1):
    path = work/f'{name}.json'
    cmd = [str(Path(exe).resolve()), str(Path(input_path).resolve()), 'avwis.pressure_reference=1',
           f'avwis.metadata_file={path}', f'avwis.dt={dt:.17g}', f'avwis.max_grid_size={grid}',
           f'avwis.pressure_reference_warmup={warmup}', 'fabarray.mfiter_tile_size=4 4 2']
    if rhs:
        cmd.append(f'avwis.pressure_reference_rhs={rhs}')
    if ranks > 1:
        cmd = ['mpiexec', '-n', str(ranks), *cmd]
    (work/f'{name}.command.json').write_text(json.dumps(cmd, indent=2)+'\n')
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
    try:
        log, _ = proc.communicate(timeout=180)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            log, _ = proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            log, _ = proc.communicate()
        (work/f'{name}.log').write_text(log)
        raise RuntimeError(f'{name} timed out; evidence preserved')
    (work/f'{name}.log').write_text(log)
    require(proc.returncode == 0, f'{name}: exit {proc.returncode}; see log')
    v = json.loads(path.read_text())
    require(v['ranks'] == ranks, 'runtime ranks mismatch')
    return v


def peers(a, b):
    for k in ('n_cell', 'dt', 'warmup', 'epsilon', 'scale', 'stretch', 'spacing', 'locked_sha'):
        require(a[k] == b[k], f'peer metadata {k}')
    require(set(a['fields']) == set(b['fields']), 'peer field set')
    # All captured fields remain visible, including divergence and gauges. An
    # isolated action comparison does not replace the original temporal gate.
    result = {k: difference(a['fields'][k], b['fields'][k], a['fields']['cell_volume'] if k.endswith('Phi') else None)
              for k in a['fields']}
    return {'fields': result,
            'frozen_rhs_exact': a['fields']['solve_rhs'] == b['fields']['solve_rhs'],
            'operator_probe_gate': 'PASS' if all(result[k]['gate'] == 'PASS' for k in
                ('probe_DG', 'probe_DG_diag', 'probe_DG_cross', 'probe_DIC')) else 'BLOCKED',
            'frozen_Phi_gate': result['solve_Phi']['gate'],
            'actual_step_Phi_gate': result['step_Phi']['gate']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpu', required=True)
    parser.add_argument('--input', required=True)
    parser.add_argument('--work', required=True)
    parser.add_argument('--mpi')
    parser.add_argument('--cuda')
    parser.add_argument('--require-backend', action='store_true')
    args = parser.parse_args()
    root = Path(args.work).resolve(); root.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix='run-', dir=root))
    evidence = {'status': 'INCOMPLETE', 'work': str(work), 'pressure_time_gate': 'NOT_RUN',
                'original_pressure_time_gate': 'BLOCKED', 'original_fine_dt_backend_gate': 'BLOCKED',
                'runs': {}, 'peers': {}}
    try:
        snapshots = {}
        for n, dt in enumerate((2.5e-4, 1.25e-4, 6.25e-5)):
            name = f'cpu-dt{2**n}'
            v = run(args.cpu, args.input, work, name, dt)
            snapshots[name] = v; evidence['runs'][name] = inspect(v)
        dt = 6.25e-5
        fine = run(args.cpu, args.input, work, 'cpu-short16', dt, warmup=15)
        evidence['runs']['cpu-short16'] = inspect(fine)
        rhs = work/'frozen_rhs.txt'
        rhs.write_text('\n'.join(format(x, '.17g') for x in fine['fields']['formed_rhs'])+'\n')
        evidence['frozen_rhs_sha256'] = hashlib.sha256(rhs.read_bytes()).hexdigest()
        # Baseline is already the same formed RHS; import peers receives identical doubles.
        for name, exe, grid, ranks in [('cpu-layout16', args.cpu, 16, 1),
                                       ('mpi-one', args.mpi, 8, 1), ('mpi-four', args.mpi, 8, 4),
                                       ('cuda', args.cuda, 8, 1)]:
            if not exe:
                continue
            v = run(exe, args.input, work, name, dt, grid=grid, warmup=15, rhs=rhs, ranks=ranks)
            require(v['backend'] == ('CUDA' if name == 'cuda' else 'CPU'), f'{name} backend')
            evidence['runs'][name] = inspect(v)
            snapshots[name] = v
            evidence['peers'][name] = peers(fine, v)
            require(evidence['peers'][name]['frozen_rhs_exact'], 'frozen RHS changed')
        evidence['rhs_response'] = {}
        for name, peer in snapshots.items():
            if name not in evidence['peers'] or evidence['peers'][name]['actual_step_Phi_gate'] != 'BLOCKED':
                continue
            peer_rhs = work/f'{name}-formed-rhs.txt'
            peer_rhs.write_text('\n'.join(format(x, '.17g') for x in peer['fields']['formed_rhs'])+'\n')
            response = run(args.cpu, args.input, work, f'cpu-response-{name}', dt, rhs=peer_rhs)
            inspect(response)
            require(response['fields']['solve_rhs'] == peer['fields']['formed_rhs'], 'response RHS changed')
            evidence['rhs_response'][name] = difference(response['fields']['solve_Phi'], peer['fields']['step_Phi'], fine['fields']['cell_volume'])
        evidence['status'] = 'DIAGNOSTICS_COMPLETE'
        evidence['mpi_launcher_version'] = subprocess.check_output(['mpiexec', '--version'], text=True) if args.mpi else 'NOT_RUN'
        evidence['isolated_backend_gate'] = ('PASS' if all(p['frozen_Phi_gate'] == 'PASS' and p['operator_probe_gate'] == 'PASS'
                                                          and p['actual_step_Phi_gate'] == 'PASS' for p in evidence['peers'].values())
                                             else 'BLOCKED')
        require(bool(evidence['peers']), 'no layout/backend comparison')
    except Exception as error:
        evidence['error'] = str(error)
        (work/'evidence.json').write_text(json.dumps(evidence, indent=2, allow_nan=False)+'\n')
        raise
    (work/'evidence.json').write_text(json.dumps(evidence, indent=2, allow_nan=False)+'\n')
    print(f"Diagnostics complete: {work}; isolated backend {evidence['isolated_backend_gate']}; original gates remain BLOCKED")
    if args.require_backend:
        require(evidence['isolated_backend_gate'] == 'PASS', 'isolated backend BLOCKED; see saved evidence')


if __name__ == '__main__':
    main()
