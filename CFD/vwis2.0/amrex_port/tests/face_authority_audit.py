#!/usr/bin/env python3
"""Audit actual face-authoritative advances and preserve every numerical gate.

Independent NumPy I, DG and R act on canonical valid snapshots. Backend
comparisons retain the original relative L2 <= 1e-12 even for near-zero fields.
Evidence collection may finish with BLOCKED; --require-decision then fails.
P and Phi are discrete multipliers; no physical-pressure claim is made.
"""
import argparse
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import tempfile

import numpy as np
from pressure_phi_reference import LOCK, difference, divergence, face_extend, require


REGISTERED = {'Ucat', 'Ucat_old', 'Ucat_older', 'P', 'Phi', 'Nvert'} | {
    f'Ucont{layer}_{d}' for layer in ('', '_old', '_older') for d in range(3)}


def decode(v):
    nx, ny, nz = v['n_cell']
    result = {}
    for name, values in v['fields'].items():
        shape = [nz, ny, nx]
        face = re.search(r'(?:Ucont(?:_old|_older)?|Q|S|beta)_([012])$', name)
        if face:
            shape[2-int(face[1])] += 1
        components = (9 if name == 'grad_xi' else 3 if name.startswith(('Q_', 'S_')) or
                      re.search(r'Ucat(?:_old|_older)?$', name) or name.endswith('_rate') else 1)
        a = np.asarray(values, dtype=float)
        require(a.size == math.prod(shape)*components and np.isfinite(a).all(),
                f'canonical shape/finite {name}')
        result[name] = a.reshape((components, *shape) if components > 1 else shape)
    require((result['cell_volume'] > 0).all(), 'positive volumes')
    return result


def interpolate(f, velocity):
    return [sum(f[f'S_{d}'][c]*face_extend(
        (velocity[c]+np.roll(velocity[c], 1, axis=2-d))*.5, d)
        for c in range(3)) for d in range(3)]


def gradient_flux(v, f, scalar):
    """Full affine DG face gradient including cross terms, serial and periodic."""
    fluxes = []
    for d in range(3):
        left = np.roll(scalar, 1, axis=2-d)
        q = f[f'Q_{d}']
        flux = q[d]*face_extend((scalar-left)/v['spacing'][d], d)
        for t in range(3):
            if t != d:
                tangent = (np.roll(scalar, -1, axis=2-t)-np.roll(scalar, 1, axis=2-t)
                           + np.roll(left, -1, axis=2-t)-np.roll(left, 1, axis=2-t))/(4*v['spacing'][t])
                flux += q[t]*face_extend(tangent, d)
        fluxes.append(f[f'beta_{d}']*flux)
    return fluxes


def recover(v, f, faces):
    """Independent triangular six-face reconstruction, including shear."""
    u = []
    for d in range(3):
        normal = faces[d]/f[f'S_{d}'][d]
        axis = 2-d
        lo = np.take(normal, np.arange(normal.shape[axis]-1), axis=axis)
        hi = np.take(normal, np.arange(1, normal.shape[axis]), axis=axis)
        u.append((lo+hi)*.5)
    u[0] += v['epsilon']*u[1]
    return np.asarray(u)


def inspect(v):
    require(v['schema'] == 'avwis-face-authority-v1' and v['status'] == 'OBSERVATION', 'schema')
    require(v['locked_sha'] == LOCK and v['amrex_version'] == '26.04', 'dependency lock')
    require(v['state_contract'] == 'face_authoritative' and v['pressure_contract'] == 'instantaneous_pressure', 'authority pair')
    require(v['stretch'] == [0, 0, 0] and v['epsilon'] != 0, 'affine scope')
    require(v['active_ranks'] == min(v['ranks'], v['boxes']), 'active rank ownership')
    require(set(v['registered_fields']) == REGISTERED and len(v['registered_fields']) == 15, 'registered field coverage')
    f = decode(v)
    for prefix in ('initial', 'final', 'perturb_before', 'perturb_base', 'perturb_after', 'failed_before', 'failed_after'):
        require({prefix+'_'+name for name in REGISTERED} <= set(f), f'incomplete {prefix} state')
    require(len(v['projections']) == v['steps'], 'actual projection count')
    for p in v['projections']:
        require(p['before'] > 1e-8 and p['after'] <= 1e-10 and p['residual'] <= 1e-11 and
                p['outer'] > 0 and p['inner'] > 0, 'original nontrivial projection gates')
    require(v['initial_projection']['before'] > 1e-3 and v['initial_projection']['after'] <= 1e-10 and
            v['initial_projection']['residual'] <= 1e-11, 'initial projection gate')
    require(v['history_error'] == 0 and v['R_error'] == 0 and v['momentum_drift'] <= 1e-12 and
            v['predictor_correction_error'] <= 1e-12, 'production state invariants')
    require(v['max_cfl'] <= 1 and v['max_diffusion'] <= 1, 'explicit stability')
    initial, final = v['initial_metadata'], v['final_metadata']
    require(final[1] == initial[1]+v['steps'] and final[2] == 3 and final[3] == final[4] and
            abs(final[0]-initial[0]-v['steps']*v['dt']) <= 1e-13, 'time/history/epoch metadata')
    initial_momentum = np.sum(f['initial_Ucat']*f['cell_volume'], axis=(1, 2, 3))
    final_momentum = np.sum(f['final_Ucat']*f['cell_volume'], axis=(1, 2, 3))
    require(np.max(np.abs(final_momentum-initial_momentum)) <= 1e-12, 'independent volume momentum')
    identities = {}
    for prefix in [f'step{n}' for n in range(v['steps'])]+['force_probe']:
        before = [f[f'{prefix}_before_Ucont_{d}'] for d in range(3)]
        predictor = [f[f'{prefix}_predictor_Ucont_{d}'] for d in range(3)]
        after = [f[f'{prefix}_after_Ucont_{d}'] for d in range(3)]
        interpolated = interpolate(f, f[prefix+'_rate'])
        g = gradient_flux(v, f, f[prefix+'_Phi'])
        predictor_checks = {str(d): difference(predictor[d], before[d]+v['dt']*interpolated[d]) for d in range(3)}
        correction_checks = {str(d): difference(after[d], before[d]+v['dt']*(interpolated[d]-g[d])) for d in range(3)}
        recovery_check = difference(f[prefix+'_Ucat'], recover(v, f, after))
        require(np.array_equal(f[prefix+'_P'], f[prefix+'_Phi']), 'exact instantaneous P=Phi')
        require(np.max(np.abs(f[prefix+'_Phi'])) > 1e-6, 'nonzero multiplier')
        require(np.max(np.abs(divergence(after, f['cell_volume']))) <= 1e-10, 'independent projected divergence')
        source = divergence(predictor, f['cell_volume'])/v['dt']
        residual = source-divergence(g, f['cell_volume'])
        relative_residual = float(np.max(np.abs(residual))/np.max(np.abs(source)))
        actual_momentum = np.sum(f[prefix+'_Ucat']*f['cell_volume'], axis=(1, 2, 3))
        expected_momentum = initial_momentum if prefix != 'force_probe' else (
            final_momentum+v['dt']*np.asarray(v['force_probe_body_force'])*np.sum(f['cell_volume']))
        momentum_error = float(np.max(np.abs(actual_momentum-expected_momentum)))
        require(momentum_error <= 1e-12, 'independent per-step volume momentum and nonzero force')
        identities[prefix] = {'I_predictor': predictor_checks, 'DG_correction': correction_checks,
                              'R': recovery_check, 'P_equals_Phi_exact': True,
                              'volume_momentum_error': momentum_error,
                              'full_DG_relative_residual': relative_residual,
                              'full_DG_residual_gate': 'PASS' if relative_residual <= 1e-11 else 'BLOCKED',
                              'gate': 'PASS' if relative_residual <= 1e-11 and all(x['gate'] == 'PASS' for x in
                                  [*predictor_checks.values(), *correction_checks.values(), recovery_check]) else 'BLOCKED'}
    # Verify all registered fields; include all histories and Phi/Nvert even
    # where they are near zero. The perturbed old pressure itself is nontrivial.
    require(np.max(np.abs(f['perturb_before_P']-f['final_P'])) > 1, 'old pressure perturbation witness')
    for name in REGISTERED:
        require(np.array_equal(f['perturb_base_'+name], f['perturb_after_'+name]) and
                v['perturb_changes'][name] == [0, 0], f'old P influenced {name}')
    before, after = v['failed_metadata_before'], v['failed_metadata_after']
    require(before[:3] == after[:3] and before[6] == after[6], 'failed time/step/history-depth/timing commit')
    require('deferred pressure exhausted max_iterations=1' in v['failure_reason'], 'precise actual failure')
    require(set(v['failed_changes']) == REGISTERED, 'complete failed state change norms')
    for name in REGISTERED:
        observed = float(np.max(np.abs(f['failed_after_'+name]-f['failed_before_'+name])))
        require(observed == v['failed_changes'][name][0], f'failed canonical change {name}')
    for name in ('P', 'Ucat', 'Nvert'):
        require(v['failed_changes'][name] == [0, 0], f'failed pressure/R/mask committed {name}')
    for d in range(3):
        require(v['failed_changes'][f'Ucont_{d}'][0] > 0, 'real failed face predictor witness')
    return {'invariants': 'PASS', 'independent_full_field_identities': identities,
            'identity_gate': 'PASS' if all(x['gate'] == 'PASS' for x in identities.values()) else 'BLOCKED',
            'old_P_independence_all_state_exact': 'PASS', 'instantaneous_P_equals_Phi_exact': 'PASS',
            'volume_momentum_max_drift': float(np.max(np.abs(final_momentum-initial_momentum))),
            'failed_pressure_R_counter_commit_gate': 'PASS',
            'entire_advance_rollback': {'gate': 'BLOCKED', 'contract': v['failure_contract'],
                'changes': v['failed_changes'], 'metadata_before': before, 'metadata_after': after},
            'physical_pressure_convergence': 'NOT_ESTABLISHED'}


def launch(exe, input_path, work, name, steps, dt, ranks=1, grid=8, extra=(), reject=None,
           mpi_exec='mpiexec', timeout=180):
    path = work/f'{name}.json'
    cmd = [str(Path(exe).resolve()), str(Path(input_path).resolve()), 'avwis.face_authority_reference=1',
           'avwis.state_contract=face_authoritative', 'avwis.pressure_contract=instantaneous_pressure',
           'avwis.mapping.stretch=0 0 0', f'avwis.run_steps={steps}', f'avwis.dt={dt:.17g}',
           f'avwis.metadata_file={path}', f'avwis.max_grid_size={grid}',
           'fabarray.mfiter_tile_size=4 4 2', *extra]
    if ranks > 1:
        cmd = [mpi_exec, '-n', str(ranks), *cmd]
    (work/f'{name}.command.json').write_text(json.dumps(cmd, indent=2)+'\n')
    process = subprocess.Popen(cmd, cwd=work, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, start_new_session=True)
    try:
        log, _ = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            log, _ = process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            log, _ = process.communicate()
        (work/f'{name}.log').write_text(log)
        raise RuntimeError(f'{name} exceeded {timeout}s; see saved log')
    (work/f'{name}.log').write_text(log)
    if reject is not None:
        require(process.returncode != 0 and reject in log, f'{name}: precise rejection {reject!r} missing')
        return {'gate': 'PASS', 'exit': process.returncode, 'reason': reject}
    require(process.returncode == 0, f'{name} exit {process.returncode}; see saved log')
    v = json.loads(path.read_text())
    require(v['ranks'] == ranks and v['steps'] == steps and abs(v['dt']-dt) <= 1e-18, 'actual run parameters')
    return v


def temporal(states):
    result = {}
    for name in ('Ucat', 'Ucont_0', 'Ucont_1', 'Ucont_2', 'Phi', 'P'):
        a, b, c = [np.asarray(v['fields']['final_'+name]) for v in states]
        coarse, fine = float(np.linalg.norm(a-b)), float(np.linalg.norm(b-c))
        ratio = coarse/fine if fine > 0 else None
        result[name] = {'dt_dt2_l2': coarse, 'dt2_dt4_l2': fine, 'ratio': ratio,
                        'observed_order': math.log2(ratio) if ratio is not None and ratio > 0 else None,
                        'gate': 'PASS' if coarse > 1e-14 and fine > 1e-14 and ratio is not None and 1.8 <= ratio <= 2.2 else 'BLOCKED'}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    executable = parser.add_mutually_exclusive_group(required=True)
    executable.add_argument('--cpu', help='CPU baseline for the full backend matrix')
    executable.add_argument('--exe', help='Single executable (also runs its alternate CPU layout when CPU)')
    parser.add_argument('--backend', choices=('CPU', 'CUDA'), default='CPU')
    parser.add_argument('--mpi')
    parser.add_argument('--cuda')
    parser.add_argument('--mpi-exec', default='mpiexec')
    parser.add_argument('--input', required=True)
    parser.add_argument('--work', required=True)
    parser.add_argument('--steps', type=int, default=4)
    parser.add_argument('--final-time', type=float, default=.002)
    parser.add_argument('--timeout', type=float, default=180)
    parser.add_argument('--require-decision', '--strict', action='store_true')
    args = parser.parse_args()
    require(1 <= args.steps <= 1024 and args.final_time > 0 and math.isfinite(args.final_time) and
            args.timeout > 0 and math.isfinite(args.timeout), 'step/time/timeout range')
    Path(args.work).mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix='run-', dir=args.work)).resolve()
    evidence = {'status': 'INCOMPLETE', 'scope': 'nonzero affine periodic face authority; explicit Euler',
                'runs': {}, 'backends': {}, 'rejections': {}, 'physical_pressure_convergence': 'NOT_ESTABLISHED'}
    exe = args.cpu or args.exe
    backend = 'CPU' if args.cpu else args.backend
    try:
        states = []
        for count in (args.steps, 2*args.steps, 4*args.steps):
            name = f'{backend.lower()}-{count}'
            v = launch(exe, args.input, work, name, count, args.final_time/count, timeout=args.timeout)
            require(v['backend'] == backend, 'actual backend')
            evidence['runs'][name] = inspect(v);states.append(v)
        require(all(states[0]['fields']['initial_'+k] == s['fields']['initial_'+k]
                    for s in states[1:] for k in REGISTERED), 'initial state depends on dt')
        evidence['fixed_time'] = temporal(states)
        evidence['velocity_time_gate'] = 'PASS' if all(evidence['fixed_time'][k]['gate'] == 'PASS' for k in
            ('Ucat', 'Ucont_0', 'Ucont_1', 'Ucont_2')) else 'BLOCKED'
        evidence['multiplier_time_gate'] = 'PASS' if all(evidence['fixed_time'][k]['gate'] == 'PASS' for k in ('Phi', 'P')) else 'BLOCKED'
        peers = [('layout16', exe if backend == 'CPU' else None, 1, 16, backend),
                 ('mpi4', args.mpi, 4, 8, 'CPU'), ('cuda', args.cuda, 1, 8, 'CUDA')]
        for name, peer_exe, ranks, grid, peer_backend in peers:
            if not peer_exe:
                evidence['backends'][name] = {'status': 'NOT_TESTED', 'full_field_gate': 'BLOCKED'}
                continue
            peer = launch(peer_exe, args.input, work, name, 4*args.steps, args.final_time/(4*args.steps),
                          ranks, grid, mpi_exec=args.mpi_exec, timeout=args.timeout)
            require(peer['backend'] == peer_backend, 'peer actual backend')
            evidence['runs'][name] = inspect(peer)
            base = states[2]
            for key in ('schema', 'locked_sha', 'amrex_version', 'n_cell', 'scale', 'stretch', 'epsilon', 'dt', 'steps'):
                require(base[key] == peer[key], f'peer configuration {key}')
            require(set(base['fields']) == set(peer['fields']), 'peer complete field coverage')
            comparisons = {key: difference(base['fields'][key], peer['fields'][key]) for key in base['fields']}
            evidence['backends'][name] = {'status': 'DIAGNOSTICS_COMPLETE', 'fields': comparisons,
                'full_field_gate': 'PASS' if all(x['gate'] == 'PASS' for x in comparisons.values()) else 'BLOCKED',
                'blocked_fields': [k for k, x in comparisons.items() if x['gate'] != 'PASS']}
        cases = [
            ('non-affine', ['avwis.mapping.stretch=0.1 0 0'], 'requires nonzero affine analytic_shear'),
            ('zero-epsilon', ['avwis.mapping.epsilon=0'], 'requires nonzero affine analytic_shear'),
            ('dual-instantaneous', ['avwis.state_contract=experimental_dual_state'], 'requires pressure_contract=accumulated_correction'),
            ('face-accumulated', ['avwis.pressure_contract=accumulated_correction'], 'requires pressure_contract=instantaneous_pressure'),
            ('unknown-pressure', ['avwis.state_contract=experimental_dual_state', 'avwis.pressure_contract=unknown'], 'requires pressure_contract=accumulated_correction'),
            ('nonperiodic', ['avwis.is_periodic=1 1 0'], 'mapped non-periodic directions require an explicit boundary configuration'),
            ('selector-conflict', ['avwis.consistent_reference=1'], 'reference selectors are mutually exclusive'),
            ('rollback-claim', ['avwis.failure_contract=rollback'], 'failure_contract supports only fatal; entire-advance rollback is not implemented'),
            ('unsupported-mode', ['avwis.mode=contract_c3_shear'], 'face_authoritative supports solve or contract_c3_temporal'),
            ('checkpoint', ['avwis.mode=solve', 'avwis.run_steps=0', 'avwis.checkpoint_file='+str(work/'unsupported-checkpoint')], 'analytic_shear checkpoint/restart provenance is not implemented'),
        ]
        for name, extra, reason in cases:
            evidence['rejections'][name] = launch(exe, args.input, work, name, args.steps, args.final_time/args.steps,
                                                  extra=extra, reject=reason, timeout=args.timeout)
        evidence['status'] = 'DIAGNOSTICS_COMPLETE'
        numerical = [v['identity_gate'] for v in evidence['runs'].values()]+[
            evidence['velocity_time_gate'], evidence['multiplier_time_gate']]+[
            v['full_field_gate'] for v in evidence['backends'].values()]
        evidence['decision_gate'] = 'PASS' if all(x == 'PASS' for x in numerical) else 'BLOCKED'
        evidence['entire_advance_rollback_gate'] = 'BLOCKED'
        evidence['decision_scope'] = 'face-state, multiplier time behavior and full backend fields; fatal failure allows predictor/history mutation'
    except Exception as error:
        evidence['error'] = str(error)
        (work/'evidence.json').write_text(json.dumps(evidence, indent=2, allow_nan=False)+'\n')
        raise
    (work/'evidence.json').write_text(json.dumps(evidence, indent=2, allow_nan=False)+'\n')
    print(f"Diagnostics complete: {work}; velocity time {evidence['velocity_time_gate']}; "
          f"DG multiplier time {evidence['multiplier_time_gate']}; decision {evidence['decision_gate']}; "
          "physical pressure convergence NOT_ESTABLISHED; entire-advance rollback BLOCKED")
    if args.require_decision:
        require(evidence['decision_gate'] == 'PASS', 'face authority decision BLOCKED; complete evidence preserved')


if __name__ == '__main__':
    main()
