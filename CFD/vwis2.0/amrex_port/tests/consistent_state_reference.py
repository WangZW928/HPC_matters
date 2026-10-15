#!/usr/bin/env python3
"""Affine streamfunction reference: evidence completion is not algorithm acceptance.

All production fields remain visible. The original stretched temporal and Phi
backend gates remain BLOCKED. The optional strict decision flag rejects those
known open contracts only after writing a complete diagnostic artifact.
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
from pressure_phi_reference import LOCK, difference, divergence, face_extend, operator, require


def decode(v):
    nx, ny, nz = v['n_cell']
    f = {}
    for name, values in v['fields'].items():
        shape = [nz, ny, nx]
        face = re.search(r'(?:Ucont(?:_old|_older)?|I_Ucat|Q|S|beta)_([012])$', name)
        if face:
            shape[2-int(face[1])] += 1
        components = 9 if name == 'grad_xi' else 3 if (name.startswith(('Q_', 'S_')) or
            (re.search(r'Ucat(?:_old|_older)?$', name) and '_DI_' not in name) or name == 'initial_rate') else 1
        data = np.asarray(values, dtype=float)
        require(data.size == math.prod(shape)*components and np.isfinite(data).all(), f'field shape/finite {name}')
        f[name] = data.reshape((components, *shape) if components > 1 else shape)
    require((f['cell_volume'] > 0).all(), 'positive volume')
    return f


def independent_DI(v, f, velocity):
    fluxes = [sum(f[f'S_{d}'][c]*face_extend((velocity[c]+np.roll(velocity[c], 1, axis=2-d))*.5, d)
                  for c in range(3)) for d in range(3)]
    return divergence(fluxes, f['cell_volume'])


def fft_multiplier(v, source):
    """Independent affine 11-point DG Fourier inverse; only constant datum set.

    No source mean is subtracted. The constant Fourier coefficient is validated
    as compatible before assigning the pressure constant coefficient to zero.
    DIC has additional checkerboard null modes on this even grid and is not
    silently substituted for DG.
    """
    sx, sy, sz = v['scale']
    eps = v['epsilon']
    h = v['spacing']
    nz, ny, nx = source.shape
    tx = 2*np.pi*np.fft.fftfreq(nx)[None, None, :]
    ty = 2*np.pi*np.fft.fftfreq(ny)[None, :, None]
    tz = 2*np.pi*np.fft.fftfreq(nz)[:, None, None]
    theta = [tx, ty, tz]
    diagonal = [(1+eps*eps)/sx**2, 1/sy**2, 1/sz**2]
    eig = sum(-4*q*np.sin(t/2)**2/hd**2 for q, t, hd in zip(diagonal, theta, h))
    eig = eig+2*eps*np.sin(tx)*np.sin(ty)/(sx*sy*h[0]*h[1])
    dic_eig = sum(-q*np.sin(t)**2/hd**2 for q, t, hd in zip(diagonal, theta, h))
    dic_eig = dic_eig+2*eps*np.sin(tx)*np.sin(ty)/(sx*sy*h[0]*h[1])
    shat = np.fft.fftn(source)
    require(abs(shat[0, 0, 0]) <= 1e-12*source.size*np.max(np.abs(source)), 'FFT source compatibility; no mean repair')
    mask = eig != 0
    require(np.count_nonzero(~mask) == 1, 'DG unexpected null modes')
    phat = np.zeros_like(shat)
    phat[mask] = shat[mask]/eig[mask]
    # Count analytically defined modes instead of thresholding the sin(pi) roundoff.
    dic_null_modes = math.prod(2 if n % 2 == 0 else 1 for n in source.shape)
    return np.fft.ifftn(phat).real, dic_null_modes, float(np.max(np.abs(eig-dic_eig)))


def inspect(v):
    require(v['schema'] == 'avwis-consistent-reference-v1' and v['status'] == 'OBSERVATION', 'schema')
    require(v['locked_sha'] == LOCK and v['amrex_version'] == '26.04', 'lock')
    require(v['stretch'] == [0, 0, 0] and v['epsilon'] != 0 and 1 <= v['steps'] <= 64, 'affine scope')
    require(v['active_ranks'] == min(v['ranks'], v['boxes']), 'active ranks')
    f = decode(v)
    vol = f['cell_volume']
    initial_constraints = {k: float(np.max(np.abs(f[k]))) for k in ('initial_DI_Ucat', 'initial_D_Ucont')}
    require(all(x <= 1e-10 for x in initial_constraints.values()), 'consistent initial constraint')
    initial_faces_exact = all(np.array_equal(f[f'initial_I_Ucat_{d}'], f[f'initial_Ucont_{d}']) for d in range(3))
    require(initial_faces_exact, 'initial cell authority and face consistency')
    nx, ny, _ = v['n_cell']
    _, j, i = np.indices(vol.shape)
    x, y = 2*np.pi*(i+0.5)*v['spacing'][0], 2*np.pi*(j+0.5)*v['spacing'][1]
    a = .04*np.sin(x)*np.cos(y)*np.sin(2*np.pi*v['spacing'][1])/v['spacing'][1]
    b = -.04*np.cos(x)*np.sin(y)*np.sin(2*np.pi*v['spacing'][0])/v['spacing'][0]
    sx, sy, _ = v['scale']
    expected = np.array([sx*a+v['epsilon']*sy*b, sy*b, .1*np.sin(x+y)])
    initial_velocity = difference(f['initial_Ucat'], expected)
    require(initial_velocity['gate'] == 'PASS', 'independent streamfunction velocity')
    initial_DI_reference = difference(f['initial_DI_Ucat'], independent_DI(v, f, f['initial_Ucat']))
    source_reference = difference(f['initial_pressure_source'], independent_DI(v, f, f['initial_rate']))
    # Near-zero DI still uses the original relative comparator and may be
    # BLOCKED under CUDA FMA rounding; retain that diagnostic without relaxing it.
    require(source_reference['gate'] == 'PASS', 'independent pressure source')
    fft, dic_null_modes, symbol_gap = fft_multiplier(v, f['initial_pressure_source'])
    pressure_reference = difference(f['initial_multiplier'], fft, vol)
    require(pressure_reference['gate'] == 'PASS', 'independent DG Fourier pressure reference')
    require(np.max(np.abs(fft)) > 1e-3, 'nonzero multiplier reference')
    action_checks = {}
    for prefix, scalar in [('initial_multiplier', 'initial_multiplier'), ('final_phi', 'final_Phi')]:
        for dic in (False, True):
            key = prefix+('_DIC' if dic else '_DG')
            action_checks[key] = difference(f[key], operator(v, f, f[scalar], dic=dic))
            require(action_checks[key]['gate'] == 'PASS', f'independent action {key}')
    source = f['initial_pressure_source']
    dg_residual = f['initial_multiplier_DG']-source
    dic_residual = f['initial_multiplier_DIC']-source
    dg_relative = float(np.max(np.abs(dg_residual))/np.max(np.abs(source)))
    require(dg_relative <= 1e-11, 'DG source residual')
    first_identity = difference(f['first_DI_Ucat'], f['initial_DI_Ucat']+
        v['dt']*(source-operator(v, f, f['first_Phi'], dic=True)))
    logical_gradient = [(np.roll(f['first_Phi'], -1, axis=2-d)-np.roll(f['first_Phi'], 1, axis=2-d))/(2*v['spacing'][d]) for d in range(3)]
    cell_gradient = np.array([sum(f['grad_xi'][3*d+c]*logical_gradient[d] for d in range(3)) for c in range(3)])
    velocity_identity = difference(f['first_Ucat'], f['initial_Ucat']+v['dt']*(f['initial_rate']-cell_gradient))
    require(velocity_identity['gate'] == 'PASS', 'first full velocity update identity')
    first_pressure = difference(f['first_Phi'], fft, vol)
    for r in v['projections']:
        require(0 <= r['after'] <= 1e-10 and 0 <= r['full_relative'] <= 1e-11, 'per-step projection gates')
    require(len(v['projections']) == v['steps'], 'projection count')
    required_fields = {'Ucat', 'Ucat_old', 'Ucat_older', 'P', 'Phi', 'Nvert'} | {
        f'Ucont{layer}_{d}' for layer in ('', '_old', '_older') for d in range(3)}
    require(set(v['direct_changes']) == required_fields == set(v['failed_changes']), 'complete transaction field set')
    direct_flow = max(values[0] for name, values in v['direct_changes'].items() if name != 'Phi')
    direct_halo = max(values[1] for name, values in v['direct_changes'].items() if name != 'Phi')
    for prefix in ('direct', 'failed'):
        for name in required_fields:
            observed = float(np.max(np.abs(f[f'{prefix}_after_{name}']-f[f'{prefix}_before_{name}'])))
            require(observed == v[f'{prefix}_changes'][name][0], f'{prefix} canonical exact change {name}')
    before, after = v['failed_metadata_before'], v['failed_metadata_after']
    require(before[:3] == after[:3] and before[6] == after[6], 'failed advance changed counters or last-success timing')
    require(v['failure_reason'].startswith('AVWiS deferred pressure exhausted max_iterations=1'), 'precise failure reason')
    require(v['failed_changes']['P'] == [0, 0] and v['failed_changes']['Nvert'] == [0, 0], 'failed pressure/mask changed')
    require(v['failed_changes']['Ucat'][0] > 0 and v['failed_changes']['Ucat_old'][0] > 0 and
            v['failed_changes']['Ucat_older'][0] > 0 and after[3] > before[3], 'missing actual advance failure witness')
    for d in range(3):
        require(v['failed_changes'][f'Ucont_{d}'][0] > 0, 'missing failed face predictor witness')
    datum = float(np.sum(f['initial_multiplier']*vol)/np.sum(vol))
    require(abs(datum) <= 1e-12, 'reference pressure datum')
    return {'initial_constraints': initial_constraints, 'initial_faces_exact': True,
            'independent_streamfunction': initial_velocity, 'independent_initial_DI': initial_DI_reference, 'independent_pressure_source': source_reference, 'DG_Fourier_reference': pressure_reference,
            'independent_actions': action_checks, 'initial_DG_source_relative_residual': dg_relative,
            'initial_DIC_source_relative_residual': float(np.max(np.abs(dic_residual))/np.max(np.abs(source))),
            'DG_DIC_symbol_max_gap': symbol_gap, 'DIC_checkerboard_null_modes': dic_null_modes,
            'first_DI_identity': first_identity, 'first_velocity_identity': velocity_identity, 'first_Phi_vs_initial_multiplier': first_pressure,
            'first_Phi_max': float(np.max(np.abs(f['first_Phi']))), 'reference_datum': datum,
            'final_DI_Ucat_max': float(np.max(np.abs(f['final_DI_Ucat']))),
            'final_D_Ucont_max': float(np.max(np.abs(f['final_D_Ucont']))),
            'direct_face_projection': {'flow_valid_change': direct_flow, 'flow_including_ghost_change': direct_halo,
                'Phi_change': v['direct_changes']['Phi'], 'full_registered_state_gate': 'PASS' if max(x[1] for x in v['direct_changes'].values()) <= 1e-12 else 'BLOCKED', 'gate': 'PASS' if direct_flow <= 1e-12 and direct_halo <= 1e-12 else 'BLOCKED'},
            'reinterpolation_projection': {'before': v['reinterpolated_divergence_before'],
                'after': v['reinterpolated_divergence_after'], 'gate': 'BLOCKED' if v['reinterpolated_divergence_before'] > 1e-10 else 'PASS'},
            'entire_advance_rollback': {'gate': 'BLOCKED', 'rejection_observed': 'PASS',
                'changes': v['failed_changes'], 'metadata_before': before, 'metadata_after': after, 'reason': v['failure_reason']}}


def launch(exe, input_path, work, name, steps, dt, ranks=1, grid=8, extra=(), reject=None):
    path = work/f'{name}.json'
    cmd = [str(Path(exe).resolve()), str(Path(input_path).resolve()), 'avwis.consistent_reference=1',
           'avwis.mapping.stretch=0 0 0', f'avwis.run_steps={steps}', f'avwis.dt={dt:.17g}',
           f'avwis.metadata_file={path}', f'avwis.max_grid_size={grid}',
           'fabarray.mfiter_tile_size=4 4 2', *extra]
    if ranks > 1:
        cmd = ['mpiexec', '-n', str(ranks), *cmd]
    (work/f'{name}.command.json').write_text(json.dumps(cmd, indent=2)+'\n')
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
    try:
        log, _ = process.communicate(timeout=180)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            log, _ = process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            log, _ = process.communicate()
        (work/f'{name}.log').write_text(log)
        raise RuntimeError(f'{name} timed out')
    (work/f'{name}.log').write_text(log)
    if reject:
        require(process.returncode != 0 and reject in log, f'{name} accurate rejection missing')
        return {'gate': 'PASS', 'exit': process.returncode, 'reason': reject}
    require(process.returncode == 0, f'{name} exit {process.returncode}; see saved log')
    v = json.loads(path.read_text())
    require(v['ranks'] == ranks, 'runtime ranks')
    return v


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', '--exe', dest='cpu', required=True)
    parser.add_argument('--mpi')
    parser.add_argument('--cuda')
    parser.add_argument('--input', required=True)
    parser.add_argument('--work', required=True)
    parser.add_argument('--require-decision', action='store_true')
    args = parser.parse_args()
    Path(args.work).mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix='run-', dir=args.work)).resolve()
    evidence = {'status': 'INCOMPLETE', 'scope': 'affine shear consistent initial state; test-only reference',
                'original_pressure_temporal_gate': 'BLOCKED', 'original_fine_dt_Phi_gate': 'BLOCKED',
                'production_numerics_changed': False, 'runs': {}, 'backends': {}, 'rejections': {}}
    try:
        states = []
        for steps in (4, 8, 16):
            v = launch(args.cpu, args.input, work, f'cpu-{steps}', steps, .002/steps)
            evidence['runs'][f'cpu-{steps}'] = inspect(v)
            states.append(v)
            evidence['primary_backend'] = v['backend']
        norms = {}
        for name in ('final_Ucat', 'final_Ucont_0', 'final_Ucont_1', 'final_Ucont_2', 'final_P', 'final_Phi'):
            a = np.asarray(states[0]['fields'][name]);b = np.asarray(states[1]['fields'][name]);c = np.asarray(states[2]['fields'][name])
            coarse = float(np.linalg.norm(a-b));fine = float(np.linalg.norm(b-c))
            ratio = coarse/fine if fine else None
            norms[name] = {'coarse_difference_l2': coarse, 'fine_difference_l2': fine, 'ratio': ratio,
                           'gate': 'PASS' if ratio is not None and 1.8 <= ratio <= 2.2 else 'BLOCKED'}
        evidence['minimal_fixed_time_differences'] = norms
        first = [np.asarray(v['fields']['first_Phi']) for v in states]
        evidence['first_multiplier_dt_differences_l2'] = [float(np.linalg.norm(first[i]-first[i+1])) for i in range(2)]
        for name, exe, ranks, grid in [('cpu-layout16', args.cpu, 1, 16), ('mpi4', args.mpi, 4, 8), ('cuda', args.cuda, 1, 8)]:
            if not exe:
                evidence['backends'][name] = {'status': 'NOT TESTED'}
                continue
            peer = launch(exe, args.input, work, name, 16, .002/16, ranks, grid)
            evidence['runs'][name] = inspect(peer)
            base = states[2]
            require(set(base['fields']) == set(peer['fields']), 'backend full field set')
            comparisons = {k: difference(base['fields'][k], peer['fields'][k]) for k in base['fields']}
            evidence['backends'][name] = {'status': 'DIAGNOSTICS_COMPLETE', 'fields': comparisons,
                'diagnostic_field_gate': 'PASS' if all(x['gate'] == 'PASS' for x in comparisons.values()) else 'BLOCKED',
                'blocked_diagnostic_fields': [k for k, x in comparisons.items() if x['gate'] != 'PASS'],
                'transport_gate': 'PASS' if all(comparisons[k]['gate'] == 'PASS' for k in
                    ('final_Ucat', 'final_P', 'final_Phi', 'final_Ucont_0', 'final_Ucont_1', 'final_Ucont_2')) else 'BLOCKED'}
        cases = [('mutual-selectors', ['avwis.pressure_reference=1'], 'reference selectors are mutually exclusive'),
                 ('non-affine', ['avwis.mapping.stretch=0.1 0 0'], 'consistent reference requires nonzero affine shear'),
                 ('step-count', ['avwis.run_steps=65'], 'consistent reference requires nonzero affine shear')]
        for name, extra, reason in cases:
            evidence['rejections'][name] = launch(args.cpu, args.input, work, name, 4, .0005, extra=extra, reject=reason)
        evidence['status'] = 'DIAGNOSTICS_COMPLETE'
        evidence['mathematical_decision_gate'] = 'BLOCKED'
    except Exception as error:
        evidence['error'] = str(error)
        (work/'evidence.json').write_text(json.dumps(evidence, indent=2, allow_nan=False)+'\n')
        raise
    (work/'evidence.json').write_text(json.dumps(evidence, indent=2, allow_nan=False)+'\n')
    print(f'Diagnostics complete: {work}; pressure semantics, single authority and advance rollback BLOCKED')
    if args.require_decision:
        require(evidence['mathematical_decision_gate'] == 'PASS', 'mathematical decision BLOCKED; evidence preserved')


if __name__ == '__main__':
    main()
