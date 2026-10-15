#!/usr/bin/env python3
"""Audit real long shear advances; report velocity and pressure time gates separately."""
import argparse
import json
import math
import os
import signal
from pathlib import Path
import subprocess

LOCK_SHA = '9219ba416b7ba2073dd1b12bf19fdce27391f17b'


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def finite_tree(value):
    if isinstance(value, dict):
        return all(finite_tree(v) for v in value.values())
    if isinstance(value, list):
        return all(finite_tree(v) for v in value)
    return not isinstance(value, (int, float)) or math.isfinite(value)


def validate(value, steps, dt, ranks, backend):
    require(finite_tree(value), 'nonfinite evidence')
    require(value['schema'] == 'avwis-c3-temporal-v1' and
            value['invariant_status'] == 'PASS', 'snapshot schema/invariants')
    require(value['locked_sha'] == LOCK_SHA and value['amrex_version'] == '26.04', 'dependency provenance')
    require(value['ranks'] == ranks and value['backend'] == backend, 'actual rank/backend mismatch')
    require(value['active_ranks'] == min(ranks, value['boxes']), 'idle rank ownership')
    require(value['steps'] == steps and value['step'] == steps and value['history_depth'] == 3,
            'step/history metadata')
    require(abs(value['dt']-dt) <= 1e-18 and abs(value['time']-steps*dt) <= 1e-13, 'fixed end time')
    require(value['history_error'] == 0 and value['momentum_drift'] <= 1e-12 and
            value['mass_error'] <= 1e-12, 'history/momentum/mass gate')
    require(value['max_cfl'] <= 1 and value['max_diffusion'] <= 1, 'explicit stability guard')
    require(len(value['projections']) == steps, 'missing actual step diagnostics')
    for p in value['projections']:
        require(p['before'] > 1e-8 and p['after'] <= 1e-10 and p['residual'] <= 1e-11,
                'per-step nontrivial pressure/residual/divergence gate')
        if value['epsilon'] != 0:
            require(0 < p['outer'] <= 40 and p['inner'] > 0, 'deferred iterations')
    nx, ny, nz = value['n_cell']
    cells = nx*ny*nz
    shapes = {'initial_Ucat': 3*cells, 'Ucat': 3*cells, 'P': cells, 'Phi': cells,
              'divergence': cells, 'interpolated_divergence': cells, 'cell_volume': cells,
              'Ucont_0': (nx+1)*ny*nz, 'Ucont_1': nx*(ny+1)*nz, 'Ucont_2': nx*ny*(nz+1)}
    require(set(value['fields']) == set(shapes), 'field set')
    for key, length in shapes.items():
        require(len(value['fields'][key]) == length, f'{key}: truncated field')
    require(min(value['fields']['cell_volume']) > 0, 'positive volumes')
    require(max(abs(x) for x in value['fields']['divergence']) <= 1e-10, 'full-field divergence')


def weighted_rms(values, volumes):
    return math.sqrt(math.fsum(x*x*volumes[i % len(volumes)] for i, x in enumerate(values)) /
                     math.fsum(volumes))


def pressure_without_datum(value):
    p, volumes = value['fields']['P'], value['fields']['cell_volume']
    mean = math.fsum(x*v for x, v in zip(p, volumes))/math.fsum(volumes)
    return [x-mean for x in p]


def temporal_evidence(runs):
    evidence = {}
    for key in ('Ucat', 'Ucont_0', 'Ucont_1', 'Ucont_2', 'P'):
        values = [pressure_without_datum(v) if key == 'P' else v['fields'][key] for v in runs]
        errors = []
        for level in (0, 1):
            diff = [a-b for a, b in zip(values[level], values[level+1])]
            if key in ('P', 'Ucat'):
                errors.append(weighted_rms(diff, runs[level]['fields']['cell_volume']))
            else:
                errors.append(math.sqrt(math.fsum(x*x for x in diff)/len(diff)))
        require(min(errors) > 1e-14, f'{key}: temporal signal below comparison floor')
        ratio = errors[0]/errors[1]
        evidence[key] = {'dt_dt2_l2': errors[0], 'dt2_dt4_l2': errors[1],
                         'ratio': ratio, 'observed_order': math.log2(ratio),
                         'gate': 'PASS' if 1.8 <= ratio <= 2.2 else 'BLOCKED'}
    volumes = runs[0]['fields']['cell_volume']
    evidence['pressure_l2'] = [weighted_rms(pressure_without_datum(v), volumes) for v in runs]
    evidence['dt_pressure_l2'] = [v['dt']*p for v, p in zip(runs, evidence['pressure_l2'])]
    return evidence


def compare(reference, candidate, observe=False):
    for key in ('schema', 'locked_sha', 'amrex_version', 'n_cell', 'epsilon', 'scale', 'stretch', 'steps', 'dt'):
        require(reference[key] == candidate[key], f'peer mismatch: {key}')
    differences = {}
    for key, a in reference['fields'].items():
        b = candidate['fields'][key]
        require(len(a) == len(b), f'peer shape: {key}')
        norm = math.sqrt(math.fsum((x-y)**2 for x, y in zip(a, b)))
        scale = max(math.sqrt(math.fsum(x*x for x in a)), math.sqrt(math.fsum(y*y for y in b)))
        if key in ('divergence', 'interpolated_divergence'):
            scale = max(scale, math.sqrt(len(a)))  # Physical U/L=1; avoid roundoff normalization.
        relative = norm/scale if scale else norm
        pointwise = all(abs(x-y) <= 2e-10+2e-11*max(abs(x), abs(y)) for x, y in zip(a, b))
        if not observe:
            require(relative <= 1e-12, f'peer relative L2: {key} {relative}')
            require(pointwise, f'peer pointwise: {key}')
        differences[key] = {'relative_l2': relative, 'max_absolute': max(abs(x-y) for x, y in zip(a, b)),
                            'gate': 'PASS' if relative <= 1e-12 and pointwise else 'BLOCKED'}
    return differences


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe', required=True)
    p.add_argument('--input', required=True)
    p.add_argument('--work', required=True)
    p.add_argument('--backend', choices=('CPU', 'CUDA'), default='CPU')
    p.add_argument('--steps', type=int, default=32)
    p.add_argument('--dt', type=float, default=2.5e-4)
    p.add_argument('--timeout', type=float, default=180)
    p.add_argument('--max-grid', type=int, default=8)
    p.add_argument('--epsilon', type=float, default=0.15)
    p.add_argument('--mpi-exec')
    p.add_argument('--ranks', type=int, default=1)
    p.add_argument('--mpi-preflag', action='append', default=[])
    p.add_argument('--mpi-postflag', action='append', default=[])
    p.add_argument('--mpi-numproc-flag', default='-n')
    p.add_argument('--tile')
    p.add_argument('--reference', help='Existing independently validated audit summary for full-field comparison')
    p.add_argument('--require-pressure-convergence', action='store_true')
    p.add_argument('--observe-peer', action='store_true',
                   help='Keep failed peer field gates BLOCKED in evidence; require all advance invariants')
    args = p.parse_args()
    require(args.steps >= 16 and 8*args.steps <= 4096 and args.ranks > 0, 'audit step/rank range')
    require(math.isfinite(args.dt) and args.dt > 0, 'positive finite dt')
    require(math.isfinite(args.timeout) and args.timeout > 0, 'positive timeout budget')
    require(args.mpi_exec or args.ranks == 1, 'multiple ranks require MPI launcher')
    exe, source, work = Path(args.exe).resolve(), Path(args.input).resolve(), Path(args.work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    summary = work/'audit.json'
    summary.unlink(missing_ok=True)
    runs = {}
    # Triple fixed-time refinement plus an eight-times-longer coupled trajectory.
    for name, steps, dt in [('dt', args.steps, args.dt), ('dt2', 2*args.steps, args.dt/2),
                            ('dt4', 4*args.steps, args.dt/4), ('long', 8*args.steps, args.dt)]:
        require(steps <= 4096, 'long step range')
        report = work/f'{name}.json'
        report.unlink(missing_ok=True)
        command = [str(exe), str(source), f'avwis.dt={dt}', f'avwis.run_steps={steps}',
                   f'avwis.max_grid_size={args.max_grid}', f'avwis.mapping.epsilon={args.epsilon}',
                   f'avwis.metadata_file={report}']
        if args.tile:
            command.append(f'fabarray.mfiter_tile_size={args.tile}')
        if args.mpi_exec:
            command = [args.mpi_exec, *args.mpi_preflag, args.mpi_numproc_flag, str(args.ranks),
                       *command, *args.mpi_postflag]
        # Reap the entire launcher group on timeout, including MPI child ranks.
        process = subprocess.Popen(command, cwd=work, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, text=True, start_new_session=True)
        timed_out = False
        try:
            stdout, stderr = process.communicate(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                stdout, stderr = process.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                stdout, stderr = process.communicate()
        (work/f'{name}.log').write_text(stdout+stderr)
        require(not timed_out, f'{name}: runtime budget {args.timeout}s exhausted; see {work / (name+".log")}')
        require(process.returncode == 0, f'{name}: exit {process.returncode}; see {work / (name+".log")}')
        v = json.loads(report.read_text())
        validate(v, steps, dt, args.ranks, args.backend)
        runs[name] = v
    for name in ('dt2', 'dt4', 'long'):
        require(runs[name]['fields']['cell_volume'] == runs['dt']['fields']['cell_volume'], 'metric changed')
        require(runs[name]['fields']['initial_Ucat'] == runs['dt']['fields']['initial_Ucat'], 'initial state depends on dt')
    temporal = temporal_evidence([runs[k] for k in ('dt', 'dt2', 'dt4')])
    require(all(temporal[k]['gate'] == 'PASS' for k in ('Ucat', 'Ucont_0', 'Ucont_1', 'Ucont_2')),
            'velocity/face temporal gate blocked')
    peer = {}
    if args.reference:
        reference = json.loads(Path(args.reference).read_text())
        for name, v in runs.items():
            old = reference['runs'][name]
            validate(old, v['steps'], v['dt'], old['ranks'], old['backend'])
            peer[name] = compare(old, v, observe=args.observe_peer)
    backend_gate = ('PASS' if all(f['gate'] == 'PASS' for run in peer.values() for f in run.values())
                    else 'BLOCKED') if peer else 'NOT_RUN'
    value = {'backend_field_gate': backend_gate, 'invariant_status': 'PASS', 'velocity_time_gate': 'PASS',
             'pressure_time_gate': temporal['P']['gate'], 'state_contract': 'experimental_dual_state',
             'temporal': temporal, 'peer_fields': peer, 'runs': runs}
    summary.write_text(json.dumps(value, indent=2)+'\n')
    print(f"C3 long advance invariants PASS; velocity time gate PASS; pressure time gate {value['pressure_time_gate']}; peer field gate {backend_gate}")
    if args.require_pressure_convergence:
        require(value['pressure_time_gate'] == 'PASS', 'pressure temporal gate BLOCKED; P is accumulated correction history')


if __name__ == '__main__':
    main()
