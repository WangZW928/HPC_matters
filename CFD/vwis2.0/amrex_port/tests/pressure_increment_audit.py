#!/usr/bin/env python3
"""Small fixed-time acceptance of actual incremental-pressure production advances.

No continuous physical-pressure or backend gate is promoted by this audit.
"""
import argparse
import json
import os
import signal
from pathlib import Path
import subprocess
import numpy as np


def require(ok, why):
    if not ok:
        raise RuntimeError(why)


def run(args, name, dt, steps, instantaneous=False):
    path = args.work / (name+'.json')
    command = ([args.mpi_exec, '-n', str(args.ranks)] if args.mpi_exec else []) + [
        str(args.exe), str(args.input), f'avwis.dt={dt:.17g}', f'avwis.run_steps={steps}',
        f'avwis.metadata_file={path}']
    if instantaneous:
        command += ['avwis.pressure_contract=instantaneous_pressure',
                    'avwis.pressure_increment_reference=0', 'avwis.face_authority_reference=1']
    env = dict(os.environ, OMPI_ALLOW_RUN_AS_ROOT='1', OMPI_ALLOW_RUN_AS_ROOT_CONFIRM='1')
    (args.work/(name+'-command.json')).write_text(json.dumps(command, indent=2))
    with (args.work/(name+'.log')).open('w') as log:
        result = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        try:
            result.wait(timeout=300)
        except subprocess.TimeoutExpired:
            os.killpg(result.pid, signal.SIGTERM)
            try:
                result.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(result.pid, signal.SIGKILL)
                result.wait()
            raise RuntimeError(f'{name} runtime timeout; see saved log')
    require(result.returncode == 0, f'{name} runtime failed; see {args.work/(name+".log")}')
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exe', type=Path, required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--mpi-exec')
    parser.add_argument('--ranks', type=int, default=4)
    args = parser.parse_args()
    args.exe = args.exe.resolve(); args.input = args.input.resolve(); args.work = args.work.resolve()
    args.work.mkdir(parents=True, exist_ok=True)
    series = [run(args, f'dt{factor}', .0005/factor, 4*factor) for factor in (1, 2, 4)]
    final_p, final_phi = [], []
    for v in series:
        require(v['status'] == 'PASS' and v['schema'] == 'avwis-pressure-increment-v1', 'schema/status')
        require(v['update_error'] == 0 and v['momentum_drift'] <= 1e-12 and
                v['max_divergence'] <= 1e-10 and v['max_full_relative_residual'] <= 1e-11,
                'original conservation/projection gates')
        require(all(v[k] for k in ('first_step_P_equals_Phi', 'standalone_replay_rejected',
                                  'failed_advance_all_state_exact', 'retry_commits_once', 'zero_update_idempotent')), 'state lifecycle')
        previous = np.zeros_like(v['series'][0]['P'])
        for step in v['series']:
            p = np.asarray(step['P']); phi = np.asarray(step['Phi'])
            require(np.isfinite(p).all() and np.isfinite(phi).all(), 'finite P/Phi')
            require(np.array_equal(previous+phi, p), 'independent exact once pressure recurrence')
            previous = p
        require(np.array_equal(v['series'][0]['P'], v['series'][0]['Phi']), 'P^0=0 first-step semantics')
        final_p.append(np.asarray(v['series'][-1]['P']))
        final_phi.append(np.asarray(v['series'][-1]['Phi']))
    rms = lambda x: float(np.sqrt(np.mean(x*x)))
    p_differences = [rms(final_p[n]-final_p[n+1]) for n in (0, 1)]
    phi_norms = [rms(x) for x in final_phi]
    require(p_differences[0] > 0 and p_differences[1] > 0, 'nontrivial pressure time refinement')
    p_ratio = p_differences[0]/p_differences[1]
    phi_ratios = [phi_norms[n]/phi_norms[n+1] for n in (0, 1)]
    require(1.5 <= p_ratio <= 2.5 and all(1.5 <= r <= 2.5 for r in phi_ratios),
            'fixed-T discrete P refinement and O(dt) last increment')
    old = run(args, 'instantaneous', .0005, 4, True)
    require(old['pressure_contract'] == 'instantaneous_pressure', 'old opt-in pressure contract')
    comparisons = []
    for n in range(4):
        p = np.asarray(old['fields'][f'step{n}_P'])
        phi = np.asarray(old['fields'][f'step{n}_Phi'])
        require(np.array_equal(p, phi), 'unchanged exact instantaneous P=Phi')
        error = rms(p-np.asarray(series[0]['series'][n]['P']))
        require(error <= 1e-10, 'incremental cumulative multiplier agrees with instantaneous multiplier')
        comparisons.append(error)
    report = {'status': 'PASS', 'final_time': .002,
              'discrete_pressure_time_gate': 'PASS', 'pressure_difference_rms': p_differences,
              'pressure_refinement_ratio': p_ratio, 'final_P_rms': [rms(x) for x in final_p],
              'final_Phi_rms': phi_norms, 'Phi_refinement_ratios': phi_ratios,
              'instantaneous_comparison_rms': comparisons,
              'physical_pressure_convergence': 'NOT_TESTED',
              'historical_pressure_backend_gate': 'BLOCKED',
              'model': 'face multiplier; explicit Euler old-P predictor, alpha=1'}
    (args.work/'audit.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
