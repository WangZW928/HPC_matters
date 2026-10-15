#!/usr/bin/env python3
"""Focused production-algorithm regression: MMS orders and weak-shear scan."""
import argparse
import json
import math
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exe', required=True)
    parser.add_argument('--input', required=True)
    parser.add_argument('--work', required=True)
    args = parser.parse_args()
    exe, source = Path(args.exe).resolve(), Path(args.input).resolve()
    work = Path(args.work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    summary = work / 'regression.json'
    summary.unlink(missing_ok=True)
    results = {}
    runs = [(f'mms_{n}', n, 0.15, False) for n in (16, 32, 64)]
    runs += [(f'scan_{epsilon}', 16, epsilon, False) for epsilon in (-0.25, -0.1, 0.0, 0.1, 0.25)]
    runs += [('affine', 16, 0.15, True), ('single_box', 16, 0.15, False)]
    for name, n, epsilon, affine in runs:
        report = work / f'{name}.json'
        report.unlink(missing_ok=True)
        command = [str(exe), str(source), f'avwis.n_cell={n} {n} 4',
                   f'avwis.mapping.epsilon={epsilon}', f'avwis.metadata_file={report}']
        if name == 'single_box':
            command += ['avwis.max_grid_size=16']
        if affine:
            command += ['avwis.mapping.stretch=0 0 0']
        run = subprocess.run(command, cwd=work, capture_output=True, text=True, timeout=120)
        (work / f'{name}.log').write_text(run.stdout + run.stderr)
        if run.returncode:
            raise RuntimeError(f'{name}: exit {run.returncode}; see {work / (name + ".log")}')
        value = json.loads(report.read_text())
        assert value['status'] == 'PASS' and value['cells'] == n
        assert abs(value['epsilon'] - epsilon) < 1e-15
        for key, number in value.items():
            if isinstance(number, (int, float)):
                assert math.isfinite(number), (name, key, number)
        assert value['coupled_steps'] == 4 and len(value['step_projections']) == 4
        assert value['history_error'] == 0
        assert value['momentum_drift'] < 1e-12 and value['state_update_error'] < 1e-11
        for projection in value['step_projections']:
            assert projection['divergence_before'] > 1e-8
            assert projection['divergence_after'] <= 1e-10
            assert projection['full_relative_residual'] <= 1e-11
            if epsilon != 0:
                assert projection['outer'] > 0 and projection['inner'] > 0
        results[name] = value
    for key in ('lap_l2', 'adv_l2', 'gradient_l2'):
        for coarse, fine in ((16, 32), (32, 64)):
            ratio = results[f'mms_{coarse}'][key] / results[f'mms_{fine}'][key]
            assert ratio >= 2**1.8, (key, coarse, fine, ratio)
    for key in ('lap_l2', 'adv_l2', 'gradient_l2', 'energy_rate'):
        a, b = results['mms_16'][key], results['single_box'][key]
        assert abs(a-b)/max(abs(a), abs(b)) <= 1e-12, ('single_box', key, a, b)
    # The normalized six-face LS tensor has eigenvalues 1 +/- chi, 1,
    # chi=|epsilon|/sqrt(1+epsilon^2). Its conditioning is controlled here.
    assert max(v['outer_iterations'] for v in results.values()) <= 40
    summary.write_text(json.dumps({'status': 'PASS', 'runs': results}, indent=2) + '\n')
    print('C3 shear MMS, epsilon scan, cross action, conservation, projection, and free stream: PASS')


if __name__ == '__main__':
    main()
