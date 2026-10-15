#!/usr/bin/env python3
"""Exercise the executable's alpha scope without changing any numerical gate."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile

from c3_temporal_audit import require, validate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exe', required=True)
    parser.add_argument('--inputs', required=True)
    parser.add_argument('--work', required=True)
    parser.add_argument('--backend', choices=('CPU', 'CUDA'), default='CPU')
    parser.add_argument('--mpi-exec')
    parser.add_argument('--mpi-numproc-flag', default='-n')
    parser.add_argument('--mpi-preflag', action='append', default=[])
    parser.add_argument('--mpi-postflag', action='append', default=[])
    parser.add_argument('--ranks', type=int, default=1)
    args = parser.parse_args()
    require(args.ranks > 0 and (args.mpi_exec or args.ranks == 1), 'MPI launcher/ranks')
    exe, inputs = Path(args.exe).resolve(), Path(args.inputs).resolve()
    parent = Path(args.work).resolve()
    parent.mkdir(parents=True, exist_ok=True)
    # Retain fresh evidence on failure; never overwrite old runs or stale PASS.
    work = Path(tempfile.mkdtemp(prefix='run-', dir=parent))
    results = []

    def run(name, source, overrides, expected=None):
        command = [str(exe), str(source), *overrides]
        if args.mpi_exec:
            command = [args.mpi_exec, *args.mpi_preflag, args.mpi_numproc_flag,
                       str(args.ranks), *command, *args.mpi_postflag]
        process = subprocess.Popen(command, cwd=work, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, start_new_session=True)
        try:
            output, _ = process.communicate(timeout=120)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            output, _ = process.communicate()
            (work / (name + '.log')).write_text(output)
            raise RuntimeError(f'{name}: timeout; see {work}')
        (work / (name + '.log')).write_text(output)
        if expected is None:
            require(process.returncode == 0, f'{name}: failed; see {work}')
        else:
            require(process.returncode != 0 and expected in output,
                    f'{name}: missing precise rejection; see {work}')
        results.append({'name': name, 'exit': process.returncode})
        return output

    diagnostic = 'P5-005 projection_time_coefficient must be 1 outside standalone projection contracts'
    # All modes that do not consume alpha, including non-stepping modes. A bad
    # coordinate selector proves rejection precedes mapping/solver construction.
    modes = ('solve', 'contract_base', 'contract_base_p2', 'contract_p2', 'contract_p3',
             'contract_p5_advection', 'contract_p5_viscous', 'contract_p5_time',
             'contract_p5_orthogonal_time', 'contract_p8_restart', 'contract_p8_sampling',
             'contract_c7_backend', 'contract_c3_shear', 'contract_c3_temporal',
             'benchmark_cartesian', 'benchmark_cartesian_mms', 'benchmark_physical_channel',
             'benchmark_mapped_channel_reference', 'case_lid_cavity')
    for mode in modes:
        run('reject_' + mode, inputs / 'p1_smoke.in',
            [f'avwis.mode={mode}', 'avwis.coordinates=unimplemented',
             'avwis.projection_time_coefficient=1.5'], diagnostic)
    for value in ('0', '-1', 'nan', 'inf', '1.000000000000001'):
        run('reject_value_' + value, inputs / 'p1_smoke.in',
            ['avwis.projection_time_coefficient=' + value], diagnostic)
    legacy = work / 'legacy.in'
    legacy.write_text('vwis.mode=solve\nvwis.projection_time_coefficient=1.5\n')
    run('reject_legacy', legacy, [], diagnostic)

    projections = (('p4_periodic_projection.in', 'P4 Cartesian projection contract: PASS'),
                   ('p5_analytic_orthogonal_projection.in', 'C2.2 orthogonal solver projection contract: PASS'),
                   ('p5_c5_mapped_periodic_boundary.in', 'C5.1 mapped boundary contract: PASS'),
                   ('p5_c52_mapped_inflow_outflow.in', 'C5.2 mapped boundary contract: PASS'))
    for source, marker in projections:
        output = run('allow_' + source, inputs / source, ['avwis.projection_time_coefficient=1.5'])
        require(marker in output, f'{source}: missing numerical contract; see {work}')

    runs = []
    for name, overrides in (('default', []), ('explicit_one', ['avwis.projection_time_coefficient=1'])):
        report = work / (name + '.json')
        run(name, inputs / 'c3_temporal.in',
            ['avwis.run_steps=16', 'avwis.max_grid_size=16',
             f'avwis.metadata_file={report}', *overrides])
        value = json.loads(report.read_text())
        validate(value, 16, 2.5e-4, args.ranks, args.backend)
        value.pop('audited_advance_seconds_max')
        runs.append(value)
    require(runs[0] == runs[1], 'default/explicit alpha=1 changed actual state or diagnostics')
    summary = {'contract_status': 'PASS', 'ranks': args.ranks, 'backend': args.backend,
               'cases': results, 'default_alpha_one_exact': True,
               'scope': 'input rejection and 16-step invariants; no pressure-time/backend acceptance'}
    (work / 'contract.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(f'P5-005 alpha scope PASS: {len(results)} cases; 32 real advances; evidence {work}')


if __name__ == '__main__':
    main()
