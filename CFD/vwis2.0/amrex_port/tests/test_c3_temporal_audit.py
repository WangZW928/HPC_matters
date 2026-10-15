#!/usr/bin/env python3
"""Guard audit acceptance against corrupt evidence and pressure-only nonconvergence."""
import copy
import unittest
from c3_temporal_audit import LOCK_SHA, compare, temporal_evidence, validate


def snapshot(level=0):
    n = 8
    dt = 0.001/2**level
    value = {'schema': 'avwis-c3-temporal-v1', 'invariant_status': 'PASS',
             'locked_sha': LOCK_SHA, 'amrex_version': '26.04', 'ranks': 1, 'backend': 'CPU',
             'active_ranks': 1, 'boxes': 1, 'n_cell': [2, 2, 2], 'epsilon': 0.15,
             'scale': [1, 1, 1], 'stretch': [0, 0, 0], 'dt': dt, 'steps': 16*2**level,
             'step': 16*2**level, 'history_depth': 3, 'time': 0.016,
             'initial_mass_divergence': 1e-12, 'initial_interpolated_divergence': 0.01,
             'history_error': 0, 'momentum_drift': 0, 'mass_error': 0,
             'max_cfl': 0.1, 'max_diffusion': 0.1,
             'projections': [{'before': 0.1, 'after': 1e-12, 'residual': 1e-12,
                              'phi_max': 0.1, 'outer': 10, 'inner': 100} for _ in range(16*2**level)]}
    value['fields'] = {'initial_Ucat': [1.0]*(3*n), 'Ucat': [1+dt]*(3*n),
                       'P': [(-1)**i/dt for i in range(n)], 'Phi': [1.0]*n,
                       'divergence': [0.0]*n, 'interpolated_divergence': [0.01]*n,
                       'cell_volume': [1/n]*n}
    for direction in range(3):
        value['fields'][f'Ucont_{direction}'] = [1+dt]*12
    return value


class AuditContract(unittest.TestCase):
    def test_velocity_cannot_mask_pressure_nonconvergence(self):
        runs = [snapshot(level) for level in range(3)]
        for v in runs:
            validate(v, v['steps'], v['dt'], 1, 'CPU')
        evidence = temporal_evidence(runs)
        self.assertEqual(evidence['Ucat']['gate'], 'PASS')
        self.assertEqual(evidence['P']['gate'], 'BLOCKED')
        self.assertAlmostEqual(evidence['P']['ratio'], 0.5)

    def test_corrupt_or_missing_step_evidence(self):
        for key, changed in [('time', 0.015), ('ranks', 2), ('step', 15),
                             ('projections', []), ('history_error', 1e-8)]:
            with self.subTest(key=key):
                v = snapshot();v[key] = changed
                with self.assertRaises(RuntimeError):
                    validate(v, 16, 0.001, 1, 'CPU')

    def test_residual_and_nontrivial_advance_are_required(self):
        for key, changed in [('residual', 2e-11), ('after', 2e-10), ('before', 0), ('outer', 0)]:
            with self.subTest(key=key):
                v = snapshot();v['projections'][7][key] = changed
                with self.assertRaises(RuntimeError):
                    validate(v, 16, 0.001, 1, 'CPU')

    def test_full_field_corruption_cannot_hide_in_norms(self):
        a = snapshot()
        for key in a['fields']:
            with self.subTest(key=key):
                b = copy.deepcopy(a);b['fields'][key][0] += 1e-6
                with self.assertRaises(RuntimeError):
                    compare(a, b)
        self.assertEqual(compare(a, copy.deepcopy(a))['Ucat']['relative_l2'], 0)

    def test_observation_preserves_failed_gate(self):
        a=snapshot();b=copy.deepcopy(a);b['fields']['Phi'][0]+=1e-6
        result=compare(a,b,observe=True)
        self.assertEqual(result['Phi']['gate'],'BLOCKED')
        self.assertEqual(result['Ucat']['gate'],'PASS')
        with self.assertRaises(RuntimeError):
            compare(a,b)

    def test_nested_nonfinite_and_truncation_rejected(self):
        for changed in (float('nan'), float('inf')):
            v = snapshot();v['fields']['P'][3] = changed
            with self.assertRaises(RuntimeError):
                validate(v, 16, 0.001, 1, 'CPU')
        v = snapshot();v['fields']['Ucont_2'].pop()
        with self.assertRaises(RuntimeError):
            validate(v, 16, 0.001, 1, 'CPU')


if __name__ == '__main__':
    unittest.main()
