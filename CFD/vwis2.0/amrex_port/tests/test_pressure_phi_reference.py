#!/usr/bin/env python3
"""Independent operator reference and fail-closed evidence checks."""
import copy
import math
import unittest
import numpy as np
from pressure_phi_reference import arrays, difference, operator, peers


class ReferenceChecks(unittest.TestCase):
    def fixture(self):
        n = 8
        shape = (n, n, n)
        h = 1/n
        v = {'n_cell': [n]*3, 'spacing': [h]*3}
        f = {'cell_volume': np.full(shape, h**3), 'grad_xi': np.zeros((9, *shape))}
        for d in range(3):
            fs = list(shape); fs[2-d] += 1
            f[f'Q_{d}'] = np.zeros((3, *fs)); f[f'Q_{d}'][d] = h*h
            f[f'S_{d}'] = np.zeros((3, *fs)); f[f'S_{d}'][d] = h*h
            f[f'beta_{d}'] = np.ones(fs); f['grad_xi'][3*d+d] = 1
        x = np.indices(shape)[2]
        p = np.cos(2*math.pi*(x+0.5)/n)
        return v, f, p, h

    def test_discrete_eigenvalues_separate_DG_DIC(self):
        v, f, p, h = self.fixture()
        dg = operator(v, f, p)
        dic = operator(v, f, p, dic=True)
        self.assertEqual(difference(dg, -4*math.sin(math.pi/8)**2/h**2*p)['gate'], 'PASS')
        self.assertEqual(difference(dic, -math.sin(2*math.pi/8)**2/h**2*p)['gate'], 'PASS')
        self.assertGreater(np.max(np.abs(dg-dic)), 1)

    def test_zero_cross_affine_metric(self):
        v, f, p, _ = self.fixture()
        self.assertTrue(np.array_equal(operator(v, f, p, part='cross'), np.zeros_like(p)))

    def test_constant_gauge_does_not_remove_spatial_failure(self):
        a = np.array([1., -1., 1., -1.])
        b = a+np.array([4e-12, 4e-12, -4e-12, -4e-12])
        r = difference(a, b, np.ones(4))
        self.assertEqual(r['gate'], 'BLOCKED')
        self.assertLess(abs(r['weighted_mean_difference']), 1e-15)
        self.assertGreater(r['centered_difference_rms'], 1e-12)

    def test_nonfinite_rejected(self):
        with self.assertRaises(RuntimeError):
            difference([1, math.nan], [1, 2])

    def test_truncated_or_extra_component_rejected(self):
        v = {'n_cell': [2]*3, 'fields': {'cell_volume': [1]*8, 'P': [0]*7}}
        with self.assertRaises(RuntimeError):
            arrays(v)
        v['fields']['P'] = [0]*16
        with self.assertRaises(RuntimeError):
            arrays(v)

    def test_peer_metadata_and_field_set_rejected(self):
        v = {k: 1 for k in ('n_cell', 'dt', 'warmup', 'epsilon', 'scale', 'stretch', 'spacing', 'locked_sha')}
        v['fields'] = {'P': [1]}
        peer = copy.deepcopy(v); peer['dt'] = 2
        with self.assertRaises(RuntimeError):
            peers(v, peer)
        peer = copy.deepcopy(v); peer['fields']['Phi'] = [1]
        with self.assertRaises(RuntimeError):
            peers(v, peer)


if __name__ == '__main__':
    unittest.main()
