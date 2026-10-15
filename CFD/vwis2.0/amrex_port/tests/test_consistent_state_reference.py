#!/usr/bin/env python3
"""Fail-closed affine Fourier reference checks."""
import math
import unittest

import numpy as np
from consistent_state_reference import decode, fft_multiplier
from pressure_phi_reference import difference


class AffineReferenceChecks(unittest.TestCase):
    def fixture(self):
        n = 16
        v = {'n_cell': [n, n, 4], 'scale': [1.1, .9, 1.05],
             'epsilon': .15, 'spacing': [1/n, 1/n, .25]}
        _, j, i = np.indices((4, n, n))
        p = np.sin(2*np.pi*(i+.5)/n+2*np.pi*(j+.5)/n)
        k = 2*np.pi/n
        sx, sy, _ = v['scale']
        eig = -4*((1+v['epsilon']**2)/sx**2+1/sy**2)*math.sin(k/2)**2*n*n
        eig += 2*v['epsilon']/(sx*sy)*math.sin(k)**2*n*n
        return v, p, eig*p

    def test_cross_sign_and_affine_DG_inverse(self):
        v, p, source = self.fixture()
        recovered, nulls, gap = fft_multiplier(v, source)
        self.assertEqual(difference(p, recovered)['gate'], 'PASS')
        self.assertEqual(nulls, 8)
        self.assertGreater(gap, 100)

    def test_incompatible_source_rejected_without_repair(self):
        v, p, _ = self.fixture()
        with self.assertRaisesRegex(RuntimeError, 'compatibility; no mean repair'):
            fft_multiplier(v, np.ones_like(p))

    def test_odd_grid_has_only_one_DIC_datum_mode(self):
        v, _, _ = self.fixture()
        source = np.zeros((5, 15, 15))
        _, nulls, _ = fft_multiplier(v, source)
        self.assertEqual(nulls, 1)

    def test_malformed_transaction_field_rejected(self):
        v = {'n_cell': [2, 2, 2], 'fields': {'cell_volume': [1]*8, 'failed_after_Ucat_old': [0]*23}}
        with self.assertRaisesRegex(RuntimeError, 'shape/finite'):
            decode(v)
        v['fields']['failed_after_Ucat_old'] = [float('nan')]*24
        with self.assertRaisesRegex(RuntimeError, 'shape/finite'):
            decode(v)


if __name__ == '__main__':
    unittest.main()
