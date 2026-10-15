"""Negative contracts for the C7 comparison gate, with no AMReX dependency."""

import copy
import unittest

from c7_backend_compare import compare, validate, LOCKED_SHA, SCHEMA


def fixture():
    fields = {"Ucat": [0.3]*24, "Ucat_old": [0.2]*24, "Ucat_older": [0.1]*24,
              "P": [2.0]*8, "Phi": [0.01]*8, "advection_initial": [0.2]*24,
              "viscous_initial": [0.1]*24, "divergence": [0.0]*8, "cell_volume": [0.125]*8}
    for direction in range(3):
        for name in ("Ucont", "Ucont_old", "Ucont_older"):
            fields[f"{name}_{direction}"] = [0.05]*12
    return {"schema": SCHEMA, "amrex_version": "26.04", "locked_sha": LOCKED_SHA,
            "backend": "CPU", "nprocs": 1, "active_ranks": 1, "boxes": 2,
            "n_cell": [2, 2, 2], "step": 3, "history_depth": 3, "time": 0.0003,
            "advance_seconds_max": 0.1, "reductions": [1.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.1],
            "fields": fields}


class ComparisonContract(unittest.TestCase):
    def test_roundoff_and_changed_timing_allowed(self):
        left = fixture()
        right = copy.deepcopy(left)
        right["advance_seconds_max"] = 100.0
        right["fields"]["Ucont_0"][2] += 1e-14
        validate(right, 1, "CPU")
        self.assertGreater(compare(left, right)["max_abs_errors"]["Ucont_0"], 0)

    def test_relative_norm_gate_is_stricter_than_pointwise_gate(self):
        right = fixture()
        right["fields"]["Ucont_0"] = [v+1e-11 for v in right["fields"]["Ucont_0"]]
        with self.assertRaisesRegex(ValueError, "relative L2 mismatch"):
            compare(fixture(), right)

    def test_each_field_perturbation_rejected(self):
        left = fixture()
        for name in left["fields"]:
            with self.subTest(field=name):
                right = copy.deepcopy(left)
                right["fields"][name][0] += 1e-5
                with self.assertRaisesRegex(ValueError, "pointwise mismatch"):
                    compare(left, right)

    def test_nonfinite_and_truncated_fields_rejected(self):
        for value in (float("nan"), float("inf")):
            right = fixture()
            right["fields"]["P"][0] = value
            with self.assertRaises(ValueError):
                validate(right, 1, "CPU")
            with self.assertRaises(ValueError):
                compare(fixture(), right)
        right = fixture()
        right["fields"]["Ucont_0"].pop()
        with self.assertRaises(ValueError):
            validate(right, 1, "CPU")
        with self.assertRaises(ValueError):
            compare(fixture(), right)

    def test_missing_field_rank_backend_and_history_rejected(self):
        for key, value in (("nprocs", 2), ("active_ranks", 0), ("backend", "CUDA"),
                           ("history_depth", 1), ("locked_sha", "wrong"), ("time", 0.1)):
            right = fixture()
            right[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate(right, 1, "CPU")
        right = fixture()
        del right["fields"]["Ucat_older"]
        with self.assertRaises(ValueError):
            validate(right, 1, "CPU")
        with self.assertRaises(ValueError):
            compare(fixture(), right)

    def test_mass_reduction_and_frozen_history_rejected(self):
        right = fixture()
        right["reductions"][4] = 1e-4
        with self.assertRaises(ValueError):
            validate(right, 1, "CPU")
        right = fixture()
        right["fields"]["Ucat_old"] = right["fields"]["Ucat"][:]
        with self.assertRaises(ValueError):
            validate(right, 1, "CPU")


if __name__ == "__main__":
    unittest.main()
