"""C7 pointwise backend/rank/layout regression; Python standard library only."""

import argparse
import json
import math
from pathlib import Path
import subprocess


SCHEMA = "avwis-c7-backend-v1"
LOCK = dict(line.split("=", 1) for line in
            (Path(__file__).resolve().parents[1] / "amrex_version.lock").read_text().splitlines()
            if line and not line.startswith("#"))
LOCKED_SHA = LOCK["AMReX_GIT_SHA"]


def validate(report, ranks, backend):
    if (report["schema"] != SCHEMA or report["amrex_version"] != LOCK["AMReX_VERSION"]
            or report["locked_sha"] != LOCKED_SHA or report["backend"] != backend
            or report["nprocs"] != ranks
            or report["active_ranks"] != min(ranks, report["boxes"])
            or report["step"] != 3 or report["history_depth"] != 3):
        raise ValueError("C7 provenance/rank/history mismatch")
    nx, ny, nz = report["n_cell"]
    cells = nx * ny * nz
    lengths = {name: cells * components for name, components in {
        "Ucat": 3, "Ucat_old": 3, "Ucat_older": 3, "P": 1, "Phi": 1,
        "advection_initial": 3, "viscous_initial": 3, "divergence": 1,
        "cell_volume": 1}.items()}
    for direction, faces in enumerate(((nx+1)*ny*nz, nx*(ny+1)*nz, nx*ny*(nz+1))):
        for layer in ("Ucont", "Ucont_old", "Ucont_older"):
            lengths[f"{layer}_{direction}"] = faces
    if set(report["fields"]) != set(lengths):
        raise ValueError("C7 field set mismatch")
    for name, length in lengths.items():
        values = report["fields"][name]
        if len(values) != length or not all(math.isfinite(v) for v in values):
            raise ValueError(f"C7 field shape/nonfinite: {name}")
    reductions = report["reductions"]
    if len(reductions) != 7 or not all(math.isfinite(v) for v in reductions):
        raise ValueError("C7 reduction shape/nonfinite")
    before, after, divergence, integral, boundary, cfl, diffusion = reductions
    if (before <= 1e-3 or after >= 1e-10 or divergence >= 1e-10
            or abs(integral-boundary) > 1e-12 or not 0 <= cfl <= 1
            or not 0 <= diffusion <= 1 or not math.isfinite(report["time"])
            or not math.isclose(report["time"], 3e-4, rel_tol=1e-14, abs_tol=0.0)
            or not math.isfinite(report["advance_seconds_max"])
            or report["advance_seconds_max"] < 0):
        raise ValueError("C7 projection/mass/stability/timing gate failed")
    if (report["fields"]["Ucat"] == report["fields"]["Ucat_old"]
            or report["fields"]["Ucat_old"] == report["fields"]["Ucat_older"]
            or max(map(abs, report["fields"]["advection_initial"])) == 0
            or max(map(abs, report["fields"]["viscous_initial"])) == 0):
        raise ValueError("C7 requires nontrivial RHS and evolving history")


def compare(reference, candidate, atol=2e-10, rtol=2e-11):
    for key in ("schema", "amrex_version", "locked_sha", "n_cell", "step", "history_depth", "time"):
        if reference[key] != candidate[key]:
            raise ValueError(f"C7 configuration mismatch: {key}")
    if set(reference["fields"]) != set(candidate["fields"]):
        raise ValueError("C7 comparison field set mismatch")
    errors, norm_errors = {}, {}
    pairs = [("reductions", reference["reductions"], candidate["reductions"])]
    pairs += [(name, reference["fields"][name], candidate["fields"][name])
              for name in reference["fields"]]
    for name, left, right in pairs:
        if len(left) != len(right) or not left:
            raise ValueError(f"C7 comparison shape mismatch: {name}")
        maximum = 0.0
        for index, (a, b) in enumerate(zip(left, right)):
            error = abs(a-b)
            if not (math.isfinite(a) and math.isfinite(b)) or error > atol + rtol*max(abs(a), abs(b)):
                raise ValueError(f"C7 pointwise mismatch: {name}[{index}] {a} vs {b}")
            maximum = max(maximum, error)
        errors[name] = maximum
        if name != "reductions":
            difference_norm = math.sqrt(math.fsum((a-b)**2 for a, b in zip(left, right)))
            reference_norm = math.sqrt(math.fsum(a*a for a in left))
            candidate_norm = math.sqrt(math.fsum(b*b for b in right))
            # Divergence has exact reference zero. Use the case's U/L=1
            # physical scale rather than dividing by a roundoff residual.
            scale = math.sqrt(len(left)) if name == "divergence" else 1e-30
            relative = difference_norm / max(reference_norm, candidate_norm, scale)
            if relative > 1e-12:
                raise ValueError(f"C7 relative L2 mismatch: {name} {relative}")
            norm_errors[name] = relative
    return {"max_abs_errors": errors, "relative_l2_errors": norm_errors}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", required=True, type=Path)
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--work", required=True, type=Path)
    parser.add_argument("--backend", choices=("CPU", "CUDA"), default="CPU")
    parser.add_argument("--mpi-exec")
    parser.add_argument("--mpi-numproc-flag", default="-n")
    parser.add_argument("--mpi-preflag", action="append", default=[])
    parser.add_argument("--mpi-postflag", action="append", default=[])
    parser.add_argument("--peer-exe", type=Path)
    parser.add_argument("--peer-backend", choices=("CPU", "CUDA"), default="CUDA")
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    (args.work / "comparison.json").unlink(missing_ok=True)
    summary = {"schema": "avwis-c7-comparison-v1", "atol": 2e-10, "rtol": 2e-11,
               "relative_l2_tolerance": 1e-12, "divergence_reference_scale": 1.0,
               "comparisons": []}

    def run(case, exe, backend, ranks, grid, launch_mpi, role=""):
        stem = f"{role}{case}_{backend}_{ranks}rank_grid{grid}"
        path = args.work / f"{stem}.json"
        # Discard only this exact previous report: failure cannot reuse a stale PASS.
        path.unlink(missing_ok=True)
        command = []
        if launch_mpi:
            command += [args.mpi_exec, args.mpi_numproc_flag, str(ranks), *args.mpi_preflag]
        command += [str(exe.resolve())]
        if launch_mpi:
            command += args.mpi_postflag
        command += [str((args.inputs / f"c7_mapped_{case}.in").resolve()),
                    f"avwis.max_grid_size={grid}", f"avwis.metadata_file={path.resolve()}"]
        result = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, timeout=120, check=False)
        (args.work / f"{stem}.log").write_text(result.stdout)
        if result.returncode:
            raise RuntimeError(f"C7 run failed ({result.returncode}): {command}\n{result.stdout}")
        report = json.loads(path.read_text())
        validate(report, ranks, backend)
        return report, stem

    for case in ("periodic", "boundary"):
        reference, reference_name = run(case, args.exe, args.backend, 1, 8, bool(args.mpi_exec))
        matrix = [(1, 4), (1, 16)]
        if args.mpi_exec:
            matrix += [(2, 4), (2, 8), (4, 4), (4, 8)]
        for ranks, grid in matrix:
            candidate, name = run(case, args.exe, args.backend, ranks, grid, bool(args.mpi_exec))
            summary["comparisons"].append({"reference": reference_name, "candidate": name,
                                           **compare(reference, candidate),
                                           "advance_seconds_max": candidate["advance_seconds_max"]})
        if args.peer_exe:
            for grid in (4, 8, 16):
                candidate, name = run(case, args.peer_exe, args.peer_backend, 1, grid, False, "peer_")
                summary["comparisons"].append({"reference": reference_name, "candidate": name,
                                               **compare(reference, candidate),
                                               "advance_seconds_max": candidate["advance_seconds_max"]})
    summary["status"] = "PASS"
    (args.work / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"C7 pointwise comparison PASS: {len(summary['comparisons'])} comparisons")


if __name__ == "__main__":
    main()
