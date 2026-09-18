#!/usr/bin/env python3
"""Render the checked-in C8-G1 profile and convergence evidence."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedFormatter, FixedLocator, NullLocator
from PIL import Image


RESOLUTIONS = (16, 32, 64)
COLORS = {16: "#0072B2", 32: "#E69F00", 64: "#009E73"}
MARKERS = {16: "o", 32: "s", 64: "^"}


def read_profile(path: Path) -> dict[str, list[float]]:
    columns = {"y": [], "w_numerical": [], "w_exact": [], "error": []}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            for name in columns:
                columns[name].append(float(row[name]))
    return columns


def read_summary(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def atomic_savefig(fig: mpl.figure.Figure, path: Path, *, dpi: int | None = None) -> None:
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    fig.savefig(
        temporary,
        format=path.suffix.lstrip("."),
        dpi=dpi,
        facecolor="white",
        metadata={"Creator": f"Matplotlib {mpl.__version__}; C8-G1 plot_results.py"},
    )
    if path.suffix == ".png":
        rgb_temporary = path.with_name(f".{path.stem}.tmp-rgb.png")
        with Image.open(temporary) as raster:
            raster.convert("RGB").save(rgb_temporary, format="PNG", dpi=(dpi, dpi))
        temporary.unlink()
        temporary = rgb_temporary
    elif path.suffix == ".svg":
        svg = temporary.read_text(encoding="utf-8")
        temporary.write_text(
            "\n".join(line.rstrip() for line in svg.splitlines()) + "\n",
            encoding="utf-8",
        )
    os.replace(temporary, path)


def require_writable_outputs(paths: list[Path], force: bool) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing and not force:
        raise SystemExit("refusing to overwrite existing outputs; pass --force: " + ", ".join(existing))


def plot_profiles(
    profiles: dict[int, dict[str, list[float]]], output_stem: Path
) -> list[Path]:
    fig, axes = plt.subplots(1, 2, figsize=(180 / 25.4, 76 / 25.4), layout="constrained")
    ax_profile, ax_error = axes

    exact_y = [index / 400 for index in range(401)]
    exact_w = [6.0 * y * (1.0 - y) for y in exact_y]
    ax_profile.plot(exact_w, exact_y, color="#222222", linewidth=1.8, label="Analytic")
    for resolution in RESOLUTIONS:
        profile = profiles[resolution]
        markevery = max(1, resolution // 8)
        ax_profile.plot(
            profile["w_numerical"],
            profile["y"],
            color=COLORS[resolution],
            marker=MARKERS[resolution],
            markevery=markevery,
            markersize=4.0,
            linewidth=1.1,
            label=rf"$N_y={resolution}$",
        )
        ax_error.semilogx(
            [abs(value) for value in profile["error"]],
            profile["y"],
            color=COLORS[resolution],
            marker=MARKERS[resolution],
            markevery=markevery,
            markersize=4.0,
            linewidth=1.1,
            label=rf"$N_y={resolution}$",
        )

    ax_profile.set(
        xlabel=r"Streamwise velocity $w$",
        ylabel=r"Wall-normal coordinate $y$",
        xlim=(0.0, 1.58),
        ylim=(0.0, 1.0),
        title="(a) Velocity profile",
    )
    ax_error.set(
        xlabel=r"Absolute pointwise error $|w_h-w|$ (log scale)",
        ylabel=r"Wall-normal coordinate $y$",
        ylim=(0.0, 1.0),
        title="(b) Profile error",
    )
    for axis in axes:
        axis.grid(True, color="#D9D9D9", linewidth=0.6)
        axis.legend(frameon=False, fontsize=8)

    outputs = [output_stem.with_suffix(".png"), output_stem.with_suffix(".svg")]
    atomic_savefig(fig, outputs[0], dpi=301)
    atomic_savefig(fig, outputs[1])
    plt.close(fig)
    return outputs


def plot_convergence(summaries: dict[int, dict[str, object]], output_stem: Path) -> list[Path]:
    grids = list(RESOLUTIONS)
    velocity = [float(summaries[n]["velocity_l2"]) for n in grids]
    residual = [float(summaries[n]["steady_residual_l2"]) for n in grids]
    flow = [float(summaries[n]["flow_rate_relative_error"]) for n in grids]
    shear = [float(summaries[n]["wall_shear_max_abs_error"]) for n in grids]

    fig, axes = plt.subplots(1, 2, figsize=(180 / 25.4, 76 / 25.4), layout="constrained")
    ax_field, ax_functional = axes
    ax_field.loglog(grids, velocity, color="#0072B2", marker="o", label=r"Velocity $L_2$")
    ax_field.loglog(grids, residual, color="#D55E00", marker="s", label=r"Steady residual $L_2$")
    ax_field.loglog(
        grids,
        [velocity[0] * (grids[0] / n) ** 2 for n in grids],
        color="#555555",
        linestyle="--",
        label=r"$O(N_y^{-2})$ guide",
    )
    ax_functional.loglog(grids, flow, color="#009E73", marker="^", label="Flow relative error")
    ax_functional.loglog(grids, shear, color="#CC79A7", marker="D", label="Wall-shear abs. error")
    ax_functional.loglog(
        grids,
        [flow[0] * (grids[0] / n) ** 2 for n in grids],
        color="#555555",
        linestyle="--",
        label=r"$O(N_y^{-2})$ guide",
    )
    ax_functional.loglog(
        grids,
        [shear[0] * grids[0] / n for n in grids],
        color="#777777",
        linestyle=":",
        label=r"$O(N_y^{-1})$ guide",
    )

    ax_field.set(
        xlabel=r"Wall-normal cells $N_y$ (log scale)",
        ylabel="Error / residual magnitude (log scale)",
        title="(a) Field and residual",
    )
    ax_functional.set(
        xlabel=r"Wall-normal cells $N_y$ (log scale)",
        ylabel="Functional error magnitude (log scale)",
        title="(b) Flow and wall shear",
    )
    for axis in axes:
        axis.xaxis.set_major_locator(FixedLocator(grids))
        axis.xaxis.set_major_formatter(FixedFormatter([str(value) for value in grids]))
        axis.xaxis.set_minor_locator(NullLocator())
        axis.grid(True, which="both", color="#D9D9D9", linewidth=0.6)
        axis.legend(frameon=False, fontsize=8)

    outputs = [output_stem.with_suffix(".png"), output_stem.with_suffix(".svg")]
    atomic_savefig(fig, outputs[0], dpi=301)
    atomic_savefig(fig, outputs[1])
    plt.close(fig)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="replace existing figures and manifest")
    args = parser.parse_args()

    case_dir = Path(__file__).resolve().parent
    raw_dir = case_dir / "raw"
    figure_dir = case_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    profiles = {
        n: read_profile(raw_dir / f"mapped_channel_{n}.csv") for n in RESOLUTIONS
    }
    summaries = {
        n: read_summary(raw_dir / f"mapped_channel_{n}.json") for n in RESOLUTIONS
    }
    output_paths = [
        figure_dir / "profiles.png",
        figure_dir / "profiles.svg",
        figure_dir / "convergence.png",
        figure_dir / "convergence.svg",
        raw_dir / "figure_manifest.json",
    ]
    require_writable_outputs(output_paths, args.force)

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "lines.linewidth": 1.3,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    produced = []
    produced.extend(plot_profiles(profiles, figure_dir / "profiles"))
    produced.extend(plot_convergence(summaries, figure_dir / "convergence"))

    manifest = {
        "schema": "avwis-c8-g1-figure-manifest-v1",
        "source_data": [
            f"raw/mapped_channel_{n}.{extension}"
            for n in RESOLUTIONS
            for extension in ("csv", "json")
        ],
        "outputs": [str(path.relative_to(case_dir)) for path in produced],
        "transformations": [
            "absolute value of the signed pointwise velocity error for log display",
            "logarithmic axes applied only to strictly positive error and residual values",
            "continuous analytic profile w=6*y*(1-y) sampled at 401 points",
            "O(N_y^-2) and O(N_y^-1) reference guides normalized at N_y=16",
            "no filtering, smoothing, exclusion, or interpolation of numerical observations",
        ],
        "matplotlib_version": mpl.__version__,
        "png_dpi": 301,
        "figure_width_mm": 180,
        "background": "opaque white",
    }
    temporary_manifest = raw_dir / ".figure_manifest.tmp.json"
    temporary_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary_manifest, raw_dir / "figure_manifest.json")


if __name__ == "__main__":
    main()
