#!/usr/bin/env python3
"""Render unsmoothed cavity results from the preserved CSV artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


CASE_DIR = Path(__file__).resolve().parent
RAW_DIR = CASE_DIR / "raw"
FIGURE_DIR = CASE_DIR / "figures"


def read_csv(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise RuntimeError(f"empty CSV: {path}")
    return {name: np.asarray([row[name] for row in rows], dtype=float) for name in rows[0]}


def save(fig: plt.Figure, stem: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / f"{stem}.png", dpi=200, facecolor="white")
    fig.savefig(FIGURE_DIR / f"{stem}.svg", facecolor="white")
    plt.close(fig)


def main() -> None:
    with (RAW_DIR / "summary.json").open(encoding="utf-8") as stream:
        summary = json.load(stream)
    field = read_csv(RAW_DIR / "final_field.csv")
    history = read_csv(RAW_DIR / "history.csv")

    with (RAW_DIR / "centerlines.csv").open(newline="", encoding="utf-8") as stream:
        centerline_rows = list(csv.DictReader(stream))
    profiles: dict[str, tuple[list[float], list[float]]] = {}
    for row in centerline_rows:
        coordinate, velocity = profiles.setdefault(row["profile"], ([], []))
        coordinate.append(float(row["coordinate"]))
        velocity.append(float(row["velocity"]))

    nx, ny, _ = summary["grid"]
    x = field["x"].reshape(ny, nx)[0, :]
    y = field["y"].reshape(ny, nx)[:, 0]
    u = field["u"].reshape(ny, nx)
    v = field["v"].reshape(ny, nx)
    speed = field["velocity_magnitude"].reshape(ny, nx)

    with plt.rc_context({
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "axes.grid": False,
        "savefig.transparent": False,
    }):
        fig, ax = plt.subplots(figsize=(6.4, 5.2), layout="constrained")
        mesh = ax.pcolormesh(x, y, speed, shading="nearest", cmap="viridis")
        ax.streamplot(x, y, u, v, color="white", density=1.25, linewidth=0.7,
                      arrowsize=0.8)
        fig.colorbar(mesh, ax=ax, label=r"Velocity magnitude $|\mathbf{u}|/U_{lid}$")
        ax.set(xlabel=r"$x/L$", ylabel=r"$y/L$", aspect="equal",
               title=f"Lid-driven cavity, Re={summary['reynolds_number']:.0f}, tU/L={summary['final_time']:.1f}")
        save(fig, "velocity_streamlines")

        fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.8), layout="constrained")
        u_coord, u_value = profiles["u_at_x_0.5"]
        v_coord, v_value = profiles["v_at_y_0.5"]
        axes[0].plot(u_value, u_coord, color="#0072B2", marker="o", markersize=2.8,
                     linewidth=1.2, label=r"$u/U_{lid}$")
        axes[0].axvline(0.0, color="#555555", linewidth=0.8, linestyle="--")
        axes[0].set(xlabel=r"$u/U_{lid}$", ylabel=r"$y/L$", title=r"Vertical centerline, $x/L=0.5$")
        axes[1].plot(v_coord, v_value, color="#D55E00", marker="s", markersize=2.6,
                     linewidth=1.2, label=r"$v/U_{lid}$")
        axes[1].axhline(0.0, color="#555555", linewidth=0.8, linestyle="--")
        axes[1].set(xlabel=r"$x/L$", ylabel=r"$v/U_{lid}$", title=r"Horizontal centerline, $y/L=0.5$")
        for axis in axes:
            axis.grid(True, color="#D0D0D0", linewidth=0.6)
        save(fig, "centerlines")

        fig, axes = plt.subplots(3, 1, figsize=(7.0, 7.2), sharex=True, layout="constrained")
        time = history["time"]
        axes[0].semilogy(time, np.maximum(history["post_projection_max_abs_divergence"], 1e-30),
                        color="#0072B2", linewidth=1.1, label=r"$\|\nabla\cdot\mathbf{u}\|_\infty$")
        axes[0].semilogy(time, np.maximum(np.abs(history["net_mass_flux"]), 1e-30),
                        color="#D55E00", linewidth=1.0, linestyle="--", label="absolute net wall flux")
        axes[0].set(ylabel="Divergence / flux\n(nondimensional)")
        axes[0].legend(frameon=False, fontsize=8)
        axes[1].plot(time, history["kinetic_energy"], color="#009E73", linewidth=1.2)
        axes[1].set(ylabel=r"Kinetic energy $E_k$")
        axes[2].plot(time, history["center_u"], color="#0072B2", linewidth=1.1, label=r"center-cell $u/U_{lid}$")
        axes[2].plot(time, history["center_v"], color="#D55E00", linewidth=1.0,
                     linestyle="--", label=r"center-cell $v/U_{lid}$")
        axes[2].set(xlabel=r"Time $tU_{lid}/L$", ylabel="Representative\nvelocity")
        axes[2].legend(frameon=False, fontsize=8)
        for axis in axes:
            axis.grid(True, color="#D0D0D0", linewidth=0.6)
        save(fig, "time_history")

    manifest = {
        "schema": "vwis-cavity-figure-manifest-v1",
        "sources": ["raw/summary.json", "raw/final_field.csv", "raw/centerlines.csv", "raw/history.csv"],
        "transformations": [
            "reshape final cell-centred CSV rows to the recorded Ny by Nx grid",
            "Matplotlib streamplot interpolation used only to integrate streamline paths",
            "no field smoothing, filtering, averaging, or resampling",
            "semilog display floors exact plotted zeros at 1e-30; raw values remain in history.csv",
        ],
        "centerline_method": summary["centerline_method"],
        "matplotlib_version": matplotlib.__version__,
        "numpy_version": np.__version__,
        "outputs": [
            "figures/velocity_streamlines.png", "figures/velocity_streamlines.svg",
            "figures/centerlines.png", "figures/centerlines.svg",
            "figures/time_history.png", "figures/time_history.svg",
        ],
    }
    with (RAW_DIR / "figure_manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
