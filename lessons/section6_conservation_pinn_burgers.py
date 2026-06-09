"""Section 6 lab: baseline PINN vs Conservation-PINN for viscous Burgers.

This file uses ``# %%`` cells, so it can be opened as a notebook-like script in
VS Code or run from the command line:

    python section6_conservation_pinn_burgers.py --quick
    python section6_conservation_pinn_burgers.py --epochs 3000
"""

# %% Imports and configuration
from __future__ import annotations

import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn


@dataclass
class Config:
    nu: float = 0.01 / np.pi
    train_t_max: float = 1.0
    test_t_max: float = 2.0
    epochs: int = 3000
    learning_rate: float = 1e-3
    n_ic: int = 128
    n_bc: int = 128
    n_f: int = 3000
    n_cons_t: int = 50
    n_cons_x: int = 200
    hidden_dim: int = 64
    num_hidden: int = 4
    lambda_ic: float = 10.0
    lambda_bc: float = 1.0
    lambda_f: float = 1.0
    lambda_cons: float = 1.0
    seed: int = 42
    print_every: int = 100
    ref_nx: int = 256
    ref_dt: float = 5e-4
    device: str = "auto"
    output_dir: str = "results/section6_burgers"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(preference: str = "auto") -> torch.device:
    """Resolve auto/mps/cuda/cpu while giving useful availability errors."""
    preference = preference.lower()
    if preference not in {"auto", "mps", "cuda", "cpu"}:
        raise ValueError("device must be one of: auto, mps, cuda, cpu")
    if preference == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available in this PyTorch build")
    if preference == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if preference != "auto":
        return torch.device(preference)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def initial_condition(x):
    """Smooth periodic initial condition with exact mass M0 = 1."""
    if isinstance(x, torch.Tensor):
        return 1.0 + 0.5 * torch.sin(2.0 * torch.pi * x)
    return 1.0 + 0.5 * np.sin(2.0 * np.pi * x)


# %% A periodic pseudo-spectral reference solver
def solve_reference(
    times: np.ndarray, nu: float, nx: int = 256, dt: float = 5e-4
) -> tuple[np.ndarray, np.ndarray]:
    """Solve viscous Burgers on [0, 1) using spectral derivatives and RK4."""
    times = np.asarray(times, dtype=np.float64)
    if times.ndim != 1 or np.any(np.diff(times) < 0):
        raise ValueError("times must be a sorted one-dimensional array")

    x = np.arange(nx, dtype=np.float64) / nx
    wave_numbers = 2.0 * np.pi * np.fft.fftfreq(nx, d=1.0 / nx)
    cutoff = nx // 3
    dealias_mask = np.abs(np.fft.fftfreq(nx) * nx) <= cutoff
    u = initial_condition(x)

    def rhs(values: np.ndarray) -> np.ndarray:
        u_hat = np.fft.fft(values)
        flux_hat = np.fft.fft(0.5 * values**2)
        flux_hat[~dealias_mask] = 0.0
        flux_x = np.fft.ifft(1j * wave_numbers * flux_hat).real
        u_xx = np.fft.ifft(-(wave_numbers**2) * u_hat).real
        return -flux_x + nu * u_xx

    snapshots = np.empty((len(times), nx), dtype=np.float64)
    current_t = 0.0
    for index, target_t in enumerate(times):
        while current_t < target_t - 1e-14:
            step = min(dt, target_t - current_t)
            k1 = rhs(u)
            k2 = rhs(u + 0.5 * step * k1)
            k3 = rhs(u + 0.5 * step * k2)
            k4 = rhs(u + step * k3)
            u = u + step * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
            current_t += step
        snapshots[index] = u
    return x, snapshots


# %% PINN model and automatic differentiation
class PINN(nn.Module):
    def __init__(self, hidden_dim: int = 64, num_hidden: int = 4):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(2, hidden_dim), nn.Tanh()]
        for _ in range(num_hidden - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, t], dim=1))


def gradient(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(
        y,
        x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        retain_graph=True,
    )[0]


def burgers_residual(
    model: PINN, x: torch.Tensor, t: torch.Tensor, nu: float
) -> torch.Tensor:
    """f = u_t + u u_x - nu u_xx."""
    u = model(x, t)
    u_t = gradient(u, t)
    u_x = gradient(u, x)
    u_xx = gradient(u_x, x)
    return u_t + u * u_x - nu * u_xx


def periodic_bc_loss(model: PINN, t: torch.Tensor) -> torch.Tensor:
    """Enforce both u(0,t)=u(1,t) and u_x(0,t)=u_x(1,t)."""
    x_left = torch.zeros_like(t, requires_grad=True)
    x_right = torch.ones_like(t, requires_grad=True)
    u_left = model(x_left, t)
    u_right = model(x_right, t)
    ux_left = gradient(u_left, x_left)
    ux_right = gradient(u_right, x_right)
    return torch.mean((u_left - u_right) ** 2) + torch.mean(
        (ux_left - ux_right) ** 2
    )


def conservation_loss(
    model: PINN, times: torch.Tensor, n_x: int, target_mass: float = 1.0
) -> torch.Tensor:
    """Approximate M(t)=integral_0^1 u(x,t) dx on uniform periodic points."""
    x = torch.arange(n_x, device=times.device, dtype=times.dtype) / n_x
    x = x.reshape(1, -1).expand(len(times), -1).reshape(-1, 1)
    t = times.reshape(-1, 1).expand(-1, n_x).reshape(-1, 1)
    mass = model(x, t).reshape(len(times), n_x).mean(dim=1)
    return torch.mean((mass - target_mass) ** 2)


# %% Training
def train_model(
    name: str, config: Config, device: torch.device, lambda_cons: float
) -> tuple[PINN, dict[str, list[float]]]:
    set_seed(config.seed)
    model = PINN(config.hidden_dim, config.num_hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    history = {key: [] for key in ["total", "ic", "bc", "pde", "cons"]}

    for epoch in range(1, config.epochs + 1):
        optimizer.zero_grad()

        x_ic = torch.rand(config.n_ic, 1, device=device)
        t_ic = torch.zeros_like(x_ic)
        loss_ic = torch.mean((model(x_ic, t_ic) - initial_condition(x_ic)) ** 2)

        t_bc = config.train_t_max * torch.rand(config.n_bc, 1, device=device)
        loss_bc = periodic_bc_loss(model, t_bc)

        x_f = torch.rand(config.n_f, 1, device=device, requires_grad=True)
        t_f = (
            config.train_t_max
            * torch.rand(config.n_f, 1, device=device, requires_grad=True)
        )
        residual = burgers_residual(model, x_f, t_f, config.nu)
        loss_pde = torch.mean(residual**2)

        t_cons = config.train_t_max * torch.rand(config.n_cons_t, 1, device=device)
        if lambda_cons > 0.0:
            loss_cons = conservation_loss(model, t_cons, config.n_cons_x)
        else:
            # Keep a diagnostic value for the baseline without building its
            # conservation-loss graph into backpropagation.
            with torch.no_grad():
                loss_cons = conservation_loss(model, t_cons, config.n_cons_x)

        loss = (
            config.lambda_ic * loss_ic
            + config.lambda_bc * loss_bc
            + config.lambda_f * loss_pde
            + lambda_cons * loss_cons
        )
        loss.backward()
        optimizer.step()

        values = [loss, loss_ic, loss_bc, loss_pde, loss_cons]
        for key, value in zip(history, values):
            history[key].append(float(value.detach().cpu()))

        if epoch == 1 or epoch % config.print_every == 0:
            print(
                f"{name:>13s} | epoch {epoch:5d}/{config.epochs} | "
                f"loss={history['total'][-1]:.3e} | "
                f"pde={history['pde'][-1]:.3e} | "
                f"cons={history['cons'][-1]:.3e}"
            )
    return model, history


# %% Evaluation helpers
@torch.no_grad()
def predict(
    model: PINN, x: np.ndarray, times: np.ndarray, device: torch.device
) -> np.ndarray:
    xx, tt = np.meshgrid(x, times)
    inputs_x = torch.tensor(xx.reshape(-1, 1), dtype=torch.float32, device=device)
    inputs_t = torch.tensor(tt.reshape(-1, 1), dtype=torch.float32, device=device)
    values = model(inputs_x, inputs_t).cpu().numpy()
    return values.reshape(len(times), len(x))


def metrics(prediction: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    mass_error = np.abs(prediction.mean(axis=1) - 1.0)
    return {
        "relative_l2": float(np.linalg.norm(prediction - reference) / np.linalg.norm(reference)),
        "mean_cons_error": float(mass_error.mean()),
        "max_cons_error": float(mass_error.max()),
        "final_cons_error": float(mass_error[-1]),
    }


def save_metrics(rows: list[dict[str, float | str]], output_dir: Path) -> None:
    fieldnames = [
        "method",
        "relative_l2",
        "mean_cons_error",
        "max_cons_error",
        "final_cons_error",
    ]
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# %% Figures required by the lesson
def plot_training(
    histories: dict[str, dict[str, list[float]]], output_dir: Path
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, history in histories.items():
        axes[0].semilogy(history["total"], label=name)
        axes[1].semilogy(history["cons"], label=name)
    axes[0].set(title="Total training loss", xlabel="epoch", ylabel="loss")
    axes[1].set(title="Mass loss during training", xlabel="epoch", ylabel="loss")
    for axis in axes:
        axis.grid(alpha=0.3)
        axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "01_training_loss.png", dpi=180)
    plt.close(fig)


def plot_solution_fields(
    x: np.ndarray,
    times: np.ndarray,
    reference: np.ndarray,
    predictions: dict[str, np.ndarray],
    output_dir: Path,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True, sharey=True)
    extent = [x[0], x[-1], times[-1], times[0]]
    fields = [reference, predictions["Baseline"], predictions["Conservation"]]
    titles = ["Reference", "Baseline PINN", "Conservation-PINN"]
    vmax = max(np.max(np.abs(field - reference)) for field in fields[1:])
    for column, (field, title) in enumerate(zip(fields, titles)):
        image = axes[0, column].imshow(field, aspect="auto", extent=extent, cmap="viridis")
        axes[0, column].set_title(title)
        fig.colorbar(image, ax=axes[0, column])
        error = np.abs(field - reference)
        image = axes[1, column].imshow(
            error, aspect="auto", extent=extent, cmap="magma", vmin=0.0, vmax=vmax
        )
        axes[1, column].set_title(f"{title} absolute error")
        fig.colorbar(image, ax=axes[1, column])
    for axis in axes[:, 0]:
        axis.set_ylabel("t")
    for axis in axes[-1]:
        axis.set_xlabel("x")
    fig.tight_layout()
    fig.savefig(output_dir / "02_solution_and_error.png", dpi=180)
    plt.close(fig)


def plot_mass_drift(
    times: np.ndarray, predictions: dict[str, np.ndarray], output_dir: Path
) -> None:
    fig, axis = plt.subplots(figsize=(7, 4))
    for name, prediction in predictions.items():
        axis.semilogy(times, np.abs(prediction.mean(axis=1) - 1.0) + 1e-16, label=name)
    axis.axvline(1.0, color="black", linestyle="--", linewidth=1, label="training limit")
    axis.set(xlabel="t", ylabel="|M(t) - M(0)|", title="Mass conservation error")
    axis.grid(alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "03_mass_drift.png", dpi=180)
    plt.close(fig)


def plot_time_slices(
    x: np.ndarray,
    times: np.ndarray,
    reference: np.ndarray,
    predictions: dict[str, np.ndarray],
    output_dir: Path,
) -> None:
    requested = [0.0, 0.5, 1.0, 1.5, 2.0]
    indices = [int(np.argmin(np.abs(times - value))) for value in requested]
    fig, axes = plt.subplots(1, len(indices), figsize=(18, 3.5), sharey=True)
    for axis, index in zip(axes, indices):
        axis.plot(x, reference[index], color="black", linewidth=2, label="Reference")
        for name, prediction in predictions.items():
            axis.plot(x, prediction[index], linestyle="--", label=name)
        axis.set_title(f"t = {times[index]:.1f}")
        axis.set_xlabel("x")
        axis.grid(alpha=0.3)
    axes[0].set_ylabel("u(x,t)")
    axes[-1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "04_time_slices.png", dpi=180)
    plt.close(fig)


# %% Complete experiment
def run_experiment(config: Config) -> None:
    set_seed(config.seed)
    device = select_device(config.device)
    output_dir = Path(__file__).resolve().parent / config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"device: {device}")
    print(f"results: {output_dir}")

    baseline, baseline_history = train_model("Baseline", config, device, lambda_cons=0.0)
    conservation, conservation_history = train_model(
        "Conservation", config, device, lambda_cons=config.lambda_cons
    )

    times = np.linspace(0.0, config.test_t_max, 101)
    x, reference = solve_reference(times, config.nu, config.ref_nx, config.ref_dt)
    predictions = {
        "Baseline": predict(baseline, x, times, device),
        "Conservation": predict(conservation, x, times, device),
    }

    rows: list[dict[str, float | str]] = []
    for name, prediction in predictions.items():
        row: dict[str, float | str] = {"method": name}
        row.update(metrics(prediction, reference))
        rows.append(row)
        print(name, {key: f"{value:.3e}" for key, value in row.items() if key != "method"})

    save_metrics(rows, output_dir)
    plot_training(
        {"Baseline": baseline_history, "Conservation": conservation_history}, output_dir
    )
    plot_solution_fields(x, times, reference, predictions, output_dir)
    plot_mass_drift(times, predictions, output_dir)
    plot_time_slices(x, times, reference, predictions, output_dir)
    torch.save(baseline.state_dict(), output_dir / "baseline_pinn.pt")
    torch.save(conservation.state_dict(), output_dir / "conservation_pinn.pt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=None, help="training epochs per model")
    parser.add_argument("--lambda-cons", type=float, default=1.0)
    parser.add_argument("--device", choices=["auto", "mps", "cuda", "cpu"], default="auto")
    parser.add_argument("--quick", action="store_true", help="small CPU smoke-test experiment")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = Config(lambda_cons=args.lambda_cons, device=args.device)
    if args.lambda_cons != 1.0:
        label = str(args.lambda_cons).replace(".", "p")
        cfg.output_dir = f"results/section6_burgers_lambda_{label}"
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.quick:
        cfg.epochs = args.epochs or 10
        cfg.n_ic = 32
        cfg.n_bc = 32
        cfg.n_f = 128
        cfg.n_cons_t = 8
        cfg.n_cons_x = 32
        cfg.hidden_dim = 32
        cfg.num_hidden = 3
        cfg.print_every = 1
        cfg.ref_nx = 128
        cfg.ref_dt = 1e-3
        cfg.output_dir = "results/section6_burgers_quick"
    run_experiment(cfg)
