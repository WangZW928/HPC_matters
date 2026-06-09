"""Build self-contained notebooks by combining each lesson's Markdown and code.

This helper uses only Python's standard library so the notebooks can be rebuilt
before NumPy, PyTorch, or nbformat are installed.
"""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def lines(text: str) -> list[str]:
    return text.splitlines(keepends=True)


def markdown_sections(text: str) -> list[str]:
    """Split lesson Markdown at level-one headings outside fenced code blocks."""
    sections: list[list[str]] = []
    current: list[str] = []
    in_fence = False
    for line in lines(text):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
        if line.startswith("# ") and current and not in_fence:
            sections.append(current)
            current = []
        current.append(line)
    if current:
        sections.append(current)
    return ["".join(section).strip() + "\n" for section in sections]


def code_sections(text: str) -> list[tuple[str, str]]:
    """Split a percent-format Python script into named notebook code cells."""
    matches = list(re.finditer(r"^# %%\s*(.*)$", text, flags=re.MULTILINE))
    sections: list[tuple[str, str]] = []
    for index, match in enumerate(matches):
        start = match.end() + 1
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        source = text[start:end].strip()
        if "def parse_args()" in source:
            source = source.split("def parse_args()", 1)[0].rstrip()
        source = source.replace(
            'Path(__file__).resolve().parent / config.output_dir',
            'Path.cwd() / config.output_dir',
        )
        source = source.replace(
            'Path(__file__).resolve().parent / "results" / output_name',
            'Path.cwd() / "results" / output_name',
        )
        sections.append((match.group(1).strip(), source + "\n"))
    return sections


def markdown_cell(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": lines(source)}


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines(source),
    }


def build_notebook(
    markdown_name: str,
    script_name: str,
    notebook_name: str,
    run_cell: str,
) -> None:
    markdown = (ROOT / markdown_name).read_text(encoding="utf-8-sig")
    script = (ROOT / script_name).read_text(encoding="utf-8")
    cells: list[dict] = []

    for section in markdown_sections(markdown):
        cells.append(markdown_cell(section))

    cells.append(
        markdown_cell(
            "# 配套实验：可逐单元运行的完整代码\n\n"
            "下面的代码单元与讲义内容配套。首次运行建议保留 "
            "`RUN_QUICK = True`，确认环境和流程正确后再运行正式实验。\n"
        )
    )
    for title, source in code_sections(script):
        cells.append(markdown_cell(f"## 实验单元：{title}\n"))
        cells.append(code_cell(source))
    cells.append(markdown_cell("## 运行实验\n"))
    cells.append(code_cell(run_cell.strip() + "\n"))

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (ROOT / notebook_name).write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )


build_notebook(
    "section6_conservation_pinn_burgers.md",
    "section6_conservation_pinn_burgers.py",
    "section6_conservation_pinn_burgers.ipynb",
    """
RUN_QUICK = True

cfg = Config()
if RUN_QUICK:
    cfg.epochs = 10
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
""",
)

build_notebook(
    "section7_double_pendulum_hamiltonian.md",
    "section7_double_pendulum_hamiltonian.py",
    "section7_double_pendulum_hamiltonian.ipynb",
    """
RUN_QUICK = True

if RUN_QUICK:
    run_experiment(t_end=1.0, dt=0.02, output_name="section7_double_pendulum_quick")
else:
    run_experiment(t_end=10.0, dt=0.005, output_name="section7_double_pendulum_dt_0p005")
""",
)
