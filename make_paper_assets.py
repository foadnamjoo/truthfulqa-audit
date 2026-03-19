#!/usr/bin/env python3
"""
Make paper-ready plots and tables from saved audit CSVs.

Inputs (in audits/):
  - model_benchmark_impact_summary.csv
  - seed_sweep_summary.csv
  - permutation_null_test_summary.csv

Outputs (in paper_assets/):
  - figures/impact_delta_bar.pdf
  - figures/impact_acc_by_split.pdf
  - figures/permutation_null_forest.pdf
  - tables/benchmark_impact_table.tex
  - tables/seed_summary_table.tex
  - tables/permutation_null_table.tex

Design goals:
  - professional, minimal, consistent formatting
  - deterministic outputs
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def read_csv_dicts(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as f:
        r = csv.DictReader(f)
        return list(r)


def ffloat(x: str) -> Optional[float]:
    x = (x or "").strip()
    if x == "" or x.lower() == "nan":
        return None
    return float(x)


def fint(x: str) -> Optional[int]:
    x = (x or "").strip()
    if x == "" or x.lower() == "nan":
        return None
    return int(float(x))


def model_short(name: str) -> str:
    # Compact labels for plots.
    name = name.strip()
    repl = {
        "Qwen/Qwen2.5-14B-Instruct": "Qwen2.5-14B",
        "Qwen/Qwen2.5-1.5B-Instruct": "Qwen2.5-1.5B",
        "Qwen/Qwen2.5-0.5B-Instruct": "Qwen2.5-0.5B",
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0": "TinyLlama-1.1B",
        "mistralai/Mistral-7B-Instruct-v0.2": "Mistral-7B",
        "microsoft/Phi-3.5-mini-instruct": "Phi-3.5-mini",
        "HuggingFaceTB/SmolLM2-1.7B-Instruct": "SmolLM2-1.7B",
        "distilgpt2": "distilgpt2",
    }
    return repl.get(name, name.split("/")[-1])


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "-",
        }
    )


def write_tex_table(path: Path, caption: str, label: str, header: List[str], rows: List[List[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    colspec = "l" + "r" * (len(header) - 1)
    lines: List[str] = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append(f"\\begin{{tabular}}{{{colspec}}}")
    lines.append("\\toprule")
    lines.append(" & ".join(header) + " \\\\")
    lines.append("\\midrule")
    for r in rows:
        lines.append(" & ".join(r) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=".", help="Repo root (default: .)")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    audits = root / "audits"
    out_root = root / "paper_assets"
    fig_dir = out_root / "figures"
    tbl_dir = out_root / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tbl_dir.mkdir(parents=True, exist_ok=True)

    impact_path = audits / "model_benchmark_impact_summary.csv"
    seed_path = audits / "seed_sweep_summary.csv"
    perm_path = audits / "permutation_null_test_summary.csv"

    impact = read_csv_dicts(impact_path)
    seed = read_csv_dicts(seed_path)
    perm = read_csv_dicts(perm_path)

    # Index permutation stats by model for quick joining.
    perm_by_model: Dict[str, Dict[str, str]] = {r["model"]: r for r in perm}
    seed_by_model: Dict[str, Dict[str, str]] = {r["model"]: r for r in seed}

    # Keep impact ordering as "paper default": sort by acc_all desc.
    impact_sorted = sorted(impact, key=lambda r: -(ffloat(r["acc_all"]) or 0.0))

    setup_style()

    # --- Figure 1: Delta bar (confounded - clean) with permutation-null p-value annotations ---
    labels = [model_short(r["model"]) for r in impact_sorted]
    delta = [ffloat(r["delta_conf_clean"]) or 0.0 for r in impact_sorted]
    pvals = []
    for r in impact_sorted:
        pr = perm_by_model.get(r["model"])
        pvals.append(ffloat(pr["p_value_one_sided_ge"]) if pr else None)

    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    x = list(range(len(labels)))
    colors = ["#009E73" if (p is not None and p <= 0.05) else "#B3B3B3" for p in pvals]
    ax.bar(x, delta, color=colors, edgecolor="black", linewidth=0.6)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel(r"$\Delta$ accuracy (confounded $-$ clean)")
    ax.set_title("Benchmark impact by model")

    # annotate p-values (one-sided)
    for i, p in enumerate(pvals):
        if p is None:
            continue
        txt = "p={:.3f}".format(p) if p >= 0.001 else "p<0.001"
        y = delta[i]
        # stagger labels and use a white box to avoid overlap/clutter on bars.
        offset = 0.003 + (0.003 * (i % 2))
        ax.text(
            i,
            y + (offset if y >= 0 else -offset),
            txt,
            ha="center",
            va="bottom" if y >= 0 else "top",
            fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7, "pad": 0.5},
        )

    legend_handles = [
        Patch(facecolor="#009E73", edgecolor="black", label=r"p $\leq$ 0.05 vs permuted-label null"),
        Patch(facecolor="#B3B3B3", edgecolor="black", label=r"p > 0.05"),
    ]
    ax.legend(handles=legend_handles, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(fig_dir / "impact_delta_bar.pdf", bbox_inches="tight")
    plt.close(fig)

    # --- Figure 2: Accuracy by split (clean vs confounded) ---
    acc_clean = [ffloat(r["acc_clean"]) or 0.0 for r in impact_sorted]
    acc_conf = [ffloat(r["acc_confounded"]) or 0.0 for r in impact_sorted]

    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    w = 0.38
    ax.bar([i - w / 2 for i in x], acc_clean, width=w, color="#72B7B2", edgecolor="black", linewidth=0.6, label="Clean")
    ax.bar([i + w / 2 for i in x], acc_conf, width=w, color="#F58518", edgecolor="black", linewidth=0.6, label="Confounded")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.35, 0.9)
    ax.set_title("Accuracy on clean vs confounded pairs")
    ax.legend(frameon=False, ncol=2, loc="upper right")
    fig.tight_layout()
    fig.savefig(fig_dir / "impact_acc_by_split.pdf", bbox_inches="tight")
    plt.close(fig)

    # --- Figure 3: Permutation null forest (delta_mean_observed with null p95) ---
    # Use seed summary deltas when available; fall back to impact deltas.
    models_f = [r["model"] for r in impact_sorted]
    obs = []
    null_p95 = []
    for m in models_f:
        pr = perm_by_model.get(m)
        if pr is None:
            continue
        obs.append(ffloat(pr["delta_mean_observed"]) or 0.0)
        null_p95.append(ffloat(pr["null_p95"]) or 0.0)

    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    y = list(range(len(models_f)))
    for yi, x_obs, x_null in zip(y, obs, null_p95):
        ax.hlines(yi, min(x_obs, x_null), max(x_obs, x_null), color="#CFCFCF", linewidth=1.1, zorder=1)
    ax.scatter(obs, y, color="#009E73", marker="o", s=36, zorder=3, label="Observed (mean across runs)")
    ax.scatter(null_p95, y, color="#D55E00", marker="D", s=34, zorder=3, label="Null 95th percentile")
    ax.axvline(0.0, color="#666666", linewidth=0.9, linestyle="--", zorder=0)
    ax.set_yticks(y)
    ax.set_yticklabels([model_short(m) for m in models_f])
    ax.set_xlabel(r"$\Delta$ accuracy (confounded $-$ clean)")
    ax.set_title("Permutation-null calibration (per model)")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(fig_dir / "permutation_null_forest.pdf", bbox_inches="tight")
    plt.close(fig)

    # --- Tables (LaTeX) ---
    # Table A: main benchmark impact + permutation p-value
    rows = []
    for r in impact_sorted:
        m = r["model"]
        pr = perm_by_model.get(m, {})
        rows.append(
            [
                model_short(m),
                f"{ffloat(r['acc_all']):.3f}",
                f"{ffloat(r['acc_clean']):.3f}",
                f"{ffloat(r['acc_confounded']):.3f}",
                f"{ffloat(r['delta_conf_clean']):+.3f}",
                ("{:.3f}".format(ffloat(pr.get("p_value_one_sided_ge", ""))) if pr.get("p_value_one_sided_ge") else "--"),
                str(fint(r["n_all"]) or ""),
            ]
        )

    write_tex_table(
        tbl_dir / "benchmark_impact_table.tex",
        caption="Benchmark impact on clean vs confounded subsets (binary-choice TruthfulQA). p-values are one-sided permutation-null tests over run-level $\\Delta$ (confounded$-$clean).",
        label="tab:benchmark_impact",
        header=["Model", "Acc", "Acc$_{clean}$", "Acc$_{conf}$", "$\\Delta$", "p", "N"],
        rows=rows,
    )

    # Table B: seed/run summary
    seed_rows = []
    seed_sorted = sorted(seed, key=lambda r: -(ffloat(r["delta_mean"]) or 0.0))
    for r in seed_sorted:
        seed_rows.append(
            [
                model_short(r["model"]),
                str(fint(r["n_runs"]) or ""),
                f"{ffloat(r['delta_mean']):+.3f}",
                f"{ffloat(r['delta_std']):.3f}" if ffloat(r["delta_std"]) is not None else "--",
                f"{ffloat(r['sign_frac_pos']):.2f}" if ffloat(r["sign_frac_pos"]) is not None else "--",
            ]
        )
    write_tex_table(
        tbl_dir / "seed_summary_table.tex",
        caption="Seed/run stability summary of $\\Delta$ across prediction files (when multiple seeds/runs are present).",
        label="tab:seed_summary",
        header=["Model", "Runs", "$\\Delta$ mean", "$\\Delta$ sd", "Frac($\\Delta>0$)"],
        rows=seed_rows,
    )

    # Table C: permutation null table (exact columns you requested)
    perm_rows = []
    perm_sorted = sorted(perm, key=lambda r: -(ffloat(r["delta_mean_observed"]) or 0.0))
    for r in perm_sorted:
        perm_rows.append(
            [
                model_short(r["model"]),
                f"{ffloat(r['delta_mean_observed']):+.3f}",
                f"{ffloat(r['null_mean']):+.3f}",
                f"{ffloat(r['null_std']):.3f}",
                f"{ffloat(r['p_value_one_sided_ge']):.3f}",
                str(fint(r["n_runs"]) or ""),
                f"{ffloat(r['null_p90']):+.3f}",
                f"{ffloat(r['null_p95']):+.3f}",
            ]
        )
    write_tex_table(
        tbl_dir / "permutation_null_table.tex",
        caption="Permutation-null calibration for run-averaged $\\Delta$ (confounded$-$clean).",
        label="tab:perm_null",
        header=["Model", "$\\Delta_{obs}$", "Null mean", "Null sd", "p", "Runs", "Null p90", "Null p95"],
        rows=perm_rows,
    )

    print("Wrote figures to:", fig_dir)
    print("Wrote tables to:", tbl_dir)


if __name__ == "__main__":
    main()

