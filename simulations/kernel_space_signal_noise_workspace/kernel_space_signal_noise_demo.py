from __future__ import annotations

"""
Kernel-space signal-vs-noise demo.

This workspace intentionally focuses on two high-dimensional regimes:

1. Strong kernel-space covariance signal:
   A dense same-mean / different-covariance case in dimension 220.
   The feature-space covariance gap is at least competitive with its estimation
   error, and KDMLP can beat KFDA.

2. Weak kernel-space covariance signal:
   A dense mean-gap / tiny-covariance-gap case in dimension 220.
   The feature-space covariance gap is smaller than the estimation error, so
   the covariance part is not trustworthy on its own.

The point is to show the two stories side by side with the same stage-1/stage-2
pipeline, not to introduce another approximate implementation.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_DIR = ROOT.parent / "redo_projection_workspace"
sys.path.insert(0, str(BASE_DIR))
import redo_projection_cv_experiment as base  # noqa: E402


sns.set_theme(style="whitegrid", context="talk")

R_VALUES = base.R_VALUES


def build_hd220_regimes(seed: int = 20260327):
    rng = np.random.default_rng(seed)
    p = 220

    sigma_strong_0 = base.random_dense_spd(rng, p, eig_low=0.6, eig_high=2.2)
    sigma_weak_0 = base.random_dense_spd(rng, p, eig_low=0.6, eig_high=2.1)

    strong_T = base.dense_congruence_transform(rng, p, strength=0.95)
    weak_T = base.dense_congruence_transform(rng, p, strength=0.12)

    sigma_strong_1 = base.matched_trace_congruence(sigma_strong_0, strong_T)
    sigma_weak_1 = base.matched_trace_congruence(sigma_weak_0, weak_T)

    v = base.random_unit_vector(rng, p)

    strong = base.Regime(
        name="strong220",
        title="Strong Case: d=220, Same Mean, Different Dense Covariances",
        p=p,
        n_train_per_class=660,
        n_test_per_class=2500,
        n_ref_per_class=3000,
        mean0=np.zeros(p),
        mean1=np.zeros(p),
        Sigma0=sigma_strong_0,
        Sigma1=sigma_strong_1,
        note="High-dimensional dense covariance difference with matched trace; no mean difference.",
    )
    weak = base.Regime(
        name="weak220",
        title="Weak Case: d=220, Mean Gap Plus Tiny Dense Covariance Gap",
        p=p,
        n_train_per_class=660,
        n_test_per_class=2500,
        n_ref_per_class=3000,
        mean0=-0.55 * v,
        mean1=0.55 * v,
        Sigma0=sigma_weak_0,
        Sigma1=sigma_weak_1,
        note="High-dimensional case where the covariance gap is real but weaker than its estimation error.",
    )
    return strong, weak


def run_strong_bestkernel_case(repeats: int = 1, seed: int = 123):
    regime, _ = build_hd220_regimes()
    df_raw, df_sel, df_cov, plot_payload = base.run_regime(
        regime,
        repeats=repeats,
        anchor_per_class=40,
        reg=1e-6,
        seed=seed,
    )
    df_acc = base.summarize_accuracy(df_raw)
    df_cov_summary = base.summarize_covariance(df_cov)
    df_cov_summary["signal_to_noise_ratio"] = (
        df_cov_summary["ref_D_sigma_feature"] / df_cov_summary["cov_gap_error_feature"]
    )
    return regime, df_raw, df_sel, df_acc, df_cov, df_cov_summary, plot_payload


def run_weak_bestkernel_case(repeats: int = 1, seed: int = 123):
    _, regime = build_hd220_regimes()
    df_raw, df_sel, df_cov, plot_payload = base.run_regime(
        regime,
        repeats=repeats,
        anchor_per_class=40,
        reg=1e-6,
        seed=seed,
    )
    df_acc = base.summarize_accuracy(df_raw)
    df_cov_summary = base.summarize_covariance(df_cov)
    df_cov_summary["signal_to_noise_ratio"] = (
        df_cov_summary["ref_D_sigma_feature"] / df_cov_summary["cov_gap_error_feature"]
    )
    return regime, df_raw, df_sel, df_acc, df_cov, df_cov_summary, plot_payload


def collapse_by_kernel(df_cov_summary: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "ref_D_mu_feature",
        "emp_D_mu_feature",
        "ref_D_sigma_feature",
        "emp_D_sigma_feature",
        "cov_gap_error_feature",
        "cov0_error_feature",
        "cov1_error_feature",
        "true_mean_gap_input",
        "emp_mean_gap_input",
        "true_cov_gap_input",
        "emp_cov_gap_input",
        "cov_gap_error_input",
        "cov0_error_input",
        "cov1_error_input",
    ]
    out = df_cov_summary.groupby("kernel", as_index=False)[cols].mean()
    out["signal_to_noise_ratio"] = out["ref_D_sigma_feature"] / out["cov_gap_error_feature"]
    return out


def plot_accuracy_curves(strong_acc: pd.DataFrame, weak_acc: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    colors = {"KFDA-1D": "#111827", "MY-large_mu": "#b91c1c", "MY-small_mu": "#1d4ed8"}

    # Strong best-kernel case.
    ax = axes[0]
    sub = strong_acc.copy()
    kfda = sub[sub["variant"] == "KFDA-1D"].iloc[0]
    ax.axhline(float(kfda["mean"]), color=colors["KFDA-1D"], linestyle="--", linewidth=2.2, label="KFDA-1D")
    for variant in ("MY-large_mu", "MY-small_mu"):
        cur = sub[sub["variant"] == variant].sort_values("r")
        ax.plot(cur["r"], cur["mean"], marker="o", linewidth=2.2, color=colors[variant], label=variant)
        ax.fill_between(cur["r"], cur["mean"] - cur["std"], cur["mean"] + cur["std"], alpha=0.18, color=colors[variant])
    ax.set_title("Strong Case\nBest nested-selected stage-1 kernel")
    ax.set_xlabel("projection dimension r")
    ax.set_ylabel("test accuracy")
    ax.set_xticks(list(R_VALUES))
    ax.legend(frameon=True, fontsize=9)

    # Weak best-kernel case.
    ax = axes[1]
    sub = weak_acc.copy()
    kfda = sub[sub["variant"] == "KFDA-1D"].iloc[0]
    ax.axhline(float(kfda["mean"]), color=colors["KFDA-1D"], linestyle="--", linewidth=2.2, label="KFDA-1D")
    for variant in ("MY-large_mu", "MY-small_mu"):
        cur = sub[sub["variant"] == variant].sort_values("r")
        ax.plot(cur["r"], cur["mean"], marker="o", linewidth=2.2, color=colors[variant], label=variant)
        ax.fill_between(cur["r"], cur["mean"] - cur["std"], cur["mean"] + cur["std"], alpha=0.18, color=colors[variant])
    ax.set_title("Weak Case\nCovariance signal below noise floor")
    ax.set_xlabel("projection dimension r")
    ax.set_xticks(list(R_VALUES))
    ax.legend(frameon=True, fontsize=9)

    fig.tight_layout()
    out = OUT_DIR / "accuracy_curves.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_signal_vs_noise(strong_cov: pd.DataFrame, weak_cov: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=False)

    def one_panel(ax, df, title, highlight_kernel=None):
        sub = df.copy().sort_values("signal_to_noise_ratio", ascending=False)
        x = np.arange(len(sub))
        ax.bar(x - 0.18, sub["ref_D_sigma_feature"], width=0.36, label=r"$D_\sigma^\phi$", color="#2563eb")
        ax.bar(x + 0.18, sub["cov_gap_error_feature"], width=0.36, label="covariance error", color="#dc2626")
        ax.set_xticks(x)
        ax.set_xticklabels(sub["kernel"], rotation=30, ha="right")
        ax.set_title(title)
        ax.set_ylabel("feature-space magnitude")
        ax.legend(frameon=True, fontsize=10)
        for i, (_, row) in enumerate(sub.iterrows()):
            ax.text(i, max(row["ref_D_sigma_feature"], row["cov_gap_error_feature"]) * 1.02, f"{row['signal_to_noise_ratio']:.2f}",
                    ha="center", va="bottom", fontsize=9)
        if highlight_kernel is not None:
            for tick, kern in zip(ax.get_xticklabels(), sub["kernel"]):
                if kern == highlight_kernel:
                    tick.set_color("#7c3aed")
                    tick.set_fontweight("bold")

    one_panel(axes[0], strong_cov, "Strong case: feature covariance signal vs error")
    one_panel(axes[1], weak_cov, "Weak case: feature covariance signal vs error")

    fig.tight_layout()
    out = OUT_DIR / "feature_space_signal_vs_noise.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_projection_views(strong_payload: dict, strong_acc: pd.DataFrame, weak_payload: dict, weak_acc: pd.DataFrame):
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Strong case: fixed poly2
    best_large = strong_acc[strong_acc["variant"] == "MY-large_mu"].sort_values("mean", ascending=False).iloc[0]
    best_small = strong_acc[strong_acc["variant"] == "MY-small_mu"].sort_values("mean", ascending=False).iloc[0]
    panels = [("KFDA-1D", 1), ("MY-large_mu", int(best_large["r"])), ("MY-small_mu", int(best_small["r"]))]

    for ax, (variant, r) in zip(axes[0], panels):
        payload = strong_payload[(variant, r)]
        acc_val = float(
            strong_acc[(strong_acc["variant"] == variant) & (strong_acc["r"] == r)]["mean"].iloc[0]
        )
        Z2 = base.project_to_plot_2d(payload["Z_show"])
        y = payload["y_show"]
        for cls, color, label in [(0, "#1f77b4", "class 0"), (1, "#d62728", "class 1")]:
            idx = y == cls
            ax.scatter(Z2[idx, 0], Z2[idx, 1], s=10, alpha=0.55, c=color, label=label)
        ax.set_title(f"Strong case\n{variant}, r={r}\nacc={acc_val:.3f}")
        ax.set_xlabel("2D view 1")
        ax.set_ylabel("2D view 2")
    axes[0, 0].legend(frameon=True, fontsize=9)

    # Weak case: best nested-selected kernels from repeat 1 payload.
    best_large = weak_acc[weak_acc["variant"] == "MY-large_mu"].sort_values("mean", ascending=False).iloc[0]
    best_small = weak_acc[weak_acc["variant"] == "MY-small_mu"].sort_values("mean", ascending=False).iloc[0]
    panels = [("KFDA-1D", 1), ("MY-large_mu", int(best_large["r"])), ("MY-small_mu", int(best_small["r"]))]

    for ax, (variant, r) in zip(axes[1], panels):
        payload = weak_payload[(variant, r)]
        acc_val = float(
            weak_acc[(weak_acc["variant"] == variant) & (weak_acc["r"] == r)]["mean"].iloc[0]
        )
        Z2 = base.project_to_plot_2d(payload["Z_show"])
        y = payload["y_show"]
        for cls, color, label in [(0, "#1f77b4", "class 0"), (1, "#d62728", "class 1")]:
            idx = y == cls
            ax.scatter(Z2[idx, 0], Z2[idx, 1], s=10, alpha=0.55, c=color, label=label)
        ax.set_title(f"Weak case\n{variant}, r={r}\nacc={acc_val:.3f}\n{payload['stage1_label']}")
        ax.set_xlabel("2D view 1")
        ax.set_ylabel("2D view 2")

    fig.tight_layout()
    out = OUT_DIR / "projection_views.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    strong_regime, df_strong_raw, df_strong_sel, df_strong_acc, df_strong_cov, df_strong_cov_summary, strong_payload = run_strong_bestkernel_case()
    weak_regime, df_weak_raw, df_weak_sel, df_weak_acc, df_weak_cov, df_weak_cov_summary, weak_payload = run_weak_bestkernel_case()

    # Keep only one row per kernel after averaging repeats.
    strong_cov_export = collapse_by_kernel(df_strong_cov_summary)
    weak_cov_export = collapse_by_kernel(df_weak_cov_summary)
    strong_cov_export["case"] = "strong_kernel_covariance_signal"
    weak_cov_export["case"] = "weak_kernel_covariance_signal"
    df_signal_noise = pd.concat([strong_cov_export, weak_cov_export], ignore_index=True)

    # Save tables.
    df_strong_raw.to_csv(OUT_DIR / "strong_bestkernel_test_results.csv", index=False)
    df_strong_sel.to_csv(OUT_DIR / "strong_bestkernel_selection_log.csv", index=False)
    df_strong_acc.to_csv(OUT_DIR / "strong_bestkernel_accuracy_summary.csv", index=False)
    df_strong_cov.to_csv(OUT_DIR / "strong_bestkernel_covariance_diagnostics_per_repeat.csv", index=False)
    df_weak_raw.to_csv(OUT_DIR / "weak_bestkernel_test_results.csv", index=False)
    df_weak_sel.to_csv(OUT_DIR / "weak_bestkernel_selection_log.csv", index=False)
    df_weak_acc.to_csv(OUT_DIR / "weak_bestkernel_accuracy_summary.csv", index=False)
    df_weak_cov.to_csv(OUT_DIR / "weak_bestkernel_covariance_diagnostics_per_repeat.csv", index=False)
    df_signal_noise.to_csv(OUT_DIR / "feature_space_signal_noise_summary.csv", index=False)

    pd.DataFrame(
        [
            {
                "case": strong_regime.name,
                "title": strong_regime.title,
                "dimension": strong_regime.p,
                "mean_gap_input": float(np.linalg.norm(strong_regime.mean1 - strong_regime.mean0)),
                "cov_gap_input": float(np.linalg.norm(strong_regime.Sigma1 - strong_regime.Sigma0, ord="fro")),
                "note": strong_regime.note,
            },
            {
                "case": weak_regime.name,
                "title": weak_regime.title,
                "dimension": weak_regime.p,
                "mean_gap_input": float(np.linalg.norm(weak_regime.mean1 - weak_regime.mean0)),
                "cov_gap_input": float(np.linalg.norm(weak_regime.Sigma1 - weak_regime.Sigma0, ord="fro")),
                "note": weak_regime.note,
            },
        ]
    ).to_csv(OUT_DIR / "case_setup.csv", index=False)

    acc_plot = plot_accuracy_curves(df_strong_acc, df_weak_acc)
    sig_plot = plot_signal_vs_noise(strong_cov_export, weak_cov_export)
    proj_plot = plot_projection_views(strong_payload, df_strong_acc, weak_payload, df_weak_acc)

    print("Saved outputs:")
    print(f"- {OUT_DIR / 'strong_bestkernel_accuracy_summary.csv'}")
    print(f"- {OUT_DIR / 'weak_bestkernel_accuracy_summary.csv'}")
    print(f"- {OUT_DIR / 'feature_space_signal_noise_summary.csv'}")
    print(f"- {OUT_DIR / 'case_setup.csv'}")
    print(f"- {acc_plot}")
    print(f"- {sig_plot}")
    print(f"- {proj_plot}")


if __name__ == "__main__":
    main()
