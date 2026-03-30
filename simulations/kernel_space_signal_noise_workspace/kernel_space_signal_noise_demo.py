from __future__ import annotations

"""
Feature-space signal-vs-noise demo.

This workspace studies two high-dimensional Gaussian regimes using the full
stage-1 kernel menu:

1. Large covariance case:
   same mean / different covariance, tuned so that some nonlinear kernels have
   a true feature-space covariance difference that is clearly larger than the
   corresponding estimation error.

2. Small covariance case:
   mean gap / small covariance gap, where the feature-space covariance
   difference stays below the estimation-noise floor across kernels.

The point is to show a clean feature-space story: we evaluate all stage-1
kernels, then summarize each case by the kernel that gives the strongest
feature covariance signal-to-error ratio.
"""

import argparse
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

R_VALUES = (1, 2, 3, 4)


def build_terminal_summary(
    df_strong_acc: pd.DataFrame,
    df_weak_acc: pd.DataFrame,
    df_case_summary: pd.DataFrame,
) -> str:
    strong_best = df_strong_acc[df_strong_acc["variant"] != "KFDA-1D"].sort_values("mean", ascending=False).iloc[0]
    strong_kfda = df_strong_acc[df_strong_acc["variant"] == "KFDA-1D"].iloc[0]
    weak_best = df_weak_acc[df_weak_acc["variant"] != "KFDA-1D"].sort_values("mean", ascending=False).iloc[0]
    weak_kfda = df_weak_acc[df_weak_acc["variant"] == "KFDA-1D"].iloc[0]
    row_a = df_case_summary[df_case_summary["label"] == "A"].iloc[0]
    row_b = df_case_summary[df_case_summary["label"] == "B"].iloc[0]
    return (
        "Summary: in the large covariance case, the representative kernel "
        f"({row_a['representative_kernel']}) has a true feature covariance difference larger than its estimation "
        f"error (ratio {row_a['signal_to_noise_ratio']:.3f}), and KDMLP improves over KFDA "
        f"({strong_best['mean']:.3f} vs {strong_kfda['mean']:.3f}). "
        "In the small covariance case, even the best feature-space signal/error ratio stays below 1 "
        f"({row_b['representative_kernel']}, ratio {row_b['signal_to_noise_ratio']:.3f}), so the covariance part "
        f"is below the noise floor; the best KDMLP run reaches {weak_best['mean']:.3f} versus {weak_kfda['mean']:.3f} "
        "for KFDA and should be interpreted more cautiously."
    )


def build_regimes(dimension: int = 220, seed: int = 20260327):
    rng = np.random.default_rng(seed)
    p = int(dimension)

    sigma_large_0 = base.random_dense_spd(rng, p, eig_low=0.5, eig_high=5.5)
    sigma_small_0 = base.random_dense_spd(rng, p, eig_low=0.6, eig_high=2.1)

    large_T = base.dense_congruence_transform(rng, p, strength=4.5)
    small_T = base.dense_congruence_transform(rng, p, strength=0.12)

    sigma_large_1 = base.matched_trace_congruence(sigma_large_0, large_T)
    sigma_small_1 = base.matched_trace_congruence(sigma_small_0, small_T)

    v = base.random_unit_vector(rng, p)

    large_case = base.Regime(
        name=f"largecov{p}",
        title=f"Large Covariance Case: d={p}, Same Mean, Different Covariances",
        p=p,
        n_train_per_class=1200,
        n_test_per_class=2500,
        n_ref_per_class=3000,
        mean0=np.zeros(p),
        mean1=np.zeros(p),
        Sigma0=sigma_large_0,
        Sigma1=sigma_large_1,
        note="High-dimensional large-covariance regime with matched trace; no mean difference.",
    )
    small_case = base.Regime(
        name=f"smallcov{p}",
        title=f"Small Covariance Case: d={p}, Mean Gap Plus Small Covariance Gap",
        p=p,
        n_train_per_class=660,
        n_test_per_class=2500,
        n_ref_per_class=3000,
        mean0=-0.55 * v,
        mean1=0.55 * v,
        Sigma0=sigma_small_0,
        Sigma1=sigma_small_1,
        note="High-dimensional case where the covariance gap is real but weaker than its estimation error.",
    )
    return large_case, small_case


def run_bestkernel_case(regime: base.Regime, repeats: int = 1, seed: int = 123):
    raw_rows = []
    selection_rows = []
    cov_rows = []
    plot_payload = {}

    rng_ref = np.random.default_rng(seed + 9999)
    X0_ref = base.sample_gaussian_antithetic(regime.n_ref_per_class, regime.mean0, regime.Sigma0, rng_ref)
    X1_ref = base.sample_gaussian_antithetic(regime.n_ref_per_class, regime.mean1, regime.Sigma1, rng_ref)

    for rep in range(repeats):
        rng = np.random.default_rng(seed + rep)
        X0_train = base.sample_gaussian_antithetic(regime.n_train_per_class, regime.mean0, regime.Sigma0, rng)
        X1_train = base.sample_gaussian_antithetic(regime.n_train_per_class, regime.mean1, regime.Sigma1, rng)
        X0_test = base.sample_gaussian_antithetic(regime.n_test_per_class, regime.mean0, regime.Sigma0, rng)
        X1_test = base.sample_gaussian_antithetic(regime.n_test_per_class, regime.mean1, regime.Sigma1, rng)

        X_train = np.vstack([X0_train, X1_train])
        y_train = np.hstack([np.zeros(len(X0_train), int), np.ones(len(X1_train), int)])
        X_test = np.vstack([X0_test, X1_test])
        y_test = np.hstack([np.zeros(len(X0_test), int), np.ones(len(X1_test), int)])

        splitter = base.StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=seed + 500 + rep)
        fit_idx, val_idx = next(splitter.split(X_train, y_train))
        X_fit, y_fit = X_train[fit_idx], y_train[fit_idx]
        X_val, y_val = X_train[val_idx], y_train[val_idx]

        original_diag = base.original_space_diagnostics(X0_train, X1_train, regime)
        for metric, raw_params in base.STAGE1_KERNELS:
            params_ref = base.normalize_kernel_params(metric, raw_params, X_train)
            feat_diag = base.feature_space_diagnostics(X0_train, X1_train, X0_ref, X1_ref, metric, params_ref)
            cov_rows.append(
                {
                    "regime": regime.name,
                    "repeat": rep + 1,
                    "kernel": metric,
                    "kernel_label": base.kernel_label(metric, params_ref),
                    **original_diag,
                    **feat_diag,
                }
            )

        for variant in base.VARIANTS:
            dims = (1,) if variant == "KFDA-1D" else R_VALUES
            for r in dims:
                candidates = []
                for metric, raw_params in base.STAGE1_KERNELS:
                    cand = base.evaluate_variant_on_validation(
                        variant=variant,
                        proj_dim=r,
                        X_fit=X_fit,
                        y_fit=y_fit,
                        X_val=X_val,
                        y_val=y_val,
                        anchor_per_class=40,
                        stage1_seed=seed + 1000 * rep + 17 * r,
                        metric=metric,
                        raw_params=raw_params,
                        reg=1e-6,
                    )
                    cand["regime"] = regime.name
                    cand["repeat"] = rep + 1
                    selection_rows.append(cand)
                    candidates.append(cand)

                best_cand = max(candidates, key=lambda row: row["val_acc"])
                full = base.refit_variant_on_full_train(
                    variant=variant,
                    proj_dim=r,
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    anchor_per_class=40,
                    stage1_seed=seed + 2000 * rep + 17 * r,
                    metric=best_cand["stage1_kernel"],
                    raw_or_norm_params=best_cand["stage1_params"],
                    reg=1e-6,
                )
                raw_rows.append(
                    {
                        "regime": regime.name,
                        "repeat": rep + 1,
                        "variant": variant,
                        "r": int(r),
                        "stage1_kernel": full["stage1_kernel"],
                        "stage1_label": full["stage1_label"],
                        "test_acc": full["test_acc"],
                        "val_acc": best_cand["val_acc"],
                        "stage2_kernel": full["best_svm_kernel"],
                        "stage2_params": "{}",
                    }
                )
                if rep == 0:
                    plot_payload[(variant, int(r))] = {
                        "Z_show": full["Z_show"],
                        "y_show": full["y_show"],
                        "stage1_label": full["stage1_label"],
                    }

    df_raw = pd.DataFrame(raw_rows)
    df_sel = pd.DataFrame(selection_rows)
    df_cov = pd.DataFrame(cov_rows)
    df_acc = base.summarize_accuracy(df_raw)
    df_cov_summary = base.summarize_covariance(df_cov)
    df_cov_summary["signal_to_noise_ratio"] = (
        df_cov_summary["ref_D_sigma_feature"] / df_cov_summary["cov_gap_error_feature"]
    )
    return df_raw, df_sel, df_acc, df_cov, df_cov_summary, plot_payload


def run_fixed_kernel_case(
    regime: base.Regime,
    metric: str,
    raw_params: dict,
    repeats: int = 1,
    seed: int = 123,
):
    raw_rows = []
    plot_payload = {}

    for rep in range(repeats):
        rng = np.random.default_rng(seed + rep)
        X0_train = base.sample_gaussian_antithetic(regime.n_train_per_class, regime.mean0, regime.Sigma0, rng)
        X1_train = base.sample_gaussian_antithetic(regime.n_train_per_class, regime.mean1, regime.Sigma1, rng)
        X0_test = base.sample_gaussian_antithetic(regime.n_test_per_class, regime.mean0, regime.Sigma0, rng)
        X1_test = base.sample_gaussian_antithetic(regime.n_test_per_class, regime.mean1, regime.Sigma1, rng)

        X_train = np.vstack([X0_train, X1_train])
        y_train = np.hstack([np.zeros(len(X0_train), int), np.ones(len(X1_train), int)])
        X_test = np.vstack([X0_test, X1_test])
        y_test = np.hstack([np.zeros(len(X0_test), int), np.ones(len(X1_test), int)])

        for variant in base.VARIANTS:
            dims = (1,) if variant == "KFDA-1D" else R_VALUES
            for r in dims:
                full = base.refit_variant_on_full_train(
                    variant=variant,
                    proj_dim=r,
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    anchor_per_class=40,
                    stage1_seed=seed + 2000 * rep + 17 * r,
                    metric=metric,
                    raw_or_norm_params=raw_params,
                    reg=1e-6,
                )
                raw_rows.append(
                    {
                        "regime": regime.name,
                        "repeat": rep + 1,
                        "variant": variant,
                        "r": int(r),
                        "stage1_kernel": full["stage1_kernel"],
                        "stage1_label": full["stage1_label"],
                        "test_acc": full["test_acc"],
                    }
                )
                if rep == 0:
                    plot_payload[(variant, int(r))] = {
                        "Z_show": full["Z_show"],
                        "y_show": full["y_show"],
                        "stage1_label": full["stage1_label"],
                    }

    df_raw = pd.DataFrame(raw_rows)
    df_acc = base.summarize_accuracy(df_raw)
    return df_raw, df_acc, plot_payload


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


def build_case_signal_summary(df_signal_noise: pd.DataFrame) -> pd.DataFrame:
    rows = []
    case_map = {
        "large_covariance_case": ("A", "Large covariance case"),
        "small_covariance_case": ("B", "Small covariance case"),
    }
    for case, (label, title) in case_map.items():
        sub = df_signal_noise[df_signal_noise["case"] == case].copy()
        best = sub.sort_values("signal_to_noise_ratio", ascending=False).iloc[0]
        ratio = float(best["signal_to_noise_ratio"])
        rows.append(
            {
                "label": label,
                "case": case,
                "title": title,
                "representative_kernel": best["kernel"],
                "true_feature_covariance_difference": float(best["ref_D_sigma_feature"]),
                "feature_covariance_error": float(best["cov_gap_error_feature"]),
                "signal_to_noise_ratio": ratio,
                "relation": "signal > error" if ratio > 1.0 else "signal < error",
            }
        )
    return pd.DataFrame(rows)


def plot_accuracy_curves(large_acc: pd.DataFrame, small_acc: pd.DataFrame, large_label: str, small_label: str):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
    colors = {"KFDA-1D": "#111827", "MY-large_mu": "#b91c1c", "MY-small_mu": "#1d4ed8"}

    ax = axes[0]
    sub = large_acc.copy()
    kfda = sub[sub["variant"] == "KFDA-1D"].iloc[0]
    ax.axhline(float(kfda["mean"]), color=colors["KFDA-1D"], linestyle="--", linewidth=2.2, label="KFDA-1D")
    for variant in ("MY-large_mu", "MY-small_mu"):
        cur = sub[sub["variant"] == variant].sort_values("r")
        ax.plot(cur["r"], cur["mean"], marker="o", linewidth=2.2, color=colors[variant], label=base.variant_display(variant))
        ax.fill_between(cur["r"], cur["mean"] - cur["std"], cur["mean"] + cur["std"], alpha=0.18, color=colors[variant])
    ax.set_title(f"A: Large covariance case\n{large_label}")
    ax.set_xlabel("projection dimension r")
    ax.set_ylabel("test accuracy")
    ax.set_xticks(list(R_VALUES))
    ax.legend(frameon=True, fontsize=9)

    ax = axes[1]
    sub = small_acc.copy()
    kfda = sub[sub["variant"] == "KFDA-1D"].iloc[0]
    ax.axhline(float(kfda["mean"]), color=colors["KFDA-1D"], linestyle="--", linewidth=2.2, label="KFDA-1D")
    for variant in ("MY-large_mu", "MY-small_mu"):
        cur = sub[sub["variant"] == variant].sort_values("r")
        ax.plot(cur["r"], cur["mean"], marker="o", linewidth=2.2, color=colors[variant], label=base.variant_display(variant))
        ax.fill_between(cur["r"], cur["mean"] - cur["std"], cur["mean"] + cur["std"], alpha=0.18, color=colors[variant])
    ax.set_title(f"B: Small covariance case\n{small_label}")
    ax.set_xlabel("projection dimension r")
    ax.set_xticks(list(R_VALUES))
    ax.legend(frameon=True, fontsize=9)

    fig.tight_layout()
    out = OUT_DIR / "accuracy_curves.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_signal_vs_noise(large_cov: pd.DataFrame, small_cov: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=False)

    def one_panel(ax, df, title):
        sub = df.copy().sort_values("signal_to_noise_ratio", ascending=False)
        highlight_kernel = sub.iloc[0]["kernel"]
        x = np.arange(len(sub))
        ax.bar(x - 0.18, sub["ref_D_sigma_feature"], width=0.36, label=r"$D_\sigma^\phi$", color="#2563eb")
        ax.bar(x + 0.18, sub["cov_gap_error_feature"], width=0.36, label="covariance error", color="#dc2626")
        ax.set_xticks(x)
        ax.set_xticklabels(sub["kernel"], rotation=30, ha="right")
        ax.set_title(title)
        ax.set_ylabel("feature-space magnitude")
        ax.legend(frameon=True, fontsize=10)
        for i, (_, row) in enumerate(sub.iterrows()):
            ax.text(
                i,
                max(row["ref_D_sigma_feature"], row["cov_gap_error_feature"]) * 1.02,
                f"{row['signal_to_noise_ratio']:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        for tick, kern in zip(ax.get_xticklabels(), sub["kernel"]):
            if kern == highlight_kernel:
                tick.set_color("#7c3aed")
                tick.set_fontweight("bold")

    one_panel(axes[0], large_cov, "A: Large covariance case\ntrue covariance difference > error")
    one_panel(axes[1], small_cov, "B: Small covariance case\ntrue covariance difference < error")

    fig.tight_layout()
    out = OUT_DIR / "feature_space_signal_vs_noise.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_case_signal_summary(df_case_summary: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(df_case_summary))
    width = 0.34

    ax.bar(
        x - width / 2,
        df_case_summary["true_feature_covariance_difference"],
        width=width,
        color="#2563eb",
        label="true feature covariance difference",
    )
    ax.bar(
        x + width / 2,
        df_case_summary["feature_covariance_error"],
        width=width,
        color="#dc2626",
        label="feature covariance error",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [
            f"{row['label']}: {row['title']}\n{row['representative_kernel']}"
            for _, row in df_case_summary.iterrows()
        ]
    )
    ax.set_ylabel("feature-space magnitude")
    ax.set_title("Case-Level Feature Covariance Signal vs Error")
    ax.legend(frameon=True, fontsize=10)

    for i, (_, row) in enumerate(df_case_summary.iterrows()):
        ymax = max(row["true_feature_covariance_difference"], row["feature_covariance_error"])
        ax.text(
            i,
            ymax * 1.03,
            f"ratio={row['signal_to_noise_ratio']:.2f}\n{row['relation']}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    fig.tight_layout()
    out = OUT_DIR / "signal_noise_case_summary.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_projection_views(
    large_payload: dict,
    large_acc: pd.DataFrame,
    small_payload: dict,
    small_acc: pd.DataFrame,
):
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    best_large = large_acc[large_acc["variant"] == "MY-large_mu"].sort_values("mean", ascending=False).iloc[0]
    best_small = large_acc[large_acc["variant"] == "MY-small_mu"].sort_values("mean", ascending=False).iloc[0]
    panels = [("KFDA-1D", 1), ("MY-large_mu", int(best_large["r"])), ("MY-small_mu", int(best_small["r"]))]

    for ax, (variant, r) in zip(axes[0], panels):
        payload = large_payload[(variant, r)]
        acc_val = float(large_acc[(large_acc["variant"] == variant) & (large_acc["r"] == r)]["mean"].iloc[0])
        Z2 = base.project_to_plot_2d(payload["Z_show"])
        y = payload["y_show"]
        for cls, color, label in [(0, "#1f77b4", "class 0"), (1, "#d62728", "class 1")]:
            idx = y == cls
            ax.scatter(Z2[idx, 0], Z2[idx, 1], s=10, alpha=0.55, c=color, label=label)
        ax.set_title(
            f"Large covariance case\n{base.variant_display(variant)}, r={r}\nacc={acc_val:.3f}\n{payload['stage1_label']}"
        )
        ax.set_xlabel("2D view 1")
        ax.set_ylabel("2D view 2")
    axes[0, 0].legend(frameon=True, fontsize=9)

    best_large = small_acc[small_acc["variant"] == "MY-large_mu"].sort_values("mean", ascending=False).iloc[0]
    best_small = small_acc[small_acc["variant"] == "MY-small_mu"].sort_values("mean", ascending=False).iloc[0]
    panels = [("KFDA-1D", 1), ("MY-large_mu", int(best_large["r"])), ("MY-small_mu", int(best_small["r"]))]

    for ax, (variant, r) in zip(axes[1], panels):
        payload = small_payload[(variant, r)]
        acc_val = float(small_acc[(small_acc["variant"] == variant) & (small_acc["r"] == r)]["mean"].iloc[0])
        Z2 = base.project_to_plot_2d(payload["Z_show"])
        y = payload["y_show"]
        for cls, color, label in [(0, "#1f77b4", "class 0"), (1, "#d62728", "class 1")]:
            idx = y == cls
            ax.scatter(Z2[idx, 0], Z2[idx, 1], s=10, alpha=0.55, c=color, label=label)
        ax.set_title(
            f"Small covariance case\n{base.variant_display(variant)}, r={r}\nacc={acc_val:.3f}\n{payload['stage1_label']}"
        )
        ax.set_xlabel("2D view 1")
        ax.set_ylabel("2D view 2")

    fig.tight_layout()
    out = OUT_DIR / "projection_views.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def parse_args():
    parser = argparse.ArgumentParser(description="Feature-space signal vs estimation-noise demo.")
    parser.add_argument("--dimension", type=int, default=220, help="Input-space dimension for both synthetic regimes.")
    parser.add_argument("--repeats", type=int, default=1, help="Number of repeated outer runs.")
    parser.add_argument("--seed", type=int, default=123, help="Random seed for repeated evaluations.")
    parser.add_argument(
        "--output-subdir",
        type=str,
        default="outputs",
        help="Workspace-relative output folder to write CSVs and plots into.",
    )
    return parser.parse_args()


def kernel_raw_params_for_name(metric: str) -> dict:
    for name, raw_params in base.STAGE1_KERNELS:
        if name == metric:
            return raw_params
    raise ValueError(f"Unknown stage-1 kernel: {metric}")


def main():
    args = parse_args()
    global OUT_DIR
    OUT_DIR = ROOT / args.output_subdir
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    large_regime, small_regime = build_regimes(dimension=args.dimension)
    df_large_raw, df_large_sel, df_large_acc, df_large_cov, df_large_cov_summary, large_payload = run_bestkernel_case(
        large_regime,
        repeats=args.repeats,
        seed=args.seed,
    )
    df_small_raw, df_small_sel, df_small_acc, df_small_cov, df_small_cov_summary, small_payload = run_bestkernel_case(
        small_regime,
        repeats=args.repeats,
        seed=args.seed,
    )

    large_cov_export = collapse_by_kernel(df_large_cov_summary)
    small_cov_export = collapse_by_kernel(df_small_cov_summary)
    large_cov_export["case"] = "large_covariance_case"
    small_cov_export["case"] = "small_covariance_case"
    df_signal_noise = pd.concat([large_cov_export, small_cov_export], ignore_index=True)
    df_case_summary = build_case_signal_summary(df_signal_noise)

    df_large_raw.to_csv(OUT_DIR / "strong_bestkernel_test_results.csv", index=False)
    df_large_sel.to_csv(OUT_DIR / "strong_bestkernel_selection_log.csv", index=False)
    df_large_acc.to_csv(OUT_DIR / "strong_bestkernel_accuracy_summary.csv", index=False)
    df_large_cov.to_csv(OUT_DIR / "strong_bestkernel_covariance_diagnostics_per_repeat.csv", index=False)
    df_small_raw.to_csv(OUT_DIR / "weak_bestkernel_test_results.csv", index=False)
    df_small_sel.to_csv(OUT_DIR / "weak_bestkernel_selection_log.csv", index=False)
    df_small_acc.to_csv(OUT_DIR / "weak_bestkernel_accuracy_summary.csv", index=False)
    df_small_cov.to_csv(OUT_DIR / "weak_bestkernel_covariance_diagnostics_per_repeat.csv", index=False)
    df_signal_noise.to_csv(OUT_DIR / "feature_space_signal_noise_summary.csv", index=False)
    df_case_summary.to_csv(OUT_DIR / "signal_noise_case_summary.csv", index=False)

    pd.DataFrame(
        [
            {
                "case": large_regime.name,
                "title": large_regime.title,
                "dimension": large_regime.p,
                "mean_gap_input": float(np.linalg.norm(large_regime.mean1 - large_regime.mean0)),
                "cov_gap_input": float(np.linalg.norm(large_regime.Sigma1 - large_regime.Sigma0, ord="fro")),
                "note": large_regime.note,
            },
            {
                "case": small_regime.name,
                "title": small_regime.title,
                "dimension": small_regime.p,
                "mean_gap_input": float(np.linalg.norm(small_regime.mean1 - small_regime.mean0)),
                "cov_gap_input": float(np.linalg.norm(small_regime.Sigma1 - small_regime.Sigma0, ord="fro")),
                "note": small_regime.note,
            },
        ]
    ).to_csv(OUT_DIR / "case_setup.csv", index=False)

    large_kernel = str(df_case_summary[df_case_summary["label"] == "A"]["representative_kernel"].iloc[0])
    small_kernel = str(df_case_summary[df_case_summary["label"] == "B"]["representative_kernel"].iloc[0])

    _, df_large_fixed_acc, large_fixed_payload = run_fixed_kernel_case(
        large_regime,
        metric=large_kernel,
        raw_params=kernel_raw_params_for_name(large_kernel),
        repeats=args.repeats,
        seed=args.seed,
    )
    _, df_small_fixed_acc, small_fixed_payload = run_fixed_kernel_case(
        small_regime,
        metric=small_kernel,
        raw_params=kernel_raw_params_for_name(small_kernel),
        repeats=args.repeats,
        seed=args.seed,
    )
    df_large_fixed_acc.to_csv(OUT_DIR / "large_case_representative_kernel_accuracy_summary.csv", index=False)
    df_small_fixed_acc.to_csv(OUT_DIR / "small_case_representative_kernel_accuracy_summary.csv", index=False)

    acc_plot = plot_accuracy_curves(
        df_large_fixed_acc,
        df_small_fixed_acc,
        f"{large_kernel} stage-1 kernel",
        f"{small_kernel} stage-1 kernel",
    )
    sig_plot = plot_signal_vs_noise(large_cov_export, small_cov_export)
    case_sig_plot = plot_case_signal_summary(df_case_summary)
    proj_plot = plot_projection_views(large_fixed_payload, df_large_fixed_acc, small_fixed_payload, df_small_fixed_acc)

    print("Saved outputs:")
    print(f"- {OUT_DIR / 'strong_bestkernel_accuracy_summary.csv'}")
    print(f"- {OUT_DIR / 'weak_bestkernel_accuracy_summary.csv'}")
    print(f"- {OUT_DIR / 'feature_space_signal_noise_summary.csv'}")
    print(f"- {OUT_DIR / 'signal_noise_case_summary.csv'}")
    print(f"- {OUT_DIR / 'case_setup.csv'}")
    print(f"- dimension={args.dimension}, repeats={args.repeats}, seed={args.seed}")
    print(f"- {acc_plot}")
    print(f"- {sig_plot}")
    print(f"- {case_sig_plot}")
    print(f"- {proj_plot}")

    with pd.option_context("display.max_columns", None, "display.width", 240):
        print("\nLarge-covariance accuracy summary:")
        print(df_large_acc)
        print("\nSmall-covariance accuracy summary:")
        print(df_small_acc)
        print("\nRepresentative-kernel accuracy summary:")
        print(df_large_fixed_acc)
        print(df_small_fixed_acc)
        print("\nFeature-space signal vs noise summary:")
        print(df_signal_noise)
        print("\nCase-level signal vs error summary:")
        print(df_case_summary)

    print("\nInterpretation:")
    print(build_terminal_summary(df_large_acc, df_small_acc, df_case_summary))

    base.open_pngs(OUT_DIR)


if __name__ == "__main__":
    main()
