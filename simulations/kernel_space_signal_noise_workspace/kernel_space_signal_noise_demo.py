from __future__ import annotations

"""
Feature-space signal-vs-noise demo.

This workspace intentionally fixes the stage-1 kernel to linear and then studies
two high-dimensional regimes:

1. Strong feature-space covariance signal:
   same mean / different covariance, with the true covariance difference larger
   than its estimation error.

2. Weak feature-space covariance signal:
   mean gap / small covariance gap, with the true covariance difference smaller
   than its estimation error.

The point is to present one clean signal-vs-error comparison without mixing
multiple kernels inside the same explanation.
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
LINEAR_KERNEL = "linear"
LINEAR_PARAMS: dict = {}


def build_terminal_summary(df_strong_acc: pd.DataFrame, df_weak_acc: pd.DataFrame, df_signal_noise: pd.DataFrame) -> str:
    strong_best = df_strong_acc[df_strong_acc["variant"] != "KFDA-1D"].sort_values("mean", ascending=False).iloc[0]
    strong_kfda = df_strong_acc[df_strong_acc["variant"] == "KFDA-1D"].iloc[0]
    weak_best = df_weak_acc[df_weak_acc["variant"] != "KFDA-1D"].sort_values("mean", ascending=False).iloc[0]
    weak_kfda = df_weak_acc[df_weak_acc["variant"] == "KFDA-1D"].iloc[0]
    strong_ratio = float(df_signal_noise[df_signal_noise["case"] == "strong_feature_covariance_signal"]["signal_to_noise_ratio"].iloc[0])
    weak_ratio = float(df_signal_noise[df_signal_noise["case"] == "weak_feature_covariance_signal"]["signal_to_noise_ratio"].iloc[0])
    return (
        "Summary (linear stage-1 kernel only): in the strong case, the true feature covariance difference is larger than "
        f"its estimation error (ratio {strong_ratio:.3f}), and KDMLP improves over KFDA "
        f"({strong_best['mean']:.3f} vs {strong_kfda['mean']:.3f}). "
        "In the weak case, the true feature covariance difference is smaller than its estimation error "
        f"(ratio {weak_ratio:.3f}), so the covariance part is below the noise floor; "
        f"the best KDMLP run reaches {weak_best['mean']:.3f} versus {weak_kfda['mean']:.3f} for KFDA, "
        "which should be read as a joint mean/covariance effect rather than as a pure covariance win."
    )


def build_regimes(dimension: int = 220, seed: int = 20260327):
    rng = np.random.default_rng(seed)
    p = int(dimension)

    sigma_strong_0 = base.random_dense_spd(rng, p, eig_low=0.6, eig_high=2.2)
    sigma_weak_0 = base.random_dense_spd(rng, p, eig_low=0.6, eig_high=2.1)

    strong_T = base.dense_congruence_transform(rng, p, strength=0.95)
    weak_T = base.dense_congruence_transform(rng, p, strength=0.12)

    sigma_strong_1 = base.matched_trace_congruence(sigma_strong_0, strong_T)
    sigma_weak_1 = base.matched_trace_congruence(sigma_weak_0, weak_T)

    v = base.random_unit_vector(rng, p)

    strong = base.Regime(
        name=f"strong{p}",
        title=f"Strong Case: d={p}, Same Mean, Different Covariances",
        p=p,
        n_train_per_class=660,
        n_test_per_class=2500,
        n_ref_per_class=3000,
        mean0=np.zeros(p),
        mean1=np.zeros(p),
        Sigma0=sigma_strong_0,
        Sigma1=sigma_strong_1,
        note="High-dimensional covariance difference with matched trace; no mean difference.",
    )
    weak = base.Regime(
        name=f"weak{p}",
        title=f"Weak Case: d={p}, Mean Gap Plus Small Covariance Gap",
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


def run_linear_case(regime: base.Regime, repeats: int = 1, seed: int = 123):
    raw_rows = []
    selection_rows = []
    cov_rows = []
    plot_payload = {}

    X0_ref = base.sample_gaussian_antithetic(regime.n_ref_per_class, regime.mean0, regime.Sigma0, np.random.default_rng(seed + 901))
    X1_ref = base.sample_gaussian_antithetic(regime.n_ref_per_class, regime.mean1, regime.Sigma1, np.random.default_rng(seed + 902))

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
        feat_diag = base.feature_space_diagnostics(X0_train, X1_train, X0_ref, X1_ref, LINEAR_KERNEL, LINEAR_PARAMS)
        cov_rows.append(
            {
                "regime": regime.name,
                "repeat": rep + 1,
                "kernel": LINEAR_KERNEL,
                "kernel_label": LINEAR_KERNEL,
                **original_diag,
                **feat_diag,
            }
        )

        for variant in base.VARIANTS:
            dims = (1,) if variant == "KFDA-1D" else R_VALUES
            for r in dims:
                cand = base.evaluate_variant_on_validation(
                    variant=variant,
                    proj_dim=r,
                    X_fit=X_fit,
                    y_fit=y_fit,
                    X_val=X_val,
                    y_val=y_val,
                    anchor_per_class=40,
                    stage1_seed=seed + 1000 * rep + 17 * r,
                    metric=LINEAR_KERNEL,
                    raw_params=LINEAR_PARAMS,
                    reg=1e-6,
                )
                cand["regime"] = regime.name
                cand["repeat"] = rep + 1
                selection_rows.append(cand)

                full = base.refit_variant_on_full_train(
                    variant=variant,
                    proj_dim=r,
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    anchor_per_class=40,
                    stage1_seed=seed + 2000 * rep + 17 * r,
                    metric=LINEAR_KERNEL,
                    raw_or_norm_params=LINEAR_PARAMS,
                    reg=1e-6,
                )
                raw_rows.append(
                    {
                        "regime": regime.name,
                        "repeat": rep + 1,
                        "variant": variant,
                        "r": int(r),
                        "stage1_kernel": LINEAR_KERNEL,
                        "stage1_label": LINEAR_KERNEL,
                        "test_acc": full["test_acc"],
                        "val_acc": cand["val_acc"],
                        "stage2_kernel": full["best_svm_kernel"],
                        "stage2_params": "{}",
                    }
                )
                if rep == 0:
                    plot_payload[(variant, int(r))] = {
                        "Z_show": full["Z_show"],
                        "y_show": full["y_show"],
                        "stage1_label": LINEAR_KERNEL,
                    }

    df_raw = pd.DataFrame(raw_rows)
    df_sel = pd.DataFrame(selection_rows)
    df_cov = pd.DataFrame(cov_rows)
    df_acc = base.summarize_accuracy(df_raw)
    df_cov_summary = base.summarize_covariance(df_cov)
    df_cov_summary["signal_to_noise_ratio"] = df_cov_summary["ref_D_sigma_feature"] / df_cov_summary["cov_gap_error_feature"]
    return df_raw, df_sel, df_acc, df_cov, df_cov_summary, plot_payload


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
        "strong_feature_covariance_signal": ("A", "Strong feature covariance signal"),
        "weak_feature_covariance_signal": ("B", "Weak feature covariance signal"),
    }
    for case, (label, title) in case_map.items():
        sub = df_signal_noise[df_signal_noise["case"] == case].copy()
        row = sub[sub["kernel"] == LINEAR_KERNEL].iloc[0] if (sub["kernel"] == LINEAR_KERNEL).any() else sub.iloc[0]
        ratio = float(row["signal_to_noise_ratio"])
        rows.append(
            {
                "label": label,
                "case": case,
                "title": title,
                "representative_kernel": row["kernel"],
                "true_feature_covariance_difference": float(row["ref_D_sigma_feature"]),
                "feature_covariance_error": float(row["cov_gap_error_feature"]),
                "signal_to_noise_ratio": ratio,
                "relation": "signal > error" if ratio > 1.0 else "signal < error",
            }
        )
    return pd.DataFrame(rows)


def plot_accuracy_curves(strong_acc: pd.DataFrame, weak_acc: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    colors = {"KFDA-1D": "#111827", "MY-large_mu": "#b91c1c", "MY-small_mu": "#1d4ed8"}

    # Strong linear-kernel case.
    ax = axes[0]
    sub = strong_acc.copy()
    kfda = sub[sub["variant"] == "KFDA-1D"].iloc[0]
    ax.axhline(float(kfda["mean"]), color=colors["KFDA-1D"], linestyle="--", linewidth=2.2, label="KFDA-1D")
    for variant in ("MY-large_mu", "MY-small_mu"):
        cur = sub[sub["variant"] == variant].sort_values("r")
        ax.plot(cur["r"], cur["mean"], marker="o", linewidth=2.2, color=colors[variant], label=base.variant_display(variant))
        ax.fill_between(cur["r"], cur["mean"] - cur["std"], cur["mean"] + cur["std"], alpha=0.18, color=colors[variant])
    ax.set_title("Strong Case\nLinear stage-1 kernel")
    ax.set_xlabel("projection dimension r")
    ax.set_ylabel("test accuracy")
    ax.set_xticks(list(R_VALUES))
    ax.legend(frameon=True, fontsize=9)

    # Weak linear-kernel case.
    ax = axes[1]
    sub = weak_acc.copy()
    kfda = sub[sub["variant"] == "KFDA-1D"].iloc[0]
    ax.axhline(float(kfda["mean"]), color=colors["KFDA-1D"], linestyle="--", linewidth=2.2, label="KFDA-1D")
    for variant in ("MY-large_mu", "MY-small_mu"):
        cur = sub[sub["variant"] == variant].sort_values("r")
        ax.plot(cur["r"], cur["mean"], marker="o", linewidth=2.2, color=colors[variant], label=base.variant_display(variant))
        ax.fill_between(cur["r"], cur["mean"] - cur["std"], cur["mean"] + cur["std"], alpha=0.18, color=colors[variant])
    ax.set_title("Weak Case\nLinear stage-1 kernel")
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

    one_panel(axes[0], strong_cov, "Strong case (linear kernel)\ntrue covariance difference > error", highlight_kernel=LINEAR_KERNEL)
    one_panel(axes[1], weak_cov, "Weak case (linear kernel)\ntrue covariance difference < error", highlight_kernel=LINEAR_KERNEL)

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
    ax.set_title("Case-Level Feature Covariance Signal vs Error (Linear Kernel)")
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


def plot_projection_views(strong_payload: dict, strong_acc: pd.DataFrame, weak_payload: dict, weak_acc: pd.DataFrame):
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Strong case: linear stage-1 kernel
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
        ax.set_title(f"Strong case, linear kernel\n{base.variant_display(variant)}, r={r}\nacc={acc_val:.3f}")
        ax.set_xlabel("2D view 1")
        ax.set_ylabel("2D view 2")
    axes[0, 0].legend(frameon=True, fontsize=9)

    # Weak case: linear stage-1 kernel.
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
        ax.set_title(f"Weak case, linear kernel\n{base.variant_display(variant)}, r={r}\nacc={acc_val:.3f}")
        ax.set_xlabel("2D view 1")
        ax.set_ylabel("2D view 2")

    fig.tight_layout()
    out = OUT_DIR / "projection_views.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def parse_args():
    parser = argparse.ArgumentParser(description="Feature-space signal vs estimation-noise demo (linear stage-1 kernel).")
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


def main():
    args = parse_args()
    global OUT_DIR
    OUT_DIR = ROOT / args.output_subdir
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    strong_regime, weak_regime = build_regimes(dimension=args.dimension)
    df_strong_raw, df_strong_sel, df_strong_acc, df_strong_cov, df_strong_cov_summary, strong_payload = run_linear_case(
        strong_regime,
        repeats=args.repeats,
        seed=args.seed,
    )
    df_weak_raw, df_weak_sel, df_weak_acc, df_weak_cov, df_weak_cov_summary, weak_payload = run_linear_case(
        weak_regime,
        repeats=args.repeats,
        seed=args.seed,
    )

    # Keep only one row per kernel after averaging repeats.
    strong_cov_export = collapse_by_kernel(df_strong_cov_summary)
    weak_cov_export = collapse_by_kernel(df_weak_cov_summary)
    strong_cov_export["case"] = "strong_feature_covariance_signal"
    weak_cov_export["case"] = "weak_feature_covariance_signal"
    df_signal_noise = pd.concat([strong_cov_export, weak_cov_export], ignore_index=True)
    df_case_summary = build_case_signal_summary(df_signal_noise)

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
    df_case_summary.to_csv(OUT_DIR / "signal_noise_case_summary.csv", index=False)

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
    case_sig_plot = plot_case_signal_summary(df_case_summary)
    proj_plot = plot_projection_views(strong_payload, df_strong_acc, weak_payload, df_weak_acc)

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
        print("\nStrong-case accuracy summary:")
        print(df_strong_acc)
        print("\nWeak-case accuracy summary:")
        print(df_weak_acc)
        print("\nFeature-space signal vs noise summary:")
        print(df_signal_noise)
        print("\nCase-level signal vs error summary:")
        print(df_case_summary)

    print("\nInterpretation:")
    print(build_terminal_summary(df_strong_acc, df_weak_acc, df_signal_noise))

    base.open_pngs(OUT_DIR)


if __name__ == "__main__":
    main()
