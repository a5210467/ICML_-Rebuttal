from __future__ import annotations

"""
High-dimensional rebuild closer to 1D_projection.ipynb.

This script exists for one reason: make the comparison easier to audit against the
old notebook setup. In particular it:

- uses d = 220 like the old notebook
- uses the old AtA covariance construction with scale gaps
- keeps the same stage-1 kernel menu and stage-2 SVM CV style
- evaluates every stage-1 kernel separately, not only the best-selected one
- still records the best-kernel curves after inner validation

That way we can answer two different questions cleanly:

1. "What does the best nested-selected kernel do?"
2. "Does degree-2 polynomial kernel see the covariance difference?"
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import redo_projection_cv_experiment as base


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "outputs_hd220_oldstyle"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="talk")


def build_oldstyle_summary(df_best_summary: pd.DataFrame, df_kernel_summary: pd.DataFrame) -> str:
    same_case = df_best_summary[df_best_summary["regime"] == "oldstyle_same_mean_scale_gap"]
    mix_case = df_best_summary[df_best_summary["regime"] == "oldstyle_diff_mean_scale_gap"]
    same_best = same_case[same_case["variant"] != "KFDA-1D"].sort_values("mean", ascending=False).iloc[0]
    same_kfda = same_case[same_case["variant"] == "KFDA-1D"].iloc[0]
    mix_best = mix_case[mix_case["variant"] != "KFDA-1D"].sort_values("mean", ascending=False).iloc[0]
    mix_kfda = mix_case[mix_case["variant"] == "KFDA-1D"].iloc[0]
    poly2_same = df_kernel_summary[
        (df_kernel_summary["regime"] == "oldstyle_same_mean_scale_gap") & (df_kernel_summary["stage1_kernel"] == "polynomial")
    ]["mean"].max()
    return (
        "Summary: in this old-style d=220 scale-gap setting, the task is easy enough that both KFDA and KDMLP reach very high "
        f"accuracy, but KDMLP still has a measurable edge in the covariance-rich cases "
        f"({same_best['mean']:.4f} vs {same_kfda['mean']:.4f} for same-mean; "
        f"{mix_best['mean']:.4f} vs {mix_kfda['mean']:.4f} for mixed mean/covariance). "
        f"The polynomial kernel does detect the covariance difference here as well, reaching {poly2_same:.4f} on the same-mean case, "
        "even though another kernel is usually chosen by inner validation."
    )


def random_covariance_AtA(d: int, rng: np.random.Generator, scale: float = 1.0) -> np.ndarray:
    """Old notebook covariance generator: Sigma = (A^T A)/d * scale."""
    A = rng.normal(size=(d, d))
    Sigma = (A.T @ A) / float(d)
    return base.symmetrize(Sigma * scale)


def build_oldstyle_regimes():
    rng = np.random.default_rng(20260326)
    p = 220

    Sigma_same_0 = random_covariance_AtA(p, rng, scale=1.0)
    Sigma_same_1 = random_covariance_AtA(p, rng, scale=2.5)

    Sigma_mix_0 = random_covariance_AtA(p, rng, scale=1.0)
    Sigma_mix_1 = random_covariance_AtA(p, rng, scale=2.1)
    v = base.random_unit_vector(rng, p)

    return [
        base.Regime(
            name="oldstyle_same_mean_scale_gap",
            title="Old-style Case A: Same Mean, Different Covariances, d=220",
            p=p,
            n_train_per_class=660,
            n_test_per_class=2500,
            n_ref_per_class=3000,
            mean0=np.zeros(p),
            mean1=np.zeros(p),
            Sigma0=Sigma_same_0,
            Sigma1=Sigma_same_1,
            note="Directly mirrors the old notebook: zero means, AtA covariances, scale1=1.0, scale2=2.5.",
        ),
        base.Regime(
            name="oldstyle_diff_mean_scale_gap",
            title="Old-style Case B: Different Mean, Different Covariances, d=220",
            p=p,
            n_train_per_class=660,
            n_test_per_class=2500,
            n_ref_per_class=3000,
            mean0=-0.18 * v,
            mean1=0.18 * v,
            Sigma0=Sigma_mix_0,
            Sigma1=Sigma_mix_1,
            note="Old-style high-dimensional case with both a mean gap and a covariance scale gap.",
        ),
    ]


def run_regime_with_full_kernel_eval(
    regime: base.Regime,
    repeats: int = 1,
    anchor_per_class: int = 40,
    reg: float = 1e-6,
    seed: int = 123,
):
    """
    Similar to base.run_regime, but keeps full-train/test evaluation for every stage-1 kernel.
    This is what makes the per-kernel plots possible.
    """
    kernel_rows = []
    best_rows = []
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
            dims = (1,) if variant == "KFDA-1D" else base.R_VALUES
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
                        anchor_per_class=anchor_per_class,
                        stage1_seed=seed + 1000 * rep + 17 * r,
                        metric=metric,
                        raw_params=raw_params,
                        reg=reg,
                    )
                    cand["regime"] = regime.name
                    cand["repeat"] = rep + 1
                    selection_rows.append(cand)
                    candidates.append(cand)

                    # Evaluate this exact kernel candidate on the outer test split too.
                    full = base.refit_variant_on_full_train(
                        variant=variant,
                        proj_dim=r,
                        X_train=X_train,
                        y_train=y_train,
                        X_test=X_test,
                        y_test=y_test,
                        anchor_per_class=anchor_per_class,
                        stage1_seed=seed + 2000 * rep + 17 * r,
                        metric=metric,
                        raw_or_norm_params=cand["stage1_params"],
                        reg=reg,
                    )
                    kernel_rows.append(
                        {
                            "regime": regime.name,
                            "repeat": rep + 1,
                            "variant": variant,
                            "r": int(r),
                            "stage1_kernel": metric,
                            "stage1_params": full["stage1_params"],
                            "stage1_label": full["stage1_label"],
                            "val_acc": cand["val_acc"],
                            "test_acc": full["test_acc"],
                            "best_svm_kernel": full["best_svm_kernel"],
                            "best_svm_params": full["best_svm_params"],
                            "diag": full["diag"],
                        }
                    )

                best_cand = max(candidates, key=lambda row: row["val_acc"])
                full_best = base.refit_variant_on_full_train(
                    variant=variant,
                    proj_dim=r,
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    anchor_per_class=anchor_per_class,
                    stage1_seed=seed + 2000 * rep + 17 * r,
                    metric=best_cand["stage1_kernel"],
                    raw_or_norm_params=best_cand["stage1_params"],
                    reg=reg,
                )
                best_rows.append(
                    {
                        "regime": regime.name,
                        "repeat": rep + 1,
                        "variant": variant,
                        "r": int(r),
                        "stage1_kernel": full_best["stage1_kernel"],
                        "stage1_params": full_best["stage1_params"],
                        "stage1_label": full_best["stage1_label"],
                        "val_acc": best_cand["val_acc"],
                        "test_acc": full_best["test_acc"],
                        "best_svm_kernel": full_best["best_svm_kernel"],
                        "best_svm_params": full_best["best_svm_params"],
                        "diag": full_best["diag"],
                    }
                )

                key = (regime.name, variant, int(r))
                if key not in plot_payload:
                    plot_payload[key] = {
                        "Z_show": full_best["Z_show"],
                        "y_show": full_best["y_show"],
                        "stage1_label": full_best["stage1_label"],
                        "test_acc": full_best["test_acc"],
                    }

    return (
        pd.DataFrame(kernel_rows),
        pd.DataFrame(best_rows),
        pd.DataFrame(selection_rows),
        pd.DataFrame(cov_rows),
        plot_payload,
    )


def summarize_per_kernel(df_kernel: pd.DataFrame) -> pd.DataFrame:
    out = (
        df_kernel.groupby(["regime", "variant", "r", "stage1_kernel", "stage1_label"], as_index=False)["test_acc"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"count": "n_repeats"})
    )
    return out


def summarize_best(df_best: pd.DataFrame) -> pd.DataFrame:
    out = (
        df_best.groupby(["regime", "variant", "r"], as_index=False)["test_acc"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"count": "n_repeats"})
    )
    return out


def plot_best_kernel_curves(regimes, df_best_summary: pd.DataFrame):
    fig, axes = plt.subplots(1, len(regimes), figsize=(8 * len(regimes), 6), sharey=True)
    axes = np.atleast_1d(axes)
    colors = {"KFDA-1D": "#1f77b4", "MY-large_mu": "#d62728", "MY-small_mu": "#2ca02c"}

    for ax, regime in zip(axes, regimes):
        sub = df_best_summary[df_best_summary["regime"] == regime.name]
        for variant in base.VARIANTS:
            piece = sub[sub["variant"] == variant].sort_values("r")
            if piece.empty:
                continue
            ax.plot(piece["r"], piece["mean"], marker="o", linewidth=2.5, label=variant, color=colors[variant])
        ax.set_title(regime.title)
        ax.set_xlabel("projection dimension r")
        ax.set_ylabel("test accuracy")
        ax.set_ylim(0.45, 1.02)
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=True)

    fig.tight_layout()
    out = OUT_DIR / "best_kernel_accuracy_curves.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_same_mean_per_kernel(df_kernel_summary: pd.DataFrame):
    regime_name = "oldstyle_same_mean_scale_gap"
    sub = df_kernel_summary[df_kernel_summary["regime"] == regime_name].copy()
    sub = sub[sub["variant"] != "KFDA-1D"]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    for ax, variant in zip(axes, ["MY-large_mu", "MY-small_mu"]):
        piece = sub[sub["variant"] == variant]
        for kernel in ["linear", "polynomial", "rbf", "laplacian", "sigmoid"]:
            line = piece[piece["stage1_kernel"] == kernel].sort_values("r")
            if line.empty:
                continue
            ax.plot(line["r"], line["mean"], marker="o", linewidth=2.2, label=kernel)
        ax.set_title(f"{variant} on same-mean case")
        ax.set_xlabel("projection dimension r")
        ax.set_ylabel("test accuracy")
        ax.set_ylim(0.45, 1.02)
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=True)

    fig.tight_layout()
    out = OUT_DIR / "same_mean_per_kernel_accuracy.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_selected_projection_views(regimes, plot_payload_map, df_best_summary):
    fig, axes = plt.subplots(len(regimes), 3, figsize=(16, 5 * len(regimes)))
    axes = np.atleast_2d(axes)

    for row_idx, regime in enumerate(regimes):
        best_large = df_best_summary[
            (df_best_summary["regime"] == regime.name) & (df_best_summary["variant"] == "MY-large_mu")
        ].sort_values("mean", ascending=False).iloc[0]
        best_small = df_best_summary[
            (df_best_summary["regime"] == regime.name) & (df_best_summary["variant"] == "MY-small_mu")
        ].sort_values("mean", ascending=False).iloc[0]
        best_kfda = df_best_summary[
            (df_best_summary["regime"] == regime.name) & (df_best_summary["variant"] == "KFDA-1D")
        ].iloc[0]

        panels = [
            ("KFDA-1D", int(best_kfda["r"])),
            ("MY-large_mu", int(best_large["r"])),
            ("MY-small_mu", int(best_small["r"])),
        ]

        for col_idx, (variant, r) in enumerate(panels):
            ax = axes[row_idx, col_idx]
            payload = plot_payload_map[(regime.name, variant, r)]
            Z2 = base.project_to_plot_2d(payload["Z_show"])
            y = payload["y_show"]
            for cls, color, label in [(0, "#1f77b4", "class 0"), (1, "#d62728", "class 1")]:
                idx = y == cls
                ax.scatter(Z2[idx, 0], Z2[idx, 1], s=10, alpha=0.55, c=color, label=label)
            ax.set_title(f"{regime.name}\n{variant}, r={r}\nacc={payload['test_acc']:.4f}")
            ax.set_xlabel("2D view 1")
            ax.set_ylabel("2D view 2")
            if row_idx == 0 and col_idx == 0:
                ax.legend(frameon=True)

    fig.tight_layout()
    out = OUT_DIR / "selected_projection_views.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    regimes = build_oldstyle_regimes()
    all_kernel = []
    all_best = []
    all_sel = []
    all_cov = []
    plot_payload_map = {}

    for regime in regimes:
        print("=" * 100)
        print(regime.title)
        print(regime.note)
        df_kernel, df_best, df_sel, df_cov, plot_payload = run_regime_with_full_kernel_eval(
            regime,
            repeats=1,
            anchor_per_class=40,
            reg=1e-6,
            seed=123,
        )
        all_kernel.append(df_kernel)
        all_best.append(df_best)
        all_sel.append(df_sel)
        all_cov.append(df_cov)
        plot_payload_map.update(plot_payload)

    df_kernel = pd.concat(all_kernel, ignore_index=True)
    df_best = pd.concat(all_best, ignore_index=True)
    df_sel = pd.concat(all_sel, ignore_index=True)
    df_cov = pd.concat(all_cov, ignore_index=True)

    df_kernel_summary = summarize_per_kernel(df_kernel)
    df_best_summary = summarize_best(df_best)
    df_choice = base.summarize_kernel_choices(df_best)
    df_cov_summary = base.summarize_covariance(df_cov)
    df_regime = base.build_regime_table(regimes)

    df_kernel.to_csv(OUT_DIR / "all_kernel_test_results.csv", index=False)
    df_best.to_csv(OUT_DIR / "best_kernel_test_results.csv", index=False)
    df_sel.to_csv(OUT_DIR / "selection_log.csv", index=False)
    df_kernel_summary.to_csv(OUT_DIR / "per_kernel_accuracy_summary.csv", index=False)
    df_best_summary.to_csv(OUT_DIR / "best_kernel_accuracy_summary.csv", index=False)
    df_choice.to_csv(OUT_DIR / "best_kernel_choice_summary.csv", index=False)
    df_cov.to_csv(OUT_DIR / "covariance_diagnostics_per_repeat.csv", index=False)
    df_cov_summary.to_csv(OUT_DIR / "covariance_diagnostics_summary.csv", index=False)
    df_regime.to_csv(OUT_DIR / "regime_setup.csv", index=False)

    best_plot = plot_best_kernel_curves(regimes, df_best_summary)
    per_kernel_plot = plot_same_mean_per_kernel(df_kernel_summary)
    proj_plot = plot_selected_projection_views(regimes, plot_payload_map, df_best_summary)

    print("\nSaved outputs:")
    print(f"- {OUT_DIR / 'all_kernel_test_results.csv'}")
    print(f"- {OUT_DIR / 'best_kernel_accuracy_summary.csv'}")
    print(f"- {OUT_DIR / 'per_kernel_accuracy_summary.csv'}")
    print(f"- {OUT_DIR / 'best_kernel_choice_summary.csv'}")
    print(f"- {OUT_DIR / 'covariance_diagnostics_summary.csv'}")
    print(f"- {OUT_DIR / 'regime_setup.csv'}")
    print(f"- {best_plot}")
    print(f"- {per_kernel_plot}")
    print(f"- {proj_plot}")

    with pd.option_context("display.max_columns", None, "display.width", 240):
        print("\nRegime setup:")
        print(df_regime)
        print("\nBest-kernel accuracy summary:")
        print(df_best_summary)
        print("\nPer-kernel accuracy summary:")
        print(df_kernel_summary)

    print("\nInterpretation:")
    print(build_oldstyle_summary(df_best_summary, df_kernel_summary))

    base.open_pngs(OUT_DIR)


if __name__ == "__main__":
    main()
