from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / "supcon_comparison_workspace"))
sys.path.insert(0, str(ROOT / "redo_projection_workspace"))

from compare_supcon_vs_ours import load_notebook_namespace, train_supcon_feature_model, NOTEBOOK_PATH  # type: ignore
import redo_projection_cv_experiment as base  # type: ignore


sns.set_theme(style="whitegrid", context="talk")


def run_gaussian_experiment():
    ns = load_notebook_namespace(NOTEBOOK_PATH)
    ns["set_global_seed"](0)
    device = ns["pick_device"]()

    stage1_kernels = ns["STAGE1_KERNELS_FULL"]
    r_values = (1, 2, 3, 4, 6)
    regimes = base.build_regimes()

    rows = []
    diag_rows = []

    for regime in regimes:
        rng_ref = np.random.default_rng(10_000 + hash(regime.name) % 1000)
        X0_ref = base.sample_gaussian_antithetic(regime.n_ref_per_class, regime.mean0, regime.Sigma0, rng_ref)
        X1_ref = base.sample_gaussian_antithetic(regime.n_ref_per_class, regime.mean1, regime.Sigma1, rng_ref)

        ours_rows = []
        sup_rows = []
        for rep in range(3):
            rng = np.random.default_rng(2000 + 37 * rep + hash(regime.name) % 1000)
            X0_train = base.sample_gaussian_antithetic(regime.n_train_per_class, regime.mean0, regime.Sigma0, rng)
            X1_train = base.sample_gaussian_antithetic(regime.n_train_per_class, regime.mean1, regime.Sigma1, rng)
            X0_test = base.sample_gaussian_antithetic(regime.n_test_per_class, regime.mean0, regime.Sigma0, rng)
            X1_test = base.sample_gaussian_antithetic(regime.n_test_per_class, regime.mean1, regime.Sigma1, rng)

            Xtr = np.vstack([X0_train, X1_train])
            ytr = np.hstack([np.zeros(len(X0_train), dtype=int), np.ones(len(X1_train), dtype=int)])
            Xte = np.vstack([X0_test, X1_test])
            yte = np.hstack([np.zeros(len(X0_test), dtype=int), np.ones(len(X1_test), dtype=int)])

            ours_rows.extend(
                ns["evaluate_methods_on_fixed_split_fast"](
                    Xtr,
                    ytr,
                    Xte,
                    yte,
                    task_name=regime.name,
                    rep=rep,
                    r_values=r_values,
                    anchors_per_class=40,
                    mu_reg=1e-6,
                    svm_folds=2,
                    stage1_kernels=stage1_kernels,
                    basis_mode="anchors",
                    center_kernel=True,
                    seed=3000 + rep,
                    verbose=False,
                )
            )

            sup = train_supcon_feature_model(
                Xtr,
                ytr,
                Xte,
                yte,
                device=device,
                seed=4000 + rep,
                epochs=18,
            )
            sup_rows.append({"regime": regime.name, "rep": rep, "acc": float(sup["acc"])})

            for metric, raw_params in stage1_kernels:
                params = base.normalize_kernel_params(metric, raw_params, X0_ref)
                feat_diag = base.feature_space_diagnostics(X0_train, X1_train, X0_ref, X1_ref, metric, params)
                input_diag = base.original_space_diagnostics(X0_train, X1_train, regime)
                diag_rows.append(
                    {
                        "regime": regime.name,
                        "rep": rep,
                        "kernel": metric,
                        **input_diag,
                        **feat_diag,
                    }
                )

        df_ours = pd.DataFrame(ours_rows)
        summary = ns["summarize_multidim_results"](df_ours)
        best = summary["best_kernel"]
        best_large = best[best["family"] == "MY-large_mu"].sort_values("acc_mean", ascending=False).iloc[0].to_dict()
        best_small = best[best["family"] == "MY-small_mu"].sort_values("acc_mean", ascending=False).iloc[0].to_dict()
        best_kfda = best[best["family"] == "KFDA"].sort_values("acc_mean", ascending=False).iloc[0].to_dict()
        sup_acc = float(pd.DataFrame(sup_rows)["acc"].mean())
        sup_std = float(pd.DataFrame(sup_rows)["acc"].std(ddof=1)) if len(sup_rows) > 1 else 0.0

        rows.extend(
            [
                {
                    "regime": regime.name,
                    "title": regime.title,
                    "method": "KFDA-1D",
                    "acc_mean": best_kfda["acc_mean"],
                    "acc_std": best_kfda["acc_std"],
                    "stage1_kernel": best_kfda["stage1_kernel"],
                    "r": 1,
                },
                {
                    "regime": regime.name,
                    "title": regime.title,
                    "method": "MY-large_mu",
                    "acc_mean": best_large["acc_mean"],
                    "acc_std": best_large["acc_std"],
                    "stage1_kernel": best_large["stage1_kernel"],
                    "r": int(best_large["r"]),
                },
                {
                    "regime": regime.name,
                    "title": regime.title,
                    "method": "MY-small_mu",
                    "acc_mean": best_small["acc_mean"],
                    "acc_std": best_small["acc_std"],
                    "stage1_kernel": best_small["stage1_kernel"],
                    "r": int(best_small["r"]),
                },
                {
                    "regime": regime.name,
                    "title": regime.title,
                    "method": "SupCon-feature",
                    "acc_mean": sup_acc,
                    "acc_std": sup_std,
                    "stage1_kernel": "n/a",
                    "r": -1,
                },
            ]
        )

    df_summary = pd.DataFrame(rows)
    df_diag = pd.DataFrame(diag_rows)
    df_summary.to_csv(OUT / "gaussian_summary.csv", index=False)
    df_diag.to_csv(OUT / "gaussian_diagnostics.csv", index=False)

    fig, axes = plt.subplots(1, len(regimes), figsize=(6 * len(regimes), 5), sharey=True)
    if len(regimes) == 1:
        axes = [axes]
    for ax, regime in zip(axes, regimes):
        sub = df_summary[df_summary["regime"] == regime.name].copy()
        ax.bar(sub["method"], sub["acc_mean"], yerr=sub["acc_std"], color=["#577590", "#43aa8b", "#f8961e", "#f94144"])
        ax.set_title(regime.title)
        ax.set_ylim(0.45, 1.0)
        ax.tick_params(axis="x", rotation=18)
    axes[0].set_ylabel("Accuracy")
    plt.tight_layout()
    fig.savefig(OUT / "gaussian_accuracy_comparison.png", dpi=180)
    plt.close(fig)

    signal = (
        df_diag.groupby(["regime", "kernel"], as_index=False)[["ref_D_sigma_feature", "cov_gap_error_feature"]]
        .mean()
    )
    fig, axes = plt.subplots(1, len(regimes), figsize=(6 * len(regimes), 5), sharey=True)
    if len(regimes) == 1:
        axes = [axes]
    for ax, regime in zip(axes, regimes):
        sub = signal[signal["regime"] == regime.name].copy()
        x = np.arange(len(sub))
        ax.bar(x - 0.18, sub["ref_D_sigma_feature"], width=0.36, label="feature gap", color="#b91c1c")
        ax.bar(x + 0.18, sub["cov_gap_error_feature"], width=0.36, label="feature error", color="#9ca3af")
        ax.set_xticks(x)
        ax.set_xticklabels(sub["kernel"], rotation=20)
        ax.set_title(regime.title)
    axes[0].set_ylabel("Magnitude")
    axes[-1].legend()
    plt.tight_layout()
    fig.savefig(OUT / "gaussian_signal_vs_error.png", dpi=180)
    plt.close(fig)

    print(df_summary.to_string(index=False))
    print(f"\nSaved outputs to {OUT}")


if __name__ == "__main__":
    run_gaussian_experiment()
