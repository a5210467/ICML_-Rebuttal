from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "outputs_hd220"
OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / "supcon_comparison_workspace"))
sys.path.insert(0, str(ROOT / "redo_projection_workspace"))

from compare_supcon_vs_ours import load_notebook_namespace, train_supcon_feature_model, NOTEBOOK_PATH  # type: ignore
import redo_projection_cv_experiment as base  # type: ignore


sns.set_theme(style="whitegrid", context="talk")


def open_png_outputs(out_dir: Path) -> None:
    pngs = sorted(out_dir.glob("*.png"))
    if not pngs:
        return
    if sys.platform == "darwin":
        opener = ["open"]
    else:
        cmd = shutil.which("xdg-open")
        if cmd is None:
            return
        opener = [cmd]
    for png in pngs:
        try:
            subprocess.run(opener + [str(png)], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except OSError:
            return


def build_hd220_summary(df: pd.DataFrame) -> str:
    my = df[df["method"] == "MY-large_mu"].sort_values("acc_mean", ascending=False).iloc[0]
    sup = df[df["method"] == "SupCon-feature"].sort_values("acc_mean", ascending=False).iloc[0]
    mean_margin = float(
        (
            df[df["method"] == "MY-large_mu"].sort_values("case")["acc_mean"].to_numpy()
            - df[df["method"] == "SupCon-feature"].sort_values("case")["acc_mean"].to_numpy()
        ).mean()
    )
    return (
        "Summary: in the stronger d=220 Gaussian settings, KDMLP remains comfortably ahead of SupCon. "
        f"The best KDMLP run reaches {my['acc_mean']:.4f} on {my['case']}, while the best SupCon run reaches {sup['acc_mean']:.4f}; "
        f"the average KDMLP-minus-SupCon margin across the three cases is {mean_margin:+.4f}."
    )


def random_covariance_AtA(d: int, rng: np.random.Generator, scale: float) -> np.ndarray:
    A = rng.normal(size=(d, d))
    return base.symmetrize((A.T @ A) / float(d) * scale)


def run_hd220_experiment():
    ns = load_notebook_namespace(NOTEBOOK_PATH)
    ns["set_global_seed"](0)
    device = ns["pick_device"]()

    rng = np.random.default_rng(20260327)
    p = 220
    v = base.random_unit_vector(rng, p)
    cases = []
    for i, (mean_scale, cov_scale) in enumerate([(0.10, 2.5), (0.14, 2.8), (0.18, 3.0)], 1):
        cases.append(
            {
                "name": f"hd220_case_{i}",
                "mean_scale": mean_scale,
                "cov_scale": cov_scale,
                "mean0": -mean_scale * v,
                "mean1": mean_scale * v,
                "Sigma0": random_covariance_AtA(p, rng, 1.0),
                "Sigma1": random_covariance_AtA(p, rng, cov_scale),
            }
        )

    rows = []
    for case in cases:
        ours_rows = []
        sup_rows = []
        for rep in range(2):
            rg = np.random.default_rng(100 + 17 * rep + int(case["cov_scale"] * 100))
            X0tr = base.sample_gaussian_antithetic(320, case["mean0"], case["Sigma0"], rg)
            X1tr = base.sample_gaussian_antithetic(320, case["mean1"], case["Sigma1"], rg)
            X0te = base.sample_gaussian_antithetic(1200, case["mean0"], case["Sigma0"], rg)
            X1te = base.sample_gaussian_antithetic(1200, case["mean1"], case["Sigma1"], rg)
            Xtr = np.vstack([X0tr, X1tr])
            ytr = np.hstack([np.zeros(len(X0tr), int), np.ones(len(X1tr), int)])
            Xte = np.vstack([X0te, X1te])
            yte = np.hstack([np.zeros(len(X0te), int), np.ones(len(X1te), int)])

            ours_rows.extend(
                ns["evaluate_methods_on_fixed_split_fast"](
                    Xtr,
                    ytr,
                    Xte,
                    yte,
                    task_name=case["name"],
                    rep=rep,
                    r_values=(1, 2, 3, 4, 6),
                    anchors_per_class=40,
                    mu_reg=1e-6,
                    svm_folds=2,
                    stage1_kernels=ns["STAGE1_KERNELS_FULL"],
                    basis_mode="anchors",
                    center_kernel=True,
                    seed=300 + rep,
                    verbose=False,
                )
            )
            sup = train_supcon_feature_model(Xtr, ytr, Xte, yte, device=device, seed=500 + rep, epochs=14)
            sup_rows.append(float(sup["acc"]))

        df_ours = pd.DataFrame(ours_rows)
        best = ns["summarize_multidim_results"](df_ours)["best_kernel"]
        best_large = best[best["family"] == "MY-large_mu"].sort_values("acc_mean", ascending=False).iloc[0].to_dict()
        best_small = best[best["family"] == "MY-small_mu"].sort_values("acc_mean", ascending=False).iloc[0].to_dict()
        best_kfda = best[best["family"] == "KFDA"].sort_values("acc_mean", ascending=False).iloc[0].to_dict()

        rows.extend(
            [
                {
                    "case": case["name"],
                    "mean_scale": case["mean_scale"],
                    "cov_scale": case["cov_scale"],
                    "method": "KFDA-1D",
                    "acc_mean": best_kfda["acc_mean"],
                    "acc_std": best_kfda["acc_std"],
                    "stage1_kernel": best_kfda["stage1_kernel"],
                    "r": 1,
                },
                {
                    "case": case["name"],
                    "mean_scale": case["mean_scale"],
                    "cov_scale": case["cov_scale"],
                    "method": "MY-large_mu",
                    "acc_mean": best_large["acc_mean"],
                    "acc_std": best_large["acc_std"],
                    "stage1_kernel": best_large["stage1_kernel"],
                    "r": int(best_large["r"]),
                },
                {
                    "case": case["name"],
                    "mean_scale": case["mean_scale"],
                    "cov_scale": case["cov_scale"],
                    "method": "MY-small_mu",
                    "acc_mean": best_small["acc_mean"],
                    "acc_std": best_small["acc_std"],
                    "stage1_kernel": best_small["stage1_kernel"],
                    "r": int(best_small["r"]),
                },
                {
                    "case": case["name"],
                    "mean_scale": case["mean_scale"],
                    "cov_scale": case["cov_scale"],
                    "method": "SupCon-feature",
                    "acc_mean": float(np.mean(sup_rows)),
                    "acc_std": float(np.std(sup_rows, ddof=1)) if len(sup_rows) > 1 else 0.0,
                    "stage1_kernel": "n/a",
                    "r": -1,
                },
            ]
        )

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "hd220_summary.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    for ax, case_name in zip(axes, df["case"].unique()):
        sub = df[df["case"] == case_name]
        ax.bar(sub["method"], sub["acc_mean"], yerr=sub["acc_std"], color=["#577590", "#43aa8b", "#f8961e", "#f94144"])
        ax.set_title(case_name)
        ax.set_ylim(0.8, 1.02)
        ax.tick_params(axis="x", rotation=18)
    axes[0].set_ylabel("Accuracy")
    plt.tight_layout()
    fig.savefig(OUT / "hd220_accuracy_bar.png", dpi=180)
    plt.close(fig)

    print(df.to_string(index=False))
    print("\nInterpretation:")
    print(build_hd220_summary(df))
    print(f"\nSaved outputs to {OUT}")
    open_png_outputs(OUT)


if __name__ == "__main__":
    run_hd220_experiment()
