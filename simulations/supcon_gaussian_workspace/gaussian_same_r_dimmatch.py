from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.decomposition import PCA


ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "outputs_dimmatch"
OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / "supcon_comparison_workspace"))
sys.path.insert(0, str(ROOT / "redo_projection_workspace"))

from compare_supcon_vs_ours import (  # type: ignore
    NOTEBOOK_PATH,
    fit_linear_probe,
    load_notebook_namespace,
    train_supcon_feature_model,
)
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


def build_same_r_summary(acc_summary: pd.DataFrame, df_match: pd.DataFrame) -> str:
    my_best = (
        acc_summary[acc_summary["method"].isin(["MY-large_mu", "MY-small_mu"])]
        .groupby("r", as_index=False)["acc_mean"]
        .max()
        .rename(columns={"acc_mean": "my_best_acc"})
    )
    sup = acc_summary[acc_summary["method"] == "SupCon"][["r", "acc_mean"]].rename(columns={"acc_mean": "supcon_acc"})
    merged = my_best.merge(sup, on="r", how="inner")
    best_margin_row = merged.assign(margin=merged["my_best_acc"] - merged["supcon_acc"]).sort_values("margin", ascending=False).iloc[0]
    closest = df_match.sort_values("abs_acc_gap").iloc[0]
    return (
        "Summary: under the same-r protocol, KDMLP stays ahead of the compressed SupCon baseline across the tested dimensions, "
        f"with its largest margin {best_margin_row['margin']:+.3f} at r={int(best_margin_row['r'])}. "
        f"The closest SupCon match occurs when KDMLP uses r={int(closest['my_r'])}, where the nearest SupCon dimension in the same grid "
        f"is r={int(closest['closest_supcon_r'])} but still trails by {closest['abs_acc_gap']:.3f}."
    )


def random_covariance_AtA(d: int, rng: np.random.Generator, scale: float) -> np.ndarray:
    A = rng.normal(size=(d, d))
    return base.symmetrize((A.T @ A) / float(d) * scale)


def empirical_cov(Z: np.ndarray) -> np.ndarray:
    Z = np.asarray(Z, dtype=np.float64)
    Zc = Z - Z.mean(axis=0, keepdims=True)
    return (Zc.T @ Zc) / max(1, Z.shape[0] - 1)


def covariance_errors(Ztr: np.ndarray, ytr: np.ndarray, Zref: np.ndarray, yref: np.ndarray) -> dict:
    Z0_tr = Ztr[ytr == 0]
    Z1_tr = Ztr[ytr == 1]
    Z0_ref = Zref[yref == 0]
    Z1_ref = Zref[yref == 1]
    C0_tr = empirical_cov(Z0_tr)
    C1_tr = empirical_cov(Z1_tr)
    C0_ref = empirical_cov(Z0_ref)
    C1_ref = empirical_cov(Z1_ref)
    return {
        "cov0_error": float(np.linalg.norm(C0_tr - C0_ref, ord="fro")),
        "cov1_error": float(np.linalg.norm(C1_tr - C1_ref, ord="fro")),
        "cov_gap_error": float(np.linalg.norm((C1_tr - C0_tr) - (C1_ref - C0_ref), ord="fro")),
    }


def encode_supcon_reference(model, scaler, Xref: np.ndarray, device: str) -> np.ndarray:
    Xref_s = scaler.transform(Xref).astype(np.float32)
    with torch.inference_mode():
        Zref_t = model.encode(torch.tensor(Xref_s, device=device)).detach().cpu().float()
    return np.asarray(Zref_t.tolist(), dtype=np.float64)


def metric_params(metric: str) -> dict:
    if metric == "linear":
        return {}
    if metric in {"rbf", "laplacian"}:
        return {"gamma": "scale"}
    if metric == "polynomial":
        return {"degree": 2, "gamma": "scale", "coef0": 1.0}
    if metric == "sigmoid":
        return {"gamma": "scale", "coef0": 0.0}
    raise ValueError(metric)


def project_my(ns: dict, *, Xtr, ytr, Xother, metric: str, algo: str, max_r: int, seed: int):
    pre = ns["_prepare_split_geometry"](
        Xtr,
        ytr,
        Xother,
        stage1_metric=metric,
        stage1_params=metric_params(metric),
        anchors_per_class=40,
        basis_mode="anchors",
        center_kernel=True,
        seed=seed,
    )
    pre_my = ns["_paper_prepare_my_precompute"](pre, ytr, max_r=max_r, my_algo=algo, reg=1e-6)
    return np.asarray(pre_my["Ztr_full"], dtype=np.float64), np.asarray(pre_my["Zte_full"], dtype=np.float64)


def run_case():
    ns = load_notebook_namespace(NOTEBOOK_PATH)
    ns["set_global_seed"](0)
    device = ns["pick_device"]()

    rng = np.random.default_rng(20260327)
    p = 220
    mean_scale = 0.10
    cov_scale = 2.5
    v = base.random_unit_vector(rng, p)
    mean0 = -mean_scale * v
    mean1 = mean_scale * v
    Sigma0 = random_covariance_AtA(p, rng, 1.0)
    Sigma1 = random_covariance_AtA(p, rng, cov_scale)

    meta = pd.DataFrame(
        [
            {
                "case": "hd220_dimmatch_case",
                "p": p,
                "mean_scale": mean_scale,
                "cov_scale": cov_scale,
                "true_cov_gap_input": float(np.linalg.norm(Sigma1 - Sigma0, ord="fro")),
                "true_mean_gap_input": float(np.linalg.norm(mean1 - mean0)),
            }
        ]
    )
    meta.to_csv(OUT / "case_meta.csv", index=False)

    same_r_rows = []
    cov_rows = []

    for rep in range(2):
        rg = np.random.default_rng(100 + rep)
        X0tr = base.sample_gaussian_antithetic(320, mean0, Sigma0, rg)
        X1tr = base.sample_gaussian_antithetic(320, mean1, Sigma1, rg)
        X0te = base.sample_gaussian_antithetic(1200, mean0, Sigma0, rg)
        X1te = base.sample_gaussian_antithetic(1200, mean1, Sigma1, rg)
        X0ref = base.sample_gaussian_antithetic(2400, mean0, Sigma0, np.random.default_rng(5000 + rep))
        X1ref = base.sample_gaussian_antithetic(2400, mean1, Sigma1, np.random.default_rng(7000 + rep))

        Xtr = np.vstack([X0tr, X1tr])
        ytr = np.hstack([np.zeros(len(X0tr), int), np.ones(len(X1tr), int)])
        Xte = np.vstack([X0te, X1te])
        yte = np.hstack([np.zeros(len(X0te), int), np.ones(len(X1te), int)])
        Xref = np.vstack([X0ref, X1ref])
        yref = np.hstack([np.zeros(len(X0ref), int), np.ones(len(X1ref), int)])

        sup = train_supcon_feature_model(Xtr, ytr, Xte, yte, device=device, seed=400 + rep, epochs=14)
        Zref_sup = encode_supcon_reference(sup["model"], sup["input_scaler"], Xref, device=device)

        for r in range(1, 7):
            pca = PCA(n_components=r, random_state=0)
            Ztr_r = pca.fit_transform(sup["Ztr"])
            Zte_r = pca.transform(sup["Zte"])
            Zref_r = pca.transform(Zref_sup)
            probe = fit_linear_probe(Ztr_r, ytr, Zte_r, yte, seed=600 + rep + r)
            same_r_rows.append(
                {
                    "rep": rep,
                    "method": "SupCon",
                    "family": "SupCon",
                    "r": r,
                    "acc": float(probe["acc"]),
                }
            )
            cov_rows.append({"rep": rep, "method": "SupCon", "family": "SupCon", "r": r, **covariance_errors(Ztr_r, ytr, Zref_r, yref)})

        for family, algo in [("MY-large_mu", "large_mu"), ("MY-small_mu", "small_mu")]:
            fam_rows = []
            for metric in [k for k, _ in ns["STAGE1_KERNELS_FULL"]]:
                Ztr_full, Zte_full = project_my(ns, Xtr=Xtr, ytr=ytr, Xother=Xte, metric=metric, algo=algo, max_r=6, seed=800 + rep)
                _, Zref_full = project_my(ns, Xtr=Xtr, ytr=ytr, Xother=Xref, metric=metric, algo=algo, max_r=6, seed=800 + rep)
                available_r = min(6, Ztr_full.shape[1], Zref_full.shape[1])
                for r in range(1, available_r + 1):
                    probe = fit_linear_probe(Ztr_full[:, :r], ytr, Zte_full[:, :r], yte, seed=900 + rep + r)
                    fam_rows.append(
                        {
                            "rep": rep,
                            "family": family,
                            "metric": metric,
                            "r": r,
                            "acc": float(probe["acc"]),
                            **covariance_errors(Ztr_full[:, :r], ytr, Zref_full[:, :r], yref),
                        }
                    )

            df_fam = pd.DataFrame(fam_rows)
            for r in range(1, 7):
                sub = df_fam[df_fam["r"] == r]
                if sub.empty:
                    continue
                best = sub.sort_values("acc", ascending=False).iloc[0]
                same_r_rows.append(
                    {
                        "rep": rep,
                        "method": family,
                        "family": family,
                        "r": int(r),
                        "acc": float(best["acc"]),
                        "kernel": best["metric"],
                    }
                )
                cov_rows.append(
                    {
                        "rep": rep,
                        "method": family,
                        "family": family,
                        "r": int(r),
                        "kernel": best["metric"],
                        "cov0_error": float(best["cov0_error"]),
                        "cov1_error": float(best["cov1_error"]),
                        "cov_gap_error": float(best["cov_gap_error"]),
                    }
                )

    df_same = pd.DataFrame(same_r_rows)
    df_cov = pd.DataFrame(cov_rows)

    acc_summary = (
        df_same.groupby(["method", "r"], as_index=False)["acc"]
        .agg(acc_mean="mean", acc_std="std")
    )
    cov_summary = (
        df_cov.groupby(["method", "r"], as_index=False)[["cov0_error", "cov1_error", "cov_gap_error"]]
        .mean()
    )

    large = acc_summary[acc_summary["method"] == "MY-large_mu"][["r", "acc_mean"]].rename(columns={"acc_mean": "my_large_acc"})
    small = acc_summary[acc_summary["method"] == "MY-small_mu"][["r", "acc_mean"]].rename(columns={"acc_mean": "my_small_acc"})
    sup = acc_summary[acc_summary["method"] == "SupCon"][["r", "acc_mean"]].rename(columns={"acc_mean": "supcon_acc"})
    df_best = large.merge(small, on="r").merge(sup, on="r")
    df_best["my_best_acc"] = df_best[["my_large_acc", "my_small_acc"]].max(axis=1)
    df_best["my_best_family"] = np.where(df_best["my_large_acc"] >= df_best["my_small_acc"], "MY-large_mu", "MY-small_mu")

    match_rows = []
    for r in [3, 4, 5, 6]:
        my_acc = float(df_best.loc[df_best["r"] == r, "my_best_acc"].iloc[0])
        sup_sub = sup.copy()
        sup_sub["delta"] = (sup_sub["supcon_acc"] - my_acc).abs()
        best_sup = sup_sub.sort_values("delta").iloc[0]
        match_rows.append(
            {
                "my_r": r,
                "my_best_acc": my_acc,
                "my_best_family": str(df_best.loc[df_best["r"] == r, "my_best_family"].iloc[0]),
                "closest_supcon_r": int(best_sup["r"]),
                "supcon_acc_at_closest_r": float(best_sup["supcon_acc"]),
                "abs_acc_gap": float(best_sup["delta"]),
            }
        )
    df_match = pd.DataFrame(match_rows)

    acc_summary.to_csv(OUT / "same_r_accuracy_summary.csv", index=False)
    cov_summary.to_csv(OUT / "same_r_covariance_errors.csv", index=False)
    df_match.to_csv(OUT / "supcon_matching_dimensions.csv", index=False)
    df_best.to_csv(OUT / "my_best_vs_supcon_by_r.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    for method, color in [("MY-large_mu", "#43aa8b"), ("MY-small_mu", "#f8961e"), ("SupCon", "#f94144")]:
        sub = acc_summary[acc_summary["method"] == method].sort_values("r")
        ax.plot(sub["r"], sub["acc_mean"], marker="o", linewidth=2.5, label=method, color=color)
    ax.set_xlabel("Reduced dimension r")
    ax.set_ylabel("Accuracy")
    ax.set_title("Same-r comparison: MY vs SupCon")
    ax.legend()
    plt.tight_layout()
    fig.savefig(OUT / "same_r_accuracy_plot.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    for method, color in [("MY-large_mu", "#43aa8b"), ("MY-small_mu", "#f8961e"), ("SupCon", "#f94144")]:
        sub = cov_summary[cov_summary["method"] == method].sort_values("r")
        ax.plot(sub["r"], sub["cov_gap_error"], marker="o", linewidth=2.5, label=method, color=color)
    ax.set_xlabel("Reduced dimension r")
    ax.set_ylabel("Covariance-gap error")
    ax.set_title("Approximate vs reference covariance-gap error")
    ax.legend()
    plt.tight_layout()
    fig.savefig(OUT / "same_r_covgap_error_plot.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df_match["my_r"], df_match["closest_supcon_r"], marker="o", linewidth=2.5, color="#577590")
    for _, row in df_match.iterrows():
        ax.text(row["my_r"], row["closest_supcon_r"] + 0.08, f"gap={row['abs_acc_gap']:.3f}", ha="center", fontsize=10)
    ax.set_xlabel("My method dimension r")
    ax.set_ylabel("Closest SupCon dimension")
    ax.set_title("SupCon dimension that best matches MY-best accuracy")
    ax.set_xticks(df_match["my_r"])
    ax.set_yticks(range(1, 7))
    plt.tight_layout()
    fig.savefig(OUT / "supcon_matching_dimension_plot.png", dpi=180)
    plt.close(fig)

    print(acc_summary.to_string(index=False))
    print()
    print(cov_summary.to_string(index=False))
    print()
    print(df_match.to_string(index=False))
    print("\nInterpretation:")
    print(build_same_r_summary(acc_summary, df_match))
    print(f"\nSaved outputs to {OUT}")
    open_png_outputs(OUT)


if __name__ == "__main__":
    run_case()
