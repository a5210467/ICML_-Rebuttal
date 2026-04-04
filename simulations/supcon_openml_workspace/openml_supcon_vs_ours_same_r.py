from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedShuffleSplit


ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = ROOT.parent
PROJECT_ROOT = REPO_ROOT.parent
OUT = Path(__file__).resolve().parent / "outputs_same_r"
OUT.mkdir(parents=True, exist_ok=True)

for candidate in (
    ROOT / "supcon_comparison_workspace",
    REPO_ROOT / "supcon_comparison_workspace",
    PROJECT_ROOT / "supcon_comparison_workspace",
):
    if candidate.exists():
        sys.path.insert(0, str(candidate))
        break

from compare_supcon_vs_ours import (  # type: ignore
    NOTEBOOK_PATH,
    fit_linear_probe,
    load_notebook_namespace,
    train_supcon_feature_model,
)


sns.set_theme(style="whitegrid", context="talk")

for candidate in (
    ROOT / "OPENML_KFDA_vs_Our_method_multidim.ipynb",
    REPO_ROOT / "OPENML_KFDA_vs_Our_method_multidim.ipynb",
    PROJECT_ROOT / "OPENML_KFDA_vs_Our_method_multidim.ipynb",
):
    if candidate.exists():
        OPENML_NOTEBOOK = candidate
        break
else:
    OPENML_NOTEBOOK = ROOT / "OPENML_KFDA_vs_Our_method_multidim.ipynb"
R_VALUES = (1, 2, 3, 4, 5, 6, 32, 64)


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


def build_openml_same_r_summary(df_dim32_64: pd.DataFrame) -> str:
    lines = []
    for r in (32, 64):
        sub = df_dim32_64[df_dim32_64["r"] == r]
        if len(sub) == 0:
            continue
        wins_sup = int((sub["my_best_acc"] > sub["supcon_acc"]).sum())
        losses_sup = int((sub["my_best_acc"] < sub["supcon_acc"]).sum())
        mean_margin_sup = float((sub["my_best_acc"] - sub["supcon_acc"]).mean())
        wins_kfda = int((sub["my_best_acc"] > sub["kfda_acc"]).sum())
        losses_kfda = int((sub["my_best_acc"] < sub["kfda_acc"]).sum())
        mean_margin_kfda = float((sub["my_best_acc"] - sub["kfda_acc"]).mean())
        lines.append(
            f"r={r}: KDMLP beats SupCon on {wins_sup} datasets and KFDA on {wins_kfda} datasets "
            f"(margins {mean_margin_sup:+.3f} vs SupCon, {mean_margin_kfda:+.3f} vs KFDA; "
            f"losses {losses_sup} and {losses_kfda}, respectively)"
        )
    if not lines:
        return "Summary: no r=32 or r=64 rows were generated."
    return "Summary: " + "; ".join(lines) + ". Binary KFDA remains rank-1, so its same-r reference is repeated across the tested r grid."


def load_openml_namespace(path: Path) -> dict:
    nb = json.loads(path.read_text())
    ns: dict = {}
    for idx in (0, 1):
        exec("".join(nb["cells"][idx]["source"]), ns)
    return ns


def summarize_my_same_r(df_ours: pd.DataFrame) -> pd.DataFrame:
    df_mean = (
        df_ours.groupby(["openml_id", "task", "family", "r", "stage1_kernel"], as_index=False)["acc"]
        .agg(acc_mean="mean", acc_std=lambda s: float(np.std(s, ddof=1)) if len(s) > 1 else 0.0)
    )
    df_best = (
        df_mean[df_mean["family"].isin(["MY-large_mu", "MY-small_mu"])]
        .sort_values(["openml_id", "task", "family", "r", "acc_mean"], ascending=[True, True, True, True, False])
        .groupby(["openml_id", "task", "family", "r"], as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    return df_best


def summarize_kfda_same_r(df_ours: pd.DataFrame) -> pd.DataFrame:
    df_mean = (
        df_ours[df_ours["family"] == "KFDA"]
        .groupby(["openml_id", "task", "stage1_kernel"], as_index=False)["acc"]
        .agg(acc_mean="mean", acc_std=lambda s: float(np.std(s, ddof=1)) if len(s) > 1 else 0.0)
    )
    return (
        df_mean.sort_values(["openml_id", "task", "acc_mean"], ascending=[True, True, False])
        .groupby(["openml_id", "task"], as_index=False)
        .head(1)
        .reset_index(drop=True)
    )


def run_openml_same_r():
    kernel_ns = load_notebook_namespace(NOTEBOOK_PATH)
    openml_ns = load_openml_namespace(OPENML_NOTEBOOK)
    kernel_ns["set_global_seed"](0)
    device = kernel_ns["pick_device"]()

    dataset_specs = openml_ns["dataset_specs"]
    stage1_kernels = kernel_ns["STAGE1_KERNELS_QUICK"]

    rows_my = []
    rows_sup = []
    compact_rows = []
    task_to_id = {}

    for spec in dataset_specs:
        openml_id = int(spec["id"])
        X, y_raw, name = openml_ns["load_openml_numeric"](openml_id)
        y01, mask = openml_ns["make_binary_labels"](
            y_raw,
            mode=spec.get("label_mode", "auto"),
            positive=spec.get("positive", None),
            class_a=spec.get("class_a", None),
            class_b=spec.get("class_b", None),
        )
        if mask is not None:
            X = X[mask]
        y01 = np.asarray(y01, dtype=int)
        task_to_id[name] = openml_id

        print(f"[openml-same-r] id={openml_id} name={name} shape={X.shape}")
        sss = StratifiedShuffleSplit(n_splits=1, train_size=0.6, random_state=0)
        tri, tei = next(sss.split(X, y01))
        Xtr, ytr = X[tri], y01[tri]
        Xte, yte = X[tei], y01[tei]

        rows_my.extend(
            kernel_ns["evaluate_methods_on_fixed_split_fast"](
                Xtr,
                ytr,
                Xte,
                yte,
                task_name=name,
                rep=0,
                r_values=R_VALUES,
                anchors_per_class=int(spec.get("anchors_per_class", 100)),
                mu_reg=1e-6,
                svm_folds=2,
                stage1_kernels=stage1_kernels,
                basis_mode="anchors",
                center_kernel=True,
                seed=1000 + openml_id,
                verbose=False,
            )
        )

        sup = train_supcon_feature_model(
            Xtr,
            ytr,
            Xte,
            yte,
            device=device,
            seed=2000 + openml_id,
            epochs=18,
        )
        max_sup_r = min(max(R_VALUES), sup["Ztr"].shape[1])
        pca = PCA(n_components=max_sup_r, random_state=0)
        Ztr_pca = pca.fit_transform(sup["Ztr"])
        Zte_pca = pca.transform(sup["Zte"])

        for r in R_VALUES:
            if r > Ztr_pca.shape[1]:
                continue
            probe = fit_linear_probe(Ztr_pca[:, :r], ytr, Zte_pca[:, :r], yte, seed=3000 + openml_id + r)
            rows_sup.append(
                {
                    "openml_id": openml_id,
                    "dataset": name,
                    "method": "SupCon",
                    "r": int(r),
                    "acc": float(probe["acc"]),
                }
            )

    df_my = pd.DataFrame(rows_my)
    df_my["openml_id"] = df_my["task"].map(task_to_id)
    df_sup = pd.DataFrame(rows_sup)

    df_my_best = summarize_my_same_r(df_my)
    df_kfda_best = summarize_kfda_same_r(df_my)
    df_sup_mean = (
        df_sup.groupby(["openml_id", "dataset", "r"], as_index=False)["acc"]
        .agg(acc_mean="mean", acc_std=lambda s: float(np.std(s, ddof=1)) if len(s) > 1 else 0.0)
    )

    for openml_id, dataset in sorted(df_sup[["openml_id", "dataset"]].drop_duplicates().itertuples(index=False)):
        my_large = df_my_best[(df_my_best["openml_id"] == openml_id) & (df_my_best["family"] == "MY-large_mu")][["r", "acc_mean"]].rename(columns={"acc_mean": "my_large_acc"})
        my_small = df_my_best[(df_my_best["openml_id"] == openml_id) & (df_my_best["family"] == "MY-small_mu")][["r", "acc_mean"]].rename(columns={"acc_mean": "my_small_acc"})
        sup_sub = df_sup_mean[(df_sup_mean["openml_id"] == openml_id)][["r", "acc_mean"]].rename(columns={"acc_mean": "supcon_acc"})
        kfda_row = df_kfda_best[df_kfda_best["openml_id"] == openml_id]
        kfda_acc = float(kfda_row["acc_mean"].iloc[0]) if not kfda_row.empty else np.nan
        kfda_kernel = str(kfda_row["stage1_kernel"].iloc[0]) if not kfda_row.empty else ""
        kfda_sub = pd.DataFrame({"r": list(R_VALUES), "kfda_acc": [kfda_acc] * len(R_VALUES), "kfda_kernel": [kfda_kernel] * len(R_VALUES)})
        merged = my_large.merge(my_small, on="r", how="outer").merge(sup_sub, on="r", how="outer").merge(kfda_sub, on="r", how="left").sort_values("r")
        merged["my_best_acc"] = merged[["my_large_acc", "my_small_acc"]].max(axis=1)
        merged["my_best_family"] = np.where(merged["my_large_acc"] >= merged["my_small_acc"], "MY-large_mu", "MY-small_mu")
        merged["margin_vs_kfda"] = merged["my_best_acc"] - merged["kfda_acc"]
        merged["margin_vs_supcon"] = merged["my_best_acc"] - merged["supcon_acc"]
        merged["openml_id"] = openml_id
        merged["dataset"] = dataset
        compact_rows.append(merged)

    df_compact = pd.concat(compact_rows, ignore_index=True)
    df_dim32_64 = df_compact[df_compact["r"].isin([32, 64])].copy()

    df_my.to_csv(OUT / "openml_my_full_same_r.csv", index=False)
    df_my_best.to_csv(OUT / "openml_my_best_same_r.csv", index=False)
    df_kfda_best.to_csv(OUT / "openml_kfda_best_same_r.csv", index=False)
    df_sup_mean.to_csv(OUT / "openml_supcon_same_r.csv", index=False)
    df_compact.to_csv(OUT / "openml_same_r_compact.csv", index=False)
    df_dim32_64.to_csv(OUT / "openml_dim32_64_comparison.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    mean_curve = (
        df_compact.groupby("r", as_index=False)[["my_best_acc", "supcon_acc", "kfda_acc"]]
        .mean()
        .sort_values("r")
    )
    ax.plot(mean_curve["r"], mean_curve["my_best_acc"], marker="o", linewidth=2.5, color="#43aa8b", label="MY-best")
    ax.plot(mean_curve["r"], mean_curve["supcon_acc"], marker="o", linewidth=2.5, color="#f94144", label="SupCon")
    ax.plot(mean_curve["r"], mean_curve["kfda_acc"], marker="o", linewidth=2.5, linestyle="--", color="#577590", label="KFDA")
    ax.set_xlabel("Dimension r")
    ax.set_ylabel("Mean accuracy across OpenML datasets")
    ax.set_title("OpenML same-r comparison (KDMLP vs SupCon vs KFDA)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(OUT / "openml_same_r_mean_curve.png", dpi=180)
    plt.close(fig)

    plot32 = df_dim32_64.melt(
        id_vars=["openml_id", "dataset", "r"],
        value_vars=["my_best_acc", "supcon_acc", "kfda_acc"],
        var_name="method",
        value_name="acc",
    )
    fig, ax = plt.subplots(figsize=(13, 7))
    sns.barplot(data=plot32, x="dataset", y="acc", hue="method", ax=ax)
    ax.set_title("OpenML: KDMLP vs SupCon vs KFDA at r=32 or 64")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    fig.savefig(OUT / "openml_dim32_64_barplot.png", dpi=180)
    plt.close(fig)

    print(df_dim32_64.to_string(index=False))
    print("\nInterpretation:")
    print(build_openml_same_r_summary(df_dim32_64))
    print(f"\nSaved outputs to {OUT}")
    open_png_outputs(OUT)


if __name__ == "__main__":
    run_openml_same_r()
