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
from sklearn.model_selection import StratifiedShuffleSplit


ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / "supcon_comparison_workspace"))

from compare_supcon_vs_ours import load_notebook_namespace, train_supcon_feature_model, NOTEBOOK_PATH  # type: ignore


sns.set_theme(style="whitegrid", context="talk")

DISPLAY_METHOD = {
    "KFDA-1D": "KFDA-1D",
    "MY-large_mu": "KDMLP large",
    "MY-small_mu": "KDMLP small",
    "SupCon": "SupCon",
}


OPENML_NOTEBOOK = ROOT / "OPENML_KFDA_vs_Our_method_multidim.ipynb"


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


def build_openml_summary(df_summary: pd.DataFrame) -> str:
    wins = int((df_summary["margin_my_minus_supcon"] > 0).sum())
    ties = int((df_summary["margin_my_minus_supcon"].abs() < 1e-12).sum())
    losses = int((df_summary["margin_my_minus_supcon"] < 0).sum())
    best = df_summary.iloc[0]
    worst = df_summary.iloc[-1]
    return (
        f"Summary: KDMLP beats SupCon on {wins} OpenML datasets, ties on {ties}, and trails on {losses}. "
        f"The largest positive margin is on {best['dataset']} ({best['margin_my_minus_supcon']:+.3f}), "
        f"while the hardest dataset for KDMLP in this scan is {worst['dataset']} ({worst['margin_my_minus_supcon']:+.3f})."
    )


def load_openml_namespace(path: Path) -> dict:
    nb = json.loads(path.read_text())
    ns: dict = {}
    for idx in (0, 1):
        exec("".join(nb["cells"][idx]["source"]), ns)
    return ns


def run_openml_experiment():
    kernel_ns = load_notebook_namespace(NOTEBOOK_PATH)
    openml_ns = load_openml_namespace(OPENML_NOTEBOOK)
    kernel_ns["set_global_seed"](0)
    device = kernel_ns["pick_device"]()

    dataset_specs = openml_ns["dataset_specs"]
    stage1_kernels = kernel_ns["STAGE1_KERNELS_QUICK"]
    r_values = (1, 2, 3, 4, 6)

    summary_rows = []
    full_rows = []
    sup_rows = []

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
        print(f"[openml] id={openml_id} name={name} shape={X.shape}")
        sss = StratifiedShuffleSplit(n_splits=1, train_size=0.6, random_state=0)
        ours_rows = []
        sup_local = []
        for rep, (tri, tei) in enumerate(sss.split(X, y01)):
            Xtr, ytr = X[tri], y01[tri]
            Xte, yte = X[tei], y01[tei]

            ours_rows.extend(
                kernel_ns["evaluate_methods_on_fixed_split_fast"](
                    Xtr,
                    ytr,
                    Xte,
                    yte,
                    task_name=name,
                    rep=rep,
                    r_values=r_values,
                    anchors_per_class=int(spec.get("anchors_per_class", 100)),
                    mu_reg=1e-6,
                    svm_folds=2,
                    stage1_kernels=stage1_kernels,
                    basis_mode="anchors",
                    center_kernel=True,
                    seed=1000 + 17 * rep + openml_id,
                    verbose=False,
                )
            )

            sup = train_supcon_feature_model(
                Xtr,
                ytr,
                Xte,
                yte,
                device=device,
                seed=2000 + 17 * rep + openml_id,
                epochs=18,
            )
            sup_local.append({"dataset": name, "openml_id": openml_id, "rep": rep, "acc": float(sup["acc"])})

        df_ours = pd.DataFrame(ours_rows)
        summary = kernel_ns["summarize_multidim_results"](df_ours)
        df_best = summary["best_kernel"].copy()
        full_rows.append(df_ours.assign(openml_id=openml_id, dataset=name))

        best_kfda = df_best[df_best["family"] == "KFDA"].sort_values("acc_mean", ascending=False).iloc[0].to_dict()
        best_large = df_best[df_best["family"] == "MY-large_mu"].sort_values("acc_mean", ascending=False).iloc[0].to_dict()
        best_small = df_best[df_best["family"] == "MY-small_mu"].sort_values("acc_mean", ascending=False).iloc[0].to_dict()
        df_sup = pd.DataFrame(sup_local)
        sup_rows.append(df_sup)
        sup_mean = float(df_sup["acc"].mean())
        sup_std = float(df_sup["acc"].std(ddof=1)) if len(df_sup) > 1 else 0.0

        my_best_acc = max(best_large["acc_mean"], best_small["acc_mean"])
        my_best_family = "MY-large_mu" if best_large["acc_mean"] >= best_small["acc_mean"] else "MY-small_mu"

        summary_rows.append(
            {
                "openml_id": openml_id,
                "dataset": name,
                "kfda_acc": best_kfda["acc_mean"],
                "kfda_kernel": best_kfda["stage1_kernel"],
                "my_large_acc": best_large["acc_mean"],
                "my_large_kernel": best_large["stage1_kernel"],
                "my_large_r": int(best_large["r"]),
                "my_small_acc": best_small["acc_mean"],
                "my_small_kernel": best_small["stage1_kernel"],
                "my_small_r": int(best_small["r"]),
                "my_best_acc": my_best_acc,
                "my_best_family": my_best_family,
                "supcon_acc": sup_mean,
                "supcon_std": sup_std,
                "margin_my_minus_supcon": float(my_best_acc - sup_mean),
            }
        )

    df_summary = pd.DataFrame(summary_rows).sort_values("margin_my_minus_supcon", ascending=False).reset_index(drop=True)
    df_full = pd.concat(full_rows, ignore_index=True)
    df_sup = pd.concat(sup_rows, ignore_index=True)

    df_summary.to_csv(OUT / "openml_summary.csv", index=False)
    df_full.to_csv(OUT / "openml_ours_full.csv", index=False)
    df_sup.to_csv(OUT / "openml_supcon_rows.csv", index=False)

    fig, ax = plt.subplots(figsize=(11, 7))
    sub = df_summary.sort_values("margin_my_minus_supcon")
    colors = ["#43aa8b" if v > 0 else "#f94144" for v in sub["margin_my_minus_supcon"]]
    ax.barh(sub["dataset"], sub["margin_my_minus_supcon"], color=colors)
    ax.axvline(0.0, color="black", linewidth=1.2)
    ax.set_xlabel("MY-best minus SupCon accuracy")
    ax.set_title("OpenML: our method vs SupCon")
    plt.tight_layout()
    fig.savefig(OUT / "openml_margin_vs_supcon.png", dpi=180)
    plt.close(fig)

    plot_rows = []
    for _, row in df_summary.iterrows():
        plot_rows.extend(
            [
                {"dataset": row["dataset"], "method": "KFDA-1D", "acc": row["kfda_acc"]},
                {"dataset": row["dataset"], "method": DISPLAY_METHOD.get(row["my_best_family"], row["my_best_family"]), "acc": row["my_best_acc"]},
                {"dataset": row["dataset"], "method": "SupCon", "acc": row["supcon_acc"]},
            ]
        )
    df_plot = pd.DataFrame(plot_rows)
    fig, ax = plt.subplots(figsize=(13, 7))
    sns.barplot(data=df_plot, x="dataset", y="acc", hue="method", ax=ax)
    ax.set_title("OpenML summary comparison")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    fig.savefig(OUT / "openml_summary_barplot.png", dpi=180)
    plt.close(fig)

    print(df_summary.to_string(index=False))
    print("\nInterpretation:")
    print(build_openml_summary(df_summary))
    print(f"\nSaved outputs to {OUT}")
    open_png_outputs(OUT)


if __name__ == "__main__":
    run_openml_experiment()
