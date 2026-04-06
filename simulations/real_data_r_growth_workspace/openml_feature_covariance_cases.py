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
REPO_ROOT = ROOT.parent
PROJECT_ROOT = REPO_ROOT.parent
WORKSPACE = Path(__file__).resolve().parent
OUT = WORKSPACE / "outputs" / "openml_feature_covariance_cases"
OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / "redo_projection_workspace"))
import redo_projection_cv_experiment as base  # type: ignore


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


OPENML_OUT = ROOT / "supcon_openml_workspace" / "outputs_same_r"
TARGETS = (
    ("A", "banana"),
    ("B", "ringnorm"),
    ("C", "diabetes"),
)
STAGE1_KERNELS = (
    ("linear", {}),
    ("rbf", {"gamma": "scale"}),
    ("laplacian", {"gamma": "scale"}),
)

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


def load_openml_namespace(path: Path) -> dict:
    nb = json.loads(path.read_text())
    ns: dict = {}
    for idx in (0, 1):
        exec("".join(nb["cells"][idx]["source"]), ns)
    return ns


def load_accuracy_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    df_compact = pd.read_csv(OPENML_OUT / "openml_same_r_compact.csv")
    df_my_best = pd.read_csv(OPENML_OUT / "openml_my_best_same_r.csv")
    return df_compact, df_my_best


def representative_kernel_for_dataset(dataset: str, df_compact: pd.DataFrame, df_my_best: pd.DataFrame) -> dict:
    sub = df_compact[df_compact["dataset"] == dataset].copy().sort_values(["my_best_acc", "r"], ascending=[False, True])
    if sub.empty:
        raise ValueError(f"Missing compact rows for dataset {dataset!r}.")
    best = sub.iloc[0]
    fam = str(best["my_best_family"])
    r = int(best["r"])
    sub_my = df_my_best[
        (df_my_best["task"] == dataset)
        & (df_my_best["family"] == fam)
        & (df_my_best["r"] == r)
    ].copy().sort_values("acc_mean", ascending=False)
    if sub_my.empty:
        raise ValueError(f"Missing KDMLP best rows for dataset {dataset!r}, family {fam!r}, r={r}.")
    row = sub_my.iloc[0]
    return {
        "dataset": dataset,
        "family": fam,
        "best_r": r,
        "best_acc": float(best["my_best_acc"]),
        "rep_kernel": str(row["stage1_kernel"]),
    }


def compute_feature_diagnostics(dataset: str, ns: dict) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    spec = next(spec for spec in ns["dataset_specs"] if ns["load_openml_numeric"](int(spec["id"]))[2] == dataset)
    X, y_raw, _ = ns["load_openml_numeric"](int(spec["id"]))
    y01, mask = ns["make_binary_labels"](
        y_raw,
        mode=spec.get("label_mode", "auto"),
        positive=spec.get("positive", None),
        class_a=spec.get("class_a", None),
        class_b=spec.get("class_b", None),
    )
    if mask is not None:
        X = X[mask]
    y01 = np.asarray(y01, dtype=int)

    sss = StratifiedShuffleSplit(n_splits=1, train_size=0.6, random_state=0)
    tri, tei = next(sss.split(X, y01))
    Xtr, ytr = X[tri], y01[tri]
    Xte, yte = X[tei], y01[tei]
    X0tr, X1tr = Xtr[ytr == 0], Xtr[ytr == 1]
    X0te, X1te = Xte[yte == 0], Xte[yte == 1]

    rows = []
    for metric, raw_params in STAGE1_KERNELS:
        params = base.normalize_kernel_params(metric, raw_params, Xtr)
        diag = base.feature_space_diagnostics(X0tr, X1tr, X0te, X1te, metric, params)
        rows.append(
            {
                "dataset": dataset,
                "kernel": metric,
                "ref_D_sigma_feature": float(diag["ref_D_sigma_feature"]),
                "cov_gap_error_feature": float(diag["cov_gap_error_feature"]),
                "ratio": float(diag["ref_D_sigma_feature"] / max(diag["cov_gap_error_feature"], 1e-12)),
                "ref_D_mu_feature": float(diag["ref_D_mu_feature"]),
            }
        )
    return pd.DataFrame(rows), Xtr, Xte


def build_interpretation(row: pd.Series) -> str:
    dataset = row["dataset"]
    ratio = float(row["ratio"])
    if dataset == "banana":
        return (
            f"On {dataset}, the selected {row['rep_kernel']} kernel shows a held-out feature-space covariance signal "
            f"that is clearly larger than the train/test estimation discrepancy (ratio {ratio:.2f}), and KDMLP "
            "improves strongly over KFDA."
        )
    if dataset == "ringnorm":
        return (
            f"On {dataset}, the selected {row['rep_kernel']} kernel still has a measurable feature-space covariance signal "
            f"(ratio {ratio:.2f}), but that signal does not translate into a gain over KFDA; here the binary KFDA baseline "
            "is already very strong and remains slightly better."
        )
    return (
        f"On {dataset}, the selected {row['rep_kernel']} kernel has feature-space covariance signal below its train/test "
        f"estimation discrepancy (ratio {ratio:.2f}), so KDMLP does not get a stable covariance advantage and only shows "
        "a marginal improvement at the largest r rather than a robust win over KFDA."
    )


def plot_dataset_case(
    label: str,
    dataset: str,
    diag_df: pd.DataFrame,
    curve_df: pd.DataFrame,
    rep_kernel: str,
    rep_family: str,
    out_path: Path,
) -> None:
    diag_sub = diag_df.copy().reset_index(drop=True)
    rep_row = diag_sub[diag_sub["kernel"] == rep_kernel].iloc[0]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))

    ax = axes[0]
    x = np.arange(len(diag_sub))
    width = 0.35
    bars1 = ax.bar(x - width / 2, diag_sub["ref_D_sigma_feature"], width=width, color="#277da1", label="reference covariance gap")
    bars2 = ax.bar(x + width / 2, diag_sub["cov_gap_error_feature"], width=width, color="#f94144", label="train/test covariance error")
    for i, row in enumerate(diag_sub.itertuples(index=False)):
        ratio = float(row.ratio)
        ax.text(i, max(row.ref_D_sigma_feature, row.cov_gap_error_feature) * 1.02, f"{ratio:.2f}", ha="center", va="bottom", fontsize=10)
        if row.kernel == rep_kernel:
            bars1[i].set_edgecolor("black")
            bars2[i].set_edgecolor("black")
            bars1[i].set_linewidth(2.5)
            bars2[i].set_linewidth(2.5)
    ax.set_xticks(x)
    ax.set_xticklabels(diag_sub["kernel"])
    ax.set_yscale("log")
    ax.set_ylabel("feature-space magnitude")
    ax.set_title(f"{label}: {dataset}\nFeature-space covariance gap vs estimation error")
    ax.legend(loc="upper right", fontsize=10)

    ax = axes[1]
    ax.plot(curve_df["r"], curve_df["my_best_acc"], marker="o", linewidth=2.8, color="#43aa8b", label="KDMLP best")
    ax.plot(curve_df["r"], curve_df["supcon_acc"], marker="o", linewidth=2.4, color="#f94144", label="SupCon")
    ax.plot(curve_df["r"], curve_df["kfda_acc"], marker="o", linewidth=2.2, linestyle="--", color="#577590", label="KFDA")
    ax.set_xlabel("target dimension r")
    ax.set_ylabel("test accuracy")
    ax.set_title(f"{label}: {dataset}\nAccuracy curves")
    ax.text(
        0.03,
        0.04,
        (
            f"selected KDMLP family: {rep_family.replace('MY-', 'KDMLP ').replace('_mu', '')}\n"
            f"selected kernel: {rep_kernel}\n"
            f"ref gap = {rep_row['ref_D_sigma_feature']:.4f}\n"
            f"error = {rep_row['cov_gap_error_feature']:.4f}\n"
            f"ratio = {rep_row['ratio']:.2f}"
        ),
        transform=ax.transAxes,
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc"},
    )
    ax.legend(loc="best", fontsize=10)

    plt.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_summary_text(df_summary: pd.DataFrame) -> str:
    banana = df_summary[df_summary["dataset"] == "banana"].iloc[0]
    ringnorm = df_summary[df_summary["dataset"] == "ringnorm"].iloc[0]
    diabetes = df_summary[df_summary["dataset"] == "diabetes"].iloc[0]
    return (
        f"banana: selected kernel {banana['rep_kernel']}, covariance-gap ratio {banana['ratio']:.2f}, "
        f"KDMLP best {banana['kdmlp_best']:.3f} vs KFDA {banana['kfda']:.3f}; "
        f"ringnorm: selected kernel {ringnorm['rep_kernel']}, ratio {ringnorm['ratio']:.2f}, "
        f"KDMLP best {ringnorm['kdmlp_best']:.3f} vs KFDA {ringnorm['kfda']:.3f}; "
        f"diabetes: selected kernel {diabetes['rep_kernel']}, ratio {diabetes['ratio']:.2f}, "
        f"KDMLP best {diabetes['kdmlp_best']:.3f} vs KFDA {diabetes['kfda']:.3f}."
    )


def main() -> None:
    ns = load_openml_namespace(OPENML_NOTEBOOK)
    df_compact, df_my_best = load_accuracy_tables()

    diag_rows = []
    summary_rows = []

    for label, dataset in TARGETS:
        rep = representative_kernel_for_dataset(dataset, df_compact, df_my_best)
        diag_df, _, _ = compute_feature_diagnostics(dataset, ns)
        diag_df.to_csv(OUT / f"{dataset}_feature_covariance_diagnostics.csv", index=False)
        curve_df = df_compact[df_compact["dataset"] == dataset].copy().sort_values("r").reset_index(drop=True)
        plot_dataset_case(
            label=label,
            dataset=dataset,
            diag_df=diag_df,
            curve_df=curve_df,
            rep_kernel=rep["rep_kernel"],
            rep_family=rep["family"],
            out_path=OUT / f"{dataset}_feature_covariance_case.png",
        )

        rep_diag = diag_df[diag_df["kernel"] == rep["rep_kernel"]].iloc[0]
        row = {
            "label": label,
            "dataset": dataset,
            "rep_family": rep["family"],
            "rep_kernel": rep["rep_kernel"],
            "best_r": rep["best_r"],
            "ref_D_sigma_feature": float(rep_diag["ref_D_sigma_feature"]),
            "cov_gap_error_feature": float(rep_diag["cov_gap_error_feature"]),
            "ratio": float(rep_diag["ratio"]),
            "kdmlp_best": float(curve_df["my_best_acc"].max()),
            "supcon_best": float(curve_df["supcon_acc"].max()),
            "kfda": float(curve_df["kfda_acc"].iloc[0]),
        }
        row["interpretation"] = build_interpretation(pd.Series(row))
        summary_rows.append(row)
        diag_rows.append(diag_df.assign(label=label))

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(OUT / "openml_feature_covariance_summary.csv", index=False)
    pd.concat(diag_rows, ignore_index=True).to_csv(OUT / "openml_feature_covariance_all_kernels.csv", index=False)

    print("OpenML feature-space covariance summary:")
    print(df_summary[["label", "dataset", "rep_family", "rep_kernel", "best_r", "ref_D_sigma_feature", "cov_gap_error_feature", "ratio", "kdmlp_best", "supcon_best", "kfda"]].to_string(index=False))
    print()
    for _, row in df_summary.iterrows():
        print(f"- {row['interpretation']}")
    print()
    print("Output files:")
    print(f"- {OUT / 'openml_feature_covariance_summary.csv'}")
    for _, dataset in TARGETS:
        print(f"- {OUT / f'{dataset}_feature_covariance_case.png'}")

    open_png_outputs(OUT)


if __name__ == "__main__":
    main()
