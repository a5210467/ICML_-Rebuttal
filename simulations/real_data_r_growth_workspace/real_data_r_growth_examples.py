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
OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

OPENML_OUT = ROOT / "supcon_openml_workspace" / "outputs_same_r"
CIFAR_SAME_R_OUT = ROOT / "supcon_cifar100_workspace" / "outputs_same_r"
CIFAR_MORE_OUT = ROOT / "supcon_cifar100_workspace" / "outputs_more_examples"

OPENML_CASES = [
    {"label": "A", "task": "sonar", "caption": "larger r helps substantially"},
    {"label": "B", "task": "hypothyroid", "caption": "larger r does not help much"},
]

CIFAR_CASES = [
    {"label": "A", "tag": "lion_vs_possum", "caption": "larger r helps substantially"},
    {"label": "B", "tag": "bear_vs_palm_tree", "caption": "larger r does not help much"},
]


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


def load_openml_case(task: str) -> pd.DataFrame:
    df = pd.read_csv(OPENML_OUT / "openml_same_r_compact.csv")
    sub = df[df["dataset"] == task].copy().sort_values("r")
    if sub.empty:
        raise ValueError(f"Could not find OpenML task {task!r}.")
    return sub.reset_index(drop=True)


def load_cifar_case(tag: str) -> pd.DataFrame:
    for folder in (CIFAR_SAME_R_OUT, CIFAR_MORE_OUT):
        compact_path = folder / f"{tag}_same_r_compact.csv"
        kfda_path = folder / f"{tag}_kfda_rows.csv"
        if compact_path.exists() and kfda_path.exists():
            sub = pd.read_csv(compact_path).copy().sort_values("r")
            kfda_rows = pd.read_csv(kfda_path)
            kfda_acc = float(kfda_rows["acc"].mean())
            if "kfda_acc" not in sub.columns:
                sub["kfda_acc"] = kfda_acc
            else:
                sub["kfda_acc"] = float(kfda_acc)
            return sub.reset_index(drop=True)
    raise ValueError(f"Could not find CIFAR case {tag!r}.")


def summarize_case(df: pd.DataFrame, *, domain: str, label: str, name: str, caption: str) -> dict:
    r1 = df[df["r"] == df["r"].min()].iloc[0]
    best = df.sort_values("my_best_acc", ascending=False).iloc[0]
    sup_best = df.sort_values("supcon_acc", ascending=False).iloc[0]
    return {
        "domain": domain,
        "label": label,
        "task": name,
        "caption": caption,
        "r1": int(r1["r"]),
        "kdmlp_r1": float(r1["my_best_acc"]),
        "best_r": int(best["r"]),
        "kdmlp_best": float(best["my_best_acc"]),
        "kdmlp_gain": float(best["my_best_acc"] - r1["my_best_acc"]),
        "supcon_r1": float(r1["supcon_acc"]),
        "supcon_best": float(sup_best["supcon_acc"]),
        "kfda": float(df["kfda_acc"].iloc[0]),
    }


def plot_case_grid(cases: list[tuple[dict, pd.DataFrame]], *, domain_title: str, out_name: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    colors = {"KDMLP best": "#43aa8b", "SupCon": "#f94144", "KFDA": "#577590"}

    for ax, (meta, df) in zip(axes, cases):
        ax.plot(df["r"], df["my_best_acc"], marker="o", linewidth=2.8, color=colors["KDMLP best"], label="KDMLP best")
        ax.plot(df["r"], df["supcon_acc"], marker="o", linewidth=2.4, color=colors["SupCon"], label="SupCon")
        ax.plot(df["r"], df["kfda_acc"], marker="o", linewidth=2.2, linestyle="--", color=colors["KFDA"], label="KFDA")
        best = df.sort_values("my_best_acc", ascending=False).iloc[0]
        start = df.iloc[0]
        gain = float(best["my_best_acc"] - start["my_best_acc"])
        ax.set_title(f"{meta['label']}: {meta['task']}\n{meta['caption']}")
        ax.set_xlabel("target dimension r")
        ax.set_ylabel("test accuracy")
        ax.text(
            0.02,
            0.05,
            f"KDMLP gain from r={int(start['r'])} to best r={int(best['r'])}: {gain:+.3f}",
            transform=ax.transAxes,
            fontsize=10,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "#cccccc"},
        )
        ax.legend(loc="best", fontsize=10)

    fig.suptitle(domain_title, y=1.03, fontsize=20)
    plt.tight_layout()
    fig.savefig(OUT / out_name, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_summary(df_summary: pd.DataFrame) -> str:
    openml_help = df_summary[(df_summary["domain"] == "OpenML") & (df_summary["label"] == "A")].iloc[0]
    openml_flat = df_summary[(df_summary["domain"] == "OpenML") & (df_summary["label"] == "B")].iloc[0]
    cifar_help = df_summary[(df_summary["domain"] == "CIFAR100") & (df_summary["label"] == "A")].iloc[0]
    cifar_flat = df_summary[(df_summary["domain"] == "CIFAR100") & (df_summary["label"] == "B")].iloc[0]
    return (
        f"On OpenML, {openml_help['task']} is a clear case where larger r helps: KDMLP best rises from "
        f"{openml_help['kdmlp_r1']:.3f} at r={int(openml_help['r1'])} to {openml_help['kdmlp_best']:.3f} at r={int(openml_help['best_r'])}, "
        f"while {openml_flat['task']} stays flat at {openml_flat['kdmlp_best']:.3f}. "
        f"On CIFAR-100, {cifar_help['task']} shows the same pattern ({cifar_help['kdmlp_r1']:.3f} to {cifar_help['kdmlp_best']:.3f}), "
        f"whereas {cifar_flat['task']} remains essentially unchanged. "
        f"KFDA is the flat binary reference in all four panels, and SupCon is shown on the same r grid for the same datasets."
    )


def main() -> None:
    openml_loaded = []
    for case in OPENML_CASES:
        df = load_openml_case(case["task"])
        openml_loaded.append((case, df))

    cifar_loaded = []
    for case in CIFAR_CASES:
        df = load_cifar_case(case["tag"])
        task_name = str(df["task"].iloc[0])
        case = {**case, "task": task_name}
        cifar_loaded.append((case, df))

    rows = []
    for meta, df in openml_loaded:
        rows.append(summarize_case(df, domain="OpenML", label=meta["label"], name=meta["task"], caption=meta["caption"]))
    for meta, df in cifar_loaded:
        rows.append(summarize_case(df, domain="CIFAR100", label=meta["label"], name=meta["task"], caption=meta["caption"]))

    df_summary = pd.DataFrame(rows)
    df_summary.to_csv(OUT / "selected_cases_summary.csv", index=False)

    openml_tidy = []
    for meta, df in openml_loaded:
        tmp = df[["r", "my_best_acc", "supcon_acc", "kfda_acc"]].copy()
        tmp["label"] = meta["label"]
        tmp["task"] = meta["task"]
        openml_tidy.append(tmp)
    pd.concat(openml_tidy, ignore_index=True).to_csv(OUT / "openml_selected_curves.csv", index=False)

    cifar_tidy = []
    for meta, df in cifar_loaded:
        tmp = df[["r", "my_best_acc", "supcon_acc", "kfda_acc"]].copy()
        tmp["label"] = meta["label"]
        tmp["task"] = meta["task"]
        cifar_tidy.append(tmp)
    pd.concat(cifar_tidy, ignore_index=True).to_csv(OUT / "cifar100_selected_curves.csv", index=False)

    plot_case_grid(openml_loaded, domain_title="OpenML: one case where larger r helps and one where it does not", out_name="openml_r_growth_examples.png")
    plot_case_grid(cifar_loaded, domain_title="CIFAR-100: one case where larger r helps and one where it does not", out_name="cifar100_r_growth_examples.png")

    print(df_summary.to_string(index=False))
    print("\nInterpretation:")
    print(build_summary(df_summary))
    print(f"\nSaved outputs to {OUT}")
    open_png_outputs(OUT)


if __name__ == "__main__":
    main()
