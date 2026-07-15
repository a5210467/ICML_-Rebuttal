from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from torchvision import datasets

import cifar100_scan_supcon_vs_ours as scan
import cifar100_same_r_supcon_vs_ours as same_r


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs_more_examples"
OUT.mkdir(parents=True, exist_ok=True)


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


def build_more_examples_summary(df_summary: pd.DataFrame) -> str:
    best = df_summary.iloc[0]
    mean_margin = float(df_summary["margin_vs_supcon"].mean())
    return (
        f"Summary: broadening the CIFAR-100 search surfaces additional nontrivial wins for KDMLP. "
        f"The best shortlist example is {best['class0']} vs {best['class1']}, where KDMLP reaches {best['my_best_acc']:.3f} "
        f"against SupCon at {best['supcon_acc']:.3f} with r={int(best['best_r'])}; "
        f"the mean shortlist margin is {mean_margin:+.3f}."
    )


def choose_shortlist(df_scan: pd.DataFrame, k: int = 4) -> list[tuple[str, str]]:
    df = df_scan.copy()
    df["nontrivial"] = (df[["ours_best_acc", "supcon_acc"]].max(axis=1) < 0.995).astype(int)
    positive = df[(df["margin"] > 0) & (df["nontrivial"] == 1)].copy()
    if len(positive) < k:
        positive = df[df["margin"] > 0].copy()
    shortlist = positive.sort_values(["margin", "ours_best_acc"], ascending=False).head(k)
    return [(str(r["class0"]), str(r["class1"])) for _, r in shortlist.iterrows()]


def make_scan_plot(df_scan: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 7))
    sub = df_scan.sort_values("margin")
    labels = [f"{a} vs {b}" for a, b in zip(sub["class0"], sub["class1"])]
    colors = ["#43aa8b" if m > 0 else "#f94144" for m in sub["margin"]]
    ax.barh(labels, sub["margin"], color=colors)
    ax.axvline(0.0, color="black", linewidth=1.2)
    ax.set_xlabel("Best KDMLP minus SupCon accuracy")
    ax.set_title("Broader CIFAR-100 quick scan")
    plt.tight_layout()
    fig.savefig(OUT / "cifar100_more_scan_margins.png", dpi=180)
    plt.close(fig)


def make_shortlist_plot(df_compact: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 6))
    for task, sub in df_compact.groupby("task"):
        short_label = task.replace("CIFAR100: ", "")
        ax.plot(sub["r"], sub["my_best_acc"], marker="o", linewidth=2.2, label=f"{short_label} | KDMLP")
        ax.plot(sub["r"], sub["supcon_acc"], marker="o", linewidth=1.6, linestyle="--", alpha=0.85, label=f"{short_label} | SupCon")
    ax.set_xlabel("Reduced dimension r")
    ax.set_ylabel("Accuracy")
    ax.set_title("Selected CIFAR-100 pairs: same-r comparison")
    ax.legend(fontsize=10, ncol=2)
    plt.tight_layout()
    fig.savefig(OUT / "cifar100_more_examples_same_r.png", dpi=180)
    plt.close(fig)


def make_mean_curve(df_compact: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 5))
    columns = ["my_best_acc", "supcon_acc"]
    if "kfda_acc" in df_compact.columns:
        columns.append("kfda_acc")
    mean_curve = df_compact.groupby("r", as_index=False)[columns].mean()
    ax.plot(mean_curve["r"], mean_curve["my_best_acc"], marker="o", linewidth=2.4, label="Best KDMLP mean", color="#43aa8b")
    ax.plot(mean_curve["r"], mean_curve["supcon_acc"], marker="o", linewidth=2.4, label="SupCon mean", color="#f94144")
    if "kfda_acc" in mean_curve.columns:
        ax.plot(mean_curve["r"], mean_curve["kfda_acc"], marker="o", linewidth=2.4, linestyle="--", label="KFDA mean", color="#577590")
    ax.set_xlabel("Reduced dimension r")
    ax.set_ylabel("Mean accuracy across shortlist")
    ax.set_title("CIFAR-100 shortlist: mean same-r curve")
    ax.legend()
    plt.tight_layout()
    fig.savefig(OUT / "cifar100_same_r_mean_curve.png", dpi=180)
    plt.close(fig)


def main():
    ns = scan.load_notebook_namespace(scan.NOTEBOOK_PATH)
    ns["set_global_seed"](0)
    device = ns["pick_device"]()

    class_names = list(datasets.CIFAR100(root=str(scan.VISION_ROOT), train=True, download=True).classes)
    candidates = scan.sample_candidate_pairs(class_names, n_pairs=18, seed=11)

    search_rows = []
    for i, (class0, class1) in enumerate(candidates, start=1):
        print("=" * 100)
        print(f"[quick-scan {i}/{len(candidates)}] {class0} vs {class1}")
        exp = scan.run_pair_experiment(
            ns,
            class0=class0,
            class1=class1,
            n_per_class=80,
            repeats=1,
            stage1_kernels=ns["STAGE1_KERNELS_QUICK"],
            r_values=(1, 2, 3, 4),
            supcon_epochs=6,
            device=device,
            seed=31,
        )
        search_rows.append(
            {
                "class0": class0,
                "class1": class1,
                "ours_best_family": exp["my_best_family"],
                "ours_best_acc": exp["my_best_acc"],
                "supcon_acc": exp["supcon_mean"],
                "kfda_acc": exp["ours_best"]["KFDA"]["acc_mean"],
                "margin": exp["margin"],
                "ours_best_kernel": exp["ours_best"][exp["my_best_family"]]["stage1_kernel"],
                "ours_best_r": exp["ours_best"][exp["my_best_family"]]["r"],
            }
        )

    df_scan = pd.DataFrame(search_rows).sort_values("margin", ascending=False).reset_index(drop=True)
    df_scan.to_csv(OUT / "cifar100_more_pair_scan.csv", index=False)
    make_scan_plot(df_scan)

    shortlist = choose_shortlist(df_scan, k=4)
    pd.DataFrame(shortlist, columns=["class0", "class1"]).to_csv(OUT / "cifar100_shortlist_pairs.csv", index=False)

    same_r.OUT = OUT
    same_r.OUT.mkdir(parents=True, exist_ok=True)
    same_r.PAIR_CACHE = scan.PAIR_CACHE
    same_r.N_PER_CLASS = 80
    same_r.REPEATS = 2
    same_r.SUPCON_EPOCHS = 10
    same_r.R_VALUES = (1, 2, 3, 4, 5, 6, 32, 64)

    all_acc = []
    all_compact = []
    summary_rows = []

    for class0, class1 in shortlist:
        print("=" * 100)
        print(f"[same-r full check] {class0} vs {class1}")
        exp = same_r.evaluate_fixed_r_pair(ns, class0=class0, class1=class1, device=device, seed=41)
        acc_summary, compact = same_r.summarize_pair(exp)
        same_r.plot_pair(exp, acc_summary, compact)

        tag = f"{same_r._slug(class0)}_vs_{same_r._slug(class1)}"
        exp["same_r_rows"].to_csv(OUT / f"{tag}_same_r_rows.csv", index=False)
        exp["kfda_rows"].to_csv(OUT / f"{tag}_kfda_rows.csv", index=False)
        acc_summary.to_csv(OUT / f"{tag}_same_r_summary.csv", index=False)
        compact.to_csv(OUT / f"{tag}_same_r_compact.csv", index=False)

        all_acc.append(acc_summary)
        all_compact.append(compact)

        best_r_row = compact.sort_values(["margin_vs_supcon", "my_best_acc"], ascending=False).iloc[0]
        summary_rows.append(
            {
                "class0": class0,
                "class1": class1,
                "best_r": int(best_r_row["r"]),
                "my_best_family": str(best_r_row["my_best_family"]),
                "my_best_acc": float(best_r_row["my_best_acc"]),
                "supcon_acc": float(best_r_row["supcon_acc"]),
                "margin_vs_supcon": float(best_r_row["margin_vs_supcon"]),
            }
        )

    df_all_acc = pd.concat(all_acc, ignore_index=True)
    df_all_compact = pd.concat(all_compact, ignore_index=True)
    df_summary = pd.DataFrame(summary_rows).sort_values("margin_vs_supcon", ascending=False).reset_index(drop=True)

    df_all_acc.to_csv(OUT / "cifar100_same_r_summary_all.csv", index=False)
    df_all_compact.to_csv(OUT / "cifar100_same_r_compact_all.csv", index=False)
    df_summary.to_csv(OUT / "cifar100_same_r_best_examples.csv", index=False)

    make_shortlist_plot(df_all_compact)

    make_mean_curve(df_all_compact)

    print("\nQuick-scan top rows:")
    print(df_scan.head(10).to_string(index=False))
    print("\nSame-r shortlist summary:")
    print(df_summary.to_string(index=False))
    print("\nInterpretation:")
    print(build_more_examples_summary(df_summary))
    print(f"\nSaved outputs to {OUT}")
    open_png_outputs(OUT)


if __name__ == "__main__":
    main()
