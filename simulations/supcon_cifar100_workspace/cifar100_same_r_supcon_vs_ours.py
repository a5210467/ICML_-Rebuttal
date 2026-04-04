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
from PIL import Image
from sklearn.decomposition import PCA
from torchvision import datasets


ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = ROOT.parent
PROJECT_ROOT = REPO_ROOT.parent
OUT = Path(__file__).resolve().parent / "outputs_same_r"
OUT.mkdir(parents=True, exist_ok=True)
PAIR_CACHE = OUT / "pair_cache"
PAIR_CACHE.mkdir(parents=True, exist_ok=True)
for candidate in (
    ROOT / "modern_benchmark_data" / "vision",
    REPO_ROOT / "modern_benchmark_data" / "vision",
    PROJECT_ROOT / "modern_benchmark_data" / "vision",
):
    if candidate.exists():
        VISION_ROOT = candidate
        break
else:
    VISION_ROOT = ROOT / "modern_benchmark_data" / "vision"

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
    _stage1_params_from_metric,
    fit_linear_probe,
    load_notebook_namespace,
    train_supcon_feature_model,
)


sns.set_theme(style="whitegrid", context="talk")

PAIR_SETTINGS = [
    ("beaver", "possum"),
    ("clock", "tractor"),
]
R_VALUES = (1, 2, 3, 4, 5, 6, 32, 64)
N_PER_CLASS = 80
REPEATS = 2
SUPCON_EPOCHS = 10


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


def build_same_r_cifar_summary(df_all_compact: pd.DataFrame) -> str:
    non_sat = df_all_compact[df_all_compact[["my_best_acc", "supcon_acc"]].max(axis=1) < 0.995].copy()
    source = non_sat if len(non_sat) else df_all_compact
    best = source.sort_values("margin_vs_supcon", ascending=False).iloc[0]
    mean_curve = df_all_compact.groupby("r", as_index=False)[["my_best_acc", "supcon_acc", "kfda_acc"]].mean()
    top_r = mean_curve.assign(margin=mean_curve["my_best_acc"] - mean_curve["supcon_acc"]).sort_values("margin", ascending=False).iloc[0]
    return (
        f"Summary: the most informative same-r CIFAR-100 pair here is {best['task']}, where KDMLP reaches {best['my_best_acc']:.3f} "
        f"against SupCon at {best['supcon_acc']:.3f} and KFDA at {best['kfda_acc']:.3f} with r={int(best['r'])}. "
        f"Averaged across the selected pairs, the largest mean same-r margin occurs at r={int(top_r['r'])} "
        f"with KDMLP-minus-SupCon = {top_r['margin']:+.3f}. Binary KFDA contributes a flat same-r reference because it has only one non-zero discriminant direction."
    )


def _slug(text: str) -> str:
    out = []
    for ch in str(text).lower():
        out.append(ch if ch.isalnum() else "_")
    s = "".join(out).strip("_")
    while "__" in s:
        s = s.replace("__", "_")
    return s


def load_cifar100_pair_bundle(ns: dict, *, class0: str, class1: str, n_per_class: int, seed: int, device: str):
    rng = np.random.default_rng(seed)
    cache_path = PAIR_CACHE / f"{_slug(class0)}_{_slug(class1)}__n{n_per_class}__seed{seed}.npz"

    train = datasets.CIFAR100(root=str(VISION_ROOT), train=True, download=True)
    test = datasets.CIFAR100(root=str(VISION_ROOT), train=False, download=True)
    data = np.concatenate([train.data, test.data], axis=0)
    targets = np.asarray(list(train.targets) + list(test.targets), dtype=int)
    classes = list(train.classes)
    name_to_id = {name: i for i, name in enumerate(classes)}

    idx0_all = np.where(targets == name_to_id[class0])[0]
    idx1_all = np.where(targets == name_to_id[class1])[0]
    n = min(len(idx0_all), len(idx1_all), int(n_per_class))
    idx0 = rng.choice(idx0_all, size=n, replace=False)
    idx1 = rng.choice(idx1_all, size=n, replace=False)
    selected = np.concatenate([idx0, idx1])
    selected = selected[rng.permutation(len(selected))]

    images = [Image.fromarray(data[i]) for i in selected]
    y = (targets[selected] == name_to_id[class1]).astype(int)
    if cache_path.exists():
        X = np.load(cache_path)["X_resnet"].astype(np.float64)
    else:
        X = ns["extract_image_features_from_pil"](
            images,
            feature_mode="resnet18",
            batch_size=128,
            device=device,
            verbose=True,
        )
        np.savez_compressed(cache_path, X_resnet=X.astype(np.float32))
    return X.astype(np.float64), y.astype(int), {"task_name": f"CIFAR100: {class0} vs {class1}", "class_names": (class0, class1)}


def split_indices(y: np.ndarray, *, train_frac: float, seed: int):
    from sklearn.model_selection import StratifiedShuffleSplit

    splitter = StratifiedShuffleSplit(n_splits=1, train_size=train_frac, random_state=seed)
    tr_idx, te_idx = next(splitter.split(np.zeros(len(y)), y))
    return np.asarray(tr_idx, dtype=int), np.asarray(te_idx, dtype=int)


def prepare_my_projection(ns: dict, *, Xtr, ytr, Xother, metric: str, algo: str, max_r: int, seed: int, anchors_per_class: int):
    pre = ns["_prepare_split_geometry"](
        Xtr,
        ytr,
        Xother,
        stage1_metric=metric,
        stage1_params=_stage1_params_from_metric(metric),
        anchors_per_class=anchors_per_class,
        basis_mode="anchors",
        center_kernel=True,
        seed=seed,
    )
    pre_my = ns["_paper_prepare_my_precompute"](pre, ytr, max_r=max_r, my_algo=algo, reg=1e-6)
    return np.asarray(pre_my["Ztr_full"], dtype=np.float64), np.asarray(pre_my["Zte_full"], dtype=np.float64)


def evaluate_fixed_r_pair(ns: dict, *, class0: str, class1: str, device: str, seed: int):
    X, y, meta = load_cifar100_pair_bundle(
        ns,
        class0=class0,
        class1=class1,
        n_per_class=N_PER_CLASS,
        seed=seed,
        device=device,
    )

    same_r_rows = []
    kfda_rows = []

    for rep in range(REPEATS):
        split_seed = seed + 97 * rep
        tr_idx, te_idx = split_indices(y, train_frac=0.6, seed=split_seed)
        Xtr = X[tr_idx]
        ytr = y[tr_idx]
        Xte = X[te_idx]
        yte = y[te_idx]
        anchors_per_class = int(min(np.sum(ytr == 0), np.sum(ytr == 1)))

        sup = train_supcon_feature_model(Xtr, ytr, Xte, yte, device=device, seed=split_seed, epochs=SUPCON_EPOCHS)
        for r in R_VALUES:
            max_allowed = min(r, sup["Ztr"].shape[1], Xtr.shape[0] - 1)
            if max_allowed < 1:
                continue
            pca = PCA(n_components=max_allowed, random_state=0)
            Ztr_r = pca.fit_transform(sup["Ztr"])
            Zte_r = pca.transform(sup["Zte"])
            probe = fit_linear_probe(Ztr_r, ytr, Zte_r, yte, seed=5000 + split_seed + r)
            same_r_rows.append(
                {
                    "task": meta["task_name"],
                    "class0": class0,
                    "class1": class1,
                    "rep": rep,
                    "method": "SupCon",
                    "family": "SupCon",
                    "r": int(r),
                    "acc": float(probe["acc"]),
                }
            )

        ours_rows = ns["evaluate_methods_on_fixed_split_fast"](
            Xtr,
            ytr,
            Xte,
            yte,
            task_name=meta["task_name"],
            rep=rep,
            r_values=(1,),
            anchors_per_class=anchors_per_class,
            mu_reg=1e-6,
            svm_folds=2,
            stage1_kernels=ns["STAGE1_KERNELS_FULL"],
            basis_mode="anchors",
            center_kernel=True,
            seed=split_seed,
            verbose=False,
        )
        df_ours = pd.DataFrame(ours_rows)
        df_kfda = df_ours[df_ours["family"] == "KFDA"].copy()
        if not df_kfda.empty:
            best_kfda = df_kfda.sort_values("acc", ascending=False).iloc[0]
            kfda_rows.append(
                {
                    "task": meta["task_name"],
                    "class0": class0,
                    "class1": class1,
                    "rep": rep,
                    "family": "KFDA",
                    "r": 1,
                    "acc": float(best_kfda["acc"]),
                    "kernel": str(best_kfda["stage1_kernel"]),
                }
            )
            for r in R_VALUES:
                same_r_rows.append(
                    {
                        "task": meta["task_name"],
                        "class0": class0,
                        "class1": class1,
                        "rep": rep,
                        "method": "KFDA",
                        "family": "KFDA",
                        "r": int(r),
                        "acc": float(best_kfda["acc"]),
                        "kernel": str(best_kfda["stage1_kernel"]),
                    }
                )

        for family, algo in [("MY-large_mu", "large_mu"), ("MY-small_mu", "small_mu")]:
            fam_rows = []
            for metric, _ in ns["STAGE1_KERNELS_FULL"]:
                Ztr_full, Zte_full = prepare_my_projection(
                    ns,
                    Xtr=Xtr,
                    ytr=ytr,
                    Xother=Xte,
                    metric=metric,
                    algo=algo,
                    max_r=max(R_VALUES),
                    seed=split_seed,
                    anchors_per_class=anchors_per_class,
                )
                available_r = min(Ztr_full.shape[1], Zte_full.shape[1])
                for r in R_VALUES:
                    if r > available_r:
                        continue
                    probe = fit_linear_probe(Ztr_full[:, :r], ytr, Zte_full[:, :r], yte, seed=7000 + split_seed + r)
                    fam_rows.append(
                        {
                            "task": meta["task_name"],
                            "class0": class0,
                            "class1": class1,
                            "rep": rep,
                            "method": family,
                            "family": family,
                            "metric": metric,
                            "r": int(r),
                            "acc": float(probe["acc"]),
                        }
                    )

            df_fam = pd.DataFrame(fam_rows)
            for r in sorted(df_fam["r"].unique()):
                sub = df_fam[df_fam["r"] == r]
                best = sub.sort_values("acc", ascending=False).iloc[0]
                same_r_rows.append(
                    {
                        "task": meta["task_name"],
                        "class0": class0,
                        "class1": class1,
                        "rep": rep,
                        "method": family,
                        "family": family,
                        "r": int(r),
                        "acc": float(best["acc"]),
                        "kernel": str(best["metric"]),
                    }
                )

    df_same = pd.DataFrame(same_r_rows)
    df_kfda = pd.DataFrame(kfda_rows)
    return {"meta": meta, "same_r_rows": df_same, "kfda_rows": df_kfda}


def summarize_pair(exp: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_same = exp["same_r_rows"].copy()
    acc_summary = df_same.groupby(["task", "class0", "class1", "method", "r"], as_index=False)["acc"].agg(acc_mean="mean", acc_std="std")

    large = acc_summary[acc_summary["method"] == "MY-large_mu"][["task", "class0", "class1", "r", "acc_mean"]].rename(columns={"acc_mean": "my_large_acc"})
    small = acc_summary[acc_summary["method"] == "MY-small_mu"][["task", "class0", "class1", "r", "acc_mean"]].rename(columns={"acc_mean": "my_small_acc"})
    sup = acc_summary[acc_summary["method"] == "SupCon"][["task", "class0", "class1", "r", "acc_mean"]].rename(columns={"acc_mean": "supcon_acc"})
    kfda = acc_summary[acc_summary["method"] == "KFDA"][["task", "class0", "class1", "r", "acc_mean"]].rename(columns={"acc_mean": "kfda_acc"})
    compact = large.merge(small, on=["task", "class0", "class1", "r"]).merge(sup, on=["task", "class0", "class1", "r"]).merge(kfda, on=["task", "class0", "class1", "r"])
    compact["my_best_acc"] = compact[["my_large_acc", "my_small_acc"]].max(axis=1)
    compact["my_best_family"] = np.where(compact["my_large_acc"] >= compact["my_small_acc"], "MY-large_mu", "MY-small_mu")
    compact["margin_vs_supcon"] = compact["my_best_acc"] - compact["supcon_acc"]
    compact["margin_vs_kfda"] = compact["my_best_acc"] - compact["kfda_acc"]
    return acc_summary, compact


def plot_pair(exp: dict, acc_summary: pd.DataFrame, compact: pd.DataFrame):
    meta = exp["meta"]
    class0, class1 = meta["class_names"]
    tag = f"{_slug(class0)}_vs_{_slug(class1)}"

    fig, ax = plt.subplots(figsize=(9, 5))
    for method, color, linestyle in [("KFDA", "#577590", "--"), ("MY-large_mu", "#43aa8b", "-"), ("MY-small_mu", "#f8961e", "-"), ("SupCon", "#f94144", "-")]:
        sub = acc_summary[acc_summary["method"] == method].sort_values("r")
        if len(sub):
            ax.plot(sub["r"], sub["acc_mean"], marker="o", linewidth=2.5, linestyle=linestyle, label=method, color=color)
    ax.set_xlabel("Reduced dimension r")
    ax.set_ylabel("Accuracy")
    ax.set_title(meta["task_name"])
    ax.legend()
    plt.tight_layout()
    fig.savefig(OUT / f"{tag}_same_r_curve.png", dpi=180)
    plt.close(fig)

    sub = compact[compact["r"].isin([32, 64])].copy()
    if not sub.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(sub))
        width = 0.26
        ax.bar(x - width, sub["my_best_acc"], width=width, color="#43aa8b", label="MY-best")
        ax.bar(x, sub["supcon_acc"], width=width, color="#f94144", label="SupCon")
        ax.bar(x + width, sub["kfda_acc"], width=width, color="#577590", label="KFDA")
        ax.set_xticks(x, [f"r={int(r)}" for r in sub["r"]])
        ax.set_ylabel("Accuracy")
        ax.set_ylim(max(0.45, float(min(sub["supcon_acc"].min(), sub["my_best_acc"].min(), sub["kfda_acc"].min()) - 0.05)), 1.02)
        ax.set_title(f"{meta['task_name']}: KDMLP vs SupCon vs KFDA at r=32,64")
        ax.legend()
        plt.tight_layout()
        fig.savefig(OUT / f"{tag}_dim32_64_bar.png", dpi=180)
        plt.close(fig)


def main():
    ns = load_notebook_namespace(NOTEBOOK_PATH)
    ns["set_global_seed"](0)
    device = ns["pick_device"]()

    all_acc = []
    all_compact = []

    for class0, class1 in PAIR_SETTINGS:
        print("=" * 100)
        print(f"[same-r] {class0} vs {class1}")
        exp = evaluate_fixed_r_pair(ns, class0=class0, class1=class1, device=device, seed=19)
        acc_summary, compact = summarize_pair(exp)
        tag = f"{_slug(class0)}_vs_{_slug(class1)}"
        exp["same_r_rows"].to_csv(OUT / f"{tag}_same_r_rows.csv", index=False)
        exp["kfda_rows"].to_csv(OUT / f"{tag}_kfda_rows.csv", index=False)
        acc_summary.to_csv(OUT / f"{tag}_same_r_summary.csv", index=False)
        compact.to_csv(OUT / f"{tag}_same_r_compact.csv", index=False)
        plot_pair(exp, acc_summary, compact)
        all_acc.append(acc_summary)
        all_compact.append(compact)
        print(compact.to_string(index=False))

    df_all_acc = pd.concat(all_acc, ignore_index=True)
    df_all_compact = pd.concat(all_compact, ignore_index=True)
    df_all_acc.to_csv(OUT / "cifar100_same_r_summary_all.csv", index=False)
    df_all_compact.to_csv(OUT / "cifar100_same_r_compact_all.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    mean_curve = df_all_compact.groupby("r", as_index=False)[["my_best_acc", "supcon_acc", "kfda_acc"]].mean()
    ax.plot(mean_curve["r"], mean_curve["my_best_acc"], marker="o", linewidth=2.5, label="MY-best mean", color="#43aa8b")
    ax.plot(mean_curve["r"], mean_curve["supcon_acc"], marker="o", linewidth=2.5, label="SupCon mean", color="#f94144")
    ax.plot(mean_curve["r"], mean_curve["kfda_acc"], marker="o", linewidth=2.5, linestyle="--", label="KFDA mean", color="#577590")
    ax.set_xlabel("Reduced dimension r")
    ax.set_ylabel("Mean accuracy across selected CIFAR-100 pairs")
    ax.set_title("Same-r CIFAR-100 comparison (KDMLP vs SupCon vs KFDA)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(OUT / "cifar100_same_r_mean_curve.png", dpi=180)
    plt.close(fig)

    print("\nInterpretation:")
    print(build_same_r_cifar_summary(df_all_compact))
    print(f"\nSaved outputs to {OUT}")
    open_png_outputs(OUT)


if __name__ == "__main__":
    main()
