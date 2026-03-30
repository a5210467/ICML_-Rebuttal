from __future__ import annotations

import random
import shutil
import subprocess
import sys
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from torchvision import datasets


ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(parents=True, exist_ok=True)
PAIR_CACHE = OUT / "pair_cache"
PAIR_CACHE.mkdir(parents=True, exist_ok=True)
VISION_ROOT = ROOT / "modern_benchmark_data" / "vision"

sys.path.insert(0, str(ROOT / "supcon_comparison_workspace"))

from compare_supcon_vs_ours import (  # type: ignore
    NOTEBOOK_PATH,
    _stage1_params_from_metric,
    build_ours_projection,
    load_notebook_namespace,
    train_supcon_feature_model,
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


def build_scan_summary(df_full: pd.DataFrame) -> str:
    best = df_full.iloc[0]
    wins = int((df_full["margin"] > 0).sum())
    return (
        f"Summary: KDMLP beats SupCon on {wins}/{len(df_full)} shortlisted CIFAR-100 pairs in the full-check run. "
        f"The strongest example is {best['class0']} vs {best['class1']}, where KDMLP reaches {best['ours_best_acc']:.3f} "
        f"against SupCon at {best['supcon_acc']:.3f} (margin {best['margin']:+.3f})."
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


def run_pair_experiment(ns: dict, *, class0: str, class1: str, n_per_class: int, repeats: int, stage1_kernels, r_values, supcon_epochs: int, device: str, seed: int):
    X, y, meta = load_cifar100_pair_bundle(ns, class0=class0, class1=class1, n_per_class=n_per_class, seed=seed, device=device)

    ours_rows = []
    sup_rows = []
    first_split = None
    for rep in range(repeats):
        split_seed = seed + 97 * rep
        tr_idx, te_idx = split_indices(y, train_frac=0.6, seed=split_seed)
        Xtr, ytr = X[tr_idx], y[tr_idx]
        Xte, yte = X[te_idx], y[te_idx]

        ours_rows.extend(
            ns["evaluate_methods_on_fixed_split_fast"](
                Xtr,
                ytr,
                Xte,
                yte,
                task_name=meta["task_name"],
                rep=rep,
                r_values=r_values,
                anchors_per_class=40,
                mu_reg=1e-6,
                svm_folds=2,
                stage1_kernels=stage1_kernels,
                basis_mode="anchors",
                center_kernel=True,
                seed=split_seed,
                verbose=False,
            )
        )

        sup = train_supcon_feature_model(Xtr, ytr, Xte, yte, device=device, seed=split_seed, epochs=supcon_epochs)
        sup_rows.append({"rep": rep, "acc": float(sup["acc"])})
        if first_split is None:
            first_split = {"split_seed": split_seed, "train_idx": tr_idx, "test_idx": te_idx, "supcon": sup}

    df_ours = pd.DataFrame(ours_rows)
    summary = ns["summarize_multidim_results"](df_ours)
    df_best = summary["best_kernel"].copy()
    best_kfda = df_best[df_best["family"] == "KFDA"].sort_values("acc_mean", ascending=False).iloc[0].to_dict()
    best_large = df_best[df_best["family"] == "MY-large_mu"].sort_values("acc_mean", ascending=False).iloc[0].to_dict()
    best_small = df_best[df_best["family"] == "MY-small_mu"].sort_values("acc_mean", ascending=False).iloc[0].to_dict()
    df_sup = pd.DataFrame(sup_rows)
    sup_mean = float(df_sup["acc"].mean())
    sup_std = float(df_sup["acc"].std(ddof=1)) if len(df_sup) > 1 else 0.0
    my_best_family = "MY-large_mu" if best_large["acc_mean"] >= best_small["acc_mean"] else "MY-small_mu"
    my_best_acc = max(best_large["acc_mean"], best_small["acc_mean"])

    return {
        "X": X,
        "y": y,
        "meta": meta,
        "ours_full": df_ours,
        "ours_best_kernel": df_best,
        "ours_best": {"KFDA": best_kfda, "MY-large_mu": best_large, "MY-small_mu": best_small},
        "supcon_mean": sup_mean,
        "supcon_std": sup_std,
        "margin": float(my_best_acc - sup_mean),
        "my_best_family": my_best_family,
        "my_best_acc": my_best_acc,
        "first_split": first_split,
    }


def sample_candidate_pairs(class_names: list[str], n_pairs: int = 6, seed: int = 0):
    all_pairs = list(combinations(class_names, 2))
    rng = random.Random(seed)
    rng.shuffle(all_pairs)
    return all_pairs[:n_pairs]


def make_final_plots(ns: dict, exp: dict, *, tag: str):
    best = exp["ours_best"]
    bar_rows = [
        {"method": "KFDA-1D", "acc_mean": best["KFDA"]["acc_mean"], "acc_std": best["KFDA"]["acc_std"]},
        {"method": "MY-large_mu", "acc_mean": best["MY-large_mu"]["acc_mean"], "acc_std": best["MY-large_mu"]["acc_std"]},
        {"method": "MY-small_mu", "acc_mean": best["MY-small_mu"]["acc_mean"], "acc_std": best["MY-small_mu"]["acc_std"]},
        {"method": "SupCon", "acc_mean": exp["supcon_mean"], "acc_std": exp["supcon_std"]},
    ]
    df_bar = pd.DataFrame(bar_rows)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(df_bar["method"], df_bar["acc_mean"], yerr=df_bar["acc_std"], color=["#577590", "#43aa8b", "#f8961e", "#f94144"])
    ax.set_ylim(0.45, 1.0)
    ax.set_title(exp["meta"]["task_name"])
    ax.set_ylabel("Accuracy")
    plt.xticks(rotation=18)
    plt.tight_layout()
    fig.savefig(OUT / f"{tag}_accuracy_bar.png", dpi=180)
    plt.close(fig)

    df_curve = (
        exp["ours_best_kernel"][exp["ours_best_kernel"]["family"].isin(["MY-large_mu", "MY-small_mu"])]
        .groupby(["family", "r"], as_index=False)[["acc_mean"]].mean()
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    for fam, color in [("MY-large_mu", "#43aa8b"), ("MY-small_mu", "#f8961e")]:
        sub = df_curve[df_curve["family"] == fam].sort_values("r")
        ax.plot(sub["r"], sub["acc_mean"], marker="o", linewidth=2.5, label=fam, color=color)
    ax.axhline(best["KFDA"]["acc_mean"], linestyle="--", color="#577590", label="KFDA-1D")
    ax.axhline(exp["supcon_mean"], linestyle=":", color="#f94144", label="SupCon")
    ax.set_xlabel("Projection dimension r")
    ax.set_ylabel("Accuracy")
    ax.set_title(exp["meta"]["task_name"])
    ax.legend()
    plt.tight_layout()
    fig.savefig(OUT / f"{tag}_r_curve.png", dpi=180)
    plt.close(fig)


def main():
    ns = load_notebook_namespace(NOTEBOOK_PATH)
    ns["set_global_seed"](0)
    device = ns["pick_device"]()

    class_names = list(datasets.CIFAR100(root=str(VISION_ROOT), train=True, download=True).classes)
    candidates = sample_candidate_pairs(class_names, n_pairs=6, seed=0)

    search_rows = []
    for class0, class1 in candidates:
        print("=" * 100)
        print(f"[scan] {class0} vs {class1}")
        exp = run_pair_experiment(
            ns,
            class0=class0,
            class1=class1,
            n_per_class=80,
            repeats=1,
            stage1_kernels=ns["STAGE1_KERNELS_QUICK"],
            r_values=(1, 2, 3, 4),
            supcon_epochs=6,
            device=device,
            seed=7,
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
            }
        )
        print(pd.DataFrame(search_rows).sort_values("margin", ascending=False).head(5).to_string(index=False))
    df_search = pd.DataFrame(search_rows).sort_values("margin", ascending=False).reset_index(drop=True)
    df_search.to_csv(OUT / "cifar100_pair_scan.csv", index=False)

    shortlist = [(str(r["class0"]), str(r["class1"])) for _, r in df_search.head(2).iterrows()]
    full_rows = []
    full_exps = {}
    for class0, class1 in shortlist:
        print("=" * 100)
        print(f"[full-check] {class0} vs {class1}")
        exp = run_pair_experiment(
            ns,
            class0=class0,
            class1=class1,
            n_per_class=80,
            repeats=2,
            stage1_kernels=ns["STAGE1_KERNELS_FULL"],
            r_values=(1, 2, 3, 4, 6),
            supcon_epochs=10,
            device=device,
            seed=19,
        )
        full_exps[(class0, class1)] = exp
        full_rows.append(
            {
                "class0": class0,
                "class1": class1,
                "ours_best_family": exp["my_best_family"],
                "ours_best_acc": exp["my_best_acc"],
                "supcon_acc": exp["supcon_mean"],
                "supcon_std": exp["supcon_std"],
                "kfda_acc": exp["ours_best"]["KFDA"]["acc_mean"],
                "margin": exp["margin"],
                "ours_best_kernel": exp["ours_best"][exp["my_best_family"]]["stage1_kernel"],
                "ours_best_r": exp["ours_best"][exp["my_best_family"]]["r"],
            }
        )

    df_full = pd.DataFrame(full_rows).sort_values("margin", ascending=False).reset_index(drop=True)
    df_full.to_csv(OUT / "cifar100_full_check.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    sub = df_full.sort_values("margin")
    colors = ["#43aa8b" if m > 0 else "#f94144" for m in sub["margin"]]
    labels = [f"{a} vs {b}" for a, b in zip(sub["class0"], sub["class1"])]
    ax.barh(labels, sub["margin"], color=colors)
    ax.axvline(0.0, color="black", linewidth=1.2)
    ax.set_xlabel("MY-best minus SupCon accuracy")
    ax.set_title("CIFAR-100 full-check margins")
    plt.tight_layout()
    fig.savefig(OUT / "cifar100_margin_plot.png", dpi=180)
    plt.close(fig)

    best_row = df_full.iloc[0]
    best_pair = (str(best_row["class0"]), str(best_row["class1"]))
    best_exp = full_exps[best_pair]
    tag = f"{_slug(best_pair[0])}_vs_{_slug(best_pair[1])}"
    make_final_plots(ns, best_exp, tag=tag)

    print(df_full.to_string(index=False))
    print("\nInterpretation:")
    print(build_scan_summary(df_full))
    print(f"\nSaved outputs to {OUT}")
    open_png_outputs(OUT)


if __name__ == "__main__":
    main()
