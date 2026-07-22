"""Large-sample CIFAR-10 comparison of KDMLP, ME-KDMLP, KFDA, and SupCon.

The script deliberately screens a fixed set of six CIFAR-10 binary tasks for
which frozen ResNet-18 features were already cached by the earlier experiment.
Every task uses 500 observations per class and the same repeated 60/40 splits.
ME-KDMLP learns Stage 1 from disjoint training bags but trains and tests the
downstream linear probe on original single observations.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


WORKSPACE = Path(__file__).resolve().parent
PROJECT_ROOT = WORKSPACE.parents[2]
SIMULATION_ROOT = WORKSPACE.parent
OLD_CIFAR10_WORKSPACE = PROJECT_ROOT / "supcon_comparison_workspace"
OLD_CIFAR100_WORKSPACE = SIMULATION_ROOT / "supcon_cifar100_workspace"
CIFAR100_ME_WORKSPACE = SIMULATION_ROOT / "cifar100_me_kdmlp_dense_r_workspace"
BAG_WORKSPACE = SIMULATION_ROOT / "bag_kdmlp_workspace"
for source in (
    OLD_CIFAR10_WORKSPACE,
    OLD_CIFAR100_WORKSPACE,
    CIFAR100_ME_WORKSPACE,
    BAG_WORKSPACE,
):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

import compare_supcon_vs_ours as old_cifar10  # noqa: E402
import cifar100_same_r_supcon_vs_ours as old_projection  # noqa: E402
import run_cifar100_me_dense_r as me_core  # noqa: E402


DEFAULT_OUTPUT = WORKSPACE / "outputs"
OLD_PAIR_CACHE = OLD_CIFAR10_WORKSPACE / "outputs" / "pair_cache"
DEFAULT_PAIRS = (
    ("airplane", "bird"),
    ("airplane", "ship"),
    ("automobile", "truck"),
    ("cat", "dog"),
    ("deer", "horse"),
    ("ship", "truck"),
)
DEFAULT_R_VALUES = (1, 2, 4, 6, 8, 10, 16, 24)
DEFAULT_M_VALUES = (4, 8, 16)
DEFAULT_REPEATS = 3
DEFAULT_SEED = 0
N_PER_CLASS = 500
TRAIN_FRACTION = 0.60
ANCHORS_PER_CLASS = 80
KERNEL_ORDER = {
    "linear": 0,
    "rbf": 1,
    "laplacian": 2,
    "polynomial": 3,
    "sigmoid": 4,
}

COLORS = {
    "KFDA": "#2b2d42",
    "Original KDMLP": "#577590",
    "ME large-mu": "#c1121f",
    "ME small-mu": "#2455d6",
    "ME best": "#2a9d8f",
    "SupCon": "#f4a261",
}

sns.set_theme(style="whitegrid", context="talk")


def parse_int_values(text: str) -> tuple[int, ...]:
    values = tuple(dict.fromkeys(int(item.strip()) for item in text.split(",")))
    if not values or min(values) < 1:
        raise argparse.ArgumentTypeError("values must be positive integers")
    return values


def parse_pairs(text: str) -> tuple[tuple[str, str], ...]:
    pairs = []
    for item in text.split(","):
        names = tuple(name.strip() for name in item.split(":"))
        if len(names) != 2 or not all(names):
            raise argparse.ArgumentTypeError(
                "pairs must use class0:class1,class0:class1 syntax"
            )
        pairs.append((names[0], names[1]))
    return tuple(pairs)


def pair_name(class0: str, class1: str) -> str:
    return f"{class0} vs {class1}"


def load_pair(
    namespace: dict,
    *,
    class0: str,
    class1: str,
    data_seed: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Load the existing n=500-per-class frozen ResNet-18 feature cache."""

    old_cifar10.PAIR_CACHE = OLD_PAIR_CACHE
    bundle = old_cifar10.load_cifar10_pair_bundle(
        namespace,
        class0=class0,
        class1=class1,
        n_per_class=N_PER_CLASS,
        seed=data_seed,
        device=device,
    )
    return (
        np.asarray(bundle.X_resnet, dtype=np.float64),
        np.asarray(bundle.y, dtype=int),
    )


def evaluate_original_split(
    namespace: dict,
    features_train: np.ndarray,
    labels_train: np.ndarray,
    features_test: np.ndarray,
    labels_test: np.ndarray,
    *,
    class0: str,
    class1: str,
    rep: int,
    split_seed: int,
    r_values: tuple[int, ...],
    stage1_kernels: list[tuple[str, dict]],
) -> tuple[list[dict], dict]:
    """Use the historical point-KDMLP algebra with the old 80-anchor cap."""

    task = pair_name(class0, class1)
    safe_anchors = min(
        ANCHORS_PER_CLASS,
        int(np.sum(labels_train == 0)),
        int(np.sum(labels_train == 1)),
    )
    baseline_rows = namespace["evaluate_methods_on_fixed_split_fast"](
        features_train,
        labels_train,
        features_test,
        labels_test,
        task_name=f"CIFAR10: {task}",
        rep=rep,
        r_values=(1,),
        anchors_per_class=safe_anchors,
        mu_reg=1e-6,
        svm_folds=2,
        stage1_kernels=stage1_kernels,
        basis_mode="anchors",
        center_kernel=True,
        seed=split_seed,
        verbose=False,
    )
    baseline = pd.DataFrame(baseline_rows)
    kfda = baseline[baseline["family"] == "KFDA"].copy()
    kfda["kernel_order"] = kfda["stage1_kernel"].map(KERNEL_ORDER)
    kfda = kfda.sort_values(
        ["acc", "kernel_order"], ascending=[False, True]
    ).iloc[0]
    kfda_row = {
        "pair": task,
        "class0": class0,
        "class1": class1,
        "rep": int(rep),
        "r": 1,
        "acc": float(kfda["acc"]),
        "stage1_kernel": str(kfda["stage1_kernel"]),
    }

    rows = []
    max_r = max(r_values)
    for family, algorithm in (
        ("large_mu", "large_mu"),
        ("small_mu", "small_mu"),
    ):
        for metric, _ in stage1_kernels:
            projected_train, projected_test = old_projection.prepare_my_projection(
                namespace,
                Xtr=features_train,
                ytr=labels_train,
                Xother=features_test,
                metric=metric,
                algo=algorithm,
                max_r=max_r,
                seed=split_seed,
                anchors_per_class=safe_anchors,
            )
            available = min(projected_train.shape[1], projected_test.shape[1])
            for r in r_values:
                if r > available:
                    continue
                probe = old_cifar10.fit_linear_probe(
                    projected_train[:, :r],
                    labels_train,
                    projected_test[:, :r],
                    labels_test,
                    seed=7000 + split_seed + int(r),
                )
                rows.append(
                    {
                        "pair": task,
                        "class0": class0,
                        "class1": class1,
                        "rep": int(rep),
                        "family": family,
                        "r": int(r),
                        "stage1_kernel": metric,
                        "acc": float(probe["acc"]),
                        "probe_C": float(probe["C"]),
                    }
                )
    return rows, kfda_row


def evaluate_supcon_split(
    features_train: np.ndarray,
    labels_train: np.ndarray,
    features_test: np.ndarray,
    labels_test: np.ndarray,
    *,
    class0: str,
    class1: str,
    rep: int,
    split_seed: int,
    r_values: tuple[int, ...],
    epochs: int,
    device: str,
) -> list[dict]:
    """Train the historical feature-level SupCon model and match final r."""

    model = old_cifar10.train_supcon_feature_model(
        features_train,
        labels_train,
        features_test,
        labels_test,
        device=device,
        seed=split_seed,
        epochs=epochs,
    )
    rows = []
    for r in r_values:
        if r > model["Ztr"].shape[1]:
            continue
        pca = PCA(n_components=r, random_state=split_seed)
        projected_train = pca.fit_transform(model["Ztr"])
        projected_test = pca.transform(model["Zte"])
        probe = old_cifar10.fit_linear_probe(
            projected_train,
            labels_train,
            projected_test,
            labels_test,
            seed=5000 + split_seed + int(r),
        )
        rows.append(
            {
                "pair": pair_name(class0, class1),
                "class0": class0,
                "class1": class1,
                "rep": int(rep),
                "r": int(r),
                "acc": float(probe["acc"]),
                "probe_C": float(probe["C"]),
                "epochs": int(epochs),
            }
        )
    return rows


def select_best_kernel(
    candidates: pd.DataFrame,
    group_columns: list[str],
) -> pd.DataFrame:
    candidates = candidates.copy()
    candidates["kernel_order"] = candidates["stage1_kernel"].map(KERNEL_ORDER)
    return (
        candidates.sort_values(
            group_columns + ["acc", "kernel_order"],
            ascending=[True] * len(group_columns) + [False, True],
        )
        .groupby(group_columns, as_index=False)
        .head(1)
        .drop(columns="kernel_order")
        .reset_index(drop=True)
    )


def summarize(
    original_candidates: pd.DataFrame,
    kfda_rows: pd.DataFrame,
    me_candidates: pd.DataFrame,
    supcon_rows: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    original_selected = select_best_kernel(
        original_candidates,
        ["pair", "class0", "class1", "rep", "family", "r"],
    )
    valid_me = me_candidates[me_candidates["status"] == "ok"].copy()
    me_selected = select_best_kernel(
        valid_me,
        ["pair", "class0", "class1", "rep", "m", "family", "r"],
    )

    original = (
        original_selected.groupby(
            ["pair", "class0", "class1", "family", "r"], as_index=False
        )["acc"]
        .agg(acc_mean="mean", acc_std="std")
        .fillna({"acc_std": 0.0})
    )
    me = (
        me_selected.groupby(
            ["pair", "class0", "class1", "m", "family", "r"],
            as_index=False,
        )["acc"]
        .agg(acc_mean="mean", acc_std="std")
        .fillna({"acc_std": 0.0})
    )
    kfda = (
        kfda_rows.groupby(["pair", "class0", "class1"], as_index=False)["acc"]
        .agg(kfda_acc="mean", kfda_std="std")
        .fillna({"kfda_std": 0.0})
    )
    supcon = (
        supcon_rows.groupby(
            ["pair", "class0", "class1", "r"], as_index=False
        )["acc"]
        .agg(supcon_acc="mean", supcon_std="std")
        .fillna({"supcon_std": 0.0})
    )

    def family_table(
        data: pd.DataFrame,
        family: str,
        prefix: str,
        *,
        include_m: bool,
    ) -> pd.DataFrame:
        keys = ["pair", "class0", "class1"]
        if include_m:
            keys.append("m")
        keys.append("r")
        return data[data["family"] == family][
            keys + ["acc_mean", "acc_std"]
        ].rename(
            columns={
                "acc_mean": f"{prefix}_acc",
                "acc_std": f"{prefix}_std",
            }
        )

    original_large = family_table(
        original, "large_mu", "original_large", include_m=False
    )
    original_small = family_table(
        original, "small_mu", "original_small", include_m=False
    )
    reference = original_large.merge(
        original_small,
        on=["pair", "class0", "class1", "r"],
        validate="one_to_one",
    ).merge(
        kfda,
        on=["pair", "class0", "class1"],
        validate="many_to_one",
    ).merge(
        supcon,
        on=["pair", "class0", "class1", "r"],
        validate="one_to_one",
    )
    reference["original_best_acc"] = reference[
        ["original_large_acc", "original_small_acc"]
    ].max(axis=1)
    reference["original_best_family"] = np.where(
        reference["original_large_acc"] >= reference["original_small_acc"],
        "large_mu",
        "small_mu",
    )

    me_large = family_table(me, "large_mu", "me_large", include_m=True)
    me_small = family_table(me, "small_mu", "me_small", include_m=True)
    compact = me_large.merge(
        me_small,
        on=["pair", "class0", "class1", "m", "r"],
        how="outer",
        validate="one_to_one",
    ).merge(
        reference,
        on=["pair", "class0", "class1", "r"],
        how="left",
        validate="many_to_one",
    )
    compact["me_best_acc"] = compact[["me_large_acc", "me_small_acc"]].max(
        axis=1
    )
    compact["me_best_family"] = np.where(
        compact["me_large_acc"].fillna(-np.inf)
        >= compact["me_small_acc"].fillna(-np.inf),
        "large_mu",
        "small_mu",
    )
    compact["me_minus_original"] = (
        compact["me_best_acc"] - compact["original_best_acc"]
    )
    compact["me_minus_kfda"] = compact["me_best_acc"] - compact["kfda_acc"]
    compact["me_minus_supcon"] = (
        compact["me_best_acc"] - compact["supcon_acc"]
    )
    compact["me_original_abs_gap"] = compact["me_minus_original"].abs()
    compact["minimum_baseline_margin"] = compact[
        ["me_minus_kfda", "me_minus_supcon"]
    ].min(axis=1)
    compact["me_retains_original_and_wins"] = (
        (compact["me_minus_original"] >= -0.02 - 1e-12)
        & (compact["me_minus_kfda"] > 0.0)
        & (compact["me_minus_supcon"] > 0.0)
    )
    return original_selected, me_selected, compact


def choose_representatives(compact: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pair, group in compact.groupby("pair", sort=False):
        successful = group[group["me_retains_original_and_wins"]].copy()
        source = successful if len(successful) else group.copy()
        best = source.sort_values(
            [
                "minimum_baseline_margin",
                "me_original_abs_gap",
                "me_best_acc",
                "m",
                "r",
            ],
            ascending=[False, True, False, True, True],
        ).iloc[0]
        rows.append(best)
    return pd.DataFrame(rows).reset_index(drop=True)


def plot_all_pairs(
    compact: pd.DataFrame,
    *,
    m: int,
    r_values: tuple[int, ...],
    output_path: Path,
) -> None:
    pairs = list(dict.fromkeys(compact["pair"]))
    fig, axes = plt.subplots(2, 3, figsize=(20, 11))
    positions = np.arange(len(r_values))
    for axis, pair in zip(axes.reshape(-1), pairs):
        data = compact[(compact["pair"] == pair) & (compact["m"] == m)]
        data = data.drop_duplicates("r").set_index("r").reindex(r_values)
        for column, label, style in (
            ("kfda_acc", "KFDA", ":"),
            ("original_best_acc", "Original KDMLP", "--"),
            ("me_large_acc", "ME large-mu", "-"),
            ("me_small_acc", "ME small-mu", "-"),
            ("supcon_acc", "SupCon", "-."),
        ):
            axis.plot(
                positions,
                data[column],
                marker="o",
                linewidth=2.1,
                linestyle=style,
                color=COLORS[label],
                label=label,
            )
        valid = int(data["me_best_acc"].notna().sum())
        axis.set_title(f"{pair}\n{valid}/{len(r_values)} ME dimensions valid")
        axis.set_xlabel("projection dimension r")
        axis.set_ylabel("test accuracy")
        axis.set_xticks(positions, [str(value) for value in r_values])
        axis.tick_params(axis="x", labelsize=10)
    handles, labels = axes.reshape(-1)[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.97),
        ncol=5,
        frameon=True,
    )
    fig.suptitle(
        f"CIFAR-10 large-sample comparison at fixed bag size m={m}",
        y=0.998,
        fontsize=23,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_selected_pair(
    compact: pd.DataFrame,
    representative: pd.Series,
    *,
    m_values: tuple[int, ...],
    r_values: tuple[int, ...],
    output_path: Path,
) -> None:
    pair = str(representative["pair"])
    fig, axes = plt.subplots(1, len(m_values), figsize=(20, 6), sharey=True)
    axes_array = np.atleast_1d(axes)
    positions = np.arange(len(r_values))
    for axis, m in zip(axes_array, m_values):
        data = compact[(compact["pair"] == pair) & (compact["m"] == m)]
        data = data.drop_duplicates("r").set_index("r").reindex(r_values)
        for column, label, style in (
            ("kfda_acc", "KFDA", ":"),
            ("original_best_acc", "Original KDMLP", "--"),
            ("me_large_acc", "ME large-mu", "-"),
            ("me_small_acc", "ME small-mu", "-"),
            ("supcon_acc", "SupCon", "-."),
        ):
            axis.plot(
                positions,
                data[column],
                marker="o",
                linewidth=2.2,
                linestyle=style,
                color=COLORS[label],
                label=label,
            )
        axis.set_title(f"m={m}")
        axis.set_xlabel("projection dimension r")
        axis.set_xticks(positions, [str(value) for value in r_values])
        axis.tick_params(axis="x", labelsize=10)
    axes_array[0].set_ylabel("test accuracy")
    handles, labels = axes_array[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=5,
        frameon=True,
    )
    fig.suptitle(
        f"Selected larger-sample task: {pair}", y=1.0, fontsize=23
    )
    plt.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_screening_summary(summary: pd.DataFrame, output_path: Path) -> None:
    ordered = summary.sort_values("minimum_baseline_margin")
    fig, ax = plt.subplots(figsize=(11, 6.5))
    y = np.arange(len(ordered))
    width = 0.22
    ax.barh(
        y - 1.5 * width,
        ordered["kfda_acc"],
        height=width,
        color=COLORS["KFDA"],
        label="KFDA",
    )
    ax.barh(
        y - 0.5 * width,
        ordered["supcon_acc"],
        height=width,
        color=COLORS["SupCon"],
        label="SupCon",
    )
    ax.barh(
        y + 0.5 * width,
        ordered["original_best_acc"],
        height=width,
        color=COLORS["Original KDMLP"],
        label="Original KDMLP",
    )
    ax.barh(
        y + 1.5 * width,
        ordered["me_best_acc"],
        height=width,
        color=COLORS["ME best"],
        label="ME-KDMLP",
    )
    ax.set_yticks(y, ordered["pair"])
    ax.set_xlabel("mean test accuracy")
    ax.set_title("Representative configuration from each declared CIFAR-10 task")
    ax.set_xlim(max(0.45, float(ordered[["kfda_acc", "supcon_acc"]].min().min()) - 0.04), 1.01)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.97),
        frameon=True,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def open_png_outputs(output_dir: Path) -> None:
    if sys.platform == "darwin":
        opener = ["open"]
    else:
        executable = shutil.which("xdg-open")
        if executable is None:
            return
        opener = [executable]
    for path in sorted(output_dir.glob("*.png")):
        subprocess.run(
            opener + [str(path)],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def save_checkpoints(
    output_dir: Path,
    *,
    original_rows: list[dict],
    kfda_rows: list[dict],
    me_rows: list[dict],
    supcon_rows: list[dict],
) -> None:
    pd.DataFrame(original_rows).to_csv(
        output_dir / "original_candidates.csv", index=False
    )
    pd.DataFrame(kfda_rows).to_csv(output_dir / "kfda_selected.csv", index=False)
    pd.DataFrame(me_rows).to_csv(output_dir / "me_candidates.csv", index=False)
    pd.DataFrame(supcon_rows).to_csv(
        output_dir / "supcon_same_r.csv", index=False
    )


def load_checkpoint_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return pd.read_csv(path).to_dict("records")


def split_row_count(rows: list[dict], *, pair: str, rep: int) -> int:
    return sum(
        str(row.get("pair")) == pair and int(row.get("rep", -1)) == rep
        for row in rows
    )


def without_split(rows: list[dict], *, pair: str, rep: int) -> list[dict]:
    return [
        row
        for row in rows
        if not (str(row.get("pair")) == pair and int(row.get("rep", -1)) == rep)
    ]


def run(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    namespace = old_cifar10.load_notebook_namespace(old_cifar10.NOTEBOOK_PATH)
    namespace["set_global_seed"](args.seed)
    device = namespace["pick_device"]()
    stage1_kernels = (
        namespace["STAGE1_KERNELS_FULL"]
        if args.kernels == "full"
        else namespace["STAGE1_KERNELS_QUICK"]
    )
    print(f"device: {device}")
    print(f"pairs: {len(args.pairs)}, repeats: {args.repeats}")
    print(f"r grid: {args.r_values}; m grid: {args.m_values}")

    if args.resume:
        original_rows = load_checkpoint_records(
            output_dir / "original_candidates.csv"
        )
        kfda_rows = load_checkpoint_records(output_dir / "kfda_selected.csv")
        me_rows = load_checkpoint_records(output_dir / "me_candidates.csv")
        supcon_rows = load_checkpoint_records(output_dir / "supcon_same_r.csv")
    else:
        original_rows = []
        kfda_rows = []
        me_rows = []
        supcon_rows = []
    setup_rows = []
    started = time.perf_counter()

    for pair_index, (class0, class1) in enumerate(args.pairs, start=1):
        features, labels = load_pair(
            namespace,
            class0=class0,
            class1=class1,
            data_seed=args.data_seed,
            device=device,
        )
        counts = np.bincount(labels, minlength=2)
        print(
            f"\n[{pair_index}/{len(args.pairs)}] {pair_name(class0, class1)} "
            f"features={features.shape}, class counts={counts.tolist()}"
        )
        for rep in range(args.repeats):
            split_seed = args.seed + 97 * rep
            train_indices, test_indices = old_cifar10.split_indices(
                labels, train_frac=TRAIN_FRACTION, seed=split_seed
            )
            features_train = features[train_indices]
            labels_train = labels[train_indices]
            features_test = features[test_indices]
            labels_test = labels[test_indices]
            split_started = time.perf_counter()

            train_counts = np.bincount(labels_train, minlength=2)
            test_counts = np.bincount(labels_test, minlength=2)
            setup_rows.append(
                {
                    "pair": pair_name(class0, class1),
                    "rep": rep,
                    "data_seed": int(args.data_seed),
                    "split_seed": int(split_seed),
                    "feature_dimension": features.shape[1],
                    "class0_total": int(counts[0]),
                    "class1_total": int(counts[1]),
                    "class0_train": int(train_counts[0]),
                    "class1_train": int(train_counts[1]),
                    "class0_test": int(test_counts[0]),
                    "class1_test": int(test_counts[1]),
                }
            )

            task = pair_name(class0, class1)
            expected_original = 2 * len(stage1_kernels) * len(args.r_values)
            expected_me = (
                len(args.m_values)
                * 2
                * len(stage1_kernels)
                * len(args.r_values)
            )
            complete = (
                split_row_count(original_rows, pair=task, rep=rep)
                == expected_original
                and split_row_count(kfda_rows, pair=task, rep=rep) == 1
                and split_row_count(me_rows, pair=task, rep=rep) == expected_me
                and split_row_count(supcon_rows, pair=task, rep=rep)
                == len(args.r_values)
            )
            if complete:
                print(f"  rep {rep + 1}/{args.repeats} resumed from checkpoint")
                continue

            # A split is atomic: discard any partial rows before recomputing it.
            original_rows = without_split(original_rows, pair=task, rep=rep)
            kfda_rows = without_split(kfda_rows, pair=task, rep=rep)
            me_rows = without_split(me_rows, pair=task, rep=rep)
            supcon_rows = without_split(supcon_rows, pair=task, rep=rep)

            new_original, new_kfda = evaluate_original_split(
                namespace,
                features_train,
                labels_train,
                features_test,
                labels_test,
                class0=class0,
                class1=class1,
                rep=rep,
                split_seed=split_seed,
                r_values=args.r_values,
                stage1_kernels=stage1_kernels,
            )
            original_rows.extend(new_original)
            kfda_rows.append(new_kfda)

            scaler = StandardScaler()
            standardized_train = scaler.fit_transform(features_train)
            standardized_test = scaler.transform(features_test)
            for m in args.m_values:
                me_rows.extend(
                    me_core.evaluate_me_split(
                        namespace,
                        standardized_train,
                        labels_train,
                        standardized_test,
                        labels_test,
                        class0=class0,
                        class1=class1,
                        rep=rep,
                        split_seed=split_seed,
                        bag_size=m,
                        r_values=args.r_values,
                        stage1_kernels=stage1_kernels,
                    )
                )

            supcon_rows.extend(
                evaluate_supcon_split(
                    features_train,
                    labels_train,
                    features_test,
                    labels_test,
                    class0=class0,
                    class1=class1,
                    rep=rep,
                    split_seed=split_seed,
                    r_values=args.r_values,
                    epochs=args.supcon_epochs,
                    device=device,
                )
            )
            save_checkpoints(
                output_dir,
                original_rows=original_rows,
                kfda_rows=kfda_rows,
                me_rows=me_rows,
                supcon_rows=supcon_rows,
            )
            print(
                f"  rep {rep + 1}/{args.repeats} finished in "
                f"{time.perf_counter() - split_started:.1f}s"
            )

    original_candidates = pd.DataFrame(original_rows)
    kfda = pd.DataFrame(kfda_rows)
    me_candidates = pd.DataFrame(me_rows)
    supcon = pd.DataFrame(supcon_rows)
    original_selected, me_selected, compact = summarize(
        original_candidates, kfda, me_candidates, supcon
    )
    representatives = choose_representatives(compact)
    successful = representatives[
        representatives["me_retains_original_and_wins"]
    ].copy()

    original_selected.to_csv(output_dir / "original_selected.csv", index=False)
    me_selected.to_csv(output_dir / "me_selected.csv", index=False)
    compact.to_csv(output_dir / "cifar10_large_sample_compact.csv", index=False)
    representatives.to_csv(output_dir / "pair_representatives.csv", index=False)
    pd.DataFrame(setup_rows).to_csv(output_dir / "dataset_setup.csv", index=False)

    rank_rows = []
    for pair in compact["pair"].drop_duplicates():
        pair_candidates = me_candidates[me_candidates["pair"] == pair]
        for m in args.m_values:
            for family in ("large_mu", "small_mu"):
                group = pair_candidates[
                    (pair_candidates["m"] == m)
                    & (pair_candidates["family"] == family)
                ]
                valid = group[group["status"] == "ok"]
                rank_rows.append(
                    {
                        "pair": pair,
                        "m": m,
                        "family": family,
                        "bags_per_class": int(group["class0_bags"].iloc[0]),
                        "largest_requested_r_supported": int(valid["r"].max()),
                        "valid_requested_dimensions": int(valid["r"].nunique()),
                    }
                )
    pd.DataFrame(rank_rows).to_csv(output_dir / "rank_support.csv", index=False)

    for m in args.m_values:
        plot_all_pairs(
            compact,
            m=m,
            r_values=args.r_values,
            output_path=output_dir / f"cifar10_all_pairs_fixed_m{m}.png",
        )
    selected_row = (
        successful.sort_values(
            ["minimum_baseline_margin", "me_original_abs_gap"],
            ascending=[False, True],
        ).iloc[0]
        if len(successful)
        else representatives.sort_values(
            ["minimum_baseline_margin", "me_original_abs_gap"],
            ascending=[False, True],
        ).iloc[0]
    )
    plot_selected_pair(
        compact,
        selected_row,
        m_values=args.m_values,
        r_values=args.r_values,
        output_path=output_dir / "selected_pair_all_m.png",
    )
    plot_screening_summary(
        representatives, output_dir / "screening_summary.png"
    )

    print("\nRepresentative configuration per declared pair:")
    display_columns = [
        "pair",
        "m",
        "r",
        "me_best_family",
        "me_best_acc",
        "original_best_acc",
        "kfda_acc",
        "supcon_acc",
        "me_minus_original",
        "me_minus_kfda",
        "me_minus_supcon",
        "me_retains_original_and_wins",
    ]
    print(representatives[display_columns].to_string(index=False))
    print(
        f"\n{len(successful)}/{len(representatives)} declared pairs contain "
        "a configuration where ME-KDMLP is no more than 0.02 below original "
        "KDMLP "
        "and exceeds both KFDA and SupCon."
    )
    print(
        f"Selected example: {selected_row['pair']}, m={int(selected_row['m'])}, "
        f"r={int(selected_row['r'])}, family={selected_row['me_best_family']}, "
        f"ME={selected_row['me_best_acc']:.4f}, "
        f"original={selected_row['original_best_acc']:.4f}, "
        f"KFDA={selected_row['kfda_acc']:.4f}, "
        f"SupCon={selected_row['supcon_acc']:.4f}."
    )
    print(f"Total wall-clock time: {(time.perf_counter() - started) / 60:.1f} min")
    print(f"Outputs: {output_dir}")
    if args.open_plots:
        open_png_outputs(output_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pairs", type=parse_pairs, default=DEFAULT_PAIRS)
    parser.add_argument("--r-values", type=parse_int_values, default=DEFAULT_R_VALUES)
    parser.add_argument("--m-values", type=parse_int_values, default=DEFAULT_M_VALUES)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--data-seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--supcon-epochs", type=int, default=16)
    parser.add_argument("--kernels", choices=("quick", "full"), default="full")
    parser.add_argument("--no-resume", action="store_false", dest="resume")
    parser.set_defaults(resume=True)
    parser.add_argument("--open-plots", action="store_true")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
