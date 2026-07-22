"""Compare historical and mean-embedding KDMLP on selected CIFAR-100 pairs."""

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
from sklearn.preprocessing import StandardScaler


WORKSPACE = Path(__file__).resolve().parent
SIMULATION_ROOT = WORKSPACE.parent
OLD_CIFAR_WORKSPACE = SIMULATION_ROOT / "supcon_cifar100_workspace"
BAG_WORKSPACE = SIMULATION_ROOT / "bag_kdmlp_workspace"
for source in (OLD_CIFAR_WORKSPACE, BAG_WORKSPACE):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

import cifar100_same_r_supcon_vs_ours as old_cifar  # noqa: E402
from bag_kdmlp import (  # noqa: E402
    bag_kernel_blocks,
    balanced_anchor_indices,
    kernel_params_for_bags,
    make_disjoint_class_bags,
    make_singleton_observations,
    validate_homogeneous_bags,
)
from me_kdmlp import (  # noqa: E402
    gaussian_regime_from_terms,
    prepare_me_kdmlp_from_kernel_blocks,
)


DEFAULT_OUT = WORKSPACE / "outputs"
OLD_PAIR_CACHE = OLD_CIFAR_WORKSPACE / "outputs" / "pair_cache"
OLD_RESULTS = OLD_CIFAR_WORKSPACE / "outputs_more_examples"
DEFAULT_PAIRS = (
    ("lion", "possum"),
    ("bee", "spider"),
    ("bear", "palm_tree"),
    ("palm_tree", "train"),
)
DEFAULT_R_VALUES = (1, 2, 4, 6, 8, 10, 16, 24)
DEFAULT_M_VALUES = (1, 2, 4, 8, 16)
DEFAULT_REPEATS = 2
DEFAULT_SEED = 41
N_PER_CLASS = 80
ORIGINAL_PROTOCOL = "historical_cifar100_linear_probe_dense_r_v1"
ME_PROTOCOL = "disjoint_bag_coordinate_me_single_point_probe_v1"
FAMILIES = ("large_mu", "small_mu")
KERNEL_ORDER = {
    "linear": 0,
    "rbf": 1,
    "laplacian": 2,
    "polynomial": 3,
    "sigmoid": 4,
}

COLORS = {
    "KFDA (rank 1)": "#2b2d42",
    "Original large-mu": "#bc6c25",
    "Original small-mu": "#457b9d",
    "Original KDMLP best": "#577590",
    "ME-KDMLP large-mu": "#c1121f",
    "ME-KDMLP small-mu": "#2455d6",
    "ME-KDMLP best": "#2a9d8f",
}

sns.set_theme(style="whitegrid", context="talk")


def parse_int_values(text: str) -> tuple[int, ...]:
    values = tuple(dict.fromkeys(int(value.strip()) for value in text.split(",")))
    if not values or min(values) < 1:
        raise argparse.ArgumentTypeError("values must be positive integers")
    return values


def parse_pairs(text: str) -> tuple[tuple[str, str], ...]:
    pairs = []
    for item in text.split(","):
        classes = tuple(value.strip() for value in item.split(":"))
        if len(classes) != 2 or not all(classes):
            raise argparse.ArgumentTypeError(
                "pairs must use class0:class1,class0:class1 syntax"
            )
        pairs.append((classes[0], classes[1]))
    return tuple(pairs)


def slug(text: str) -> str:
    output = []
    for character in str(text).lower():
        output.append(character if character.isalnum() else "_")
    value = "".join(output).strip("_")
    while "__" in value:
        value = value.replace("__", "_")
    return value


def pair_name(class0: str, class1: str) -> str:
    return f"{class0} vs {class1}"


def load_pair(
    namespace: dict,
    *,
    class0: str,
    class1: str,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Load the exact cached feature sample used by the old experiment."""

    old_cifar.PAIR_CACHE = OLD_PAIR_CACHE
    features, labels, _ = old_cifar.load_cifar100_pair_bundle(
        namespace,
        class0=class0,
        class1=class1,
        n_per_class=N_PER_CLASS,
        seed=DEFAULT_SEED,
        device=device,
    )
    return np.asarray(features, dtype=np.float64), np.asarray(labels, dtype=int)


def select_historical_original(
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    """Apply the exact old per-split, per-family descriptive kernel rule."""

    candidates = candidates.copy()
    candidates["kernel_order"] = candidates["stage1_kernel"].map(KERNEL_ORDER)
    return (
        candidates.sort_values(
            ["pair", "rep", "family", "r", "acc", "kernel_order"],
            ascending=[True, True, True, True, False, True],
        )
        .groupby(["pair", "class0", "class1", "rep", "family", "r"], as_index=False)
        .head(1)
        .drop(columns="kernel_order")
        .reset_index(drop=True)
    )


def evaluate_historical_split(
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
    """Reproduce the old point-level CIFAR evaluator on a denser r grid."""

    name = pair_name(class0, class1)
    anchors_per_class = int(
        min(np.sum(labels_train == 0), np.sum(labels_train == 1))
    )
    kfda_rows = namespace["evaluate_methods_on_fixed_split_fast"](
        features_train,
        labels_train,
        features_test,
        labels_test,
        task_name=f"CIFAR100: {name}",
        rep=rep,
        r_values=(1,),
        anchors_per_class=anchors_per_class,
        mu_reg=1e-6,
        svm_folds=2,
        stage1_kernels=stage1_kernels,
        basis_mode="anchors",
        center_kernel=True,
        seed=split_seed,
        verbose=False,
    )
    kfda = pd.DataFrame(kfda_rows)
    kfda = kfda[kfda["family"] == "KFDA"].copy()
    kfda["kernel_order"] = kfda["stage1_kernel"].map(KERNEL_ORDER)
    kfda = kfda.sort_values(["acc", "kernel_order"], ascending=[False, True])
    best_kfda = kfda.iloc[0]
    kfda_row = {
        "pair": name,
        "class0": class0,
        "class1": class1,
        "rep": int(rep),
        "family": "KFDA",
        "r": 1,
        "acc": float(best_kfda["acc"]),
        "stage1_kernel": str(best_kfda["stage1_kernel"]),
        "evaluation_protocol": ORIGINAL_PROTOCOL,
    }

    rows = []
    max_r = max(r_values)
    for family, algorithm in (
        ("MY-large_mu", "large_mu"),
        ("MY-small_mu", "small_mu"),
    ):
        for metric, _ in stage1_kernels:
            projected_train, projected_test = old_cifar.prepare_my_projection(
                namespace,
                Xtr=features_train,
                ytr=labels_train,
                Xother=features_test,
                metric=metric,
                algo=algorithm,
                max_r=max_r,
                seed=split_seed,
                anchors_per_class=anchors_per_class,
            )
            available = min(projected_train.shape[1], projected_test.shape[1])
            for r in r_values:
                if r > available:
                    continue
                probe = old_cifar.fit_linear_probe(
                    projected_train[:, :r],
                    labels_train,
                    projected_test[:, :r],
                    labels_test,
                    seed=7000 + split_seed + int(r),
                )
                rows.append(
                    {
                        "pair": name,
                        "class0": class0,
                        "class1": class1,
                        "rep": int(rep),
                        "family": family,
                        "r": int(r),
                        "acc": float(probe["acc"]),
                        "probe_C": float(probe["C"]),
                        "stage1_kernel": metric,
                        "evaluation_protocol": ORIGINAL_PROTOCOL,
                    }
                )
    return rows, kfda_row


def invalid_me_rows(
    *,
    class0: str,
    class1: str,
    rep: int,
    bag_size: int,
    r_values: tuple[int, ...],
    kernel_names: tuple[str, ...],
    status: str,
    detail: str,
    class_bag_counts: tuple[int, int],
) -> list[dict]:
    rows = []
    for metric in kernel_names:
        for family in FAMILIES:
            for r in r_values:
                rows.append(
                    {
                        "pair": pair_name(class0, class1),
                        "class0": class0,
                        "class1": class1,
                        "rep": int(rep),
                        "m": int(bag_size),
                        "r": int(r),
                        "stage1_kernel": metric,
                        "family": family,
                        "status": status,
                        "status_detail": detail,
                        "class0_bags": class_bag_counts[0],
                        "class1_bags": class_bag_counts[1],
                        "acc": np.nan,
                        "evaluation_protocol": ME_PROTOCOL,
                    }
                )
    return rows


def evaluate_me_split(
    namespace: dict,
    standardized_train: np.ndarray,
    labels_train: np.ndarray,
    standardized_test: np.ndarray,
    labels_test: np.ndarray,
    *,
    class0: str,
    class1: str,
    rep: int,
    split_seed: int,
    bag_size: int,
    r_values: tuple[int, ...],
    stage1_kernels: list[tuple[str, dict]],
) -> list[dict]:
    """Learn ME Stage 1 from disjoint bags and classify original points."""

    kernel_names = tuple(metric for metric, _ in stage1_kernels)
    try:
        train_bags = make_disjoint_class_bags(
            labels_train,
            bag_size,
            seed=split_seed + 10_000 * int(bag_size) + 1,
        )
    except ValueError as error:
        counts = tuple(
            int(np.sum(labels_train == label) // bag_size) for label in (0, 1)
        )
        return invalid_me_rows(
            class0=class0,
            class1=class1,
            rep=rep,
            bag_size=bag_size,
            r_values=r_values,
            kernel_names=kernel_names,
            status="insufficient_bags",
            detail=str(error),
            class_bag_counts=counts,
        )

    validate_homogeneous_bags(train_bags)
    class_bag_counts = tuple(
        int(np.sum(train_bags.labels == label)) for label in (0, 1)
    )
    if min(class_bag_counts) < 2:
        return invalid_me_rows(
            class0=class0,
            class1=class1,
            rep=rep,
            bag_size=bag_size,
            r_values=r_values,
            kernel_names=kernel_names,
            status="insufficient_bags",
            detail="each class needs at least two disjoint bags",
            class_bag_counts=class_bag_counts,
        )

    test_points = make_singleton_observations(labels_test)
    rows = []
    for kernel_index, (metric, raw_params) in enumerate(stage1_kernels):
        geometry_seed = split_seed + 31 * kernel_index
        anchor_indices, safe_anchors = balanced_anchor_indices(
            train_bags.labels,
            anchors_per_class=min(class_bag_counts),
            seed=geometry_seed,
        )
        params = kernel_params_for_bags(
            namespace,
            metric,
            raw_params,
            standardized_train,
            train_bags,
            anchor_indices,
        )
        bag_gram, bag_to_train, bag_to_test = bag_kernel_blocks(
            standardized_train,
            standardized_test,
            train_bags,
            test_points,
            anchor_indices,
            metric=metric,
            params=params,
            kernel_function=namespace["safe_pairwise_kernels"],
            project_training_points=True,
        )
        try:
            projection = prepare_me_kdmlp_from_kernel_blocks(
                bag_gram,
                bag_to_train,
                bag_to_test,
                train_bags.labels[anchor_indices],
                max_r=max(r_values),
                covariance_ridge=1e-6,
                center_kernel=True,
            )
        except (ValueError, np.linalg.LinAlgError) as error:
            rows.extend(
                invalid_me_rows(
                    class0=class0,
                    class1=class1,
                    rep=rep,
                    bag_size=bag_size,
                    r_values=r_values,
                    kernel_names=(metric,),
                    status="projection_failure",
                    detail=str(error),
                    class_bag_counts=class_bag_counts,
                )
            )
            continue

        family_coordinates = {
            "large_mu": (
                projection["Ztr_large_full"],
                projection["Zte_large_full"],
                int(projection["large_direction_count"]),
                safe_anchors,
            ),
            "small_mu": (
                projection["Ztr_small_full"],
                projection["Zte_small_full"],
                int(projection["small_direction_count"]),
                max(0, safe_anchors - 1),
            ),
        }
        for family_index, family in enumerate(FAMILIES):
            projected_train, projected_test, available, stable_limit = (
                family_coordinates[family]
            )
            for r in r_values:
                common = {
                    "pair": pair_name(class0, class1),
                    "class0": class0,
                    "class1": class1,
                    "rep": int(rep),
                    "m": int(bag_size),
                    "r": int(r),
                    "stage1_kernel": metric,
                    "family": family,
                    "class0_bags": class_bag_counts[0],
                    "class1_bags": class_bag_counts[1],
                    "stable_r_limit": int(stable_limit),
                    "coordinate_rank": int(projection["coordinate_rank"]),
                    "available_directions": int(available),
                    "D_mu": float(projection["D_mu"]),
                    "D_covariance": float(projection["D_covariance"]),
                    "heuristic_family": gaussian_regime_from_terms(
                        projection["D_mu"],
                        projection["D_covariance"],
                        r,
                    ),
                    "evaluation_protocol": ME_PROTOCOL,
                }
                if r > stable_limit or r > available:
                    rows.append(
                        {
                            **common,
                            "status": "rank_limited",
                            "status_detail": (
                                f"requested r={r}, stable limit={stable_limit}, "
                                f"available directions={available}"
                            ),
                            "acc": np.nan,
                            "probe_C": np.nan,
                        }
                    )
                    continue
                probe = old_cifar.fit_linear_probe(
                    projected_train[:, :r],
                    labels_train,
                    projected_test[:, :r],
                    labels_test,
                    seed=(
                        9000
                        + split_seed
                        + 100 * int(bag_size)
                        + 10 * kernel_index
                        + family_index
                        + int(r)
                    ),
                )
                rows.append(
                    {
                        **common,
                        "status": "ok",
                        "status_detail": "",
                        "acc": float(probe["acc"]),
                        "probe_C": float(probe["C"]),
                    }
                )
    return rows


def audit_historical_overlap(
    selected: pd.DataFrame,
    kfda: pd.DataFrame,
    *,
    pairs: tuple[tuple[str, str], ...],
    r_values: tuple[int, ...],
) -> pd.DataFrame:
    overlap = sorted(set(r_values).intersection({1, 2, 3, 4, 5, 6, 32, 64}))
    audit_rows = []
    for class0, class1 in pairs:
        tag = f"{slug(class0)}_vs_{slug(class1)}"
        old_rows = pd.read_csv(OLD_RESULTS / f"{tag}_same_r_rows.csv")
        old_rows = old_rows[
            old_rows["method"].isin(["MY-large_mu", "MY-small_mu"])
            & old_rows["r"].isin(overlap)
        ][["rep", "family", "r", "acc", "kernel"]]
        new_rows = selected[
            (selected["class0"] == class0)
            & (selected["class1"] == class1)
            & selected["r"].isin(overlap)
        ][["rep", "family", "r", "acc", "stage1_kernel"]]
        merged = new_rows.merge(
            old_rows,
            on=["rep", "family", "r"],
            suffixes=("_new", "_old"),
            validate="one_to_one",
        )
        merged["pair"] = pair_name(class0, class1)
        merged["acc_abs_error"] = (merged["acc_new"] - merged["acc_old"]).abs()
        merged["kernel_match"] = (
            merged["stage1_kernel"].astype(str) == merged["kernel"].astype(str)
        )
        audit_rows.extend(merged.to_dict("records"))

        old_kfda = pd.read_csv(OLD_RESULTS / f"{tag}_kfda_rows.csv")
        new_kfda = kfda[
            (kfda["class0"] == class0) & (kfda["class1"] == class1)
        ]
        kfda_merge = new_kfda.merge(
            old_kfda[["rep", "acc", "kernel"]],
            on="rep",
            suffixes=("_new", "_old"),
            validate="one_to_one",
        )
        for row in kfda_merge.itertuples(index=False):
            audit_rows.append(
                {
                    "rep": int(row.rep),
                    "family": "KFDA",
                    "r": 1,
                    "acc_new": float(row.acc_new),
                    "acc_old": float(row.acc_old),
                    "stage1_kernel": str(row.stage1_kernel),
                    "kernel": str(row.kernel),
                    "pair": pair_name(class0, class1),
                    "acc_abs_error": abs(float(row.acc_new) - float(row.acc_old)),
                    "kernel_match": str(row.stage1_kernel) == str(row.kernel),
                }
            )
    audit = pd.DataFrame(audit_rows)
    if len(audit):
        if audit["acc_abs_error"].max() > 1e-12 or not audit["kernel_match"].all():
            raise RuntimeError("dense-r historical CIFAR results failed overlap audit")
    return audit


def summarize_results(
    original_candidates: pd.DataFrame,
    kfda: pd.DataFrame,
    me_candidates: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    original_selected = select_historical_original(original_candidates)
    original_summary = (
        original_selected.groupby(
            ["pair", "class0", "class1", "family", "r"], as_index=False
        )["acc"]
        .agg(acc_mean="mean", acc_std="std")
        .fillna({"acc_std": 0.0})
    )
    kfda_summary = (
        kfda.groupby(["pair", "class0", "class1"], as_index=False)["acc"]
        .agg(kfda_acc="mean", kfda_std="std")
        .fillna({"kfda_std": 0.0})
    )

    valid_me = me_candidates[me_candidates["status"] == "ok"].copy()
    valid_me["kernel_order"] = valid_me["stage1_kernel"].map(KERNEL_ORDER)
    me_selected = (
        valid_me.sort_values(
            ["pair", "rep", "m", "family", "r", "acc", "kernel_order"],
            ascending=[True, True, True, True, True, False, True],
        )
        .groupby(
            ["pair", "class0", "class1", "rep", "m", "family", "r"],
            as_index=False,
        )
        .head(1)
        .drop(columns="kernel_order")
        .reset_index(drop=True)
    )
    me_summary = (
        me_selected.groupby(
            ["pair", "class0", "class1", "m", "family", "r"], as_index=False
        )["acc"]
        .agg(acc_mean="mean", acc_std="std")
        .fillna({"acc_std": 0.0})
    )

    large_original = original_summary[original_summary["family"] == "MY-large_mu"][
        ["pair", "class0", "class1", "r", "acc_mean", "acc_std"]
    ].rename(
        columns={
            "acc_mean": "original_large_acc",
            "acc_std": "original_large_std",
        }
    )
    small_original = original_summary[original_summary["family"] == "MY-small_mu"][
        ["pair", "class0", "class1", "r", "acc_mean", "acc_std"]
    ].rename(
        columns={
            "acc_mean": "original_small_acc",
            "acc_std": "original_small_std",
        }
    )
    reference = large_original.merge(
        small_original,
        on=["pair", "class0", "class1", "r"],
        validate="one_to_one",
    ).merge(
        kfda_summary,
        on=["pair", "class0", "class1"],
        validate="many_to_one",
    )
    reference["original_best_acc"] = reference[
        ["original_large_acc", "original_small_acc"]
    ].max(axis=1)
    reference["original_best_family"] = np.where(
        reference["original_large_acc"] >= reference["original_small_acc"],
        "large_mu",
        "small_mu",
    )

    large_me = me_summary[me_summary["family"] == "large_mu"][
        ["pair", "class0", "class1", "m", "r", "acc_mean", "acc_std"]
    ].rename(columns={"acc_mean": "me_large_acc", "acc_std": "me_large_std"})
    small_me = me_summary[me_summary["family"] == "small_mu"][
        ["pair", "class0", "class1", "m", "r", "acc_mean", "acc_std"]
    ].rename(columns={"acc_mean": "me_small_acc", "acc_std": "me_small_std"})
    compact = large_me.merge(
        small_me,
        on=["pair", "class0", "class1", "m", "r"],
        how="outer",
        validate="one_to_one",
    ).merge(
        reference,
        on=["pair", "class0", "class1", "r"],
        how="left",
        validate="many_to_one",
    )
    compact["me_best_acc"] = compact[["me_large_acc", "me_small_acc"]].max(axis=1)
    compact["me_best_family"] = np.where(
        compact["me_large_acc"].fillna(-np.inf)
        >= compact["me_small_acc"].fillna(-np.inf),
        "large_mu",
        "small_mu",
    )
    return original_selected, original_summary, me_selected, compact


def build_rank_support(
    me_candidates: pd.DataFrame,
    *,
    pairs: tuple[tuple[str, str], ...],
    m_values: tuple[int, ...],
) -> pd.DataFrame:
    rows = []
    for class0, class1 in pairs:
        for bag_size in m_values:
            for family in FAMILIES:
                group = me_candidates[
                    (me_candidates["class0"] == class0)
                    & (me_candidates["class1"] == class1)
                    & (me_candidates["m"] == bag_size)
                    & (me_candidates["family"] == family)
                ]
                valid = group[group["status"] == "ok"]
                rows.append(
                    {
                        "pair": pair_name(class0, class1),
                        "class0": class0,
                        "class1": class1,
                        "m": int(bag_size),
                        "family": family,
                        "valid_requested_dimensions": int(valid["r"].nunique()),
                        "largest_requested_r_supported": (
                            int(valid["r"].max()) if len(valid) else np.nan
                        ),
                        "class0_bags": (
                            int(group["class0_bags"].dropna().iloc[0])
                            if group["class0_bags"].notna().any()
                            else np.nan
                        ),
                        "class1_bags": (
                            int(group["class1_bags"].dropna().iloc[0])
                            if group["class1_bags"].notna().any()
                            else np.nan
                        ),
                    }
                )
    return pd.DataFrame(rows)


def plot_grid(
    data: pd.DataFrame,
    *,
    r_values: tuple[int, ...],
    series: tuple[tuple[str, str, str], ...],
    title: str,
    output_path: Path,
    valid_column: str | None = None,
) -> None:
    pairs = list(dict.fromkeys(data["pair"]))
    positions = np.arange(len(r_values))
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.reshape(-1)
    for axis, name in zip(axes, pairs):
        subset = data[data["pair"] == name].drop_duplicates("r").set_index("r")
        subset = subset.reindex(r_values)
        for column, label, linestyle in series:
            axis.plot(
                positions,
                subset[column],
                marker="o",
                linewidth=2.2,
                linestyle=linestyle,
                color=COLORS[label],
                label=label,
            )
        subtitle = name
        if valid_column is not None:
            subtitle += f"\n{subset[valid_column].notna().sum()}/{len(r_values)} dimensions valid"
        axis.set_title(subtitle)
        axis.set_xlabel("projection dimension r")
        axis.set_ylabel("test accuracy")
        axis.set_xticks(positions, [str(value) for value in r_values])
        axis.tick_params(axis="x", labelsize=10)
    for axis in axes[len(pairs) :]:
        axis.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=len(series),
        frameon=True,
    )
    fig.suptitle(title, y=0.998, fontsize=21)
    plt.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_outputs(
    reference: pd.DataFrame,
    compact: pd.DataFrame,
    *,
    r_values: tuple[int, ...],
    m_values: tuple[int, ...],
    output_dir: Path,
) -> None:
    plot_grid(
        reference,
        r_values=r_values,
        series=(
            ("kfda_acc", "KFDA (rank 1)", ":"),
            ("original_large_acc", "Original large-mu", "--"),
            ("original_small_acc", "Original small-mu", "--"),
            ("original_best_acc", "Original KDMLP best", "-"),
        ),
        title="Historical point-level KDMLP on selected CIFAR-100 pairs",
        output_path=output_dir / "cifar100_original_large_small_dense_r.png",
    )
    for bag_size in m_values:
        subset = compact[compact["m"] == bag_size]
        plot_grid(
            subset,
            r_values=r_values,
            series=(
                ("kfda_acc", "KFDA (rank 1)", ":"),
                ("original_best_acc", "Original KDMLP best", "--"),
                ("me_large_acc", "ME-KDMLP large-mu", "-"),
                ("me_small_acc", "ME-KDMLP small-mu", "-"),
                ("me_best_acc", "ME-KDMLP best", "-."),
            ),
            title=f"CIFAR-100 single-point test with fixed training-bag size m={bag_size}",
            output_path=output_dir / f"cifar100_me_dense_r_fixed_m{bag_size}.png",
            valid_column="me_best_acc",
        )


def build_best_summary(
    reference: pd.DataFrame,
    compact: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for (name, bag_size), group in compact.groupby(["pair", "m"]):
        valid = group[group["me_best_acc"].notna()]
        large = group[group["me_large_acc"].notna()].sort_values(
            ["me_large_acc", "r"], ascending=[False, True]
        )
        small = group[group["me_small_acc"].notna()].sort_values(
            ["me_small_acc", "r"], ascending=[False, True]
        )
        best = valid.sort_values(["me_best_acc", "r"], ascending=[False, True])
        reference_group = reference[reference["pair"] == name]
        original_best = reference_group.sort_values(
            ["original_best_acc", "r"], ascending=[False, True]
        ).iloc[0]
        rows.append(
            {
                "pair": name,
                "m": int(bag_size),
                "kfda_acc": float(reference_group["kfda_acc"].iloc[0]),
                "original_best_acc": float(original_best["original_best_acc"]),
                "original_best_r": int(original_best["r"]),
                "me_large_best_acc": float(large.iloc[0]["me_large_acc"]),
                "me_large_best_r": int(large.iloc[0]["r"]),
                "me_small_best_acc": (
                    float(small.iloc[0]["me_small_acc"]) if len(small) else np.nan
                ),
                "me_small_best_r": int(small.iloc[0]["r"]) if len(small) else np.nan,
                "me_best_acc": float(best.iloc[0]["me_best_acc"]),
                "me_best_r": int(best.iloc[0]["r"]),
                "valid_requested_dimensions": int(valid["r"].nunique()),
            }
        )
    return pd.DataFrame(rows).sort_values(["m", "pair"])


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pairs",
        type=parse_pairs,
        default=DEFAULT_PAIRS,
        help="comma-separated class0:class1 pairs",
    )
    parser.add_argument("--r-values", type=parse_int_values, default=DEFAULT_R_VALUES)
    parser.add_argument("--m-values", type=parse_int_values, default=DEFAULT_M_VALUES)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--kernels", choices=("quick", "full"), default="full")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--open-plots", action="store_true")
    return parser


def load_checkpoint(path: Path, protocol: str, resume: bool) -> pd.DataFrame:
    if not resume or not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    if "evaluation_protocol" not in frame or not frame["evaluation_protocol"].eq(protocol).all():
        return pd.DataFrame()
    return frame


def main() -> None:
    args = build_parser().parse_args()
    pairs = tuple(args.pairs)
    r_values = tuple(args.r_values)
    m_values = tuple(args.m_values)
    if args.repeats < 1:
        raise ValueError("repeats must be positive")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    original_path = output_dir / "original_candidates.csv"
    kfda_path = output_dir / "kfda_selected.csv"
    me_path = output_dir / "me_candidates.csv"
    resume = not args.no_resume

    namespace = old_cifar.load_notebook_namespace(old_cifar.NOTEBOOK_PATH)
    namespace["set_global_seed"](0)
    device = namespace["pick_device"]()
    stage1_kernels = (
        namespace["STAGE1_KERNELS_FULL"]
        if args.kernels == "full"
        else namespace["STAGE1_KERNELS_QUICK"]
    )

    original_existing = load_checkpoint(original_path, ORIGINAL_PROTOCOL, resume)
    kfda_existing = load_checkpoint(kfda_path, ORIGINAL_PROTOCOL, resume)
    me_existing = load_checkpoint(me_path, ME_PROTOCOL, resume)
    original_rows = original_existing.to_dict("records")
    kfda_rows = kfda_existing.to_dict("records")
    me_rows = me_existing.to_dict("records")
    original_done = set(
        zip(original_existing.get("class0", []), original_existing.get("class1", []), original_existing.get("rep", []))
    )
    me_done = set(
        zip(
            me_existing.get("class0", []),
            me_existing.get("class1", []),
            me_existing.get("rep", []),
            me_existing.get("m", []),
        )
    )
    setup_rows = []
    started = time.perf_counter()

    for class0, class1 in pairs:
        features, labels = load_pair(
            namespace,
            class0=class0,
            class1=class1,
            device=device,
        )
        print(f"[pair] {class0} vs {class1} features={features.shape}")
        for rep in range(args.repeats):
            split_seed = DEFAULT_SEED + 97 * rep
            train_indices, test_indices = old_cifar.split_indices(
                labels,
                train_frac=0.6,
                seed=split_seed,
            )
            features_train, labels_train = features[train_indices], labels[train_indices]
            features_test, labels_test = features[test_indices], labels[test_indices]
            print(
                f"  [split {rep}] train={len(labels_train)} test={len(labels_test)} "
                f"per-class={np.bincount(labels_train).tolist()}"
            )

            if (class0, class1, rep) in original_done:
                print("    [resume] historical point-level methods")
            else:
                step_started = time.perf_counter()
                candidates, kfda_row = evaluate_historical_split(
                    namespace,
                    features_train,
                    labels_train,
                    features_test,
                    labels_test,
                    class0=class0,
                    class1=class1,
                    rep=rep,
                    split_seed=split_seed,
                    r_values=r_values,
                    stage1_kernels=stage1_kernels,
                )
                original_rows.extend(candidates)
                kfda_rows.append(kfda_row)
                pd.DataFrame(original_rows).to_csv(original_path, index=False)
                pd.DataFrame(kfda_rows).to_csv(kfda_path, index=False)
                original_done.add((class0, class1, rep))
                print(
                    f"    [finished] historical point-level methods in "
                    f"{time.perf_counter() - step_started:.1f}s"
                )

            scaler = StandardScaler(with_mean=True, with_std=True)
            standardized_train = scaler.fit_transform(features_train)
            standardized_test = scaler.transform(features_test)
            for bag_size in m_values:
                if (class0, class1, rep, bag_size) in me_done:
                    print(f"    [resume] ME-KDMLP m={bag_size}")
                    continue
                step_started = time.perf_counter()
                me_rows.extend(
                    evaluate_me_split(
                        namespace,
                        standardized_train,
                        labels_train,
                        standardized_test,
                        labels_test,
                        class0=class0,
                        class1=class1,
                        rep=rep,
                        split_seed=split_seed,
                        bag_size=bag_size,
                        r_values=r_values,
                        stage1_kernels=stage1_kernels,
                    )
                )
                pd.DataFrame(me_rows).to_csv(me_path, index=False)
                me_done.add((class0, class1, rep, bag_size))
                print(
                    f"    [finished] ME-KDMLP m={bag_size} in "
                    f"{time.perf_counter() - step_started:.1f}s"
                )

            setup_rows.append(
                {
                    "pair": pair_name(class0, class1),
                    "class0": class0,
                    "class1": class1,
                    "rep": rep,
                    "input_feature_dimension": features.shape[1],
                    "n_train": len(labels_train),
                    "n_test": len(labels_test),
                    "n0_train": int(np.sum(labels_train == 0)),
                    "n1_train": int(np.sum(labels_train == 1)),
                    "stage2_train_unit": "original_point",
                    "test_unit": "original_point",
                }
            )

    original_candidates = pd.DataFrame(original_rows)
    kfda = pd.DataFrame(kfda_rows)
    me_candidates = pd.DataFrame(me_rows)
    original_candidates.to_csv(original_path, index=False)
    kfda.to_csv(kfda_path, index=False)
    me_candidates.to_csv(me_path, index=False)
    pd.DataFrame(setup_rows).to_csv(output_dir / "dataset_setup.csv", index=False)

    original_selected, original_summary, me_selected, compact = summarize_results(
        original_candidates,
        kfda,
        me_candidates,
    )
    reference_large = original_summary[original_summary["family"] == "MY-large_mu"][
        ["pair", "class0", "class1", "r", "acc_mean"]
    ].rename(columns={"acc_mean": "original_large_acc"})
    reference_small = original_summary[original_summary["family"] == "MY-small_mu"][
        ["pair", "class0", "class1", "r", "acc_mean"]
    ].rename(columns={"acc_mean": "original_small_acc"})
    reference = reference_large.merge(
        reference_small,
        on=["pair", "class0", "class1", "r"],
        validate="one_to_one",
    ).merge(
        kfda.groupby(["pair", "class0", "class1"], as_index=False)["acc"]
        .mean()
        .rename(columns={"acc": "kfda_acc"}),
        on=["pair", "class0", "class1"],
        validate="many_to_one",
    )
    reference["original_best_acc"] = reference[
        ["original_large_acc", "original_small_acc"]
    ].max(axis=1)

    overlap = pd.DataFrame()
    if args.kernels == "full" and set(pairs).issubset(set(DEFAULT_PAIRS)):
        overlap = audit_historical_overlap(
            original_selected,
            kfda,
            pairs=pairs,
            r_values=r_values,
        )
    rank_support = build_rank_support(
        me_candidates,
        pairs=pairs,
        m_values=m_values,
    )
    best_summary = build_best_summary(reference, compact)

    original_selected.to_csv(output_dir / "original_selected.csv", index=False)
    original_summary.to_csv(output_dir / "original_summary.csv", index=False)
    me_selected.to_csv(output_dir / "me_selected.csv", index=False)
    compact.to_csv(output_dir / "cifar100_me_dense_r_compact.csv", index=False)
    overlap.to_csv(output_dir / "historical_overlap_audit.csv", index=False)
    rank_support.to_csv(output_dir / "rank_support.csv", index=False)
    best_summary.to_csv(output_dir / "accuracy_summary.csv", index=False)
    plot_outputs(
        reference,
        compact,
        r_values=r_values,
        m_values=m_values,
        output_dir=output_dir,
    )

    print("\nHistorical overlap audit:")
    if len(overlap):
        print(f"maximum absolute error: {overlap['acc_abs_error'].max():.3e}")
        print(f"all selected kernels match: {bool(overlap['kernel_match'].all())}")
    else:
        print("skipped for non-default pairs or quick kernels")
    print("\nLargest supported requested r:")
    print(
        rank_support.pivot_table(
            index=["pair", "family"],
            columns="m",
            values="largest_requested_r_supported",
        )
    )
    print("\nBest descriptive accuracies:")
    print(best_summary.to_string(index=False))
    print(
        f"\nSaved outputs to {output_dir} in "
        f"{(time.perf_counter() - started) / 60.0:.1f} minutes"
    )
    if args.open_plots:
        open_png_outputs(output_dir)


if __name__ == "__main__":
    main()
