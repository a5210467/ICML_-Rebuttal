"""Bag utilities that preserve the paper-style KDMLP algebra.

The original OpenML experiment represents one observation by ``phi(x)``.
Here a bag is represented by its empirical RKHS mean

    mean_phi(B) = (1 / m) * sum_{x in B} phi(x).

These mean embeddings define the Stage-1 anchor geometry. Original training
and held-out points can then be projected through a bag-to-point cross kernel,
which averages only over the anchor bag.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy import sparse


@dataclass(frozen=True)
class HomogeneousBags:
    """A collection of class-homogeneous bootstrap bags.

    ``weights[a, i]`` is the empirical frequency of source observation ``i``
    in bag ``a``. Rows sum to one, so multiplying ``weights`` by pointwise
    projected coordinates averages those coordinates within each bag.
    ``labels`` contains one label per bag and ``source_labels`` contains one
    label per original observation.
    """

    weights: sparse.csr_matrix
    members: np.ndarray
    labels: np.ndarray
    source_labels: np.ndarray

    @property
    def bag_size(self) -> int:
        return int(self.members.shape[1])


def make_class_homogeneous_bags(
    labels: np.ndarray,
    bag_size: int,
    *,
    seed: int,
) -> HomogeneousBags:
    """Make one bag per source observation without crossing class boundaries.

    For ``bag_size=1`` the membership matrix is exactly the identity. For a
    larger bag, members are sampled with replacement from the same class and
    only from the supplied train or test pool. Keeping the number and order of
    bags equal to the source observations makes the singleton case an exact
    numerical check against the original OpenML experiment.
    """

    labels = np.asarray(labels, dtype=int)
    if labels.ndim != 1 or len(labels) == 0:
        raise ValueError("labels must be a nonempty one-dimensional array")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("exactly two classes encoded as 0 and 1 are required")
    if int(bag_size) < 1:
        raise ValueError("bag_size must be positive")

    bag_size = int(bag_size)
    n = len(labels)
    if bag_size == 1:
        members = np.arange(n, dtype=int).reshape(-1, 1)
    else:
        rng = np.random.default_rng(seed)
        members = np.empty((n, bag_size), dtype=int)
        for class_label in (0, 1):
            rows = np.flatnonzero(labels == class_label)
            members[rows] = rng.choice(
                rows,
                size=(len(rows), bag_size),
                replace=True,
            )

    rows = np.repeat(np.arange(n, dtype=int), bag_size)
    cols = members.reshape(-1)
    values = np.full(n * bag_size, 1.0 / bag_size, dtype=np.float64)
    weights = sparse.coo_matrix((values, (rows, cols)), shape=(n, n)).tocsr()
    return HomogeneousBags(
        weights=weights,
        members=members,
        labels=labels.copy(),
        source_labels=labels.copy(),
    )


def make_disjoint_class_bags(
    labels: np.ndarray,
    bag_size: int,
    *,
    seed: int,
) -> HomogeneousBags:
    """Partition each class into disjoint equal-size bags.

    A class contributes ``floor(n_c / bag_size)`` bags. Its shuffled leftover
    observations are not used to learn the bag-level Stage-1 geometry, but all
    original observations remain available to the pointwise Stage-2 model.
    For ``bag_size=1`` the identity ordering is retained as a direct kernel
    sanity check.
    """

    labels = np.asarray(labels, dtype=int)
    if labels.ndim != 1 or len(labels) == 0:
        raise ValueError("labels must be a nonempty one-dimensional array")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("exactly two classes encoded as 0 and 1 are required")
    if int(bag_size) < 1:
        raise ValueError("bag_size must be positive")

    bag_size = int(bag_size)
    n = len(labels)
    if bag_size == 1:
        members = np.arange(n, dtype=int).reshape(-1, 1)
        bag_labels = labels.copy()
    else:
        rng = np.random.default_rng(seed)
        class_members = []
        class_labels = []
        for class_label in (0, 1):
            indices = np.flatnonzero(labels == class_label).copy()
            rng.shuffle(indices)
            bag_count = len(indices) // bag_size
            if bag_count == 0:
                raise ValueError(
                    f"class {class_label} has fewer observations than bag_size={bag_size}"
                )
            used = indices[: bag_count * bag_size]
            class_members.append(used.reshape(bag_count, bag_size))
            class_labels.append(np.full(bag_count, class_label, dtype=int))
        members = np.vstack(class_members)
        bag_labels = np.concatenate(class_labels)

    rows = np.repeat(np.arange(len(members), dtype=int), bag_size)
    cols = members.reshape(-1)
    values = np.full(len(rows), 1.0 / bag_size, dtype=np.float64)
    weights = sparse.coo_matrix(
        (values, (rows, cols)),
        shape=(len(members), n),
    ).tocsr()
    return HomogeneousBags(
        weights=weights,
        members=members,
        labels=bag_labels,
        source_labels=labels.copy(),
    )


def make_singleton_observations(labels: np.ndarray) -> HomogeneousBags:
    """Represent every observation by itself, independently of its label.

    Labels are retained only so the common structural validator can be used;
    they do not affect this identity membership matrix.
    """

    labels = np.asarray(labels, dtype=int)
    if labels.ndim != 1 or len(labels) == 0:
        raise ValueError("labels must be a nonempty one-dimensional array")
    n = len(labels)
    members = np.arange(n, dtype=int).reshape(-1, 1)
    weights = sparse.eye(n, format="csr", dtype=np.float64)
    return HomogeneousBags(
        weights=weights,
        members=members,
        labels=labels.copy(),
        source_labels=labels.copy(),
    )


def validate_homogeneous_bags(bags: HomogeneousBags) -> None:
    """Raise if a bag contains a source observation from the other class."""

    member_labels = bags.source_labels[bags.members]
    if not np.all(member_labels == bags.labels[:, None]):
        raise ValueError("a bag contains members from more than one class")
    row_sums = np.asarray(bags.weights.sum(axis=1)).reshape(-1)
    if not np.allclose(row_sums, 1.0, atol=1e-12, rtol=0.0):
        raise ValueError("bag membership weights must sum to one")


def balanced_anchor_indices(
    labels: np.ndarray,
    anchors_per_class: int,
    *,
    seed: int,
) -> tuple[np.ndarray, int]:
    """Use the same balanced random-anchor rule as the source evaluator."""

    labels = np.asarray(labels, dtype=int)
    idx0_all = np.flatnonzero(labels == 0)
    idx1_all = np.flatnonzero(labels == 1)
    safe = min(int(anchors_per_class), len(idx0_all), len(idx1_all))
    if safe < 2:
        raise ValueError("each class needs at least two training bags")
    rng = np.random.default_rng(seed)
    idx0 = rng.choice(idx0_all, safe, replace=False)
    idx1 = rng.choice(idx1_all, safe, replace=False)
    return np.concatenate([idx0, idx1]), safe


def kernel_params_for_bags(
    namespace: dict,
    metric: str,
    raw_params: dict,
    standardized_train: np.ndarray,
    train_bags: HomogeneousBags,
    anchor_indices: np.ndarray,
) -> dict:
    """Resolve ``gamma='scale'`` from members of the selected anchor bags.

    With singleton bags this reference array is exactly the point-anchor array
    used by the original implementation.
    """

    member_indices = train_bags.members[np.asarray(anchor_indices, dtype=int)].reshape(-1)
    reference = standardized_train[member_indices]
    return namespace["normalize_kernel_params"](metric, raw_params, reference)


def bag_kernel_blocks(
    standardized_train: np.ndarray,
    standardized_test: np.ndarray,
    train_bags: HomogeneousBags,
    test_units: HomogeneousBags,
    anchor_indices: np.ndarray,
    *,
    metric: str,
    params: dict,
    kernel_function: Callable,
    project_training_points: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return raw kernels ``K(A,A)``, ``K(A,tr)``, and ``K(A,te)``.

    Only source observations appearing in an anchor bag are used on the left
    side of a point-kernel evaluation. This computes the exact empirical
    average while avoiding a dense all-pairs expansion for large ``m``.
    ``test_units`` may contain mean bags or singleton original observations.
    If ``project_training_points`` is true, the returned training cross-kernel
    maps the original training points rather than the mean training bags. The
    anchor Gram matrix remains a bag-to-bag matrix in either case.
    """

    anchor_indices = np.asarray(anchor_indices, dtype=int)
    support = np.unique(train_bags.members[anchor_indices].reshape(-1))
    anchor_weights = train_bags.weights[anchor_indices][:, support]

    support_to_train = kernel_function(
        standardized_train[support],
        standardized_train,
        metric=metric,
        **params,
    )
    support_to_test = kernel_function(
        standardized_train[support],
        standardized_test,
        metric=metric,
        **params,
    )

    anchor_to_source_train = np.asarray(anchor_weights @ support_to_train)
    anchor_to_source_test = np.asarray(anchor_weights @ support_to_test)
    anchor_to_train_bags = np.asarray(
        (train_bags.weights @ anchor_to_source_train.T).T,
        dtype=np.float64,
    )
    anchor_to_test = np.asarray(
        (test_units.weights @ anchor_to_source_test.T).T,
        dtype=np.float64,
    )
    anchor_gram = anchor_to_train_bags[:, anchor_indices]
    anchor_gram = 0.5 * (anchor_gram + anchor_gram.T)
    anchor_to_train = (
        anchor_to_source_train
        if project_training_points
        else anchor_to_train_bags
    )
    return anchor_gram, anchor_to_train, anchor_to_test


def paper_precompute_from_kernel_blocks(
    namespace: dict,
    anchor_gram_raw: np.ndarray,
    anchor_to_train_raw: np.ndarray,
    anchor_to_test_raw: np.ndarray,
    *,
    anchors_per_class: int,
    max_r: int,
    algorithm: str,
    center_kernel: bool = True,
    reg: float = 1e-6,
) -> dict:
    """Run the source KDMLP block algebra using precomputed bag kernels.

    This is a kernel-input version of ``_paper_prepare_my_precompute`` from
    ``Modern_Benchmarks_KFDA_vs_Our_method_multidim.ipynb``. The equations,
    direction ordering, centering, and regularization are unchanged.
    """

    l = int(anchors_per_class)
    m = int(anchors_per_class)
    one_l = np.ones(l, dtype=np.float64)
    one_m = np.ones(m, dtype=np.float64)

    if center_kernel:
        kernel_basis = namespace["_center_train_kernel"](anchor_gram_raw)
        kernel_train = namespace["_center_cross_kernel"](
            anchor_gram_raw,
            anchor_to_train_raw,
        )
        kernel_test = namespace["_center_cross_kernel"](
            anchor_gram_raw,
            anchor_to_test_raw,
        )
    else:
        kernel_basis = np.asarray(anchor_gram_raw, dtype=np.float64).copy()
        kernel_train = np.asarray(anchor_to_train_raw, dtype=np.float64).copy()
        kernel_test = np.asarray(anchor_to_test_raw, dtype=np.float64).copy()

    kernel_x = kernel_basis[:l, :l]
    kernel_xy = kernel_basis[:l, l:]
    kernel_y = kernel_basis[l:, l:]
    matrix_a, matrix_b = namespace["_paper_build_block_matrices"](
        kernel_x,
        kernel_y,
        kernel_xy,
    )
    thetas, eigenvectors, matrix_a_reg = namespace["_paper_generalized_eig"](
        matrix_b,
        matrix_a,
        reg=reg,
    )
    lambdas = (float(l) / float(max(1, m))) * thetas

    valid = np.isfinite(lambdas) & (lambdas > 1e-12)
    g_values = np.full_like(lambdas, -np.inf, dtype=np.float64)
    g_values[valid] = 0.5 * (
        np.log(lambdas[valid]) - 1.0 + 1.0 / lambdas[valid]
    )

    delta_x = (kernel_xy @ one_m) / m - (kernel_x @ one_l) / l
    delta_y = (kernel_y @ one_m) / m - (kernel_xy.T @ one_l) / l
    alpha_all = eigenvectors[:l, :]
    beta_all = eigenvectors[l:, :]
    projected_mean = alpha_all.T @ delta_x + beta_all.T @ delta_y

    divergence_scores = np.full_like(lambdas, -np.inf, dtype=np.float64)
    divergence_scores[valid] = 0.5 * (
        np.log(lambdas[valid])
        - 1.0
        + (1.0 + np.square(projected_mean[valid])) / lambdas[valid]
    )
    order_g = [
        int(i)
        for i in np.argsort(g_values)[::-1]
        if np.isfinite(g_values[int(i)])
    ]
    order_divergence = [
        int(i)
        for i in np.argsort(divergence_scores)[::-1]
        if np.isfinite(divergence_scores[int(i)])
    ]

    kx_train = kernel_train[:l, :]
    ky_train = kernel_train[l:, :]
    kx_test = kernel_test[:l, :]
    ky_test = kernel_test[l:, :]
    eigen_train = np.asarray(
        kx_train.T @ alpha_all + ky_train.T @ beta_all,
        dtype=np.float64,
    )
    eigen_test = np.asarray(
        kx_test.T @ alpha_all + ky_test.T @ beta_all,
        dtype=np.float64,
    )

    rhs = (kernel_y @ one_m) - (float(m) / float(l)) * (kernel_xy.T @ one_l)
    coefficients = np.linalg.pinv(
        kernel_y @ kernel_y + float(reg) * np.eye(m)
    ) @ rhs
    stage1_train = np.asarray(ky_train.T @ coefficients[:, None], dtype=np.float64)
    stage1_test = np.asarray(ky_test.T @ coefficients[:, None], dtype=np.float64)

    max_r = int(max_r)
    if algorithm == "large_mu":
        selected = order_g[: max(0, max_r - 1)]
        train_columns = [stage1_train]
        test_columns = [stage1_test]
        if selected:
            train_columns.append(eigen_train[:, selected])
            test_columns.append(eigen_test[:, selected])
        projected_train = np.hstack(train_columns)
        projected_test = np.hstack(test_columns)
    elif algorithm == "small_mu":
        selected = order_divergence[:max_r]
        if not selected:
            raise ValueError("no valid KDMLP covariance direction was found")
        projected_train = eigen_train[:, selected]
        projected_test = eigen_test[:, selected]
    else:
        raise ValueError("algorithm must be 'large_mu' or 'small_mu'")

    return {
        "thetas": thetas,
        "lambdas": lambdas,
        "gvals": g_values,
        "D_scores": divergence_scores,
        "A_block": matrix_a,
        "A_reg": matrix_a_reg,
        "B_block": matrix_b,
        "Ztr_full": np.asarray(projected_train, dtype=np.float64),
        "Zte_full": np.asarray(projected_test, dtype=np.float64),
    }
