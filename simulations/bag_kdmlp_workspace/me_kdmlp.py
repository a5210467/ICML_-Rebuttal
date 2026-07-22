"""Paper-aligned mean-embedding KDMLP projection in empirical coordinates.

The implementation follows the two algorithms in the supplied manuscript:
the large-mean regime keeps one explicit mean direction and ranks the
remaining directions with ``g(gamma)``, while the small-mean regime ranks all
directions with the full one-dimensional Gaussian KLD score ``D_j``.
"""

from __future__ import annotations

import numpy as np


def _center_train_kernel(kernel: np.ndarray) -> np.ndarray:
    kernel = np.asarray(kernel, dtype=np.float64)
    row_mean = kernel.mean(axis=1, keepdims=True)
    col_mean = kernel.mean(axis=0, keepdims=True)
    return kernel - row_mean - col_mean + float(kernel.mean())


def _center_cross_kernel(
    train_kernel: np.ndarray,
    cross_kernel: np.ndarray,
) -> np.ndarray:
    train_kernel = np.asarray(train_kernel, dtype=np.float64)
    cross_kernel = np.asarray(cross_kernel, dtype=np.float64)
    return (
        cross_kernel
        - train_kernel.mean(axis=1, keepdims=True)
        - cross_kernel.mean(axis=0, keepdims=True)
        + float(train_kernel.mean())
    )


def _inverse_sqrt_spd(matrix: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh(0.5 * (matrix + matrix.T))
    floor = max(1e-14, float(np.max(values)) * 1e-12)
    values = np.maximum(values, floor)
    return (vectors * (1.0 / np.sqrt(values))) @ vectors.T


def _append_independent_direction(
    basis: list[np.ndarray],
    direction: np.ndarray,
    *,
    tolerance: float,
) -> bool:
    candidate = np.asarray(direction, dtype=np.float64).reshape(-1).copy()
    for vector in basis:
        candidate -= vector * float(vector @ candidate)
    norm = float(np.linalg.norm(candidate))
    if not np.isfinite(norm) or norm <= tolerance:
        return False
    basis.append(candidate / norm)
    return True


def _project_from_coordinate_basis(
    eigenvectors: np.ndarray,
    eigenvalues: np.ndarray,
    coordinate_basis: np.ndarray,
    bag_to_train: np.ndarray,
    bag_to_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dual_coefficients = eigenvectors @ (
        coordinate_basis / np.sqrt(eigenvalues)[:, None]
    )
    projected_train = np.asarray(
        bag_to_train.T @ dual_coefficients,
        dtype=np.float64,
    )
    projected_test = np.asarray(
        bag_to_test.T @ dual_coefficients,
        dtype=np.float64,
    )
    return dual_coefficients, projected_train, projected_test


def prepare_me_kdmlp_from_kernel_blocks(
    bag_gram_raw: np.ndarray,
    bag_to_train_raw: np.ndarray,
    bag_to_test_raw: np.ndarray,
    bag_labels: np.ndarray,
    *,
    max_r: int,
    covariance_ridge: float = 1e-6,
    center_kernel: bool = True,
    eigen_tolerance: float = 1e-10,
) -> dict:
    """Prepare nested large- and small-mean ME-KDMLP point projections.

    The rows/columns of ``bag_gram_raw`` represent the Stage-1 bags.
    ``bag_to_train_raw`` and ``bag_to_test_raw`` contain kernels from those
    same bags to original point observations. No Stage-2 bag averaging occurs.
    """

    bag_gram_raw = np.asarray(bag_gram_raw, dtype=np.float64)
    bag_to_train_raw = np.asarray(bag_to_train_raw, dtype=np.float64)
    bag_to_test_raw = np.asarray(bag_to_test_raw, dtype=np.float64)
    bag_labels = np.asarray(bag_labels, dtype=int)
    max_r = int(max_r)

    if bag_gram_raw.ndim != 2 or bag_gram_raw.shape[0] != bag_gram_raw.shape[1]:
        raise ValueError("bag_gram_raw must be square")
    bag_count = bag_gram_raw.shape[0]
    if len(bag_labels) != bag_count:
        raise ValueError("bag_labels must contain one label per bag")
    if set(np.unique(bag_labels)) != {0, 1}:
        raise ValueError("exactly two bag classes encoded as 0 and 1 are required")
    if bag_to_train_raw.shape[0] != bag_count or bag_to_test_raw.shape[0] != bag_count:
        raise ValueError("cross-kernel rows must match the bag Gram matrix")
    if max_r < 1:
        raise ValueError("max_r must be positive")
    if covariance_ridge <= 0:
        raise ValueError("covariance_ridge must be positive")

    if center_kernel:
        bag_gram = _center_train_kernel(bag_gram_raw)
        bag_to_train = _center_cross_kernel(bag_gram_raw, bag_to_train_raw)
        bag_to_test = _center_cross_kernel(bag_gram_raw, bag_to_test_raw)
    else:
        bag_gram = bag_gram_raw.copy()
        bag_to_train = bag_to_train_raw.copy()
        bag_to_test = bag_to_test_raw.copy()

    bag_gram = 0.5 * (bag_gram + bag_gram.T)
    eigenvalues_all, eigenvectors_all = np.linalg.eigh(bag_gram)
    largest = max(1.0, float(np.max(np.abs(eigenvalues_all))))
    keep = eigenvalues_all > float(eigen_tolerance) * largest
    if not np.any(keep):
        raise ValueError("the centered bag Gram matrix has no positive eigenvalue")
    eigenvalues = eigenvalues_all[keep]
    eigenvectors = eigenvectors_all[:, keep]
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    # Xi has one empirical feature coordinate in each row and one bag per column.
    coordinates = np.sqrt(eigenvalues)[:, None] * eigenvectors.T
    class_means = []
    class_covariances = []
    class_counts = []
    for class_label in (0, 1):
        class_coordinates = coordinates[:, bag_labels == class_label]
        count = class_coordinates.shape[1]
        if count < 2:
            raise ValueError("each class needs at least two disjoint bags")
        mean = class_coordinates.mean(axis=1)
        centered = class_coordinates - mean[:, None]
        covariance = centered @ centered.T / float(count)
        class_means.append(mean)
        class_covariances.append(0.5 * (covariance + covariance.T))
        class_counts.append(count)

    coordinate_rank = len(eigenvalues)
    identity = np.eye(coordinate_rank)
    covariance_1 = class_covariances[0] + float(covariance_ridge) * identity
    covariance_2 = class_covariances[1] + float(covariance_ridge) * identity
    mean_difference = class_means[1] - class_means[0]

    inverse_sqrt_1 = _inverse_sqrt_spd(covariance_1)
    whitened_covariance = inverse_sqrt_1 @ covariance_2 @ inverse_sqrt_1
    whitened_covariance = 0.5 * (whitened_covariance + whitened_covariance.T)
    gamma, whitened_vectors = np.linalg.eigh(whitened_covariance)
    gamma_floor = max(1e-14, float(np.max(np.abs(gamma))) * 1e-12)
    valid = np.isfinite(gamma) & (gamma > gamma_floor)
    gamma = gamma[valid]
    whitened_vectors = whitened_vectors[:, valid]
    if len(gamma) == 0:
        raise ValueError("the whitened covariance problem has no positive direction")

    eta = inverse_sqrt_1 @ mean_difference
    projected_mean = whitened_vectors.T @ eta
    g_scores = 0.5 * (np.log(gamma) - 1.0 + 1.0 / gamma)
    divergence_scores = 0.5 * (
        np.log(gamma)
        - 1.0
        + (1.0 + np.square(projected_mean)) / gamma
    )
    covariance_directions = inverse_sqrt_1 @ whitened_vectors

    small_order = np.argsort(divergence_scores)[::-1]
    small_count = min(max_r, len(small_order))
    small_basis = covariance_directions[:, small_order[:small_count]]
    small_dual, small_train, small_test = _project_from_coordinate_basis(
        eigenvectors,
        eigenvalues,
        small_basis,
        bag_to_train,
        bag_to_test,
    )

    mean_direction = np.linalg.solve(covariance_2, mean_difference)
    large_basis_vectors: list[np.ndarray] = []
    direction_tolerance = max(1e-12, float(np.linalg.norm(mean_direction)) * 1e-10)
    mean_direction_available = _append_independent_direction(
        large_basis_vectors,
        mean_direction,
        tolerance=direction_tolerance,
    )
    if mean_direction_available:
        for index in np.argsort(g_scores)[::-1]:
            if len(large_basis_vectors) >= max_r:
                break
            _append_independent_direction(
                large_basis_vectors,
                covariance_directions[:, int(index)],
                tolerance=1e-10,
            )

    if large_basis_vectors:
        large_basis = np.column_stack(large_basis_vectors)
        large_dual, large_train, large_test = _project_from_coordinate_basis(
            eigenvectors,
            eigenvalues,
            large_basis,
            bag_to_train,
            bag_to_test,
        )
    else:
        large_basis = np.empty((coordinate_rank, 0), dtype=np.float64)
        large_dual = np.empty((bag_count, 0), dtype=np.float64)
        large_train = np.empty((bag_to_train.shape[1], 0), dtype=np.float64)
        large_test = np.empty((bag_to_test.shape[1], 0), dtype=np.float64)

    sign_1, logdet_1 = np.linalg.slogdet(covariance_1)
    sign_2, logdet_2 = np.linalg.slogdet(covariance_2)
    if sign_1 <= 0 or sign_2 <= 0:
        raise ValueError("regularized covariance matrices must be positive definite")
    mean_term = 0.5 * float(
        mean_difference @ np.linalg.solve(covariance_2, mean_difference)
    )
    covariance_term = 0.5 * float(
        logdet_2
        - logdet_1
        - coordinate_rank
        + np.trace(np.linalg.solve(covariance_2, covariance_1))
    )

    return {
        "coordinate_rank": int(coordinate_rank),
        "class_bag_counts": tuple(int(value) for value in class_counts),
        "eigenvalues": eigenvalues,
        "gamma": gamma,
        "g_scores": g_scores,
        "D_scores": divergence_scores,
        "D_mu": mean_term,
        "D_covariance": covariance_term,
        "large_mean_direction_available": bool(mean_direction_available),
        "large_direction_count": int(large_train.shape[1]),
        "small_direction_count": int(small_train.shape[1]),
        "large_dual_coefficients": large_dual,
        "small_dual_coefficients": small_dual,
        "Ztr_large_full": large_train,
        "Zte_large_full": large_test,
        "Ztr_small_full": small_train,
        "Zte_small_full": small_test,
    }


def gaussian_regime_from_terms(
    mean_term: float,
    covariance_term: float,
    r: int,
) -> str | None:
    """Return the manuscript's heuristic regime diagnostic.

    This is a diagnostic only. The legacy OpenML figure evaluates both
    regimes and reports their better test result instead of using this rule.
    The manuscript defines the displayed inequality only for ``r > 1``;
    consequently, no heuristic family is returned at ``r = 1``.
    """

    if int(r) <= 1:
        return None
    threshold = covariance_term / float(int(r) - 1)
    return "large_mu" if mean_term >= threshold else "small_mu"


def estimate_regularized_rho3(
    projected_samples: np.ndarray,
    *,
    covariance_ridge: float = 1e-6,
) -> float:
    """Estimate the standardized third moment used by the Bentkus diagnostic.

    ``projected_samples`` must contain an independent class-specific
    alignment sample after applying a fixed projection. This coordinate-space
    calculation is algebraically identical to the Gram-matrix expression in
    the manuscript because each row is ``Theta.T @ k_X(x)``.
    """

    projected_samples = np.asarray(projected_samples, dtype=np.float64)
    if projected_samples.ndim == 1:
        projected_samples = projected_samples[:, None]
    if projected_samples.ndim != 2 or projected_samples.shape[0] < 2:
        raise ValueError("projected_samples must contain at least two rows")
    if projected_samples.shape[1] < 1:
        raise ValueError("projected_samples must contain at least one direction")
    if covariance_ridge <= 0:
        raise ValueError("covariance_ridge must be positive")

    centered = projected_samples - projected_samples.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / float(len(centered))
    covariance = 0.5 * (covariance + covariance.T)
    covariance += float(covariance_ridge) * np.eye(covariance.shape[0])
    squared_mahalanobis = np.einsum(
        "ij,ji->i",
        centered,
        np.linalg.solve(covariance, centered.T),
    )
    squared_mahalanobis = np.maximum(squared_mahalanobis, 0.0)
    return float(np.mean(np.power(squared_mahalanobis, 1.5)))


def bentkus_gaussianity_diagnostic(
    projected_alignment: np.ndarray,
    labels_alignment: np.ndarray,
    *,
    bag_size: int,
    covariance_ridge: float = 1e-6,
    class_priors: tuple[float, float] | None = None,
) -> dict[str, float]:
    """Compute classwise and prior-weighted constant-free Bentkus quantities.

    The caller is responsible for keeping ``projected_alignment`` independent
    of projection fitting and final test evaluation. The returned values are
    regularized plug-in diagnostics, not finite-sample confidence bounds.
    """

    projected_alignment = np.asarray(projected_alignment, dtype=np.float64)
    labels_alignment = np.asarray(labels_alignment, dtype=int)
    if projected_alignment.ndim == 1:
        projected_alignment = projected_alignment[:, None]
    if projected_alignment.ndim != 2 or len(projected_alignment) != len(labels_alignment):
        raise ValueError("projected_alignment and labels_alignment must align by row")
    if set(np.unique(labels_alignment)) != {0, 1}:
        raise ValueError("exactly two alignment classes encoded as 0 and 1 are required")
    if int(bag_size) < 1:
        raise ValueError("bag_size must be positive")

    if class_priors is None:
        priors = np.array(
            [np.mean(labels_alignment == label) for label in (0, 1)],
            dtype=np.float64,
        )
    else:
        priors = np.asarray(class_priors, dtype=np.float64)
        if priors.shape != (2,) or np.any(priors < 0) or not np.isclose(priors.sum(), 1.0):
            raise ValueError("class_priors must be two nonnegative values summing to one")

    rho3 = np.array(
        [
            estimate_regularized_rho3(
                projected_alignment[labels_alignment == label],
                covariance_ridge=covariance_ridge,
            )
            for label in (0, 1)
        ],
        dtype=np.float64,
    )
    scale = float(projected_alignment.shape[1]) ** 0.25 / np.sqrt(float(bag_size))
    classwise = scale * rho3
    return {
        "rho3_class0": float(rho3[0]),
        "rho3_class1": float(rho3[1]),
        "bentkus_class0": float(classwise[0]),
        "bentkus_class1": float(classwise[1]),
        "bentkus_weighted": float(priors @ classwise),
    }
