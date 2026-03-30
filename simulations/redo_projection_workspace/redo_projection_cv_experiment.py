from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.linalg as spla
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import KernelCenterer
from sklearn.svm import SVC


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# We keep the stage-1 kernel menu close to 1D_projection.ipynb.
STAGE1_KERNELS = [
    ("linear", {}),
    ("rbf", {"gamma": "scale"}),
    ("laplacian", {"gamma": "scale"}),
    ("polynomial", {"degree": 2, "gamma": "scale", "coef0": 1.0}),
    ("sigmoid", {"gamma": "scale", "coef0": 0.0}),
]
VARIANTS = ("KFDA-1D", "MY-large_mu", "MY-small_mu")
VARIANT_DISPLAY = {
    "KFDA-1D": "KFDA-1D",
    "MY-large_mu": "KDMLP large",
    "MY-small_mu": "KDMLP small",
}
# Default ladder used across Gaussian reruns. The effective grid is always
# capped by the original dimension p so that r remains a true reduction size.
R_VALUES = (1, 2, 3, 4, 6, 8, 10, 12, 16, 24, 32)


sns.set_theme(style="whitegrid", context="talk")
np.set_printoptions(suppress=True, precision=4)


def variant_display(variant: str) -> str:
    return VARIANT_DISPLAY.get(str(variant), str(variant))


def parse_r_values(raw: str) -> tuple[int, ...]:
    vals = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(int(part))
    vals = sorted(set(vals))
    if not vals:
        raise ValueError("At least one r value must be provided.")
    return tuple(vals)


def cap_r_values_for_dimension(r_values: tuple[int, ...], p: int) -> tuple[int, ...]:
    capped = tuple(r for r in r_values if int(r) <= int(p))
    if not capped:
        raise ValueError(f"No valid r values remain after enforcing r <= p={p}.")
    return capped


def gamma_scale_value(X_ref: np.ndarray) -> float:
    """sklearn-like gamma='scale' used in the old notebook."""
    var = float(np.var(X_ref))
    if var <= 0:
        var = 1.0
    return 1.0 / (X_ref.shape[1] * var)


def normalize_kernel_params(metric: str, params: dict, X_ref: np.ndarray) -> dict:
    params = dict(params)
    if "gamma" in params and isinstance(params["gamma"], str):
        if params["gamma"] == "scale":
            params["gamma"] = gamma_scale_value(X_ref)
        elif params["gamma"] == "auto":
            params["gamma"] = 1.0 / X_ref.shape[1]
        else:
            raise ValueError(f"Unknown gamma string: {params['gamma']}")
    return params


def kernel_label(metric: str, params: dict) -> str:
    if not params:
        return metric
    ordered = ", ".join([f"{k}={params[k]!r}" for k in sorted(params)])
    return f"{metric}({ordered})"


def symmetrize(M: np.ndarray) -> np.ndarray:
    M = np.asarray(M, dtype=np.float64)
    return 0.5 * (M + M.T)


def regularize_psd(M: np.ndarray, reg: float = 1e-6) -> np.ndarray:
    M = symmetrize(M)
    scale = float(np.trace(M) / max(1, M.shape[0]))
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    return M + float(reg) * scale * np.eye(M.shape[0], dtype=np.float64)


def solve_psd_system(M: np.ndarray, b: np.ndarray, reg: float = 1e-6) -> np.ndarray:
    M_reg = regularize_psd(M, reg=reg)
    try:
        return np.linalg.solve(M_reg, b)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(M_reg) @ b


def top_eigvec_of_invN_M(M: np.ndarray, N: np.ndarray, reg: float = 1e-6):
    """Same KFDA helper idea as in 1D_projection.ipynb, but with stabilization."""
    N_reg = regularize_psd(N, reg=reg)
    evals, evecs = spla.eigh(symmetrize(M), N_reg)
    j = int(np.argmax(evals))
    return float(evals[j]), np.asarray(evecs[:, [j]], dtype=np.float64)


def build_A_B(KX: np.ndarray, KY: np.ndarray, KXY: np.ndarray):
    """
    Paper block matrices:
      A = [[KX^2, KX KXY], [KYX KX, KYX KXY]]
      B = [[KXY KYX, KXY KY], [KYX KXY, KY^2]]
    """
    KYX = KXY.T
    A = np.block([[KX @ KX, KX @ KXY], [KYX @ KX, KYX @ KXY]])
    B = np.block([[KXY @ KYX, KXY @ KY], [KYX @ KXY, KY @ KY]])
    return symmetrize(A), symmetrize(B)


def solve_block_gep(B: np.ndarray, A: np.ndarray, reg: float = 1e-6):
    A_reg = regularize_psd(A, reg=reg)
    evals, Z = spla.eigh(symmetrize(B), A_reg)
    Z = np.asarray(Z, dtype=np.float64)
    for j in range(Z.shape[1]):
        zj = Z[:, j]
        denom = float(zj @ (A_reg @ zj))
        if np.isfinite(denom) and denom > 1e-18:
            Z[:, j] = zj / np.sqrt(denom)
    return np.asarray(evals, dtype=np.float64), Z


def random_unit_vector(rng: np.random.Generator, p: int) -> np.ndarray:
    v = rng.normal(size=p)
    return v / np.linalg.norm(v)


def random_dense_spd(rng: np.random.Generator, p: int, eig_low: float = 0.4, eig_high: float = 2.2) -> np.ndarray:
    """Dense, non-diagonal SPD matrix Q diag(lambda) Q^T."""
    Q, _ = np.linalg.qr(rng.normal(size=(p, p)))
    eigs = np.exp(rng.uniform(np.log(eig_low), np.log(eig_high), size=p))
    return symmetrize(Q @ np.diag(eigs) @ Q.T)


def dense_congruence_transform(rng: np.random.Generator, p: int, strength: float) -> np.ndarray:
    """Dense SPD transform used to change shape without collapsing rank."""
    Q, _ = np.linalg.qr(rng.normal(size=(p, p)))
    shifts = rng.uniform(-strength, strength, size=p)
    shifts = shifts - shifts.mean()
    return symmetrize(Q @ np.diag(np.exp(shifts)) @ Q.T)


def matched_trace_congruence(base_sigma: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """
    Make a different dense covariance but keep the same total variance.
    This avoids a trivial trace-only discrimination story.
    """
    sigma_new = symmetrize(transform @ base_sigma @ transform.T)
    scale = float(np.trace(base_sigma) / max(np.trace(sigma_new), 1e-12))
    return sigma_new * scale


def covariance_sqrt(Sigma: np.ndarray) -> np.ndarray:
    vals, vecs = np.linalg.eigh(symmetrize(Sigma))
    vals = np.clip(vals, 0.0, None)
    return vecs @ np.diag(np.sqrt(vals))


def sample_gaussian_antithetic(n: int, mean: np.ndarray, Sigma: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Reuses the antithetic idea from 1D_projection.ipynb:
    if n is even, empirical class mean is exactly equal to the target mean.
    """
    mean = np.asarray(mean, dtype=np.float64)
    p = mean.shape[0]
    L = covariance_sqrt(Sigma)
    if n % 2 == 0:
        half = n // 2
        Z = rng.normal(size=(half, p))
        X = Z @ L.T
        return np.vstack([mean + X, mean - X])
    half = (n - 1) // 2
    Z = rng.normal(size=(half, p))
    X = Z @ L.T
    return np.vstack([mean + X, mean - X, mean.reshape(1, -1)])


def empirical_cov(X: np.ndarray) -> np.ndarray:
    Xc = X - X.mean(axis=0, keepdims=True)
    return (Xc.T @ Xc) / max(1, X.shape[0] - 1)


def ensure_2d_features(Z):
    Z = np.asarray(Z, dtype=np.float64)
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)
    return Z


def standardize_features(Z_tr, Z_te):
    Z_tr = ensure_2d_features(Z_tr)
    Z_te = ensure_2d_features(Z_te)
    mu = Z_tr.mean(axis=0, keepdims=True)
    sd = Z_tr.std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return (Z_tr - mu) / sd, (Z_te - mu) / sd


def cv_select_svm(Z_train, y_train, n_splits=3, seed=0, max_iter=20000):
    """
    Stage-2 SVM selection reused from the old notebook: linear / rbf / poly / sigmoid
    on the projected features.
    """
    Z = ensure_2d_features(Z_train)
    y = np.asarray(y_train).astype(int)

    grid = []
    for C in (0.5, 1.0, 2.0):
        grid.append(("linear", {"C": C}))
    for C in (0.5, 1.0, 2.0):
        for gamma in ("scale", 0.5):
            grid.append(("rbf", {"C": C, "gamma": gamma}))
    for C in (0.5, 1.0, 2.0):
        grid.append(("poly", {"C": C, "gamma": "scale", "degree": 2, "coef0": 1.0}))
    for C in (0.5, 1.0, 2.0):
        grid.append(("sigmoid", {"C": C, "gamma": "scale", "coef0": 0.0}))

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    best = {"cv": -1.0, "kernel": None, "params": None}

    for kernel, params in grid:
        accs = []
        for tri, vai in skf.split(Z, y):
            Z_tr = Z[tri]
            Z_va = Z[vai]
            Z_trs, Z_vas = standardize_features(Z_tr, Z_va)
            clf = SVC(kernel=kernel, cache_size=500, max_iter=max_iter, **params)
            clf.fit(Z_trs, y[tri])
            pred = clf.predict(Z_vas)
            accs.append(accuracy_score(y[vai], pred))
        cv = float(np.mean(accs))
        if cv > best["cv"]:
            best = {"cv": cv, "kernel": kernel, "params": dict(params)}
    return best


@dataclass(frozen=True)
class Regime:
    name: str
    title: str
    p: int
    n_train_per_class: int
    n_test_per_class: int
    n_ref_per_class: int
    mean0: np.ndarray
    mean1: np.ndarray
    Sigma0: np.ndarray
    Sigma1: np.ndarray
    note: str


def build_regimes(p: int):
    rng = np.random.default_rng(20260326)

    base_same = random_dense_spd(rng, p, eig_low=0.6, eig_high=2.2)
    base_tiny = random_dense_spd(rng, p, eig_low=0.6, eig_high=2.1)
    base_strong = random_dense_spd(rng, p, eig_low=0.6, eig_high=2.4)
    base_mix = random_dense_spd(rng, p, eig_low=0.6, eig_high=2.3)

    tiny_T = dense_congruence_transform(rng, p, strength=0.12)
    strong_T = dense_congruence_transform(rng, p, strength=0.95)
    mix_T = dense_congruence_transform(rng, p, strength=0.90)

    tiny_alt = matched_trace_congruence(base_tiny, tiny_T)
    strong_alt = matched_trace_congruence(base_strong, strong_T)
    mix_alt = matched_trace_congruence(base_mix, mix_T)

    v_tiny = random_unit_vector(rng, p)
    v_mix = random_unit_vector(rng, p)

    return [
        Regime(
            name="same_mean_diff_covariance",
            title="Case A: Same Mean, Different Covariances",
            p=p,
            n_train_per_class=260,
            n_test_per_class=1200,
            n_ref_per_class=1400,
            mean0=np.zeros(p),
            mean1=np.zeros(p),
            Sigma0=base_strong,
            Sigma1=strong_alt,
            note="Pure covariance separation with non-diagonal covariances and matched trace.",
        ),
        Regime(
            name="diff_mean_diff_covariance",
            title="Case B: Different Mean, Different Covariances",
            p=p,
            n_train_per_class=220,
            n_test_per_class=1200,
            n_ref_per_class=1400,
            mean0=-0.45 * v_mix,
            mean1=0.45 * v_mix,
            Sigma0=base_mix,
            Sigma1=mix_alt,
            note="Both mean and covariance differ, with the covariance gap still matched in trace.",
        ),
        Regime(
            name="diff_mean_tiny_covariance_gap",
            title="Case C: Different Mean, Small Covariance Gap",
            p=p,
            n_train_per_class=160,
            n_test_per_class=1200,
            n_ref_per_class=1400,
            mean0=-0.55 * v_tiny,
            mean1=0.55 * v_tiny,
            Sigma0=base_tiny,
            Sigma1=tiny_alt,
            note="This is the weak-gap case: the covariance gap is real but small relative to estimation noise, so gains from larger r should be interpreted cautiously.",
        ),
    ]


def select_anchor_indices(y: np.ndarray, per_class: int, rng: np.random.Generator) -> np.ndarray:
    idx0_all = np.where(y == 0)[0]
    idx1_all = np.where(y == 1)[0]
    m = min(per_class, len(idx0_all), len(idx1_all))
    idx0 = rng.choice(idx0_all, size=m, replace=False)
    idx1 = rng.choice(idx1_all, size=m, replace=False)
    return np.concatenate([idx0, idx1])


def kfda_fit_paper_binary_inverse(Kc_aa: np.ndarray, y01_anchor: np.ndarray, reg: float = 1e-6):
    y = np.asarray(y01_anchor).astype(int)
    L = Kc_aa.shape[0]

    N = np.zeros((L, L), dtype=np.float64)
    Mc = {}
    for c in [0, 1]:
        idx = np.where(y == c)[0]
        lc = len(idx)
        Kj = Kc_aa[:, idx]
        ones = np.ones((lc, 1), dtype=np.float64)
        s = Kj @ ones
        Mc[c] = (s / lc).reshape(-1)
        N += (Kj @ Kj.T) - (1.0 / lc) * (s @ s.T)

    d = (Mc[0] - Mc[1]).reshape(L, 1)
    M = d @ d.T
    d_norm = float(np.linalg.norm(d))
    lam, a = top_eigvec_of_invN_M(M, N, reg=reg)
    return lam, a, d_norm


def kfda_project_scores(X_any, Xa, metric, params, kc: KernelCenterer, a):
    K = pairwise_kernels(X_any, Xa, metric=metric, **params)
    Kc = kc.transform(K)
    return (Kc @ a).reshape(-1)


def fit_kfda_candidate(X_anchor, y_anchor, X_fit, y_fit, X_eval, metric, params, reg: float = 1e-6):
    Kaa = pairwise_kernels(X_anchor, X_anchor, metric=metric, **params)
    kc = KernelCenterer()
    Kaa_c = kc.fit_transform(Kaa)
    top_lam, a, d_norm = kfda_fit_paper_binary_inverse(Kaa_c, y_anchor, reg=reg)
    z_fit = kfda_project_scores(X_fit, X_anchor, metric, params, kc, a)
    z_eval = kfda_project_scores(X_eval, X_anchor, metric, params, kc, a)
    return ensure_2d_features(z_fit), ensure_2d_features(z_eval), {
        "top_lambda": top_lam,
        "d_norm": d_norm,
    }


def fit_my_large_mu_candidate(X0_anchor, X1_anchor, X_fit, X_eval, metric, params, proj_dim, reg: float = 1e-6):
    l = X0_anchor.shape[0]
    m = X1_anchor.shape[0]
    KX = pairwise_kernels(X0_anchor, X0_anchor, metric=metric, **params)
    KY = pairwise_kernels(X1_anchor, X1_anchor, metric=metric, **params)
    KXY = pairwise_kernels(X0_anchor, X1_anchor, metric=metric, **params)

    ones0 = np.ones(l, dtype=np.float64)
    ones1 = np.ones(m, dtype=np.float64)
    rhs = KY @ ones1 - (float(m) / float(max(1, l))) * (KXY.T @ ones0)
    c = solve_psd_system(KY @ KY, rhs, reg=reg)

    A, B = build_A_B(KX, KY, KXY)
    thetas, Z = solve_block_gep(B, A, reg=reg)
    lambdas = (float(l) / float(max(1, m))) * thetas

    valid = np.isfinite(lambdas) & (lambdas > 1e-12)
    gvals = np.full_like(lambdas, -np.inf, dtype=np.float64)
    gvals[valid] = 0.5 * (np.log(lambdas[valid]) - 1.0 + 1.0 / lambdas[valid])
    order = np.argsort(gvals)[::-1]
    alpha = Z[:l, order]
    beta = Z[l:, order]

    kX_fit = pairwise_kernels(X0_anchor, X_fit, metric=metric, **params)
    kY_fit = pairwise_kernels(X1_anchor, X_fit, metric=metric, **params)
    kX_eval = pairwise_kernels(X0_anchor, X_eval, metric=metric, **params)
    kY_eval = pairwise_kernels(X1_anchor, X_eval, metric=metric, **params)

    s1_fit = (c @ kY_fit).reshape(-1, 1)
    s1_eval = (c @ kY_eval).reshape(-1, 1)
    eig_fit = (alpha.T @ kX_fit + beta.T @ kY_fit).T
    eig_eval = (alpha.T @ kX_eval + beta.T @ kY_eval).T

    if proj_dim <= 1:
        Z_fit = s1_fit
        Z_eval = s1_eval
    else:
        take = min(int(proj_dim) - 1, eig_fit.shape[1])
        Z_fit = np.hstack([s1_fit, eig_fit[:, :take]])
        Z_eval = np.hstack([s1_eval, eig_eval[:, :take]])

    return Z_fit, Z_eval, {
        "thetas": thetas[order][: int(max(1, proj_dim))],
        "lambdas": lambdas[order][: int(max(1, proj_dim))],
        "gvals": gvals[order][: int(max(1, proj_dim))],
    }


def fit_my_small_mu_candidate(X0_anchor, X1_anchor, X_fit, X_eval, metric, params, proj_dim, reg: float = 1e-6):
    l = X0_anchor.shape[0]
    m = X1_anchor.shape[0]
    KX = pairwise_kernels(X0_anchor, X0_anchor, metric=metric, **params)
    KY = pairwise_kernels(X1_anchor, X1_anchor, metric=metric, **params)
    KXY = pairwise_kernels(X0_anchor, X1_anchor, metric=metric, **params)

    A, B = build_A_B(KX, KY, KXY)
    thetas, Z = solve_block_gep(B, A, reg=reg)
    lambdas = (float(l) / float(max(1, m))) * thetas

    ones0 = np.ones(l, dtype=np.float64)
    ones1 = np.ones(m, dtype=np.float64)
    delta0 = (KXY @ ones1) / float(max(1, m)) - (KX @ ones0) / float(max(1, l))
    delta1 = (KY @ ones1) / float(max(1, m)) - (KXY.T @ ones0) / float(max(1, l))

    alpha_all = Z[:l, :]
    beta_all = Z[l:, :]
    mu_proj = alpha_all.T @ delta0 + beta_all.T @ delta1

    valid = np.isfinite(lambdas) & (lambdas > 1e-12)
    D_scores = np.full_like(lambdas, -np.inf, dtype=np.float64)
    D_scores[valid] = 0.5 * (
        np.log(lambdas[valid]) - 1.0 + (1.0 + np.square(mu_proj[valid])) / lambdas[valid]
    )
    order = np.argsort(D_scores)[::-1]
    alpha = alpha_all[:, order]
    beta = beta_all[:, order]

    kX_fit = pairwise_kernels(X0_anchor, X_fit, metric=metric, **params)
    kY_fit = pairwise_kernels(X1_anchor, X_fit, metric=metric, **params)
    kX_eval = pairwise_kernels(X0_anchor, X_eval, metric=metric, **params)
    kY_eval = pairwise_kernels(X1_anchor, X_eval, metric=metric, **params)

    Z_fit = (alpha.T @ kX_fit + beta.T @ kY_fit).T
    Z_eval = (alpha.T @ kX_eval + beta.T @ kY_eval).T
    take = min(int(proj_dim), Z_fit.shape[1])
    return Z_fit[:, :take], Z_eval[:, :take], {
        "thetas": thetas[order][: int(max(1, proj_dim))],
        "lambdas": lambdas[order][: int(max(1, proj_dim))],
        "D_scores": D_scores[order][: int(max(1, proj_dim))],
    }


def centered_cross_gram(Xa, Xb, metric: str, params: dict):
    na = Xa.shape[0]
    nb = Xb.shape[0]
    Ha = np.eye(na) - np.ones((na, na)) / float(max(1, na))
    Hb = np.eye(nb) - np.ones((nb, nb)) / float(max(1, nb))
    Kab = pairwise_kernels(Xa, Xb, metric=metric, **params)
    return Ha @ Kab @ Hb


def covariance_inner_hs(Xa, Xb, metric: str, params: dict) -> float:
    G = centered_cross_gram(Xa, Xb, metric, params)
    return float(np.sum(G * G) / float(max(1, Xa.shape[0] * Xb.shape[0])))


def covariance_gap_norm_feature(X0, X1, metric: str, params: dict) -> float:
    c00 = covariance_inner_hs(X0, X0, metric, params)
    c11 = covariance_inner_hs(X1, X1, metric, params)
    c01 = covariance_inner_hs(X0, X1, metric, params)
    return math.sqrt(max(c11 + c00 - 2.0 * c01, 0.0))


def covariance_operator_error_feature(X_emp, X_ref, metric: str, params: dict) -> float:
    emp = covariance_inner_hs(X_emp, X_emp, metric, params)
    ref = covariance_inner_hs(X_ref, X_ref, metric, params)
    cross = covariance_inner_hs(X_emp, X_ref, metric, params)
    return math.sqrt(max(emp + ref - 2.0 * cross, 0.0))


def covariance_gap_error_feature(X0_emp, X1_emp, X0_ref, X1_ref, metric: str, params: dict) -> float:
    emp_sq = covariance_gap_norm_feature(X0_emp, X1_emp, metric, params) ** 2
    ref_sq = covariance_gap_norm_feature(X0_ref, X1_ref, metric, params) ** 2
    cross = (
        covariance_inner_hs(X1_emp, X1_ref, metric, params)
        - covariance_inner_hs(X1_emp, X0_ref, metric, params)
        - covariance_inner_hs(X0_emp, X1_ref, metric, params)
        + covariance_inner_hs(X0_emp, X0_ref, metric, params)
    )
    return math.sqrt(max(emp_sq + ref_sq - 2.0 * cross, 0.0))


def feature_mean_gap(X0, X1, metric: str, params: dict) -> float:
    n0 = X0.shape[0]
    n1 = X1.shape[0]
    K00 = pairwise_kernels(X0, X0, metric=metric, **params)
    K11 = pairwise_kernels(X1, X1, metric=metric, **params)
    K01 = pairwise_kernels(X0, X1, metric=metric, **params)
    term = (
        float(np.sum(K11)) / float(n1 * n1)
        + float(np.sum(K00)) / float(n0 * n0)
        - 2.0 * float(np.sum(K01)) / float(n0 * n1)
    )
    return math.sqrt(max(term, 0.0))


def original_space_diagnostics(X0_train, X1_train, regime: Regime):
    C0 = empirical_cov(X0_train)
    C1 = empirical_cov(X1_train)
    delta_true = regime.Sigma1 - regime.Sigma0
    delta_emp = C1 - C0
    return {
        "true_mean_gap_input": float(np.linalg.norm(regime.mean1 - regime.mean0)),
        "emp_mean_gap_input": float(np.linalg.norm(X1_train.mean(axis=0) - X0_train.mean(axis=0))),
        "true_cov_gap_input": float(np.linalg.norm(delta_true, ord="fro")),
        "emp_cov_gap_input": float(np.linalg.norm(delta_emp, ord="fro")),
        "cov_gap_error_input": float(np.linalg.norm(delta_emp - delta_true, ord="fro")),
        "cov0_error_input": float(np.linalg.norm(C0 - regime.Sigma0, ord="fro")),
        "cov1_error_input": float(np.linalg.norm(C1 - regime.Sigma1, ord="fro")),
    }


def feature_space_diagnostics(X0_train, X1_train, X0_ref, X1_ref, metric: str, params: dict):
    return {
        "ref_D_mu_feature": feature_mean_gap(X0_ref, X1_ref, metric, params),
        "emp_D_mu_feature": feature_mean_gap(X0_train, X1_train, metric, params),
        "ref_D_sigma_feature": covariance_gap_norm_feature(X0_ref, X1_ref, metric, params),
        "emp_D_sigma_feature": covariance_gap_norm_feature(X0_train, X1_train, metric, params),
        "cov_gap_error_feature": covariance_gap_error_feature(X0_train, X1_train, X0_ref, X1_ref, metric, params),
        "cov0_error_feature": covariance_operator_error_feature(X0_train, X0_ref, metric, params),
        "cov1_error_feature": covariance_operator_error_feature(X1_train, X1_ref, metric, params),
    }


def evaluate_variant_on_validation(
    variant: str,
    proj_dim: int,
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    anchor_per_class: int,
    stage1_seed: int,
    metric: str,
    raw_params: dict,
    reg: float = 1e-6,
):
    """
    Inner validation for one method/variant/kernel/dimension candidate.
    We choose anchors only from the fit split to avoid leaking validation information.
    """
    params = normalize_kernel_params(metric, raw_params, X_fit)
    rng_anchor = np.random.default_rng(stage1_seed)
    anchor_idx = select_anchor_indices(y_fit, per_class=anchor_per_class, rng=rng_anchor)
    Xa = X_fit[anchor_idx]
    ya = y_fit[anchor_idx]
    X0_anchor = Xa[ya == 0]
    X1_anchor = Xa[ya == 1]

    if variant == "KFDA-1D":
        Z_fit, Z_val, diag = fit_kfda_candidate(Xa, ya, X_fit, y_fit, X_val, metric, params, reg=reg)
        final_r = 1
    elif variant == "MY-large_mu":
        Z_fit, Z_val, diag = fit_my_large_mu_candidate(X0_anchor, X1_anchor, X_fit, X_val, metric, params, proj_dim, reg=reg)
        final_r = proj_dim
    elif variant == "MY-small_mu":
        Z_fit, Z_val, diag = fit_my_small_mu_candidate(X0_anchor, X1_anchor, X_fit, X_val, metric, params, proj_dim, reg=reg)
        final_r = proj_dim
    else:
        raise ValueError(f"Unknown variant: {variant}")

    best_svm = cv_select_svm(Z_fit, y_fit, n_splits=3, seed=stage1_seed)
    Z_fits, Z_vals = standardize_features(Z_fit, Z_val)
    clf = SVC(kernel=best_svm["kernel"], cache_size=500, max_iter=20000, **best_svm["params"])
    clf.fit(Z_fits, y_fit)
    pred = clf.predict(Z_vals)
    val_acc = float(accuracy_score(y_val, pred))
    return {
        "variant": variant,
        "r": int(final_r),
        "stage1_kernel": metric,
        "stage1_params": params,
        "stage1_label": kernel_label(metric, params),
        "val_acc": val_acc,
        "best_svm_kernel": best_svm["kernel"],
        "best_svm_params": best_svm["params"],
        "diag": diag,
    }


def refit_variant_on_full_train(
    variant: str,
    proj_dim: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    anchor_per_class: int,
    stage1_seed: int,
    metric: str,
    raw_or_norm_params: dict,
    reg: float = 1e-6,
):
    """After inner selection, refit on the full train split and evaluate on outer test."""
    params = normalize_kernel_params(metric, raw_or_norm_params, X_train) if any(
        isinstance(v, str) for v in raw_or_norm_params.values()
    ) else dict(raw_or_norm_params)
    rng_anchor = np.random.default_rng(stage1_seed)
    anchor_idx = select_anchor_indices(y_train, per_class=anchor_per_class, rng=rng_anchor)
    Xa = X_train[anchor_idx]
    ya = y_train[anchor_idx]
    X0_anchor = Xa[ya == 0]
    X1_anchor = Xa[ya == 1]

    if variant == "KFDA-1D":
        Z_train, Z_test, diag = fit_kfda_candidate(Xa, ya, X_train, y_train, X_test, metric, params, reg=reg)
        final_r = 1
    elif variant == "MY-large_mu":
        Z_train, Z_test, diag = fit_my_large_mu_candidate(X0_anchor, X1_anchor, X_train, X_test, metric, params, proj_dim, reg=reg)
        final_r = proj_dim
    elif variant == "MY-small_mu":
        Z_train, Z_test, diag = fit_my_small_mu_candidate(X0_anchor, X1_anchor, X_train, X_test, metric, params, proj_dim, reg=reg)
        final_r = proj_dim
    else:
        raise ValueError(f"Unknown variant: {variant}")

    best_svm = cv_select_svm(Z_train, y_train, n_splits=3, seed=stage1_seed)
    Z_trs, Z_tes = standardize_features(Z_train, Z_test)
    clf = SVC(kernel=best_svm["kernel"], cache_size=500, max_iter=20000, **best_svm["params"])
    clf.fit(Z_trs, y_train)
    pred = clf.predict(Z_tes)
    test_acc = float(accuracy_score(y_test, pred))

    # Visualization uses the same selected stage-1 features sent to stage-2 SVM.
    X_show = np.vstack([X_train, X_test])
    if variant == "KFDA-1D":
        Z_show, _, _ = fit_kfda_candidate(Xa, ya, X_show, np.concatenate([y_train, y_test]), X_show, metric, params, reg=reg)
        # fit_kfda_candidate returns the first features for X_show and duplicate eval. Use the first.
    elif variant == "MY-large_mu":
        Z_show, _dummy, _ = fit_my_large_mu_candidate(X0_anchor, X1_anchor, X_show, X_show, metric, params, proj_dim, reg=reg)
    else:
        Z_show, _dummy, _ = fit_my_small_mu_candidate(X0_anchor, X1_anchor, X_show, X_show, metric, params, proj_dim, reg=reg)

    return {
        "variant": variant,
        "r": int(final_r),
        "stage1_kernel": metric,
        "stage1_params": params,
        "stage1_label": kernel_label(metric, params),
        "test_acc": test_acc,
        "best_svm_kernel": best_svm["kernel"],
        "best_svm_params": best_svm["params"],
        "diag": diag,
        "Z_train": ensure_2d_features(Z_train),
        "Z_test": ensure_2d_features(Z_test),
        "Z_show": ensure_2d_features(Z_show),
        "y_show": np.concatenate([y_train, y_test]),
    }


def project_to_plot_2d(Z: np.ndarray) -> np.ndarray:
    Z = ensure_2d_features(Z)
    if Z.shape[1] >= 2:
        if Z.shape[1] == 2:
            return Z
        return PCA(n_components=2, random_state=0).fit_transform(Z)
    jitter = np.random.default_rng(0).normal(0.0, 0.05, size=(Z.shape[0], 1))
    return np.hstack([Z, jitter])


def run_regime(regime: Regime, repeats: int = 3, anchor_per_class: int = 60, reg: float = 1e-6, seed: int = 123):
    raw_rows = []
    selection_rows = []
    cov_rows = []
    timing_rows = []
    plot_payload = {}

    rng_ref = np.random.default_rng(seed + 9999)
    X0_ref = sample_gaussian_antithetic(regime.n_ref_per_class, regime.mean0, regime.Sigma0, rng_ref)
    X1_ref = sample_gaussian_antithetic(regime.n_ref_per_class, regime.mean1, regime.Sigma1, rng_ref)

    for rep in range(repeats):
        rng = np.random.default_rng(seed + rep)
        X0_train = sample_gaussian_antithetic(regime.n_train_per_class, regime.mean0, regime.Sigma0, rng)
        X1_train = sample_gaussian_antithetic(regime.n_train_per_class, regime.mean1, regime.Sigma1, rng)
        X0_test = sample_gaussian_antithetic(regime.n_test_per_class, regime.mean0, regime.Sigma0, rng)
        X1_test = sample_gaussian_antithetic(regime.n_test_per_class, regime.mean1, regime.Sigma1, rng)

        X_train = np.vstack([X0_train, X1_train])
        y_train = np.hstack([np.zeros(len(X0_train), int), np.ones(len(X1_train), int)])
        X_test = np.vstack([X0_test, X1_test])
        y_test = np.hstack([np.zeros(len(X0_test), int), np.ones(len(X1_test), int)])

        # Inner validation split used only to choose the best stage-1 kernel for each variant/r.
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=seed + 500 + rep)
        fit_idx, val_idx = next(splitter.split(X_train, y_train))
        X_fit, y_fit = X_train[fit_idx], y_train[fit_idx]
        X_val, y_val = X_train[val_idx], y_train[val_idx]

        original_diag = original_space_diagnostics(X0_train, X1_train, regime)

        for metric, raw_params in STAGE1_KERNELS:
            params_ref = normalize_kernel_params(metric, raw_params, X_train)
            feat_diag = feature_space_diagnostics(X0_train, X1_train, X0_ref, X1_ref, metric, params_ref)
            cov_rows.append(
                {
                    "regime": regime.name,
                    "repeat": rep + 1,
                    "kernel": metric,
                    "kernel_label": kernel_label(metric, params_ref),
                    **original_diag,
                    **feat_diag,
                }
            )

        for variant in VARIANTS:
            dims = (1,) if variant == "KFDA-1D" else R_VALUES
            for r in dims:
                wall_start = time.perf_counter()
                candidates = []
                for metric, raw_params in STAGE1_KERNELS:
                    cand = evaluate_variant_on_validation(
                        variant=variant,
                        proj_dim=r,
                        X_fit=X_fit,
                        y_fit=y_fit,
                        X_val=X_val,
                        y_val=y_val,
                        anchor_per_class=anchor_per_class,
                        stage1_seed=seed + 1000 * rep + 17 * r,
                        metric=metric,
                        raw_params=raw_params,
                        reg=reg,
                    )
                    cand["regime"] = regime.name
                    cand["repeat"] = rep + 1
                    selection_rows.append(cand)
                    candidates.append(cand)

                validation_seconds = time.perf_counter() - wall_start
                refit_start = time.perf_counter()
                best_cand = max(candidates, key=lambda row: row["val_acc"])
                full = refit_variant_on_full_train(
                    variant=variant,
                    proj_dim=r,
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    anchor_per_class=anchor_per_class,
                    stage1_seed=seed + 2000 * rep + 17 * r,
                    metric=best_cand["stage1_kernel"],
                    raw_or_norm_params=best_cand["stage1_params"],
                    reg=reg,
                )
                refit_seconds = time.perf_counter() - refit_start
                total_seconds = time.perf_counter() - wall_start
                raw_rows.append(
                    {
                        "regime": regime.name,
                        "repeat": rep + 1,
                        "variant": variant,
                        "r": int(r),
                        "stage1_kernel": full["stage1_kernel"],
                        "stage1_label": full["stage1_label"],
                        "test_acc": full["test_acc"],
                        "val_acc": best_cand["val_acc"],
                        "stage2_kernel": full["best_svm_kernel"],
                        "stage2_params": json.dumps(full["best_svm_params"], sort_keys=True),
                    }
                )
                timing_rows.append(
                    {
                        "regime": regime.name,
                        "repeat": rep + 1,
                        "variant": variant,
                        "r": int(r),
                        "validation_seconds": validation_seconds,
                        "refit_seconds": refit_seconds,
                        "total_seconds": total_seconds,
                    }
                )
                if rep == 0:
                    key = (variant, int(r))
                    plot_payload[key] = {
                        "Z_show": full["Z_show"],
                        "y_show": full["y_show"],
                        "stage1_label": full["stage1_label"],
                    }

    return pd.DataFrame(raw_rows), pd.DataFrame(selection_rows), pd.DataFrame(cov_rows), pd.DataFrame(timing_rows), plot_payload


def summarize_accuracy(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = (
        df_raw.groupby(["regime", "variant", "r"], as_index=False)["test_acc"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"count": "n_repeats"})
    )
    df["std"] = df["std"].fillna(0.0)
    return df


def summarize_kernel_choices(df_raw: pd.DataFrame) -> pd.DataFrame:
    return (
        df_raw.groupby(["regime", "variant", "r", "stage1_kernel", "stage1_label"], as_index=False)
        .size()
        .rename(columns={"size": "n_selected"})
        .sort_values(["regime", "variant", "r", "n_selected"], ascending=[True, True, True, False])
    )


def summarize_covariance(df_cov: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "true_mean_gap_input",
        "emp_mean_gap_input",
        "true_cov_gap_input",
        "emp_cov_gap_input",
        "cov_gap_error_input",
        "cov0_error_input",
        "cov1_error_input",
        "ref_D_mu_feature",
        "emp_D_mu_feature",
        "ref_D_sigma_feature",
        "emp_D_sigma_feature",
        "cov_gap_error_feature",
        "cov0_error_feature",
        "cov1_error_feature",
    ]
    return df_cov.groupby(["regime", "kernel", "kernel_label"], as_index=False)[cols].mean()


def summarize_timing(df_time: pd.DataFrame) -> pd.DataFrame:
    df = (
        df_time.groupby(["regime", "variant", "r"], as_index=False)[["validation_seconds", "refit_seconds", "total_seconds"]]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    flat_cols = []
    for col in df.columns:
        if isinstance(col, tuple):
            pieces = [str(part) for part in col if part not in ("", None)]
            flat_cols.append("_".join(pieces))
        else:
            flat_cols.append(str(col))
    df.columns = flat_cols
    if "validation_seconds_count" in df.columns:
        df = df.rename(columns={"validation_seconds_count": "n_repeats"})
    drop_cols = [c for c in ("refit_seconds_count", "total_seconds_count") if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    for col in [
        "validation_seconds_mean",
        "validation_seconds_std",
        "refit_seconds_mean",
        "refit_seconds_std",
        "total_seconds_mean",
        "total_seconds_std",
    ]:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)
    return df


def build_terminal_summary(df_acc: pd.DataFrame, df_regime: pd.DataFrame) -> str:
    best_rows = {}
    for regime in df_acc["regime"].unique():
        sub = df_acc[df_acc["regime"] == regime]
        kfda = sub[sub["variant"] == "KFDA-1D"].iloc[0]
        best = sub[sub["variant"] != "KFDA-1D"].sort_values("mean", ascending=False).iloc[0]
        best_rows[regime] = (kfda, best)

    rich_regimes = [r for r in best_rows if r != "diff_mean_tiny_covariance_gap"]
    rich_wins = []
    for regime in rich_regimes:
        kfda, best = best_rows[regime]
        rich_wins.append(f"{variant_display(best['variant'])} at r={int(best['r'])} ({best['mean']:.3f} vs {kfda['mean']:.3f} for KFDA)")

    weak_regime = "diff_mean_tiny_covariance_gap"
    weak_sentence = ""
    if weak_regime in best_rows:
        kfda, best = best_rows[weak_regime]
        margin = float(best["mean"] - kfda["mean"])
        weak_sentence = (
            f"In the weak-gap regime, the covariance difference is small relative to estimation noise, "
            f"so the gain stays modest: the best KDMLP setting reaches {best['mean']:.3f} at r={int(best['r'])}, "
            f"compared with {kfda['mean']:.3f} for KFDA (margin {margin:+.3f})."
        )

    return (
        "Summary: the covariance-rich Gaussian regimes continue to benefit from larger projection dimension, "
        + "; ".join(rich_wins)
        + ". "
        + weak_sentence
    )


def build_timing_paragraph(df_timing: pd.DataFrame) -> str:
    lines = []
    for regime in df_timing["regime"].unique():
        sub = df_timing[df_timing["regime"] == regime]
        kfda = sub[sub["variant"] == "KFDA-1D"].iloc[0]
        best = sub[sub["variant"] != "KFDA-1D"].sort_values("total_seconds_mean").iloc[0]
        lines.append(
            f"{regime}: KFDA averages {kfda['total_seconds_mean']:.2f}s total, while the fastest KDMLP setting "
            f"averages {best['total_seconds_mean']:.2f}s ({variant_display(best['variant'])}, r={int(best['r'])})."
        )
    return "Wall-clock summary: " + " ".join(lines)


def open_pngs(out_dir: Path):
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


def plot_accuracy_curves(regimes, df_acc: pd.DataFrame):
    fig, axes = plt.subplots(1, len(regimes), figsize=(6 * len(regimes), 5), squeeze=False)
    axes = axes.ravel()
    colors = {"KFDA-1D": "#111827", "MY-large_mu": "#b91c1c", "MY-small_mu": "#1d4ed8"}

    for ax, regime in zip(axes, regimes):
        sub = df_acc[df_acc["regime"] == regime.name].copy()
        kfda = sub[sub["variant"] == "KFDA-1D"].iloc[0]
        ax.axhline(float(kfda["mean"]), color=colors["KFDA-1D"], linestyle="--", linewidth=2.2, label="KFDA-1D")

        for variant in ("MY-large_mu", "MY-small_mu"):
            cur = sub[sub["variant"] == variant].sort_values("r")
            ax.plot(cur["r"], cur["mean"], marker="o", linewidth=2.2, color=colors[variant], label=variant_display(variant))
            ax.fill_between(cur["r"], cur["mean"] - cur["std"], cur["mean"] + cur["std"], alpha=0.18, color=colors[variant])

        ax.set_title(regime.title)
        ax.set_xlabel("projection dimension r")
        ax.set_ylabel("test accuracy")
        ax.set_xticks(list(R_VALUES))
        ax.legend(frameon=True, fontsize=9)

    plt.tight_layout()
    out = OUT_DIR / "accuracy_curves_best_kernel.png"
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_covariance_diagnostics(regimes, df_cov_summary: pd.DataFrame):
    fig, axes = plt.subplots(len(regimes), 2, figsize=(15, 4.5 * len(regimes)))
    if len(regimes) == 1:
        axes = np.array([axes])

    kernel_order = [metric for metric, _ in STAGE1_KERNELS]
    for ridx, regime in enumerate(regimes):
        sub = df_cov_summary[df_cov_summary["regime"] == regime.name].copy()
        sub["kernel"] = pd.Categorical(sub["kernel"], categories=kernel_order, ordered=True)
        sub = sub.sort_values("kernel")
        x = np.arange(len(sub))
        width = 0.24

        ax0 = axes[ridx, 0]
        ax0.bar(x - width, sub["true_cov_gap_input"], width=width, color="#111827", label="true input gap")
        ax0.bar(x, sub["emp_cov_gap_input"], width=width, color="#6b7280", label="emp input gap")
        ax0.bar(x + width, sub["cov_gap_error_input"], width=width, color="#9ca3af", label="input error")
        ax0.set_xticks(x)
        ax0.set_xticklabels(sub["kernel"], rotation=25)
        ax0.set_title(f"{regime.title}\nOriginal-space covariance diagnostics")
        ax0.set_ylabel("Frobenius magnitude")
        ax0.legend(fontsize=8)

        ax1 = axes[ridx, 1]
        ax1.bar(x - width, sub["ref_D_sigma_feature"], width=width, color="#b91c1c", label="ref feature gap")
        ax1.bar(x, sub["emp_D_sigma_feature"], width=width, color="#f87171", label="emp feature gap")
        ax1.bar(x + width, sub["cov_gap_error_feature"], width=width, color="#6b7280", label="feature error")
        ax1.set_xticks(x)
        ax1.set_xticklabels(sub["kernel"], rotation=25)
        ax1.set_title(f"{regime.title}\nFeature-space covariance diagnostics")
        ax1.set_ylabel("HS/Frobenius magnitude")
        ax1.legend(fontsize=8)

    plt.tight_layout()
    out = OUT_DIR / "covariance_diagnostics.png"
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_projection_panels(regimes, plot_payload_map: dict, df_acc: pd.DataFrame):
    fig, axes = plt.subplots(len(regimes), 3, figsize=(16, 5 * len(regimes)))
    if len(regimes) == 1:
        axes = np.array([axes])

    for ridx, regime in enumerate(regimes):
        # pick the best r separately for large and small mu
        sub = df_acc[df_acc["regime"] == regime.name]
        best_large = int(sub[sub["variant"] == "MY-large_mu"].sort_values("mean", ascending=False).iloc[0]["r"])
        best_small = int(sub[sub["variant"] == "MY-small_mu"].sort_values("mean", ascending=False).iloc[0]["r"])
        configs = [("KFDA-1D", 1), ("MY-large_mu", best_large), ("MY-small_mu", best_small)]

        for cidx, (variant, r) in enumerate(configs):
            ax = axes[ridx, cidx]
            payload = plot_payload_map[regime.name][(variant, int(r))]
            Z2 = project_to_plot_2d(payload["Z_show"])
            y = payload["y_show"]
            ax.scatter(Z2[y == 0, 0], Z2[y == 0, 1], s=14, alpha=0.60, color="#1d4ed8", label="class 0")
            ax.scatter(Z2[y == 1, 0], Z2[y == 1, 1], s=14, alpha=0.60, color="#b91c1c", label="class 1")
            title = variant_display(variant) if variant == "KFDA-1D" else f"{variant_display(variant)}, r={r}"
            ax.set_title(f"{regime.title}\n{title}\n{payload['stage1_label']}")
            ax.set_xlabel("2D view coord 1")
            ax.set_ylabel("2D view coord 2")
            ax.legend(frameon=True, fontsize=8)

    plt.tight_layout()
    out = OUT_DIR / "selected_projection_views.png"
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def build_regime_table(regimes) -> pd.DataFrame:
    rows = []
    for regime in regimes:
        rows.append(
            {
                "regime": regime.name,
                "title": regime.title,
                "dimension": regime.p,
                "mean_gap_input": float(np.linalg.norm(regime.mean1 - regime.mean0)),
                "cov_gap_input_fro": float(np.linalg.norm(regime.Sigma1 - regime.Sigma0, ord="fro")),
                "trace_cov0": float(np.trace(regime.Sigma0)),
                "trace_cov1": float(np.trace(regime.Sigma1)),
                "note": regime.note,
            }
        )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Gaussian KDMLP vs KFDA sweep.")
    parser.add_argument("--dimension", type=int, default=120, help="Original Gaussian input dimension p.")
    parser.add_argument("--repeats", type=int, default=1, help="Number of repeated train/test splits.")
    parser.add_argument(
        "--r-values",
        type=str,
        default="1,2,3,4,6,8,10,12,16,24,32",
        help="Comma-separated KDMLP target dimensions.",
    )
    parser.add_argument(
        "--output-subdir",
        type=str,
        default="outputs",
        help="Output subdirectory under the workspace folder.",
    )
    parser.add_argument("--anchor-per-class", type=int, default=60, help="Number of anchors per class.")
    parser.add_argument("--seed", type=int, default=123, help="Base random seed for repeated runs.")
    parser.add_argument("--reg", type=float, default=1e-6, help="PSD regularization strength.")
    args = parser.parse_args()

    global OUT_DIR, R_VALUES
    OUT_DIR = ROOT / args.output_subdir
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    requested_r = parse_r_values(args.r_values)
    R_VALUES = cap_r_values_for_dimension(requested_r, args.dimension)

    regimes = build_regimes(args.dimension)
    all_raw = []
    all_sel = []
    all_cov = []
    all_time = []
    plot_payload_map = {}

    for regime in regimes:
        print("=" * 100)
        print(regime.title)
        print(regime.note)
        df_raw, df_sel, df_cov, df_time, plot_payload = run_regime(
            regime,
            repeats=args.repeats,
            anchor_per_class=args.anchor_per_class,
            reg=args.reg,
            seed=args.seed,
        )
        all_raw.append(df_raw)
        all_sel.append(df_sel)
        all_cov.append(df_cov)
        all_time.append(df_time)
        plot_payload_map[regime.name] = plot_payload

    df_raw = pd.concat(all_raw, ignore_index=True)
    df_sel = pd.concat(all_sel, ignore_index=True)
    df_cov = pd.concat(all_cov, ignore_index=True)
    df_time = pd.concat(all_time, ignore_index=True)
    df_acc = summarize_accuracy(df_raw)
    df_choice = summarize_kernel_choices(df_raw)
    df_cov_summary = summarize_covariance(df_cov)
    df_time_summary = summarize_timing(df_time)
    df_regime = build_regime_table(regimes)

    df_raw.to_csv(OUT_DIR / "test_results.csv", index=False)
    df_sel.to_csv(OUT_DIR / "selection_log.csv", index=False)
    df_acc.to_csv(OUT_DIR / "accuracy_summary.csv", index=False)
    df_choice.to_csv(OUT_DIR / "kernel_choice_summary.csv", index=False)
    df_cov.to_csv(OUT_DIR / "covariance_diagnostics_per_repeat.csv", index=False)
    df_cov_summary.to_csv(OUT_DIR / "covariance_diagnostics_summary.csv", index=False)
    df_time.to_csv(OUT_DIR / "timing_per_repeat.csv", index=False)
    df_time_summary.to_csv(OUT_DIR / "timing_summary.csv", index=False)
    df_regime.to_csv(OUT_DIR / "regime_setup.csv", index=False)

    acc_plot = plot_accuracy_curves(regimes, df_acc)
    cov_plot = plot_covariance_diagnostics(regimes, df_cov_summary)
    proj_plot = plot_projection_panels(regimes, plot_payload_map, df_acc)

    print("\nSaved outputs:")
    print(f"- {OUT_DIR / 'test_results.csv'}")
    print(f"- {OUT_DIR / 'selection_log.csv'}")
    print(f"- {OUT_DIR / 'accuracy_summary.csv'}")
    print(f"- {OUT_DIR / 'kernel_choice_summary.csv'}")
    print(f"- {OUT_DIR / 'covariance_diagnostics_summary.csv'}")
    print(f"- {OUT_DIR / 'timing_summary.csv'}")
    print(f"- {OUT_DIR / 'regime_setup.csv'}")
    print(f"- {acc_plot}")
    print(f"- {cov_plot}")
    print(f"- {proj_plot}")

    with pd.option_context("display.max_columns", None, "display.width", 260):
        print("\nRegime setup:")
        print(df_regime)
        print("\nAccuracy summary:")
        print(df_acc)
        print("\nKernel choice summary:")
        print(df_choice)
        print("\nCovariance diagnostics summary:")
        print(df_cov_summary)
        print("\nTiming summary:")
        print(df_time_summary)

    print("\nInterpretation:")
    print(build_terminal_summary(df_acc, df_regime))
    print(build_timing_paragraph(df_time_summary))
    if R_VALUES != requested_r:
        print(f"Using r grid capped by p={args.dimension}: {R_VALUES}")

    open_pngs(OUT_DIR)


if __name__ == "__main__":
    main()
