"""
Uncertainty propagation from plane parameters theta = [u; v; h] (9,) to pixel UV outputs.

Math (first-order / delta method):
  Let y(theta) = uv_flat in R^{2N}. For theta ~ (approximately) N(mu, Sigma_theta),
  Taylor-expand at mu: y(theta) ~ y(mu) + J (theta - mu),  J = dy/dtheta |_{mu}.
  Then  Cov(y) ~ J Sigma_theta J^T  (same formula as linear Gaussian pushforward).

  Correlation: Corr_ij = Cov_ij / sqrt(Var_i * Var_j)  (diagonal of Corr is 1 if Var_i > 0).

  Monte Carlo: sample theta_k ~ N(mu, Sigma_theta), form y_k = y(theta_k); empirical
  covariance of {y_k} estimates Cov(y) without linearizing (nonlinear truth check).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple

import numpy as np

from my_plane import MyPlane

THETA_DIM = 9


def plane_from_theta(theta: np.ndarray, *, skip_checks: bool = True) -> MyPlane:
    """
    Build MyPlane from theta = concat(u, v, h), each length 3.

    Finite-difference and Monte Carlo paths use skip_checks=True so small
    departures from the default orthogonality heuristic do not abort.
    """
    # Parameter vector: theta = [u_0,u_1,u_2, v_0,v_1,v_2, h_0,h_1,h_2]^T  in R^9.
    t = np.asarray(theta, dtype=float).reshape(-1)
    if t.shape[0] != THETA_DIM:
        raise ValueError(f"theta must have shape ({THETA_DIM},), got {t.shape}")
    u = t[0:3].copy()
    v = t[3:6].copy()
    h = t[6:9].copy()
    return MyPlane(u, v, h, skip_checks=skip_checks)


def simulate_uv(theta: np.ndarray, photobleach_pattern: Any) -> np.ndarray:
    """
    Map plane parameters to pixel coordinates for each pattern landmark.

    Parameters
    ----------
    theta : (9,) array
        Stacked u, v, h (um / um per pixel basis as in MyPlane).
    photobleach_pattern
        Any object with ``forward_model_nonparametric(plane)`` returning a tuple
        of physical points (A, B, C, ...).

    Returns
    -------
    uv : (N, 2) float array
        Rows are (u_pix, v_pix) for each landmark in forward-model order.
    """
    # Forward model: plane -> physical points A,B,... on pattern lines, then (u_pix,v_pix) = g(plane).
    # Composing with theta -> plane gives y(theta) in R^{2N} after flattening.
    plane = plane_from_theta(theta, skip_checks=True)
    raw = photobleach_pattern.forward_model_nonparametric(plane)
    if not isinstance(raw, tuple):
        raw = (raw,)
    out = []
    for pt in raw:
        pt = np.asarray(pt, dtype=float).reshape(3)
        if np.any(~np.isfinite(pt)):
            u_pix, v_pix = np.nan, np.nan
        else:
            u_pix, v_pix = plane.physical_to_pix(pt)
        out.append((u_pix, v_pix))
    return np.asarray(out, dtype=float)


def uv_flatten(uv: np.ndarray) -> np.ndarray:
    """(N, 2) -> [u1, v1, u2, v2, ...]"""
    # Fixed ordering y = [u^(1), v^(1), u^(2), v^(2), ...] so Jacobian rows match Sigma_uv indices.
    return np.asarray(uv, dtype=float).reshape(-1)


def _fd_step_abs(theta_j: float, eps: float) -> float:
    """Scale step for component j: avoids huge relative steps when |theta_j| is tiny."""
    # Step h_j = eps * max(1, |theta_j|): blends absolute and relative scaling for stable central differences.
    return eps * max(1.0, abs(theta_j))


def compute_jacobian(
    theta: np.ndarray,
    simulate_uv_fn: Callable[[np.ndarray], np.ndarray],
    eps: float = 1e-6,
    relative_steps: bool = True,
) -> np.ndarray:
    """
    Central finite-difference Jacobian J with J[i, j] = d(uv_flat)_i / d(theta_j).

    Output shape (2N, 9).
    """
    # Jacobian J in R^{(2N)x9}:  J_{ij} = partial y_i / partial theta_j  at the current theta.
    # Central difference per column j:  J_{:j} ~ (y(theta + h e_j) - y(theta - h e_j)) / (2h).
    theta = np.asarray(theta, dtype=float).reshape(-1)
    if theta.shape[0] != THETA_DIM:
        raise ValueError(f"theta must have shape ({THETA_DIM},), got {theta.shape}")

    uv0 = simulate_uv_fn(theta)
    uv0f = uv_flatten(uv0)
    m = uv0f.size
    J = np.zeros((m, THETA_DIM), dtype=float)

    for j in range(THETA_DIM):
        h = _fd_step_abs(theta[j], eps) if relative_steps else float(eps)
        if h <= 0:
            raise ValueError("Finite-difference step must be positive.")
        e = np.zeros(THETA_DIM, dtype=float)
        e[j] = 1.0
        uv_p = simulate_uv_fn(theta + h * e)
        uv_m = simulate_uv_fn(theta - h * e)
        J[:, j] = (uv_flatten(uv_p) - uv_flatten(uv_m)) / (2.0 * h)

    return J


def propagate_covariance(J: np.ndarray, Sigma_theta: np.ndarray) -> np.ndarray:
    """Sigma_uv = J Sigma_theta J.T"""
    # If theta has mean mu and Cov(theta)=Sigma_theta, linearization gives
    #   Cov(y) = Cov(J(theta-mu)) = J Sigma_theta J^T  (y in R^{2N}).
    J = np.asarray(J, dtype=float)
    S = np.asarray(Sigma_theta, dtype=float)
    if S.shape != (J.shape[1], J.shape[1]):
        raise ValueError(
            f"Sigma_theta must be ({J.shape[1]}, {J.shape[1]}), got {S.shape}"
        )
    return J @ S @ J.T


def covariance_to_correlation(Sigma: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    """
    Corr[i,j] = Sigma[i,j] / sqrt(Sigma[i,i] * Sigma[j,j]).

    Near-zero variances on the diagonal: that row/column are set to 0 off-diagonal;
    diagonal entries are 1 when variance > eps, else 0.
    """
    # Standardize:  Corr_ij = Sigma_ij / (sigma_i sigma_j),  sigma_i = sqrt(Sigma_ii).
    # Same as Corr = D^{-1} Sigma D^{-1} with D = diag(sigma_i).
    Sigma = np.asarray(Sigma, dtype=float)
    d = np.sqrt(np.maximum(np.diag(Sigma), 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        Corr = Sigma / np.outer(np.maximum(d, eps), np.maximum(d, eps))
    Corr = np.nan_to_num(Corr, nan=0.0, posinf=0.0, neginf=0.0)
    for i in range(Sigma.shape[0]):
        if d[i] > eps:
            Corr[i, i] = 1.0
        else:
            Corr[i, i] = 0.0
    return Corr


def monte_carlo_covariance(
    theta: np.ndarray,
    Sigma_theta: np.ndarray,
    simulate_uv_fn: Callable[[np.ndarray], np.ndarray],
    n_samples: int = 1000,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Draw theta_k ~ N(theta, Sigma_theta), collect uv_flat samples, return empirical covariance.

    Returns
    -------
    Sigma_uv_empirical : (2N, 2N)
    uv_samples : (n_samples, 2N)
    """
    # theta_k ~ N(mu, Sigma_theta);  y_k = y(theta_k). Unbiased sample covariance:
    #   Sigma_y ~ (1/(K-1)) sum_k (y_k - y_bar)(y_k - y_bar)^T  (K = n_samples).
    rng = rng or np.random.default_rng()
    theta = np.asarray(theta, dtype=float).reshape(-1)
    S = np.asarray(Sigma_theta, dtype=float)
    samples_theta = rng.multivariate_normal(theta, S, size=n_samples)
    rows = []
    for t in samples_theta:
        rows.append(uv_flatten(simulate_uv_fn(t)))
    U = np.asarray(rows, dtype=float)
    Uc = U - np.mean(U, axis=0, keepdims=True)
    if n_samples < 2:
        raise ValueError("n_samples must be at least 2 for empirical covariance.")
    Sigma_emp = (Uc.T @ Uc) / (n_samples - 1)
    return Sigma_emp, U


def compare_covariance_matrices(
    Sigma_a: np.ndarray, Sigma_b: np.ndarray
) -> Tuple[float, float]:
    """Frobenius norm of difference and max absolute entrywise difference."""
    # ||Sigma_a - Sigma_b||_F = sqrt(sum_ij (Delta_ij)^2);  max_abs = max_ij |Delta_ij|.
    D = Sigma_a - Sigma_b
    frob = float(np.linalg.norm(D, ord="fro"))
    max_abs = float(np.max(np.abs(D)))
    return frob, max_abs


def _block_indices(n_points: int) -> Tuple[np.ndarray, np.ndarray]:
    """u indices 0,2,... and v indices 1,3,... in flattened length-2N vector."""
    # With y = [u^(1), v^(1), ...], u-block rows/cols are even indices; v-block are odd.
    flat = np.arange(2 * n_points)
    return flat[0::2], flat[1::2]


def plot_uv_covariance_heatmaps(
    Sigma_uv: np.ndarray,
    Corr_uv: Optional[np.ndarray] = None,
    n_points: Optional[int] = None,
    title_prefix: str = "",
    show_blocks: bool = True,
) -> None:
    """
    Plot covariance and (optionally) correlation heatmaps; optional u-u, v-v, u-v blocks.

    Requires matplotlib.
    """
    # Heatmaps: full Sigma_uv and Corr; sub-blocks extract Cov(u_i,u_j), Cov(u_i,v_j) from indices iu, iv.
    import matplotlib.pyplot as plt

    Sigma_uv = np.asarray(Sigma_uv, dtype=float)
    m = Sigma_uv.shape[0]
    if n_points is None:
        if m % 2 != 0:
            raise ValueError("Sigma_uv must have even size for UV layout.")
        n_points = m // 2

    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    ax = axes.ravel()

    im0 = ax[0].imshow(Sigma_uv, cmap="viridis", aspect="auto")
    ax[0].set_title(f"{title_prefix}Sigma_uv (covariance)")
    plt.colorbar(im0, ax=ax[0])

    if Corr_uv is not None:
        C = np.asarray(Corr_uv, dtype=float)
    else:
        C = covariance_to_correlation(Sigma_uv)
    im1 = ax[1].imshow(C, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    ax[1].set_title(f"{title_prefix}Correlation")
    plt.colorbar(im1, ax=ax[1])

    if show_blocks:
        iu, iv = _block_indices(n_points)
        im2 = ax[2].imshow(Sigma_uv[np.ix_(iu, iu)], cmap="viridis", aspect="auto")
        ax[2].set_title("u-u block (covariance)")
        plt.colorbar(im2, ax=ax[2])

        im3 = ax[3].imshow(Sigma_uv[np.ix_(iu, iv)], cmap="RdBu_r", aspect="auto")
        ax[3].set_title("u-v cross block (covariance)")
        plt.colorbar(im3, ax=ax[3])
    else:
        ax[2].axis("off")
        ax[3].axis("off")

    plt.tight_layout()
    plt.show()


def plot_uv_correlation_blocks(
    Corr_uv: np.ndarray,
    n_points: int,
    title_prefix: str = "",
) -> None:
    """Separate figures for u-u, v-v, and u-v correlation blocks."""
    # Partition Corr:  u-u = C[iu,iu],  v-v = C[iv,iv],  u-v = C[iu,iv] (cross-correlation block).
    import matplotlib.pyplot as plt

    C = np.asarray(Corr_uv, dtype=float)
    iu, iv = _block_indices(n_points)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, block, name in zip(
        axes,
        (C[np.ix_(iu, iu)], C[np.ix_(iv, iv)], C[np.ix_(iu, iv)]),
        ("u-u", "v-v", "u-v"),
    ):
        im = ax.imshow(block, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
        ax.set_title(f"{title_prefix}{name}")
        plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.show()


@dataclass
class UvUncertaintyResult:
    # Bundles J, analytical Cov(y)=Sigma_uv, and Corr(y) for one (mu, Sigma_theta) choice.
    J: np.ndarray
    Sigma_uv: np.ndarray
    Corr_uv: np.ndarray


def run_uv_uncertainty_pipeline(
    theta: np.ndarray,
    photobleach_pattern: Any,
    sigma_theta: float | np.ndarray,
    *,
    jacobian_eps: float = 1e-6,
    correlation_matrix: Optional[np.ndarray] = None,
) -> UvUncertaintyResult:
    """
    Convenience: build Sigma_theta (scalar * I or full), Jacobian, Sigma_uv, Corr_uv.

    Parameters
    ----------
    sigma_theta
        If scalar, Sigma_theta = sigma_theta**2 * R (R from correlation_matrix or I).
        If array shape (9,9), used as Sigma_theta directly.
    correlation_matrix
        Optional (9,9) correlation matrix; combined with sigma if scalar scale is used
        (Sigma_theta = sigma^2 * R). If None, identity.
    """
    theta = np.asarray(theta, dtype=float).reshape(-1)

    if np.ndim(sigma_theta) == 0:
        # Scalar sigma: interpret as per-component std if R=I, so Var(theta_j)=sigma^2 on diagonal.
        var = float(sigma_theta) ** 2
        R = (
            np.eye(THETA_DIM)
            if correlation_matrix is None
            else np.asarray(correlation_matrix, dtype=float)
        )
        if R.shape != (THETA_DIM, THETA_DIM):
            raise ValueError("correlation_matrix must be (9, 9).")
        Sigma_theta = var * R
    else:
        # Full covariance on theta (must be PSD in applications; MC assumes it is valid for sampling).
        Sigma_theta = np.asarray(sigma_theta, dtype=float)
        if Sigma_theta.shape != (THETA_DIM, THETA_DIM):
            raise ValueError(f"Sigma_theta must be ({THETA_DIM}, {THETA_DIM}).")

    def sim(th: np.ndarray) -> np.ndarray:
        return simulate_uv(th, photobleach_pattern)

    # Chain: J <- finite diff of y;  Sigma_uv <- J Sigma_theta J^T;  Corr_uv <- standardize Sigma_uv.
    J = compute_jacobian(theta, sim, eps=jacobian_eps)
    Sigma_uv = propagate_covariance(J, Sigma_theta)
    Corr_uv = covariance_to_correlation(Sigma_uv)
    return UvUncertaintyResult(J=J, Sigma_uv=Sigma_uv, Corr_uv=Corr_uv)
