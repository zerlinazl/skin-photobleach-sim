"""
Inverse uncertainty propagation: observed UV noise -> recovered plane parameter noise.

Let z = uv_flat (length 2N), and theta_hat = f(z) in R^9 where theta_hat stacks
[u_vec, v_vec, h_vec] from a chosen recovery solver.

First-order propagation (delta method):
  J_inv = d(theta_hat)/d(z)  (shape 9 x 2N)
  Sigma_theta_hat ~ J_inv Sigma_obs J_inv^T
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

from my_plane import MyPlane
from photobleach_pattern import PhotobleachPattern5Z2
from rank_lines_utils import solve_plane_from_pixels_and_pattern5z2


THETA_DIM = 9


def flatten_uv_observations(uv_points: np.ndarray) -> np.ndarray:
    """(N,2) observed UV -> flat [u1, v1, u2, v2, ...]."""
    arr = np.asarray(uv_points, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("uv_points must have shape (N, 2).")
    return arr.reshape(-1)


def unflatten_uv_observations(uv_flat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """uv_flat -> (u_points, v_points), each length N."""
    z = np.asarray(uv_flat, dtype=float).reshape(-1)
    if z.size % 2 != 0:
        raise ValueError("uv_flat must have even length 2N.")
    uv = z.reshape(-1, 2)
    return uv[:, 0].copy(), uv[:, 1].copy()


def theta_from_plane(plane: MyPlane) -> np.ndarray:
    """Pack MyPlane into theta = [u(3), v(3), h(3)]."""
    return np.concatenate([plane.u, plane.v, plane.h]).astype(float)


# def recover_theta_pattern5z2(uv_flat: np.ndarray, pattern: PhotobleachPattern5Z2) -> np.ndarray:
#     """
#     Recover theta from observed UV for pattern 5Z2 using the existing linear solve.
#     """
#     u_points, v_points = unflatten_uv_observations(uv_flat)
#     if u_points.size != 5:
#         raise ValueError("pattern5z2 recovery expects exactly 5 points (A..E).")
#     plane = solve_plane_from_pixels_and_pattern5z2(u_points, v_points, pattern)
#     return theta_from_plane(plane)

def recover_theta_pattern5splitz(uv_flat: np.ndarray, pattern: PhotobleachPattern5splitZ) -> np.ndarray:
    """
    Recover theta from observed UV for pattern 5splitZ using the existing linear solve.
    """
    u_points, v_points = unflatten_uv_observations(uv_flat)
    if u_points.size != 5:
        raise ValueError("pattern5splitZ recovery expects exactly 5 points (A..E).")
    plane = solve_plane_from_pixels_and_pattern5splitz(u_points, v_points, pattern)
    return theta_from_plane(plane)

def compute_inverse_jacobian(
    uv_flat: np.ndarray,
    recover_theta_fn: Callable[[np.ndarray], np.ndarray],
    eps: float = 1e-6,
    relative_steps: bool = True,
) -> np.ndarray:
    """
    Numerical Jacobian of inverse map: J_inv[i,j] = d(theta_hat_i)/d(uv_flat_j).

    Output shape: (9, 2N).
    """
    z = np.asarray(uv_flat, dtype=float).reshape(-1)
    theta0 = np.asarray(recover_theta_fn(z), dtype=float).reshape(-1)
    if theta0.size != THETA_DIM:
        raise ValueError(f"recover_theta_fn must return length-{THETA_DIM} theta.")

    m = z.size
    J_inv = np.zeros((THETA_DIM, m), dtype=float)

    for j in range(m):
        step = eps * max(1.0, abs(z[j])) if relative_steps else float(eps)
        if step <= 0:
            raise ValueError("Finite-difference step must be positive.")
        e = np.zeros(m, dtype=float)
        e[j] = 1.0
        t_plus = np.asarray(recover_theta_fn(z + step * e), dtype=float).reshape(-1)
        t_minus = np.asarray(recover_theta_fn(z - step * e), dtype=float).reshape(-1)
        J_inv[:, j] = (t_plus - t_minus) / (2.0 * step)

    return J_inv


def propagate_observation_covariance(J_inv: np.ndarray, Sigma_obs: np.ndarray) -> np.ndarray:
    """Sigma_theta_hat = J_inv Sigma_obs J_inv.T."""
    J = np.asarray(J_inv, dtype=float)
    S = np.asarray(Sigma_obs, dtype=float)
    if S.shape != (J.shape[1], J.shape[1]):
        raise ValueError(f"Sigma_obs must be ({J.shape[1]}, {J.shape[1]}), got {S.shape}")
    return J @ S @ J.T


def covariance_to_correlation(Sigma: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    """Normalize covariance into correlation matrix."""
    Sigma = np.asarray(Sigma, dtype=float)
    d = np.sqrt(np.maximum(np.diag(Sigma), 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        C = Sigma / np.outer(np.maximum(d, eps), np.maximum(d, eps))
    C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
    for i in range(C.shape[0]):
        C[i, i] = 1.0 if d[i] > eps else 0.0
    return C


def monte_carlo_parameter_covariance(
    uv_flat: np.ndarray,
    Sigma_obs: np.ndarray,
    recover_theta_fn: Callable[[np.ndarray], np.ndarray],
    n_samples: int = 2000,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample observed UV vectors and estimate empirical covariance of recovered theta.
    """
    rng = rng or np.random.default_rng()
    z0 = np.asarray(uv_flat, dtype=float).reshape(-1)
    S = np.asarray(Sigma_obs, dtype=float)
    z_samples = rng.multivariate_normal(z0, S, size=n_samples)
    theta_rows = [np.asarray(recover_theta_fn(zs), dtype=float).reshape(-1) for zs in z_samples]
    T = np.asarray(theta_rows, dtype=float)
    Tc = T - np.mean(T, axis=0, keepdims=True)
    if n_samples < 2:
        raise ValueError("n_samples must be at least 2.")
    Sigma_emp = (Tc.T @ Tc) / (n_samples - 1)
    return Sigma_emp, T


def compare_covariance_matrices(Sigma_a: np.ndarray, Sigma_b: np.ndarray) -> Tuple[float, float]:
    """Return (Frobenius norm diff, max abs diff)."""
    D = np.asarray(Sigma_a, dtype=float) - np.asarray(Sigma_b, dtype=float)
    return float(np.linalg.norm(D, ord="fro")), float(np.max(np.abs(D)))


def plot_theta_covariance_heatmaps(
    Sigma_theta: np.ndarray,
    Corr_theta: Optional[np.ndarray] = None,
    title_prefix: str = "",
    show_blocks: bool = True,
) -> None:
    """
    Plot covariance/correlation for recovered theta = [u(3), v(3), h(3)].
    """

    S = np.asarray(Sigma_theta, dtype=float)
    if S.shape != (THETA_DIM, THETA_DIM):
        raise ValueError(f"Sigma_theta must be ({THETA_DIM}, {THETA_DIM}).")
    C = covariance_to_correlation(S) if Corr_theta is None else np.asarray(Corr_theta, dtype=float)

    labels = ["u_x", "u_y", "u_z", "v_x", "v_y", "v_z", "h_x", "h_y", "h_z"]
    iu = np.arange(0, 3)
    iv = np.arange(3, 6)
    ih = np.arange(6, 9)

    if show_blocks:
        fig, axes = plt.subplots(2, 3, figsize=(13, 8))
        ax = axes.ravel()
    else:
        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        ax = np.asarray(axes).ravel()

    im0 = ax[0].imshow(S, cmap="viridis", aspect="auto")
    ax[0].set_title(f"{title_prefix}Sigma_theta (covariance)")
    ax[0].set_xticks(range(THETA_DIM))
    ax[0].set_yticks(range(THETA_DIM))
    ax[0].set_xticklabels(labels, rotation=45, ha="right")
    ax[0].set_yticklabels(labels)
    plt.colorbar(im0, ax=ax[0])

    im1 = ax[1].imshow(C, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    ax[1].set_title(f"{title_prefix}Correlation")
    ax[1].set_xticks(range(THETA_DIM))
    ax[1].set_yticks(range(THETA_DIM))
    ax[1].set_xticklabels(labels, rotation=45, ha="right")
    ax[1].set_yticklabels(labels)
    plt.colorbar(im1, ax=ax[1])

    if show_blocks:
        im2 = ax[2].imshow(S[np.ix_(iu, iu)], cmap="viridis", aspect="auto")
        ax[2].set_title("u-u block")
        plt.colorbar(im2, ax=ax[2])

        im3 = ax[3].imshow(S[np.ix_(iv, iv)], cmap="viridis", aspect="auto")
        ax[3].set_title("v-v block")
        plt.colorbar(im3, ax=ax[3])

        im4 = ax[4].imshow(S[np.ix_(ih, ih)], cmap="viridis", aspect="auto")
        ax[4].set_title("h-h block")
        plt.colorbar(im4, ax=ax[4])

        # Show coupling between orientation vectors u and v.
        im5 = ax[5].imshow(S[np.ix_(iu, iv)], cmap="RdBu_r", aspect="auto")
        ax[5].set_title("u-v cross block")
        plt.colorbar(im5, ax=ax[5])

    plt.tight_layout()
    plt.show()


def plot_theta_analytical_vs_mc(
    Sigma_theta_analytical: np.ndarray,
    Sigma_theta_mc: np.ndarray,
    title_prefix: str = "",
) -> None:
    """
    Side-by-side plot of analytical, Monte Carlo, and difference covariance matrices.
    """
    import matplotlib.pyplot as plt

    Sa = np.asarray(Sigma_theta_analytical, dtype=float)
    Sm = np.asarray(Sigma_theta_mc, dtype=float)
    D = Sa - Sm
    vmax = float(np.max(np.abs(D))) if np.max(np.abs(D)) > 0 else 1.0

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    im0 = axes[0].imshow(Sa, cmap="viridis", aspect="auto")
    axes[0].set_title(f"{title_prefix}Analytical")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(Sm, cmap="viridis", aspect="auto")
    axes[1].set_title(f"{title_prefix}Monte Carlo")
    plt.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(D, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    axes[2].set_title(f"{title_prefix}Difference")
    plt.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    plt.show()
