"""Tests for UV uncertainty propagation (linear validation + Monte Carlo check)."""

import numpy as np
import numpy.testing as npt

from photobleach_pattern import PhotobleachPattern
from uv_covariance import (
    compare_covariance_matrices,
    compute_jacobian,
    covariance_to_correlation,
    monte_carlo_covariance,
    propagate_covariance,
    simulate_uv,
    uv_flatten,
)


def test_propagate_covariance_linear_algebra():
    """If y = A x, Sigma_y = A Sigma_x A.T."""
    rng = np.random.default_rng(42)
    m, n = 12, 9
    A = rng.standard_normal((m, n))
    Sigma_x = rng.standard_normal((n, n))
    Sigma_x = Sigma_x @ Sigma_x.T
    Sigma_y = propagate_covariance(A, Sigma_x)
    expected = A @ Sigma_x @ A.T
    npt.assert_allclose(Sigma_y, expected, rtol=1e-12, atol=1e-12)


def test_jacobian_recovers_linear_map():
    """Central differences on y = A theta should yield J ~= A."""
    rng = np.random.default_rng(0)
    A = rng.standard_normal((6, 9))
    theta0 = rng.standard_normal(9)

    def sim(th: np.ndarray) -> np.ndarray:
        return (A @ th).reshape(3, 2)

    J = compute_jacobian(theta0, sim, eps=1e-5)
    npt.assert_allclose(J, A, rtol=1e-6, atol=1e-5)


def test_monte_carlo_matches_linear_analytical():
    rng = np.random.default_rng(1)
    A = rng.standard_normal((6, 9))
    theta0 = rng.standard_normal(9)
    Sigma_theta = rng.standard_normal((9, 9))
    Sigma_theta = Sigma_theta @ Sigma_theta.T * 0.01

    def sim(th: np.ndarray) -> np.ndarray:
        return (A @ th).reshape(3, 2)

    Sigma_analytical = propagate_covariance(A, Sigma_theta)
    Sigma_mc, _ = monte_carlo_covariance(
        theta0, Sigma_theta, sim, n_samples=50_000, rng=rng
    )
    frob, max_abs = compare_covariance_matrices(Sigma_analytical, Sigma_mc)
    assert frob < 0.15
    assert max_abs < 0.02


def test_covariance_to_correlation_identity():
    n = 5
    S = np.eye(n) * 4.0
    C = covariance_to_correlation(S)
    npt.assert_allclose(C, np.eye(n), atol=1e-10)


def test_simulate_uv_runs_with_default_pattern():
    u = np.array([1.0, 0.0, 0.0])
    v = np.array([0.0, 0.0, 1.0])
    h = np.array([0.0, 80.0, 0.0])
    theta = np.concatenate([u, v, h])
    pattern = PhotobleachPattern(
        A_x_um=0.0,
        A_z_um=0.0,
        beta1=1.0,
        beta0_um=0.0,
        B_z_um=50.0,
        C_x_um=200.0,
        C_z_um=0.0,
    )
    uv = simulate_uv(theta, pattern)
    assert uv.shape == (3, 2)
    assert np.all(np.isfinite(uv_flatten(uv)))


def test_nonlinear_mc_close_to_analytical_small_noise():
    """With small Sigma_theta, delta-method covariance ~ empirical MC."""
    rng = np.random.default_rng(7)
    u = np.array([1.0, 0.0, 0.0])
    v = np.array([0.0, 0.0, 1.0])
    h = np.array([0.0, 80.0, 0.0])
    theta = np.concatenate([u, v, h])
    pattern = PhotobleachPattern(
        A_x_um=0.0,
        A_z_um=0.0,
        beta1=1.0,
        beta0_um=0.0,
        B_z_um=50.0,
        C_x_um=200.0,
        C_z_um=0.0,
    )
    sigma = 1e-4
    Sigma_theta = (sigma**2) * np.eye(9)

    def sim(th: np.ndarray) -> np.ndarray:
        return simulate_uv(th, pattern)

    J = compute_jacobian(theta, sim, eps=1e-5)
    Sigma_a = propagate_covariance(J, Sigma_theta)
    Sigma_mc, _ = monte_carlo_covariance(
        theta, Sigma_theta, sim, n_samples=20_000, rng=rng
    )
    frob, max_abs = compare_covariance_matrices(Sigma_a, Sigma_mc)
    assert frob < 0.02
    assert max_abs < 5e-3
