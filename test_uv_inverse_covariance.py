import numpy as np
import numpy.testing as npt

from my_plane import MyPlane
from photobleach_pattern import PhotobleachPattern5Z2
from uv_covariance import simulate_uv, uv_flatten
from uv_inverse_covariance import (
    compare_covariance_matrices,
    compute_inverse_jacobian,
    monte_carlo_parameter_covariance,
    propagate_observation_covariance,
    recover_theta_pattern5z2,
)


def test_inverse_jacobian_linear_map():
    rng = np.random.default_rng(0)
    B = rng.standard_normal((9, 10))
    z0 = rng.standard_normal(10)

    def recover(z: np.ndarray) -> np.ndarray:
        return B @ z

    J = compute_inverse_jacobian(z0, recover, eps=1e-6)
    npt.assert_allclose(J, B, rtol=1e-6, atol=1e-5)


def test_propagate_observation_covariance_linear():
    rng = np.random.default_rng(1)
    J = rng.standard_normal((9, 10))
    S = rng.standard_normal((10, 10))
    S = S @ S.T
    got = propagate_observation_covariance(J, S)
    expected = J @ S @ J.T
    npt.assert_allclose(got, expected, rtol=1e-12, atol=1e-12)


def test_monte_carlo_parameter_covariance_linear():
    rng = np.random.default_rng(2)
    B = rng.standard_normal((9, 10))
    z0 = rng.standard_normal(10)
    S = rng.standard_normal((10, 10))
    S = S @ S.T * 0.01

    def recover(z: np.ndarray) -> np.ndarray:
        return B @ z

    Sigma_a = B @ S @ B.T
    Sigma_mc, _ = monte_carlo_parameter_covariance(
        z0, S, recover, n_samples=50_000, rng=rng
    )
    frob, max_abs = compare_covariance_matrices(Sigma_a, Sigma_mc)
    assert frob < 0.15
    assert max_abs < 0.02


def test_pattern5z2_recovery_roundtrip_shape():
    pattern = PhotobleachPattern5Z2(
        A_x_um=0.0,
        A_z_um=0.0,
        beta1=1.0,
        beta0_um=0.0,
        B_z_um=0.0,
        C_x_um=100.0,
        C_z_um=0.0,
        d1=-1.0,
        d0_um=200.0,
        D_z_um=0.0,
        E_x_um=200.0,
        E_z_delta=0.5,
        E_z0=50.0,
    )
    plane = MyPlane(
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([0.0, 80.0, 0.0]),
    )
    theta_true = np.concatenate([plane.u, plane.v, plane.h])
    z = uv_flatten(simulate_uv(theta_true, pattern))
    theta_hat = recover_theta_pattern5z2(z, pattern)
    assert theta_hat.shape == (9,)
    assert np.all(np.isfinite(theta_hat))
