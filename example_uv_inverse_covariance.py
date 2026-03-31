#!/usr/bin/env python3
"""
Example: propagate observed UV pixel uncertainty to recovered plane parameters.
"""

import numpy as np

from my_plane import MyPlane
from photobleach_pattern5_split import PhotobleachPattern5splitZ
from uv_covariance import simulate_uv, uv_flatten
from uv_inverse_covariance import (
    compare_covariance_matrices,
    compute_inverse_jacobian,
    covariance_to_correlation,
    monte_carlo_parameter_covariance,
    plot_theta_analytical_vs_mc,
    plot_theta_covariance_heatmaps,
    propagate_observation_covariance,
    recover_theta_pattern5splitz,
)


def main() -> None:
    pattern = PhotobleachPattern5splitZ(
        A_x_um=0,
        A_z_um=0,
        beta1=1,
        beta0_um=0,
        B_z_um=0,
        B_x_start=0,
        B_x_end=100,
        C_x_um=100,
        C_z_delta=1,
        d1=-1,
        d0_um=200,
        D_z_um=0,
        D_x_start=100,
        D_x_end=200,
        E_x_um=200,
        E_z_um=50
    )

    # Ground-truth plane -> synthetic observed uv.
    plane_true = MyPlane(
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([0.0, 80.0, 0.0]),
    )
    theta_true = np.concatenate([plane_true.u, plane_true.v, plane_true.h])
    uv_obs = simulate_uv(theta_true, pattern)
    uv_flat_obs = uv_flatten(uv_obs)

    def recover_fn(z: np.ndarray) -> np.ndarray:
        return recover_theta_pattern5splitz(z, pattern)

    # Observation covariance: independent pixel noise for each u_i, v_i.
    sigma_pix = 0.05
    Sigma_obs = (sigma_pix**2) * np.eye(uv_flat_obs.size)

    J_inv = compute_inverse_jacobian(uv_flat_obs, recover_fn, eps=1e-6)
    Sigma_theta_analytical = propagate_observation_covariance(J_inv, Sigma_obs)
    Corr_theta = covariance_to_correlation(Sigma_theta_analytical)

    Sigma_theta_mc, _ = monte_carlo_parameter_covariance(
        uv_flat_obs, Sigma_obs, recover_fn, n_samples=5000, rng=np.random.default_rng(123)
    )
    frob, max_abs = compare_covariance_matrices(Sigma_theta_analytical, Sigma_theta_mc)

    print("Inverse Jacobian shape:", J_inv.shape)
    print("Recovered-parameter covariance shape:", Sigma_theta_analytical.shape)
    print("theta std (analytical):", np.sqrt(np.clip(np.diag(Sigma_theta_analytical), 0.0, None)))
    print("theta correlation (first 3x3):\n", Corr_theta[:3, :3])
    print("Analytical vs MC (Sigma_theta_hat)")
    print(f"  Frobenius norm difference: {frob:.6e}")
    print(f"  Max absolute difference:   {max_abs:.6e}")

    plot_theta_covariance_heatmaps(
        Sigma_theta_analytical,
        Corr_theta,
        title_prefix="Recovered parameters ",
        show_blocks=False,
    )
    plot_theta_analytical_vs_mc(
        Sigma_theta_analytical,
        Sigma_theta_mc,
        title_prefix="Recovered parameters ",
    )


if __name__ == "__main__":
    main()
