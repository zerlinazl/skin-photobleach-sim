#!/usr/bin/env python3
"""
Example: propagate uncertainty in plane parameters (u, v, h) to pixel UV coordinates.

Uses a 3-point PhotobleachPattern (A, B, C). Adjust ``sigma_theta`` and
``Sigma_theta`` for your noise model (independent vs full covariance).
"""

import numpy as np

from photobleach_pattern import PhotobleachPattern, PhotobleachPattern3Z
from photobleach_pattern_4 import PhotobleachPattern4a, PhotobleachPattern4splitZ1
from photobleach_pattern_5_split import PhotobleachPattern5split, PhotobleachPattern5splitZ

from uv_covariance import (
    compare_covariance_matrices,
    compute_jacobian,
    covariance_to_correlation,
    monte_carlo_covariance,
    plot_uv_correlation_blocks,
    plot_uv_covariance_heatmaps,
    propagate_covariance,
    run_uv_uncertainty_pipeline,
    simulate_uv,
    uv_flatten,
)

def run_analysis(pattern, n_points):
    u = np.array([1.0, 0.0, 0.0])
    v = np.array([0.0, 0.0, 1.0])
    h = np.array([0.0, 80.0, 0.0])
    theta = np.concatenate([u, v, h])


    uv = simulate_uv(theta, pattern)
    print("UV (N x 2):\n", uv)
    print("uv_flat:", uv_flatten(uv))

    # --- Delta method: build J, then Sigma_uv = J Sigma_theta J.T with Sigma_theta = sigma_theta^2 I (first block). ---
    sigma_theta = 1e-3
    result = run_uv_uncertainty_pipeline(
        theta, pattern, sigma_theta, jacobian_eps=1e-5
    )
    print("Jacobian shape:", result.J.shape)
    print("Sigma_uv shape:", result.Sigma_uv.shape)

    N = uv.shape[0]
    u_idx = np.arange(0, 2*N, 2)
    v_idx = np.arange(1, 2*N, 2)

    Sigma_cross = result.Sigma_uv[np.ix_(u_idx, v_idx)]
    print("max |Cov(u,v)| =", np.max(np.abs(Sigma_cross)))


    # N = uv.shape[0]               # number of landmarks, e.g. 3
    # v_idx = np.arange(1, 2*N, 2) # v indices for uv_flat ordering

    # print("v diag:", np.diag(result.Sigma_uv)[v_idx])
    # # print("sigma_theta diag:", np.diag(result.Sigma_theta))
    # print("J v row energies:", np.sum(result.J[v_idx, :]**2, axis=1))
    # print("any nonzero J rows for v?", np.any(result.J[v_idx, :] != 0))

    # --- Same sigma, but Sigma_theta = sigma^2 * R (R has off-diagonal correlation between theta_0 and theta_3). ---
    R = np.eye(9)
    R[0, 3] = R[3, 0] = 0.3
    result2 = run_uv_uncertainty_pipeline(
        theta, pattern, sigma_theta, correlation_matrix=R, jacobian_eps=1e-5
    )
    print("Correlation (full model) diagonal sample:", np.diag(result2.Corr_uv)[:4])

    # --- Compare linearized Sigma_uv to empirical covariance from theta ~ N(mu, Sigma_theta) (nonlinear check). ---
    Sigma_theta = (sigma_theta**2) * np.eye(9)

    def sim(th: np.ndarray) -> np.ndarray:
        return simulate_uv(th, pattern)

    Sigma_a = propagate_covariance(
        compute_jacobian(theta, sim, eps=1e-5),
        Sigma_theta,
    )
    Sigma_mc, _ = monte_carlo_covariance(
        theta, Sigma_theta, sim, n_samples=8000, rng=np.random.default_rng(123)
    )
    frob, max_abs = compare_covariance_matrices(Sigma_a, Sigma_mc)
    print("\nAnalytical vs Monte Carlo (Sigma_uv):")
    print(f"  Frobenius norm of difference: {frob:.6e}")
    print(f"  Max absolute difference:       {max_abs:.6e}")

    # --- Visualize full matrix and u/u, u/v blocks (indices follow uv_flat order). ---
    plot_uv_covariance_heatmaps(
        Sigma_a,
        covariance_to_correlation(Sigma_a),
        n_points=n_points,
        title_prefix="Analytical ",
        show_blocks=False,
    )
    # plot_uv_correlation_blocks(
    #     covariance_to_correlation(Sigma_a), n_points=3, title_prefix=""
    # )

def main():
    pattern_3_1 = PhotobleachPattern(
        A_x_um=0.0,
        A_z_um=0.0,
        beta1=1.0,
        beta0_um=0.0,
        B_z_um=50.0,
        C_x_um=200.0,
        C_z_um=0.0,
    )
    # run_analysis(pattern_3_1, 3)

    pattern_4_a = PhotobleachPattern4a(
        A_x_um=0.0,
        A_z_um=0.0,
        beta1=1.0,
        beta0_um=0.0,
        B_z_um=50.0,
        C_x_um=200.0,
        C_z_um=0.0,
        d1=-1.0,
        d0_um=-200.0,
        D_z_um=30.0,
    )
    # run_analysis(pattern_4_a, 4)

    
    pattern_5_split_1 = PhotobleachPattern5split(
        A_x_um=0,
        A_z_um=0,
        beta1=1,
        beta0_um=0,
        B_z_um=0,
        B_x_start=0,
        B_x_end=100,
        C_x_um=100,
        C_z_um=0,
        d1=-1,
        d0_um=200,
        D_z_um=50,
        D_x_start=100,
        D_x_end=200,
        E_x_um=200,
        E_z_um=0
    )
    # run_analysis(pattern_5_split_1, 5)


    pattern_5_split_3 = PhotobleachPattern5split(
        A_x_um=0,
        A_z_um=0,
        beta1=1,
        beta0_um=0,
        B_z_um=0,
        B_x_start=0,
        B_x_end=100,
        C_x_um=200,
        C_z_um=50,
        d1=-1,
        d0_um=200,
        D_z_um=0,
        D_x_start=100,
        D_x_end=200,
        E_x_um=250,
        E_z_um=0
    )
    # run_analysis(pattern_5_split_3, 5)

    photobleach_pattern_3z = PhotobleachPattern3Z(
        A_x_um=0,
        A_z_um=0,
        beta1=1,
        beta0_um=0,
        B_z_um=0,
        C_x_um=100,
        C_z_delta=1,
    )
    # run_analysis(photobleach_pattern_3z, 3)

    photobleach_pattern_4splitZ1 = PhotobleachPattern4splitZ1(
        A_x_um=0,
        A_z_um=0,
        beta1=1,
        beta0_um=0,
        B_z_um=50,
        b_x_start=0,
        b_x_end=100,
        C_x_um=200,
        C_z_delta=1,
        d1=-1,
        d0_um=200,
        D_z_um=0,
        d_x_start=100,
        d_x_end=200,
    )
    # run_analysis(photobleach_pattern_4splitZ1, 4)

    photobleach_pattern_5splitZ1 = PhotobleachPattern5splitZ(
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
    run_analysis(photobleach_pattern_5splitZ1, 5)

if __name__ == "__main__":
    main()
