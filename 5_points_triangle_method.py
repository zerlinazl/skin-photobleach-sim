from __future__ import annotations

from typing import Literal

import numpy as np

from evaluate_solver import (
    plane_rms_mismatch_uv_triangles,
    recover_plane_dual_annealing_5splitz,
    recovery_loss,
    synthetic_observed_uv,
)
from my_plane import MyPlane
from photobleach_pattern_5_split import PhotobleachPattern5splitZ


def default_pattern() -> PhotobleachPattern5splitZ:
    return PhotobleachPattern5splitZ(
        A_x_um=0.0,
        A_z_um=0.0,
        beta1=1.0,
        beta0_um=0.0,
        B_z_um=0.0,
        B_x_start=0.0,
        B_x_end=100.0,
        C_x_um=100.0,
        C_z_delta=1.0,
        d1=-1.0,
        d0_um=200.0,
        D_z_um=0.0,
        D_x_start=100.0,
        D_x_end=200.0,
        E_x_um=200.0,
        E_z_um=50.0,
    )

def half_slope_pattern() -> PhotobleachPattern5splitZ:
    return PhotobleachPattern5splitZ(
        A_x_um=0.0,
        A_z_um=0.0,
        beta1=0.5,
        beta0_um=0.0,
        B_z_um=0.0,
        B_x_start=0.0,
        B_x_end=100.0,
        C_x_um=100.0,
        C_z_delta=0.1,
        d1=-0.5,
        d0_um=100.0,
        D_z_um=0.0,
        D_x_start=100.0,
        D_x_end=200.0,
        E_x_um=200.0,
        E_z_um=50.0,
    )

def half_slope_ground_truth_plane() -> MyPlane:
    return MyPlane(
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([0.0, 20.0, 0.0]),
        skip_checks=True,
    )

def default_ground_truth_plane() -> MyPlane:
    return MyPlane(
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([0.0, 80.0, 0.0]),
        skip_checks=True,
    )


def add_pixel_noise(
    observed_uv: np.ndarray,
    magnitude_px: float,
    rng: np.random.Generator,
    mode: Literal["gaussian", "uniform"] = "gaussian",
) -> np.ndarray:
    """
    Perturb every observed u and v coordinate (same pixel units as ``MyPlane`` UV).

    Parameters
    ----------
    magnitude_px
        Size of the noise in **pixels**:
        - ``mode="gaussian"``: i.i.d. ``Normal(0, magnitude_px)`` per coordinate (σ = magnitude).
        - ``mode="uniform"``: i.i.d. ``Uniform(-magnitude_px, magnitude_px)`` per coordinate.
    mode
        ``"gaussian"`` (default) or ``"uniform"``.

    Use ``magnitude_px=0`` for an unchanged copy.
    """
    clean = np.asarray(observed_uv, dtype=float).copy()
    if magnitude_px <= 0.0:
        return clean
    if mode == "gaussian":
        clean += rng.normal(0.0, magnitude_px, size=clean.shape)
    else:
        clean += rng.uniform(-magnitude_px, magnitude_px, size=clean.shape)
    return clean


def add_isotropic_pixel_noise(
    observed_uv: np.ndarray,
    pixel_noise_std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Deprecated name: use ``add_pixel_noise(..., magnitude_px=pixel_noise_std, mode=\"gaussian\")``."""
    return add_pixel_noise(observed_uv, pixel_noise_std, rng, mode="gaussian")


def _mean_rms_over_restarts(
    res,
    u_vec: np.ndarray,
    v_vec: np.ndarray,
    h_vec: np.ndarray,
    u_A: float,
    v_A: float,
    u_B: float,
    v_B: float,
    u_C: float,
    v_C: float,
    u_D: float,
    v_D: float,
    u_E: float,
    v_E: float,
    rms_seed: int,
) -> float:
    """
    RMS vs ground-truth plane for each restart's recovered (u,v,h), then mean.
    Uses the same RNG seed for each ``plane_rms_mismatch_uv_triangles`` call so UV samples match.
    """
    pr = res.per_restart_params
    if pr.size == 0:
        return float("nan")
    rms_vals = []
    for i in range(pr.shape[0]):
        row = pr[i]
        u_r, v_r, h_r = row[0:3], row[3:6], row[6:9]
        rms_vals.append(
            plane_rms_mismatch_uv_triangles(
                u_vec,
                v_vec,
                h_vec,
                u_r,
                v_r,
                h_r,
                u_A,
                v_A,
                u_B,
                v_B,
                u_C,
                v_C,
                u_D,
                v_D,
                u_E,
                v_E,
                rng=np.random.default_rng(rms_seed),
            )
        )
    return float(np.mean(rms_vals))


def _run_dual_annealing_best_of_k_loop(
    *,
    pattern: PhotobleachPattern5splitZ,
    plane_true: MyPlane,
    observed_uv_solver: np.ndarray,
    uv_for_rms_polygon: np.ndarray,
    trials: int,
    niters: tuple[int, ...],
    rms_rng: np.random.Generator,
    rms_seed: int,
    print_truth_residual_on_solver_uv: bool,
    truth_residual_note: str = "",
) -> None:
    """
    Run best-of-k recovery on ``observed_uv_solver``; RMS vs truth uses ``uv_for_rms_polygon``
    corners (typically clean UV so the chart matches the ground-truth plane).
    """
    u_vec, v_vec, h_vec = plane_true.u, plane_true.v, plane_true.h

    if print_truth_residual_on_solver_uv:
        loss_at_truth = recovery_loss(pattern, observed_uv_solver, u_vec, v_vec, h_vec)
        print(truth_residual_note, f"{loss_at_truth:.6e}")

    poly = np.asarray(uv_for_rms_polygon, dtype=float)
    u_A, v_A = poly[0]
    u_B, v_B = poly[1]
    u_C, v_C = poly[2]
    u_D, v_D = poly[3]
    u_E, v_E = poly[4]

    print(f"Running dual annealing best-of-{trials} (restarts={trials}, seeds 0..{trials - 1})...")
    for niter in niters:
        res = recover_plane_dual_annealing_5splitz(
            pattern,
            observed_uv_solver,
            initial_u_vec=u_vec,
            initial_v_vec=v_vec,
            initial_h_vec=h_vec,
            maxiter=niter,
            random_seed=0,
            restarts=trials,
        )
        loss_rec = recovery_loss(
            pattern,
            observed_uv_solver,
            res.u_vec,
            res.v_vec,
            res.h_vec,
        )
        bi = res.best_restart_index if res.best_restart_index is not None else 0
        print(
            f"  best restart index: {bi} (of {res.n_restarts} seeds), "
            f"recovery_loss={loss_rec:.6e} (optimizer objective={res.objective_value:.6e})"
        )

        rms = plane_rms_mismatch_uv_triangles(
            u_vec,
            v_vec,
            h_vec,
            res.u_vec,
            res.v_vec,
            res.h_vec,
            u_A,
            v_A,
            u_B,
            v_B,
            u_C,
            v_C,
            u_D,
            v_D,
            u_E,
            v_E,
            rng=rms_rng,
        )
        rms_mean = _mean_rms_over_restarts(
            res,
            u_vec,
            v_vec,
            h_vec,
            u_A,
            v_A,
            u_B,
            v_B,
            u_C,
            v_C,
            u_D,
            v_D,
            u_E,
            v_E,
            rms_seed,
        )

        np.set_printoptions(precision=4, suppress=True)
        print("u:", res.u_vec)
        print("v:", res.v_vec)
        print("h:", res.h_vec)
        print(f"\tN_iters: {niter}, RMS (best-of-{trials}): {rms:.6f}")
        print(
            f"\tN_iters: {niter}, mean RMS (avg over {res.n_restarts} restarts): {rms_mean:.6f}"
        )


def run_demo_init_at_true_plane(
    *,
    trials: int = 5,
    niters: tuple[int, ...] = (200,),
) -> None:
    """
    Build synthetic observed UV from a known plane, verify zero loss at truth, then run
    dual annealing with **best-of-k restarts** (``restarts=trials``, seeds ``0..trials-1``)
    and report recovery loss + RMS plane mismatch over UV triangles ABC + ACD.
    """
    pattern = half_slope_pattern()
    plane_true = half_slope_ground_truth_plane()
    # pattern = default_pattern()
    # plane_true = default_ground_truth_plane()
    observed_uv = synthetic_observed_uv(pattern, plane_true)

    _run_dual_annealing_best_of_k_loop(
        pattern=pattern,
        plane_true=plane_true,
        observed_uv_solver=observed_uv,
        uv_for_rms_polygon=observed_uv,
        trials=trials,
        niters=niters,
        rms_rng=np.random.default_rng(42),
        rms_seed=42,
        print_truth_residual_on_solver_uv=True,
        truth_residual_note="Residual loss at ground-truth plane on clean UV (should be ~0):",
    )


def run_demo_with_observation_noise(
    *,
    noise_magnitude_px: float,
    noise_mode: Literal["gaussian", "uniform"] = "gaussian",
    noise_seed: int = 123,
    trials: int = 15,
    noise_trials: int = 1,
    niters: tuple[int, ...] = (200,),
    rms_seed: int = 42,
) -> None:
    """
    Same workflow as ``run_demo_init_at_true_plane``, but add pixel noise to every observed
    u and v before recovery.

    **Specifying “3 px of noise” (typical):** pass ``noise_magnitude_px=3.0``. With the default
    ``noise_mode="gaussian"``, each coordinate gets independent Gaussian noise with **σ = 3**
    pixels. With ``noise_mode="uniform"``, each coordinate is shifted uniformly in
    **(-3, 3)** pixels.

    Recovery fits the **noisy** observations. RMS vs the ground-truth plane still uses
    **clean** UV corners for the ABC/ACD sampling region so the chart matches truth.
    """
    pattern = half_slope_pattern()
    plane_true = half_slope_ground_truth_plane()
    # pattern = default_pattern()
    # plane_true = default_ground_truth_plane()

    observed_clean = synthetic_observed_uv(pattern, plane_true)
    if noise_mode == "gaussian":
        print(f"Observation noise: Gaussian σ = {noise_magnitude_px} px per u or v coordinate")
    else:
        print(f"Observation noise: Uniform in (-{noise_magnitude_px}, {noise_magnitude_px}) px per coordinate")

    if noise_trials < 1:
        raise ValueError("noise_trials must be >= 1.")

    u_vec, v_vec, h_vec = plane_true.u, plane_true.v, plane_true.h
    poly = np.asarray(observed_clean, dtype=float)
    u_A, v_A = poly[0]
    u_B, v_B = poly[1]
    u_C, v_C = poly[2]
    u_D, v_D = poly[3]
    u_E, v_E = poly[4]

    best_rms_all = np.zeros((noise_trials, len(niters)), dtype=float)
    mean_rms_all = np.zeros((noise_trials, len(niters)), dtype=float)
    best_loss_all = np.zeros((noise_trials, len(niters)), dtype=float)

    for noise_t in range(noise_trials):
        # Each noise trial gets a different RNG stream.
        rng_noise = np.random.default_rng(noise_seed + noise_t)
        observed_noisy = add_pixel_noise(
            observed_clean, noise_magnitude_px, rng_noise, mode=noise_mode
        )

        print(f"\n=== noise trial {noise_t + 1}/{noise_trials} ===")
        if noise_magnitude_px > 0:
            print(
                "Residual at ground-truth (u,v,h) on this noisy UV:",
                f"{recovery_loss(pattern, observed_noisy, plane_true.u, plane_true.v, plane_true.h):.6e}",
            )

        for niter_idx, niter in enumerate(niters):
            # Use best-of-k restarts; seeds 0..trials-1 are determined inside recover_*.
            res = recover_plane_dual_annealing_5splitz(
                pattern,
                observed_noisy,
                initial_u_vec=u_vec,
                initial_v_vec=v_vec,
                initial_h_vec=h_vec,
                maxiter=niter,
                random_seed=0,
                restarts=trials,
            )

            loss_rec = recovery_loss(
                pattern, observed_noisy, res.u_vec, res.v_vec, res.h_vec
            )
            # Deterministic RMS sampling for fair comparison across restarts/noise trials.
            rms_seed_this = rms_seed + 1000 * noise_t + 10 * niter_idx
            rms_rng_this = np.random.default_rng(rms_seed_this)
            rms_best = plane_rms_mismatch_uv_triangles(
                u_vec,
                v_vec,
                h_vec,
                res.u_vec,
                res.v_vec,
                res.h_vec,
                u_A,
                v_A,
                u_B,
                v_B,
                u_C,
                v_C,
                u_D,
                v_D,
                u_E,
                v_E,
                rng=rms_rng_this,
            )
            rms_mean = _mean_rms_over_restarts(
                res,
                u_vec,
                v_vec,
                h_vec,
                u_A,
                v_A,
                u_B,
                v_B,
                u_C,
                v_C,
                u_D,
                v_D,
                u_E,
                v_E,
                rms_seed=rms_seed_this,
            )

            best_rms_all[noise_t, niter_idx] = rms_best
            mean_rms_all[noise_t, niter_idx] = rms_mean
            best_loss_all[noise_t, niter_idx] = loss_rec

            print(
                f"niter={niter}: best RMS={rms_best:.6f}, mean RMS(restarts)={rms_mean:.6f}, "
                f"recovery_loss(best)={loss_rec:.6e}, best_restart_index={res.best_restart_index}"
            )

    print("\n=== summary across noise trials ===")
    for niter_idx, niter in enumerate(niters):
        print(
            f"niter={niter}: "
            f"avg best RMS={best_rms_all[:, niter_idx].mean():.6f} (std {best_rms_all[:, niter_idx].std():.6f}), "
            f"avg mean RMS(restarts)={mean_rms_all[:, niter_idx].mean():.6f} (std {mean_rms_all[:, niter_idx].std():.6f})"
        )


def main() -> None:
    # run_demo_init_at_true_plane(trials=20, niters=(200,))
    # Example: σ = 3 px Gaussian on each u,v coordinate (10 coordinates total)
    print("--------------------------------")
    run_demo_with_observation_noise(noise_magnitude_px=1.0, trials=20, niters=(200,), noise_trials=50)


if __name__ == "__main__":
    main()
