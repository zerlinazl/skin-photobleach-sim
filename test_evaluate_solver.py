import numpy as np

from evaluate_solver import recover_plane_dual_annealing_5splitz
from my_plane import MyPlane
from photobleach_pattern_5_split import PhotobleachPattern5splitZ


def _make_pattern() -> PhotobleachPattern5splitZ:
    return PhotobleachPattern5splitZ(
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


def _simulate_observed_uv(pattern: PhotobleachPattern5splitZ, plane: MyPlane) -> np.ndarray:
    pts = pattern.forward_model_nonparametric(plane)
    uv = [plane.physical_to_pix(np.asarray(pt, dtype=float)) for pt in pts]
    return np.asarray(uv, dtype=float)


def test_recover_plane_dual_annealing_5splitz_synthetic():
    pattern = _make_pattern()
    plane_true = MyPlane(
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([0.0, 80.0, 0.0]),
    )
    observed_uv = _simulate_observed_uv(pattern, plane_true)

    result = recover_plane_dual_annealing_5splitz(
        pattern,
        observed_uv,
        initial_u_vec=plane_true.u,
        initial_v_vec=plane_true.v,
        initial_h_vec=plane_true.h,
        maxiter=200,
        random_seed=0,
    )

    assert result.n_restarts == 1
    assert result.best_restart_index is None
    assert result.per_restart_objectives.shape == (1,)
    assert result.per_restart_params.shape == (1, 9)

    # Objective should be effectively zero on noiseless synthetic data.
    assert result.objective_value < 0.1 #1e-8

    # Recovery should be close to truth for this deterministic setup.
    assert np.linalg.norm(result.u_vec - plane_true.u) < 1e-1
    assert np.linalg.norm(result.v_vec - plane_true.v) < 1e-1
    assert np.linalg.norm(result.h_vec - plane_true.h) < 1e-0


def test_recover_best_of_k_restarts_metadata():
    pattern = _make_pattern()
    plane_true = MyPlane(
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([0.0, 80.0, 0.0]),
    )
    observed_uv = _simulate_observed_uv(pattern, plane_true)
    r = recover_plane_dual_annealing_5splitz(
        pattern,
        observed_uv,
        initial_u_vec=plane_true.u,
        initial_v_vec=plane_true.v,
        initial_h_vec=plane_true.h,
        maxiter=80,
        random_seed=0,
        restarts=4,
    )
    assert r.n_restarts == 4
    assert r.best_restart_index is not None
    assert 0 <= r.best_restart_index < 4
    assert r.per_restart_objectives.shape == (4,)
    assert r.per_restart_params.shape == (4, 9)
    assert np.isclose(
        r.per_restart_objectives[r.best_restart_index], r.objective_value, rtol=0, atol=1e-12
    )
