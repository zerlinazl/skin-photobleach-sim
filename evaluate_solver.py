"""
Dual-annealing plane recovery for 5-point observed UV measurements.

This module exposes a clean API to recover the plane basis vectors and offset:
    u_vec, v_vec, h_vec
from observed pixel coordinates for points A..E on a `PhotobleachPattern5splitZ`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.optimize import OptimizeResult, dual_annealing

from my_plane import MyPlane
from photobleach_pattern_5_split import PhotobleachPattern5splitZ


@dataclass
class RecoveredPlaneResult:
    u_vec: np.ndarray
    v_vec: np.ndarray
    h_vec: np.ndarray
    objective_value: float
    scipy_result: OptimizeResult
    # When restarts > 1: which seed index produced the minimum objective (0 .. restarts-1).
    best_restart_index: Optional[int] = None
    n_restarts: int = 1
    # Per-restart outcomes (row i = restart i): objectives and packed [u(3), v(3), h(3)].
    per_restart_objectives: np.ndarray = field(default_factory=lambda: np.array([]))
    per_restart_params: np.ndarray = field(default_factory=lambda: np.zeros((0, 9)))


def _pack_params(u_vec: np.ndarray, v_vec: np.ndarray, h_vec: np.ndarray) -> np.ndarray:
    return np.concatenate([u_vec, v_vec, h_vec]).astype(float)


def _unpack_params(params: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p = np.asarray(params, dtype=float).reshape(-1)
    if p.size != 9:
        raise ValueError("Expected 9 parameters [u(3), v(3), h(3)].")
    return p[0:3], p[3:6], p[6:9]


def _vector_bounds(center: np.ndarray, relative: float, abs_floor: float) -> list[tuple[float, float]]:
    norm = float(np.linalg.norm(center))
    delta = max(abs_floor, relative * max(1.0, norm))
    return [(float(x - delta), float(x + delta)) for x in center]


def _objective_from_observed_uv(
    params: np.ndarray,
    observed_uv: np.ndarray,
    pattern: PhotobleachPattern5splitZ,
    degeneracy_penalty: float,
) -> float:
    u_vec, v_vec, h_vec = _unpack_params(params)

    # Penalize degenerate basis vectors where normal vector is ill-defined.
    cross_norm = float(np.linalg.norm(np.cross(u_vec, v_vec)))
    if cross_norm < 1e-8:
        return degeneracy_penalty

    try:
        plane = MyPlane(u_vec, v_vec, h_vec, skip_checks=True)
    except Exception:
        return degeneracy_penalty

    model_pts = pattern.forward_model_nonparametric(plane)
    if len(model_pts) != observed_uv.shape[0]:
        return degeneracy_penalty

    loss = 0.0
    for i, pt_model in enumerate(model_pts):
        if np.any(~np.isfinite(pt_model)):
            return degeneracy_penalty
        u_pix, v_pix = observed_uv[i]
        pt_parametric = plane.pix_to_physical(float(u_pix), float(v_pix))
        diff = np.asarray(pt_model, dtype=float) - np.asarray(pt_parametric, dtype=float)
        loss += float(np.dot(diff, diff))

    return loss


def recovery_loss(
    pattern: PhotobleachPattern5splitZ,
    observed_uv: np.ndarray,
    u_vec: np.ndarray,
    v_vec: np.ndarray,
    h_vec: np.ndarray,
    degeneracy_penalty: float = 1e12,
) -> float:
    """
    Same residual as the dual-annealing objective: sum of squared 3D mismatches
    between nonparametric line intersections and parametric points at observed UV.
    """
    params = _pack_params(
        np.asarray(u_vec, dtype=float),
        np.asarray(v_vec, dtype=float),
        np.asarray(h_vec, dtype=float),
    )
    return _objective_from_observed_uv(
        params,
        np.asarray(observed_uv, dtype=float),
        pattern,
        degeneracy_penalty,
    )


def recover_plane_dual_annealing_5splitz(
    pattern: PhotobleachPattern5splitZ,
    observed_uv: np.ndarray,
    *,
    initial_u_vec: Optional[np.ndarray] = None,
    initial_v_vec: Optional[np.ndarray] = None,
    initial_h_vec: Optional[np.ndarray] = None,
    maxiter: int = 2000,
    u_v_relative_bound: float = 0.3,
    h_abs_bound_um: float = 25.0,
    random_seed: Optional[int] = None,
    initial_temp: float = 5230.0,
    visit: float = 2.7,
    accept: float = -5.0,
    no_local_search: bool = False,
    degeneracy_penalty: float = 1e12,
    restarts: int = 1,
) -> RecoveredPlaneResult:
    """
    Recover (u_vec, v_vec, h_vec) from five observed (u, v) pixel coordinates.

    Parameters
    ----------
    pattern
        Photobleach pattern instance (`PhotobleachPattern5splitZ`) used by the solver.
    observed_uv
        Array-like shape (5, 2), rows ordered as points A, B, C, D, E.
    initial_u_vec, initial_v_vec, initial_h_vec
        Initial center for optimizer bounds. Defaults to [1,0,0], [0,0,1], [0,80,0].
    maxiter
        Dual annealing iteration budget **per restart**.
    u_v_relative_bound
        Relative bound size for each component of u/v vectors.
    h_abs_bound_um
        Absolute +/- bound around initial h values (in um).
    restarts
        Run dual annealing this many times with seeds ``random_seed + i`` (if ``random_seed``
        is an int) or default scipy randomness if ``random_seed`` is None. Return the
        (u,v,h) with minimum objective among runs (best-of-k).

    Returns
    -------
    RecoveredPlaneResult
        Contains recovered vectors plus scipy result for the **winning** restart.
        ``best_restart_index`` and ``n_restarts`` are set when ``restarts > 1``.
    """
    obs = np.asarray(observed_uv, dtype=float)
    if obs.shape != (5, 2):
        raise ValueError("observed_uv must have shape (5, 2) ordered as A,B,C,D,E.")
    if np.any(~np.isfinite(obs)):
        raise ValueError("observed_uv contains non-finite values.")

    u0 = np.array([1.0, 0.0, 0.0]) if initial_u_vec is None else np.asarray(initial_u_vec, dtype=float)
    v0 = np.array([0.0, 0.0, 1.0]) if initial_v_vec is None else np.asarray(initial_v_vec, dtype=float)
    h0 = np.array([0.0, 80.0, 0.0]) if initial_h_vec is None else np.asarray(initial_h_vec, dtype=float)

    if u0.shape != (3,) or v0.shape != (3,) or h0.shape != (3,):
        raise ValueError("Initial vectors must all be shape (3,).")
    if restarts < 1:
        raise ValueError("restarts must be >= 1.")

    bounds: list[tuple[float, float]] = []
    bounds.extend(_vector_bounds(u0, relative=u_v_relative_bound, abs_floor=0.5))
    bounds.extend(_vector_bounds(v0, relative=u_v_relative_bound, abs_floor=0.5))
    bounds.extend(
        [
            (float(h0[0] - h_abs_bound_um), float(h0[0] + h_abs_bound_um)),
            (float(h0[1] - h_abs_bound_um), float(h0[1] + h_abs_bound_um)),
            (float(h0[2] - h_abs_bound_um), float(h0[2] + h_abs_bound_um)),
        ]
    )

    def objective(params: np.ndarray) -> float:
        return _objective_from_observed_uv(
            params=params,
            observed_uv=obs,
            pattern=pattern,
            degeneracy_penalty=degeneracy_penalty,
        )

    best_result: Optional[OptimizeResult] = None
    best_fun = float("inf")
    best_idx = 0
    per_obj: list[float] = []
    per_params: list[np.ndarray] = []

    for i in range(restarts):
        seed_i = random_seed + i if random_seed is not None else None
        result = dual_annealing(
            objective,
            bounds=bounds,
            maxiter=maxiter,
            seed=seed_i,
            initial_temp=initial_temp,
            visit=visit,
            accept=accept,
            no_local_search=no_local_search,
        )
        fun = float(result.fun)
        per_obj.append(fun)
        per_params.append(np.asarray(result.x, dtype=float).reshape(-1).copy())
        if fun < best_fun:
            best_fun = fun
            best_result = result
            best_idx = i

    assert best_result is not None
    u_opt, v_opt, h_opt = _unpack_params(best_result.x)
    return RecoveredPlaneResult(
        u_vec=u_opt,
        v_vec=v_opt,
        h_vec=h_opt,
        objective_value=best_fun,
        scipy_result=best_result,
        best_restart_index=best_idx if restarts > 1 else None,
        n_restarts=restarts,
        per_restart_objectives=np.asarray(per_obj, dtype=float),
        per_restart_params=np.stack(per_params, axis=0) if per_params else np.zeros((0, 9)),
    )


def synthetic_observed_uv(pattern: PhotobleachPattern5splitZ, plane: MyPlane) -> np.ndarray:
    """
    Ideal observed UV for points A..E: project each nonparametric intersection to pixel coords.
    Shape (5, 2), rows ordered A..E.
    """
    pts = pattern.forward_model_nonparametric(plane)
    uv = [plane.physical_to_pix(np.asarray(pt, dtype=float)) for pt in pts]
    return np.asarray(uv, dtype=float)


def plane_rms_mismatch_uv_triangles(
    u_vec: np.ndarray,
    v_vec: np.ndarray,
    h_vec: np.ndarray,
    u_vec2: np.ndarray,
    v_vec2: np.ndarray,
    h_vec2: np.ndarray,
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
    n_samples: int = 10_000,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """
    RMS 3D distance between two plane parameterizations over a UV region: triangles ABC and ACD
    (same construction as the original notebook script). Landmark E is not used in the sampling region.
    """
    rng = rng or np.random.default_rng()
    u1 = np.asarray(u_vec, dtype=float)
    v1 = np.asarray(v_vec, dtype=float)
    h1 = np.asarray(h_vec, dtype=float)
    u2 = np.asarray(u_vec2, dtype=float)
    v2 = np.asarray(v_vec2, dtype=float)
    h2 = np.asarray(h_vec2, dtype=float)

    A = np.array([u_A, v_A])
    B = np.array([u_B, v_B])
    C = np.array([u_C, v_C])
    D = np.array([u_D, v_D])

    def sample_triangle(P0: np.ndarray, P1: np.ndarray, P2: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
        r1 = rng.random(n)
        r2 = rng.random(n)
        mask = r1 + r2 > 1
        r1 = np.where(mask, 1 - r1, r1)
        r2 = np.where(mask, 1 - r2, r2)
        samples = P0 + r1[:, None] * (P1 - P0) + r2[:, None] * (P2 - P0)
        return samples[:, 0], samples[:, 1]

    def triangle_area(P0: np.ndarray, P1: np.ndarray, P2: np.ndarray) -> float:
        return 0.5 * abs(
            (P1[0] - P0[0]) * (P2[1] - P0[1]) - (P2[0] - P0[0]) * (P1[1] - P0[1])
        )

    area1 = triangle_area(A, B, C)
    area2 = triangle_area(A, C, D)
    total = area1 + area2
    if total <= 0:
        return float("nan")

    n1 = int(n_samples * area1 / total)
    n2 = n_samples - n1
    uu1, vv1 = sample_triangle(A, B, C, n1)
    uu2, vv2 = sample_triangle(A, C, D, n2)
    u_samples = np.concatenate([uu1, uu2])
    v_samples = np.concatenate([vv1, vv2])

    sq = 0.0
    for uu, vv in zip(u_samples, v_samples):
        p1 = h1 + uu * u1 + vv * v1
        p2 = h2 + uu * u2 + vv * v2
        d = p1 - p2
        sq += float(np.dot(d, d))
    return float(np.sqrt(sq / n_samples))


def _demo() -> None:
    pattern = PhotobleachPattern5splitZ(
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

    plane_true = MyPlane(
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([0.0, 80.0, 0.0]),
    )
    observed_uv = synthetic_observed_uv(pattern, plane_true)

    res = recover_plane_dual_annealing_5splitz(
        pattern,
        observed_uv,
        initial_u_vec=plane_true.u,
        initial_v_vec=plane_true.v,
        initial_h_vec=plane_true.h,
        maxiter=400,
        random_seed=0,
    )

    print("Recovered u_vec:", np.round(res.u_vec, 6))
    print("Recovered v_vec:", np.round(res.v_vec, 6))
    print("Recovered h_vec:", np.round(res.h_vec, 6))
    print("Objective value:", f"{res.objective_value:.6e}")
    print("SciPy success:", bool(res.scipy_result.success))
    print("SciPy message:", res.scipy_result.message)


if __name__ == "__main__":
    _demo()

