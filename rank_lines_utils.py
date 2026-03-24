import numpy as np
from my_plane import MyPlane
from scipy.linalg import qr
from photobleach_pattern import (
    PhotobleachPattern,
    PhotobleachPattern4,
    PhotobleachPatternVarZC,
    PhotobleachPattern4Z,
    PhotobleachPattern5,
    PhotobleachPattern5Z,
    PhotobleachPattern5Z2,
)
from test_forward_model_sensitivity_utils import (
    compare_error_between_two_designs,
    compare_error_between_two_designs4,
    compare_error_between_two_designs5,
)

def get_rank(M):
    return np.linalg.matrix_rank(M)

def get_normalized_vector_and_magnitude_for_y_translation(input_value, my_plane, photobleach_pattern, n_points):
    modified_plane = my_plane.copy()
    modified_plane.h[1] = modified_plane.h[1] - input_value # Moving 1 micron = 1 pixel

    if n_points == 3:
        normalized_vector, magnitude = compare_error_between_two_designs(
            photobleach_pattern, my_plane,
            photobleach_pattern, modified_plane)
    elif n_points == 4:
        normalized_vector, magnitude = compare_error_between_two_designs4(
            photobleach_pattern, my_plane,
            photobleach_pattern, modified_plane)
    elif n_points == 5:
        normalized_vector, magnitude = compare_error_between_two_designs5(
            photobleach_pattern, my_plane,
            photobleach_pattern, modified_plane)
    else:
        raise ValueError(f"Invalid number of points: {n_points}")
    return normalized_vector, magnitude

def get_normalized_vector_and_magnitude_for_z_translation(input_value, my_plane, photobleach_pattern, n_points):
    modified_plane = my_plane.copy()
    modified_plane.h = modified_plane.h + np.array([0, 0, input_value])

    if n_points == 3:
        normalized_vector, magnitude = compare_error_between_two_designs(
            photobleach_pattern, my_plane,
            photobleach_pattern, modified_plane)
    elif n_points == 4:
        normalized_vector, magnitude = compare_error_between_two_designs4(
            photobleach_pattern, my_plane,
            photobleach_pattern, modified_plane)
    elif n_points == 5:
        normalized_vector, magnitude = compare_error_between_two_designs5(
            photobleach_pattern, my_plane,
            photobleach_pattern, modified_plane)
    else:
        raise ValueError(f"Invalid number of points: {n_points}")
    return normalized_vector, magnitude

def get_normalized_vector_and_magnitude_for_x_translation(input_value, my_plane, photobleach_pattern, n_points):
    modified_plane = my_plane.copy()
    modified_plane.h = modified_plane.h + np.array([input_value, 0, 0])

    if n_points == 3:
        normalized_vector, magnitude = compare_error_between_two_designs(
            photobleach_pattern, my_plane,
            photobleach_pattern, modified_plane)
    elif n_points == 4:
        normalized_vector, magnitude = compare_error_between_two_designs4(
            photobleach_pattern, my_plane,
            photobleach_pattern, modified_plane)
    elif n_points == 5:
        normalized_vector, magnitude = compare_error_between_two_designs5(
            photobleach_pattern, my_plane,
            photobleach_pattern, modified_plane)
    else:
        raise ValueError(f"Invalid number of points: {n_points}")
    return normalized_vector, magnitude

def get_normalized_vector_and_magnitude_for_z_rotation(input_value, my_plane, photobleach_pattern, n_points):
    theta = np.deg2rad(input_value)
    modified_plane = my_plane.copy()
    modified_plane.u = np.array([np.cos(theta), -np.sin(theta), 0.0])
    modified_plane.v = np.array([0.0, 0.0, 1.0])

    if n_points == 3:
        normalized_vector, magnitude = compare_error_between_two_designs(
            photobleach_pattern, my_plane,
            photobleach_pattern, modified_plane)
    elif n_points == 4:
        normalized_vector, magnitude = compare_error_between_two_designs4(
            photobleach_pattern, my_plane,
            photobleach_pattern, modified_plane)
    elif n_points == 5:
        normalized_vector, magnitude = compare_error_between_two_designs5(
            photobleach_pattern, my_plane,
            photobleach_pattern, modified_plane)
    else:
        raise ValueError(f"Invalid number of points: {n_points}")
    return normalized_vector, magnitude

def get_normalized_vector_and_magnitude_for_x_rotation(input_value, my_plane, photobleach_pattern, n_points):
    theta = np.deg2rad(input_value)
    modified_plane = my_plane.copy()
    modified_plane.u = np.array([1.0, 0.0, 0.0])
    modified_plane.v = np.array([0.0, -np.sin(theta), np.cos(theta)])

    if n_points == 3:
        normalized_vector, magnitude = compare_error_between_two_designs(
            photobleach_pattern, my_plane,
            photobleach_pattern, modified_plane)
    elif n_points == 4:
        normalized_vector, magnitude = compare_error_between_two_designs4(
            photobleach_pattern, my_plane,
            photobleach_pattern, modified_plane)
    elif n_points == 5:
        normalized_vector, magnitude = compare_error_between_two_designs5(
            photobleach_pattern, my_plane,
            photobleach_pattern, modified_plane)
    else:
        raise ValueError(f"Invalid number of points: {n_points}")

    return normalized_vector, magnitude

def get_normalized_vector_and_magnitude_for_y_rotation(input_value, my_plane, photobleach_pattern, n_points):
    theta = np.deg2rad(input_value)
    modified_plane = my_plane.copy()
    modified_plane.u = np.array([np.cos(theta), 0.0, np.sin(theta)])
    modified_plane.v = np.array([-np.sin(theta), 0.0,  np.cos(theta)])

    if n_points == 3:
        normalized_vector, magnitude = compare_error_between_two_designs(
            photobleach_pattern, my_plane,
            photobleach_pattern, modified_plane)
    elif n_points == 4:
        normalized_vector, magnitude = compare_error_between_two_designs4(
            photobleach_pattern, my_plane,
            photobleach_pattern, modified_plane)
    elif n_points == 5:
        normalized_vector, magnitude = compare_error_between_two_designs5(
            photobleach_pattern, my_plane,
            photobleach_pattern, modified_plane)
    else:
        raise ValueError(f"Invalid number of points: {n_points}")
    return normalized_vector, magnitude

def get_normalized_vector_and_magnitude_for_scale_x(input_value, my_plane, photobleach_pattern, n_points):
    modified_plane = my_plane.copy()
    modified_plane.u = np.array([input_value, 0.0, 0.0])
    modified_plane.v = np.array([0.0, 0.0, 1.0])

    if n_points == 3:
        normalized_vector, magnitude = compare_error_between_two_designs(
            photobleach_pattern, my_plane,
            photobleach_pattern, modified_plane)
    elif n_points == 4:
        normalized_vector, magnitude = compare_error_between_two_designs4(
            photobleach_pattern, my_plane,
            photobleach_pattern, modified_plane)
    elif n_points == 5:
        normalized_vector, magnitude = compare_error_between_two_designs5(
            photobleach_pattern, my_plane,
            photobleach_pattern, modified_plane)
    else:
        raise ValueError(f"Invalid number of points: {n_points}")
    return normalized_vector, magnitude

def get_normalized_vector_and_magnitude_for_scale_z(input_value, my_plane, photobleach_pattern, n_points):
    modified_plane = my_plane.copy()
    modified_plane.v = np.array([0.0, 0.0, input_value])

    if n_points == 3:
        normalized_vector, magnitude = compare_error_between_two_designs(
            photobleach_pattern, my_plane,
            photobleach_pattern, modified_plane)
    elif n_points == 4:
        normalized_vector, magnitude = compare_error_between_two_designs4(
            photobleach_pattern, my_plane,
            photobleach_pattern, modified_plane)
    elif n_points == 5:
        normalized_vector, magnitude = compare_error_between_two_designs5(
            photobleach_pattern, my_plane,
        photobleach_pattern, modified_plane)
    else:
        raise ValueError(f"Invalid number of points: {n_points}")
    return normalized_vector, magnitude

def get_normalized_vector_and_magnitude_for_shear_xz(input_value, my_plane, photobleach_pattern, n_points):
    modified_plane = my_plane.copy()

    s = input_value

    modified_plane.u = np.array([1.0, 0.0, 0.0])
    modified_plane.v = np.array([s, 0.0, 1.0])

    if n_points == 3:
        normalized_vector, magnitude = compare_error_between_two_designs(
            photobleach_pattern, my_plane,
            photobleach_pattern, modified_plane)
    elif n_points == 4:
        normalized_vector, magnitude = compare_error_between_two_designs4(
            photobleach_pattern, my_plane,
            photobleach_pattern, modified_plane)
    elif n_points == 5:
        normalized_vector, magnitude = compare_error_between_two_designs5(
            photobleach_pattern, my_plane,
        photobleach_pattern, modified_plane)
    else:
        raise ValueError(f"Invalid number of points: {n_points}")

    return normalized_vector, magnitude

def round_to_bins(arr, bins):
    # arr_abs = np.abs(arr)
    arr_abs = arr
    bins = np.array(bins)
    # compute distance to each bin
    idx = np.argmin(np.abs(arr_abs[..., None] - bins), axis=-1)
    return bins[idx]

def get_binned_M(M, bins):
    return round_to_bins(M, bins)

def find_dependent_columns(M, tol=1e-8):
    """
    Returns indices of independent and dependent **columns** of matrix M.
    Columns correspond to individual transforms (movement patterns).
    """
    Q, R, piv = qr(M, pivoting=True)

    rank = np.sum(np.abs(np.diag(R)) > tol)

    independent_cols = piv[:rank]
    dependent_cols = piv[rank:]

    return independent_cols, dependent_cols


def find_column_combination(M, target_col_idx, tol=1e-8):
    """
    Express the target column as a linear combination of the other columns
    (in least‑squares sense). Returns list of (column_index, coefficient).
    """
    target = M[:, target_col_idx]

    # remove the target column
    other_indices = [j for j in range(M.shape[1]) if j != target_col_idx]
    A = M[:, other_indices]

    # solve least squares: A @ coeffs ≈ target
    coeffs, residuals, rank, s = np.linalg.lstsq(A, target, rcond=None)

    # identify columns that contribute
    contributing_columns = [
        (int(other_indices[i]), coeffs[i])
        for i in range(len(coeffs))
        if abs(coeffs[i]) > tol
    ]

    return contributing_columns

def find_identical_columns(M):
    """
    Returns groups of column indices that are identical.
    Each group contains indices of columns that are exactly the same.
    """
    unique_cols, inverse = np.unique(M, axis=1, return_inverse=True)

    groups = {}
    for j, group_id in enumerate(inverse):
        groups.setdefault(group_id, []).append(j)

    # keep only groups with duplicates
    identical_groups = [cols for cols in groups.values() if len(cols) > 1]

    return identical_groups


def solve_plane_from_pixels_and_pattern5z2(
    u_points, v_points, pattern: PhotobleachPattern5Z2
) -> MyPlane:
    """
    Solve for plane parameters (u, v, h) given pixel coordinates for A, B, C, D, E
    and a PhotobleachPattern5Z2 that defines the underlying physical lines.

    Assumes:
    - u_points, v_points are length‑5 iterables giving pixel coords for
      points [A, B, C, D, E] in this order.
    - The plane mapping is: pt = h + u * u_pix + v * v_pix.
    - Points lie on the lines encoded in PhotobleachPattern5Z2.
    """

    u_points = np.asarray(u_points, dtype=float)
    v_points = np.asarray(v_points, dtype=float)

    if u_points.shape[0] != 5 or v_points.shape[0] != 5:
        raise ValueError("u_points and v_points must each have length 5 (A, B, C, D, E).")

    # Unknown vector X has 14 components:
    # [u_x, u_y, u_z,
    #  v_x, v_y, v_z,
    #  h_x, h_y, h_z,
    #  sA, sB, sC, sD, sE]
    n_unknowns = 14
    rows = []
    rhs = []

    def add_equation(coeffs, value):
        rows.append(coeffs)
        rhs.append(value)

    # Convenient aliases
    A_u, B_u, C_u, D_u, E_u = u_points
    A_v, B_v, C_v, D_v, E_v = v_points

    # A lies on line: x = A_x_um, z = A_z_um, y = sA (free)
    Ax, Az = pattern.A_x_um, pattern.A_z_um
    # x‑component: h_x + u_x*A_u + v_x*A_v = Ax
    add_equation(
        [A_u, 0.0, 0.0,  A_v, 0.0, 0.0,  1.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0, 0.0],
        Ax,
    )
    # y‑component: h_y + u_y*A_u + v_y*A_v = sA
    add_equation(
        [0.0, A_u, 0.0,  0.0, A_v, 0.0,  0.0, 1.0, 0.0,  -1.0, 0.0, 0.0, 0.0, 0.0],
        0.0,
    )
    # z‑component: h_z + u_z*A_u + v_z*A_v = Az
    add_equation(
        [0.0, 0.0, A_u,  0.0, 0.0, A_v,  0.0, 0.0, 1.0,  0.0, 0.0, 0.0, 0.0, 0.0],
        Az,
    )

    # B lies on angled line: y = beta1 * x + beta0, z = B_z_um
    beta1, beta0 = pattern.beta1, pattern.beta0_um
    Bz = pattern.B_z_um
    # Let x = sB.
    # x‑component: h_x + u_x*B_u + v_x*B_v = sB
    add_equation(
        [B_u, 0.0, 0.0,  B_v, 0.0, 0.0,  1.0, 0.0, 0.0,  0.0, -1.0, 0.0, 0.0, 0.0],
        0.0,
    )
    # y‑component: h_y + u_y*B_u + v_y*B_v = beta1 * sB + beta0
    add_equation(
        [0.0, B_u, 0.0,  0.0, B_v, 0.0,  0.0, 1.0, 0.0,  0.0, -beta1, 0.0, 0.0, 0.0],
        beta0,
    )
    # z‑component: h_z + u_z*B_u + v_z*B_v = Bz
    add_equation(
        [0.0, 0.0, B_u,  0.0, 0.0, B_v,  0.0, 0.0, 1.0,  0.0, 0.0, 0.0, 0.0, 0.0],
        Bz,
    )

    # C lies on line: x = C_x_um, z = C_z_um, y = sC
    Cx, Cz = pattern.C_x_um, pattern.C_z_um
    # x‑component
    add_equation(
        [C_u, 0.0, 0.0,  C_v, 0.0, 0.0,  1.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0, 0.0],
        Cx,
    )
    # y‑component
    add_equation(
        [0.0, C_u, 0.0,  0.0, C_v, 0.0,  0.0, 1.0, 0.0,  0.0, 0.0, -1.0, 0.0, 0.0],
        0.0,
    )
    # z‑component
    add_equation(
        [0.0, 0.0, C_u,  0.0, 0.0, C_v,  0.0, 0.0, 1.0,  0.0, 0.0, 0.0, 0.0, 0.0],
        Cz,
    )

    # D lies on angled line: y = d1 * x + d0_um, z = D_z_um
    d1, d0 = pattern.d1, pattern.d0_um
    Dz = pattern.D_z_um
    # Let x = sD.
    # x‑component
    add_equation(
        [D_u, 0.0, 0.0,  D_v, 0.0, 0.0,  1.0, 0.0, 0.0,  0.0, 0.0, 0.0, -1.0, 0.0],
        0.0,
    )
    # y‑component
    add_equation(
        [0.0, D_u, 0.0,  0.0, D_v, 0.0,  0.0, 1.0, 0.0,  0.0, 0.0, 0.0, -d1, 0.0],
        d0,
    )
    # z‑component
    add_equation(
        [0.0, 0.0, D_u,  0.0, 0.0, D_v,  0.0, 0.0, 1.0,  0.0, 0.0, 0.0, 0.0, 0.0],
        Dz,
    )

    # E lies on variable‑z line:
    #   x = E_x_um
    #   y = sE
    #   z = E_z0 + E_z_delta * sE
    Ex = pattern.E_x_um
    Ez0 = pattern.E_z0
    Ez_delta = pattern.E_z_delta
    # x‑component
    add_equation(
        [E_u, 0.0, 0.0,  E_v, 0.0, 0.0,  1.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0, 0.0],
        Ex,
    )
    # y‑component
    add_equation(
        [0.0, E_u, 0.0,  0.0, E_v, 0.0,  0.0, 1.0, 0.0,  0.0, 0.0, 0.0, 0.0, -1.0],
        0.0,
    )
    # z‑component
    add_equation(
        [0.0, 0.0, E_u,  0.0, 0.0, E_v,  0.0, 0.0, 1.0,  0.0, 0.0, 0.0, 0.0, -Ez_delta],
        Ez0,
    )

    A_mat = np.asarray(rows, dtype=float)
    b_vec = np.asarray(rhs, dtype=float)

    # Solve the (slightly overdetermined) linear system in least‑squares sense.
    X, *_ = np.linalg.lstsq(A_mat, b_vec, rcond=None)

    u = X[0:3]
    v = X[3:6]
    h = X[6:9]

    return MyPlane(u, v, h)