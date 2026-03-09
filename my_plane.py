import numpy as np


def _check_dim(vec):
    if not isinstance(vec, np.ndarray) or vec.shape != (3,):
        raise ValueError("Expected a 3-element vector (1D NumPy array).")


class MyPlane:
    def __init__(self, u: np.array, v: np.array, h: np.array):
        """ Initialize from plane parameters u(vec), v(vec), h(vec)"""

        # Check dimensions
        _check_dim(u)
        _check_dim(v)
        _check_dim(h)

        # Check orthogonality
        dot_product = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
        if abs(dot_product) > 0.2:
            raise ValueError("u and v are not orthogonal.")

        # Store
        self.u = u
        self.v = v
        self.h = h

    def pix_to_physical (self, u_pix: float, v_pix: float) -> np.array:
        """ Convert pixel coordinates to physical coordinates """
        return u_pix * self.u + v_pix * self.v + self.h

    def physical_to_pix (self, pt: np.array) -> (float, float):
        """
        Convert physical coordinates [x,y,z] to pixel coordinates
        Returns (u_pix, v_pix)
        """

        _check_dim(pt)

        rhs = pt - self.h
        uu = np.dot(self.u, self.u)
        uv = np.dot(self.u, self.v)
        vv = np.dot(self.v, self.v)
        ru = np.dot(rhs, self.u)
        rv = np.dot(rhs, self.v)

        # Check normal is well-defined
        _ = self.normal_vector
        det = uu * vv - uv * uv

        u_pix = (ru * vv - rv * uv) / det
        v_pix = (rv * uu - ru * uv) / det
        return float(u_pix), float(v_pix)

    @property
    def normal_vector(self) -> np.array:
        normal = np.cross(self.u, self.v)
        normal_norm = np.linalg.norm(normal)
        if normal_norm < 1e-6:
            raise ValueError("Plane basis vectors 'u' and 'v' must be linearly independent.")

        return  normal/normal_norm

    def copy(self) -> "MyPlane":
        """Create an independent copy of this plane."""
        return MyPlane(np.copy(self.u), np.copy(self.v), np.copy(self.h))

    def distance_to_point(self, pt: np.array) -> float:
        """Compute Euclidean distance from point pt to this plane."""
        _check_dim(pt)

        return float(abs(np.dot(pt - self.h, self.normal_vector)))

    def compute_line_intersection_angled(self, beta0: float, beta1: float, z0: float) -> np.array:
        """
        Compute the physical position of intersection of a line of the form [x, y=beta1*x+beta0, z0] with the plane
        """

        nx, ny, nz = self.normal_vector
        hx, hy, hz = self.h

        x = (nx * hx + ny * (hy - beta0) - nz * (z0 - hz)) / (nx + ny * beta1)
        y = beta1 * x + beta0
        return np.array([x, y, z0])

    def compute_line_intersection_parallel_x(self, x0: float, z0: float) -> np.array:
        """
        Compute the physical position of intersection of a line of the form [x0, y, z0] with the plane
        """

        nx, ny, nz = self.normal_vector
        hx, hy, hz = self.h

        y = hy - (nx * (x0 - hx) + nz * (z0 - hz)) / ny
        return np.array([x0, y, z0])
