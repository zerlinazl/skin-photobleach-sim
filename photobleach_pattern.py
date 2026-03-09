from dataclasses import dataclass
import numpy as np
from my_plane import MyPlane

class PhotobleachPattern:
    def __init__(
            self,
            A_x_um: float, A_z_um: float,
            beta1: float, beta0_um: float, B_z_um: float,
            C_x_um: float, C_z_um: float):
        """
        Initialize the triangle pattern defined here:
        pt_parallel_x = plane.compute_line_intersection_parallel_x(x0, z0)
        """

        self.A_x_um = A_x_um
        self.A_z_um = A_z_um
        self.beta1 = beta1
        self.beta0_um = beta0_um
        self.B_z_um = B_z_um
        self.C_x_um = C_x_um
        self.C_z_um = C_z_um

    def copy(self) -> "PhotobleachPattern":
        """Create an independent copy of this photobleach pattern."""
        return PhotobleachPattern(
            A_x_um=float(self.A_x_um),
            A_z_um=float(self.A_z_um),
            beta1=float(self.beta1),
            beta0_um=float(self.beta0_um),
            B_z_um=float(self.B_z_um),
            C_x_um=float(self.C_x_um),
            C_z_um=float(self.C_z_um),
        )

    def forward_model_parametric(self, my_plane: MyPlane, A_u_pix, A_v_pix, B_u_pix, B_v_pix, C_u_pix, C_v_pix) -> (
            np.array, np.array, np.array, np.array, np.array, np.array):
        """
        Accepts a plane and pixel positions of the intersection and returns the calculation of intersection
        points A,B,C
        """

        A_parametric = my_plane.pix_to_physical(A_u_pix, A_v_pix)
        B_parametric = my_plane.pix_to_physical(B_u_pix, B_v_pix)
        C_parametric = my_plane.pix_to_physical(C_u_pix, C_v_pix)

        return A_parametric, B_parametric, C_parametric

    def forward_model_nonparametric(self, my_plane: MyPlane):
        """
        Accepts a plane  returns the calculation of intersection
        points A,B,C
        """

        A_nonparametric = my_plane.compute_line_intersection_parallel_x(self.A_x_um, self.A_z_um)
        B_nonparametric = my_plane.compute_line_intersection_angled(self.beta0_um, self.beta1, self.B_z_um)
        C_nonparametric = my_plane.compute_line_intersection_parallel_x(self.C_x_um, self.C_z_um)

        return A_nonparametric, B_nonparametric, C_nonparametric



class PhotobleachPattern4:
    def __init__(
            self,
            A_x_um: float, A_z_um: float,
            beta1: float, beta0_um: float, B_z_um: float,
            C_x_um: float, C_z_um: float,
            d1: float, d0_um: float, D_z_um: float):
        """
        Initialize the triangle pattern defined here:
        pt_parallel_x = plane.compute_line_intersection_parallel_x(x0, z0)
        """

        self.A_x_um = A_x_um
        self.A_z_um = A_z_um
        self.beta1 = beta1
        self.beta0_um = beta0_um
        self.B_z_um = B_z_um
        self.C_x_um = C_x_um
        self.C_z_um = C_z_um
        self.d1 = d1
        self.d0_um = d0_um
        self.D_z_um = D_z_um

    def copy(self) -> "PhotobleachPattern":
        """Create an independent copy of this photobleach pattern."""
        return PhotobleachPattern(
            A_x_um=float(self.A_x_um),
            A_z_um=float(self.A_z_um),
            beta1=float(self.beta1),
            beta0_um=float(self.beta0_um),
            B_z_um=float(self.B_z_um),
            C_x_um=float(self.C_x_um),
            C_z_um=float(self.C_z_um),
            d1=float(self.d1),
            d0_um=float(self.d0_um),
            D_z_um=float(self.D_z_um),
        )

    def forward_model_parametric(self, my_plane: MyPlane, A_u_pix, A_v_pix, B_u_pix, B_v_pix, C_u_pix, C_v_pix, D_u_pix, D_v_pix) -> (
            np.array, np.array, np.array, np.array, np.array, np.array, np.array, np.array):
        """
        Accepts a plane and pixel positions of the intersection and returns the calculation of intersection
        points A,B,C
        """

        A_parametric = my_plane.pix_to_physical(A_u_pix, A_v_pix)
        B_parametric = my_plane.pix_to_physical(B_u_pix, B_v_pix)
        C_parametric = my_plane.pix_to_physical(C_u_pix, C_v_pix)
        D_parametric = my_plane.pix_to_physical(D_u_pix, D_v_pix)

        return A_parametric, B_parametric, C_parametric, D_parametric

    def forward_model_nonparametric(self, my_plane: MyPlane):
        """
        Accepts a plane  returns the calculation of intersection
        points A,B,C,D
        """

        A_nonparametric = my_plane.compute_line_intersection_parallel_x(self.A_x_um, self.A_z_um)
        B_nonparametric = my_plane.compute_line_intersection_angled(self.beta0_um, self.beta1, self.B_z_um)
        C_nonparametric = my_plane.compute_line_intersection_parallel_x(self.C_x_um, self.C_z_um)
        D_nonparametric = my_plane.compute_line_intersection_angled(self.d0_um, self.d1, self.D_z_um)

        return A_nonparametric, B_nonparametric, C_nonparametric, D_nonparametric
