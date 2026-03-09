"""
Photobleach pattern forward model sensitivity test. How much photobleach pattern impacts the forward model error
"""
import unittest
import numpy as np
from my_plane import MyPlane
from photobleach_pattern import PhotobleachPattern
from test_forward_model_sensitivity_utils import (
    compare_error_between_two_designs,
    verify_error_vector_is_according_to_expectation,
)


class ForwardModelSensitivity(unittest.TestCase):
    BETA1 = 0.4

    def setUp(self):
        # Setting up photobleach pattern
        AC_z_um = 0
        self.beta0_um = 0
        self.beta1 = float(self.BETA1)
        self.B_z_um = 50
        self.photobleach_pattern = PhotobleachPattern(
            # A line
            A_x_um=self.beta0_um,
            A_z_um=AC_z_um,
            # B Line
            beta1=self.beta1,
            beta0_um=self.beta0_um,  # Recommendation: use beta0 = A_x so the two lines meet
            B_z_um=self.B_z_um,
            C_x_um=200,
            C_z_um=AC_z_um
        )

        # Define the cutting plane
        u = np.array([1, 0, 0])  # um/pix [1,0,0] is 1um per pixel parallel to x
        v = np.array([0, 0, 1])  # um/pix [0,0,1] is 1um per pixel parallel to z
        h = np.array([0, 100, 0])  # um
        self.h = h

        self.my_plane = MyPlane(u, v, h)

    def test_sensitivity_to_h_changes_in_normal_direction(self):
        """ Verify that changes in h perpendicular to plane impacting B only as 1/beta """

        input_value=10 # out-plane displacement
        expected_normalized_vector=[0, 0, 1, 0, 0, 0] #Au, Av, Bu, Bv, Cu, Cv
        expected_magnitude = input_value * 1/self.beta1

        # Create and modify plane
        modified_plane = self.my_plane.copy()
        modified_plane.h[1] = modified_plane.h[1] - input_value # Moving 1 micron = 1 pixel

        normalized_vector, magnitude = compare_error_between_two_designs(
            self.photobleach_pattern, self.my_plane,
            self.photobleach_pattern, modified_plane)

        verify_error_vector_is_according_to_expectation(
            expected_normalized_vector, expected_magnitude,
            normalized_vector, magnitude
        )
        
    def test_sensitivity_to_h_changes_in_parallel_direction(self):
        """ Verify that changes in h parallel to plane impacting A,B,C by the amount of the change in h """

        input_value=[0, 10]   # in-plane displacement x,z
        expected_normalized_vector=[0, 1, 0, 1, 0, 1]
        expected_magnitude = input_value[1]  # Assuming 1um/pix, the displacement is the same

        modified_plane = self.my_plane.copy()
        modified_plane.h = modified_plane.h + np.array([input_value[0], 0, input_value[1]])

        normalized_vector, magnitude = compare_error_between_two_designs(
            self.photobleach_pattern, self.my_plane,
            self.photobleach_pattern, modified_plane)
        
        verify_error_vector_is_according_to_expectation(
            expected_normalized_vector, expected_magnitude,
            normalized_vector, magnitude
        )

        # other direction
        input_value=[10,0]   # in-plane displacement x,z
        expected_normalized_vector=[1, 0, 1, 0, 1, 0]
        expected_magnitude = input_value[0]  # Assuming 1um/pix, the displacement is the same

        modified_plane = self.my_plane.copy()
        modified_plane.h = modified_plane.h + np.array([input_value[0], 0, input_value[1]])

        normalized_vector, magnitude = compare_error_between_two_designs(
            self.photobleach_pattern, self.my_plane,
            self.photobleach_pattern, modified_plane)
        
        verify_error_vector_is_according_to_expectation(
            expected_normalized_vector, expected_magnitude,
            normalized_vector, magnitude
        )

    def test_impact_of_small_rotation_along_z_axis(self):
        """
        Verify that changes u,v such that small rotation is applied to plane impacts A,B,C
        by the amount of x distance from the origin
        """
        input_value=np.deg2rad(5) # Rotation Angle (small-angle regime)
        expected_normalized_vector=[0, 0, 1, 0, 0, 0]
        expected_magnitude=(self.h[1] - self.beta0_um) / self.beta1 * input_value * \
        (1.0 / self.beta1) * \
        (1 - (1/self.beta1 + self.beta1/2) * input_value)
            # Small angle approximation.
            # factor_1: Rotation along axis creates y displacement ΔB_y = B_x * theta
            # factor_2: This displacement is translated to pixel wise displacement by: 1/beta1.
            # factor_3: We see some beta_1 related amplification - for small betas the effect is weaker.

        theta = input_value
        modified_plane = self.my_plane.copy()
        modified_plane.u = np.array([np.cos(theta), -np.sin(theta), 0.0])
        modified_plane.v = np.array([0.0, 0.0, 1.0])

        normalized_vector, magnitude = compare_error_between_two_designs(
            self.photobleach_pattern, self.my_plane,
            self.photobleach_pattern, modified_plane)

        # print("normalized_vector:", normalized_vector)
        # print("magnitude:", magnitude)
        # print("expected_normalized_vector:", expected_normalized_vector)
        # print("expected_magnitude:", expected_magnitude)   
        
        verify_error_vector_is_according_to_expectation(
            expected_normalized_vector, expected_magnitude,
            normalized_vector, magnitude
        )

    def test_impact_of_small_rotation_along_x_axis(self):
        """
        Verify that changes u,v such that small tilt is applied to plane impacts A,B,C
        by the amount of z distance from the origin
        """
        input_value=np.deg2rad(5) # Rotation Angle (small-angle regime)
        expected_normalized_vector=[0, 0, 1, 0, 0, 0]
        expected_magnitude= (self.h[2] - self.B_z_um)  * input_value * (1.0 / self.beta1) 
    
        theta = input_value
        modified_plane = self.my_plane.copy()
        modified_plane.u = np.array([1.0, 0.0, 0.0])
        modified_plane.v = np.array([0.0, -np.sin(theta), np.cos(theta)])

        normalized_vector, magnitude = compare_error_between_two_designs(
            self.photobleach_pattern, self.my_plane,
            self.photobleach_pattern, modified_plane)
        
        verify_error_vector_is_according_to_expectation(
            expected_normalized_vector, expected_magnitude,
            normalized_vector, magnitude
        )

    def test_impact_of_small_rotation_along_y_axis(self):
        """
        Small rotation about y-axis:
            Δu ≈ -z θ
            Δv ≈  x θ
        """
        input_value=np.deg2rad(5)
        expected_normalized_vector= np.array([ 0, 0, 0, 1, 0, 1 ]) # actual for 5 deg: [ 0.,  0.,   0.245, -1.,  0.0352, -0.807]
        expected_magnitude= 21 # idk

        theta = input_value
        modified_plane = self.my_plane.copy()

        modified_plane.u = np.array([np.cos(theta), 0.0, -np.sin(theta)])
        modified_plane.v = np.array([np.sin(theta), 0.0,  np.cos(theta)])

        normalized_vector, magnitude = compare_error_between_two_designs(
            self.photobleach_pattern, self.my_plane,
            self.photobleach_pattern, modified_plane)

        print("theta:", theta)
        print("normalized_vector:", normalized_vector)
        print("magnitude:", magnitude)
        print("expected_normalized_vector:", expected_normalized_vector)
        print("expected_magnitude:", expected_magnitude)

        verify_error_vector_is_according_to_expectation(
            expected_normalized_vector, expected_magnitude,
            normalized_vector, magnitude
        )

def test_impact_of_small_scale_along_x_axis(self):
    """
    Verify that small scaling along x affects only u pixel coordinates.
    """
    input_value=0.05
    expected_normalized_vector=[1, 0, 1, 0, 1, 0]  # only u components move
    expected_magnitude=self._max_abs_x() * input_value

    eps = input_value
    modified_plane = self.my_plane.copy()

    # Scale x component of basis
    modified_plane.u = np.array([1.0 + eps, 0.0, 0.0])
    modified_plane.v = np.array([0.0, 0.0, 1.0])

    normalized_vector, magnitude = compare_error_between_two_designs(
        self.photobleach_pattern, self.my_plane,
        self.photobleach_pattern, modified_plane)

    verify_error_vector_is_according_to_expectation(
            expected_normalized_vector, expected_magnitude,
            normalized_vector, magnitude
    )

def test_impact_of_small_scale_along_z_axis(self):
    """
    Small scaling along z affects only v pixel coordinates.
    """
    input_value=0.05
    expected_normalized_vector=[0, 1, 0, 1, 0, 1]
    expected_magnitude=self._max_abs_z() * input_value

    eps = input_value
    modified_plane = self.my_plane.copy()

    modified_plane.u = np.array([1.0, 0.0, 0.0])
    modified_plane.v = np.array([0.0, 0.0, 1.0 + eps])

    normalized_vector, magnitude = compare_error_between_two_designs(
        self.photobleach_pattern, self.my_plane,
        self.photobleach_pattern, modified_plane)

    verify_error_vector_is_according_to_expectation(
            expected_normalized_vector, expected_magnitude,
            normalized_vector, magnitude
    )
