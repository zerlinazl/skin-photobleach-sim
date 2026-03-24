import numpy as np
from my_plane import MyPlane
from scipy.linalg import qr
from photobleach_pattern import PhotobleachPattern, PhotobleachPattern4, PhotobleachPatternVarZC, PhotobleachPattern4Z, PhotobleachPattern5, PhotobleachPattern5a, PhotobleachPattern5Z, PhotobleachPattern5Z2
from photobleach_pattern_4 import PhotobleachPattern4a
from photobleach_pattern_5_split import PhotobleachPattern5split
from test_forward_model_sensitivity_utils import (
    compare_error_between_two_designs,
    compare_error_between_two_designs4,
    compare_error_between_two_designs5,
    verify_error_vector_is_according_to_expectation,
)
from rank_lines_utils import (
    get_normalized_vector_and_magnitude_for_y_translation,
    get_normalized_vector_and_magnitude_for_z_translation,
    get_normalized_vector_and_magnitude_for_x_translation,
    get_normalized_vector_and_magnitude_for_z_rotation,
    get_normalized_vector_and_magnitude_for_x_rotation,
    get_normalized_vector_and_magnitude_for_y_rotation,
    get_normalized_vector_and_magnitude_for_scale_x,
    get_normalized_vector_and_magnitude_for_scale_z,
    get_normalized_vector_and_magnitude_for_shear_xz,
    get_rank,
    round_to_bins,
    find_dependent_columns,
    find_column_combination,
    find_identical_columns
)

class RankLines3():
    def setup(self):
        self.BETA1 = 1

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
            C_z_um=AC_z_um,
        )


        # Define the cutting plane
        u = np.array([1, 0, 0])  # um/pix [1,0,0] is 1um per pixel parallel to x
        v = np.array([0, 0, 1])  # um/pix [0,0,1] is 1um per pixel parallel to z
        h = np.array([0, 80, 0])  # um
        self.h = h

        self.my_plane = MyPlane(u, v, h)

        self.transforms = {
            'deg_rot_x': 5,
            'deg_rot_y': 5,
            'deg_rot_z': 5,
            'scale_x': 1.2,
            'scale_z': 1.2,
            'translation_x': 10,
            'translation_y': 10,
            'translation_z': 10,
            'shear_xz': 0.2
        }

        self.bins = [-1, -0.5, 0, 0.5, 1]
        self.M = self.get_M()
        self.M_binned = round_to_bins(self.M, self.bins)

        self.transforms_list = [
            'translation_y',
            'translation_z',
            'translation_x',
            'x_rotation',
            'y_rotation',
            'z_rotation',
            'scale_x',
            'scale_z',
            'shear_xz'
        ]

    def get_M(self):
        theta_x = self.transforms['deg_rot_x']
        theta_y = self.transforms['deg_rot_y']
        theta_z = self.transforms['deg_rot_z']
        scale_x = self.transforms['scale_x']
        scale_z = self.transforms['scale_z']
        translation_x = self.transforms['translation_x']
        translation_y = self.transforms['translation_y']
        translation_z = self.transforms['translation_z']
        shear_xz = self.transforms['shear_xz']

        normalized_vector_y_translation, magnitude_y_translation = get_normalized_vector_and_magnitude_for_y_translation(translation_y, self.my_plane, self.photobleach_pattern, 3)
        normalized_vector_z_translation, magnitude_z_translation = get_normalized_vector_and_magnitude_for_z_translation(translation_z, self.my_plane, self.photobleach_pattern, 3)
        normalized_vector_x_translation, magnitude_x_translation = get_normalized_vector_and_magnitude_for_x_translation(translation_x, self.my_plane, self.photobleach_pattern, 3)
        normalized_vector_x_rotation, magnitude_x_rotation = get_normalized_vector_and_magnitude_for_x_rotation(theta_x, self.my_plane, self.photobleach_pattern, 3)
        normalized_vector_y_rotation, magnitude_y_rotation = get_normalized_vector_and_magnitude_for_y_rotation(theta_y, self.my_plane, self.photobleach_pattern, 3)
        normalized_vector_z_rotation, magnitude_z_rotation = get_normalized_vector_and_magnitude_for_z_rotation(theta_z, self.my_plane, self.photobleach_pattern, 3)
        normalized_vector_scale_x, magnitude_scale_x = get_normalized_vector_and_magnitude_for_scale_x(scale_x, self.my_plane, self.photobleach_pattern, 3)
        normalized_vector_scale_z, magnitude_scale_z = get_normalized_vector_and_magnitude_for_scale_z(scale_z, self.my_plane, self.photobleach_pattern, 3)
        normalized_vector_shear_xz, magnitude_shear_xz = get_normalized_vector_and_magnitude_for_shear_xz(shear_xz, self.my_plane, self.photobleach_pattern, 3)

        M_all = np.column_stack([
            normalized_vector_y_translation,
            normalized_vector_z_translation,
            normalized_vector_x_translation,
            normalized_vector_x_rotation,
            normalized_vector_y_rotation,
            normalized_vector_z_rotation,
            normalized_vector_scale_x,
            normalized_vector_scale_z,
            normalized_vector_shear_xz])

        # print(M_all)
        # print("--------------------------------")
        return M_all

    

class RankLines4():
    def setup(self):
        self.BETA1 = 1

        AC_z_um = 0
        self.beta0_um = 0
        self.beta1 = float(self.BETA1)
        self.d1 = -self.beta1
        self.d0_um = 200 * self.beta1
        self.D_z_um = 30
        self.B_z_um = 50
        self.photobleach_pattern = PhotobleachPattern4(
            # A line
            A_x_um=self.beta0_um,
            A_z_um=AC_z_um,
            # B Line
            beta1=self.beta1,
            beta0_um=self.beta0_um,  # Recommendation: use beta0 = A_x so the two lines meet
            B_z_um=self.B_z_um,
            C_x_um=200,
            C_z_um=AC_z_um,
            d1=self.d1,
            d0_um=self.d0_um,
            D_z_um=self.D_z_um
        )


        # Define the cutting plane
        u = np.array([1, 0, 0])  # um/pix [1,0,0] is 1um per pixel parallel to x
        v = np.array([0, 0, 1])  # um/pix [0,0,1] is 1um per pixel parallel to z
        h = np.array([0, 80, 0])  # um
        self.h = h

        self.my_plane = MyPlane(u, v, h)

        self.transforms = {
            'deg_rot_x': 5,
            'deg_rot_y': 5,
            'deg_rot_z': 5,
            'scale_x': 1.1,
            'scale_z': 1.1,
            'translation_x': 10,
            'translation_y': 10,
            'translation_z': 10,
            'shear_xz': 0.2
        }

        self.bins = [-1, -0.5, 0, 0.5, 1]
        self.M = self.get_M()
        self.M_binned = round_to_bins(self.M, self.bins)

        self.transforms_list = [
            'translation_y',
            'translation_z',
            'translation_x',
            'x_rotation',
            'y_rotation',
            'z_rotation',
            'scale_x',
            'scale_z',
            'shear_xz'
        ]

    def get_M(self):
        theta_x = self.transforms['deg_rot_x']
        theta_y = self.transforms['deg_rot_y']
        theta_z = self.transforms['deg_rot_z']
        scale_x = self.transforms['scale_x']
        scale_z = self.transforms['scale_z']
        translation_x = self.transforms['translation_x']
        translation_y = self.transforms['translation_y']
        translation_z = self.transforms['translation_z']
        shear_xz = self.transforms['shear_xz']

        normalized_vector_y_translation, magnitude_y_translation = get_normalized_vector_and_magnitude_for_y_translation(translation_y, self.my_plane, self.photobleach_pattern, 4)
        normalized_vector_z_translation, magnitude_z_translation = get_normalized_vector_and_magnitude_for_z_translation(translation_z, self.my_plane, self.photobleach_pattern, 4)
        normalized_vector_x_translation, magnitude_x_translation = get_normalized_vector_and_magnitude_for_x_translation(translation_x, self.my_plane, self.photobleach_pattern, 4)
        normalized_vector_x_rotation, magnitude_x_rotation = get_normalized_vector_and_magnitude_for_x_rotation(theta_x, self.my_plane, self.photobleach_pattern, 4)
        normalized_vector_y_rotation, magnitude_y_rotation = get_normalized_vector_and_magnitude_for_y_rotation(theta_y, self.my_plane, self.photobleach_pattern, 4)
        normalized_vector_z_rotation, magnitude_z_rotation = get_normalized_vector_and_magnitude_for_z_rotation(theta_z, self.my_plane, self.photobleach_pattern, 4)
        normalized_vector_scale_x, magnitude_scale_x = get_normalized_vector_and_magnitude_for_scale_x(scale_x, self.my_plane, self.photobleach_pattern, 4)
        normalized_vector_scale_z, magnitude_scale_z = get_normalized_vector_and_magnitude_for_scale_z(scale_z, self.my_plane, self.photobleach_pattern, 4)
        normalized_vector_shear_xz, magnitude_shear_xz = get_normalized_vector_and_magnitude_for_shear_xz(shear_xz, self.my_plane, self.photobleach_pattern, 4)

        M_all = np.column_stack([
            normalized_vector_y_translation,
            normalized_vector_z_translation,
            normalized_vector_x_translation,
            normalized_vector_x_rotation,
            normalized_vector_y_rotation,
            normalized_vector_z_rotation,
            normalized_vector_scale_x,
            normalized_vector_scale_z,
            normalized_vector_shear_xz])

        # print(M_all)
        # print("--------------------------------")
        return M_all


class RankLines3Z():
    def setup(self):
        self.BETA1 = 10

        AC_z_um = 0
        self.beta0_um = 0
        self.beta1 = float(self.BETA1)
        self.B_z_um = 50
        self.photobleach_pattern = PhotobleachPatternVarZC(
            # A line
            A_x_um=self.beta0_um,
            A_z_um=AC_z_um,
            # B Line
            beta1=self.beta1,
            beta0_um=self.beta0_um,  # Recommendation: use beta0 = A_x so the two lines meet
            B_z_um=self.B_z_um,
            C_x_um=200,
            C_z_delta=1.0
        )


        # Define the cutting plane
        u = np.array([1, 0, 0])  # um/pix [1,0,0] is 1um per pixel parallel to x
        v = np.array([0, 0, 1])  # um/pix [0,0,1] is 1um per pixel parallel to z
        h = np.array([0, 80, 0])  # um
        self.h = h

        self.my_plane = MyPlane(u, v, h)

        self.transforms = {
            'deg_rot_x': 5,
            'deg_rot_y': 5,
            'deg_rot_z': 5,
            'scale_x': 1.1,
            'scale_z': 1.1,
            'translation_x': 10,
            'translation_y': 10,
            'translation_z': 10,
            'shear_xz': 0.2
        }

        self.bins = [-1, -0.5, 0, 0.5, 1]
        self.M = self.get_M()
        self.M_binned = round_to_bins(self.M, self.bins)

        self.transforms_list = [
            'translation_y',
            'translation_z',
            'translation_x',
            'x_rotation',
            'y_rotation',
            'z_rotation',
            'scale_x',
            'scale_z',
            'shear_xz'
        ]

    def get_M(self):
        theta_x = self.transforms['deg_rot_x']
        theta_y = self.transforms['deg_rot_y']
        theta_z = self.transforms['deg_rot_z']
        scale_x = self.transforms['scale_x']
        scale_z = self.transforms['scale_z']
        translation_x = self.transforms['translation_x']
        translation_y = self.transforms['translation_y']
        translation_z = self.transforms['translation_z']
        shear_xz = self.transforms['shear_xz']

        normalized_vector_y_translation, magnitude_y_translation = get_normalized_vector_and_magnitude_for_y_translation(translation_y, self.my_plane, self.photobleach_pattern, 3)
        normalized_vector_z_translation, magnitude_z_translation = get_normalized_vector_and_magnitude_for_z_translation(translation_z, self.my_plane, self.photobleach_pattern, 3)
        normalized_vector_x_translation, magnitude_x_translation = get_normalized_vector_and_magnitude_for_x_translation(translation_x, self.my_plane, self.photobleach_pattern, 3)
        normalized_vector_x_rotation, magnitude_x_rotation = get_normalized_vector_and_magnitude_for_x_rotation(theta_x, self.my_plane, self.photobleach_pattern, 3)
        normalized_vector_y_rotation, magnitude_y_rotation = get_normalized_vector_and_magnitude_for_y_rotation(theta_y, self.my_plane, self.photobleach_pattern, 3)
        normalized_vector_z_rotation, magnitude_z_rotation = get_normalized_vector_and_magnitude_for_z_rotation(theta_z, self.my_plane, self.photobleach_pattern, 3)
        normalized_vector_scale_x, magnitude_scale_x = get_normalized_vector_and_magnitude_for_scale_x(scale_x, self.my_plane, self.photobleach_pattern, 3)
        normalized_vector_scale_z, magnitude_scale_z = get_normalized_vector_and_magnitude_for_scale_z(scale_z, self.my_plane, self.photobleach_pattern, 3)
        normalized_vector_shear_xz, magnitude_shear_xz = get_normalized_vector_and_magnitude_for_shear_xz(shear_xz, self.my_plane, self.photobleach_pattern, 3)

        M_all = np.column_stack([
            normalized_vector_y_translation,
            normalized_vector_z_translation,
            normalized_vector_x_translation,
            normalized_vector_x_rotation,
            normalized_vector_y_rotation,
            normalized_vector_z_rotation,
            normalized_vector_scale_x,
            normalized_vector_scale_z,
            normalized_vector_shear_xz])

        # print(M_all)
        # print("--------------------------------")
        return M_all


class RankLines4Z():
    def setup(self):
        self.BETA1 = 1

        AC_z_um = 0
        self.beta0_um = 0
        self.beta1 = float(self.BETA1)
        self.d1 = -self.beta1
        self.d0_um = 200 * self.beta1
        self.D_z_um = 30
        self.B_z_um = 50
        self.photobleach_pattern = PhotobleachPattern4Z(
            # A line
            A_x_um=self.beta0_um,
            A_z_um=AC_z_um,
            # B Line
            beta1=self.beta1,
            beta0_um=self.beta0_um,  # Recommendation: use beta0 = A_x so the two lines meet
            B_z_um=self.B_z_um,
            C_x_um=200,
            C_z_delta=1,
            d1=self.d1,
            d0_um=self.d0_um,
            D_z_um=self.D_z_um
        )


        # Define the cutting plane
        u = np.array([1, 0, 0])  # um/pix [1,0,0] is 1um per pixel parallel to x
        v = np.array([0, 0, 1])  # um/pix [0,0,1] is 1um per pixel parallel to z
        h = np.array([0, 80, 0])  # um
        self.h = h

        self.my_plane = MyPlane(u, v, h)

        self.transforms = {
            'deg_rot_x': 5,
            'deg_rot_y': 5,
            'deg_rot_z': 5,
            'scale_x': 1.1,
            'scale_z': 1.1,
            'translation_x': 10,
            'translation_y': 10,
            'translation_z': 10,
            'shear_xz': 0.2
        }

        self.bins = [-1, -0.5, 0, 0.5, 1]
        self.M = self.get_M()
        self.M_binned = round_to_bins(self.M, self.bins)

        self.transforms_list = [
            'translation_y',
            'translation_z',
            'translation_x',
            'x_rotation',
            'y_rotation',
            'z_rotation',
            'scale_x',
            'scale_z',
            'shear_xz'
        ]

    def get_M(self):
        theta_x = self.transforms['deg_rot_x']
        theta_y = self.transforms['deg_rot_y']
        theta_z = self.transforms['deg_rot_z']
        scale_x = self.transforms['scale_x']
        scale_z = self.transforms['scale_z']
        translation_x = self.transforms['translation_x']
        translation_y = self.transforms['translation_y']
        translation_z = self.transforms['translation_z']
        shear_xz = self.transforms['shear_xz']

        normalized_vector_y_translation, magnitude_y_translation = get_normalized_vector_and_magnitude_for_y_translation(translation_y, self.my_plane, self.photobleach_pattern, 4)
        normalized_vector_z_translation, magnitude_z_translation = get_normalized_vector_and_magnitude_for_z_translation(translation_z, self.my_plane, self.photobleach_pattern, 4)
        normalized_vector_x_translation, magnitude_x_translation = get_normalized_vector_and_magnitude_for_x_translation(translation_x, self.my_plane, self.photobleach_pattern, 4)
        normalized_vector_x_rotation, magnitude_x_rotation = get_normalized_vector_and_magnitude_for_x_rotation(theta_x, self.my_plane, self.photobleach_pattern, 4)
        normalized_vector_y_rotation, magnitude_y_rotation = get_normalized_vector_and_magnitude_for_y_rotation(theta_y, self.my_plane, self.photobleach_pattern, 4)
        normalized_vector_z_rotation, magnitude_z_rotation = get_normalized_vector_and_magnitude_for_z_rotation(theta_z, self.my_plane, self.photobleach_pattern, 4)
        normalized_vector_scale_x, magnitude_scale_x = get_normalized_vector_and_magnitude_for_scale_x(scale_x, self.my_plane, self.photobleach_pattern, 4)
        normalized_vector_scale_z, magnitude_scale_z = get_normalized_vector_and_magnitude_for_scale_z(scale_z, self.my_plane, self.photobleach_pattern, 4)
        normalized_vector_shear_xz, magnitude_shear_xz = get_normalized_vector_and_magnitude_for_shear_xz(shear_xz, self.my_plane, self.photobleach_pattern, 4)
        
        M_all = np.column_stack([
            normalized_vector_y_translation,
            normalized_vector_z_translation,
            normalized_vector_x_translation,
            normalized_vector_x_rotation,
            normalized_vector_y_rotation,
            normalized_vector_z_rotation,
            normalized_vector_scale_x,
            normalized_vector_scale_z,
            normalized_vector_shear_xz])

        # print(M_all)
        # print("--------------------------------")
        return M_all


class RankLines5():
    def setup(self):
        self.BETA1 = 1

        AC_z_um = 0
        self.beta0_um = 0
        self.beta1 = float(self.BETA1)
        self.d1 = -self.beta1
        self.d0_um = 200 * self.beta1
        self.D_z_um = 30
        self.B_z_um = 50
        self.e1 = self.beta1
        self.e0_um = 20
        self.E_z_um = 15

        self.photobleach_pattern = PhotobleachPattern5(
            # A line
            A_x_um=self.beta0_um,
            A_z_um=AC_z_um,
            # B Line
            beta1=self.beta1,
            beta0_um=self.beta0_um,  # Recommendation: use beta0 = A_x so the two lines meet
            B_z_um=self.B_z_um,
            C_x_um=200,
            C_z_um=AC_z_um,
            d1=self.d1,
            d0_um=self.d0_um,
            D_z_um=self.D_z_um,
            e1=self.beta1 + 0.2,
            e0_um=self.e0_um,
            E_z_um=self.E_z_um
        )


        # Define the cutting plane
        u = np.array([1, 0, 0])  # um/pix [1,0,0] is 1um per pixel parallel to x
        v = np.array([0, 0, 1])  # um/pix [0,0,1] is 1um per pixel parallel to z
        h = np.array([0, 80, 0])  # um
        self.h = h

        self.my_plane = MyPlane(u, v, h)

        self.transforms = {
            'deg_rot_x': 5,
            'deg_rot_y': 5,
            'deg_rot_z': 5,
            'scale_x': 1.1,
            'scale_z': 1.1,
            'translation_x': 10,
            'translation_y': 10,
            'translation_z': 10,
            'shear_xz': 0.2
        }

        self.bins = [-1, -0.5, 0, 0.5, 1]
        self.M = self.get_M()
        self.M_binned = round_to_bins(self.M, self.bins)

        self.transforms_list = [
            'translation_y',
            'translation_z',
            'translation_x',
            'x_rotation',
            'y_rotation',
            'z_rotation',
            'scale_x',
            'scale_z',
            'shear_xz'
        ]

    def get_M(self):
        theta_x = self.transforms['deg_rot_x']
        theta_y = self.transforms['deg_rot_y']
        theta_z = self.transforms['deg_rot_z']
        scale_x = self.transforms['scale_x']
        scale_z = self.transforms['scale_z']
        translation_x = self.transforms['translation_x']
        translation_y = self.transforms['translation_y']
        translation_z = self.transforms['translation_z']
        shear_xz = self.transforms['shear_xz']

        normalized_vector_y_translation, magnitude_y_translation = get_normalized_vector_and_magnitude_for_y_translation(translation_y, self.my_plane, self.photobleach_pattern, 5)
        normalized_vector_z_translation, magnitude_z_translation = get_normalized_vector_and_magnitude_for_z_translation(translation_z, self.my_plane, self.photobleach_pattern, 5)
        normalized_vector_x_translation, magnitude_x_translation = get_normalized_vector_and_magnitude_for_x_translation(translation_x, self.my_plane, self.photobleach_pattern, 5)
        normalized_vector_x_rotation, magnitude_x_rotation = get_normalized_vector_and_magnitude_for_x_rotation(theta_x, self.my_plane, self.photobleach_pattern, 5)
        normalized_vector_y_rotation, magnitude_y_rotation = get_normalized_vector_and_magnitude_for_y_rotation(theta_y, self.my_plane, self.photobleach_pattern, 5)
        normalized_vector_z_rotation, magnitude_z_rotation = get_normalized_vector_and_magnitude_for_z_rotation(theta_z, self.my_plane, self.photobleach_pattern, 5)
        normalized_vector_scale_x, magnitude_scale_x = get_normalized_vector_and_magnitude_for_scale_x(scale_x, self.my_plane, self.photobleach_pattern, 5)
        normalized_vector_scale_z, magnitude_scale_z = get_normalized_vector_and_magnitude_for_scale_z(scale_z, self.my_plane, self.photobleach_pattern, 5)
        normalized_vector_shear_xz, magnitude_shear_xz = get_normalized_vector_and_magnitude_for_shear_xz(shear_xz, self.my_plane, self.photobleach_pattern, 5)

        M_all = np.column_stack([
            normalized_vector_y_translation,
            normalized_vector_z_translation,
            normalized_vector_x_translation,
            normalized_vector_x_rotation,
            normalized_vector_y_rotation,
            normalized_vector_z_rotation,
            normalized_vector_scale_x,
            normalized_vector_scale_z,
            normalized_vector_shear_xz])

        # print(M_all)
        # print("--------------------------------")
        return M_all


class RankLines5a():
    def setup(self):
        # self.A_z_um = 0
        # self.A_x_um = 0

        # self.beta0_um = 0
        # self.beta1 = 1
        # self.B_z_um = 50

        # self.d1 = -self.beta1
        # self.d0_um = 200 * self.beta1
        # self.D_z_um = 30

        # self.C_z_um = 0
        # self.C_x_um = 200

        # self.E_x_um = 250
        # self.E_z_um = 15

        self.A_z_um = 0
        self.A_x_um = 0

        self.beta0_um = 0
        self.beta1 = 1
        self.B_z_um = 50

        self.d1 = -self.beta1
        self.d0_um = 200 * self.beta1
        self.D_z_um = 50

        self.C_z_um = 0
        self.C_x_um = 200

        self.E_x_um = 250
        self.E_z_um = 15

        self.photobleach_pattern = PhotobleachPattern5a(
            # A line
            A_x_um=self.beta0_um,
            A_z_um=self.A_z_um,
            # B Line
            beta1=self.beta1,
            beta0_um=self.beta0_um,  # Recommendation: use beta0 = A_x so the two lines meet
            B_z_um=self.B_z_um,
            C_x_um=self.C_x_um,
            C_z_um=self.C_z_um,
            d1=self.d1,
            d0_um=self.d0_um,
            D_z_um=self.D_z_um,
            E_x_um=self.E_x_um,
            E_z_um=self.E_z_um
        )


        # Define the cutting plane
        u = np.array([1, 0, 0])  # um/pix [1,0,0] is 1um per pixel parallel to x
        v = np.array([0, 0, 1])  # um/pix [0,0,1] is 1um per pixel parallel to z
        h = np.array([0, 40, 0])  # um
        self.h = h

        self.my_plane = MyPlane(u, v, h)

        self.transforms = {
            'deg_rot_x': 5,
            'deg_rot_y': 5,
            'deg_rot_z': 5,
            'scale_x': 1.1,
            'scale_z': 1.1,
            'translation_x': 10,
            'translation_y': 10,
            'translation_z': 10,
            'shear_xz': 0.2
        }

        self.bins = [-1, -0.5, 0, 0.5, 1]
        self.M = self.get_M()
        self.M_binned = round_to_bins(self.M, self.bins)

        self.transforms_list = [
            'translation_y',
            'translation_z',
            'translation_x',
            'x_rotation',
            'y_rotation',
            'z_rotation',
            'scale_x',
            'scale_z',
            'shear_xz'
        ]

    def get_M(self):
        theta_x = self.transforms['deg_rot_x']
        theta_y = self.transforms['deg_rot_y']
        theta_z = self.transforms['deg_rot_z']
        scale_x = self.transforms['scale_x']
        scale_z = self.transforms['scale_z']
        translation_x = self.transforms['translation_x']
        translation_y = self.transforms['translation_y']
        translation_z = self.transforms['translation_z']
        shear_xz = self.transforms['shear_xz']

        normalized_vector_y_translation, magnitude_y_translation = get_normalized_vector_and_magnitude_for_y_translation(translation_y, self.my_plane, self.photobleach_pattern, 5)
        normalized_vector_z_translation, magnitude_z_translation = get_normalized_vector_and_magnitude_for_z_translation(translation_z, self.my_plane, self.photobleach_pattern, 5)
        normalized_vector_x_translation, magnitude_x_translation = get_normalized_vector_and_magnitude_for_x_translation(translation_x, self.my_plane, self.photobleach_pattern, 5)
        normalized_vector_x_rotation, magnitude_x_rotation = get_normalized_vector_and_magnitude_for_x_rotation(theta_x, self.my_plane, self.photobleach_pattern, 5)
        normalized_vector_y_rotation, magnitude_y_rotation = get_normalized_vector_and_magnitude_for_y_rotation(theta_y, self.my_plane, self.photobleach_pattern, 5)
        normalized_vector_z_rotation, magnitude_z_rotation = get_normalized_vector_and_magnitude_for_z_rotation(theta_z, self.my_plane, self.photobleach_pattern, 5)
        normalized_vector_scale_x, magnitude_scale_x = get_normalized_vector_and_magnitude_for_scale_x(scale_x, self.my_plane, self.photobleach_pattern, 5)
        normalized_vector_scale_z, magnitude_scale_z = get_normalized_vector_and_magnitude_for_scale_z(scale_z, self.my_plane, self.photobleach_pattern, 5)
        normalized_vector_shear_xz, magnitude_shear_xz = get_normalized_vector_and_magnitude_for_shear_xz(shear_xz, self.my_plane, self.photobleach_pattern, 5)

        M_all = np.column_stack([
            normalized_vector_y_translation,
            normalized_vector_z_translation,
            normalized_vector_x_translation,
            normalized_vector_x_rotation,
            normalized_vector_y_rotation,
            normalized_vector_z_rotation,
            normalized_vector_scale_x,
            normalized_vector_scale_z,
            normalized_vector_shear_xz])

        # print(M_all)
        # print("--------------------------------")
        return M_all


class RankLines5Z():
    def setup(self):
        self.BETA1 = 1

        AC_z_um = 0
        self.beta0_um = 0
        self.beta1 = float(self.BETA1)
        self.C_z_delta = 1
        self.C_x_um = 200
        self.d1 = -self.beta1
        self.d0_um = 200 * self.beta1
        self.D_z_um = 30
        self.B_z_um = 50
        self.e1 = self.beta1 + 0.2
        self.e0_um = 20
        self.E_z_um = 15
        self.photobleach_pattern = PhotobleachPattern5Z(
            # A line
            A_x_um=self.beta0_um,
            A_z_um=AC_z_um,
            # B Line
            beta1=self.beta1,
            beta0_um=self.beta0_um,  # Recommendation: use beta0 = A_x so the two lines meet
            B_z_um=self.B_z_um,
            C_x_um=self.C_x_um,
            C_z_delta=self.C_z_delta,
            d1=self.d1,
            d0_um=self.d0_um,
            D_z_um=self.D_z_um,
            e1=self.e1,
            e0_um=self.e0_um,
            E_z_um=self.E_z_um
        )

        # Define the cutting plane
        u = np.array([1, 0, 0])  # um/pix [1,0,0] is 1um per pixel parallel to x
        v = np.array([0, 0, 1])  # um/pix [0,0,1] is 1um per pixel parallel to z
        h = np.array([0, 80, 0])  # um
        self.h = h

        self.my_plane = MyPlane(u, v, h)

        self.transforms = {
            'deg_rot_x': 5,
            'deg_rot_y': 5,
            'deg_rot_z': 5,
            'scale_x': 1.1,
            'scale_z': 1.1,
            'translation_x': 10,
            'translation_y': 10,
            'translation_z': 10,
            'shear_xz': 0.2
        }

        self.bins = [-1, -0.5, 0, 0.5, 1]
        self.M = self.get_M()
        self.M_binned = round_to_bins(self.M, self.bins)

        self.transforms_list = [
            'translation_y',
            'translation_z',
            'translation_x',
            'x_rotation',
            'y_rotation',
            'z_rotation',
            'scale_x',
            'scale_z',
            'shear_xz'
        ]
    def get_M(self):
        theta_x = self.transforms['deg_rot_x']
        theta_y = self.transforms['deg_rot_y']
        theta_z = self.transforms['deg_rot_z']
        scale_x = self.transforms['scale_x']
        scale_z = self.transforms['scale_z']
        translation_x = self.transforms['translation_x']
        translation_y = self.transforms['translation_y']
        translation_z = self.transforms['translation_z']
        shear_xz = self.transforms['shear_xz']

        normalized_vector_y_translation, magnitude_y_translation = get_normalized_vector_and_magnitude_for_y_translation(translation_y, self.my_plane, self.photobleach_pattern, 5)
        normalized_vector_z_translation, magnitude_z_translation = get_normalized_vector_and_magnitude_for_z_translation(translation_z, self.my_plane, self.photobleach_pattern, 5)
        normalized_vector_x_translation, magnitude_x_translation = get_normalized_vector_and_magnitude_for_x_translation(translation_x, self.my_plane, self.photobleach_pattern, 5)
        normalized_vector_x_rotation, magnitude_x_rotation = get_normalized_vector_and_magnitude_for_x_rotation(theta_x, self.my_plane, self.photobleach_pattern, 5)
        normalized_vector_y_rotation, magnitude_y_rotation = get_normalized_vector_and_magnitude_for_y_rotation(theta_y, self.my_plane, self.photobleach_pattern, 5)
        normalized_vector_z_rotation, magnitude_z_rotation = get_normalized_vector_and_magnitude_for_z_rotation(theta_z, self.my_plane, self.photobleach_pattern, 5)
        normalized_vector_scale_x, magnitude_scale_x = get_normalized_vector_and_magnitude_for_scale_x(scale_x, self.my_plane, self.photobleach_pattern, 5)
        normalized_vector_scale_z, magnitude_scale_z = get_normalized_vector_and_magnitude_for_scale_z(scale_z, self.my_plane, self.photobleach_pattern, 5)
        normalized_vector_shear_xz, magnitude_shear_xz = get_normalized_vector_and_magnitude_for_shear_xz(shear_xz, self.my_plane, self.photobleach_pattern, 5)

        M_all = np.column_stack([
            normalized_vector_y_translation,
            normalized_vector_z_translation,
            normalized_vector_x_translation,
            normalized_vector_x_rotation,
            normalized_vector_y_rotation,
            normalized_vector_z_rotation,
            normalized_vector_scale_x,
            normalized_vector_scale_z,
            normalized_vector_shear_xz])

        # print(M_all)
        # print("--------------------------------")
        return M_all


class RankLines5Z2():
    def setup(self):
        self.BETA1 = 1

        AC_z_um = 0
        self.beta0_um = 0
        self.beta1 = float(self.BETA1)
        self.C_z_um = 0
        self.C_x_um = 200
        self.d1 = -self.beta1
        self.d0_um = 200 * self.beta1
        self.D_z_um = 30
        self.B_z_um = 50
        self.E_x_um = 250
        self.E_z_delta = 1
        self.E_z0 = 0

        self.photobleach_pattern = PhotobleachPattern5Z2(
            # A line
            A_x_um=self.beta0_um,
            A_z_um=AC_z_um,
            # B Line
            beta1=self.beta1,
            beta0_um=self.beta0_um,  # Recommendation: use beta0 = A_x so the two lines meet
            B_z_um=self.B_z_um,
            C_x_um=self.C_x_um,
            C_z_um=self.C_z_um,
            d1=self.d1,
            d0_um=self.d0_um,
            D_z_um=self.D_z_um,
            E_x_um=self.E_x_um,
            E_z_delta=self.E_z_delta,
            E_z0=self.E_z0
        )

        # Define the cutting plane
        u = np.array([1, 0, 0])  # um/pix [1,0,0] is 1um per pixel parallel to x
        v = np.array([0, 0, 1])  # um/pix [0,0,1] is 1um per pixel parallel to z
        h = np.array([0, 80, 0])  # um
        self.h = h

        self.my_plane = MyPlane(u, v, h)

        self.transforms = {
            'deg_rot_x': 5,
            'deg_rot_y': 5,
            'deg_rot_z': 5,
            'scale_x': 1.1,
            'scale_z': 1.1,
            'translation_x': 10,
            'translation_y': 10,
            'translation_z': 10,
            'shear_xz': 0.2
        }

        self.bins = [-1, -0.5, 0, 0.5, 1]
        self.M = self.get_M()
        self.M_binned = round_to_bins(self.M, self.bins)

        self.transforms_list = [
            'translation_y',
            'translation_z',
            'translation_x',
            'x_rotation',
            'y_rotation',
            'z_rotation',
            'scale_x',
            'scale_z',
            'shear_xz'
        ]
    def get_M(self):
        theta_x = self.transforms['deg_rot_x']
        theta_y = self.transforms['deg_rot_y']
        theta_z = self.transforms['deg_rot_z']
        scale_x = self.transforms['scale_x']
        scale_z = self.transforms['scale_z']
        translation_x = self.transforms['translation_x']
        translation_y = self.transforms['translation_y']
        translation_z = self.transforms['translation_z']
        shear_xz = self.transforms['shear_xz']

        normalized_vector_y_translation, magnitude_y_translation = get_normalized_vector_and_magnitude_for_y_translation(translation_y, self.my_plane, self.photobleach_pattern, 5)
        normalized_vector_z_translation, magnitude_z_translation = get_normalized_vector_and_magnitude_for_z_translation(translation_z, self.my_plane, self.photobleach_pattern, 5)
        normalized_vector_x_translation, magnitude_x_translation = get_normalized_vector_and_magnitude_for_x_translation(translation_x, self.my_plane, self.photobleach_pattern, 5)
        normalized_vector_x_rotation, magnitude_x_rotation = get_normalized_vector_and_magnitude_for_x_rotation(theta_x, self.my_plane, self.photobleach_pattern, 5)
        normalized_vector_y_rotation, magnitude_y_rotation = get_normalized_vector_and_magnitude_for_y_rotation(theta_y, self.my_plane, self.photobleach_pattern, 5)
        normalized_vector_z_rotation, magnitude_z_rotation = get_normalized_vector_and_magnitude_for_z_rotation(theta_z, self.my_plane, self.photobleach_pattern, 5)
        normalized_vector_scale_x, magnitude_scale_x = get_normalized_vector_and_magnitude_for_scale_x(scale_x, self.my_plane, self.photobleach_pattern, 5)
        normalized_vector_scale_z, magnitude_scale_z = get_normalized_vector_and_magnitude_for_scale_z(scale_z, self.my_plane, self.photobleach_pattern, 5)
        normalized_vector_shear_xz, magnitude_shear_xz = get_normalized_vector_and_magnitude_for_shear_xz(shear_xz, self.my_plane, self.photobleach_pattern, 5)

        M_all = np.column_stack([
            normalized_vector_y_translation,
            normalized_vector_z_translation,
            normalized_vector_x_translation,
            normalized_vector_x_rotation,
            normalized_vector_y_rotation,
            normalized_vector_z_rotation,
            normalized_vector_scale_x,
            normalized_vector_scale_z,
            normalized_vector_shear_xz])

        # print(M_all)
        # print("--------------------------------")
        return M_all


class RankLines4a2():
    def setup(self):
        self.BETA1 = 1

        AC_z_um = 0
        self.beta0_um = 0
        self.beta1 = float(self.BETA1)
        self.d1 = -self.beta1
        self.d0_um = 200 * self.beta1
        self.D_z_um = 0
        self.B_z_um = 50
        self.C_x_um = 200
        self.C_z_um = 0

        self.photobleach_pattern = PhotobleachPattern4a(
            A_x_um=self.beta0_um,
            A_z_um=AC_z_um,
            beta1=self.beta1,
            beta0_um=self.beta0_um,
            B_z_um=self.B_z_um,
            C_x_um=self.C_x_um,
            C_z_um=self.C_z_um,
            d1=self.d1,
            d0_um=self.d0_um,
            D_z_um=self.D_z_um
        )

        # Define the cutting plane
        u = np.array([1, 0, 0])  # um/pix [1,0,0] is 1um per pixel parallel to x
        v = np.array([0, 0, 1])  # um/pix [0,0,1] is 1um per pixel parallel to z
        h = np.array([0, 80, 0])  # um
        self.h = h

        self.my_plane = MyPlane(u, v, h)
        
        self.transforms = {
            'deg_rot_x': 5,
            'deg_rot_y': 5,
            'deg_rot_z': 5,
            'scale_x': 1.1,
            'scale_z': 1.1,
            'translation_x': 10,
            'translation_y': 10,
            'translation_z': 10,
            'shear_xz': 0.2
        }

        self.bins = [-1, -0.5, 0, 0.5, 1]
        self.M = self.get_M()
        self.M_binned = round_to_bins(self.M, self.bins)

        self.transforms_list = [
            'translation_y',
            'translation_z',
            'translation_x',
            'x_rotation',
            'y_rotation',
            'z_rotation',
            'scale_x',
            'scale_z',
            'shear_xz'
        ]
    def get_M(self):
        theta_x = self.transforms['deg_rot_x']
        theta_y = self.transforms['deg_rot_y']
        theta_z = self.transforms['deg_rot_z']
        scale_x = self.transforms['scale_x']
        scale_z = self.transforms['scale_z']
        translation_x = self.transforms['translation_x']
        translation_y = self.transforms['translation_y']
        translation_z = self.transforms['translation_z']
        shear_xz = self.transforms['shear_xz']

        normalized_vector_y_translation, magnitude_y_translation = get_normalized_vector_and_magnitude_for_y_translation(translation_y, self.my_plane, self.photobleach_pattern, 4)
        normalized_vector_z_translation, magnitude_z_translation = get_normalized_vector_and_magnitude_for_z_translation(translation_z, self.my_plane, self.photobleach_pattern, 4)
        normalized_vector_x_translation, magnitude_x_translation = get_normalized_vector_and_magnitude_for_x_translation(translation_x, self.my_plane, self.photobleach_pattern, 4)
        normalized_vector_x_rotation, magnitude_x_rotation = get_normalized_vector_and_magnitude_for_x_rotation(theta_x, self.my_plane, self.photobleach_pattern, 4)
        normalized_vector_y_rotation, magnitude_y_rotation = get_normalized_vector_and_magnitude_for_y_rotation(theta_y, self.my_plane, self.photobleach_pattern, 4)
        normalized_vector_z_rotation, magnitude_z_rotation = get_normalized_vector_and_magnitude_for_z_rotation(theta_z, self.my_plane, self.photobleach_pattern, 4)
        normalized_vector_scale_x, magnitude_scale_x = get_normalized_vector_and_magnitude_for_scale_x(scale_x, self.my_plane, self.photobleach_pattern, 4)
        normalized_vector_scale_z, magnitude_scale_z = get_normalized_vector_and_magnitude_for_scale_z(scale_z, self.my_plane, self.photobleach_pattern, 4)
        normalized_vector_shear_xz, magnitude_shear_xz = get_normalized_vector_and_magnitude_for_shear_xz(shear_xz, self.my_plane, self.photobleach_pattern, 4)

        M_all = np.column_stack([
            normalized_vector_y_translation,
            normalized_vector_z_translation,
            normalized_vector_x_translation,
            normalized_vector_x_rotation,
            normalized_vector_y_rotation,
            normalized_vector_z_rotation,
            normalized_vector_scale_x,
            normalized_vector_scale_z,
            normalized_vector_shear_xz])

        return M_all

class RankLines4a3():
    def setup(self):
        self.BETA1 = 1

        AC_z_um = 0
        self.beta0_um = 0
        self.beta1 = float(self.BETA1)
        self.d1 = -self.beta1 / 2
        self.d0_um = 200 * self.beta1 / 2
        self.D_z_um = 25
        self.B_z_um = 50
        self.C_x_um = 200
        self.C_z_um = 0

        self.photobleach_pattern = PhotobleachPattern4a(
            A_x_um=self.beta0_um,
            A_z_um=AC_z_um,
            beta1=self.beta1,
            beta0_um=self.beta0_um,
            B_z_um=self.B_z_um,
            C_x_um=self.C_x_um,
            C_z_um=self.C_z_um,
            d1=self.d1,
            d0_um=self.d0_um,
            D_z_um=self.D_z_um
        )

        # Define the cutting plane
        u = np.array([1, 0, 0])  # um/pix [1,0,0] is 1um per pixel parallel to x
        v = np.array([0, 0, 1])  # um/pix [0,0,1] is 1um per pixel parallel to z
        h = np.array([0, 80, 0])  # um
        self.h = h

        self.my_plane = MyPlane(u, v, h)
        
        self.transforms = {
            'deg_rot_x': 5,
            'deg_rot_y': 5,
            'deg_rot_z': 5,
            'scale_x': 1.1,
            'scale_z': 1.1,
            'translation_x': 10,
            'translation_y': 10,
            'translation_z': 10,
            'shear_xz': 0.2
        }

        self.bins = [-1, -0.5, 0, 0.5, 1]
        self.M = self.get_M()
        self.M_binned = round_to_bins(self.M, self.bins)

        self.transforms_list = [
            'translation_y',
            'translation_z',
            'translation_x',
            'x_rotation',
            'y_rotation',
            'z_rotation',
            'scale_x',
            'scale_z',
            'shear_xz'
        ]
    def get_M(self):
        theta_x = self.transforms['deg_rot_x']
        theta_y = self.transforms['deg_rot_y']
        theta_z = self.transforms['deg_rot_z']
        scale_x = self.transforms['scale_x']
        scale_z = self.transforms['scale_z']
        translation_x = self.transforms['translation_x']
        translation_y = self.transforms['translation_y']
        translation_z = self.transforms['translation_z']
        shear_xz = self.transforms['shear_xz']

        normalized_vector_y_translation, magnitude_y_translation = get_normalized_vector_and_magnitude_for_y_translation(translation_y, self.my_plane, self.photobleach_pattern, 4)
        normalized_vector_z_translation, magnitude_z_translation = get_normalized_vector_and_magnitude_for_z_translation(translation_z, self.my_plane, self.photobleach_pattern, 4)
        normalized_vector_x_translation, magnitude_x_translation = get_normalized_vector_and_magnitude_for_x_translation(translation_x, self.my_plane, self.photobleach_pattern, 4)
        normalized_vector_x_rotation, magnitude_x_rotation = get_normalized_vector_and_magnitude_for_x_rotation(theta_x, self.my_plane, self.photobleach_pattern, 4)
        normalized_vector_y_rotation, magnitude_y_rotation = get_normalized_vector_and_magnitude_for_y_rotation(theta_y, self.my_plane, self.photobleach_pattern, 4)
        normalized_vector_z_rotation, magnitude_z_rotation = get_normalized_vector_and_magnitude_for_z_rotation(theta_z, self.my_plane, self.photobleach_pattern, 4)
        normalized_vector_scale_x, magnitude_scale_x = get_normalized_vector_and_magnitude_for_scale_x(scale_x, self.my_plane, self.photobleach_pattern, 4)
        normalized_vector_scale_z, magnitude_scale_z = get_normalized_vector_and_magnitude_for_scale_z(scale_z, self.my_plane, self.photobleach_pattern, 4)
        normalized_vector_shear_xz, magnitude_shear_xz = get_normalized_vector_and_magnitude_for_shear_xz(shear_xz, self.my_plane, self.photobleach_pattern, 4)

        M_all = np.column_stack([
            normalized_vector_y_translation,
            normalized_vector_z_translation,
            normalized_vector_x_translation,
            normalized_vector_x_rotation,
            normalized_vector_y_rotation,
            normalized_vector_z_rotation,
            normalized_vector_scale_x,
            normalized_vector_scale_z,
            normalized_vector_shear_xz])

        return M_all

class RankLines5Split():
    def setup(self):
        self.BETA1 = 1

        self.A_z_um = 0
        self.beta0_um = 0
        self.beta1 = float(self.BETA1)
        self.d1 = -self.beta1 
        self.d0_um = 100 * self.beta1 + 100 * self.beta1
        self.D_z_um = 50
        self.B_z_um = 50
        self.C_x_um = 100
        self.C_z_um = 0
        self.E_x_um = 200
        self.E_z_um = 0

        self.photobleach_pattern = PhotobleachPattern5split(
            A_x_um=self.beta0_um,
            A_z_um=self.A_z_um,
            beta1=self.beta1,
            beta0_um=self.beta0_um,
            B_z_um=self.B_z_um,
            B_x_start=0,
            B_x_end=100,
            C_x_um=self.C_x_um,
            C_z_um=self.C_z_um,
            d1=self.d1,
            d0_um=self.d0_um,
            D_z_um=self.D_z_um,
            D_x_start=100,
            D_x_end=200,
            E_x_um=self.E_x_um,
            E_z_um=self.E_z_um
        )

        # Define the cutting plane
        u = np.array([1, 0, 0])  # um/pix [1,0,0] is 1um per pixel parallel to x
        v = np.array([0, 0, 1])  # um/pix [0,0,1] is 1um per pixel parallel to z
        h = np.array([0, 80, 0])  # um
        self.h = h

        self.my_plane = MyPlane(u, v, h)
        
        self.transforms = {
            'deg_rot_x': 5,
            'deg_rot_y': 5,
            'deg_rot_z': 5,
            'scale_x': 1.1,
            'scale_z': 1.1,
            'translation_x': 10,
            'translation_y': 10,
            'translation_z': 10,
            'shear_xz': 0.2
        }

        self.bins = [-1, -0.5, 0, 0.5, 1]
        self.M = self.get_M()
        self.M_binned = round_to_bins(self.M, self.bins)

        self.transforms_list = [
            'translation_y',
            'translation_z',
            'translation_x',
            'x_rotation',
            'y_rotation',
            'z_rotation',
            'scale_x',
            'scale_z',
            'shear_xz'
        ]
    def get_M(self):
        theta_x = self.transforms['deg_rot_x']
        theta_y = self.transforms['deg_rot_y']
        theta_z = self.transforms['deg_rot_z']
        scale_x = self.transforms['scale_x']
        scale_z = self.transforms['scale_z']
        translation_x = self.transforms['translation_x']
        translation_y = self.transforms['translation_y']
        translation_z = self.transforms['translation_z']
        shear_xz = self.transforms['shear_xz']

        normalized_vector_y_translation, magnitude_y_translation = get_normalized_vector_and_magnitude_for_y_translation(translation_y, self.my_plane, self.photobleach_pattern, 5)
        normalized_vector_z_translation, magnitude_z_translation = get_normalized_vector_and_magnitude_for_z_translation(translation_z, self.my_plane, self.photobleach_pattern, 5)
        normalized_vector_x_translation, magnitude_x_translation = get_normalized_vector_and_magnitude_for_x_translation(translation_x, self.my_plane, self.photobleach_pattern, 5)
        normalized_vector_x_rotation, magnitude_x_rotation = get_normalized_vector_and_magnitude_for_x_rotation(theta_x, self.my_plane, self.photobleach_pattern, 5)
        normalized_vector_y_rotation, magnitude_y_rotation = get_normalized_vector_and_magnitude_for_y_rotation(theta_y, self.my_plane, self.photobleach_pattern, 5)
        normalized_vector_z_rotation, magnitude_z_rotation = get_normalized_vector_and_magnitude_for_z_rotation(theta_z, self.my_plane, self.photobleach_pattern, 5)
        normalized_vector_scale_x, magnitude_scale_x = get_normalized_vector_and_magnitude_for_scale_x(scale_x, self.my_plane, self.photobleach_pattern, 5)
        normalized_vector_scale_z, magnitude_scale_z = get_normalized_vector_and_magnitude_for_scale_z(scale_z, self.my_plane, self.photobleach_pattern, 5)
        normalized_vector_shear_xz, magnitude_shear_xz = get_normalized_vector_and_magnitude_for_shear_xz(shear_xz, self.my_plane, self.photobleach_pattern, 5)

        M_all = np.column_stack([
            normalized_vector_y_translation,
            normalized_vector_z_translation,
            normalized_vector_x_translation,
            normalized_vector_x_rotation,
            normalized_vector_y_rotation,
            normalized_vector_z_rotation,
            normalized_vector_scale_x,
            normalized_vector_scale_z,
            normalized_vector_shear_xz])

        return M_all

class RankLines5Split2():
    def setup(self):

        self.A_x_um = 0
        self.A_z_um = 0

        self.beta0_um = 0
        self.beta1 = 1
        self.B_z_um = 0
        self.B_x_start = 0
        self.B_x_end = 100

        self.d1 = 1
        self.d0_um = -100
        self.D_z_um = 0
        self.D_x_start = 100
        self.D_x_end = 150

        self.C_x_um = 100
        self.C_z_um = 0

        self.E_x_um = 150
        self.E_z_um = 50

        self.photobleach_pattern = PhotobleachPattern5split(
            A_x_um=self.A_x_um,
            A_z_um=self.A_z_um,
            beta1=self.beta1,
            beta0_um=self.beta0_um,
            B_z_um=self.B_z_um,
            B_x_start=self.B_x_start,
            B_x_end=self.B_x_end,
            C_x_um=self.C_x_um,
            C_z_um=self.C_z_um,
            d1=self.d1,
            d0_um=self.d0_um,
            D_z_um=self.D_z_um,
            D_x_start=self.D_x_start,
            D_x_end=self.D_x_end,
            E_x_um=self.E_x_um,
            E_z_um=self.E_z_um
        )

        # Define the cutting plane
        u = np.array([1, 0, 0])  # um/pix [1,0,0] is 1um per pixel parallel to x
        v = np.array([0, 0, 1])  # um/pix [0,0,1] is 1um per pixel parallel to z
        h = np.array([0, 40, 0])  # um
        self.h = h

        self.my_plane = MyPlane(u, v, h)
        
        self.transforms = {
            'deg_rot_x': 5,
            'deg_rot_y': 5,
            'deg_rot_z': 5,
            'scale_x': 1.1,
            'scale_z': 1.1,
            'translation_x': 10,
            'translation_y': 10,
            'translation_z': 10,
            'shear_xz': 0.2
        }

        self.bins = [-1, -0.5, 0, 0.5, 1]
        self.M = self.get_M()
        self.M_binned = round_to_bins(self.M, self.bins)

        self.transforms_list = [
            'translation_y',
            'translation_z',
            'translation_x',
            'x_rotation',
            'y_rotation',
            'z_rotation',
            'scale_x',
            'scale_z',
            'shear_xz'
        ]
    def get_M(self):
        theta_x = self.transforms['deg_rot_x']
        theta_y = self.transforms['deg_rot_y']
        theta_z = self.transforms['deg_rot_z']
        scale_x = self.transforms['scale_x']
        scale_z = self.transforms['scale_z']
        translation_x = self.transforms['translation_x']
        translation_y = self.transforms['translation_y']
        translation_z = self.transforms['translation_z']
        shear_xz = self.transforms['shear_xz']

        normalized_vector_y_translation, magnitude_y_translation = get_normalized_vector_and_magnitude_for_y_translation(translation_y, self.my_plane, self.photobleach_pattern, 5)
        normalized_vector_z_translation, magnitude_z_translation = get_normalized_vector_and_magnitude_for_z_translation(translation_z, self.my_plane, self.photobleach_pattern, 5)
        normalized_vector_x_translation, magnitude_x_translation = get_normalized_vector_and_magnitude_for_x_translation(translation_x, self.my_plane, self.photobleach_pattern, 5)
        normalized_vector_x_rotation, magnitude_x_rotation = get_normalized_vector_and_magnitude_for_x_rotation(theta_x, self.my_plane, self.photobleach_pattern, 5)
        normalized_vector_y_rotation, magnitude_y_rotation = get_normalized_vector_and_magnitude_for_y_rotation(theta_y, self.my_plane, self.photobleach_pattern, 5)
        normalized_vector_z_rotation, magnitude_z_rotation = get_normalized_vector_and_magnitude_for_z_rotation(theta_z, self.my_plane, self.photobleach_pattern, 5)
        normalized_vector_scale_x, magnitude_scale_x = get_normalized_vector_and_magnitude_for_scale_x(scale_x, self.my_plane, self.photobleach_pattern, 5)
        normalized_vector_scale_z, magnitude_scale_z = get_normalized_vector_and_magnitude_for_scale_z(scale_z, self.my_plane, self.photobleach_pattern, 5)
        normalized_vector_shear_xz, magnitude_shear_xz = get_normalized_vector_and_magnitude_for_shear_xz(shear_xz, self.my_plane, self.photobleach_pattern, 5)

        M_all = np.column_stack([
            normalized_vector_y_translation,
            normalized_vector_z_translation,
            normalized_vector_x_translation,
            normalized_vector_x_rotation,
            normalized_vector_y_rotation,
            normalized_vector_z_rotation,
            normalized_vector_scale_x,
            normalized_vector_scale_z,
            normalized_vector_shear_xz])

        return M_all

class RankLines5Split3():
    def setup(self):

        self.A_x_um = 0
        self.A_z_um = 0

        self.beta0_um = 0
        self.beta1 = 1
        self.B_z_um = 0
        self.B_x_start = 0
        self.B_x_end = 100

        self.d1 = -1
        self.d0_um = 200
        self.D_z_um = 0
        self.D_x_start = 100
        self.D_x_end = 200

        self.C_x_um = 200
        self.C_z_um = 50

        self.E_x_um = 250
        self.E_z_um = 0

        self.photobleach_pattern = PhotobleachPattern5split(
            A_x_um=self.A_x_um,
            A_z_um=self.A_z_um,
            beta1=self.beta1,
            beta0_um=self.beta0_um,
            B_z_um=self.B_z_um,
            B_x_start=self.B_x_start,
            B_x_end=self.B_x_end,
            C_x_um=self.C_x_um,
            C_z_um=self.C_z_um,
            d1=self.d1,
            d0_um=self.d0_um,
            D_z_um=self.D_z_um,
            D_x_start=self.D_x_start,
            D_x_end=self.D_x_end,
            E_x_um=self.E_x_um,
            E_z_um=self.E_z_um
        )

        # Define the cutting plane
        u = np.array([1, 0, 0])  # um/pix [1,0,0] is 1um per pixel parallel to x
        v = np.array([0, 0, 1])  # um/pix [0,0,1] is 1um per pixel parallel to z
        h = np.array([0, 40, 0])  # um
        self.h = h

        self.my_plane = MyPlane(u, v, h)
        
        self.transforms = {
            'deg_rot_x': 5,
            'deg_rot_y': 5,
            'deg_rot_z': 5,
            'scale_x': 1.1,
            'scale_z': 1.1,
            'translation_x': 10,
            'translation_y': 10,
            'translation_z': 10,
            'shear_xz': 0.2
        }

        self.bins = [-1, -0.5, 0, 0.5, 1]
        self.M = self.get_M()
        self.M_binned = round_to_bins(self.M, self.bins)

        self.transforms_list = [
            'translation_y',
            'translation_z',
            'translation_x',
            'x_rotation',
            'y_rotation',
            'z_rotation',
            'scale_x',
            'scale_z',
            'shear_xz'
        ]
    def get_M(self):
        theta_x = self.transforms['deg_rot_x']
        theta_y = self.transforms['deg_rot_y']
        theta_z = self.transforms['deg_rot_z']
        scale_x = self.transforms['scale_x']
        scale_z = self.transforms['scale_z']
        translation_x = self.transforms['translation_x']
        translation_y = self.transforms['translation_y']
        translation_z = self.transforms['translation_z']
        shear_xz = self.transforms['shear_xz']

        normalized_vector_y_translation, magnitude_y_translation = get_normalized_vector_and_magnitude_for_y_translation(translation_y, self.my_plane, self.photobleach_pattern, 5)
        normalized_vector_z_translation, magnitude_z_translation = get_normalized_vector_and_magnitude_for_z_translation(translation_z, self.my_plane, self.photobleach_pattern, 5)
        normalized_vector_x_translation, magnitude_x_translation = get_normalized_vector_and_magnitude_for_x_translation(translation_x, self.my_plane, self.photobleach_pattern, 5)
        normalized_vector_x_rotation, magnitude_x_rotation = get_normalized_vector_and_magnitude_for_x_rotation(theta_x, self.my_plane, self.photobleach_pattern, 5)
        normalized_vector_y_rotation, magnitude_y_rotation = get_normalized_vector_and_magnitude_for_y_rotation(theta_y, self.my_plane, self.photobleach_pattern, 5)
        normalized_vector_z_rotation, magnitude_z_rotation = get_normalized_vector_and_magnitude_for_z_rotation(theta_z, self.my_plane, self.photobleach_pattern, 5)
        normalized_vector_scale_x, magnitude_scale_x = get_normalized_vector_and_magnitude_for_scale_x(scale_x, self.my_plane, self.photobleach_pattern, 5)
        normalized_vector_scale_z, magnitude_scale_z = get_normalized_vector_and_magnitude_for_scale_z(scale_z, self.my_plane, self.photobleach_pattern, 5)
        normalized_vector_shear_xz, magnitude_shear_xz = get_normalized_vector_and_magnitude_for_shear_xz(shear_xz, self.my_plane, self.photobleach_pattern, 5)

        M_all = np.column_stack([
            normalized_vector_y_translation,
            normalized_vector_z_translation,
            normalized_vector_x_translation,
            normalized_vector_x_rotation,
            normalized_vector_y_rotation,
            normalized_vector_z_rotation,
            normalized_vector_scale_x,
            normalized_vector_scale_z,
            normalized_vector_shear_xz])

        return M_all

def main():
    # rank_lines3 = RankLines3()
    # rank_lines3.setup()
    # rank3 = get_rank(rank_lines3.M)
    # rank_binned3 = get_rank(rank_lines3.M_binned)

    # print("3 LINES:")
    # print(rank_lines3.M.T)
    # print(rank_lines3.M_binned.T)
    # print("--------------------------------")
    # print("Rank: ",rank3)
    # print("Rank binned: ", rank_binned3)
    # print()
    # independent_cols, dependent_cols = find_dependent_columns(rank_lines3.M)
    # print("Independent transform indices: ", independent_cols)
    # print("Dependent transform indices: ", dependent_cols)

    # rank_lines4 = RankLines4()
    # rank_lines4.setup()
    # rank4 = get_rank(rank_lines4.M)
    # rank_binned4 = get_rank(rank_lines4.M_binned)

    # print("Pattern 4a-1:")
    # print(rank_lines4.M_binned.T)
    # print("--------------------------------")
    # print("Rank: ",rank4)
    # print("Rank binned: ", rank_binned4)
    # print("--------------------------------")
    # independent_cols, dependent_cols = find_dependent_columns(rank_lines4.M_binned)
    # print("Independent transform indices: ", independent_cols)
    # print("Dependent transform indices: ", dependent_cols)

    # rank_lines4a2 = RankLines4a2()
    # rank_lines4a2.setup()
    # rank4a2 = get_rank(rank_lines4a2.M)
    # rank_binned4a2 = get_rank(rank_lines4a2.M_binned)

    # print("Pattern 4a-2:")
    # print(rank_lines4a2.M_binned.T)
    # print("--------------------------------")
    # print("Rank: ",rank4a2)
    # print("Rank binned: ", rank_binned4a2)
    # print("--------------------------------")

    # rank_lines4a3 = RankLines4a3()
    # rank_lines4a3.setup()
    # rank4a3 = get_rank(rank_lines4a3.M)
    # rank_binned4a3 = get_rank(rank_lines4a3.M_binned)

    # print("Pattern 4a-3:")
    # print(rank_lines4a3.M_binned.T)
    # print("--------------------------------")
    # print("Rank: ",rank4a3)
    # print("Rank binned: ", rank_binned4a3)
    # print("--------------------------------")

    rank_lines5split = RankLines5Split()
    rank_lines5split.setup()
    rank5split = get_rank(rank_lines5split.M)
    rank_binned5split = get_rank(rank_lines5split.M_binned)

    print("Pattern 5 Split:")
    print(rank_lines5split.M_binned.T)
    print("--------------------------------")
    print("Rank: ",rank5split)
    print("Rank binned: ", rank_binned5split)
    print("--------------------------------")

    rank_lines5split2 = RankLines5Split2()
    rank_lines5split2.setup()
    rank5split2 = get_rank(rank_lines5split2.M)
    rank_binned5split2 = get_rank(rank_lines5split2.M_binned)

    print("Pattern 5 Split 2:")
    print(rank_lines5split2.M_binned.T)
    print("--------------------------------")
    print("Rank: ",rank5split2)
    print("Rank binned: ", rank_binned5split2)
    print("--------------------------------")

    rank_lines5split3 = RankLines5Split3()
    rank_lines5split3.setup()
    rank5split3 = get_rank(rank_lines5split3.M)
    rank_binned5split3 = get_rank(rank_lines5split3.M_binned)
    print("Pattern 5 Split 3:")
    print(rank_lines5split3.M_binned.T)
    print("--------------------------------")
    print("Rank: ",rank5split3)
    print("Rank binned: ", rank_binned5split3)
    print("--------------------------------")

    # rank_lines3z = RankLines3Z()
    # rank_lines3z.setup()
    # rank3z = get_rank(rank_lines3z.M)
    # rank_binned3z = get_rank(rank_lines3z.M_binned)

    # print("3 LINES Z:")
    # print(rank_lines3z.M_binned.T)
    # print("--------------------------------")
    # print("Rank: ",rank3z)
    # print("Rank binned: ", rank_binned3z)

    # rank_lines4z = RankLines4Z()
    # rank_lines4z.setup()
    # rank4z = get_rank(rank_lines4z.M)
    # rank_binned4z = get_rank(rank_lines4z.M_binned)

    # print("4 LINES Z:")
    # print(rank_lines4z.M_binned.T)
    # print("--------------------------------")
    # print("Rank: ",rank4z)
    # print("Rank binned: ", rank_binned4z)

    # rank_lines5 = RankLines5()
    # rank_lines5.setup()
    # rank5 = get_rank(rank_lines5.M)
    # rank_binned5 = get_rank(rank_lines5.M_binned)

    # print("5 LINES:")
    # print(rank_lines5.M_binned.T)
    # print("--------------------------------")
    # print("Rank: ",rank5)
    # print("Rank binned: ", rank_binned5)

    # rank_lines5z = RankLines5Z()
    # rank_lines5z.setup()
    # rank5z = get_rank(rank_lines5z.M)
    # rank_binned5z = get_rank(rank_lines5z.M_binned)

    # print("5 LINES Z:")
    # print(rank_lines5z.M.T)
    # print(rank_lines5z.M_binned.T)
    # print("--------------------------------")
    # print("Rank: ",rank5z)
    # print("Rank binned: ", rank_binned5z)
    # independent_cols, dependent_cols = find_dependent_columns(rank_lines5z.M_binned)
    # print("Independent transform indices: ", independent_cols)
    # print("Dependent transform indices: ", dependent_cols)
    # column_combination = find_column_combination(rank_lines5z.M_binned, 3)
    # print("Column combination: ", column_combination)

    rank_lines5a = RankLines5a()
    rank_lines5a.setup()
    rank5a = get_rank(rank_lines5a.M)
    rank_binned5a = get_rank(rank_lines5a.M_binned)

    print("5 LINES Pattern A:")
    print(rank_lines5a.M_binned.T)
    print("--------------------------------")
    print("Rank: ",rank5a)
    print("Rank binned: ", rank_binned5a)


    # rank_lines5z2 = RankLines5Z2()
    # rank_lines5z2.setup()
    # rank5z2 = get_rank(rank_lines5z2.M)
    # rank_binned5z2 = get_rank(rank_lines5z2.M_binned)

    # print("5 LINES Z2:")
    
    # print(rank_lines5z2.M_binned.T)
    # print("--------------------------------")
    # print("Rank: ",rank5z2)
    # print("Rank binned: ",rank_binned5z2)

if __name__ == "__main__":
    main()