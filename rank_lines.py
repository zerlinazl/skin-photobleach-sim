import numpy as np
from my_plane import MyPlane
from scipy.linalg import qr
from photobleach_pattern import PhotobleachPattern, PhotobleachPattern4
from test_forward_model_sensitivity_utils import (
    compare_error_between_two_designs,
    compare_error_between_two_designs4,
    verify_error_vector_is_according_to_expectation,
)

class RankLines():
    def setup(self):
        self.BETA1 = 1


         # Setting up photobleach pattern
        # AC_z_um = 0
        # self.beta0_um = 0
        # self.beta1 = float(self.BETA1)
        # self.B_z_um = 50
        # self.photobleach_pattern = PhotobleachPattern(
        #     # A line
        #     A_x_um=self.beta0_um,
        #     A_z_um=AC_z_um,
        #     # B Line
        #     beta1=self.beta1,
        #     beta0_um=self.beta0_um,  # Recommendation: use beta0 = A_x so the two lines meet
        #     B_z_um=self.B_z_um,
        #     C_x_um=200,
        #     C_z_um=AC_z_um
        # )

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
        h = np.array([0, 100, 0])  # um
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
            'translation_z': 10
        }

        self.bins = [0,0.5, 1]
        self.M = self.get_M()
        self.M = self.get_binned_M(self.bins)

        self.transforms_list = [
            'translation_y',
            'translation_z',
            'translation_x',
            'x_rotation',
            'y_rotation',
            'z_rotation',
            'scale_x',
            'scale_z'
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

        normalized_vector_y_translation, magnitude_y_translation = self.get_normalized_vector_and_magnitude_for_y_translation(translation_y)
        normalized_vector_z_translation, magnitude_z_translation = self.get_normalized_vector_and_magnitude_for_z_translation(translation_z)
        normalized_vector_x_translation, magnitude_x_translation = self.get_normalized_vector_and_magnitude_for_x_translation(translation_x)
        normalized_vector_x_rotation, magnitude_x_rotation = self.get_normalized_vector_and_magnitude_for_x_rotation(theta_x)
        normalized_vector_y_rotation, magnitude_y_rotation = self.get_normalized_vector_and_magnitude_for_y_rotation(theta_y)
        normalized_vector_z_rotation, magnitude_z_rotation = self.get_normalized_vector_and_magnitude_for_z_rotation(theta_z)
        normalized_vector_scale_x, magnitude_scale_x = self.get_normalized_vector_and_magnitude_for_scale_x(scale_x)
        normalized_vector_scale_z, magnitude_scale_z = self.get_normalized_vector_and_magnitude_for_scale_z(scale_z)

        M_all = np.column_stack([
            normalized_vector_y_translation,
            normalized_vector_z_translation,
            normalized_vector_x_translation,
            normalized_vector_x_rotation,
            normalized_vector_y_rotation,
            normalized_vector_z_rotation,
            normalized_vector_scale_x,
            normalized_vector_scale_z])

        return M_all

    def get_rank(self):
        return np.linalg.matrix_rank(self.M)

    def get_normalized_vector_and_magnitude_for_y_translation(self, input_value):
        modified_plane = self.my_plane.copy()
        modified_plane.h[1] = modified_plane.h[1] - input_value # Moving 1 micron = 1 pixel

        normalized_vector, magnitude = compare_error_between_two_designs4(
            self.photobleach_pattern, self.my_plane,
            self.photobleach_pattern, modified_plane)
        return normalized_vector, magnitude

    def get_normalized_vector_and_magnitude_for_z_translation(self, input_value):
        modified_plane = self.my_plane.copy()
        modified_plane.h = modified_plane.h + np.array([0, 0, input_value])

        normalized_vector, magnitude = compare_error_between_two_designs4(
            self.photobleach_pattern, self.my_plane,
            self.photobleach_pattern, modified_plane)
        return normalized_vector, magnitude
    
    def get_normalized_vector_and_magnitude_for_x_translation(self, input_value):
        modified_plane = self.my_plane.copy()
        modified_plane.h = modified_plane.h + np.array([input_value, 0, 0])

        normalized_vector, magnitude = compare_error_between_two_designs4(
            self.photobleach_pattern, self.my_plane,
            self.photobleach_pattern, modified_plane)
        return normalized_vector, magnitude
    
    def get_normalized_vector_and_magnitude_for_z_rotation(self, input_value):
        theta = input_value
        modified_plane = self.my_plane.copy()
        modified_plane.u = np.array([np.cos(theta), -np.sin(theta), 0.0])
        modified_plane.v = np.array([0.0, 0.0, 1.0])

        normalized_vector, magnitude = compare_error_between_two_designs4(
            self.photobleach_pattern, self.my_plane,
            self.photobleach_pattern, modified_plane)
        return normalized_vector, magnitude
    
    def get_normalized_vector_and_magnitude_for_x_rotation(self, input_value):
        theta = input_value
        modified_plane = self.my_plane.copy()
        modified_plane.u = np.array([1.0, 0.0, 0.0])
        modified_plane.v = np.array([0.0, -np.sin(theta), np.cos(theta)])

        normalized_vector, magnitude = compare_error_between_two_designs4(
            self.photobleach_pattern, self.my_plane,
            self.photobleach_pattern, modified_plane)
        return normalized_vector, magnitude
    
    def get_normalized_vector_and_magnitude_for_y_rotation(self, input_value):
        theta = input_value
        modified_plane = self.my_plane.copy()
        modified_plane.u = np.array([np.cos(theta), 0.0, -np.sin(theta)])
        modified_plane.v = np.array([np.sin(theta), 0.0,  np.cos(theta)])

        normalized_vector, magnitude = compare_error_between_two_designs4(
            self.photobleach_pattern, self.my_plane,
            self.photobleach_pattern, modified_plane)
        return normalized_vector, magnitude

    def get_normalized_vector_and_magnitude_for_scale_x(self, input_value):
        modified_plane = self.my_plane.copy()
        modified_plane.u = np.array([1.0 + input_value, 0.0, 0.0])
        modified_plane.v = np.array([0.0, 0.0, 1.0])

        normalized_vector, magnitude = compare_error_between_two_designs4(
            self.photobleach_pattern, self.my_plane,
            self.photobleach_pattern, modified_plane)
        return normalized_vector, magnitude
    
    def get_normalized_vector_and_magnitude_for_scale_z(self, input_value):
        modified_plane = self.my_plane.copy()
        modified_plane.v = np.array([0.0, 0.0, 1.0 + input_value])

        normalized_vector, magnitude = compare_error_between_two_designs4(
            self.photobleach_pattern, self.my_plane,
            self.photobleach_pattern, modified_plane)
        return normalized_vector, magnitude

    def round_to_bins(self, arr, bins):
        arr_abs = np.abs(arr)
        bins = np.array(bins)
        # compute distance to each bin
        idx = np.argmin(np.abs(arr_abs[..., None] - bins), axis=-1)
        return bins[idx]
    
    def get_binned_M(self, bins):
        return self.round_to_bins(self.M, bins)

    def find_dependent_rows(self, tol=1e-8):
        """
        Returns indices of independent rows and dependent rows of matrix M.
        """
        Q, R, piv = qr(self.M.T, pivoting=True)

        rank = np.sum(np.abs(np.diag(R)) > tol)

        independent_rows = piv[:rank]
        dependent_rows = piv[rank:]

        return independent_rows, dependent_rows
    

    def find_row_combination(self, target_row_idx, tol=1e-8):
        target = self.M[target_row_idx]

        # remove the target row
        other_indices = [i for i in range(self.M.shape[0]) if i != target_row_idx]
        A = self.M[other_indices]

        # solve least squares
        coeffs, residuals, rank, s = np.linalg.lstsq(A.T, target, rcond=None)

        # identify rows that contribute
        contributing_rows = [
            (int(other_indices[i]), coeffs[i])
            for i in range(len(coeffs))
            if abs(coeffs[i]) > tol
        ]

        return contributing_rows

    def find_identical_rows(self):
        """
        Returns groups of row indices that are identical.
        Each group contains indices of rows that are exactly the same.
        """
        unique_rows, inverse = np.unique(self.M, axis=0, return_inverse=True)

        groups = {}
        for i, group_id in enumerate(inverse):
            groups.setdefault(group_id, []).append(i)

        # keep only groups with duplicates
        identical_groups = [rows for rows in groups.values() if len(rows) > 1]

        return identical_groups

def main():
    rank_lines = RankLines()
    rank_lines.setup()
    rank = rank_lines.get_rank()
    print("Rank: ",rank)
    # independent_rows, dependent_rows = rank_lines.find_dependent_rows()
    # print("Dependent rows: ", dependent_rows)
    # contributing_rows = rank_lines.find_row_combination(7)
    # print("Contributing rows: ", contributing_rows)
    identical_groups = rank_lines.find_identical_rows()
    print("Identical rows: ", identical_groups)
    for g in identical_groups:
        for i in g:
            print(rank_lines.transforms_list[i])
        print("--------------------------------")

if __name__ == "__main__":
    main()