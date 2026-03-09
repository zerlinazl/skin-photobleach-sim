from functools import wraps
from typing import Union, List
import numpy as np
import numpy.testing as npt
from my_plane import MyPlane
from photobleach_pattern import PhotobleachPattern


def compare_error_between_two_designs(
        photobleach_pattern_1: PhotobleachPattern,
        my_plane_1: MyPlane,
        photobleach_pattern_2: PhotobleachPattern,
        my_plane_2: MyPlane
) -> tuple[np.ndarray, float]:
    """
    Compare how many pixels the photobleach pattern moves on plane between two designs.
    Returns:
    - normalized_vector: [A_u, A_v, B_u, B_v, C_u, C_v] normalized by max abs component (range [-1, 1])
    - magnitude: max absolute component before normalization
    """

    # Design 1
    A1, B1, C1 = photobleach_pattern_1.forward_model_nonparametric(my_plane_1)
    A1_u_pix, A1_v_pix = my_plane_1.physical_to_pix(A1)
    B1_u_pix, B1_v_pix = my_plane_1.physical_to_pix(B1)
    C1_u_pix, C1_v_pix = my_plane_1.physical_to_pix(C1)

    # Design 2
    A2, B2, C2 = photobleach_pattern_2.forward_model_nonparametric(my_plane_2)
    A2_u_pix, A2_v_pix = my_plane_2.physical_to_pix(A2)
    B2_u_pix, B2_v_pix = my_plane_2.physical_to_pix(B2)
    C2_u_pix, C2_v_pix = my_plane_2.physical_to_pix(C2)

    delta_vector = np.array([
        A1_u_pix - A2_u_pix,
        A1_v_pix - A2_v_pix,
        B1_u_pix - B2_u_pix,
        B1_v_pix - B2_v_pix,
        C1_u_pix - C2_u_pix,
        C1_v_pix - C2_v_pix,
    ], dtype=float)

    magnitude = float(np.max(np.abs(delta_vector)))
    if magnitude == 0.0:
        normalized_vector = np.zeros(6, dtype=float)
    else:
        normalized_vector = delta_vector / magnitude

    return normalized_vector, magnitude


def compare_error_between_two_designs4(
        photobleach_pattern_1: PhotobleachPattern,
        my_plane_1: MyPlane,
        photobleach_pattern_2: PhotobleachPattern,
        my_plane_2: MyPlane
) -> tuple[np.ndarray, float]:
    """
    Compare how many pixels the photobleach pattern moves on plane between two designs.
    Returns:
    - normalized_vector: [A_u, A_v, B_u, B_v, C_u, C_v] normalized by max abs component (range [-1, 1])
    - magnitude: max absolute component before normalization
    """

    # Design 1
    A1, B1, C1, D1 = photobleach_pattern_1.forward_model_nonparametric(my_plane_1)
    A1_u_pix, A1_v_pix = my_plane_1.physical_to_pix(A1)
    B1_u_pix, B1_v_pix = my_plane_1.physical_to_pix(B1)
    C1_u_pix, C1_v_pix = my_plane_1.physical_to_pix(C1)
    D1_u_pix, D1_v_pix = my_plane_1.physical_to_pix(D1)

    # Design 2
    A2, B2, C2, D2 = photobleach_pattern_2.forward_model_nonparametric(my_plane_2)
    A2_u_pix, A2_v_pix = my_plane_2.physical_to_pix(A2)
    B2_u_pix, B2_v_pix = my_plane_2.physical_to_pix(B2)
    C2_u_pix, C2_v_pix = my_plane_2.physical_to_pix(C2)
    D2_u_pix, D2_v_pix = my_plane_2.physical_to_pix(D2)

    delta_vector = np.array([
        A1_u_pix - A2_u_pix,
        A1_v_pix - A2_v_pix,
        B1_u_pix - B2_u_pix,
        B1_v_pix - B2_v_pix,
        C1_u_pix - C2_u_pix,
        C1_v_pix - C2_v_pix,
        D1_u_pix - D2_u_pix,
        D1_v_pix - D2_v_pix,
    ], dtype=float)

    magnitude = float(np.max(np.abs(delta_vector)))
    if magnitude == 0.0:
        normalized_vector = np.zeros(8, dtype=float)
    else:
        normalized_vector = delta_vector / magnitude

    return normalized_vector, magnitude

# THAT IS THE TESTING PART
def verify_error_vector_is_according_to_expectation(
        expected_normalized_vector: Union[np.ndarray, List], expected_magnitude: float,
        normalized_vector: np.ndarray, magnitude: float
):
    """
    Verify that the error vector is according to the expected vector. Raise error if not.

    Equal-ish components: any entry with |expected| > 0.8 is treated as "significant";
    the actual vector must have |actual[i]| >= 0.8 for each such index. So you can mark
    multiple components as significant (e.g. [0, 0, 1, 1, 0, 0]) when the response has
    similar magnitude in both; normalization is by max component so equal components
    both become ±1 in the actual vector.
    """
    expected_normalized_vector = np.array(expected_normalized_vector)
    normalized_vector = np.array(normalized_vector)

    if expected_normalized_vector.shape != normalized_vector.shape:
        raise AssertionError(
            f"Vector shape mismatch: expected {expected_normalized_vector.shape}, got {normalized_vector.shape}."
        )

    # Compare magnitude
    npt.assert_allclose(
        np.abs(magnitude), np.abs(expected_magnitude),
        rtol=0.05,  # 5% error is acceptable
        atol=1,  # 1 pixel model error is acceptable
        err_msg="Magnitude of error vector is not as expected")

    # Find which vector elements are significant
    is_element_significant = (np.abs(expected_normalized_vector) > 0.8).astype(bool)
    significant_indices = np.where(is_element_significant)[0]
    if significant_indices.size == 0:
        raise AssertionError(
            "No significant expected vector elements were provided; expected at least one (|value| > 0.8)."
        )

    for i in significant_indices:
        if np.abs(normalized_vector[i]) < 0.8:
            raise AssertionError(f"Vector element {i} (value: {normalized_vector[i]}) is not significant enough.")

    non_significant_indices = np.where(~is_element_significant)[0]
    for i in non_significant_indices:
        if np.abs(normalized_vector[i]) > 0.25:
            raise AssertionError(f"Vector element {i} (value: {normalized_vector[i]}) is significant but should not be.")

    # Sign check is global across significant elements:
    # either all signs match expected, or all match expected after a global -1 flip.
    actual_signs = np.sign(normalized_vector[significant_indices])
    expected_signs = np.sign(expected_normalized_vector[significant_indices])
    direct_match = np.array_equal(actual_signs, expected_signs)
    flipped_match = np.array_equal(actual_signs, -expected_signs)
    if not (direct_match or flipped_match):
        raise AssertionError(
            "Sign pattern on significant elements does not match expected "
            "(neither direct nor globally flipped)."
        )


# def test_sensitivity(*, input_value, expected_normalized_vector, expected_magnitude,
#                      verifier=verify_error_vector_is_according_to_expectation):
#     """
#     Stacked case decorator for unittest tests.
#     Decorated test should return: (normalized_vector, magnitude).
#     """
#     def decorator(func):
#         if getattr(func, "_is_sensitivity_wrapper", False):
#             wrapper = func
#         else:
#             @wraps(func)
#             def wrapper(self):
#                 cases = getattr(wrapper, "_sensitivity_cases", [])
#                 if not cases:
#                     raise ValueError(f"No sensitivity cases configured for {func.__name__}")
#                 for case in cases:
#                     case_input = case["input_value"]
#                     case_vector = case["expected_normalized_vector"]
#                     case_magnitude = case["expected_magnitude"]
#                     case_verifier = case["verifier"]
#                     if callable(case_magnitude):
#                         try:
#                             case_magnitude = case_magnitude(self, case_input)
#                         except TypeError:
#                             case_magnitude = case_magnitude(case_input)
#                     with self.subTest(input_value=case_input):
#                         normalized_vector, magnitude = func(self, case_input)
#                         case_verifier(
#                             expected_normalized_vector=case_vector,
#                             expected_magnitude=case_magnitude,
#                             normalized_vector=normalized_vector,
#                             magnitude=magnitude,
#                         )

#             wrapper._sensitivity_cases = []
#             wrapper._is_sensitivity_wrapper = True

#         wrapper._sensitivity_cases.append({
#             "input_value": input_value,
#             "expected_normalized_vector": expected_normalized_vector,
#             "expected_magnitude": expected_magnitude,
#             "verifier": verifier,
#         })
#         return wrapper

#     return decorator
