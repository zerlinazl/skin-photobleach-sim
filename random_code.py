import numpy as np

def rotate_axes_xz(points, angle_deg=5):
    """
    Rotate coordinate axes in the xz-plane and compute new coordinates.

    points: array of shape (N,2) with columns [x, z]
    angle_deg: rotation angle in degrees

    returns:
        new_points: rotated coordinates
        delta: change in coordinates
    """
    
    theta = np.deg2rad(angle_deg)

    # rotation matrix for rotating axes
    R = np.array([
        [np.cos(theta),  np.sin(theta)],
        [-np.sin(theta), np.cos(theta)]
    ])

    new_points = points @ R.T
    delta = new_points - points

    return new_points, delta


# example points
points = np.array([
    [0,0],
    [100, 50],
    [200, 0]
])

new_points, delta = rotate_axes_xz(points, 5)

print("Original points:\n", points)
print("\nNew coordinates:\n", new_points)
print("\nChange in coordinates:\n", delta)