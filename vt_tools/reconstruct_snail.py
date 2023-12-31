import funcy
import numpy as np

from scipy.spatial.distance import euclidean


def calculate_segment_midpoint_and_angle(p1, p2):
    """
    Calculates the midpoint and the angle between two points.

    Args:
    p1 (Tuple): (x, y) coordinates of the first point.
    p2 (Tuple): (x, y) coordinates of the second point.
    """
    x1, y1 = p1
    x2, y2 = p2

    x_mid = min(x2, x1) + np.abs(x2 - x1) / 2
    y_mid = min(y2, y1) + np.abs(y2 - y1) / 2

    midpoint = x_mid, y_mid

    hip = euclidean(p1, p2)
    sin = (y2 - y1) / hip
    cos = (x2 - x1) / hip
    theta = np.arctan2(cos, sin)

    return midpoint, theta


def calculate_points(midpoint, angle, w_int, w_ext):
    """
    Calculates the external and the internal points with respect to the midpoint and the angle.

    Args:
    midpoint (Tuple): (x, y) coordinates of the midpoint.
    angle (float): Rotation angle in radians.
    w_int (float): Internal width.
    w_ext (float): External width.
    """
    x0, y0 = midpoint

    x_ext = w_ext
    y_ext = 0

    x_int = -w_int
    y_int = 0

    points = np.array([
        [x_ext, y_ext],
        [x_int, y_int]
    ])

    rot_mtx = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])

    rot_points = np.matmul(points, rot_mtx)
    p_ext, p_int = rot_points

    p_ext[0] += x0
    p_ext[1] += y0

    p_int[0] += x0
    p_int[1] += y0

    return p_ext, p_int


def reconstruct_apex(center, radius, int_width, ext_width, angle_rad, n_samples=20):
    """
    Reconstruct the snail apex.

    Args:
    center (Tuple): (x, y) coordinates of the central position of the apex.
    radius (float): Radius of the apex.
    int_width (float): Internal width of the apex.
    ext_width (float): External width of the apex.
    angle_rad (float): Angular position of the apex.
    n_samples (int): Number of samples in the curve.
    """
    samples = np.arange(0, n_samples) / n_samples
    radius_int = int_width + (np.sin(np.pi * (-0.5 + samples)) + 1.) / 2 * (radius - int_width)
    radius_ext = ext_width + (np.sin(np.pi * (-0.5 + samples)) + 1.) / 2 * (radius - ext_width)
    alphas = (np.pi / 2) * samples

    apex_int = np.flip(np.array([-int_width * np.cos(alphas), radius_int * np.sin(alphas)]), axis=1)
    apex_ext = np.array([ext_width * np.cos(alphas), radius_ext * np.sin(alphas)])
    apex = np.concatenate([apex_ext, apex_int], axis=1).transpose(1, 0)

    rot_mtx = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad), np.cos(angle_rad)]
    ])

    rot_apex = np.matmul(apex, rot_mtx)
    xc, yc = center
    rot_apex[:, 0] += xc
    rot_apex[:, 1] += yc

    return rot_apex


def reconstruct_snail_from_midline(midline_, width_int, width_ext, width_apex_int, width_apex_ext):
    """
    Reconstructs a snail structure from the midline.

    Args:
    midline_ (np.ndarray): Midline of the snail structure.
    width_int (float): Internal width of the snail.
    width_ext (float): External width of the snail.
    width_apex_int (float): Internal width of the snail apex.
    width_apex_ext (float): External width of the snail apex.
    """
    x_start, _ = midline_[0]
    x_end, _ = midline_[-1]

    if x_start > x_end:
        midline = np.flip(midline_.copy(), axis=0)
    else:
        midline = midline_.copy()

    midline_m1 = midline[:-1]
    midline_p1 = midline[1:]

    segments = list(zip(midline_m1, midline_p1))
    midpoints_angles = [calculate_segment_midpoint_and_angle(*segment) for segment in segments]

    decay_int = np.arctan((width_apex_int - width_int) / len(midpoints_angles))
    widths_int = funcy.lmap(lambda x: decay_int * x + width_int, range(len(midpoints_angles)))

    decay_ext = np.arctan((width_apex_ext - width_ext) / len(midpoints_angles))
    widths_ext = funcy.lmap(lambda x: decay_ext * x + width_ext, range(len(midpoints_angles)))

    midpoints = [d[0] for d in midpoints_angles]
    angles = [d[1] for d in midpoints_angles]

    snail_points = [
        calculate_points(midpoint, angle, w_int=wint, w_ext=wext)
        for (midpoint, angle), wint, wext in zip(midpoints_angles, widths_int, widths_ext)
    ]

    int_snail_points = [d[1] for d in snail_points]
    ext_snail_points = [d[0] for d in reversed(snail_points)]

    xc, yc = midpoints[-1]

    int_width = widths_int[-1]
    ext_width = widths_ext[-1]
    radius = max(int_width, ext_width) * 1.2
    apex_angle = angles[-1]
    apex = reconstruct_apex(
        center=(xc, yc),
        radius=radius,
        int_width=int_width,
        ext_width=ext_width,
        angle_rad=apex_angle
    )

    snail = np.array(int_snail_points + list(np.flip(apex, axis=0)) + ext_snail_points)

    return snail
