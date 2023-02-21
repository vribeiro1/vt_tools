import numpy as np

from numba import jit


@jit(nopython=True)
def euclidean(u, v):
    """
    Calculates the euclidean distance between two points. Use numba to accelerate the calculation.

    Args:
        u (Tuple[float, float]): First point to calculate the euclidean distance.
        v (Tuple[float, float]): Second point to calculate the euclidean distance.

    >>> u = np.array([3., 0.])
    >>> v = np.array([0., 4.])
    >>> euclidean(u, v)
    5.0
    """
    x_u, y_u = u
    x_v, y_v = v

    return np.sqrt((x_u - x_v) ** 2 + (y_u - y_v) ** 2)


@jit(nopython=True)
def p2cp(i, u, v):
    """
    Calculates the point to closest point distance between the u[i] point to the v curve.

    Args:
        i (int): Index of the point in the u-array.
        u (np.ndarray): Array of shape (N, 2) that contains the target point.
        v (np.ndarray): Array of shape (N, 2) to calculate the closest point.
    """
    ui = u[i]
    ui2cp = min(euclidean(ui, vj) for vj in v)
    return ui2cp


@jit(nopython=True)
def distance_matrix(u, v):
    """
    Calculates the distance between the elements of two arrays.

    Args:
        u (np.ndarray): Array of shape (N, 2).
        v (np.ndarray): Array of shape (N, 2).
    """
    n = len(u)
    m = len(v)

    dist_mtx = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            dist_mtx[i][j] = euclidean(u[i], v[j])

    return dist_mtx


def p2cp_mean(u_, v_):
    """
    Calculates the mean point-to-closest-point distance between two arrays.

    Args:
        u_ (np.ndarray): Array of shape (N, 2).
        v_ (np.ndarray): Array of shape (N, 2).
    """
    n = len(u_)
    m = len(v_)

    dist_mtx = distance_matrix(u_, v_)

    u2cv = dist_mtx.min(axis=1)
    v2cu = dist_mtx.min(axis=0)
    mean_p2cp = ((np.sum(u2cv) / n) + (np.sum(v2cu) / m)) / 2

    return mean_p2cp

def p2cp_rms(u_, v_):
    """
    Calculates the RMS value of the point-to-closest-point distance between two arrays.

    Args:
        u_ (np.ndarray): Array of shape (N, 2).
        v_ (np.ndarray): Array of shape (N, 2).
    """
    n = len(u_)
    m = len(v_)
    assert n == m, "Arrays have to have the same length for the RMS value."

    dist_mtx = distance_matrix(u_, v_)
    u2cv = dist_mtx.min(axis=1)
    v2cu = dist_mtx.min(axis=0)
    p2cp = (u2cv + v2cu) / 2
    p2cp_rms = (p2cp ** 2).sum() / n

    return p2cp_rms
