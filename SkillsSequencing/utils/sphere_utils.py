from typing import Union, Tuple
import numpy as np

def get_axisangle(d: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    This function gets the axis-angle representation of a point lying on a unit sphere
    Based on the function of riepybdlib (https://gitlab.martijnzeestraten.nl/martijn/riepybdlib)

    Parameters
    ----------
    :param d: point on the sphere

    Returns
    -------
    :return: axis, angle: corresponding axis and angle representation
    """
    norm = np.sqrt(d[0]**2 + d[1]**2)
    if norm < 1e-6:
        return np.array([0, 0, 1]), 0
    else:
        vec = np.array([-d[1], d[0], 0])
        return vec/norm, np.arccos(d[2])


def rotation_matrix_to_unit_sphere(R: np.ndarray) -> Union[np.ndarray, int]:
    """
    This function transforms a rotation matrix to a point lying on a sphere (i.e., unit vector).
    This function is valid for rotation matrices of dimension 2 (to S1) and 3 (to S3).

    Parameters
    ----------
    :param R: rotation matrix

    Returns
    -------
    :return: a unit vector on S1 or S3, or -1 if the dimension of the rotation matrix cannot be handled.
    """
    if R.shape[0] == 3:
        return rotation_matrix_to_quaternion(R)
    elif R.shape[0] == 2:
        return R[:, 0]
    else:
        return -1


def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """
    This function transforms a 3x3 rotation matrix into a quaternion.
    This function was implemented based on Peter Corke's robotics toolbox.

    Parameters
    ----------
    :param R: 3x3 rotation matrix

    Returns
    -------
    :return: a quaternion [scalar term, vector term]
    """

    qs = min(np.sqrt(np.trace(R) + 1)/2.0, 1.0)
    kx = R[2, 1] - R[1, 2]   # Oz - Ay
    ky = R[0, 2] - R[2, 0]   # Ax - Nz
    kz = R[1, 0] - R[0, 1]   # Ny - Ox

    if (R[0, 0] >= R[1, 1]) and (R[0, 0] >= R[2, 2]) :
        kx1 = R[0, 0] - R[1, 1] - R[2, 2] + 1 # Nx - Oy - Az + 1
        ky1 = R[1, 0] + R[0, 1]               # Ny + Ox
        kz1 = R[2, 0] + R[0, 2]               # Nz + Ax
        add = (kx >= 0)
    elif (R[1, 1] >= R[2, 2]):
        kx1 = R[1, 0] + R[0, 1]               # Ny + Ox
        ky1 = R[1, 1] - R[0, 0] - R[2, 2] + 1 # Oy - Nx - Az + 1
        kz1 = R[2, 1] + R[1, 2]               # Oz + Ay
        add = (ky >= 0)
    else:
        kx1 = R[2, 0] + R[0, 2]               # Nz + Ax
        ky1 = R[2, 1] + R[1, 2]               # Oz + Ay
        kz1 = R[2, 2] - R[0, 0] - R[1, 1] + 1 # Az - Nx - Oy + 1
        add = (kz >= 0)

    if add:
        kx = kx + kx1
        ky = ky + ky1
        kz = kz + kz1
    else:
        kx = kx - kx1
        ky = ky - ky1
        kz = kz - kz1

    nm = np.linalg.norm(np.array([kx, ky, kz]))
    if nm == 0:
        q = np.zeros(4)
    else:
        s = np.sqrt(1 - qs**2) / nm
        qv = s*np.array([kx, ky, kz])
        q = np.hstack((qs, qv))

    return q


def rotation_matrix_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    """
    Gets rotation matrix from axis angle representation using Rodriguez formula.
    Based on the function of riepybdlib (https://gitlab.martijnzeestraten.nl/martijn/riepybdlib)

    Parameters
    ----------
    :param axis: unit axis defining the axis of rotation
    :param angle: angle of rotation

    Returns
    -------
    :return: R(ax, angle) = I + sin(angle) x ax + (1 - cos(angle) ) x ax^2 with x the cross product.
    """
    utilde = vector_to_skew_matrix(axis)
    return np.eye(3) + np.sin(angle)*utilde + (1 - np.cos(angle))*utilde.dot(utilde)


def vector_to_skew_matrix(q: np.ndarray) -> np.ndarray:
    """
    Transform a vector into a skew-symmetric matrix

    Parameters
    ----------
    :param q: vector

    Returns
    -------
    :return: corresponding skew-symmetric matrix
    """
    return np.array([[0, -q[2], q[1]], [q[2], 0, -q[0]], [-q[1], q[0], 0]])
