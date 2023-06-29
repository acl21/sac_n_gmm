import numpy as np
import mayavi
from mayavi import mlab
from typing import Union, Tuple
import scipy

from lib.dynsys.utils.sphere_utils import (
    get_axisangle,
    rotation_matrix_from_axis_angle,
)


def plot_sphere(
    color: Tuple = (0.7, 0.7, 0.7),
    opacity: float = 0.8,
    radius: float = 0.99,
    n_elems: int = 100,
    figure=None,
    scalars=None,
    colormap=None,
    offset: np.ndarray = np.array([0.0, 0.0, 0.0]),
):
    """
    Plots a sphere

    Optional parameters
    -------------------
    :param color: color of the surface
    :param opacity: transparency index
    :param radius: sphere radius
    :param n_elems: number of points in the surface
    :param figure: mayavi figure handle
    :param base_point: center of the sphere

    Returns
    -------
    :return: -
    """
    u = np.linspace(0, 2 * np.pi, n_elems)
    v = np.linspace(0, np.pi, n_elems)

    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones(np.size(u)), np.cos(v))

    if scalars is not None:
        mlab.mesh(
            x + offset[0],
            y + offset[1],
            z + offset[2],
            figure=figure,
            opacity=opacity,
            scalars=scalars,
            colormap=colormap,
        )
    elif colormap is not None:
        mlab.mesh(
            x + offset[0],
            y + offset[1],
            z + offset[2],
            figure=figure,
            opacity=opacity,
            colormap=colormap,
        )
    else:
        mlab.mesh(
            x + offset[0],
            y + offset[1],
            z + offset[2],
            figure=figure,
            color=color,
            opacity=opacity,
        )


def plot_sphere_tangent_plane(
    base: np.ndarray,
    l_vert: float = 1,
    opacity: float = 0.5,
    figure=None,
    offset: np.ndarray = np.array([0.0, 0.0, 0.0]),
):
    """
    This function plots the tangent plane of a point lying on the sphere manifold.
    Based on the function of riepybdlib (https://gitlab.martijnzeestraten.nl/martijn/riepybdlib)

    Parameters
    ----------
    :param base: base point of the tangent space

    Optional parameters
    -------------------
    :param l_vert: length/width of the displayed plane
    :param opacity: transparency index
    :param figure: mayavi figure handle

    Returns
    -------
    :return: -
    """
    # Tangent axis at 0 rotation:
    T0 = np.array([[1, 0], [0, 1], [0, 0]])

    # Rotation matrix with respect to zero:
    axis, ang = get_axisangle(base)
    R = rotation_matrix_from_axis_angle(axis, -ang)

    # Tangent axis in new plane:
    T = R.T.dot(T0)

    # Compute vertices of tangent plane at g
    hl = 0.5 * l_vert
    X = [[hl, hl], [hl, -hl], [-hl, hl], [-hl, -hl]]  # p0  # p1  # p2  # p3
    X = np.array(X).T
    points = (T.dot(X).T + base).T

    # Plot tangent space
    psurf = points.reshape((-1, 2, 2))
    mlab.mesh(
        psurf[0] + offset[0],
        psurf[1] + offset[1],
        psurf[2] + offset[2],
        color=(0.9, 0.9, 0.9),
        opacity=opacity,
        figure=figure,
    )

    # Plot contours of the tangent space
    contour = points[:, [0, 1, 3, 2, 0]]
    mlab.plot3d(
        contour[0] + offset[0],
        contour[1] + offset[1],
        contour[2] + offset[2],
        color=(0, 0, 0),
        line_width=2.0,
        tube_radius=None,
        figure=figure,
    )


def plot_vector_on_tangent_plane(
    base: np.ndarray,
    vector: np.ndarray,
    color: Tuple = (0.8, 0.8, 0.8),
    line_width: float = 3.0,
    r: float = 0.03,
    figure=None,
):
    """
    This function draws a vector on the tangent space of a given point on a 2D-sphere.

    Parameters
    ----------
    :param base: base point of the tangent space
    :param vector: vector to draw

    Optional parameters
    -------------------
    :param color: color of the vector
    :param line_width: width of the vector
    :param r: maximum radius of the tip of the arrow
    :param figure: mayavi figure handle

    Returns
    -------
    :return: -
    """
    if vector.shape[0] == 2:
        x = np.ones(3)
        x[0:2] = vector
        # Rotation matrix with respect to zero:
        axis, ang = get_axisangle(base)
        R = rotation_matrix_from_axis_angle(axis, ang)
        # Vector on tangent space
        tgt_vector = R.dot(x)
    else:
        # Case where the vector is already 3-dimensional
        tgt_vector = vector + base

    draw_arrow_mayavi(
        base[0],
        base[1],
        base[2],
        tgt_vector[0] - base[0],
        tgt_vector[1] - base[1],
        tgt_vector[2] - base[2],
        r=r,
        color=color,
        line_width=line_width,
        figure=figure,
    )


def draw_arrow_mayavi(
    x: float,
    y: float,
    z: float,
    u: float,
    v: float,
    w: float,
    r: float = 0.025,
    color: Tuple = (0, 0, 0),
    line_width: float = 2.0,
    n_elems: int = 30,
    figure=None,
):
    """
    This function draws an arrow with mayavi.

    Parameters
    ----------
    :param x: x-coordinate of the starting point of the arrow
    :param y: y-coordinate of the starting point of the arrow
    :param z: z-coordinate of the starting point of the arrow
    :param u: x-length of the arrow
    :param v: y-length of the arrow
    :param w: z-length of the arrow

    Optional parameters
    -------------------
    :param r: maximum radius of the tip of the arrow
    :param color: color of the arrow
    :param line_width: line width of the line of the arrow
    :param n_elems: number of elements used to draw the tip of the arrow
    :param figure: mayavi figure handle

    Returns
    -------
    :return: -
    """

    phi = np.linspace(0, 2 * np.pi, n_elems)

    # Arrow coordinates
    base = np.stack((x, y, z))
    diff_base_tip = np.stack((u, v, w))
    tip = np.stack((u, v, w)) + base

    # Rotation of the cone
    # 90°
    axis = np.array([0, 1, 0])
    R = rotation_matrix_from_axis_angle(axis, np.pi / 2.0)

    # Arrow direction
    axis, ang = get_axisangle(diff_base_tip / np.linalg.norm(diff_base_tip))
    R = rotation_matrix_from_axis_angle(axis, ang).dot(R)

    # Points of the cone
    xyz = np.vstack(
        (r * 2 * np.ones(n_elems), r * np.sin(phi), r / np.sqrt(2) * np.cos(phi))
    )
    xyz = R.dot(xyz)

    xcone = np.vstack((np.zeros(n_elems), xyz[0])) + tip[0]
    ycone = np.vstack((np.zeros(n_elems), xyz[1])) + tip[1]
    zcone = np.vstack((np.zeros(n_elems), xyz[2])) + tip[2]

    # Plot vector
    vector = np.stack((base, tip))
    mlab.plot3d(
        vector[:, 0],
        vector[:, 1],
        vector[:, 2],
        color=color,
        line_width=line_width,
        tube_radius=None,
        figure=None,
    )
    # Plot tip
    mlab.mesh(xcone, ycone, zcone, color=color)


def plot_gaussian_mesh_on_tangent_plane(
    mu, sigma, color, offset: np.ndarray = np.array([0.0, 0.0, 0.0])
):
    """
    Plot the Gaussian with the mean mu and covariance matrix sigma in the 2D-sphere.

    Parameters
    ----------
    :param mu: The mean of the Gaussian
    :param sigma: Covariance Matrix of the Gaussian
    :param color: Color of the plotted Gaussian
    """
    # Generate Points
    num = 50
    t = np.linspace(-np.pi, np.pi, num)

    # Angle axis and rotation matrix representation of the mean of the Gaussian
    axis, angle = get_axisangle(mu)
    R = rotation_matrix_from_axis_angle(axis, angle)

    # 2D representation of the covariance in the manifold "origin" (1, 0, 0)
    sigma2d = np.dot(R.T, np.dot(sigma, R))[0:2, 0:2]

    # Eigenvalue decomposition of the 2D covariance
    D, V = np.linalg.eig(sigma2d)
    D = np.real(D)
    V = np.real(V)

    # Rotation for covariance
    R = np.eye(3)
    R[0:2, 0:2] = V

    # Angle axis representation of the mean of the Gaussian
    axis, angle = get_axisangle(mu)
    R = rotation_matrix_from_axis_angle(axis, angle).dot(R)

    # Points of the 2d covariance for inner mesh
    n_elems = 30
    u = np.linspace(0, 2 * np.pi, n_elems)
    v = np.linspace(0, np.pi, n_elems)

    x = np.sqrt(D[0]) * np.outer(np.cos(u), np.sin(v))
    y = np.sqrt(D[1]) * np.outer(np.sin(u), np.sin(v))
    z = 1e-8 * np.outer(np.ones(np.size(u)), np.cos(v))

    xyz0 = np.stack((x, y, z), axis=2)
    xyz0 = np.reshape(xyz0, (n_elems * n_elems, 3)).T
    xyz = np.dot(R, xyz0)
    xyz = np.reshape(xyz.T, (n_elems, n_elems, 3))

    x = xyz[:, :, 0] + mu[0] + offset[0]
    y = xyz[:, :, 1] + mu[1] + offset[1]
    z = xyz[:, :, 2] + mu[2] + offset[2]

    # Contour
    Rc = np.eye(3)
    Rc[0:2, 0:2] = np.real(scipy.linalg.sqrtm(sigma2d))
    Rc = rotation_matrix_from_axis_angle(axis, angle).dot(Rc)
    contour = np.vstack((np.cos(t), np.sin(t), np.ones(num)))
    contour = Rc.dot(contour) + offset[:, None]  # Points rotated in Tangent Space of mu

    mlab.mesh(x, y, z, color=color, opacity=0.15)
    # mlab.plot3d(contour[0, :], contour[1, :], contour[2, :], color=color, line_width=3, tube_radius=None, opacity=1.)
