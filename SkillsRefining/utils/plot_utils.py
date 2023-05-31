import os
import numpy as np
import scipy as sp
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.animation as animation

from SkillsRefining.utils.utils import prepare_torch

device = prepare_torch()


def plot_gaussian_covariance(
    ax,
    mean,
    covariance_matrix,
    color,
    edgecolor=None,
    transparency=0.5,
    linewidth=1,
    zorder=20,
):
    nbDrawingSeg = 35
    t = np.linspace(-np.pi, np.pi, nbDrawingSeg)

    # Create Polygon
    R = np.real(sp.linalg.sqrtm(covariance_matrix))
    points = (
        R.dot(np.array([[np.cos(t)], [np.sin(t)]]).reshape([2, nbDrawingSeg]))
        + mean[:, None]
    )

    if edgecolor is None:
        edgecolor = color

    polygon = plt.Polygon(
        points.transpose().tolist(),
        facecolor=color,
        alpha=transparency,
        linewidth=linewidth,
        zorder=zorder,
        edgecolor=edgecolor,
    )

    ax.add_patch(polygon)


def plot_animated_ds(x0, ds, offset=0, ax=None, traj=None, is_3d=False, dt=0.01, T=100):
    if is_3d:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        if traj is not None:
            ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], "k-.")
    else:
        if ax is None:
            _, ax = plt.subplots()
        if traj is not None:
            ax.plot(traj[:, 0], traj[:, 1], "k-.")

    test_traj = np.array([x0])
    for _ in range(T):
        dx = ds.forward(torch.from_numpy(x0 - offset).double().to(device)).squeeze()
        x0 = x0 + dx.detach().cpu().numpy() * dt
        test_traj = np.append(test_traj, np.expand_dims(x0, axis=0), axis=0)

        if is_3d:
            ax.plot(
                test_traj[:, 0], test_traj[:, 1], test_traj[:, 2], "b-", linewidth=3
            )
        else:
            ax.plot(
                test_traj[:, 0],
                test_traj[:, 1],
                "-",
                linewidth=2,
            )

        plt.pause(0.1)

    plt.show()


def plot_2d_ds_velocity_field(
    ax, ds, workspace, gridsize, offset=np.array([0, 0]), cmap="summer"
):
    x1 = np.linspace(workspace[0], workspace[1], gridsize)
    x2 = np.linspace(workspace[2], workspace[3], gridsize)
    xgrid = []
    for i in range(len(x1)):
        for j in range(len(x2)):
            xgrid.append(np.array([x1[i], x2[j]]) - offset)

    xgrid = np.stack(xgrid)
    grid_input = torch.from_numpy(xgrid).double().to(device)
    e = ds.forward(grid_input)
    vel = e.detach().cpu().numpy()
    X = np.reshape(xgrid[:, 0] + offset[0], newshape=(gridsize, gridsize), order="F")
    Y = np.reshape(xgrid[:, 1] + offset[1], newshape=(gridsize, gridsize), order="F")
    U = np.reshape(vel[:, 0], newshape=(gridsize, gridsize), order="F")
    V = np.reshape(vel[:, 1], newshape=(gridsize, gridsize), order="F")
    ax.streamplot(X, Y, U, V, density=[1.5, 1.5], cmap=cmap)


def plot_2d_weighted_sum_ds_velocity_field(
    ax,
    ds1,
    ds2,
    w1,
    w2,
    workspace,
    gridsize,
    offset1=np.array([0, 0]),
    offset2=np.array([0, 0]),
    cmap="summer",
):
    x1 = np.linspace(workspace[0], workspace[1], gridsize)
    x2 = np.linspace(workspace[2], workspace[3], gridsize)
    xgrid1 = []
    xgrid2 = []
    for i in range(len(x1)):
        for j in range(len(x2)):
            xgrid1.append(np.array([x1[i], x2[j]]) - offset1)
            xgrid2.append(np.array([x1[i], x2[j]]) - offset2)

    xgrid1 = np.stack(xgrid1)
    xgrid2 = np.stack(xgrid2)
    grid_input1 = torch.from_numpy(xgrid1).double().to(device)
    grid_input2 = torch.from_numpy(xgrid2).double().to(device)

    e = w1 * ds1.forward(grid_input1) + w2 * ds2.forward(grid_input2)
    vel = e.detach().cpu().numpy()
    X = np.reshape(xgrid1[:, 0] + offset1[0], newshape=(gridsize, gridsize), order="F")
    Y = np.reshape(xgrid1[:, 1] + offset1[1], newshape=(gridsize, gridsize), order="F")
    U = np.reshape(vel[:, 0], newshape=(gridsize, gridsize), order="F")
    V = np.reshape(vel[:, 1], newshape=(gridsize, gridsize), order="F")
    ax.streamplot(X, Y, U, V, density=[1.5, 1.5], cmap=cmap)


def plot_2d_energy_field(
    ax,
    ds,
    workspace,
    gridsize,
    offset=np.array([0, 0]),
    with_contour=True,
    cmap="summer",
    with_default_level=True,
):
    x1 = np.linspace(workspace[0], workspace[1], gridsize)
    x2 = np.linspace(workspace[2], workspace[3], gridsize)
    xgrid = []
    for i in range(len(x1)):
        for j in range(len(x2)):
            xgrid.append(np.array([x1[i], x2[j]]) - offset)

    xgrid = np.stack(xgrid)
    grid_input = torch.from_numpy(xgrid).double().to(device)
    e, vel = ds.forward_with_grad(grid_input)
    vel = -vel.detach().cpu().numpy()
    X = np.reshape(xgrid[:, 0], newshape=(gridsize, gridsize), order="F")
    Y = np.reshape(xgrid[:, 1], newshape=(gridsize, gridsize), order="F")
    U = np.reshape(vel[:, 0], newshape=(gridsize, gridsize), order="F")
    V = np.reshape(vel[:, 1], newshape=(gridsize, gridsize), order="F")
    e = np.reshape(e.detach().cpu().numpy(), newshape=(gridsize, gridsize), order="F")
    ax.streamplot(X, Y, U, V, density=[1.5, 1.5], color=-e, cmap=cmap)

    if with_contour:
        if not with_default_level:
            levels = set()
            minpos = np.where(e == np.min(e))
            maxpos = np.where(e == np.max(e))
            xs = np.linspace(minpos[0], maxpos[0], 20)
            ys = np.linspace(minpos[1], maxpos[1], 20)
            print(xs.shape)
            for i in range(min(len(xs), len(ys))):
                levels.add(e[int(xs[i]), int(ys[i])])

            levels = sorted(list(levels))

            cs = ax.contour(X, Y, e, colors="k", alpha=0.5, levels=levels)
        else:
            cs = ax.contour(X, Y, e, colors="k", alpha=0.5)

        ax.clabel(cs, cs.levels, inline=True, fontsize=10)  # , fmt=fmt)


def visualize_3d_gmm(points, priors, means, covariances, save_dir):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    colors = ["r", "g", "b", "c", "m", "y"]  # Color for each component

    def update_frame(i):
        ax.cla()  # Clear the previous frame

        for j in range(len(priors)):
            eigenvalues, eigenvectors = np.linalg.eigh(covariances[j])
            scaling_factors = np.sqrt(eigenvalues)
            u = np.linspace(0, 2 * np.pi, 100)
            v = np.linspace(0, np.pi, 100)
            x = scaling_factors[0] * np.outer(np.cos(u), np.sin(v))
            y = scaling_factors[1] * np.outer(np.sin(u), np.sin(v))
            z = scaling_factors[2] * np.outer(np.ones_like(u), np.cos(v))

            for k in range(len(x)):
                for l in range(len(x[k])):
                    [x[k][l], y[k][l], z[k][l]] = (
                        np.dot([x[k][l], y[k][l], z[k][l]], eigenvectors) + means[j]
                    )

            ax.plot_surface(x, y, z, color=colors[j], alpha=0.3)

        ax.scatter(points[:, 0], points[:, 1], points[:, 2], alpha=0.3, c=[[0, 0, 0]])
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("Gaussian Mixture Model")

        ax.view_init(elev=28, azim=i * 4)  # Adjust the view angle

    frames = 90  # Number of frames in the animation
    ani = animation.FuncAnimation(fig, update_frame, frames=frames, interval=50)

    ani.save(
        os.path.join(save_dir, "gmm.gif"), writer="imagemagick"
    )  # Save the animation as a GIF
    return os.path.join(save_dir, "gmm.gif")
