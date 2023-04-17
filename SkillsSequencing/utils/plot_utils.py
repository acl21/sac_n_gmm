import os
import numpy as np
import scipy as sp
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.animation as animation

from SkillsSequencing.utils.utils import prepare_torch
device = prepare_torch()


def plot_gaussian_covariance(ax, mean, covariance_matrix, color, edgecolor=None, transparency=0.5, linewidth=1, zorder=20):
    nbDrawingSeg = 35
    t = np.linspace(-np.pi, np.pi, nbDrawingSeg)

    # Create Polygon
    R = np.real(sp.linalg.sqrtm(covariance_matrix))
    points = R.dot(np.array([[np.cos(t)], [np.sin(t)]]).reshape([2, nbDrawingSeg])) + mean[:, None]

    if edgecolor is None:
        edgecolor = color

    polygon = plt.Polygon(points.transpose().tolist(), facecolor=color, alpha=transparency, linewidth=linewidth,
                          zorder=zorder, edgecolor=edgecolor)

    ax.add_patch(polygon)


def plot_animated_ds(x0, ds, offset=0, ax=None, traj=None, is_3d=False, dt=0.01, T=100):
    if is_3d:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        if traj is not None:
            ax.plot(traj[:,0], traj[:,1], traj[:,2], 'k-.')
    else:
        if ax is None:
            _, ax = plt.subplots()
        if traj is not None:
            ax.plot(traj[:,0], traj[:,1], 'k-.')

    test_traj = np.array([x0])
    for _ in range(T):
        dx = ds.forward(torch.from_numpy(x0-offset).double().to(device)).squeeze()
        x0 = x0 + dx.detach().cpu().numpy() * dt
        test_traj = np.append(test_traj, np.expand_dims(x0, axis=0), axis=0)

        if is_3d:
            ax.plot(test_traj[:, 0], test_traj[:, 1], test_traj[:,2], 'b-', linewidth=3)
        else:
            ax.plot(test_traj[:, 0], test_traj[:, 1], '-', color=kit_colors['cb'], linewidth=2)

        plt.pause(0.1)

    plt.show()


def plot_2d_ds_velocity_field(ax, ds, workspace, gridsize, offset=np.array([0,0]),  cmap='summer'):
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
    X = np.reshape(xgrid[:, 0]+offset[0], newshape=(gridsize, gridsize), order='F')
    Y = np.reshape(xgrid[:, 1]+offset[1], newshape=(gridsize, gridsize), order='F')
    U = np.reshape(vel[:, 0], newshape=(gridsize, gridsize), order='F')
    V = np.reshape(vel[:, 1], newshape=(gridsize, gridsize), order='F')
    ax.streamplot(X, Y, U, V, density=[1.5, 1.5], color=kit_colors['cb'])


def plot_2d_weighted_sum_ds_velocity_field(ax, ds1, ds2, w1, w2, workspace, gridsize, offset1=np.array([0, 0]),
                                           offset2=np.array([0, 0]), cmap='summer'):
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
    X = np.reshape(xgrid1[:, 0]+offset1[0], newshape=(gridsize, gridsize), order='F')
    Y = np.reshape(xgrid1[:, 1]+offset1[1], newshape=(gridsize, gridsize), order='F')
    U = np.reshape(vel[:, 0], newshape=(gridsize, gridsize), order='F')
    V = np.reshape(vel[:, 1], newshape=(gridsize, gridsize), order='F')
    ax.streamplot(X, Y, U, V, density=[1.5, 1.5], cmap=cmap)


def plot_2d_energy_field(ax, ds, workspace, gridsize, offset=np.array([0,0]), with_contour=True, cmap='summer',
                         with_default_level=True):
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
    X = np.reshape(xgrid[:, 0], newshape=(gridsize, gridsize), order='F')
    Y = np.reshape(xgrid[:, 1], newshape=(gridsize, gridsize), order='F')
    U = np.reshape(vel[:, 0], newshape=(gridsize, gridsize), order='F')
    V = np.reshape(vel[:, 1], newshape=(gridsize, gridsize), order='F')
    e = np.reshape(e.detach().cpu().numpy(), newshape=(gridsize, gridsize), order='F')
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

            cs = ax.contour(X, Y, e,  colors='k', alpha=0.5, levels=levels)
        else:
            cs = ax.contour(X, Y, e,  colors='k', alpha=0.5)

        ax.clabel(cs, cs.levels, inline=True,  fontsize=10)#, fmt=fmt)


# Credits: Yizhak Ben-Shabat
# Source: https://github.com/sitzikbs/gmm_tutorial/
# Notes: Slightly modified
def visualize_3d_gmm(points, w, mu, stdev, skill, export_dir, export_type='gif'):
    '''
    plots points and their corresponding gmm model in 3D
    Input: 
        points: N X 3, sampled points
        w: n_gaussians, gmm weights
        mu: 3 X n_gaussians, gmm means
        stdev: 3 X n_gaussians, gmm standard deviation (assuming diagonal covariance matrix)
    Output:
        None
    '''
    n_gaussians = mu.shape[1]
    N = int(np.floor(points.shape[0] / n_gaussians))
    # Visualize data
    fig = plt.figure(figsize=(8, 8))
    axes = fig.add_subplot(111, projection='3d')
    plt.set_cmap('Set1')
    for i in range(n_gaussians):
        idx = range(i * N, (i + 1) * N)
        axes.scatter(points[idx, 0], points[idx, 1], points[idx, 2], alpha=0.3, c=[[0,0,0]])
        plot_sphere3d(w=w[i], c=mu[:, i], r=stdev[:, i], ax=axes)

    plt.title(f'3D GMM: {skill}')
    axes.set_xlabel('X')
    axes.set_ylabel('Y')
    axes.set_zlabel('Z')

    outfile = None
    if export_type == 'gif':
        def init():
            axes.view_init(elev=28, azim=0)
            return fig,

        def animate(i):
            axes.view_init(elev=28, azim=i)
            return fig,

        # Animate
        anim = animation.FuncAnimation(fig, animate, init_func=init,
                                        frames=180, interval=30, blit=True)
        # Save
        anim.save(os.path.join(export_dir, 'gmm.gif'), fps=20)
        outfile = os.path.join(export_dir, 'gmm.gif')
    else:
        axes.view_init(elev=35.246, azim=45)
        plt.savefig(os.path.join(export_dir, 'gmm.png'), dpi=100, format='png')
        outfile = os.path.join(export_dir, 'gmm.png')
    return outfile

# Credits: Yizhak Ben-Shabat
# Source: https://github.com/sitzikbs/gmm_tutorial/
def plot_sphere3d(w=0, c=[0,0,0], r=[1, 1, 1], subdev=10, ax=None, sigma_multiplier=3):
    '''
        plot a sphere surface
        Input: 
            c: 3 elements list, sphere center
            r: 3 element list, sphere original scale in each axis ( allowing to draw elipsoids)
            subdiv: scalar, number of subdivisions (subdivision^2 points sampled on the surface)
            ax: optional pyplot axis object to plot the sphere in.
            sigma_multiplier: sphere additional scale (choosing an std value when plotting gaussians)
        Output:
            ax: pyplot axis object
    '''

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    pi = np.pi
    cos = np.cos
    sin = np.sin
    phi, theta = np.mgrid[0.0:pi:complex(0,subdev), 0.0:2.0 * pi:complex(0,subdev)]
    x = sigma_multiplier*r[0] * sin(phi) * cos(theta) + c[0]
    y = sigma_multiplier*r[1] * sin(phi) * sin(theta) + c[1]
    z = sigma_multiplier*r[2] * cos(phi) + c[2]
    cmap = cmx.ScalarMappable()
    cmap.set_cmap('prism')
    c = cmap.to_rgba(w)

    ax.plot_surface(x, y, z, color=c, alpha=0.2, linewidth=1)

    return ax