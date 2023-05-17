import os
import inspect
from SkillsRefining.skills.mps.dynsys.FNN import SimpleNN
from SkillsRefining.skills.mps.dynsys.WSAQF import WSAQF
from SkillsRefining.skills.mps.dynsys.CLFDS import CLFDS
from SkillsRefining.skills.mps.dynsys.dsdataset import DSDataSet
from SkillsRefining.utils.plot_utils import plot_2d_ds_velocity_field
import numpy as np
import matplotlib.pyplot as plt
from SkillsRefining.utils.utils import prepare_torch

device = prepare_torch()
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
os.sys.path.insert(0, current_dir)
os.sys.path.insert(0, "..")


def learn_ds_skills(filename="pick"):
    dirname = "demos"
    outdir = "pickplace_ds_skill"
    trajs = []
    _, ax = plt.subplots(1, 1)
    for i in range(1, 7):
        fname = os.path.join(dirname, filename + "_" + str(i) + ".csv")
        traj = np.loadtxt(fname, delimiter=",", skiprows=1)
        trajs.append(traj[:, [1, 2]])
        ax.plot(traj[:, 1], traj[:, 2], "k-.", label="demos")

    dataset = DSDataSet(
        trajs, with_timesteps=False, move_traj_to_zero=True, velocity_in_file=False
    )
    clf_model = WSAQF(dim=2, n_qfcn=1)
    reg_model = SimpleNN(2, 2, (20, 20))
    clfds = CLFDS(clf_model, reg_model, rho_0=0.01, kappa_0=0.0001)
    clfds.train_clf(
        dataset,
        lr=1e-3,
        max_epochs=3000,
        batch_size=100,
        load_if_possible=False,
        fname=os.path.join(outdir, filename + "_posi_clf"),
    )
    dataset = clfds.collect_ds_data(dataset)
    clfds.train_ds(
        dataset,
        lr=1e-3,
        max_epochs=3000,
        batch_size=10,
        load_if_possible=False,
        fname=os.path.join(outdir, filename + "_posi_ds"),
    )

    workspace = (-20, 20, -20, 20)
    gridsize = 1000

    plot_2d_ds_velocity_field(
        ax, clfds.reg_model, workspace, gridsize, offset=dataset.offset, cmap="summer"
    )
    if filename == "pick":
        color = [70 / 255, 100 / 255, 170 / 255, 0.5]
    else:
        color = [162 / 255, 34 / 255, 35 / 255, 0.5]

    ax.plot(dataset.offset[0], dataset.offset[1], ".", color=color, markersize=30)
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])
    plt.xlabel(r"x", fontsize=10)
    plt.ylabel(r"y", fontsize=10)
    plt.tight_layout()
    plt.savefig(filename + "_ds")
    plt.show()


def show_ds(filename="pick"):
    dirname = "demos"
    outdir = "pickplace_ds_skill"
    trajs = []
    for i in range(1, 7):
        fname = os.path.join(dirname, filename + "_" + str(i) + ".csv")
        traj = np.loadtxt(fname, delimiter=",", skiprows=1)
        trajs.append(traj[:, [1, 2]])


if __name__ == "__main__":
    learn_ds_skills("place")
    learn_ds_skills("pick")
