import matplotlib.pyplot as plt

def plot_3d_trajectories(demos, repro=None, goal=None, figsize=(4,4)):
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(projection="3d")
        if goal is not None:
            ax.scatter(goal[0], goal[1], goal[2], s=15, label='Goal')
        if repro is None:
            for i in range(demos.shape[0]):
                x_val = demos[i, :, 0]
                y_val = demos[i, :, 1]
                z_val = demos[i, :, 2]
                ax.scatter(x_val, y_val, z_val, s=10)
        else:
            ax.scatter(demos[:, 0], demos[:, 1], demos[:, 2], alpha=0.5, s=1, label="Demonstration")
            ax.scatter(repro[:, 0], repro[:, 1], repro[:, 2], s=5, label="Reproduction")
        plt.legend()
        plt.tight_layout()
        plt.show()