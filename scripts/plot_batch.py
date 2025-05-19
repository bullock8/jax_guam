import ipdb
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection


def main():
    npz = np.load("bT_state.npz")
    bT_state = npz["aircraft"]

    bT_state = bT_state[::]

    b, T, _ = bT_state.shape
    bT_pos = bT_state[:, :, 6:9]

    dt = 0.005
    T_t = np.arange(T) * dt * 10
    bT_t = np.broadcast_to(T_t, (b, T))

    arrs = [
        bT_pos[:, :, 0],
        bT_pos[:, :, 1],
        bT_pos[:, :, 2],
        bT_state[:, :, 0],
        bT_state[:, :, 1],
        bT_state[:, :, 2],
        bT_state[:, :, 3],
        bT_state[:, :, 4],
        bT_state[:, :, 5],
        bT_state[:, :, 9],
        bT_state[:, :, 10],
        bT_state[:, :, 11],
        bT_state[:, :, 12]
    ]
    labels = [r"$p_x$", r"$p_y$", r"$p_z$", r"$v_x$", r"$v_y$", r"$v_z$", r"$\omega_x$", r"$\omega_y$", r"$\omega_z$", r"$q_1$", r"$q_2$", r"$q_3$", r"$q_4$"]

    # 1: Plot overhead 2d.
    fig, ax = plt.subplots(layout="constrained")
    line_col = LineCollection(bT_pos[:, :, :2][:, :, ::-1], color="C1", lw=0.01, alpha=0.9)
    ax.add_collection(line_col)
    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.set(xlabel="East [ft]", ylabel="North [ft]")
    fig.savefig("batch_traj2d_square.pdf")

    print(f"{np.max(bT_state[:, :, 9])}")

    nrows = len(arrs)
    figsize = np.array([6, 1 * nrows])
    fig, axes = plt.subplots(nrows, figsize=figsize, sharex=True, layout="constrained")
    for ii, ax in enumerate(axes):
        # (b, T, 2)
        lines = np.stack([bT_t, arrs[ii]], axis=-1)
        line_col = LineCollection(lines, color="C1", lw=0.001, alpha=0.1*5)
        ax.add_collection(line_col)
        ax.autoscale_view()
        ax.set_ylabel(labels[ii], rotation=0, ha="right")
    axes[-1].set_xlabel("Time (s)")
    fig.savefig("batch_traj.pdf")


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()
        
