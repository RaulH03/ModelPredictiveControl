import numpy as np
import matplotlib.pyplot as plt

from run_Linear_MPC import run_mpc


def compare_horizons():
    # Fixed settings
    x0 = [0.0, np.deg2rad(20), 0.0, 0.0]
    u_max = 1.5
    theta1_max = np.deg2rad(25)
    theta2_dev_max = np.deg2rad(25)
    steps = 100

    # Fixed tuning
    q_theta1 = 50
    q_delta_theta2 = 200
    q_theta1_dot = 1
    q_theta2_dot = 1
    r_input = 1

    # Horizons to compare
    horizons = [3, 5, 10, 20]

    results = []

    for N in horizons:
        out = run_mpc(
            q_theta1=q_theta1,
            q_delta_theta2=q_delta_theta2,
            q_theta1_dot=q_theta1_dot,
            q_theta2_dot=q_theta2_dot,
            r_input=r_input,
            x0=x0,
            u_max=u_max,
            theta1_max=theta1_max,
            theta2_dev_max=theta2_dev_max,
            N=N,
            steps=steps,
        )
        results.append((fr"$N={N}$", out))

    # Three plots in one figure
    fig, axes = plt.subplots(3, 1, figsize=(5, 6), sharex=True)

    # Top plot: delta theta2
    for label, out in results:
        history_x = out["history_x"]
        axes[0].plot(np.rad2deg(history_x[1, :]), label=label)
    axes[0].set_ylabel(r"$\tilde{\theta}_2$ [deg]")
    axes[0].legend()
    axes[0].grid(True)

    # Middle plot: theta1
    for label, out in results:
        history_x = out["history_x"]
        axes[1].plot(np.rad2deg(history_x[0, :]), label=label)
    axes[1].set_ylabel(r"$\theta_1$ [deg]")
    axes[1].legend()
    axes[1].grid(True)

    # Bottom plot: control input
    for label, out in results:
        history_u = out["history_u"]
        axes[2].step(range(history_u.shape[1]), history_u[0, :], where="post", label=label)
    axes[2].set_xlabel("Time step")
    axes[2].set_ylabel(r"$u$ [Nm]")
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    compare_horizons()