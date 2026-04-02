import numpy as np
import matplotlib.pyplot as plt

from run_Linear_MPC import run_mpc   # change this import to your actual filename


def compare_parameter_cases():
    # Fixed settings
    x0 = [0.0, np.deg2rad(20), 0.0, 0.0]
    u_max = 2
    theta1_max = np.deg2rad(25)
    theta2_dev_max = np.deg2rad(25)
    N = 20
    steps = 100

    # Cases to compare
    cases = [
        {
            "label": r"$q_3=1$",
            "q_theta1": 1,
            "q_delta_theta2": 200,
            "q_theta1_dot": 1,
            "q_theta2_dot": 1,
            "r_input": 0.1,
        },
        {
            "label": r"$q_3=10$",
            "q_theta1": 1,
            "q_delta_theta2": 200,
            "q_theta1_dot": 1,
            "q_theta2_dot": 10,
            "r_input": 0.1,
        },
        {
            "label": r"$q_3=50$",
            "q_theta1": 1,
            "q_delta_theta2": 200,
            "q_theta1_dot": 1,
            "q_theta2_dot": 50,
            "r_input": 0.1,
        },
        {
            "label": r"$q_3=100$",
            "q_theta1": 1,
            "q_delta_theta2": 200,
            "q_theta1_dot": 1,
            "q_theta2_dot": 100,
            "r_input": 0.1,
        },
    ]

    results = []

    for case in cases:
        out = run_mpc(
            q_theta1=case["q_theta1"],
            q_delta_theta2=case["q_delta_theta2"],
            q_theta1_dot=case["q_theta1_dot"],
            q_theta2_dot=case["q_theta2_dot"],
            r_input=case["r_input"],
            x0=x0,
            u_max=u_max,
            theta1_max=theta1_max,
            theta2_dev_max=theta2_dev_max,
            N=N,
            steps=steps,
        )
        results.append((case["label"], out))

    # Two plots in one figure
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    # Top plot: delta theta2
    for label, out in results:
        history_x = out["history_x"]
        axes[0].plot(np.rad2deg(history_x[1, :]), label=label)
    axes[0].set_ylabel(r"$\delta \theta_2$ [deg]")
    axes[0].set_title(r"Comparison of $\delta \theta_2$ and $\theta_1$ trajectories")
    axes[0].legend()
    axes[0].grid(True)

    # Bottom plot: theta1
    for label, out in results:
        history_x = out["history_x"]
        axes[1].plot(np.rad2deg(history_x[0, :]), label=label)
    axes[1].set_xlabel("Time step")
    axes[1].set_ylabel(r"$\theta_1$ [deg]")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    compare_parameter_cases()