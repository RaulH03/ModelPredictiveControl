import numpy as np
import matplotlib.pyplot as plt

from run_Linear_MPC import run_mpc


if __name__ == "__main__":
    u_max = 1.5

    # Selected tuning
    selected = run_mpc(
        q_theta1=50,
        q_delta_theta2=200,
        q_theta1_dot=1,
        q_theta2_dot=1,
        r_input=1,
        x0=[0.0, np.deg2rad(20), 0.0, 0.0],
        u_max=u_max,
        theta1_max=np.deg2rad(25),
        theta2_dev_max=np.deg2rad(25),
        N=20,
        steps=100,
    )

    # More aggressive tuning
    aggressive = run_mpc(
        q_theta1=50,
        q_delta_theta2=500,
        q_theta1_dot=1,
        q_theta2_dot=1,
        r_input=1,
        x0=[0.0, np.deg2rad(20), 0.0, 0.0],
        u_max=u_max,
        theta1_max=np.deg2rad(25),
        theta2_dev_max=np.deg2rad(25),
        N=20,
        steps=100,
    )

    u_selected = selected["history_u"][0, :]
    u_aggressive = aggressive["history_u"][0, :]

    plt.figure(figsize=(5, 4))
    plt.step(range(len(u_selected)), u_selected, where="post", label=r"Selected tuning")
    plt.step(range(len(u_aggressive)), u_aggressive, where="post", label=r"More aggressive tuning")
    plt.axhline(u_max, color="red", linestyle="--", label=r"Constraint $\pm u_{\max}$")
    plt.axhline(-u_max, color="red", linestyle="--")
    plt.xlabel("Time step")
    plt.ylabel("Torque [Nm]")
    plt.title("Control input for different tuning choices")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()