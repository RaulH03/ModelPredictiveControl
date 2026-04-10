import numpy as np
from scipy.linalg import solve_discrete_are
import matplotlib.pyplot as plt

from MPC_double_pendulum_mechanics import Ad, Bd, Ts
from MPC_helper_functions import f_nl
from run_Nonlinear_MPC import run_mpc


def run_lqr(
    q_theta1,
    q_delta_theta2,
    q_theta1_dot,
    q_theta2_dot,
    r_input,
    x0,
    steps=100,
    clip_input=False,
    u_max=None,
):

    x_curr = np.array(x0, dtype=float).copy()

    x_eq = np.array([0.0, np.pi, 0.0, 0.0])
    u_eq = 0.0

    n_states = 4
    n_inputs = 1

    # Weighting matrices
    Q = np.diag([q_theta1, q_delta_theta2, q_theta1_dot, q_theta2_dot])
    R = np.array([[r_input]])

    # LQR ingredients
    P = solve_discrete_are(Ad, Bd, Q, R)
    K = -np.linalg.inv(R + Bd.T @ P @ Bd) @ (Bd.T @ P @ Ad)

    # Storage
    history_x = np.zeros((n_states, steps))
    history_u = np.zeros((n_inputs, steps))

    for t in range(steps):
        u_applied = float((K @ x_curr.reshape(-1, 1))[0, 0])

        if clip_input:
            if u_max is None:
                raise ValueError("u_max must be provided when clip_input=True.")
            u_applied = float(np.clip(u_applied, -u_max, u_max))

        history_x[:, t] = x_curr
        history_u[:, t] = u_applied

        # Update system
        x_curr = f_nl(x_curr, u_applied, x_eq, u_eq)

    results = {
        "history_x": history_x,
        "history_u": history_u,
        "Q": Q,
        "R": R,
        "P": P,
        "K": K,
        "simulated_steps": steps,
        "Ts": Ts,
        "x_final": x_curr,
        "clip_input": clip_input,
        "u_max": u_max,
    }

    return results

if __name__ == "__main__":
    # Same parameters for both controllers
    q_theta1 = 1
    q_delta_theta2 = 200
    q_theta1_dot = 10
    q_theta2_dot = 1
    r_input = 1
    x0 = [0.0, np.deg2rad(30), 0.0, 0.0]
    u_max = 1.5
    steps = 100

    # Run LQR
    lqr_results = run_lqr(
        q_theta1=q_theta1,
        q_delta_theta2=q_delta_theta2,
        q_theta1_dot=q_theta1_dot,
        q_theta2_dot=q_theta2_dot,
        r_input=r_input,
        x0=x0,
        steps=steps,
        clip_input=False,
        u_max=u_max,
    )

    # Run MPC
    mpc_results = run_mpc(
        q_theta1=q_theta1,
        q_delta_theta2=q_delta_theta2,
        q_theta1_dot=q_theta1_dot,
        q_theta2_dot=q_theta2_dot,
        r_input=r_input,
        x0=x0,
        u_max=u_max,
        theta1_max=np.deg2rad(90),
        theta2_dev_max=np.deg2rad(90),
        N=30,
        steps=steps,
    )

    # Saturated / clipped LQR (for physically realizable state comparison)
    sat_lqr_results = run_lqr(
        q_theta1=q_theta1,
        q_delta_theta2=q_delta_theta2,
        q_theta1_dot=q_theta1_dot,
        q_theta2_dot=q_theta2_dot,
        r_input=r_input,
        x0=x0,
        steps=steps,
        clip_input=True,
        u_max=u_max,
    )

    # Extract data
    mpc_x = mpc_results["history_x"]
    mpc_u = mpc_results["history_u"][0, :]

    lqr_x = lqr_results["history_x"]
    lqr_u = lqr_results["history_u"][0, :]

    sat_lqr_x = sat_lqr_results["history_x"]
    sat_lqr_u = sat_lqr_results["history_u"][0, :]

    # Plot control inputs together
    plt.figure(figsize=(5, 4))
    plt.step(range(len(mpc_u)), mpc_u, where="post", label="MPC")
    plt.step(range(len(lqr_u)), lqr_u, where="post", label="LQR")
    plt.axhline(u_max, color="red", linestyle="--", label=r"Constraint $\pm u_{\max}$")
    plt.axhline(-u_max, color="red", linestyle="--")
    plt.xlabel("Time step")
    plt.ylabel("Torque [Nm]")
    plt.title("Control input: MPC vs LQR")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    fig, axes = plt.subplots(2, 1, figsize=(5, 4.5), sharex=True)

    # Top plot: passive link deviation
    axes[0].plot(np.rad2deg(mpc_x[1, :]), label="MPC")
    axes[0].plot(np.rad2deg(sat_lqr_x[1, :]), label="Saturated LQR")
    axes[0].set_ylabel(r"$\tilde{\theta}_2$ [deg]")
    axes[0].legend()
    axes[0].grid(True)

    # Bottom plot: first-link angle
    axes[1].plot(np.rad2deg(mpc_x[0, :]), label="MPC")
    axes[1].plot(np.rad2deg(sat_lqr_x[0, :]), label="Saturated LQR")
    axes[1].set_xlabel("Time step")
    axes[1].set_ylabel(r"$\theta_1$ [deg]")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()