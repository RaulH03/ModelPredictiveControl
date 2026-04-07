import numpy as np
import cvxpy as cp
from scipy.linalg import solve_discrete_are

from MPC_helper_functions import compute_terminal_alpha_double_pendulum
from MPC_double_pendulum_mechanics import Ad, Bd, Ts


def run_mpc_with_measurement_noise(
    q_theta1,
    q_delta_theta2,
    q_theta1_dot,
    q_theta2_dot,
    r_input,
    x0,
    u_max,
    theta1_max,
    theta2_dev_max,
    noise_std,
    N=10,
    steps=100,
    solver=cp.SCS,
    random_seed=1,
):

    rng = np.random.default_rng(random_seed)

    # True initial state
    x_true = np.array(x0, dtype=float).copy()

    n_states = 4
    n_inputs = 1

    # Weighting matrices
    Q = np.diag([q_theta1, q_delta_theta2, q_theta1_dot, q_theta2_dot])
    R = np.array([[r_input]])

    # Terminal ingredients
    P = solve_discrete_are(Ad, Bd, Q, R)
    K = -np.linalg.inv(R + Bd.T @ P @ Bd) @ (Bd.T @ P @ Ad)

    alpha = compute_terminal_alpha_double_pendulum(
        P=P,
        K=K,
        theta1_max=theta1_max,
        theta2_dev_max=theta2_dev_max,
        u_max=u_max,
    )

    # Storage
    history_x_true = np.zeros((n_states, steps))
    history_x_meas = np.zeros((n_states, steps))
    history_noise = np.zeros((n_states, steps))
    history_u = np.zeros((1, steps))
    status_history = []
    simulated_steps = 0

    # Convert noise_std to array if scalar
    if np.isscalar(noise_std):
        noise_std = np.full(n_states, noise_std)
    else:
        noise_std = np.array(noise_std, dtype=float)

    for t in range(steps):
        # Noisy measurement used by the MPC
        noise = rng.normal(loc=0.0, scale=noise_std, size=n_states)
        x_meas = x_true + noise

        x = cp.Variable((n_states, N + 1))
        u = cp.Variable((1, N))

        cost = 0
        constraints = [x[:, 0] == x_meas]

        for k in range(N):
            cost += cp.quad_form(x[:, k], Q) + cp.quad_form(u[:, k], R)

            constraints += [x[:, k + 1] == Ad @ x[:, k] + Bd @ u[:, k]]

            constraints += [cp.abs(u[:, k]) <= u_max]
            constraints += [cp.abs(x[0, k]) <= theta1_max]
            constraints += [cp.abs(x[1, k]) <= theta2_dev_max]

        cost += cp.quad_form(x[:, N], P)
        constraints += [cp.quad_form(x[:, N], P) <= alpha]

        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve(solver=solver, verbose=False)

        status_history.append(prob.status)

        if prob.status not in ["optimal", "optimal_inaccurate"] or u.value is None:
            print(f"Solver failed at step {t}, status: {prob.status}")
            break

        u_applied = float(u.value[0, 0])

        history_x_true[:, t] = x_true
        history_x_meas[:, t] = x_meas
        history_noise[:, t] = noise
        history_u[:, t] = u_applied
        simulated_steps = t + 1

        # True system update
        x_true = Ad @ x_true + Bd.flatten() * u_applied

    # Trim histories
    history_x_true = history_x_true[:, :simulated_steps]
    history_x_meas = history_x_meas[:, :simulated_steps]
    history_noise = history_noise[:, :simulated_steps]
    history_u = history_u[:, :simulated_steps]

    results = {
        "history_x_true": history_x_true,
        "history_x_meas": history_x_meas,
        "history_noise": history_noise,
        "history_u": history_u,
        "Q": Q,
        "R": R,
        "P": P,
        "K": K,
        "alpha": alpha,
        "statuses": status_history,
        "simulated_steps": simulated_steps,
        "Ts": Ts,
        "N": N,
        "x_final_true": x_true,
        "noise_std": noise_std,
    }

    return results


import numpy as np
import matplotlib.pyplot as plt


def compare_noise_levels():
    noise_cases = [
        {
            "label": "No noise",
            "noise_std": [0.0, 0.0, 0.0, 0.0],
        },

        {
            "label": "High noise",
            "noise_std": [np.deg2rad(0.5), np.deg2rad(1.0), np.deg2rad(2.0), np.deg2rad(2.0)],
        },
    ]

    results = []

    for case in noise_cases:
        out = run_mpc_with_measurement_noise(
            q_theta1=50,
            q_delta_theta2=200,
            q_theta1_dot=1,
            q_theta2_dot=1,
            r_input=1,
            x0=[0.0, np.deg2rad(20), 0.0, 0.0],
            u_max=1.5,
            theta1_max=np.deg2rad(25),
            theta2_dev_max=np.deg2rad(25),
            noise_std=case["noise_std"],
            N=10,
            steps=100,
            random_seed=1,
        )
        results.append((case["label"], out))

    fig, axes = plt.subplots(3, 1, figsize=(6, 6), sharex=True)

    # Top: true passive-link deviation
    for label, out in results:
        x_true = out["history_x_true"]
        axes[0].plot(np.rad2deg(x_true[1, :]), label=label)
    axes[0].set_ylabel(r"$\tilde{\theta}_2$ [deg]")
    axes[0].legend()
    axes[0].grid(True)

    # Middle: true first-link angle
    for label, out in results:
        x_true = out["history_x_true"]
        axes[1].plot(np.rad2deg(x_true[0, :]), label=label)
    axes[1].set_ylabel(r"$\theta_1$ [deg]")
    axes[1].grid(True)

    # Bottom: control input
    for label, out in results:
        u = out["history_u"][0, :]
        axes[2].step(range(len(u)), u, where="post", label=label)
    axes[2].set_xlabel("Time step")
    axes[2].set_ylabel(r"$u$ [Nm]")
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    compare_noise_levels()