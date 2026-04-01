import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from scipy.linalg import solve_discrete_are

from MPC_helper_functions import compute_terminal_alpha_double_pendulum
from MPC_double_pendulum_mechanics import Ad, Bd, Ts


def run_mpc_case(q_theta1, q_delta_theta2, q_dot_theta1, q_dot_theta2, R):
    n_states = 4
    n_inputs = 1
    N = 20

    # Tuning Weights
    Q = np.diag([q_theta1, q_delta_theta2, q_dot_theta2, q_dot_theta2])
    R = np.array([[R]])

    # Terminal Cost P
    P = solve_discrete_are(Ad, Bd, Q, R)

    # Terminal LQR gain: u = Kx
    K = -np.linalg.inv(R + Bd.T @ P @ Bd) @ (Bd.T @ P @ Ad)

    # Constraints
    u_max = 0.5
    theta1_max = np.deg2rad(25)
    theta2_dev_max = np.deg2rad(25)

    # Terminal set size
    alpha = compute_terminal_alpha_double_pendulum(P, K, theta1_max, theta2_dev_max, u_max)

    # Initial State
    x_curr = np.array([0.0, np.deg2rad(10), 0.0, 0.0])

    # Simulation containers
    steps = 100
    history_x = np.zeros((n_states, steps))
    history_u = np.zeros((n_inputs, steps))

    for t in range(steps):
        x = cp.Variable((n_states, N + 1))
        u = cp.Variable((n_inputs, N))

        cost = 0
        constraints = [x[:, 0] == x_curr]

        for k in range(N):
            # Stage cost
            cost += cp.quad_form(x[:, k], Q) + cp.quad_form(u[:, k], R)

            # Dynamics
            constraints += [x[:, k + 1] == Ad @ x[:, k] + Bd @ u[:, k]]

            # Input & state constraints
            constraints += [cp.abs(u[:, k]) <= u_max]
            constraints += [cp.abs(x[0, k]) <= theta1_max]
            constraints += [cp.abs(x[1, k]) <= theta2_dev_max]

        # Terminal cost and terminal set
        cost += cp.quad_form(x[:, N], P)
        constraints += [cp.quad_form(x[:, N], P) <= alpha]

        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve(solver=cp.SCS, verbose=False)

        if prob.status not in ["optimal", "optimal_inaccurate"] or u.value is None:
            print(f"Solver failed for q_delta_theta2={q_delta_theta2} at step {t}, status: {prob.status}")
            history_x = history_x[:, :t]
            history_u = history_u[:, :t]
            break

        u_applied = u.value[0, 0]
        history_u[:, t] = u_applied
        history_x[:, t] = x_curr

        # Update actual system
        x_curr = Ad @ x_curr + Bd.flatten() * u_applied

    return history_x, history_u


# Run three cases
cases = [10, 100, 200, 500]
q1 = 1
q3 = 1
q4 = 1
R = 1
results = {}

for q2 in cases:
    history_x, history_u = run_mpc_case(q1, q2, q3, q4, R)
    results[q2] = {"x": history_x, "u": history_u}

# One figure with both graphs
fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

# Top plot: delta theta 2
for q2 in cases:
    theta2_hist = results[q2]["x"][1, :]
    axes[0].plot(np.rad2deg(theta2_hist), label=fr"$q_{{\delta \theta_2}}={q2}$")
axes[0].set_ylabel(r"$\delta \theta_2$ [deg]")
axes[0].set_title(r"Effect of $q_{\delta \theta_2}$ on the closed-loop response")
axes[0].legend()
axes[0].grid(True)

# Bottom plot: theta 1
for q2 in cases:
    theta1_hist = results[q2]["x"][0, :]
    axes[1].plot(np.rad2deg(theta1_hist), label=fr"$q_{{\delta \theta_2}}={q2}$")
axes[1].set_xlabel("Time step")
axes[1].set_ylabel(r"$\theta_1$ [deg]")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()

