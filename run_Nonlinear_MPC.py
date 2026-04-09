import numpy as np
import cvxpy as cp
from scipy.linalg import solve_discrete_are
import matplotlib.pyplot as plt

from MPC_helper_functions import compute_terminal_alpha_double_pendulum, find_nonlinear_terminal_set, f_nl
from MPC_double_pendulum_mechanics import Ad, Bd, Ts


def run_mpc(
    q_theta1,
    q_delta_theta2,
    q_theta1_dot,
    q_theta2_dot,
    r_input,
    x0,
    u_max,
    theta1_max,
    theta2_dev_max,
    N=20,
    steps=100,
    solver=cp.SCS,
):
    #Equilibrium state
    x_eq = np.array([0.0, np.pi, 0.0, 0.0])
    u_eq = 0.0

    # Convert initial condition to numpy array
    x_curr = np.array(x0, dtype=float).copy()

    # Dimensions
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

    iterations = 100
    num_samples = 10000
    L_frac = 0.9 #fraction of the decrease that is linear
    nl_alpha = find_nonlinear_terminal_set(
        P=P, 
        K=K, 
        Q=Q, 
        R=R, 
        x_eq=x_eq, 
        u_eq=u_eq, 
        alpha_initial=alpha, 
        L_frac=L_frac, 
        max_iterations=iterations, 
        num_samples=num_samples
    )


    # Storage
    history_x = np.zeros((n_states, steps))
    history_u = np.zeros((n_inputs, steps))
    status_history = []
    simulated_steps = 0

    # MPC loop
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

            # Input and state constraints
            constraints += [cp.abs(u[:, k]) <= u_max]
            constraints += [cp.abs(x[0, k]) <= theta1_max]
            constraints += [cp.abs(x[1, k]) <= theta2_dev_max]

        # Terminal cost
        cost += cp.quad_form(x[:, N], P)

        # Terminal set constraint
        constraints += [cp.quad_form(x[:, N], P) <= nl_alpha]

        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve(solver=solver, verbose=False)

        status_history.append(prob.status)

        if prob.status not in ["optimal", "optimal_inaccurate"] or u.value is None:
            print(f"Solver failed at step {t}, status: {prob.status}")
            break

        u_applied = float(u.value[0, 0])

        history_x[:, t] = x_curr
        history_u[:, t] = u_applied
        simulated_steps = t + 1

        # Update actual system using the same discrete model
        x_curr = f_nl(x_curr, u_applied, x_eq, u_eq)

    # Trim to actual simulated length
    history_x = history_x[:, :simulated_steps]
    history_u = history_u[:, :simulated_steps]

    results = {
        "history_x": history_x,
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
        "x_final": x_curr,
    }

    return results

if __name__ == "__main__":
    results = run_mpc(
        q_theta1=10,
        q_delta_theta2=200,
        q_theta1_dot=10,
        q_theta2_dot=1,
        r_input=0.1,
        x0=[0.0, np.deg2rad(5), 0.0, 0.0],
        u_max=0.5,
        theta1_max=np.deg2rad(25),
        theta2_dev_max=np.deg2rad(25),
        N=20,
        steps=100,
    )

    history_x = results["history_x"]
    history_u = results["history_u"]
    simulated_steps = results["simulated_steps"]

    # Plotting results
    plt.figure(figsize=(10, 6))

    plt.subplot(2, 1, 1)
    plt.plot(np.rad2deg(history_x[0, :]), label='Theta 1 (Lower)')
    plt.plot(np.rad2deg(history_x[1, :]), label='Theta 2 Error (Upper)')
    plt.legend()
    plt.ylabel('Degrees')
    plt.title('Pendulum Deviation Recovery')
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.step(range(simulated_steps), history_u[0, :], label='Control Torque')
    plt.ylabel('Torque [Nm]')
    plt.xlabel('Time step')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()