import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from scipy.linalg import solve_discrete_are
from scipy.integrate import solve_ivp
from MPC_helper_functions import compute_polyhedral_terminal_set_double_pendulum
from MPC_double_pendulum_mechanics import Ad, Bd, non_linear_dynamics, Ts

n_states = 4
n_inputs = 1
N = 20

x_eq = np.array([0.0, np.pi, 0.0, 0.0])
u_eq = 0.0

# Tuning Weights
Q = np.diag([10, 500, 1, 10]) 
R = np.array([[0.1]])

# Terminal Cost P
P = solve_discrete_are(Ad, Bd, Q, R)

# Terminal LQR gain: u = Kx
K = -np.linalg.inv(R + Bd.T @ P @ Bd) @ (Bd.T @ P @ Ad)

# Constraints
u_max = 0.4 # Max torque in Nm
theta1_max = np.deg2rad(25)    # 25 degrees
theta2_dev_max = np.deg2rad(25) # 25 degrees deviation

# Polyhedra terminal set
H_f, h_f = compute_polyhedral_terminal_set_double_pendulum(
    Ad=Ad,
    Bd=Bd,
    K=K,
    theta1_max=theta1_max,
    theta2_dev_max=theta2_dev_max,
    u_max=u_max
)

# Initial State (Deviation from x_e)
x_curr = np.array([0.0, np.deg2rad(15), 0.0, 0.0])       # Starting with a 5-degree offset on the upper link

# Simulation containers
steps = 150
history_x = np.zeros((n_states, steps))
history_u = np.zeros((n_inputs, steps))

# MPC Loop
for t in range(steps):
    x = cp.Variable((n_states, N + 1))
    u = cp.Variable((n_inputs, N))
    
    cost = 0
    constraints = [x[:, 0] == x_curr]
    
    for k in range(N):

        # Stage cost
        cost += cp.quad_form(x[:, k], Q) + cp.quad_form(u[:, k], R)

        # Dynamics
        constraints += [x[:, k+1] == Ad @ x[:, k] + Bd @ u[:, k]]

        # Input & State constraints
        constraints += [cp.abs(u[:, k]) <= u_max]
        constraints += [cp.abs(x[0, k]) <= theta1_max]                  # might change to soft constraint?
        constraints += [cp.abs(x[1, k]) <= theta2_dev_max]              # might change to soft constarint?
        
    # Terminal Cost
    cost += cp.quad_form(x[:, N], P)

    # Terminal set constraint
    constraints += [H_f @ x[:, N] <= h_f]
    
    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve(solver=cp.OSQP)

    if prob.status not in ["optimal", "optimal_inaccurate"]:
        print(f"Solver failed at step {t}, status: {prob.status}")
        break
    
    if u.value is None:
        print(f"Solver failed at step {t}")
        break
        
    u_applied = u.value[0, 0]
    history_u[:, t] = u_applied
    history_x[:, t] = x_curr
    
    # Update actual system (using the model)

    
    # x_curr = Ad @ x_curr + Bd @ [u_applied]

    x_true = x_curr + x_eq
    u_true = u_applied + u_eq
    
    # Update using true non-linear dynamics
    def pendulum_ode(t_sim, x_state):
        # Pass the true physical states and the true physical torque
        dx = non_linear_dynamics(*x_state, u_true)
        return np.array(dx).flatten()

    # Integrate forward by one sampling period (Ts) using true states
    sol = solve_ivp(pendulum_ode, [0, Ts], x_true, method='RK45')

    x_curr = sol.y[:, -1] - x_eq




## applying the input to the non linear system

# Define the equilibrium point exactly as it was in your SymPy script


# # ==========================================
# # 2. Apply the SAME Control Sequence to the NON-LINEAR Plant
# # ==========================================
hist_x_nl = np.zeros((n_states, steps))
x_curr_nl = history_x[:, 0] # This is the starting *deviation*

for t in range(steps):
    # Record current deviation state
    hist_x_nl[:, t] = x_curr_nl
    
    # Extract the exact input deviation the linear system used at this time step
    u_applied = history_u[0, t] 
    
    # Calculate the TRUE absolute state and input to feed the physical model
    x_true = x_curr_nl + x_eq
    u_true = u_applied + u_eq
    
    # Update using true non-linear dynamics
    def pendulum_ode(t_sim, x_state):
        # Pass the true physical states and the true physical torque
        dx = non_linear_dynamics(*x_state, u_true)
        return np.array(dx).flatten()

    # Integrate forward by one sampling period (Ts) using true states
    sol = solve_ivp(pendulum_ode, [0, Ts], x_true, method='RK45')
    
    # The solver returns the true absolute state. 
    # Subtract the equilibrium to convert it back to a deviation state for the next loop.
    x_curr_nl = sol.y[:, -1] - x_eq



# Plotting results
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(np.rad2deg(history_x[0, :]), label='Theta 1 (Lower)')
plt.plot(np.rad2deg(history_x[1, :]), label='Theta 2 Error (Upper)')
plt.plot(np.rad2deg(hist_x_nl[0, :]), label='Theta 1 NL (Lower)')
plt.plot(np.rad2deg(hist_x_nl[1, :]), label='Theta 2 NL Error (Upper)')
plt.legend()
plt.ylabel('Degrees')
plt.title('Pendulum Deviation Recovery')

plt.subplot(2, 1, 2)
plt.step(range(steps), history_u[0, :], label='Control Torque')
plt.ylabel('Torque [Nm]')
plt.legend()
plt.show()

