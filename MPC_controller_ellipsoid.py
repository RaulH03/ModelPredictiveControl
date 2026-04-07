import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from scipy.linalg import solve_discrete_are
from MPC_helper_functions import compute_terminal_alpha_double_pendulum

# Model Setup
Ad = np.array([[ 0.99100,  0.02730,  0.01910,  0.00020],
               [ 0.00910,  1.04550,  0.00010,  0.01950],
               [-0.92990,  2.85800,  0.99100,  0.02730],
               [ 0.95270,  4.78600,  0.00910,  1.04550]])

Bd = np.array([[0.0462],
               [0.0464],
               [4.8265],
               [4.8845]])

n_states = 4
n_inputs = 1
N = 20

# Tuning Weights
Q = np.diag([50, 200, 1, 1]) 
R = np.array([[1]])

# Terminal Cost P
P = solve_discrete_are(Ad, Bd, Q, R)

# Terminal LQR gain: u = Kx
K = -np.linalg.inv(R + Bd.T @ P @ Bd) @ (Bd.T @ P @ Ad)

# Constraints
u_max = 2.0 # Max torque in Nm
theta1_max = np.deg2rad(25)    # 25 degrees
theta2_dev_max = np.deg2rad(20) # 25 degrees deviation

# Conservative ellipsoidal terminal set size
alpha = compute_terminal_alpha_double_pendulum(P, K, theta1_max, theta2_dev_max, u_max)
print(alpha)

# Initial State (Deviation from x_e)
x_curr = np.array([0.0, np.deg2rad(20), 0.0, 0.0])       # Starting with a 5-degree offset on the upper link

# Simulation containers
steps = 100
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
    constraints += [cp.quad_form(x[:, N], P) <= alpha]
    
    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve(solver=cp.SCS)

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
    x_curr = Ad @ x_curr + Bd @ [u_applied]

# Plotting results
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(np.rad2deg(history_x[0, :]), label='Theta 1 (Lower)')
plt.plot(np.rad2deg(history_x[1, :]), label='Theta 2 Error (Upper)')
plt.legend()
plt.ylabel('Degrees')
plt.title('Pendulum Deviation Recovery')

plt.subplot(2, 1, 2)
plt.step(range(steps), history_u[0, :], label='Control Torque')
plt.ylabel('Torque [Nm]')
plt.legend()
plt.show()

