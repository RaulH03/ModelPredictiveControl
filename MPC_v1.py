import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from scipy.linalg import solve_discrete_are

# System Parameters
V0 = 5.0 
L = 2.5
T = 0.1
N = 10

# State-Space
A = np.array([[1, T*V0, 0], [0, 1, T*V0/L], [0, 0, 1]])
B = np.array([[0], [0], [T]])

# Simulate path
total_distance = 20.0 
steps = int(total_distance / (V0 * T)) 
s = np.linspace(0, total_distance, 200)
road_y = 2 * np.sin(0.3 * s)

# MPC Loop
x_curr = np.array([0.3, 0.0, 0.0]) # Start 30cm off-center
pos_x, pos_y = np.zeros(steps), np.zeros(steps)
x_history = np.zeros((3, steps))

# Weights and Constraints
Q = np.diag([50.0, 1.0, 0.1])
R = np.array([[0.1]])
P = solve_discrete_are(A, B, Q, R)
W_lane = 2.0
delta_max = np.deg2rad(25)
u_max = 0.5

# Terminal set
y_e_alpha = 0.1
psi_e_alpha = 0.4
delta_alpha = 0.1

for t in range(steps):
    x = cp.Variable((3, N + 1))
    u = cp.Variable((1, N))
    
    cost = 0
    constraints = [x[:, 0] == x_curr]
    
    for i in range(N):
        cost += cp.quad_form(x[:, i], Q) + cp.quad_form(u[:, i], R)
        constraints += [x[:, i+1] == A @ x[:, i] + B @ u[:, i]]
        constraints += [cp.abs(x[0, i]) <= W_lane/2]
        constraints += [cp.abs(x[2, i]) <= delta_max]
        constraints += [cp.abs(u[0, i]) <= u_max]

    cost += cp.quad_form(x[:, N], P) # Terminal cost

    constraints += [cp.abs(x[0, N]) <= y_e_alpha]
    constraints += [cp.abs(x[1, N]) <= psi_e_alpha]
    constraints += [cp.abs(x[2, N]) <= delta_alpha]
    
    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve(solver=cp.OSQP, verbose=False)

    # Check if a solution was found
    if u.value is None:
        print(f"Infeasible at step {t}.")
        break
    
    u_opt = u.value[0, 0]
    
    # Global coordinates
    current_s = t * T * V0
    idx = np.argmin(np.abs(s - current_s))
    pos_x[t] = current_s
    pos_y[t] = road_y[idx] + x_curr[0]
    
    # Update actual system
    x_curr = A @ x_curr + B @ [u_opt]

# Visualization
plt.figure(figsize=(10, 4))
plt.plot(s, road_y, 'r--', label='Road Centerline')
plt.plot(pos_x, pos_y, 'b-o', markersize=3, label='MPC Path')
plt.fill_between(s, road_y - W_lane/2, road_y + W_lane/2, color='gray', alpha=0.2)
plt.title(f"MPC Path Following (20m simulation, N={N})")
plt.xlabel("Distance [m]")
plt.ylabel("Lateral [m]")
plt.legend()
plt.axis('equal')
plt.show()