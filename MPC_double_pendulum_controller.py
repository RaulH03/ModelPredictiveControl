import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from scipy.linalg import solve_discrete_are

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
Q = np.diag([10, 500, 1, 10]) 
R = np.array([[0.1]])

# Terminal Cost P
P = solve_discrete_are(Ad, Bd, Q, R)

# Initial State (Deviation from x_e)
x_curr = np.array([0.0, np.deg2rad(5), 0.0, 0.0])       # Starting with a 5-degree tilt on the upper link

# Constraints
u_max = 5.0 # Max torque in Nm
theta1_max = np.deg2rad(25)    # 25 degrees
theta2_dev_max = np.deg2rad(25) # 25 degrees deviation

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
        cost += cp.quad_form(x[:, k], Q) + cp.quad_form(u[:, k], R)
        constraints += [x[:, k+1] == Ad @ x[:, k] + Bd @ u[:, k]]
        constraints += [cp.abs(u[:, k]) <= u_max]
        constraints += [cp.abs(x[0, k]) <= theta1_max]                  # might change to soft constraint?
        constraints += [cp.abs(x[1, k]) <= theta2_dev_max]              # might change to soft constarint?
        
    cost += cp.quad_form(x[:, N], P) # Terminal Cost
    
    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve(solver=cp.OSQP)
    
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

