import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from scipy.linalg import solve_discrete_are
from scipy.interpolate import interp1d
from scipy.optimize import linprog

# ==========================================
# 1. Maximum Admissible Invariant Set (MAS)
# ==========================================
def compute_invariant_set(A, B, P, R, W_lane, delta_max, u_max):
    """
    Computes the MAS for the terminal LQR controller.
    Returns matrices H and h such that H*x <= h defines the invariant set.
    """
    # 1. Compute the LQR gain K: u = -Kx
    K = np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)
    
    # 2. Closed-loop dynamics under LQR
    A_cl = A - B @ K
    
    # 3. Define the base physical constraints in the form F*x <= g
    F = np.array([
        [ 1,  0,  0],  # Lateral error <= W_lane/2
        [-1,  0,  0],  # -Lateral error <= W_lane/2
        [ 0,  0,  1],  # Steering <= delta_max
        [ 0,  0, -1],  # -Steering <= delta_max
        K[0],          # Steering rate (u = -Kx) <= u_max
        -K[0]          # -Steering rate <= u_max
    ])
    g = np.array([W_lane/2, W_lane/2, delta_max, delta_max, u_max, u_max])
    
    # 4. Iterate to find the invariant set
    H, h = F.copy(), g.copy()
    max_iter = 100
    
    for i in range(1, max_iter):
        F_next = F @ np.linalg.matrix_power(A_cl, i)
        all_redundant = True
        
        for j in range(F_next.shape[0]):
            f_row = F_next[j, :]
            g_val = g[j]
            # Maximize f_row * x subject to H * x <= h
            res = linprog(-f_row, A_ub=H, b_ub=h, bounds=(None, None), method='highs')
            
            # If maximum possible value exceeds our bound, the constraint is strictly necessary
            if res.success and (-res.fun > g_val + 1e-6):
                H = np.vstack((H, f_row))
                h = np.append(h, g_val)
                all_redundant = False
                
        if all_redundant:
            print(f"Terminal Invariant Set found: {H.shape[0]} inequalities.")
            break
            
    return H, h

# ==========================================
# 2. System Parameters & State-Space
# ==========================================
V0 = 5.0 
L = 2.5
T = 0.1
N = 10

# States: [lateral error (e_y), heading error (e_psi), steering angle (delta)]
# Input:  [steering rate (u)]
A = np.array([[1, T*V0, 0], [0, 1, T*V0/L], [0, 0, 1]])
B = np.array([[0], [0], [T]])

# ==========================================
# 3. Path Generation & Kinematics
# ==========================================
sim_distance = 30.0 
steps = int(sim_distance / (V0 * T)) 

# # Generate a longer track (35m) so the car doesn't run out of road visually
# x_raw = np.linspace(0, 35.0, 500)
# road_y = 2 * np.sin(0.3 * x_raw)

# Generate a track long enough to see the whole maneuver
x_raw = np.linspace(0, 40.0, 500)

# ==========================================
# NEW: Smooth Lane Change (Sigmoid / Tanh)
# ==========================================
target_lane_y = 3.0    # How far lateral to shift (e.g., a standard 3m lane)
shift_center_x = 15.0  # The X-coordinate where the lane change happens
sharpness = 1.5        # How aggressive the driver is (lower = smoother)

# Tanh creates a smooth step from -1 to 1. We scale and shift it.
road_y = (target_lane_y / 2.0) * (1 + np.tanh(sharpness * (x_raw - shift_center_x)))

# Calculate derivatives to find heading (psi) and curvature (kappa)
dx = np.gradient(x_raw)
dy = np.gradient(road_y)
psi_ref = np.arctan2(dy, dx)

ddx = np.gradient(dx)
ddy = np.gradient(dy)
kappa_raw = (dx * ddy - dy * ddx) / ((dx**2 + dy**2)**(3/2)) # Curvature

# Calculate true arc length (s)
ds = np.sqrt(dx**2 + dy**2)
s_ref = np.cumsum(ds)
s_ref -= s_ref[0]

# Continuous interpolators for the MPC to query exact path data at any 's'
interp_x = interp1d(s_ref, x_raw, kind='cubic', fill_value="extrapolate")
interp_y = interp1d(s_ref, road_y, kind='cubic', fill_value="extrapolate")
interp_psi = interp1d(s_ref, psi_ref, kind='cubic', fill_value="extrapolate")
interp_kappa = interp1d(s_ref, kappa_raw, kind='cubic', fill_value="extrapolate")

def frenet_to_cartesian(s, e_y):
    """Maps (s, e_y) to global (X, Y) using the path's normal vector."""
    x_r = interp_x(s)
    y_r = interp_y(s)
    psi_r = interp_psi(s)
    X = x_r - e_y * np.sin(psi_r)
    Y = y_r + e_y * np.cos(psi_r)
    return X, Y

# ==========================================
# 4. MPC Setup & Tuning
# ==========================================
x_curr = np.array([0.3, 0.0, 0.0]) # Start 30cm off-center
pos_x, pos_y = np.zeros(steps), np.zeros(steps)

Q = np.diag([50.0, 1.0, 0.1])
R = np.array([[0.1]])
P = solve_discrete_are(A, B, Q, R) # Terminal LQR cost

W_lane = 5.0
delta_max = np.deg2rad(45)
u_max = 10

# Compute the rigorous terminal constraint bounds offline
print("Calculating Maximum Admissible Invariant Set (MAS)...")
H_mas, h_mas = compute_invariant_set(A, B, P, R, W_lane, delta_max, u_max)

# ==========================================
# 5. Simulation Loop
# ==========================================
print("Starting MPC Simulation...")
for t in range(steps):
    x = cp.Variable((3, N + 1))
    u = cp.Variable((1, N))
    
    cost = 0
    constraints = [x[:, 0] == x_curr]
    
    # Estimate current 's' assuming constant velocity (open-loop approximation)
    current_s = t * T * V0 
    
    for i in range(N):
        # 1. Look ahead to find the curvature at future step 'i'
        future_s = current_s + (i * T * V0)
        k_future = interp_kappa(future_s)
        
        # 2. The curvature pushes the heading error (e_psi) off the path
        dist_future = np.array([0, -T * V0 * k_future, 0])
        
        # 3. Accumulate running cost
        cost += cp.quad_form(x[:, i], Q) + cp.quad_form(u[:, i], R)
        
        # 4. Predict next state (Linear Dynamics + Curvature Disturbance)
        constraints += [x[:, i+1] == A @ x[:, i] + B @ u[:, i] + dist_future]
        
        # 5. Apply physical limits
        constraints += [cp.abs(x[0, i]) <= W_lane/2]
        constraints += [cp.abs(x[2, i]) <= delta_max]
        constraints += [cp.abs(u[0, i]) <= u_max]

    # Add terminal cost (LQR matrix P)
    cost += cp.quad_form(x[:, N], P)

    # Add terminal constraints (The calculated MAS)
    constraints += [H_mas @ x[:, N] <= h_mas]
    
    # Solve the optimization problem
    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve(solver=cp.OSQP, verbose=False)

    if u.value is None:
        print(f"Infeasible at step {t}. The car crashed or cannot satisfy constraints!")
        break
    
    u_opt = u.value[0, 0]
    
    # Save true global coordinates for plotting
    global_x, global_y = frenet_to_cartesian(current_s, x_curr[0])
    pos_x[t] = global_x
    pos_y[t] = global_y
    
    # Update the "Real World" car state with the actual current curvature
    k_current = interp_kappa(current_s)
    dist_current = np.array([0, -T * V0 * k_current, 0])
    x_curr = A @ x_curr + B @ [u_opt] + dist_current

print("Simulation Complete.")

# ==========================================
# 6. Visualization
# ==========================================
plt.figure(figsize=(10, 4))
plt.plot(x_raw, road_y, 'r--', label='Road Centerline')

# Only plot valid steps if the solver failed partway through
valid_steps = t if u.value is None else steps
plt.plot(pos_x[:valid_steps], pos_y[:valid_steps], 'b-o', markersize=3, label='MPC Path')

# Geometrically accurate lane boundaries using normal vectors
bound_x_upper = x_raw - (W_lane/2)*np.sin(psi_ref)
bound_y_upper = road_y + (W_lane/2)*np.cos(psi_ref)
bound_x_lower = x_raw + (W_lane/2)*np.sin(psi_ref)
bound_y_lower = road_y - (W_lane/2)*np.cos(psi_ref)

plt.plot(bound_x_upper, bound_y_upper, color='gray', alpha=0.5, linestyle=':', label='Lane Bound')
plt.plot(bound_x_lower, bound_y_lower, color='gray', alpha=0.5, linestyle=':')

plt.title(f"MPC Path Following with Invariant Set (N={N})")
plt.xlabel("Global X [m]")
plt.ylabel("Global Y [m]")
plt.xlim(0, 25) # Focus on the area the car actually reaches
plt.legend()
plt.axis('equal')
plt.grid(True)
plt.show()