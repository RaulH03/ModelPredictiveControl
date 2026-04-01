import numpy as np
from scipy.optimize import linprog
from scipy.integrate import solve_ivp
from MPC_double_pendulum_mechanics import non_linear_dynamics, Ts

# X_f = {x | x.T P x <= alpha}
def compute_terminal_alpha_double_pendulum(P, K, theta1_max, theta2_dev_max, u_max):

    P_inv = np.linalg.inv(P)

    # Each constraint is written as |c^T x| <= b
    constraint_rows = [
        (np.array([1, 0, 0, 0]), theta1_max),         # |x[0]| <= theta1_max
        (np.array([0, 1, 0, 0]), theta2_dev_max),     # |x[1]| <= theta2_dev_max
        (K.flatten(), u_max)                          # |u| = |Kx| <= u_max
    ]

    alpha_candidates = []
    for c, b in constraint_rows:
        alpha_i = (b ** 2) / (c @ P_inv @ c)
        alpha_candidates.append(alpha_i)

    return min(alpha_candidates)


def compute_polyhedral_terminal_set_double_pendulum(
    Ad,
    Bd,
    K,
    theta1_max,
    theta2_dev_max,
    u_max,
    max_iter=100,
    tol=1e-8,
):

    # Closed-loop matrix
    A_cl = Ad + Bd @ K

    # Base admissible set: Fx <= g
    F = np.array([
        [ 1.0,  0.0,  0.0,  0.0],   #  x1 <= theta1_max
        [-1.0,  0.0,  0.0,  0.0],   # -x1 <= theta1_max
        [ 0.0,  1.0,  0.0,  0.0],   #  x2 <= theta2_dev_max
        [ 0.0, -1.0,  0.0,  0.0],   # -x2 <= theta2_dev_max
        [ 0.0,  0.0,  1.0,  0.0],   #  x3 <= theta1_dot_max
        [ 0.0,  0.0, -1.0,  0.0],   # -x3 <= theta1_dot_max
        [ 0.0,  0.0,  0.0,  1.0],   #  x4 <= theta2_dot_max
        [ 0.0,  0.0,  0.0, -1.0],   # -x4 <= theta2_dot_max
        K.flatten(),                #  Kx <= u_max
        -K.flatten()                # -Kx <= u_max
    ])

    theta1_dot_max = 2.0
    theta2_dot_max = 2.0
    
    g = np.array([
        theta1_max,
        theta1_max,
        theta2_dev_max,
        theta2_dev_max,
        theta1_dot_max,
        theta1_dot_max,
        theta2_dot_max,
        theta2_dot_max,
        u_max,
        u_max
    ])

    # Start with the one-step admissible set
    H = F.copy()
    h = g.copy()

    for i in range(1, max_iter + 1):
        H_new = F @ np.linalg.matrix_power(A_cl, i)
        added_any = False

        for row_idx in range(H_new.shape[0]):
            c = H_new[row_idx, :]
            b = g[row_idx]

            # Check if c x <= b is redundant over current set Hx <= h
            # Maximize c^T x subject to Hx <= h
            res = linprog(
                -c,                # maximize c^T x == minimize -c^T x
                A_ub=H,
                b_ub=h,
                bounds=[(None, None)] * H.shape[1],
                method="highs"
            )

            if not res.success:
                raise RuntimeError(
                    f"LP failed while checking redundancy at iteration {i}, row {row_idx}: {res.message}"
                )

            max_val = -res.fun

            # If current set allows c^T x > b, then this inequality is needed
            if max_val > b + tol:
                H = np.vstack([H, c])
                h = np.append(h, b)
                added_any = True

        if not added_any:
            print(f"Polyhedral terminal set found after {i} iterations with {H.shape[0]} inequalities.")
            break
    else:
        print(f"Warning: maximum iterations ({max_iter}) reached. Set may not be fully converged.")

    return H, h


def f_nl(x_dev, u_dev, x_eq, u_eq):
    """
    Computes x_{k+1} = f_nl(x_k, u_k) by integrating the continuous 
    nonlinear dynamics over one sample time Ts.
    """
    # Convert deviation variables to true physical variables
    x_true = x_dev + x_eq
    u_true = (u_dev + u_eq).item() # ensure scalar for the ODE
    
    # Define the ODE specifically for this constant input
    def current_ode(t, x_state):
        dx = non_linear_dynamics(*x_state, u_true)
        return np.array(dx).flatten()
    
    # Integrate over one sample time
    sol = solve_ivp(current_ode, [0, Ts], x_true, method='RK45') #, rtol=1e-10, atol=1e-10)
    
    # Extract final state and convert back to deviation variable
    x_next_true = sol.y[:, -1]
    return x_next_true - x_eq


def sample_ellipsoid_boundary(P, current_alpha, num_samples):
    """
    Generates random state vectors 'x' that lie exactly on the 
    surface of the ellipsoid x^T P x = current_alpha.
    """
    n = P.shape[0]
    # Generate random points on an n-dimensional unit sphere
    z = np.random.randn(n, num_samples)
    z /= np.linalg.norm(z, axis=0)
    
    # Transform unit sphere points to the ellipsoid
    # using the inverse square root of P
    U, S, Vh = np.linalg.svd(P)
    P_inv_sqrt = U @ np.diag(1.0 / np.sqrt(S)) @ U.T
    
    x_samples = P_inv_sqrt @ z * np.sqrt(current_alpha)
    return x_samples


def verify_nonlinear_terminal_set(P, K, Q, R, x_eq, u_eq, current_alpha, L_frac, num_samples):
    """
    Checks Points 3 and 4 of Assumption 2.14 for the sampled points.
    """
    x_samples = sample_ellipsoid_boundary(P, current_alpha, num_samples)
    
    invariance_passed = True
    descent_passed = True

    c_frac = L_frac # number between 0 and 1, higher nr says more of the energy decrease should look linear
    
    for i in range(num_samples):
        x_k = x_samples[:, i]
        u_k = K @ x_k
        
        # Calculate next state using true nonlinear dynamics
        x_next = f_nl(x_k, u_k, x_eq, u_eq)
        
        # Calculate Lyapunov function values
        V_curr = x_k.T @ P @ x_k            # Should be exactly current_alpha
        V_next = x_next.T @ P @ x_next
        
        # Calculate Stage Cost l(x, u)
        stage_cost = x_k.T @ Q @ x_k + u_k.T @ R @ u_k
        
        # Point 3: Positive Invariance (V_next <= alpha)
        # We add a tiny tolerance (1e-8) for floating point math
        if V_next > current_alpha:
            invariance_passed = False
            break
                
        # Point 4: Local Descent (V_next - V_curr <= c_frac -stage_cost)
        if (V_next - V_curr) > (c_frac * -stage_cost):
            descent_passed = False
            violation = (V_next - V_curr) - (c_frac * -stage_cost)
            break
                
    return invariance_passed, descent_passed


def find_nonlinear_terminal_set(P, K, Q, R, x_eq, u_eq, alpha_initial, L_frac,  max_iterations, num_samples):

    alpha_high = alpha_initial  # We know this is likely too big
    alpha_low = 0.0             # We know the origin is perfectly safe
    alpha_best = 0.0            # Store the highest safe alpha we find
    
    # We will stop searching when the gap between high and low is less than 0.01
    tolerance = 0.01 
    
    for i in range(1, max_iterations + 1):
        # Test the midpoint
        alpha_mid = alpha_low + (alpha_high-alpha_low) * 0.5
        
        # Stop if precision is reached
        if (alpha_high - alpha_low) < tolerance:
            break
            
        print(f"Iteration {i:02d}: Testing alpha = {alpha_mid:.5f}...", end=" ")
        
        # Run verification function
        inv_pass, desc_pass = verify_nonlinear_terminal_set(
            P, K, Q, R, x_eq, u_eq, alpha_mid, L_frac, num_samples
        )
        
        if inv_pass and desc_pass:
            print("PASSED! (Searching higher)")
            alpha_best = alpha_mid      # Save current best safe set
            alpha_low = alpha_mid       # The true max must be higher than this
        else:
            print("FAILED. (Searching lower)")
            alpha_high = alpha_mid      # The true max must be lower than this

    if alpha_best > 0:
        print(f"Maximum safe nonlinear alpha found: {alpha_best:.5f}")
    else:
        print(f"Could not find a safe nonlinear alpha.")
        
    return alpha_best