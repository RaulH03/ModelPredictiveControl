import numpy as np
from scipy.optimize import linprog

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