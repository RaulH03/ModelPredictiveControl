import numpy as np

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