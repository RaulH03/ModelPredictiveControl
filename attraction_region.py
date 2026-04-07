import numpy as np
import matplotlib.pyplot as plt

from run_Linear_MPC import run_mpc


def converges_to_equilibrium(history_x, tol_angle_deg=1.0, tol_rate_deg=2.0, n_last=10):
    tol_angle = np.deg2rad(tol_angle_deg)
    tol_rate = np.deg2rad(tol_rate_deg)

    x_last = history_x[:, -n_last:]

    cond1 = np.all(np.abs(x_last[0, :]) < tol_angle)
    cond2 = np.all(np.abs(x_last[1, :]) < tol_angle)
    cond3 = np.all(np.abs(x_last[2, :]) < tol_rate)
    cond4 = np.all(np.abs(x_last[3, :]) < tol_rate)

    return cond1 and cond2 and cond3 and cond4


def estimate_roa_slice():
    theta1_vals = np.deg2rad(np.linspace(-25, 25, 20))
    theta2_vals = np.deg2rad(np.linspace(-30, 30, 20))

    roa_map = np.zeros((len(theta2_vals), len(theta1_vals)))

    for i, th2 in enumerate(theta2_vals):
        for j, th1 in enumerate(theta1_vals):
            results = run_mpc(
                q_theta1=50,
                q_delta_theta2=200,
                q_theta1_dot=1,
                q_theta2_dot=1,
                r_input=1,
                x0=[th1, th2, 0.0, 0.0],
                u_max=1.5,
                theta1_max=np.deg2rad(25),
                theta2_dev_max=np.deg2rad(30),   # or your chosen bound
                N=10,
                steps=100,
            )

            feasible = all(s in ["optimal", "optimal_inaccurate"] for s in results["statuses"])

            if feasible and results["history_x"].shape[1] > 10:
                stable = converges_to_equilibrium(results["history_x"])
                roa_map[i, j] = 1 if stable else 0
            else:
                roa_map[i, j] = 0

            print(f"done {i} {j}")

    TH1, TH2 = np.meshgrid(np.rad2deg(theta1_vals), np.rad2deg(theta2_vals))

    plt.figure(figsize=(5, 4))
    plt.contourf(TH1, TH2, roa_map, levels=[-0.1, 0.5, 1.1], alpha=0.6)
    plt.xlabel(r"$\theta_1(0)$ [deg]")
    plt.ylabel(r"$\tilde{\theta}_2(0)$ [deg]")
    plt.title("Numerical estimate of the region of attraction")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    estimate_roa_slice()