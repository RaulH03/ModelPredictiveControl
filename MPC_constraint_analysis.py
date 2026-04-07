import numpy as np
import matplotlib.pyplot as plt

from run_Linear_MPC import run_mpc


def constraint_analysis():
    # Final selected controller parameters
    q_theta1 = 50
    q_delta_theta2 = 200
    q_theta1_dot = 1
    q_theta2_dot = 1
    r_input = 1
    N = 10
    steps = 100

    # Constraints
    u_max = 1.5
    theta1_max = np.deg2rad(25)
    # theta2_dev_max = np.deg2rad(25)

    # Vary only the initial passive-link deviation
    initial_deviations_deg = [20, 21, 22, 27]

    results = []

    for deg in initial_deviations_deg:
        x0 = [0.0, np.deg2rad(deg), 0.0, 0.0]

        out = run_mpc(
            q_theta1=q_theta1,
            q_delta_theta2=q_delta_theta2,
            q_theta1_dot=q_theta1_dot,
            q_theta2_dot=q_theta2_dot,
            r_input=r_input,
            x0=x0,
            u_max=u_max,
            theta1_max=theta1_max,
            theta2_dev_max=deg,
            N=N,
            steps=steps,
        )

        history_x = out["history_x"]
        history_u = out["history_u"]
        status_ok = all(s in ["optimal", "optimal_inaccurate"] for s in out["statuses"])

        max_u = np.max(np.abs(history_u[0, :])) if history_u.shape[1] > 0 else np.nan
        max_theta1 = np.max(np.abs(np.rad2deg(history_x[0, :]))) if history_x.shape[1] > 0 else np.nan
        max_theta2 = np.max(np.abs(np.rad2deg(history_x[1, :]))) if history_x.shape[1] > 0 else np.nan

        input_active = max_u >= u_max - 1e-3
        theta1_active = np.max(np.abs(history_x[0, :])) >= theta1_max - 1e-6 if history_x.shape[1] > 0 else False
        theta2_active = np.max(np.abs(history_x[1, :])) >= deg - 1e-6 if history_x.shape[1] > 0 else False

        results.append({
            "initial_deg": deg,
            "feasible": status_ok,
            "input_active": input_active,
            "theta1_active": theta1_active,
            "theta2_active": theta2_active,
            "max_u": max_u,
            "max_theta1_deg": max_theta1,
            "max_theta2_deg": max_theta2,
            "history_x": history_x,
            "history_u": history_u,
        })

    # Print summary table
    print("\nConstraint analysis summary")
    print("-" * 95)
    print(f"{'tilde(theta2)(0) [deg]':>22} | {'Feasible':>8} | {'Input active':>12} | {'theta1 active':>13} | {'theta2 active':>13} | {'Max |u| [Nm]':>12}")
    print("-" * 95)
    for res in results:
        print(f"{res['initial_deg']:22.1f} | "
              f"{str(res['feasible']):>8} | "
              f"{str(res['input_active']):>12} | "
              f"{str(res['theta1_active']):>13} | "
              f"{str(res['theta2_active']):>13} | "
              f"{res['max_u']:12.3f}")

    # Plot 1: control input for different initial deviations
    plt.figure(figsize=(6, 4.5))
    for res in results:
        history_u = res["history_u"]
        if history_u.shape[1] > 0:
            plt.step(
                range(history_u.shape[1]),
                history_u[0, :],
                where="post",
                label=fr"$\tilde{{\theta}}_2(0)={res['initial_deg']}^\circ$"
            )
    plt.axhline(u_max, color="red", linestyle="--", label=r"Constraint $\pm u_{\max}$")
    plt.axhline(-u_max, color="red", linestyle="--")
    plt.xlabel("Time step")
    plt.ylabel(r"$u$ [Nm]")
    plt.title("Input constraint activation for different initial deviations")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot 2: theta1 response for different initial deviations
    plt.figure(figsize=(6, 4.5))
    for res in results:
        history_x = res["history_x"]
        if history_x.shape[1] > 0:
            plt.plot(
                np.rad2deg(history_x[0, :]),
                label=fr"$\tilde{{\theta}}_2(0)={res['initial_deg']}^\circ$"
            )
    plt.axhline(np.rad2deg(theta1_max), color="red", linestyle="--", label=r"Constraint $\pm \theta_{1,\max}$")
    plt.axhline(-np.rad2deg(theta1_max), color="red", linestyle="--")
    plt.xlabel("Time step")
    plt.ylabel(r"$\theta_1$ [deg]")
    plt.title(r"First-link response for different initial deviations")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot 3: passive-link response for different initial deviations
    plt.figure(figsize=(5, 4))
    for res in results:
        history_x = res["history_x"]
        if history_x.shape[1] > 0:
            plt.plot(
                np.rad2deg(history_x[1, :]),
                label=fr"$\tilde{{\theta}}_2(0)={res['initial_deg']}^\circ$"
            )
    plt.axhline(np.rad2deg(deg), color="red", linestyle="--", label=r"Constraint $\pm \tilde{\theta}_{2,\max}$")
    plt.axhline(-np.rad2deg(deg), color="red", linestyle="--")
    plt.xlabel("Time step")
    plt.ylabel(r"$\tilde{\theta}_2$ [deg]")
    plt.title(r"Passive-link response for different initial deviations")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    constraint_analysis()


if __name__ == "__main__":
    constraint_analysis()