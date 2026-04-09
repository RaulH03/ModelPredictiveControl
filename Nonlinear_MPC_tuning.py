import os
import numpy as np
import matplotlib.pyplot as plt

from run_Nonlinear_MPC import run_mpc

def run_and_plot(cases, tuning_name, filename, fixed_params, save_fig=False):
    print(f'testing {tuning_name} cases')
    results = []
    
    for i, case in enumerate(cases):
        print(f"Testing case {i+1}")
        mpc_params = {**fixed_params}
        for key, value in case.items():
            if key != "label":
                mpc_params[key] = value
                
        out = run_mpc(**mpc_params)
        results.append((case["label"], out))

    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    for label, out in results:
        history_x = out["history_x"]
        axes[0].plot(np.rad2deg(history_x[1, :]), label=label)
    axes[0].set_ylabel(r"$\delta \theta_2$ [deg]")
    axes[0].set_title(r"Comparison of $\delta \theta_2$ and $\theta_1$ trajectories")
    axes[0].legend()
    axes[0].grid(True)

    for label, out in results:
        history_x = out["history_x"]
        axes[1].plot(np.rad2deg(history_x[0, :]), label=label)
    axes[1].set_xlabel("Time step")
    axes[1].set_ylabel(r"$\theta_1$ [deg]")
    axes[1].legend()
    axes[1].grid(True)

    if save_fig:
        fig.savefig(f'Figures/{filename}.png', dpi=300, bbox_inches='tight')
        
    plt.close(fig)

def generate_cases(param_key, param_latex_label, test_values):
    base_weights = {
        "q_theta1": 1, "q_delta_theta2": 1, 
        "q_theta1_dot": 1, "q_theta2_dot": 1, "r_input": 1
    }
    
    identity = base_weights.copy()
    identity["label"] = r"$Identity$"
    cases = [identity]
    
    for val in test_values:
        variation = base_weights.copy()
        variation[param_key] = val
        variation["label"] = rf"${param_latex_label}={val}$"
        cases.append(variation)
        
    return cases

def compare_parameter_cases(save_fig=False):
    if save_fig:
        os.makedirs('Figures', exist_ok=True)
        
    fixed_params = {
        "x0": [0.0, np.deg2rad(20), 0.0, 0.0],
        "u_max": 2,
        "theta1_max": np.deg2rad(90),
        "theta2_dev_max": np.deg2rad(90),
        "N": 20,
        "steps": 100
    }

    studies = [
        {"name": "q_theta_1",     "key": "q_theta1",       "latex": r"q_{\theta 1}",       "file": "qtheta1_tuning"},
        {"name": "q_theta_2",     "key": "q_delta_theta2", "latex": r"q_{\theta 2}",       "file": "qtheta2_tuning"},
        {"name": "q_theta_dot_1", "key": "q_theta1_dot",   "latex": r"q_{\dot{\theta} 1}", "file": "qthetadot1_tuning"},
        {"name": "q_theta_dot_2", "key": "q_theta2_dot",   "latex": r"q_{\dot{\theta} 2}", "file": "qthetadot2_tuning"},
    ]

    test_multipliers = [0.1, 10, 100]

    for study in studies:
        cases = generate_cases(study["key"], study["latex"], test_multipliers)
        run_and_plot(cases, study["name"], study["file"], fixed_params, save_fig)

if __name__ == "__main__":
    compare_parameter_cases(save_fig=True)