import os
import numpy as np
import matplotlib.pyplot as plt

from run_Nonlinear_MPC import run_mpc
def run_and_plot(cases, tuning_name, latex, filename, fixed_params, save_fig=False, only_control=False):
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

    

    # plt.rcParams.update({
    # 'font.size': 6,          # Standard IEEE text size
    # 'axes.labelsize': 6,     # Slightly larger for axis titles
    # "axes.titlesize": 6,
    # 'legend.fontsize': 6,    # Smaller for legends to save space
    # 'xtick.labelsize': 6,
    # 'ytick.labelsize': 6,
    # 'lines.linewidth': 1.2   # Slightly thicker lines for visibility
    # })

    # 1. Delta Theta 2
    if only_control:
        fig, ax = plt.subplots(1, 1, figsize=(3.5, 2.5), sharex=True)
        

        # 3. Control Input (u) - Stepped Plot
        for label, out in results:
            history_u = out["history_u"].flatten()
            # Using step plot to show Zero-Order Hold behavior
            ax.step(range(len(history_u)), history_u, where='post', label=label)
        
        u_limit = fixed_params.get("u_max")
        if u_limit:
            ax.axhline(y=u_limit, color='r', linestyle='--', alpha=0.5, label='Limit')
            ax.axhline(y=-u_limit, color='r', linestyle='--', alpha=0.5)

        ax.set_xlabel("Time step")
        ax.set_ylabel("Torque [Nm]")
        ax.grid(True)
        ax.set_title(rf"MPC Control input")
        ax.legend(loc='upper right')
        ax.grid(True)

        plt.tight_layout()

        if save_fig:
            fig.savefig(f'Figures/{filename}.png', dpi=300, bbox_inches='tight')
        else: 
            plt.show() # Added show() for immediate feedback if not saving
        

    else:
        fig, axes = plt.subplots(3, 1, figsize=(3.5, 4.5), sharex=True)

        for label, out in results:
            history_x = out["history_x"]
            axes[0].plot(np.rad2deg(history_x[1, :]), label=label)
        axes[0].set_ylabel(r"$\delta \theta_2$ [deg]")
        axes[0].set_title(rf"MPC Study: ${latex}$")
        axes[0].legend(loc='upper right')
        axes[0].grid(True)

        # 2. Theta 1
        for label, out in results:
            history_x = out["history_x"]
            axes[1].plot(np.rad2deg(history_x[0, :]), label=label)
        axes[1].set_ylabel(r"$\theta_1$ [deg]")
        axes[1].grid(True)

        # 3. Control Input (u) - Stepped Plot
        for label, out in results:
            history_u = out["history_u"].flatten()
            # Using step plot to show Zero-Order Hold behavior
            axes[2].step(range(len(history_u)), history_u, where='post', label=label)
        
        u_limit = fixed_params.get("u_max")
        if u_limit:
            axes[2].axhline(y=u_limit, color='r', linestyle='--', alpha=0.5, label='Limit')
            axes[2].axhline(y=-u_limit, color='r', linestyle='--', alpha=0.5)

        axes[2].set_xlabel("Time step")
        axes[2].set_ylabel("Torque [Nm]")
        axes[2].grid(True)

        plt.tight_layout()

        if save_fig:
            fig.savefig(f'Figures/{filename}.png', dpi=300, bbox_inches='tight')
        else: 
            plt.show() # Added show() for immediate feedback if not saving

def generate_cases(param_key, param_latex_label, test_values, base_config=None):
    # Default baseline if nothing is passed
    default_base = {
        "x0": [0.0, np.deg2rad(20), 0.0, 0.0], 
        "q_theta1": 1, "q_delta_theta2": 1, 
        "q_theta1_dot": 1, "q_theta2_dot": 1, "r_input": 1,
        "N": 20  # Adding N to the base so we can tune it too
    }
    
    # Merge user-defined base_config into the defaults
    if base_config:
        current_base = {**default_base, **base_config}
    else:
        current_base = default_base

    cases = []
    for val in test_values:
        variation = current_base.copy()
        variation[param_key] = val
        
        # Format the label nicely if we are testing the x0 array
        if param_key == "x0":
            deg_val = int(np.round(np.rad2deg(val[1])))
            variation["label"] = rf"${param_latex_label}={deg_val}^\circ$"
        else:
            variation["label"] = rf"${param_latex_label}={val}$"
            
        cases.append(variation)
        
    return cases

def compare_parameter_cases(save_fig=False):
    if save_fig:
        os.makedirs('Figures', exist_ok=True)
        
    fixed_simulation_params = {
        "u_max": 1.5,
        "theta1_max": np.deg2rad(90),
        "theta2_dev_max": np.deg2rad(90),
        "steps": 100
    }

    # define best weigths from previous study
    best_weights = {
        "q_delta_theta2": 200,  
        "q_theta1": 1,      
        "q_theta2_dot": 1,
        "q_theta1_dot": 10,
        "N": 30
    }

    studies = [
        # {
        #     "name": "q_delta_theta2 Study", 
        #     "key": "q_delta_theta2", 
        #     "latex": r"q_{\theta 2}", 
        #     "file": "qtheta2_tuning", 
        #     "test_values": [10, 100, 200, 500],
        #     "use_best_weights": False  
        # },
        # {
        #     "name": "q_theta1 Study", 
        #     "key": "q_theta1", 
        #     "latex": r"q_{\theta 1}", 
        #     "file": "qtheta1_tuning", 
        #     "test_values": [0.1, 1, 10, 100],
        #     "use_best_weights": True 
        # },

        # {
        #     "name": "q_theta2_dot Study", 
        #     "key": "q_theta2_dot", 
        #     "latex": r"q_{\dot{\theta} 2}", 
        #     "file": "qtheta2_dot_tuning", 
        #     "test_values": [0.1, 1, 10, 100],
        #     "use_best_weights": True 
        # },

        # {
        #     "name": "q_theta1_dot Study", 
        #     "key": "q_theta1_dot", 
        #     "latex": r"q_{\dot{\theta} 1}", 
        #     "file": "qtheta1_dot_tuning", 
        #     "test_values": [0.1, 1, 10, 100],
        #     "use_best_weights": True 
        # },

        # {
        #     "name": "R Study", 
        #     "key": "r_input", 
        #     "latex": "R", 
        #     "file": "r_tuning", 
        #     "test_values": [0.1, 1, 10, 100],
        #     "use_best_weights": True
        # }
        
        # {
        #     "name": "Horizon Study", 
        #     "key": "N", 
        #     "latex": "N", 
        #     "file": "horizon_study", 
        #     "test_values": [20, 30, 40],
        #     "use_best_weights": True  # Flag to use the weights above
        # }

        {
            "name": "Constraint analysis", 
            "key": "x0", 
            "latex": r"\theta_{2, 0}", # Slightly cleaner LaTeX
            "file": "constraint_analysis", 
            "test_values": [
                [0.0, np.deg2rad(20), 0.0, 0.0],
                [0.0, np.deg2rad(22), 0.0, 0.0],
                [0.0, np.deg2rad(24), 0.0, 0.0],
                [0.0, np.deg2rad(26), 0.0, 0.0],
                [0.0, np.deg2rad(28), 0.0, 0.0],
                [0.0, np.deg2rad(30), 0.0, 0.0]
            ],
            "use_best_weights": True 
        }


    ]

    for study in studies:
        # Determine which base to start from
        base = best_weights if study.get("use_best_weights") else None
        
        cases = generate_cases(
            study["key"], 
            study["latex"], 
            study["test_values"], 
            base_config=base
        )
        
        run_and_plot(cases, study["name"], study["latex"], study["file"], fixed_simulation_params, save_fig, only_control=True)

if __name__ == "__main__":
    compare_parameter_cases(save_fig=False)