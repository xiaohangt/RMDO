import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt


def plot_curves_exact(env_name, algorithm_name, num_iteration=1, is_warm_start=False, weak_warm_start="", seed=0, base=False, max_x=1e8):
    save_prefix = "/root/data/results"
    # kuhn_poker_PDO_100_wsFalse_0_exps.npy
    # '/root/data/results/' + self.algorithm + str(self.meta_iterations) + '_' + self.game_name + f'_{seed}'
    # lcfr_liars_dice_0__exps.npy
    # kuhn_poker_AdaDO_1_wsFalse_weakFalse_0.1_0_exps.npy
    # python_large_kuhn_poker_PDO_100_wsTrue_weakTrue_0.1_0_times.npy


    if base:
        data_filename = f"{save_prefix}/{algorithm_name}_{env_name}_{seed}__exps.npy"
        x_filename = f"{save_prefix}/{algorithm_name}_{env_name}_{seed}__infostates.npy"
    else:
        data_filename = f"{save_prefix}/{env_name}_{algorithm_name}_{num_iteration}_ws{is_warm_start}{weak_warm_start}_{seed}_exps.npy"
        x_filename = f"{save_prefix}/{env_name}_{algorithm_name}_{num_iteration}_ws{is_warm_start}{weak_warm_start}_{seed}_infostates.npy"

    seed_data = np.load(data_filename)
    x_values = np.load(x_filename)

    selected_inds = x_values < max_x

    plt.plot(x_values[selected_inds], seed_data[selected_inds], label=f'{algorithm_name}_{num_iteration}_ws{is_warm_start}{weak_warm_start}')


# Example usage:
num_seeds = 100  # Replace with the actual number of seeds
env_name = "your_env_name"  # Replace with your environment name
algorithm_name = "your_algorithm_name"  # Replace with your algorithm name
num_iteration = 1000  # Replace with the number of iterations
is_warm_start = True  # Replace with True or False as needed
legend_done = False

for env_name in ["python_large_kuhn_poker", "liars_dice", "leduc_poker", "leduc_poker_dummy", "kuhn_poker", "oshi_zumo"]:
    plt.cla()
    for algorithm_name in ["PDO"]:
        for num_iteration in [100]: #, 50, 100, 500]:
            plot_curves_exact(env_name, 
                                algorithm_name, 
                                num_iteration, 
                                is_warm_start=False, 
                                max_x=1e8)
            
#             for weak_warm_start in ["", "_weakTrue_0.1"]:
#                 plot_curves_exact(env_name, 
#                                   algorithm_name, 
#                                   num_iteration, 
#                                   is_warm_start=True,
#                                   weak_warm_start=weak_warm_start, 
#                                   max_x=1e9)
                
    for is_warm_start in [True, False]:
        plot_curves_exact(env_name, "AdaDO", is_warm_start=is_warm_start, weak_warm_start="_weakFalse_0.1")
        
    plot_curves_exact(env_name, "lcfr", base=True)
    plot_curves_exact(env_name, "dxdo", base=True)
    
    plt.yscale("log")
    plt.xlabel('# of Information states visited')
    plt.ylabel('Exploitability')
    plt.title(f'{env_name}')
    # if not legend_done:
    plt.legend(loc='upper right') #, bbox_to_anchor=(1.5, 1.5), fancybox=True, shadow=True, ncol=5)
        # legend_done = True
    plt.grid(True)
    plt.savefig(f"results/{env_name}_exact_new.png")
