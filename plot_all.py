import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot


def plot_curves_exact(env_name, algorithm_name, num_iteration=1, is_warm_start=False, weak_warm_start="", seed=0, base=False, max_x=1e8):
    save_prefix = "/root/data/results"
    # kuhn_poker_PDO_100_wsFalse_0_exps.npy
    # '/root/data/results/' + self.algorithm + str(self.meta_iterations) + '_' + self.game_name + f'_{seed}'
    # lcfr_liars_dice_0__exps.npy
    # kuhn_poker_AdaDO_1_wsFalse_weakFalse_0.1_0_exps.npy
    # python_large_kuhn_poker_PDO_100_wsTrue_weakTrue_0.1_0_times.npy
    # python_large_kuhn_poker_PDO_100_wsTrue_avgTrue_0.1_0_times.npy

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
    # for algorithm_name in ["PDO"]:
    #     for num_iteration in [100]: #, 50, 100, 500]:
    #         plot_curves_exact(env_name, 
    #                             algorithm_name, 
    #                             num_iteration, 
    #                             is_warm_start=False, 
    #                             max_x=1e9)
            
    #         for weak_warm_start in ["", "_avgTrue_0.1"]:
    #             plot_curves_exact(env_name, 
    #                               algorithm_name, 
    #                               num_iteration, 
    #                               is_warm_start=True,
    #                               weak_warm_start=weak_warm_start, 
    #                               max_x=1e9)
                
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


def plot_mean_curve_with_error_bars(num_seeds, env_name, algorithm_name, num_iteration, is_warm_start, color, max_x_axis, base=False):
    save_prefix = '/root/data/results/'
    # Initialize lists to store interpolated data for each seed
    interp_seeds = []
    seed_x_values = []

    for seed in range(0, num_seeds):
        # Generate filenames based on the provided format

        if base:
            data_filename = save_prefix + algorithm_name + '_' + env_name + f"_{seed}__exps.npy"
            x_filename = save_prefix + algorithm_name + '_' + env_name + f"_{seed}__infostates.npy"

        else:
            # f'/root/data/results/{self.game_name}_{self.algorithm}_{self.meta_iterations}_ws{self.is_warm_start}_{seed}'
            data_filename = f'{save_prefix}{env_name}_{algorithm_name}_{num_iteration}_ws{is_warm_start}_{seed}_exps.npy'
            x_filename = f'{save_prefix}{env_name}_{algorithm_name}_{num_iteration}_ws{is_warm_start}_{seed}_infostates.npy'
   
        # Load data for each seed and its corresponding x-values
        seed_data = np.load(data_filename)
        x_values = np.load(x_filename)
#         print(x_filename)
        # x_values = pickle.load(open(x_filename,'rb'))
        
        interp = interp1d(x_values, seed_data, kind='cubic')
        interp_seeds.append(interp)
        seed_x_values.append(x_values)

    # Find the minimum and maximum x-values across all seeds
    min_x = max([np.min(x) for x in seed_x_values])
    max_x = min([np.max(x) for x in seed_x_values] + [max_x_axis])

    # Generate a sequence of x-values from min_x to max_x
    x_sequence = np.linspace(min_x, max_x, num=1000)  # You can adjust the num parameter for the desired number of points

    # Initialize arrays to store mean and standard deviation
    mean_data = np.zeros(len(x_sequence))
    std_dev_data = np.zeros(len(x_sequence))

    # Calculate the mean and standard deviation at each x-value
    for i, x in enumerate(x_sequence):
        values_at_x = [interp(x) for interp in interp_seeds]
        mean_data[i] = np.mean(values_at_x)
        std_dev_data[i] = np.std(values_at_x)

    # Plot the curve with error bars
    plt.plot(x_sequence, mean_data, label=f'{algorithm_name}_{num_iteration}_ws{is_warm_start}', color=color)
    plt.fill_between(x_sequence, mean_data - std_dev_data, mean_data + std_dev_data, alpha=0.3, color=color)

# colors = generate_distinct_colors(50)
colors = matplotlib.cm.tab10(range(10))

# Example usage:
num_seeds = 100  # Replace with the actual number of seeds
env_name = "your_env_name"  # Replace with your environment name
algorithm_name = "your_algorithm_name"  # Replace with your algorithm name
num_iteration = 1000  # Replace with the number of iterations
is_warm_start = True  # Replace with True or False as needed
legend_done = False
max_x_axis_dict = {"kuhn_poker": 1.5e7, "python_large_kuhn_poker":5e7, "leduc_poker": 5e6, "leduc_poker_dummy": 1e9, "liars_dice": 1e9, "oshi_zumo":1e9}

for env_name in ["leduc_poker", "leduc_poker_dummy", "python_large_kuhn_poker", "kuhn_poker", "liars_dice", "oshi_zumo"]:
    plt.cla()
    i = 0
    for algorithm_name in ["SPDO"]:
        for num_iteration in [1000]: #500, 1000, 5000, 10000]:
            plot_mean_curve_with_error_bars(5, env_name, algorithm_name, num_iteration, False, colors[i], max_x_axis=max_x_axis_dict[env_name])
            i += 1
            plot_mean_curve_with_error_bars(5, env_name, algorithm_name, num_iteration, True, colors[i], max_x_axis=max_x_axis_dict[env_name])
            i += 1
    
    plot_mean_curve_with_error_bars(5, env_name, "outcome_sampling_mccfr", num_iteration, True, colors[i], max_x_axis=max_x_axis_dict[env_name], base=True)
    
    plt.yscale("log")
    plt.xlabel('# of Information states visited')
    plt.ylabel('Exploitability')
    plt.title(f'{env_name}')
    # if not legend_done:
    #     plt.legend(loc='upper center', bbox_to_anchor=(1.5, 1.5), fancybox=True, shadow=True, ncol=5)
    #     legend_done = True
    plt.legend(loc='upper right') #, bbox_to_anchor=(1.5, 1.5), fancybox=True, shadow=True, ncol=5)
    plt.grid(True)
    plt.savefig(f"results/{env_name}_stochastic_new.png")
