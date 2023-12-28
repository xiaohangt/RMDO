import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot
import pandas as pd
import csv
import pdb
import os

bars =  [0.5, 0.2, 0.09, 0.02, 0.005, 0.0003]
mult = 2
SMALL_SIZE = 8*mult
MEDIUM_SIZE = 10*mult*1.6
BIGGER_SIZE = 12*mult*1.3
def set_plot(legend_size=None):
    if not legend_size:
        legend_size = 1.6
    plt.clf()
    plt.figure(figsize=(15, 8), dpi=200)
    plt.rcParams['lines.linewidth'] = 3
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE*legend_size)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

game_name_dict = {"python_large_kuhn_poker": "Large Kuhn Poker", "blotto_20":"Blotto 20","blotto_25":"Blotto 25",
                  "blotto_30":"Blotto 30","blotto_40":"Blotto 40",
                  "liars_dice": "Liars Dice", "leduc_poker": "Leduc Poker", 
                  "leduc_poker_dummy": "Leduc Poker Dummy", "kuhn_poker": "Kuhn Poker",
                    "oshi_zumo": "Oshi Zumo", "leduc_poker_10_card":"Leduc Poker 10 Card"}

def get_algo_name(raw_algo_name, num_iteration=1, is_warm_start=False, is_mcbr="", sto_do=False):
    name_dict = {"dxdo":"XDO", "AdaDO":"AdaDO", "outcome_sampling_mccfr": "MCCFR"}
    ws = "-WS" if is_warm_start else ""
    name_capital = name_dict.get(raw_algo_name, raw_algo_name.upper())
    if raw_algo_name != "outcome_sampling_mccfr":
    #     is_mcbr = "-MCCBR" if is_mcbr else "-OBR"
        if num_iteration > 1 and "PDO" in name_capital:
            algo_name = f"{name_capital}({num_iteration}){is_mcbr}{ws}" 
        else:
            algo_name = f"{name_capital}{ws}" 
    else:
        algo_name = name_capital
    return algo_name


def get_support(env_name, algorithm_name, num_iteration=1, is_warm_start=False, weak_warm_start="", seed=0, base=False, max_x=1e8):
    save_prefix = "results"
    # kuhn_poker_PDO_100_wsFalse_0_exps.npy
    # '/root/data/results/' + self.algorithm + str(self.meta_iterations) + '_' + self.game_name + f'_{seed}'
    # lcfr_liars_dice_0__exps.npy
    # kuhn_poker_AdaDO_1_wsFalse_weakFalse_0.1_0_exps.npy
    # python_large_kuhn_poker_PDO_100_wsTrue_weakTrue_0.1_0_times.npy
    # python_large_kuhn_poker_PDO_100_wsTrue_avgTrue_0.1_0_times.npy
    info_dict = defaultdict(list)

    if base:
        data_filename = f"{save_prefix}/{algorithm_name}_{env_name}_{seed}__exps.npy"
        info_filename = f"{save_prefix}/{algorithm_name}_{env_name}_{seed}__infos.npy"
    else:
        data_filename = f"{save_prefix}/{env_name}_{algorithm_name}_{num_iteration}_ws{is_warm_start}{weak_warm_start}_{seed}_exps.npy"
        info_filename = f"{save_prefix}/{env_name}_{algorithm_name}_{num_iteration}_ws{is_warm_start}{weak_warm_start}_{seed}_infos.npy"

    infos = np.load(info_filename)
    exps = np.load(data_filename)

    ind = np.where(exps < 1e-3)
    if len(ind[0]) > 0:
        info = infos[ind[0][0]] 
    else:
        return
    
    if "DO" in algorithm_name:
        info_out = [game_name_dict[env_name], algorithm_name.upper()] + list([f"{int(ele*100)}%" for ele in info[1:]])
    else:
        info_out = [game_name_dict[env_name], algorithm_name.upper()] + list([f"{int(ele*100)}%" for ele in info])
    
    with open('support.csv', 'a') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(info_out)
        

################### Getting supports informations ###########################
def run_get_support():
    os.system('rm support.csv')

    for env_name in ["python_large_kuhn_poker", "liars_dice", "leduc_poker", "leduc_poker_dummy", "kuhn_poker", "oshi_zumo"]:
        for algorithm_name in ["PDO"]:
            for num_iteration in [100]: #, 50, 100, 500]:
                
                for weak_warm_start in ["_avgFalse_0.1"]:
                    get_support(env_name, 
                                    algorithm_name, 
                                    num_iteration, 
                                    is_warm_start=False,
                                    weak_warm_start=weak_warm_start, 
                                    max_x=1e9)
            
        get_support(env_name, "lcfr", base=True)
        # get_support(env_name, "dxdo", base=True)



def plot_curves_exact(env_name, algorithm_name, num_iteration=1, is_warm_start=False, weak_warm_start="", seed=0, base=False, max_x=1e8, ax=None):
    save_prefix = "/root/data/results"
    # kuhn_poker_PDO_100_wsFalse_0_exps.npy
    # '/root/data/results/' + self.algorithm + str(self.meta_iterations) + '_' + self.game_name + f'_{seed}'
    # lcfr_liars_dice_0__exps.npy
    # kuhn_poker_AdaDO_1_wsFalse_weakFalse_0.1_0_exps.npy
    # python_large_kuhn_poker_PDO_100_wsTrue_weakTrue_0.1_0_times.npy
    # python_large_kuhn_poker_PDO_100_wsTrue_avgTrue_0.1_0_times.npy
    info_dict = defaultdict(list)
    results, results_seeds = [], defaultdict(list)

    if base:
        data_filename = f"{save_prefix}/{algorithm_name}_{env_name}_{seed}__exps.npy"
        x_filename = f"{save_prefix}/{algorithm_name}_{env_name}_{seed}__infostates.npy"
        info_filename = f".results_support/{algorithm_name}_{env_name}_{seed}__infos.npy"
    else:
        data_filename = f"{save_prefix}/{env_name}_{algorithm_name}_{num_iteration}_ws{is_warm_start}{weak_warm_start}_{seed}_exps.npy"
        x_filename = f"{save_prefix}/{env_name}_{algorithm_name}_{num_iteration}_ws{is_warm_start}{weak_warm_start}_{seed}_infostates.npy"
        info_filename = f".results_support/{env_name}_{algorithm_name}_{num_iteration}_ws{is_warm_start}{weak_warm_start}_{seed}_infos.npy"

    print(data_filename)
    seed_data = np.load(data_filename)
    x_values = np.load(x_filename)
    min_infos = {}

    for i, bar in enumerate(bars):
        if bar in min_infos:
            min_infos[bar] = min(min_infos[bar], x_values[-1])
        else:
            min_infos[bar] = x_values[-1]
        ind = np.where(seed_data < bar)
        if len(ind[0]) > 0:
            results_seeds[bar].append(x_values[ind[0][0]] / 1e6)
        else:
            break
    
    for key in bars:
        if key not in results_seeds:
            results.append([key, "failed, infostates <= ", min_infos[bar]/1e6])
        else:
            results.append([env_name, algorithm_name, num_iteration, key, np.round(np.mean(results_seeds[key]), decimals=3)])

    with open('exact_methods.csv', 'a') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(results)

    selected_inds = x_values < max_x
    if (algorithm_name == "AdaDO") and (env_name != "leduc_poker_dummy") and (env_name != "leduc_poker"):
        sep = 500
    else:
        sep = 1

    # plt.plot(x_values[selected_inds], seed_data[selected_inds], label=f'{algorithm_name}_{num_iteration}_ws{is_warm_start}{weak_warm_start}')
    if ax is not None:
        ax.plot(x_values[selected_inds][::sep], seed_data[selected_inds][::sep], label=get_algo_name(algorithm_name, num_iteration, is_warm_start))
    else:
        plt.plot(x_values[selected_inds][::sep], seed_data[selected_inds][::sep], label=get_algo_name(algorithm_name, num_iteration, is_warm_start))


########################## Example usage of plotting exact methods ######################################
def run_plot_exact_methods():
    os.system("rm exact_methods.csv")

    num_seeds = 100  # Replace with the actual number of seeds
    env_name = "your_env_name"  # Replace with your environment name
    algorithm_name = "your_algorithm_name"  # Replace with your algorithm name
    num_iteration = 1000  # Replace with the number of iterations
    is_warm_start = True  # Replace with True or False as needed
    legend_done = False
    max_x_axis_dict = {"kuhn_poker": 1e8, "python_large_kuhn_poker": 1e8, "leduc_poker": 1e9, "leduc_poker_dummy": 1e9, "liars_dice": 1e9, "oshi_zumo":1e9}

    set_plot()
    for env_name in ["liars_dice", "python_large_kuhn_poker", "leduc_poker", "leduc_poker_dummy", "kuhn_poker", "oshi_zumo"]:
        plt.cla()
        for algorithm_name in ["PDO"]:
            for num_iteration in [100]: #, 50, 100, 500]:
                plot_curves_exact(env_name, 
                                    algorithm_name, 
                                    num_iteration, 
                                    is_warm_start=False, 
                                    max_x=max_x_axis_dict[env_name])
                
                for weak_warm_start in [""]: #, "_avgTrue_0.1"]:
                    plot_curves_exact(env_name, 
                                    algorithm_name, 
                                    num_iteration, 
                                    is_warm_start=True,
                                    weak_warm_start=weak_warm_start, 
                                    max_x=max_x_axis_dict[env_name])
                    
        # for is_warm_start in [True, False]:
        #     plot_curves_exact(env_name, "AdaDO", is_warm_start=is_warm_start, weak_warm_start="_avgFalse_0.1")
            
        plot_curves_exact(env_name, "lcfr", base=True)
        # plot_curves_exact(env_name, "dxdo", base=True)
        
        plt.yscale("log")
        plt.xlabel('Number of nodes visited')
        plt.ylabel('Exploitability')
        plt.title(f'{game_name_dict[env_name]}')
        # if not legend_done:
        plt.legend(loc='upper right') #, bbox_to_anchor=(1.5, 1.5), fancybox=True, shadow=True, ncol=5)
            # legend_done = True
        plt.grid(True)
        plt.savefig(f"results_figs/{env_name}_exact_new.png", bbox_inches='tight', dpi=200, transparent=True)


def get_sto_methods_data(out_dir, num_seeds, env_name, algorithm_name, num_iteration, is_warm_start, base=False):
    save_prefix = f'{out_dir}/'
    # Initialize lists to store interpolated data for each seed
    interp_seeds = []
    seed_x_values = []
    min_infos = {}
    results = [game_name_dict[env_name], get_algo_name(algorithm_name, num_iteration, is_warm_start)]
    results_seeds = defaultdict(list)
    for seed in range(0, num_seeds):
        # Generate filenames based on the provided format

        if base:
            data_filename = save_prefix + algorithm_name + '_' + env_name + f"_{seed}__exps.npy"
            x_filename = save_prefix + algorithm_name + '_' + env_name + f"_{seed}__infostates.npy"
        else:
            data_filename = f'{save_prefix}{env_name}_{algorithm_name}_{num_iteration}_ws{is_warm_start}_{seed}_exps.npy'
            x_filename = f'{save_prefix}{env_name}_{algorithm_name}_{num_iteration}_ws{is_warm_start}_{seed}_infostates.npy'

        # Load data for each seed and its corresponding x-values
        exps = np.load(data_filename)
        x_values = np.load(x_filename)

        for i, bar in enumerate(bars):
            if bar in min_infos:
                min_infos[bar] = min(min_infos[bar], x_values[-1])
            else:
                min_infos[bar] = x_values[-1]
            ind = np.where(exps < bar)
            if len(ind[0]) > 0:
                results_seeds[bar].append(x_values[ind[0][0]] / 1e6)
            else:
                break
    
    for key in bars:
        if key not in results_seeds:
            results.append([key, "failed, infostates <= ", min_infos[bar]/1e6])
        else:
            results.append([key, f"{np.round(np.mean(results_seeds[key]), decimals=3)}" + '$\\pm$' + f"{np.round(np.std(results_seeds[key]), decimals=3)}"])

    with open('stochastic_methods.csv', 'a') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(results)


def plot_mean_curve_with_error_bars(ax, num_seeds, env_name, algorithm_name, num_iteration, is_warm_start, is_avg_warm, color, max_x_axis, base=False, is_mcbr=""):
    save_prefix = '/home/vipuser/Documents/RMDO/results/'
    # Initialize lists to store interpolated data for each seed
    interp_seeds = []
    seed_x_values = []
    all_min_exp = []

    for seed in range(0, num_seeds):
        # Generate filenames based on the provided format

        if base:
            data_filename = save_prefix + algorithm_name + '_' + env_name + f"_{seed}__exps.npy"
            x_filename = save_prefix + algorithm_name + '_' + env_name + f"_{seed}__infostates.npy"
        elif algorithm_name=="SPDO":
            data_filename = f'{save_prefix}{env_name}_{algorithm_name}_{num_iteration}_ws{is_warm_start}{is_mcbr}_{seed}_exps.npy'
            x_filename = f'{save_prefix}{env_name}_{algorithm_name}_{num_iteration}_ws{is_warm_start}{is_mcbr}_{seed}_infostates.npy'
        else:
            # f'/root/data/results/{self.game_name}_{self.algorithm}_{self.meta_iterations}_ws{self.is_warm_start}_{seed}'
            data_filename = f'{save_prefix}{env_name}_{algorithm_name}_{num_iteration}_ws{is_warm_start}{is_mcbr}_avg{is_avg_warm}_0.1_{seed}_exps.npy'
            x_filename = f'{save_prefix}{env_name}_{algorithm_name}_{num_iteration}_ws{is_warm_start}{is_mcbr}_avg{is_avg_warm}_0.1_{seed}_infostates.npy'
   
        if data_filename == "/root/data/results/leduc_poker_SPDO_100_wsFalse_4_exps.npy":
            return 
        # Load data for each seed and its corresponding x-values
        seed_data = np.load(data_filename)
        x_values = np.load(x_filename)
        all_min_exp.append(f"{x_values[np.argmin(seed_data)] / 1e6}:{np.min(seed_data)}")
        
        interp = interp1d(x_values, seed_data, kind='linear')
        interp_seeds.append(interp)
        seed_x_values.append(x_values)

    print(data_filename, all_min_exp)
    # Find the minimum and maximum x-values across all seeds
    min_x = max([x[0] for x in seed_x_values])
    max_x = min([x[-1] for x in seed_x_values] + [max_x_axis])

    # Generate a sequence of x-values from min_x to max_x
    x_sequence = np.linspace(min_x, max_x, num=1000)  # You can adjust the num parameter for the desired number of points

    # Initialize arrays to store mean and standard deviation
    mean_data = np.ones(len(x_sequence)) * 1e10
    std_dev_data = np.ones(len(x_sequence)) * 1e10

    # Calculate the mean and standard deviation at each x-value
    for i, x in enumerate(x_sequence):
        values_at_x = [interp(x) for interp in interp_seeds]
        mean_data[i] = np.mean(values_at_x) 
        std_dev_data[i] = np.std(values_at_x)
        

    # Plot the curve with error bars
    ax.plot(x_sequence, mean_data, label=get_algo_name(algorithm_name, num_iteration, is_warm_start, is_mcbr, sto_do=(not base)), color=color)
    ax.fill_between(x_sequence, mean_data - std_dev_data, mean_data + std_dev_data, alpha=0.08, color=color)

# colors = generate_distinct_colors(50)
colors = matplotlib.cm.tab10(range(20))


##################### Example usage of plotting stochastic methods: #####################
def run_get_sto_methods_data(out_dir):
    os.system("rm stochastic_methods.csv")
    for env_name in ["leduc_poker", "leduc_poker_dummy", "python_large_kuhn_poker", "kuhn_poker"]: #, "liars_dice", "oshi_zumo"]:
        for algorithm_name in ["SPDO"]:
            for num_iteration in [100, 500, 1000, 5000, 10000]:
                get_sto_methods_data(out_dir, 5, env_name, algorithm_name, num_iteration, False)
                get_sto_methods_data(out_dir, 5, env_name, algorithm_name, num_iteration, True)

        for algo_name in ["SADO"]:
            for ws in [False, True]:
                get_sto_methods_data("/root/data/results_sto", 5, env_name, algo_name, 500, is_warm_start=ws)
  
        
        get_sto_methods_data(out_dir, 5, env_name, "outcome_sampling_mccfr", 1, True, base=True)
        

def run_plot_sto_methods():
    num_seeds = 1  # Replace with the actual number of seeds
    env_name = "your_env_name"  # Replace with your environment name
    algorithm_name = "your_algorithm_name"  # Replace with your algorithm name
    num_iteration = 1000  # Replace with the number of iterations
    is_warm_start = True  # Replace with True or False as needed
    legend_done = False
    max_x_axis_dict = {"blotto_20":1e7, "blotto_25":1e7, "blotto_30":1e8, "blotto_40":1e8,"leduc_poker_10_card":2e8,
                       "kuhn_poker": 1e7, "python_large_kuhn_poker":2e7, "leduc_poker": 2e8, "leduc_poker_dummy": 1e9, "liars_dice": 1e9, "oshi_zumo":1e9}
    set_plot()
    fig, axs = plt.subplots(1, 5, figsize=(15 * 3, 8))

    for j, env_name in enumerate(["leduc_poker_10_card", "blotto_20", "blotto_25", "blotto_30","blotto_40" ]): #, "leduc_poker_dummy", "liars_dice", "oshi_zumo"]:
        ax = axs[j]
        i = 0

        for algorithm_name in ["PDO"]:

            for num_iteration in [50, 100]: #, 1000, 5000, 10000]:
                plot_mean_curve_with_error_bars(ax, 1, env_name, algorithm_name, num_iteration, False,False, colors[i], max_x_axis=max_x_axis_dict[env_name])
                i += 1
                plot_mean_curve_with_error_bars(ax, 1, env_name, algorithm_name, num_iteration, True,False, colors[i], max_x_axis=max_x_axis_dict[env_name])
                i += 1
        plot_mean_curve_with_error_bars(ax, 1, env_name, "XODO", 1, False,False, colors[i], max_x_axis=max_x_axis_dict[env_name])
        plot_mean_curve_with_error_bars(ax, 1, env_name, "SPDO", 5000, False,False, colors[i], max_x_axis=max_x_axis_dict[env_name])
        plot_mean_curve_with_error_bars(ax, 1, env_name, "SPDO", 1000, False,False, colors[i], max_x_axis=max_x_axis_dict[env_name])
        for algo_name in ["xdo", "outcome_sampling_mccfr", "lcfr"]:
            for ws in [False]:
                plot_mean_curve_with_error_bars(ax, 1, env_name, algo_name, 500, ws, False, colors[i], max_x_axis=max_x_axis_dict[env_name],base=True)
                i += 1
        
        #plot_mean_curve_with_error_bars(ax, 1, env_name, "outcome_sampling_mccfr", num_iteration, False, False,colors[i], max_x_axis=max_x_axis_dict[env_name], base=True)
        
        ax.set_yscale("log")
        ax.set_title(f'{game_name_dict[env_name]}')
        ax.set_xlabel('Number of nodes visited')
        if env_name == "python_large_kuhn_poker":
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.3), fancybox=True, shadow=True, ncol=7, fontsize=SMALL_SIZE*1.6)
        elif env_name == "kuhn_poker":
            ax.set_ylabel('Exploitability')
        else:
            ax.set_ylim([1e-1, 100])
        plt.legend(loc='lower right') #, bbox_to_anchor=(1.5, 1.5), fancybox=True, shadow=True, ncol=5)
        ax.grid(True)
    fig.savefig(f"results/stochastic_new.pdf", bbox_inches='tight', dpi=200, transparent=True)


if __name__ == '__main__':
    # run_plot_exact_methods()
    # run_get_support()
    run_plot_sto_methods()
    # run_get_sto_methods_data("/root/data/results")


    # seed_data = np.load("/root/data/results_sto/kuhn_poker_SPDO_500_wsTrue_2_exps.npy")
    # x_values = np.load("/root/data/results_sto/kuhn_poker_SPDO_500_wsTrue_2_times.npy")
    # interps = interp1d(x_values, seed_data, kind='linear')
    # x_sequence = np.linspace(x_values[0], x_values[-1], num=1000)  # You can adjust the num parameter for the desired number of points

    # # Initialize arrays to store mean and standard deviation
    # mean_data = np.ones(len(x_sequence)) * 1e10
    # std_dev_data = np.ones(len(x_sequence)) * 1e10

    # # Calculate the mean and standard deviation at each x-value
    # for i, x in enumerate(x_sequence):
    #     values_at_x = [interp(x) for interp in [interps]]
    #     mean_data[i] = np.mean(values_at_x) 
    #     std_dev_data[i] = np.std(values_at_x)
        

    # # Plot the curve with error bars
    # plt.cla()
    # plt.plot(x_sequence, mean_data)
    # plt.yscale("log")
    # plt.scatter(x_values, seed_data)
    # plt.savefig("test.png")