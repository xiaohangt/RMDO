from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import xdo
import time
import pyspiel
import argparse
import datetime
import numpy as np
from tqdm import tqdm

from dependencies.open_spiel.python import policy
from dependencies.open_spiel.python.algorithms import cfr
from dependencies.open_spiel.python.algorithms import psro_oracle
from dependencies.open_spiel.python.algorithms import best_response
from dependencies.open_spiel.python.algorithms import exploitability
from dependencies.open_spiel.python.algorithms import get_all_states
from dependencies.open_spiel.python.algorithms import fictitious_play
from dependencies.open_spiel.python.algorithms import expected_game_score
from dependencies.open_spiel.python.algorithms.cfr_cfr import OnlineTraining
from dependencies.open_spiel.python.algorithms import external_sampling_mccfr as external_mccfr

module_path = os.path.abspath(os.path.join(''))
if module_path not in sys.path:
    sys.path.append(module_path)


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--game_name', type=str, required=False, default="leduc_poker",
                        choices=["leduc_poker", "kuhn_poker", "leduc_poker_dummy", "oshi_zumo"])
    parser.add_argument('-a', '--algorithm', required=False, default="xodo", type=str,
                        choices=["psro", "cfr", "xfp", "xdo", "xodo_eps", "lcfr_plus", "xodo", "lcfr",
                                 "xodo_eps", "lcfr_plus_eps", "cfr_eps"])
    parser.add_argument('-m', '--meta_solver', required=False, default="lcfr_plus", type=str,
                        choices=["outcome", "external", "xfp", "lcfr_plus", "cfr"])
    parser.add_argument('-r', '--random_br', default="False", type=str, choices=["True", "False"])
    parser.add_argument('-o', '--old', default="False", type=str, choices=["True", "False"])
    parser.add_argument('-i', '--iterations', default=100000000, type=int)
    parser.add_argument('-x', '--xodo_iterations', default=50, type=int)

    # Parse arguments
    commandline_args = parser.parse_args()
    iterations = commandline_args.iterations
    xodo_iterations = commandline_args.xodo_iterations
    algorithm = commandline_args.algorithm
    game_name = commandline_args.game_name
    meta_solver = commandline_args.meta_solver
    random_max_br = commandline_args.random_br
    old_schedule = commandline_args.old
    if old_schedule:
        starting_br_conv_threshold = 2 ** 4
    else:
        starting_br_conv_threshold = 0.05
    extra_info = datetime.datetime.now().strftime("%I.%M.%S%p_%b-%d-%Y")

    # Set up game environment
    if game_name == "oshi_zumo":
        COINS = 4
        SIZE = 1
        HORIZON = 6
        game = pyspiel.load_game(game_name,
                                 {
                                     "coins": pyspiel.GameParameter(COINS),
                                     "size": pyspiel.GameParameter(SIZE),
                                     "horizon": pyspiel.GameParameter(HORIZON)
                                 })
        game = pyspiel.convert_to_turn_based(game)
        pretty_game_name = f'{game_name}c{COINS}s{SIZE}h{HORIZON}'
    else:
        game = pyspiel.load_game(game_name)
        pretty_game_name = game_name


    def run(solver, iterations):
        checkpoint_period = 5
        if algorithm == 'external_mccfr':
            checkpoint_period = 15000
        start_time = time.time()
        times = []
        exps = []
        episodes = []
        cfr_infostates = []
        if algorithm == 'psro':
            size_of_game = len(get_all_states.get_all_states(game, include_chance_states=True))
            num_infostates_expanded = size_of_game * 2
        for i in range(iterations):
            if algorithm in ['cfr', 'lcfr_plus', 'lcfr']:
                solver.evaluate_and_update_policy()
            else:
                solver.iteration()
            if i % checkpoint_period == 0:
                if algorithm in ['cfr', 'lcfr_plus', 'lcfr']:
                    average_policy = solver.average_policy()
                elif algorithm == 'external_mccfr':
                    try:
                        average_policy = solver.average_policy()
                    except AttributeError:
                        print('making tabular policy from callable')
                        average_policy = policy.tabular_policy_from_callable(game, solver.callable_avg_policy())
                        print('done')
                elif algorithm == 'xfp':
                    average_policy = solver.average_policy()
                elif algorithm == 'psro':
                    average_policy = solver._current_policy
                    num_infostates_expanded += size_of_game * 2
                else:
                    raise ValueError(f"Unknown algorithm name: {algorithm}")
                print('beginning exploitability calculation')
                conv = exploitability.exploitability(game, average_policy)
                print("Iteration {} exploitability {}".format(i, conv))
                elapsed_time = time.time() - start_time
                print(elapsed_time)
                times.append(elapsed_time)
                exps.append(conv)
                episodes.append(i)
                save_prefix = './results/' + algorithm + '_' + pretty_game_name + extra_info
                ensure_dir(save_prefix)
                print(f"saving to: {save_prefix + '_times.npy'}")
                np.save(save_prefix + '_times', np.array(times))
                print(f"saving to: {save_prefix + '_exps.npy'}")
                np.save(save_prefix + '_exps', np.array(exps))
                print(f"saving to: {save_prefix + '_episodes.npy'}")
                np.save(save_prefix + '_episodes', np.array(episodes))
                if algorithm in ['cfr', 'external_mccfr', 'lcfr_plus', 'lcfr', 'xfp']:
                    cfr_infostates.append(solver.num_infostates_expanded * (i + 1))
                    print("Num infostates expanded (mil): ", solver.num_infostates_expanded * (i + 1) / 1e6)
                    print(f"saving to: {save_prefix + '_infostates.npy'}")
                    np.save(save_prefix + '_infostates', np.array(cfr_infostates))
                elif algorithm in ['psro']:
                    cfr_infostates.append(num_infostates_expanded)
                    print("Num infostates expanded (mil): ", num_infostates_expanded / 1e6)
                    print(f"saving to: {save_prefix + '_infostates.npy'}")
                    np.save(save_prefix + '_infostates', np.array(cfr_infostates))


    if algorithm == 'cfr':
        solver = cfr.CFRSolver(game)
        run(solver, iterations)
    elif algorithm == 'lcfr':
        solver = cfr.LCFRSolver(game)
        run(solver, iterations)
    elif algorithm == 'lcfr_plus':
        # In the implementation of OpSpiel, CFRPlusSolver is LCFR+
        solver = cfr.CFRPlusSolver(game)
        run(solver, iterations)
    elif algorithm == 'external_mccfr':
        solver = external_mccfr.ExternalSamplingSolver(game)
        run(solver, iterations)
    elif algorithm == 'xfp':
        solver = fictitious_play.XFPSolver(game)
        run(solver, iterations)
    elif algorithm == 'xdo':
        # brs = []
        xdo_iterations = 50
        if meta_solver == 'external':
            xdo_iterations = 200
        elif meta_solver == 'xfp':
            xdo_iterations = 5
        inner_thresh = .35

        info_test = []

        size_of_game = len(get_all_states.get_all_states(game, include_chance_states=True))
        uniform = policy.UniformRandomPolicy(game)
        br_list = []
        for pid in [0, 1]:
            br_list.append([best_response.BestResponsePolicy(game, pid, uniform)])
        ###############################
        start_time = time.time()
        cum_brtime = 0
        xdo_times = []
        xdo_exps = []
        xdo_episodes = []
        xdo_infostates = []
        xdo_brtimes = []
        xdo_outers = []
        xdo_res_game_size = []
        br_conv_threshold = starting_br_conv_threshold

        outer_loop = 0
        episode = 0
        # num_infostates = 0
        nash_support_reached = False
        prev_cum_num_infostates = 0
        num_infostates = 0
        for i in range(iterations):
            print('Iteration: ', i)
            restricted_game = xdo.WrappedGame(game, br_list)
            if meta_solver == 'external':
                solver = external_mccfr.ExternalSamplingSolver(restricted_game, external_mccfr.AverageType.SIMPLE)
            elif meta_solver == 'xfp':
                solver = fictitious_play.XFPSolver(restricted_game)
            elif meta_solver == 'lcfr_plus':
                solver = cfr.CFRPlusSolver(restricted_game)
            elif meta_solver == 'cfr':
                solver = cfr.CFRSolver(restricted_game)

            double_next_time = False
            while True:
                for inner_loop_iter in tqdm(range(int(xdo_iterations))):
                    episode += 1
                    if meta_solver in ['cfr', 'lcfr_plus']:
                        solver.evaluate_and_update_policy()
                    else:
                        solver.iteration()
                restricted_exploitability = exploitability.nash_conv(restricted_game, solver.average_policy()) / 2
                print(f'inner loop (restricted) exploitability: {restricted_exploitability}')
                # if nash_support_reached:
                # total_exploitability = exploitability.nash_conv(())
                # total_exploitability = 0
                ############################
                # make BRs and save data
                if old_schedule:
                    if restricted_exploitability < br_conv_threshold and i > 0:
                        br_conv_threshold /= 2
                        break
                print('making full policy')
                full_policy = xdo.LazyTabularPolicy(restricted_game, game, solver.average_policy())
                print('making new brs')
                brtime_start = time.time()
                avg_exploitability = 0
                new_brs = []
                for pid in [0, 1]:
                    new_br = best_response.BestResponsePolicy(game, pid, full_policy, add_noise=random_max_br)
                    new_brs.append(new_br)
                    avg_exploitability += new_br.value(game.new_initial_state()) / 2
                brtime_end = time.time()
                cum_brtime += brtime_end - brtime_start
                print(f'avg (full) exploitability: {avg_exploitability}')

                elapsed_time = time.time() - start_time
                print('Total elapsed time: ', elapsed_time)
                save_prefix = './results/' + algorithm + '_' + meta_solver + '_' + game_name + str(
                    old_schedule) + extra_info
                if meta_solver in ['cfr', 'lcfr_plus', 'external', 'xfp']:
                    # num_infostates = prev_cum_num_infostates + solver.num_infostates_expanded
                    num_infostates += solver.num_infostates_expanded * xdo_iterations
                    xdo_infostates.append(num_infostates)
                    print('Num infostates expanded (mil): ', num_infostates / 1e6)
                    print(f"saving to: {save_prefix + '_infostates.npy'}")
                    np.save(save_prefix + '_infostates', np.array(xdo_infostates))
                else:
                    num_infostates = 0
                restricted_game_size = len(get_all_states.get_all_states(restricted_game, include_chance_states=True))
                xdo_times.append(elapsed_time)
                xdo_exps.append(avg_exploitability)
                xdo_episodes.append(episode)
                xdo_brtimes.append(cum_brtime)
                xdo_outers.append(outer_loop)
                xdo_res_game_size.append(restricted_game_size)
                print(f'outer loop: {outer_loop}, restricted game size: {restricted_game_size}')
                ensure_dir(save_prefix)
                print(f"saving to: {save_prefix + '_times.npy'}")
                np.save(save_prefix + '_times', np.array(xdo_times))
                print(f"saving to: {save_prefix + '_exps.npy'}")
                np.save(save_prefix + '_exps', np.array(xdo_exps))
                print(f"saving to: {save_prefix + '_episodes.npy'}")
                np.save(save_prefix + '_episodes', np.array(xdo_episodes))
                print(f"saving to: {save_prefix + '_brtimes.npy'}")
                np.save(save_prefix + '_brtimes', np.array(xdo_brtimes))
                print(f"saving to: {save_prefix + '_outers.npy'}")
                np.save(save_prefix + '_outers', np.array(xdo_outers))
                print(f"saving to: {save_prefix + '_res_game_size.npy'}")
                np.save(save_prefix + '_res_game_size', np.array(xdo_res_game_size))
                ###############################
                if avg_exploitability != restricted_exploitability:
                    if abs(avg_exploitability - restricted_exploitability) < 0.00001:
                        print(
                            "EXPLOITABILITY IS DIFFERENT BUT BELOW ALLOWABLE TOLERANCE! THIS IS WRONG IF IT CONTINUES HAPPENING")
                        continue
                    print("no support :((((((((((((((((((((((")
                    if restricted_exploitability > inner_thresh:
                        xdo_iterations = int(xdo_iterations * 1.4)
                        continue
                    else:
                        for pid in [0, 1]:
                            br_list[pid].append(new_brs[pid])
                            # num_infostates += len(new_brs[pid].cache_value)
                            # print(len(new_brs[pid].cache_value))
                            num_infostates += size_of_game
                        outer_loop += 1
                        print(f"adding brs: brs explore {size_of_game} infostates each")
                        prev_cum_num_infostates = num_infostates
                        inner_thresh *= 0.98
                        print(f"inner loop exploitability threshold set to {inner_thresh}")
                        break
                else:
                    print("has full support!!!!!!!!!!!!!!!")
                    xdo_iterations = int(xdo_iterations * 1.02)

    elif algorithm == 'xodo':
        size_of_game = len(get_all_states.get_all_states(game, include_chance_states=True))
        brs, info_test = [], []
        br_actions = {}
        start_time = time.time()
        uniform = policy.UniformRandomPolicy(game)
        for pid in range(2):
            br = best_response.BestResponsePolicy(game, pid, uniform, add_noise=random_max_br)
            _ = br.value(game.new_initial_state())
            for key, action in br.cache_best_response_action.items():
                br_actions[key] = [action]
            brs.append(br)
        new_br = True
        br_list = [[brs[0]], [brs[1]]]
        restricted_game = xdo.WrappedGame(game, br_list)
        if meta_solver == 'external':
            solver = external_mccfr.ExternalSamplingSolver(restricted_game, external_mccfr.AverageType.SIMPLE)
        elif meta_solver == 'xfp':
            solver = fictitious_play.XFPSolver(restricted_game)
        elif meta_solver == 'lcfr_plus':
            solver = cfr.CFRPlusSolver(restricted_game)
        elif meta_solver == 'cfr':
            solver = cfr.CFRSolver(restricted_game)

        xdo_times = []
        xdo_exps = []
        xdo_exps_brtimes = []
        xdo_episodes = []
        xdo_infostates = []
        episode = 0
        num_infostates = 0
        num_infostates_prev_iteration = 0
        for i in range(iterations):
            print('Iteration: ', i)
            full_policy = xdo.LazyTabularPolicy(restricted_game, game, solver.average_policy())
            conv = exploitability.exploitability(game, full_policy)
            save_prefix = './results/' + algorithm + '_' + meta_solver + '_' + game_name + extra_info

            if (new_br and i > 0) or i % 5 == 0:
                check_start = time.time()
                start_time += (time.time() - check_start)
                print("Iteration {} exploitability {}".format(i, conv))
                elapsed_time = time.time() - start_time
                print('Total elapsed time: ', elapsed_time)
                num_infostates += solver.num_infostates_expanded * xodo_iterations
                num_infostates_prev_iteration = solver.num_infostates_expanded
                print('Num infostates expanded (mil): ', num_infostates / 1e6)
                xdo_times.append(elapsed_time)
                xdo_exps.append(conv)
                xdo_episodes.append(episode)
                xdo_infostates.append(num_infostates)
                ensure_dir(save_prefix)
                print(f"saving to: {save_prefix + '_times.npy'}")
                np.save(save_prefix + '_times', np.array(xdo_times))
                print(f"saving to: {save_prefix + '_exps.npy'}")
                np.save(save_prefix + '_exps', np.array(xdo_exps))
                print(f"saving to: {save_prefix + '_episodes.npy'}")
                np.save(save_prefix + '_episodes', np.array(xdo_episodes))
                print(f"saving to: {save_prefix + '_infostates.npy'}")
                np.save(save_prefix + '_infostates', np.array(xdo_infostates))

            if new_br and i > 0:
                num_infostates_prev_iteration = 0
                restricted_game = xdo.WrappedGame(game, br_list)
                if meta_solver == 'external':
                    solver = external_mccfr.ExternalSamplingSolver(restricted_game, external_mccfr.AverageType.SIMPLE)
                elif meta_solver == 'xfp':
                    solver = fictitious_play.XFPSolver(restricted_game)
                elif meta_solver == 'lcfr_plus':
                    solver = cfr.CFRPlusSolver(restricted_game)
                else:
                    solver = cfr.CFRSolver(restricted_game)
            for _ in tqdm(range(xodo_iterations)):
                if meta_solver in ['cfr', 'lcfr_plus']:
                    solver.evaluate_and_update_policy()
                else:
                    solver.iteration()
            new_brs = []
            new_br = False
            full_policy = xdo.LazyTabularPolicy(restricted_game, game, solver.average_policy())

            for pid in range(2):
                br = best_response.BestResponsePolicy(game, pid, full_policy, add_noise=random_max_br)
                _ = br.value(game.new_initial_state())
                # Get best response action for unvisited states
                for infostate in set(br.infosets) - set(br.cache_best_response_action):
                    br.best_response_action(infostate)
                for key, action in br.cache_best_response_action.items():
                    if key in br_actions:
                        if action not in br_actions[key]:
                            br_actions[key].append(action)
                            new_br = True
                    else:
                        br_actions[key] = [action]
                        new_br = True
                new_brs.append(br)
                num_infostates += size_of_game
            xdo_exps_brtimes.append(conv)
            print(f"saving to: {save_prefix + '_brtimes.npy'}")
            np.save(save_prefix + '_brtimes.npy', np.array(xdo_exps_brtimes))
            if new_br:
                for pid in [0, 1]:
                    br_list[pid].append(new_brs[pid])
    elif 'eps' in algorithm:
        xoso_id = 0
        oppo_id = (xoso_id + 1) % 2
        info_test = []
        br_actions, oppo_br_actions = {}, {}
        br_list, oppo_br_list = [[]] * 2, [[]] * 2
        start_time = time.time()
        uniform = policy.UniformRandomPolicy(game)
        first_action = policy.FirstActionPolicy(game)
        always_bet = policy.BetPolicy(game)

        # Get BR to the opponent's strategy
        br = best_response.BestResponsePolicy(game, xoso_id, uniform, add_noise=random_max_br)
        _ = br.value(game.new_initial_state())
        for key, action in br.cache_best_response_action.items():
            br_actions[key] = [action]
        new_br = True
        br_list[xoso_id].append(br)

        restricted_game = xdo.WrappedGame(game, br_list, br_id=xoso_id)
        if algorithm == 'xodo_eps':
            own_agent = cfr.CFRSolver(restricted_game)
        elif algorithm == 'lcfr_plus_eps':
            own_agent = cfr.CFRPlusSolver(game)
        elif algorithm == 'cfr_eps':
            own_agent = cfr.CFRSolver(game)


        opponent = cfr.CFRPlusSolver(game)
        solvers = [own_agent, opponent] if xoso_id == 0 else [opponent, own_agent]
        online_training = OnlineTraining(game, solvers, algorithm == 'lcfr_plus_eps', xoso_id)

        eps_list = []
        episode = 0
        num_infostates = 0
        num_infostates_prev_iteration = 0

        for i in range(iterations):
            print('Iteration: ', i)
            # If they are new best responses, reset
            if new_br and i > 0:
                restricted_game = xdo.WrappedGame(game, br_list, br_id=xoso_id)
                if algorithm == 'xodo_eps':
                    solvers[xoso_id] = cfr.CFRPlusSolver(restricted_game)
                elif algorithm == 'lcfr_plus_eps':
                    own_agent = cfr.CFRPlusSolver(game)
                elif algorithm == 'cfr_eps':
                    own_agent = cfr.CFRSolver(game)
                online_training = OnlineTraining(game, solvers, algorithm == 'lcfr_plus_eps', xoso_id)

            for _ in tqdm(range(1)):
                eps = online_training.evaluate_and_update_policy()
            new_brs = []
            new_br = False

            oppo_full_policy = always_bet
            full_policy = xdo.LazyTabularPolicy(restricted_game, game, solvers[xoso_id].average_policy())
            policies = [full_policy, oppo_full_policy] if xoso_id == 0 else [oppo_full_policy, full_policy]
            eps = expected_game_score.policy_value(game.new_initial_state(), policies)

            print("Exploitation: ", eps[xoso_id])
            eps_list.append(eps[xoso_id])
            save_prefix = './results/exploitation_'
            np.save(save_prefix + f'{algorithm}_eps_list.npy', np.array(eps_list))

            # Get BR
            br = best_response.BestResponsePolicy(game, xoso_id, oppo_full_policy, add_noise=random_max_br)
            _ = br.value(game.new_initial_state())
            # Get best response action for unvisited states
            for infostate in set(br.infosets) - set(br.cache_best_response_action):
                br.best_response_action(infostate)
            for key, action in br.cache_best_response_action.items():
                if key in br_actions:
                    if action not in br_actions[key]:
                        br_actions[key].append(action)
                        new_br = True
                else:
                    br_actions[key] = [action]
                    new_br = True
            if new_br:
                br_list[xoso_id].append(br)

    elif algorithm == 'psro':
        psro_num_eval_episodes = 1  # oshi zumo is deterministic
        brs = []
        info_test = []
        for i in range(2):
            br = best_response.CPPBestResponsePolicy(game, i, policy.UniformRandomPolicy(game))
            brs.append(br)
        br_list = [[brs[0]], [1], [brs[1]], [1]]
        solver = psro_oracle.PSRO(game, br_list, num_episodes=psro_num_eval_episodes)
        run(solver, iterations)
