from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import datetime
import time
import pickle
import pdb
import numpy as np
import pyspiel
import blotto
import large_kuhn_poker
from utils import get_support

from dependencies.open_spiel.python.algorithms import cfr
from dependencies.open_spiel.python.algorithms import discounted_cfr
from dependencies.open_spiel.python.algorithms import cfr_br_actions
from dependencies.open_spiel.python.algorithms import exploitability
from dependencies.open_spiel.python.algorithms import exploitability_br_actions
from dependencies.open_spiel.python.algorithms import fictitious_play
from dependencies.open_spiel.python.algorithms import outcome_sampling_mccfr
from dependencies.open_spiel.python.algorithms import psro_oracle


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def _full_best_response_policy(br_infoset_dict):
    """Turns a dictionary of best response action selections into a full policy.
  Args:
    br_infoset_dict: A dictionary mapping information state to a best response
      action.
  Returns:
    A function `state` -> list of (action, prob)
  """

    def wrap(state):
        infostate_key = state.information_state_string(state.current_player())
        br_action = br_infoset_dict[infostate_key]
        ap_list = []
        for action in state.legal_actions():
            ap_list.append((action, 1.0 if action == br_action else 0.0))
        return ap_list

    return wrap


def _policy_dict_at_state(callable_policy, state):
    """Turns a policy function into a dictionary at a specific state.
  Args:
    callable_policy: A function from `state` -> lis of (action, prob),
    state: the specific state to extract the policy from.
  Returns:
    A dictionary of action -> prob at this state.
  """

    infostate_policy_list = callable_policy(state)
    infostate_policy = {}
    for ap in infostate_policy_list:
        infostate_policy[ap[0]] = ap[1]
    return infostate_policy


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # dxdo is xdo with meta_solver cfr+
    parser.add_argument('--algorithm', type=str, choices=["psro", "cfr", "xfp", "xdo", "dxdo", "cfr_plus", "lcfr", "outcome_sampling_mccfr"], default="outcome_sampling_mccfr")
    parser.add_argument('--game_name', type=str, required=False, default="kuhn_poker",
                        choices=["leduc_poker", "kuhn_poker", "leduc_poker_dummy", "oshi_zumo",  "liars_dice",
                                 "goofspiel", "havannah", "blotto", "python_large_kuhn_poker"])
    parser.add_argument('--display', type=bool, default=False)
    parser.add_argument('--seed', type=int, required=False, default=0)
    parser.add_argument('--out_dir', type=str, required=False, default="results")  # output folder
    commandline_args = parser.parse_args()

    seed = commandline_args.seed
    algorithm = commandline_args.algorithm
    game_name = commandline_args.game_name
    display = commandline_args.display
    out_dir = commandline_args.out_dir

    extra_info = datetime.datetime.now().strftime("%I.%M.%S%p_%b-%d-%Y")
    np.random.seed(commandline_args.seed)

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
    elif game_name == "goofspiel":
        game = pyspiel.load_game(game_name, {"players": pyspiel.GameParameter(2)})
        game = pyspiel.convert_to_turn_based(game)
    elif game_name == "havannah":
        game = pyspiel.load_game(game_name)
    elif game_name == "blotto":
        game = blotto.BlottoGame()
    elif game_name == "python_large_kuhn_poker":
        game = large_kuhn_poker.KuhnPokerGame()
    else:
        game = pyspiel.load_game(game_name, {"players": pyspiel.GameParameter(2)})

    starting_br_conv_threshold = 2 ** 4
    iterations = 10000000000 if algorithm == "outcome_sampling_mccfr" else 1000000000
    xdo_iterations = 200000
    random_max_br = False
    supports = []


    def run(solver, iterations):
        start_time = time.time()
        times = []
        exps = []
        episodes = []
        cfr_infostates = []
        for i in range(iterations):
            # display_policy(solver.average_policy())
            if algorithm == 'cfr' or algorithm == 'cfr_plus' or algorithm == "lcfr":
                solver.evaluate_and_update_policy()
            else:
                solver.iteration()
            data_collect_freq = 5000 if "mccfr" in algorithm else 10
            if i % data_collect_freq == 0:
                if 'cfr' in algorithm:
                    average_policy = solver.average_policy()
                elif algorithm == 'xfp':
                    average_policy = solver.average_policy()
                elif algorithm == 'psro':
                    average_policy = solver._current_policy
                else:
                    raise ValueError(f"Unknown algorithm name: {algorithm}")

                conv = exploitability.exploitability(game, average_policy)
                print("Iteration {} exploitability {}".format(i, conv))
                elapsed_time = time.time() - start_time
                print("Time:", elapsed_time)
                times.append(elapsed_time)
                exps.append(conv)
                episodes.append(i)
                supports.append(get_support(average_policy, game))

                save_prefix = f'{out_dir}/' + algorithm + '_' + game_name + f"_{seed}_"
                ensure_dir(save_prefix)
                print(f"saving to: {save_prefix + '_times.npy'}")
                # pdb.set_trace()
                np.save(save_prefix + '_times', np.array(times))
                print(f"saving to: {save_prefix + '_exps.npy'}")
                np.save(save_prefix + '_exps', np.array(exps))
                print(f"saving to: {save_prefix + '_episodes.npy'}")
                np.save(save_prefix + '_episodes', np.array(episodes))
                cfr_infostates.append(solver.num_infostates_expanded)
                print("Num infostates expanded (mil): ", solver.num_infostates_expanded / 1e6)
                print(f"saving to: {save_prefix + '_infostates.npy'}")
                # pickle.dump(cfr_infostates, open(save_prefix + '_infostates.pkl',"wb"))
                # test = pickle.load(open(save_prefix + '_infostates.pkl','rb'))
                # print(test)
                np.save(save_prefix + '_infostates', np.array(cfr_infostates))

                print(f"saving to: {save_prefix + '_infos.npy'}")
                np.save(save_prefix + '_infos', np.array(supports))

    if algorithm == 'cfr':
        solver = cfr.CFRSolver(game)
        run(solver, iterations)
    elif algorithm == 'cfr_plus':
        solver = cfr.CFRPlusSolver(game)
        run(solver, iterations)
    elif algorithm == "lcfr":
        solver = discounted_cfr.LCFRSolver(game)
        run(solver, iterations)
    elif algorithm == 'outcome_sampling_mccfr':
        solver = outcome_sampling_mccfr.OutcomeSamplingSolver(game)
        run(solver, iterations)
    elif algorithm == 'xfp':
        solver = fictitious_play.XFPSolver(game)
        run(solver, iterations)
    elif "xdo" in algorithm:
        episode = 0
        num_infostates = 0
        # size_of_game = len(get_all_states.get_all_states(game, include_chance_states=True))

        brs = []
        info_test = []
        for i in range(2):
            br_info = exploitability.best_response(game, cfr.CFRSolver(game).average_policy(), i)
            full_br_policy = _full_best_response_policy(br_info["best_response_action"])
            info_sets = br_info['info_sets']
            info_test.append(info_sets)
            brs.append(full_br_policy)
            num_infostates += br_info["expanded_infostates"]

        br_list = [brs]
        start_time = time.time()
        xdo_times = []
        xdo_exps = []
        xdo_episodes = []
        xdo_infostates = []

        br_conv_threshold = starting_br_conv_threshold

        for i in range(iterations):
            if algorithm == "dxdo":
                cfr_br_solver = cfr_br_actions.CFRPlusSolver(game, br_list)
            else:
                cfr_br_solver = cfr_br_actions.CFRSolver(game, br_list)

            for j in range(xdo_iterations):
                cfr_br_solver.evaluate_and_update_policy()
                episode += 1
                if j % 50 == 0:
                    br_list_conv = exploitability_br_actions.exploitability(game, br_list,
                                                                            cfr_br_solver.average_policy())
                    num_infostates += cfr_br_solver.num_infostates_expanded / j # add the complexity of computing exps in restricted games
                    if display:
                        print("Br list conv: ", br_list_conv, j)
                    if br_list_conv < br_conv_threshold:
                        break

            conv = exploitability.exploitability(game, cfr_br_solver.average_policy())
            print("Iteration {} exploitability {}".format(i, conv))
            if conv < br_conv_threshold:
                br_conv_threshold /= 2
                if display:
                    print("new br threshold: ", br_conv_threshold)

            elapsed_time = time.time() - start_time
            if display:
                print('Total elapsed time: ', elapsed_time)
            num_infostates += cfr_br_solver.num_infostates_expanded
            if display:
                print('Num infostates expanded (mil): ', num_infostates / 1e6)
            xdo_times.append(elapsed_time)
            xdo_exps.append(conv)
            xdo_episodes.append(episode)
            xdo_infostates.append(num_infostates)
            supports.append(get_support(cfr_br_solver.average_policy(), game))

            brs = []
            for i in range(2):
                if random_max_br:
                    br_info = exploitability.best_response_random_max_br(game, cfr_br_solver.average_policy(), i)
                else:
                    br_info = exploitability.best_response(game, cfr_br_solver.average_policy(), i)
                full_br_policy = _full_best_response_policy(br_info["best_response_action"])
                brs.append(full_br_policy)
                num_infostates += br_info["expanded_infostates"]

            br_list.append(brs)
            save_prefix = f'{out_dir}/' + algorithm + '_' + game_name + f"_{seed}_"
            ensure_dir(save_prefix)
            if time.time() - start_time < 258000:
                np.save(save_prefix + '_times', np.array(xdo_times))
                np.save(save_prefix + '_exps', np.array(xdo_exps))
                np.save(save_prefix + '_infostates', np.array(xdo_infostates))
                np.save(save_prefix + '_infos', np.array(supports))
    elif algorithm == 'psro':
        brs = []
        info_test = []
        for i in range(2):
            br_info = exploitability.best_response(game, cfr.CFRSolver(game).average_policy(), i)
            full_br_policy = _full_best_response_policy(br_info["best_response_action"])
            info_sets = br_info['info_sets']
            info_test.append(info_sets)
            brs.append(full_br_policy)
        br_list = [[brs[0]], [1], [brs[1]], [1]]
        solver = psro_oracle.PSRO(game, br_list, num_episodes=2000)
        run(solver, iterations)
