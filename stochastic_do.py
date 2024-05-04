import os
import gc
import sys
import time
import argparse
import datetime
import pdb
import numpy as np
import pyspiel



from tqdm import tqdm
from copy import deepcopy
import blotto
import blotto_20
import blotto_25
import blotto_30
import blotto_40
import kuhn_poker_dummy

import large_kuhn_poker
from dependencies.open_spiel.python.algorithms import get_all_states
from dependencies.open_spiel.python.games import tic_tac_toe
from dependencies.open_spiel.python.games import kuhn_poker
from dependencies.open_spiel.python import policy
from dependencies.open_spiel.python.algorithms import cfr
from dependencies.open_spiel.python.algorithms import best_response
from dependencies.open_spiel.python.algorithms import exploitability
from dependencies.open_spiel.python.algorithms import outcome_sampling_mccfr as outcome_mccfr

from approximate_best_response import MCBR, mc_exploitability

module_path = os.path.abspath(os.path.join(''))
if module_path not in sys.path:
    sys.path.append(module_path)

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


class WrappedOSMCCFRSolver(outcome_mccfr.OutcomeSamplingSolver):
    def __init__(self, game):
        super().__init__(game)

    def evaluate_and_update_policy(self):
        self.iteration()


class MetaState:
    def __init__(self, game, state, brs, br_id=None):
        self.state = state
        self.brs = brs
        self.br_id = br_id
        self.game = game
        self.child_converged = None

    def legal_actions(self, player=None):
        if player is not None and player != self.state.current_player():
            # assumption
            return []
        if self.state.history_str() in self.game._legal_actions_cache:
            return self.game._legal_actions_cache[self.state.history_str()]
        if self.brs == 'empty' and self.state.current_player() == self.br_id:
            return self.state.legal_actions()

        if (self.br_id is None and self.state.current_player() in [0, 1]) or \
                (self.br_id is not None and self.state.current_player() == self.br_id):
            legal_actions = set()
            for br in self.brs[self.state.current_player()]:
                try:
                    br.update_legal_actions(self.state)
                    legal_actions.add(br.best_response_action(self.state.information_state_string()))
                except:
                    legal_actions.add(br.best_response_action(self.state.information_state_string()))

            ans = list(legal_actions)
            self.game._legal_actions_cache[self.state.history_str()] = ans
            return ans
        else:
            return self.state.legal_actions()

    def child(self, a):
        return MetaState(self.game, self.state.child(a), self.brs, br_id=self.br_id)

    def __getattr__(self, attr):
        assert attr != 'new_initial_state'
        return self.state.__getattribute__(attr)

    def clone(self):
        return MetaState(self.game, self.state.clone(), self.brs, br_id=self.br_id)


class MetaGame:
    def __init__(self, game, brs, br_id=None, it=0):
        self.game = game
        self.get_type = game.get_type
        self.num_players = game.num_players
        self.it = it

        self.brs = brs
        self.br_id = br_id
        self._legal_actions_cache = dict()

    def new_initial_state(self):
        return MetaState(self, self.game.new_initial_state(), self.brs, br_id=self.br_id)

    def __getattr__(self, attr):
        assert attr != 'new_initial_state'
        return self.game.__getattribute__(attr)
    


class ExpandTabularPolicy:
    def __init__(self, p):
        tabular_p = p.to_tabular()
        self.state_lookup = deepcopy(tabular_p.state_lookup)
        self.action_probability_array = deepcopy(tabular_p.action_probability_array)

    def action_probabilities(self, state):
        state_str = state.information_state_string()
        if state_str in self.state_lookup:
            probability = self.action_probability_array[self.state_lookup[state_str]]
            return {action: probability[action] for action in state.legal_actions()}
        return {a: 1 / len(state.legal_actions()) for a in state.legal_actions()}


def merge_two_policies(previous_policy, current_policy, iter, prev_iter):
    """Merges two policies(one is the other's subset) into single joint policy for fixed player.

    Missing states are filled with a valid uniform policy.

    Args:
      current_policy: avg policy of current window
      previous_policy: avg policy of previous windows
      game: The game corresponding to the resulting TabularPolicy.
      iter: Useful in uniform average


    Returns:
      merged_policy: A TabularPolicy with each player i's policy taken from the
        ith joint_policy.
    """
    merged_policy = current_policy
    for p_state in current_policy.state_lookup.keys():
        to_index = merged_policy.state_lookup[p_state]
        # Only copy if the state exists, otherwise fall back onto uniform.
        current_prob_array = current_policy.action_probability_array[current_policy.state_lookup[p_state]]
        if p_state in previous_policy.state_lookup:
            previous_prob_array = previous_policy.action_probability_array[
                previous_policy.state_lookup[p_state]]
            merged_policy.action_probability_array[to_index] = (previous_prob_array * prev_iter + current_prob_array * (
                    iter - prev_iter)) / iter
        else:
            merged_policy.action_probability_array[to_index] = current_prob_array
        merged_policy.action_probability_array[to_index] /= np.sum(
            merged_policy.action_probability_array[to_index])
    return merged_policy


def display_policy(current_policy):
    for p_state in current_policy.state_lookup.keys():
        print(p_state, current_policy.action_probability_array[current_policy.state_lookup[p_state]])
    return



class SPDO:
    def __init__(self, algorithm: str, game_name: str, meta_iterations: int, data_collect_frequency: int, is_warm_start: bool, is_mcbr=False, out_dir="results"):
        self.algorithm = algorithm
        self.game_name = game_name
        self.meta_iterations = meta_iterations
        self.data_collect_frequency = data_collect_frequency
        self.is_warm_start = is_warm_start
        self.br_actions = {}
        self.state_str_to_legal_actions = {}
        self.is_mcbr = is_mcbr
        self.out_dir = out_dir
        self.game_size = 0
    def reset_game(self):
        # Set up game environment
        if self.game_name == "oshi_zumo":
            COINS = 4
            SIZE = 1
            HORIZON = 6
            game = pyspiel.load_game(self.game_name,
                                     {
                                         "coins": pyspiel.GameParameter(COINS),
                                         "size": pyspiel.GameParameter(SIZE),
                                         "horizon": pyspiel.GameParameter(HORIZON)
                                     })
            game = pyspiel.convert_to_turn_based(game)
        elif game_name == "goofspiel":
            game = pyspiel.load_game(game_name, {"players": pyspiel.GameParameter(2)})
            game = pyspiel.convert_to_turn_based(game)
        elif game_name == "phantom_ttt":
            game = pyspiel.load_game(game_name)
        elif game_name == "blotto":
            game = blotto.BlottoGame()
        elif game_name == "blotto_20":
            game = blotto_20.BlottoGame()
        elif game_name == "blotto_25":
            game = blotto_25.BlottoGame()   
        elif game_name == "blotto_30":
            game = blotto_30.BlottoGame()
        elif game_name == "blotto_40":
            game = blotto_40.BlottoGame()
        elif game_name == "python_large_kuhn_poker":
            game = large_kuhn_poker.KuhnPokerGame()
        else:
            game = pyspiel.load_game(self.game_name)
        
        self.game_size = len(get_all_states.get_all_states(game, include_chance_states=True))
        return game

    def warm_star_init(self, restricted_game, old_solver, new_solver, br_list):
        ws_actions_cnt = 0
        ws_real_cnt=0
        for key, legal_action_value in old_solver.legal_actions_dict.items():
            cur_player, old_legal_actions, state = legal_action_value
            
            #modified
            #state_old = state.clone()
            state.state.game = deepcopy(restricted_game.game)
            

            
            legal_actions = set()
            for br in br_list[cur_player]:
                legal_actions.add(br.best_response_action(key))
            legal_actions = list(legal_actions)

            new_solver._infostates[key] = [
                np.ones(len(legal_actions), dtype=np.float64) / 1e6, # cum regret
                np.ones(len(legal_actions), dtype=np.float64) / 1e6, # cum strategy
            ]

            value = old_solver._infostates[key]
            count = 0
            for i, action in enumerate(legal_actions):
                #modified
                if action in old_legal_actions:
                    if not state.child(action).is_terminal():
                        new_solver._infostates[key][0][i] = value[0][count] / self.meta_iterations
                        ws_real_cnt+=1
                    count += 1
                    #if not state_old.child(action).is_terminal():
                        #ws_actions_cnt+=1
        #if ws_real_cnt!=ws_actions_cnt:
            #print(f"has some differences: new {ws_real_cnt};old {ws_actions_cnt}")
            #sys.exit(0)
        
    def reset_meta_solver(self, restricted_game, br_list=None, old_solver=None):
        meta_solver = WrappedOSMCCFRSolver(restricted_game)
        if (not old_solver) or (not self.is_warm_start):
            return meta_solver
        # modified
        self.warm_star_init(restricted_game, old_solver, meta_solver, br_list)
        return meta_solver
    
    def get_best_response(self, game, policy):
        brs = []
        new_br = False
        num_infostates_expanded = 0
        for pid in range(2):
            if not self.is_mcbr:
                br = best_response.BestResponsePolicy(game, pid, policy)
                br.expanded_infostates = 0
                _ = br.value(game.new_initial_state())
                # Get the best response action for unvisited states
                for infostate in set(br.infosets) - set(br.cache_best_response_action):
                    br.best_response_action(infostate)
            else:
                br = MCBR(game, pid, policy)
                br.value(self.game_size)

            for key, action in br.cache_best_response_action.items():
                if key in self.br_actions:
                    if action not in self.br_actions[key]:
                        self.br_actions[key].append(action)
                        new_br = True
                else:
                    self.br_actions[key] = [action]
                    new_br = True

            brs.append(br)
            num_infostates_expanded += br.expanded_infostates


        return brs, num_infostates_expanded, new_br

    def run(self, game, iterations, seed):
        k = 0
        xodo_times = []
        xodo_exps = []
        policies, windows = [], [0]
        xodo_infostates = []
        num_infostates = 0
        cumu_br_info = 0
        exp_collected_time = 0
        start_time = time.time()
        previous_avg_policy, current_window_policy, prev_iter = None, None, None

        # Compute BR
        uniform = policy.UniformRandomPolicy(game)
        brs, expanded_infostates, new_br = self.get_best_response(game, uniform)
        num_infostates += expanded_infostates
        new_br = True
        br_list = [[brs[0]], [brs[1]]]

        # Construct meta game
        restricted_game = MetaGame(game, br_list, it=0)
        meta_solver = self.reset_meta_solver(restricted_game)
        current_window_policy = ExpandTabularPolicy(meta_solver.average_policy())
        tol=0.1
        alpha=0.5
        for i in range(iterations):
            avg_policy = current_window_policy

            conv,num_infostate_expanded = exploitability.exploitability(game, avg_policy)
            # conv = mc_exploitability(game, avg_policy)
            mcbr_str = "_mcbr" if self.is_mcbr else ""
            save_prefix = f'{self.out_dir}/{self.game_name}_{self.algorithm}_{self.meta_iterations}_ws{self.is_warm_start}{mcbr_str}_{seed}'

            if (new_br and i > 0) or i % self.data_collect_frequency == 0:
                # print(avg_policy.action_probability_array)
                print("Iteration {} exploitability {}".format(i, conv))
                wall_time = time.time() - start_time
                xodo_times.append(wall_time)
                xodo_exps.append(conv)
                xodo_infostates.append(num_infostates)
                ensure_dir(save_prefix)
                if time.time() - start_time < 258000:
                    np.save(save_prefix + '_times', np.array(xodo_times))
                    np.save(save_prefix + '_exps', np.array(xodo_exps))
                    np.save(save_prefix + '_infostates', np.array(xodo_infostates))

            if new_br and i > 0:
                # If there is new BR, construct meta-game, increase window count and reset strategy
                k += 1
                restricted_game = MetaGame(game, br_list, it=i)
                old_meta_solver = meta_solver
                meta_solver = self.reset_meta_solver(restricted_game, br_list, old_meta_solver)
                if previous_avg_policy:
                    del previous_avg_policy
                previous_avg_policy = avg_policy

            # Run meta-strategy updates
            if self.algorithm == "SADO":
                all_states, depth_h, max_a = get_all_states.get_all_statistics(restricted_game, include_chance_states=False)
                s_j = len(all_states)
                if is_mcbr:
                    frequency = np.rint(np.sqrt(s_j^3 * max_a)) * 1.#modified
                else:
                    frequency = np.rint(s_j * np.sqrt(max_a) / depth_h) * 1.#modified
            else:
                frequency = self.meta_iterations

            meta_solver.num_infostates_expanded = 0
            convs = []
            frequency = max(frequency,800)
            
            for _ in range(int(frequency)):
                meta_solver.evaluate_and_update_policy()
                if _%100==0:
                    conv,num_infostate_expanded = exploitability.exploitability(game,ExpandTabularPolicy(meta_solver.average_policy()))
                    num_infostates += num_infostate_expanded
                    convs.append(conv)
                    #if len(convs)>10 and (np.mean(convs[-8:])-convs[-1])/np.mean(convs[-8:])<tol:
                    if len(convs)>1 and (convs[-2]-convs[-1])/convs[-2]<tol:
                        tol *=alpha
                        break
                    
            num_infostates += meta_solver.num_infostates_expanded

            # Compute BR
            current_window_policy = ExpandTabularPolicy(meta_solver.average_policy())
            new_brs, expanded_infostates, new_br = self.get_best_response(game, current_window_policy)
            num_infostates += expanded_infostates

            if new_br:
                for pid in [0, 1]:
                    br_list[pid].append(new_brs[pid])

            # Release unreferenced memory
            if i % 10 == 0:
                gc.collect()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', type=str, choices=["SPDO", "SADO"],
                        required=False, default="SPDO")
    parser.add_argument('--meta_iterations', type=int, required=False, default=500)
    parser.add_argument('--is_mcbr', action='store_true')  # on/off flag
    parser.add_argument('--out_dir', type=str, required=False, default="results")  # output folder
    parser.add_argument('--seed', type=int, required=False, default=0)
    parser.add_argument('-w', '--is_warm_start', action='store_true')  # on/off flag
    parser.add_argument('--game_name', type=str, required=False, default="kuhn_poker",
                        choices=["leduc_poker", "kuhn_poker", "leduc_poker_10_card","leduc_poker_dummy", "oshi_zumo", "liars_dice",
                                 "python_large_kuhn_poker","kuhn_poker_dummy",
                                 "blotto", "blotto_20", "blotto_25", "blotto_30", "blotto_40"])
    commandline_args = parser.parse_args()

    seed = commandline_args.seed
    algorithm = commandline_args.algorithm
    game_name = commandline_args.game_name
    meta_iterations = commandline_args.meta_iterations
    iterations = 100000
    data_collect_frequency = 100
    is_warm_start = commandline_args.is_warm_start
    is_mcbr = commandline_args.is_mcbr
    out_dir = commandline_args.out_dir

    print(vars(commandline_args))

    np.random.seed(seed)
    algo = SPDO(algorithm, game_name, meta_iterations, data_collect_frequency, is_warm_start, is_mcbr, out_dir)
    game = algo.reset_game()
    algo.run(game, iterations, seed)

