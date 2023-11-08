import os
import gc
import sys
import time
import pyspiel
import argparse
import numpy as np
from copy import deepcopy
import blotto
import pdb
import large_kuhn_poker
from utils import get_support, ensure_dir, merge_two_policies   
from open_spiel.python.algorithms import get_all_states

from dependencies.open_spiel.python import policy
from dependencies.open_spiel.python.algorithms import cfr
from dependencies.open_spiel.python.algorithms import best_response
from dependencies.open_spiel.python.algorithms import exploitability
from dependencies.open_spiel.python.algorithms import outcome_sampling_mccfr as outcome_mccfr

module_path = os.path.abspath(os.path.join(''))
if module_path not in sys.path:
    sys.path.append(module_path)


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
                legal_actions.add(br.best_response_action(self.state.information_state_string()))
            ans = list(legal_actions)
            self.game._legal_actions_cache[self.state.history_str()] = ans
            return ans
        else:
            return self.state.legal_actions()

    def child(self, a):
        return MetaState(self.game, self.state.child(a), self.brs, br_id=self.br_id)

    def __getattr__(self, attr):
        # hacky hacky hacky hacky
        assert attr != 'new_initial_state'
        return self.state.__getattribute__(attr)

    def clone(self):
        return MetaState(self.game, self.state.clone(), self.brs, br_id=self.br_id)


class MetaGame:
    def __init__(self, game, brs, br_id=None):
        self.alter_state = pyspiel.load_game("kuhn_poker").new_initial_state
        self.game = game
        self.get_type = game.get_type
        self.num_players = game.num_players

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


class RMDO:
    def __init__(self, algorithm: str, game_name: str, meta_iterations: int, data_collect_frequency: int,
                 meta_solver: str, is_warm_start: bool, warm_start_discount=1, is_avg_warm_start=False, is_testing=False,
                 out_dir="results"):
        self.algorithm = algorithm
        self.game_name = game_name
        self.meta_solver = meta_solver
        self.meta_iterations = meta_iterations
        self.data_collect_frequency = data_collect_frequency
        self.is_warm_start = is_warm_start
        self.warm_start_discount = warm_start_discount
        self.is_avg_warm_start = is_avg_warm_start
        self.is_testing = is_testing
        self.out_dir = out_dir

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
        elif game_name == "python_large_kuhn_poker":
            game = large_kuhn_poker.KuhnPokerGame()
        else:
            game = pyspiel.load_game(self.game_name)
        return game
    
    def warm_star_init(self, state, old_solver, new_solver):
        if state.is_terminal():
            return

        if state.is_chance_node():
            for action, unused_action_prob in state.chance_outcomes():
                self.warm_star_init(state.child(action), old_solver, new_solver)
            return

        current_player = state.current_player()
        info_state = state.information_state_string(current_player)

        info_state_node = new_solver._info_state_nodes.get(info_state)
        old_info_state_node = old_solver._info_state_nodes.get(info_state)

        assert info_state_node is not None
        for action in info_state_node.legal_actions:
            if old_info_state_node and (action in old_info_state_node.cumulative_regret):
                info_state_node.cumulative_regret[action] = old_info_state_node.cumulative_regret[action] * self.warm_start_discount
            self.warm_star_init(state.child(action), old_solver, new_solver)

    def reset_meta_solver(self, restricted_game, old_solver=None):        
        if self.meta_solver == 'cfr_plus':
            meta_solver = cfr.CFRPlusSolver(restricted_game)
        elif self.meta_solver == 'cfr':
            meta_solver = cfr.CFRSolver(restricted_game)
        else:
            raise ValueError("Algorithm unidentified")
        
        if (not old_solver) or (not self.is_warm_start):
            return meta_solver
        print('warm start')
        if self.is_avg_warm_start:
            old_solver.get_avgpolicy_regret()
        self.warm_star_init(restricted_game.new_initial_state(), old_solver, meta_solver)
        return meta_solver
    

    def test_and_save(self, game, seed, start_time, num_infostates, k, meta_solver,
                        previous_avg_policy, current_window_policy, i, prev_iter=None):
        current_window_policy = ExpandTabularPolicy(meta_solver.average_policy())

        if (self.algorithm == "XODO") and previous_avg_policy:
            avg_policy = merge_two_policies(previous_avg_policy, current_window_policy, i, prev_iter)
        else:
            avg_policy = current_window_policy

        conv = exploitability.exploitability(game, avg_policy)
        save_prefix = f'{self.out_dir}/{self.game_name}_{self.algorithm}_{str(self.meta_iterations)}' + \
            f'_ws{self.is_warm_start}_avg{self.is_avg_warm_start}_{self.warm_start_discount}_{seed}'
        print("Iteration {} exploitability {}".format(i, conv))
        wall_time = time.time() - start_time
        self.rmdo_times.append(wall_time)
        self.rmdo_exps.append(conv)
        self.rmdo_infostates.append(num_infostates)
        self.supports.append([k] + get_support(avg_policy, game))
        ensure_dir(save_prefix)
        # if (time.time() - start_time < 64500) and (not self.is_testing):
        np.save(save_prefix + '_times', np.array(self.rmdo_times))
        np.save(save_prefix + '_exps', np.array(self.rmdo_exps))
        np.save(save_prefix + '_infostates', np.array(self.rmdo_infostates))
        np.save(save_prefix + '_infos', np.array(self.supports)) # k, the lowest exp and support
        return avg_policy



    def run(self, game, iterations, seed):
        brs = []
        k = 0
        br_actions = {}
        self.rmdo_times = []
        self.rmdo_exps = []
        self.rmdo_infostates = []
        self.supports = []
        num_infostates = 0
        start_time = time.time()
        previous_avg_policy, current_window_policy, prev_iter = None, None, None

        # Compute BR
        uniform = policy.UniformRandomPolicy(game)
        for pid in range(2):
            br = best_response.BestResponsePolicy(game, pid, uniform)
            br.expanded_infostates = 0
            root_state = game.new_initial_state()
            _ = br.value(root_state)
            for key, action in br.cache_best_response_action.items():
                br_actions[key] = [action]
            brs.append(br)
            num_infostates += br.expanded_infostates
        new_br = True
        br_list = [[brs[0]], [brs[1]]]

        # Construct meta game
        restricted_game = MetaGame(game, br_list)
        meta_solver = self.reset_meta_solver(restricted_game)

        for i in range(iterations):

            if (new_br and i > 0) or i % self.data_collect_frequency == 0:
                avg_policy = self.test_and_save(game, seed, start_time, num_infostates, k, meta_solver, \
                    previous_avg_policy, current_window_policy, i, prev_iter)

            if new_br:
                k += 1
                if i > 0:
                    restricted_game = MetaGame(game, br_list)
                    meta_solver = self.reset_meta_solver(restricted_game, old_solver=meta_solver)
                    if previous_avg_policy:
                        del previous_avg_policy
                    prev_iter = i
                    previous_avg_policy = avg_policy

            # Run meta-strategy updates
            if self.algorithm == "AdaDO":
                meta_solver.num_infostates_expanded = 0
                meta_solver.evaluate_and_update_policy()
                num_infostates += meta_solver.num_infostates_expanded
                frequency = np.rint(np.sqrt(meta_solver.max_act) * len(meta_solver.all_info_states)) * 0.1 - 1
            else:
                frequency = self.meta_iterations
            for meta_i in range(int(frequency)):
                meta_solver.num_infostates_expanded = 0
                meta_solver.evaluate_and_update_policy()
                num_infostates += meta_solver.num_infostates_expanded
                if meta_i % self.data_collect_frequency * 10 == 0:
                    self.test_and_save(game, seed, start_time, num_infostates, k, meta_solver, \
                        previous_avg_policy, current_window_policy, i, prev_iter)

            # Compute BR
            new_brs = []
            new_br = False
            current_window_policy = ExpandTabularPolicy(meta_solver.average_policy())
            for pid in range(2):
                br = best_response.BestResponsePolicy(game, pid, current_window_policy)
                br.expanded_infostates = 0
                _ = br.value(game.new_initial_state())
                num_infostates += br.expanded_infostates
                # Get the best response action for unvisited states
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
            if new_br:
                for pid in [0, 1]:
                    br_list[pid].append(new_brs[pid])

            # Release unreferenced memory
            if i % 10 == 0:
                gc.collect()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', type=str, choices=["XODO", "PDO", "AdaDO"],
                        required=False, default="PDO")
    parser.add_argument('--meta_iterations', type=int, required=False, default=50)
    parser.add_argument('--iterations', type=int, required=False, default=100000)
    parser.add_argument('--meta_solver', type=str, required=False, default="cfr_plus")
    parser.add_argument('-t', '--test', action='store_true')  # on/off flag
    parser.add_argument('-w', '--is_warm_start', action='store_true')  # on/off flag
    parser.add_argument('-aws', '--is_avg_warm_start', action='store_true')  # on/off flag
    parser.add_argument('--delta', type=str, required=False, default="0.1")  # warm start discount
    parser.add_argument('--out_dir', type=str, required=False, default="results")  # output folder
    parser.add_argument('--seed', type=int, required=False, default=0) 
    parser.add_argument('--game_name', type=str, required=False, default="kuhn_poker",
                        choices=["leduc_poker", "kuhn_poker", "leduc_poker_dummy", "oshi_zumo", "liars_dice",
                                 "goofspiel", "python_large_kuhn_poker",
                                 "phantom_ttt", "blotto"])
    commandline_args = parser.parse_args()

    seed = commandline_args.seed
    algorithm = commandline_args.algorithm
    game_name = commandline_args.game_name
    iterations = commandline_args.iterations
    meta_iterations = commandline_args.meta_iterations if algorithm == "PDO" else 1
    meta_solver = commandline_args.meta_solver
    is_warm_start = commandline_args.is_warm_start
    delta = eval(commandline_args.delta)
    is_avg_warm_start = commandline_args.is_avg_warm_start
    is_testing = commandline_args.test
    out_dir = commandline_args.out_dir

    data_collect_frequency = 10
    np.random.seed(seed)

    print(vars(commandline_args))

    rmdo = RMDO(algorithm=algorithm,
                game_name=game_name,
                meta_iterations=meta_iterations,
                data_collect_frequency=data_collect_frequency,
                meta_solver=meta_solver,
                is_warm_start=is_warm_start,
                warm_start_discount=delta,
                is_avg_warm_start=is_avg_warm_start,
                is_testing=is_testing,
                out_dir=out_dir)
    game = rmdo.reset_game()
    rmdo.run(game, iterations, seed)
