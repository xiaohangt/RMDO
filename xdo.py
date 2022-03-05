import pyspiel
from dependencies.open_spiel.python import policy
from dependencies.open_spiel.python.algorithms import best_response
from dependencies.open_spiel.python.algorithms import cfr
from dependencies.open_spiel.python.algorithms import exploitability
from dependencies.open_spiel.python.algorithms import external_sampling_mccfr as external_mccfr
from dependencies.open_spiel.python.algorithms import expected_game_score
from dependencies.open_spiel.python.algorithms import get_all_states
from dependencies.open_spiel.python.observation import make_observation

import copy
import random
import time

import numpy as np
import matplotlib.pyplot as plt

# import os
# import sys
# module_path = os.path.abspath(os.path.join('..'))
# if module_path not in sys.path:
#     sys.path.append(module_path)
# from lib.utils import debug, info, debug_game, game_score_of_best_response, paste_subpolicy_onto_policy, ResultContainer, quick_debug_state

try:
    #     from tqdm.notebook import tqdm
    from tqdm import tqdm
except ImportError as e:
    print('{} -- (tqdm is a cosmetic-only progress bar) -- setting `tqdm` to the identity function instead'.format(e))
    tqdm = lambda x: x


class WrappedState:
    def __init__(self, game, state, brs, br_id=None):
        #         super().__init__(self, game)
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
            # if self.br_id is not None and self.state.current_player() != self.br_id \
            #         and self.state.current_player() in [0, 1]:
            #     num = int(len(self.state.legal_actions()) / 2)
            #     legal_actions = np.random.choice(self.state.legal_actions(), num)
            #     ans = list(legal_actions)
            #     self.game._legal_actions_cache[self.state.history_str()] = ans
            #     return ans
            # else:
            return self.state.legal_actions()

    def child(self, a):
        return WrappedState(self.game, self.state.child(a), self.brs, br_id=self.br_id)

    def __getattr__(self, attr):
        # hacky hacky hacky hacky
        assert attr != 'new_initial_state'
        return self.state.__getattribute__(attr)

    def clone(self):
        return WrappedState(self.game, self.state.clone(), self.brs, br_id=self.br_id)


class WrappedGame():
    def __init__(self, game, brs, br_id=None):
        self.game = game
        self.brs = brs
        self.br_id = br_id
        game_info = pyspiel.GameInfo(
            num_distinct_actions=self.game.num_distinct_actions(),
            max_chance_outcomes=self.game.max_chance_outcomes(),
            num_players=self.game.num_players(),
            min_utility=self.game.min_utility(),
            max_utility=self.game.max_utility(),
            utility_sum=self.game.utility_sum(),
            max_game_length=self.game.max_game_length()  # TODO: this is hardcoded
        )
        self._legal_actions_cache = dict()
        # print('aaaa')
        # print(self.game.get_type(), game_info, self.game.get_parameters())
        # super().__init__(self, self.game.get_type(), game_info, self.game.get_parameters())
        # print('bbbbb')

    #     def get_info(self):
    #         return pyspiel.GameInfo(
    #             num_distinct_actions=self.num_distinct_actions(),
    #             max_chance_outcomes=self.max_chance_outcomes(),
    #             num_players=self.game.num_players(),
    #             min_utility=self.game.min_utility(),
    #             max_utility=self.game.max_utility(),
    #             utility_sum=self.game.utility_sum(),
    #             max_game_length=self.game.max_game_length())
    def new_initial_state(self):
        return WrappedState(self, self.game.new_initial_state(), self.brs, br_id=self.br_id)

    def make_py_observer(self, *args, **kwargs):
        try:
            return make_observation(self.game, *args, **kwargs)
        except:
            class dummy:
                def __init__(self):
                    self.tensor = None

                def set_from(self, state):
                    self.state = state
                    self.tensor = state.information_state_tensor()

                def string_from(self, state, player):
                    return state.information_state_string(player)

            return dummy()

    def __getattr__(self, attr):
        # hacky hacky hacky hacky
        assert attr != 'new_initial_state'
        return self.game.__getattribute__(attr)


class LazyTabularPolicy(policy.Policy):
    def __init__(self, restricted_game, full_game, p):
        self.p = p.to_tabular()

    def action_probabilities(self, state):
        if state.information_state_string() in self.p.state_lookup:
            # should probably include actions that are in full_game but not in restricted_game, with the probability set to 0
            return self.p.action_probabilities(state)
        l = len(state.legal_actions())
        return {a: 1 / l for a in state.legal_actions()}

#### usage example #####
# uniform = policy.UniformRandomPolicy(game)
# brs = []
# for pid in [0,1]:
#     brs.append([best_response.BestResponsePolicy(game, pid, uniform)])
#
# inner_iters = 500
# for outer_loop_iter in range(40):
#     print(f'{outer_loop_iter = }')
#     restricted_game = WrappedGame(game, brs)
#     mccfr_solver = external_mccfr.ExternalSamplingSolver(restricted_game, external_mccfr.AverageType.SIMPLE)
#     while True:
#         for inner_loop_iter in tqdm(range(int(inner_iters))):
#             mccfr_solver.iteration()
#         restricted_exploitability = exploitability.nash_conv(restricted_game, mccfr_solver.average_policy())/2
#         print(f'inner loop exploitability: {restricted_exploitability}')
#         if restricted_exploitability < 0.04:
#             break
#     total_exploitability = 0
#     print('making full policy')
# #     restricted_policy = mccfr_solver.average_policy().to_tabular()
# #     full_policy = policy.TabularPolicy(game)
#     full_policy = LazyTabularPolicy(restricted_game, game, mccfr_solver.average_policy())
# #     paste_subpolicy_onto_policy(full_policy, restricted_policy)
#     print('making new brs')
#     for pid in [0,1]:
#         new_br = best_response.BestResponsePolicy(game, pid, full_policy)
#         brs[pid].append(new_br)
#         total_exploitability += new_br.value(game.new_initial_state())
#     print(f'avg exploitability: {total_exploitability/2}')
