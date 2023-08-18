import os
import gc
import sys
import pdb
import time
import pyspiel
import argparse
import datetime
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from collections import defaultdict

import blotto
import large_kuhn_poker
from dependencies.open_spiel.open_spiel.python.games import tic_tac_toe
from dependencies.open_spiel.open_spiel.python.games import kuhn_poker

from dependencies.open_spiel.open_spiel.python import policy
from dependencies.open_spiel.open_spiel.python.algorithms import cfr
from dependencies.open_spiel.open_spiel.python.algorithms import mccfr
from dependencies.open_spiel.open_spiel.python.algorithms import exploitability
from dependencies.open_spiel.open_spiel.python.algorithms import get_all_states
from dependencies.open_spiel.open_spiel.python.algorithms import outcome_sampling_mccfr as outcome_mccfr
from dependencies.open_spiel.open_spiel.python.algorithms.best_response import _memoize_method

class BestResponseWrapper():
    def __init__(self, br_policy):
        self.br_policy = br_policy
    
    def best_response_action_wrapped(self, state):
        return self.br_policy.best_response_action(state.information_state_string())
    
    def value(self, num_values=0):
        self.br_policy.value(num_values)


class MCBR(outcome_mccfr.OutcomeSamplingSolver):
    def __init__(self, game, br_id, policy):
        super().__init__(game)
        self._game = game
        self.br_id = br_id
        self.policy = policy
        self.expanded_infostates = 0   
        self.state_str_to_legal_actions = {} 

    @_memoize_method
    def best_response_action(self, info_state_key):
        legal_actions = self.state_str_to_legal_actions[info_state_key]
        if info_state_key not in self._infostates:
            return legal_actions[0]
        action_id = np.argmax(self._infostates[info_state_key][0])
        return legal_actions[action_id]
    
    def update_legal_actions(self, state):
        info_state_key  = state.information_state_string()
        self.state_str_to_legal_actions[info_state_key] = state.legal_actions()
        self.best_response_action(info_state_key)

    def value(self, num_values):
        self.expanded_infostates = 0
        for _ in range(num_values):
            state = self._game.new_initial_state()
            self._episode_br(state, self.br_id, opp_reach=1.0, sample_reach=1.0)

    def _episode_br(self, state, br_id, opp_reach, sample_reach):
        """Runs an episode of outcome sampling.

        Args:
        state: the open spiel state to run from (will be modified in-place).
        br_id: the player to update regrets for (the other players
            update average strategies)
        opp_reach: reach probability of all the opponents (including chance)
        sample_reach: reach probability of the sampling (behavior) policy

        Returns:
        util is a real value representing the utility of the update player
        """
        self.expanded_infostates += 1

        if state.is_terminal():
            return state.player_return(br_id)

        if state.is_chance_node():
            outcomes, probs = zip(*state.chance_outcomes())
            aidx = np.random.choice(range(len(outcomes)), p=probs)
            state.apply_action(outcomes[aidx])
            return self._episode_br(state, br_id, probs[aidx] * opp_reach, probs[aidx] * sample_reach)

        cur_player = state.current_player()
        info_state_key = state.information_state_string(cur_player)
        legal_actions = state.legal_actions()
        num_legal_actions = len(legal_actions)
        infostate_info = self._lookup_infostate_info(info_state_key,
                                                    num_legal_actions)

        if cur_player == br_id:
            policy = self._regret_matching(infostate_info[mccfr.REGRET_INDEX],
                                            num_legal_actions)
            uniform_policy = (
                np.ones(num_legal_actions, dtype=np.float64) / num_legal_actions)
            sample_policy = self._expl * uniform_policy + (1.0 - self._expl) * policy
        else:
            policy = self.policy.action_probabilities(state)
            sample_policy = np.fromiter(policy.values(), dtype=float)

        sampled_aidx = np.random.choice(range(num_legal_actions), p=sample_policy)
        state.apply_action(legal_actions[sampled_aidx])

        if cur_player == br_id:
            new_opp_reach = opp_reach
        else:
            new_opp_reach = opp_reach * policy[sampled_aidx]
        new_sample_reach = sample_reach * sample_policy[sampled_aidx]
        child_value = self._episode_br(state, br_id, new_opp_reach, new_sample_reach)

        # Compute each of the child estimated values.
        child_values = np.zeros(num_legal_actions, dtype=np.float64)
        for aidx in range(num_legal_actions):
            child_values[aidx] = self._baseline_corrected_child_value(
                state, infostate_info, sampled_aidx, aidx, child_value,
                sample_policy[aidx])
        value_estimate = 0
        for aidx in range(num_legal_actions):
            value_estimate += policy[sampled_aidx] * child_values[aidx]

        if cur_player == br_id:
            # Estimate for the counterfactual value of the policy.
            cf_value = value_estimate * opp_reach / sample_reach
            for aidx in range(num_legal_actions):
                # Estimate for the counterfactual value of the policy replaced by always
                # choosing sampled_aidx at this information state.
                cf_action_value = child_values[aidx] * opp_reach / sample_reach
                self._add_regret(info_state_key, aidx, cf_action_value - cf_value)
            
            # cache br policy
            if info_state_key not in self.state_str_to_legal_actions:
                self.state_str_to_legal_actions[info_state_key] = legal_actions
            else:
                assert self.state_str_to_legal_actions[info_state_key] == legal_actions
            self.best_response_action(info_state_key)

        return value_estimate



class mcts_br:
    def __init__(self, game, pid, strategy, exploration_weight=5.0):
        self.game = game
        self.pid = pid
        self.strategy = strategy
        self.exploration_weight = exploration_weight
        self.q = defaultdict(lambda: 0.0)
        self.count = defaultdict(lambda: 0.0)
        self.cache_best_response_action = defaultdict(lambda: None)
    
    def best_response_action(self, information_str):
        return self.cache_best_response_action[information_str]

    def uct(self, state_key, parent_count):
        log_n = np.log(parent_count) 
        cur_state_count = self.count[state_key]
        explore_term = self.exploration_weight * np.sqrt(log_n / cur_state_count)
        exploit_term = (self.q[state_key] / cur_state_count)
        return exploit_term + explore_term
    
    def get_info_key(self, info_state_key, child_state, action=None):
        if child_state.is_terminal():
            return info_state_key + str(action)
        else:
            return child_state.information_state_string()

    def value(self, times=100):
        self.q = defaultdict(lambda: 0.0)
        self.count = defaultdict(lambda: 1.0)
        for _ in range(times):
            self._episode(self.game.new_initial_state(), self.pid)
        print(self.q.items())
        for cur_state in get_all_states.get_all_states(self.game).values():
            if cur_state.is_terminal():
                continue
            information_str = cur_state.information_state_string()
            legal_actions = cur_state.legal_actions()
            num_legal_actions = len(legal_actions)
            max_count, max_count_action = 0, legal_actions[0]
            for action_idx in range(num_legal_actions):
                child_state = cur_state.child(legal_actions[action_idx])
                child_state_key = self.get_info_key(information_str, child_state, legal_actions[action_idx])
                if self.count[child_state_key] > max_count:
                    max_count = self.count[child_state_key]
                    max_count_action = legal_actions[action_idx]
            self.cache_best_response_action[information_str] = max_count_action

    def _episode(self, state, update_player):
        """Runs an episode of mcts.

        Args:
        state: the open spiel state to run from (will be modified in-place).
        update_player: the player to update regrets for (the other players
            update average strategies)
        Returns:
        util is a real value representing the utility of the update player
        """
        if state.is_terminal():
            return state.player_return(update_player)

        if state.is_chance_node():
            outcomes, probs = zip(*state.chance_outcomes())
            aidx = np.random.choice(range(len(outcomes)), p=probs)
            state.apply_action(outcomes[aidx])
            return self._episode(state, update_player)
        
        # print(state.information_state_string(), state.legal_actions(), self.q.items())
        cur_player = state.current_player()
        info_state_key = state.information_state_string(cur_player)
        legal_actions = state.legal_actions()
        num_legal_actions = len(legal_actions)

        if cur_player == update_player:
            uct_values = np.zeros(num_legal_actions)
            if np.random.sample() > 1:
                sampled_aidx = np.random.choice(num_legal_actions)
            else:
                for action_idx in range(num_legal_actions):
                    child_state = state.child(legal_actions[action_idx])
                    child_state_key = self.get_info_key(info_state_key, child_state, legal_actions[action_idx])
                    uct_values[action_idx] = self.uct(child_state_key, self.count[info_state_key])
                sampled_aidx = np.argmax(uct_values)
        else:
            sample_policy = np.fromiter(self.strategy.action_probabilities(state).values(), dtype=float)
            sampled_aidx = np.random.choice(range(num_legal_actions), p=sample_policy)

        state.apply_action(legal_actions[sampled_aidx])
        child_state_key = state.information_state_string(update_player)

        self.count[child_state_key] += 1
        child_value = self._episode(state, update_player)
        self.q[child_state_key] += child_value
        return child_value
