import numpy as np
import pdb
from collections import defaultdict
import os

def get_support(policy, game):
    supports = defaultdict(list)
    bars = [0, 0.0001, 0.001, 0.01, 0.1] 
    results = []
    
    def traverse(state):
        if state.is_terminal():
            return

        if state.is_chance_node():
            for action, unused_action_prob in state.chance_outcomes():
                traverse(state.child(action))
            return 

        legal_actions = state.legal_actions()
        for bar in bars:
            num_positive_probs = len(np.where(np.fromiter(policy.action_probabilities(state).values(), dtype=float) > bar)[0])
            supports[bar].append(num_positive_probs / len(legal_actions))

        for action in legal_actions:
            traverse(state.child(action))
    
    traverse(game.new_initial_state())
    return [(np.mean(supports[key]).round(2), np.sum(supports[key]).round(2)) for key in supports.keys()]


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


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

