import numpy as np
import pyspiel
from open_spiel.python.observation import IIGObserverForPublicInfoGame

_NUM_PLAYERS = 2
_HORIZON = 4
_NUM_ACTION = 20
_DECK = frozenset([0, 1])
_GAME_TYPE = pyspiel.GameType(
    short_name="python_blotto_20",
    long_name="Python Blotto 20",
    dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
    chance_mode=pyspiel.GameType.ChanceMode.DETERMINISTIC,
    information=pyspiel.GameType.Information.IMPERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.ZERO_SUM,
    reward_model=pyspiel.GameType.RewardModel.TERMINAL,
    max_num_players=_NUM_PLAYERS,
    min_num_players=_NUM_PLAYERS,
    provides_information_state_string=True,
    provides_information_state_tensor=True,
    provides_observation_string=True,
    provides_observation_tensor=True,
    provides_factored_observation_string=True)

_GAME_INFO = pyspiel.GameInfo(
    num_distinct_actions=_NUM_ACTION,
    max_chance_outcomes=len(_DECK),
    num_players=_NUM_PLAYERS,
    min_utility=-_NUM_ACTION / 2,
    max_utility=_NUM_ACTION / 2,
    utility_sum=0.0,
    max_game_length=_HORIZON)


class BlottoGame(pyspiel.Game):

    def __init__(self, params=None):
        super().__init__(_GAME_TYPE, _GAME_INFO, params or dict())

    def new_initial_state(self):
        """Returns a state corresponding to the start of a game."""
        return BlottoState(self)

    def make_py_observer(self, iig_obs_type=None, params=None):
        """Returns an object used for observing game state."""
        if ((iig_obs_type is None) or
                (iig_obs_type.public_info and not iig_obs_type.perfect_recall)):
            return BlottoObserver(params)
        else:
            return IIGObserverForPublicInfoGame(iig_obs_type, params)


class BlottoState(pyspiel.State):
    """A python version of the Kuhn poker state."""

    def __init__(self, game):
        super().__init__(game)
        self.card = None
        self.game = game
        self._is_terminal = False
        self._cur_player = 0
        self.traj = -np.ones(_HORIZON)
        self.mask = np.ones([2, _NUM_ACTION])

    def current_player(self):
        """Returns id of the next player to move, or TERMINAL if game is over."""
        return pyspiel.PlayerId.TERMINAL if self._is_terminal else self._cur_player

    def _legal_actions(self, player_id=None):
        """Returns a list of legal actions, sorted in ascending order."""
        if player_id != pyspiel.PlayerId.CHANCE:
            if player_id:
                return [a for a in range(_NUM_ACTION) if self.mask[player_id][a] == 1]
            return [a for a in range(_NUM_ACTION) if self.mask[self._cur_player][a] == 1]
        else:
            return _DECK - set(self.card)

    def _apply_action(self, action):
        """Applies the specified action to the state."""
        info_str = self.information_state_string(0)
        info_str1 = self.information_state_string(1)
        if self.is_chance_node():
            self.card = action
        else:
            self.traj[np.where(self.traj == -1)[0][0]] = action
            self.mask[self._cur_player][action] = 0
            self._cur_player = (self._cur_player + 1) % _NUM_PLAYERS

            if np.all(self.traj > -1):
                self._is_terminal = True

    def chance_outcomes(self):
        """Returns the possible chance outcomes and their probabilities."""
        if not self.is_chance_node():
            raise ValueError("chance_outcomes called on a non-chance state.")
        outcomes = _DECK - set(self.card)
        p = 1.0 / len(outcomes)
        return [(o, p) for o in outcomes]

    def _action_to_string(self, player, action):
        """Action -> string."""
        assert player == self._cur_player
        return str(action)

    # def information_state_string(self, player_id=None):
    #     cur_length = len(np.where(self.traj > -1))
    #     selected_traj = self.traj[:int(cur_length / 2) * 2]
    #     return str(self._cur_player) + ", ".join(map(str, selected_traj))

    def legal_actions_mask(self, player_id=None):
        return self.mask[player_id]

    def is_terminal(self):
        """Returns True if the game is over."""
        return self._is_terminal

    def returns(self):
        """Total reward for each player over the course of the game so far."""
        if not self._is_terminal:
            return [0., 0.]
        traj = np.array(self.traj)
        # max_returns = len(np.where(traj[::2] > traj[1::2])[0])
        max_returns = np.sum(traj[::2] - traj[1::2])
        if self.card == 0:
            return [max_returns, -max_returns]
        else:
            return [-max_returns, max_returns]

    def __str__(self):
        """String for debug purposes. No particular semantics are required."""
        return "".join(map(str, self.traj))


class BlottoObserver:
    """Observer, conforming to the PyObserver interface (see observation.py)."""

    def __init__(self, params):
        """Initializes an empty observation tensor."""
        if params:
            raise ValueError(f"Observation parameters not supported; passed {params}")
        # The observation should contain a 1-D tensor in `self.tensor` and a
        # dictionary of views onto the tensor, which may be of any shape.
        # Here the observation is indexed `(cell state, row, column)`.
        shape = (100)
        self.tensor = -np.ones(shape, np.float32)
        self.dict = {"observation": np.reshape(self.tensor, shape)}

    def set_from(self, state, player):
        """Updates `tensor` and `dict` to reflect `state` from PoV of `player`."""
        del player
        # We update the observation via the shaped tensor since indexing is more
        # convenient than with the 1-D tensor. Both are views onto the same memory.
        obs = self.dict["observation"]
        for ind, elem in state.traj:
            obs[ind] = elem

    def string_from(self, state, player):
        """Observation of `state` from the PoV of `player`, as a string."""
        return state.information_state_string(player)


pyspiel.register_game(_GAME_TYPE, BlottoGame)
