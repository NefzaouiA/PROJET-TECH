# battle_line_env.py
import random
from battle_line_game import BattleLineGame
import numpy as np

class BattleLineEnv:
    def __init__(self, opponent_policy="random"):
        """
        opponent_policy: choose how the opponent moves.
         - 'random': makes random valid moves.
         - You can later plug in a heuristic or trained model.
        """
        self.game = BattleLineGame()
        self.opponent_policy = opponent_policy

    def reset(self):
        self.game = BattleLineGame()
        return self.game.get_state_vector()

    def step(self, action):
        """
        The agent (playing as 'player') takes an action.
        After that, if the game is not over, the opponent makes a move.
        Action is an integer encoding (card_index * 9 + flag_index).
        Returns: next_state, reward, done, info
        """
        card_index, flag_index = self.game.decode_action(action)
        valid, winner = self.game.step("player", card_index, flag_index)
        if not valid:
            # Illegal move penalty
            return self.game.get_state_vector(), -0.5, True, {"error": "Invalid move"}
        # Check if game ended after player's move
        if winner is not None:
            reward = 1.0 if winner == "player" else -1.0
            return self.game.get_state_vector(), reward, True, {}
        # Now, let the opponent move (using a simple random policy)
        opp_valid_actions = self.game.state.available_actions("opponent")
        if opp_valid_actions:  # if opponent can move
            opp_move = random.choice(opp_valid_actions)
            opp_card_index, opp_flag_index = opp_move
            self.game.step("opponent", opp_card_index, opp_flag_index)
        # Check for game over after opponent's move
        winner = self.game.state.check_game_over()
        if winner is not None:
            reward = 1.0 if winner == "player" else -1.0
            done = True
        else:
            reward = 0.0
            done = False
        return self.game.get_state_vector(), reward, done, {}

    def render(self):
        self.game.render()

    def get_valid_actions(self):
        return self.game.get_valid_actions()
