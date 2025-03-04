import random
import numpy as np
from battle_line_game import BattleLineGame

class BattleLineEnv:
    def __init__(self, opponent_policy="random"):
        """
        opponent_policy: choose how the opponent moves.
         - 'random': makes random valid moves.
         - 'ai': placeholder for an AI opponent (you can implement later).
        """
        self.game = BattleLineGame()
        self.opponent_policy = opponent_policy
        # Track the previous flag score (player flags minus opponent flags)
        self.previous_score = self.get_flag_score()

    def get_flag_score(self): 
        """
        Compute a score based on completed flags.
        Each flag that the player wins adds +1,
        each flag that the opponent wins subtracts -1.
        Flags that are 'draw' or not decided count as 0.
        """
        results = [flag.get_winner() for flag in self.game.state.flags]
        player_flags = sum(1 for r in results if r == "player")
        opponent_flags = sum(1 for r in results if r == "opponent")
        return player_flags - opponent_flags

    def reset(self):
        self.game = BattleLineGame()
        self.previous_score = self.get_flag_score()
        return self.game.get_state_vector()

    def step(self, action):
        """
        The agent (playing as 'player') takes an action.
        Then, if the game is not over, the opponent makes a move.
        Action is an integer encoding (card_index * 9 + flag_index).
        Returns: next_state, reward, done, info
        """
        # If no valid actions remain, end the episode immediately.
        if not self.game.state.available_actions("player"):
            winner = self.game.state.check_game_over()
            reward = 1.0 if winner == "player" else -1.0 if winner == "opponent" else 0.0
            return self.game.get_state_vector(), reward, True, {"error": "No valid actions"}

        # --- Agent's Move ---
        card_index, flag_index = self.game.decode_action(action)
        valid, _ = self.game.step("player", card_index, flag_index)
        if not valid:
            # Illegal move penalty
            return self.game.get_state_vector(), -0.5, True, {"error": "Invalid move"}

        # Small reward for making a legal move
        move_reward = 0.5

        # Check intermediate reward by comparing flag scores
        new_score = self.get_flag_score()
        # Bonus: reward (or penalty) based on change in flag score
        intermediate_reward = 0.5 * (new_score - self.previous_score)
        self.previous_score = new_score

        reward = move_reward + intermediate_reward

        # Check if game ended after the player's move
        winner = self.game.state.check_game_over()
        if winner is not None:
            final_reward = 1.0 if winner == "player" else -1.0
            return self.game.get_state_vector(), final_reward, True, {}

        # --- Opponent's Move ---
        opp_valid_actions = self.game.state.available_actions("opponent")
        if opp_valid_actions:
            if self.opponent_policy == "random":
                # Random move
                opp_move = random.choice(opp_valid_actions)
            elif self.opponent_policy == "ai":
                # Placeholder for AI logic (could be a trained model).
                # For now, fallback to random or a heuristic.
                opp_move = random.choice(opp_valid_actions)

            opp_card_index, opp_flag_index = opp_move
            self.game.step("opponent", opp_card_index, opp_flag_index)

        # Update intermediate reward after opponent's move
        new_score = self.get_flag_score()
        intermediate_reward = 0.1 * (new_score - self.previous_score)
        self.previous_score = new_score
        reward += intermediate_reward

        # Check if game is over after opponent's move
        winner = self.game.state.check_game_over()
        if winner is not None:
            final_reward = 1.0 if winner == "player" else -1.0
            return self.game.get_state_vector(), final_reward, True, {}
        else:
            done = False

        return self.game.get_state_vector(), reward, done, {}

    def render(self):
        self.game.render()

    def get_valid_actions(self):
        """
        Returns a list of valid encoded actions for the 'player'.
        """
        return self.game.get_valid_actions()
