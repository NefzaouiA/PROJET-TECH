# battle_line_game.py
import random
from itertools import groupby
import numpy as np

# Constants for cards and colors
COLORS = ['red', 'blue', 'green']
NUM_VALUES = 10

class Card:
    def __init__(self, color, value):
        self.color = color  # e.g., 'red'
        self.value = value  # 1 to 10

    def __repr__(self):
        return f"{self.color[0].upper()}{self.value}"

    def encode(self):
        """
        Encode a card as an integer in the range 1..30.
        Calculation: color_index * NUM_VALUES + value.
        """
        return COLORS.index(self.color) * NUM_VALUES + self.value

def evaluate_hand(cards):
    """
    Given a list of 3 cards, evaluate the hand strength.
    The ranking (from best to worst):
      6: Trio (three cards of same value)
      5: Straight Flush (consecutive values, same color)
      4: Flush (all same color)
      3: Straight (consecutive values)
      2: Pair (two cards of same value)
      1: High Card
    The returned tuple is constructed so that higher tuples compare as stronger.
    """
    values = sorted([card.value for card in cards])
    colors = [card.color for card in cards]
    # Trio
    if values[0] == values[1] == values[2]:
        return (6, values[0])
    # Check straight and flush conditions
    is_straight = (values[0] + 1 == values[1] and values[1] + 1 == values[2])
    is_flush = (colors[0] == colors[1] == colors[2])
    if is_straight and is_flush:
        return (5, values[2])
    if is_flush:
        desc = sorted(values, reverse=True)
        return (4, desc[0], desc[1], desc[2])
    if is_straight:
        return (3, values[2])
    # Pair
    if values[0] == values[1] or values[1] == values[2] or values[0] == values[2]:
        if values[0] == values[1]:
            pair = values[0]
            kicker = values[2]
        elif values[1] == values[2]:
            pair = values[1]
            kicker = values[0]
        else:
            pair = values[0]
            kicker = values[1]
        return (2, pair, kicker)
    # High Card
    desc = sorted(values, reverse=True)
    return (1, desc[0], desc[1], desc[2])

class Flag:
    def __init__(self):
        # Each flag has two lists (one per player) for up to 3 cards each.
        self.slots = {"player": [], "opponent": []}

    def add_card(self, player, card):
        if len(self.slots[player]) < 3:
            self.slots[player].append(card)
            return True
        return False

    def is_complete(self):
        return len(self.slots["player"]) == 3 and len(self.slots["opponent"]) == 3

    def get_winner(self):
        """
        Once both players have placed 3 cards on this flag,
        evaluate the two hands. Returns 'player', 'opponent', or 'draw'.
        """
        if not self.is_complete():
            return None
        player_hand = evaluate_hand(self.slots["player"])
        opponent_hand = evaluate_hand(self.slots["opponent"])
        if player_hand > opponent_hand:
            return "player"
        elif opponent_hand > player_hand:
            return "opponent"
        else:
            return "draw"

class GameState:
    def __init__(self):
        # Create and shuffle the deck
        self.deck = [Card(color, value) for color in COLORS for value in range(1, NUM_VALUES + 1)]
        random.shuffle(self.deck)
        # Create 9 flags (battlefields)
        self.flags = [Flag() for _ in range(9)]
        # Deal initial hands (7 cards per player)
        self.hands = {"player": [], "opponent": []}
        for _ in range(7):
            self.hands["player"].append(self.deck.pop())
            self.hands["opponent"].append(self.deck.pop())
        self.current_turn = "player"  # 'player' starts

    def available_actions(self, player):
        """
        Return a list of valid actions for the given player.
        An action is represented as (card_index, flag_index), where:
          - card_index is the index in the player's hand.
          - flag_index is the flag (0 to 8) where the card will be placed.
        Only flags that have fewer than 3 cards for that player are allowed.
        """
        actions = []
        for i in range(len(self.hands[player])):
            for flag_index, flag in enumerate(self.flags):
                if len(flag.slots[player]) < 3:
                    actions.append((i, flag_index))
        return actions

    def play_move(self, player, card_index, flag_index):
        """
        Make a move for the given player.
        Returns True if the move is legal and executed, otherwise False.
        """
        if card_index >= len(self.hands[player]) or flag_index >= len(self.flags):
            return False  # invalid indices
        flag = self.flags[flag_index]
        if len(flag.slots[player]) >= 3:
            return False  # flag is already full for this player
        card = self.hands[player].pop(card_index)
        flag.add_card(player, card)
        # Switch turn after a valid move
        self.current_turn = "opponent" if player == "player" else "player"
        return True

    def check_game_over(self):
        """
        Evaluate each flag to count wins. Win conditions:
          - A player wins if they secure at least 5 flags.
          - Or if they secure 3 consecutive flags.
        Returns 'player', 'opponent', or None if the game continues.
        """
        results = []
        for flag in self.flags:
            results.append(flag.get_winner())
        player_wins = sum(1 for r in results if r == "player")
        opponent_wins = sum(1 for r in results if r == "opponent")
        if player_wins >= 5:
            return "player"
        if opponent_wins >= 5:
            return "opponent"
        # Check for 3 consecutive flags won
        for i in range(len(results) - 2):
            window = results[i:i+3]
            if window[0] == window[1] == window[2] == "player":
                return "player"
            if window[0] == window[1] == window[2] == "opponent":
                return "opponent"
        return None

class BattleLineGame:
    def __init__(self):
        self.state = GameState()
        self.winner = None
        self.move_count = 0

    def step(self, player, card_index, flag_index):
        """
        Execute a move for the current player.
        Returns (valid_move, winner) after the move.
        """
        valid = self.state.play_move(player, card_index, flag_index)
        self.move_count += 1
        self.winner = self.state.check_game_over()
        return valid, self.winner

    def render(self):
        """
        Render the game state to the console.
        """
        print("\n--- Battle Line State ---")
        for i, flag in enumerate(self.state.flags):
            print(f"Flag {i+1}:")
            print(f"  Player:   {flag.slots['player']}")
            print(f"  Opponent: {flag.slots['opponent']}")
        print(f"Player hand:   {self.state.hands['player']}")
        print(f"Opponent hand: {self.state.hands['opponent']}")
        print(f"Current turn:  {self.state.current_turn}")
        print("-------------------------\n")

    def get_state_vector(self):
        """
        Create a vector representation of the current state for the RL agent.
        Representation:
          - 7 slots for player's hand (card encoded as integer, 0 if empty)
          - 9 flags x 2 players x 3 slots = 54 entries for board state
          - 1 entry for current turn (0 for player, 1 for opponent)
        Total length = 7 + 54 + 1 = 62.
        """
        hand_vec = [0] * 7
        for i, card in enumerate(self.state.hands["player"]):
            hand_vec[i] = card.encode()
        board_vec = []
        for flag in self.state.flags:
            for side in ["player", "opponent"]:
                slots = flag.slots[side]
                slot_codes = [card.encode() for card in slots]
                while len(slot_codes) < 3:
                    slot_codes.append(0)
                board_vec.extend(slot_codes)
        turn = 0 if self.state.current_turn == "player" else 1
        return np.array(hand_vec + board_vec + [turn], dtype=np.float32)

    def get_valid_actions(self):
        """
        For the RL agent playing as 'player', encode each action as an integer:
          action = card_index * 9 + flag_index.
        """
        valid = []
        actions = self.state.available_actions("player")
        for (card_index, flag_index) in actions:
            valid.append(card_index * 9 + flag_index)
        return valid

    def decode_action(self, action):
        """
        Convert an action integer back into (card_index, flag_index).
        """
        card_index = action // 9
        flag_index = action % 9
        return card_index, flag_index
