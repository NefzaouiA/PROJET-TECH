import random
from itertools import groupby
import numpy as np

# Constants for cards and colors
COLORS = ['red', 'blue', 'green']
NUM_VALUES = 10

def card_to_onehot(card):
    """
    Convert a card to a one-hot vector of length 30:
    Index = color_index * 10 + (value - 1)
    For example, red 1 => index 0, blue 10 => index 29, etc.
    """
    vec = np.zeros(30, dtype=np.int32)
    color_idx = COLORS.index(card.color)
    val_idx = card.value - 1
    index = color_idx * NUM_VALUES + val_idx
    vec[index] = 1
    return vec

class Card:
    def __init__(self, color, value):
        self.color = color  # e.g., 'red'
        self.value = value  # 1 to 10

    def __repr__(self):
        return f"{self.color[0].upper()}{self.value}"

    def encode(self):
        """
        Encode a card as an integer in the range 1..30.
        Calculation: color_index * NUM_VALUES + value
        (We keep this method for backward compatibility;
         for RL, we'll use card_to_onehot() in get_state_vector().)
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
    if len(cards) < 3:
        # If less than 3 cards, we can't evaluate a standard formation,
        # so just return something very low to indicate incomplete set.
        return (0, 0)

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
        self.slots = {"player": [], "opponent": []}
        self.winner = None  # Flag is locked once winner is determined

    def add_card(self, player, card):
        if self.winner is not None:
            return False  # Flag already decided
        if len(self.slots[player]) < 3:
            self.slots[player].append(card)
            # Check if flag is now complete:
            if self.is_complete():
                self.winner = self.get_winner()
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
        An action is (card_index, flag_index).
        Skip flags that are decided or full for that player.
        """
        actions = []
        for i in range(len(self.hands[player])):
            for flag_index, flag in enumerate(self.flags):
                if flag.winner is not None:
                    continue  # Skip already decided flags
                if len(flag.slots[player]) < 3:
                    actions.append((i, flag_index))
        return actions

    def play_move(self, player, card_index, flag_index):
        """
        Make a move for the given player.
        Returns True if the move is legal and executed, otherwise False.
        """
        if card_index >= len(self.hands[player]) or flag_index >= len(self.flags):
            return False
        flag = self.flags[flag_index]
        if flag.winner is not None or len(flag.slots[player]) >= 3:
            return False
        card = self.hands[player].pop(card_index)
        if flag.add_card(player, card):
            # Draw a new card if available and if hand is below 7 cards
            if self.deck and len(self.hands[player]) < 7:
                self.hands[player].append(self.deck.pop())
            self.current_turn = "opponent" if player == "player" else "player"
            return True
        else:
            # If move failed, restore the card
            self.hands[player].insert(card_index, card)
            return False

    def check_game_over(self):
        """
        Evaluate each flag to count wins. Win conditions:
          - A player wins if they secure at least 5 flags.
          - Or if they secure 3 consecutive flags.
        Returns 'player', 'opponent', or None if the game continues.
        """
        results = [flag.get_winner() for flag in self.flags]
        player_wins = sum(1 for r in results if r == "player")
        opponent_wins = sum(1 for r in results if r == "opponent")

        # 5 flags rule
        if player_wins >= 5:
            return "player"
        if opponent_wins >= 5:
            return "opponent"

        # 3 consecutive flags
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
        Returns (valid_move, winner).
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
            status = f"(Winner: {flag.winner})" if flag.winner else ""
            print(f"Flag {i+1} {status}:")
            print(f"  Player:   {flag.slots['player']}")
            print(f"  Opponent: {flag.slots['opponent']}")
        print(f"Player hand:   {self.state.hands['player']}")
        print(f"Opponent hand: {self.state.hands['opponent']}")
        print(f"Current turn:  {self.state.current_turn}")
        print("-------------------------\n")

    def get_state_vector(self):
        """
        Returns a one-dimensional binary (one-hot) representation of the entire game state.

        Layout:
          1) Player's hand (7 cards), each card -> 30 bits => 7*30 = 210
          2) Opponent's hand (7 cards), each card -> 30 bits => 210
          3) For each of the 9 flags:
               - Player side: up to 3 cards => 3*30 = 90
               - Opponent side: up to 3 cards => 90
               - Flag status: one-hot [not decided, player, opponent] => 3 bits
            => total per flag = 90 + 90 + 3 = 183
            => for 9 flags => 9*183 = 1647
          4) Current turn (1 bit for 'player' or 'opponent'—or 2 bits if you prefer one-hot).
             We'll use 1 bit: 0 = player, 1 = opponent

        Total = 210 + 210 + 1647 + 1 = 2068
        """
        # 1) Player hand
        player_hand_vecs = []
        for i in range(7):
            if i < len(self.state.hands["player"]):
                onehot = card_to_onehot(self.state.hands["player"][i])
            else:
                onehot = np.zeros(30, dtype=np.int32)
            player_hand_vecs.append(onehot)

        # 2) Opponent hand
        opp_hand_vecs = []
        for i in range(7):
            if i < len(self.state.hands["opponent"]):
                onehot = card_to_onehot(self.state.hands["opponent"][i])
            else:
                onehot = np.zeros(30, dtype=np.int32)
            opp_hand_vecs.append(onehot)

        # 3) Flags
        flags_vecs = []
        for flag in self.state.flags:
            # Player side
            for j in range(3):
                if j < len(flag.slots["player"]):
                    onehot = card_to_onehot(flag.slots["player"][j])
                else:
                    onehot = np.zeros(30, dtype=np.int32)
                flags_vecs.append(onehot)
            # Opponent side
            for j in range(3):
                if j < len(flag.slots["opponent"]):
                    onehot = card_to_onehot(flag.slots["opponent"][j])
                else:
                    onehot = np.zeros(30, dtype=np.int32)
                flags_vecs.append(onehot)
            # Flag status: [not decided, player, opponent]
            if flag.winner is None:
                status_vec = np.array([1, 0, 0], dtype=np.int32)
            elif flag.winner == "player":
                status_vec = np.array([0, 1, 0], dtype=np.int32)
            elif flag.winner == "opponent":
                status_vec = np.array([0, 0, 1], dtype=np.int32)
            else:
                status_vec = np.array([0, 0, 0], dtype=np.int32)
            flags_vecs.append(status_vec)

        # Flatten everything
        player_hand_flat = np.concatenate(player_hand_vecs)
        opp_hand_flat = np.concatenate(opp_hand_vecs)
        flags_flat = np.concatenate(flags_vecs)

        # 4) Current turn bit
        turn_bit = np.array([1 if self.state.current_turn == "opponent" else 0], dtype=np.int32)

        # Concatenate all
        state_vec = np.concatenate([player_hand_flat, opp_hand_flat, flags_flat, turn_bit])
        return state_vec

    def get_valid_actions(self): 
        """
        For the RL agent playing as 'player', encode each action as an integer:
          action = card_index * 9 + flag_index.
        """
        valid = []
        actions = self.state.available_actions("player")
        for (c_idx, f_idx) in actions:
            valid.append(c_idx * 9 + f_idx)
        return valid

    def decode_action(self, action):
        """
        Convert an action integer back into (card_index, flag_index).
        """
        card_index = action // 9
        flag_index = action % 9
        return card_index, flag_index
