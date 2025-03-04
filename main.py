import numpy as np
from battle_line_env import BattleLineEnv
from dqn_agent import DQNAgent

# Updated for the new binary state representation
STATE_DIM = 2068  # The new state vector size
ACTION_DIM = 63   # 7 possible cards x 9 flags

def train_agent(episodes=5000):
    env = BattleLineEnv(opponent_policy="random")
    agent = DQNAgent(state_dim=STATE_DIM, action_dim=ACTION_DIM)
    win_count = 0 
    
    for ep in range(episodes):
        state = env.reset()
        done = False
        while not done:
            valid_actions = env.get_valid_actions()
            if not valid_actions:
                # No valid actions => terminal
                winner = env.game.state.check_game_over()
                reward = 1.0 if winner == "player" else -1.0 if winner == "opponent" else 0.0
                agent.store_transition(state, 0, reward, state, True)
                done = True
                break

            # Agent picks action
            action = agent.select_action(state, valid_actions)
            next_state, reward, done, info = env.step(action)
            
            # Store and update
            agent.store_transition(state, action, reward, next_state, done)
            agent.update()
            
            state = next_state

        if reward > 0:
            win_count += 1
        
        if (ep+1) % 100 == 0:
            print(f"Episode {ep+1}/{episodes} - Win rate: {win_count/100:.2f} "
                  f"(epsilon: {agent.epsilon:.2f})")
            win_count = 0

def play_game():
    """
    Simple text-based UI for a human to play against the AI.
    The player is first asked which card (by its position in the hand) they want to play,
    then which flag to target.
    """
    env = BattleLineEnv(opponent_policy="random")
    state = env.reset()
    done = False

    while not done:
        env.render()
        valid_actions = env.get_valid_actions()

        # Show the player's hand
        print("Your hand:")
        for i, card in enumerate(env.game.state.hands["player"]):
            print(f"  {i}: {card}")
        
        # Show flags
        print("\nFlags:")
        for i in range(len(env.game.state.flags)):
            print(f"  Flag {i}")
        
        # Ask for card index
        try:
            card_choice = int(input("\nWhich card do you want to play? (Enter card index) "))
        except ValueError:
            print("Invalid input. Please enter an integer for the card index.")
            continue

        # Ask for flag index
        try:
            flag_choice = int(input("Which flag do you choose? (Enter flag index, 0-8) "))
        except ValueError:
            print("Invalid input. Please enter an integer for the flag index.")
            continue

        # Encode the action
        action = card_choice * 9 + flag_choice

        # Validate and step
        if action not in valid_actions:
            print("Action not valid. Try again.\n")
            continue

        state, reward, done, info = env.step(action)
    
    env.render()
    if reward > 0:
        print("You win!")
    elif reward < 0:
        print("You lose!")
    else:
        print("Draw!")

if __name__ == '__main__':
    mode = input("Enter mode (train/play): ").strip().lower()
    if mode == "train":
        train_agent(episodes=5000)
    elif mode == "play":
        play_game()
    else:
        print("Unknown mode.")
