# main.py
import numpy as np
from battle_line_env import BattleLineEnv
from dqn_agent import DQNAgent

STATE_DIM = 62      # as defined in get_state_vector()
ACTION_DIM = 7 * 9  # maximum of 63 discrete actions (7 possible card indices x 9 flags)

def train_agent(episodes=5000):
    env = BattleLineEnv(opponent_policy="random")
    agent = DQNAgent(state_dim=STATE_DIM, action_dim=ACTION_DIM)
    win_count = 0
    
    for ep in range(episodes):
        state = env.reset()
        done = False
        while not done:
            valid_actions = env.get_valid_actions()
            action = agent.select_action(state, valid_actions)
            next_state, reward, done, info = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            agent.update()
            state = next_state
        if reward > 0:
            win_count += 1
        if (ep+1) % 100 == 0:
            print(f"Episode {ep+1}/{episodes} - Win rate: {win_count/100:.2f} (epsilon: {agent.epsilon:.2f})")
            win_count = 0

def play_game():
    """
    Simple text-based UI for a human to play against the AI.
    Here the human plays as 'player' and the AI (using random moves for now)
    plays as 'opponent'. You can extend this to use a trained DQN agent.
    """
    env = BattleLineEnv(opponent_policy="random")
    state = env.reset()
    done = False
    while not done:
        env.render()
        valid_actions = env.get_valid_actions()
        print("Valid actions (encoded as card_index * 9 + flag_index):", valid_actions)
        try:
            user_action = int(input("Enter your action: "))
        except ValueError:
            print("Invalid input.")
            continue
        if user_action not in valid_actions:
            print("Action not valid. Try again.")
            continue
        state, reward, done, info = env.step(user_action)
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
        train_agent(episodes=1000)
    elif mode == "play":
        play_game()
    else:
        print("Unknown mode.")
