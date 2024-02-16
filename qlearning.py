import numpy as np
import time
import os
import tkinter
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import gymnasium as gym
import gym_examples
env = gym.make("gym_examples/GridWorld-v0", render_mode="human")

# Calculate the number of states
num_states = (env.observation_space['agent'].high[0] + 1) * (env.observation_space['agent'].high[1] + 1)

# Create Q-table with random values
qtable = np.random.rand(num_states, env.action_space.n).tolist()

# Hyperparameters
episodes = 50
gamma = 0.1
epsilon = 0.08
decay = 0.1

# Lists to store episode returns and steps per episode
episode_returns = []
steps_per_episode = []

#  Loop for training
for i in range(episodes):
    state_dict, info = env.reset()
    state = (state_dict['agent'][0] + 1) * (state_dict['agent'][1] + 1) - 1
    steps = 0
    episode_return = 0
    done = False

    while not done:
        os.system('clear')
        print("episode #", i+1, "/", episodes)
        env.render()
        time.sleep(0.05)

        # Increment steps
        steps += 1

        # Exploration-exploitation trade-off
        if np.random.uniform() < epsilon:
            action = env.action_space.sample()
        else:
            action = qtable[state].index(max(qtable[state]))

        # Take action
        next_state_dict, reward, done, _, info = env.step(action)
        next_state = (next_state_dict['agent'][0] + 1) * (next_state_dict['agent'][1] + 1) - 1

        # Update Q-table using Bellman equation
        qtable[state][action] = reward + gamma * max(qtable[next_state])

        # Update state and episode return
        state = next_state
        episode_return += reward

    # Decay epsilon
    epsilon -= decay * epsilon

    # Append episode return and steps per episode to lists
    episode_returns.append(episode_return)
    steps_per_episode.append(steps)

    print("\nDone in", steps, "steps")
    print("\nEpisode return:", episode_return)


# Plot episode returns and steps per episode
plt.figure(figsize=(10, 5))
plt.subplot(2, 1, 1)
plt.plot(range(1, episodes + 1), episode_returns)
plt.title('Episode Returns')
plt.xlabel('Episode')
plt.ylabel('Return')

plt.subplot(2, 1, 2)
plt.plot(range(1, episodes + 1), steps_per_episode)
plt.title('Steps per Episode')
plt.xlabel('Episode')
plt.ylabel('Steps')

plt.tight_layout()
plt.show()

env.close()
