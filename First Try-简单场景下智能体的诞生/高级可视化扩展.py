import os
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, Circle, Arrow
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import seaborn as sns
from main import *

# 确保结果可以复现
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

os.makedirs("models", exist_ok=True)
os.makedirs("videos", exist_ok=True)


def evaluate(agent, num_episodes=20, render=True, save_video=False):
    env = Environment()
    rewards = []

    if save_video:
        fig, ax = plt.subplots(figsize=(10, 10))
        writer = animation.FFMpegWriter(fps=15)
        with writer.saving(fig, 'videos/evaluation_complex.mp4', dpi=100):
            for episode in tqdm(range(num_episodes), desc="Evaluating Episodes", unit="episode"):
                state = env.reset()
                episode_reward = 0
                step = 0
                trajectory = []

                while True:
                    action = agent.select_action(state)
                    next_state, reward, done, _ = env.step(action)

                    trajectory.append((env.car.x, env.car.y))

                    env.render(ax=ax, step=step)
                    ax.plot(*zip(*trajectory), color='purple', linestyle='--', linewidth=1)
                    writer.grab_frame()

                    state = next_state
                    episode_reward += reward
                    step += 1

                    if done:
                        break

                rewards.append(episode_reward)
                print(f"Evaluation Episode: {episode + 1}, Reward: {episode_reward:.2f}")
        plt.close(fig)
    else:
        fig, ax = None, None
        if render:
            fig, ax = plt.subplots(figsize=(10, 10))

        for episode in tqdm(range(num_episodes), desc="Evaluating Episodes", unit="episode"):
            state = env.reset()
            episode_reward = 0
            step = 0
            trajectory = []

            while True:
                action = agent.select_action(state)
                next_state, reward, done, _ = env.step(action)

                trajectory.append((env.car.x, env.car.y))

                if render:
                    env.render(ax=ax, step=step)
                    ax.plot(*zip(*trajectory), color='purple', linestyle='--', linewidth=1)
                    plt.pause(0.01)

                state = next_state
                episode_reward += reward
                step += 1

                if done:
                    break

            rewards.append(episode_reward)
            print(f"Evaluation Episode: {episode + 1}, Reward: {episode_reward:.2f}")

        if fig is not None:
            plt.close(fig)

    average_reward = np.mean(rewards)
    print(f"Average Reward over {num_episodes} episodes: {average_reward:.2f}")
    return average_reward

def visualize_q_values(agent, env):
    x_range = np.linspace(0, ENV_WIDTH, 20)
    y_range = np.linspace(0, ENV_HEIGHT, 20)
    q_values = np.zeros((len(x_range), len(y_range), ACTION_SPACE))

    for i, x in enumerate(x_range):
        for j, y in enumerate(y_range):
            env.car.x, env.car.y = x, y
            state = env._get_state()
            with torch.no_grad():
                q_values[i, j] = agent.policy_net(state).numpy()

    fig, axes = plt.subplots(1, ACTION_SPACE, figsize=(15, 5))
    for action in range(ACTION_SPACE):
        sns.heatmap(q_values[:, :, action], ax=axes[action], cmap='viridis')
        axes[action].set_title(f"Q-values for Action {action}")
        axes[action].set_xticks(np.arange(0, len(y_range), 5))
        axes[action].set_xticklabels(y_range[::5].astype(int))
        axes[action].set_yticks(np.arange(0, len(x_range), 5))
        axes[action].set_yticklabels(x_range[::5].astype(int))
        axes[action].invert_yaxis()

    plt.show()

if __name__ == "__main__":
    env = Environment()
    state_size = NUM_SENSORS + 2
    agent = Agent(state_size, ACTION_SPACE)

    # 加载预训练模型
    agent.load("models/best_model.pth")
    print("预训练模型已成功加载。")

    print("\n开始评估...")
    evaluate(agent, num_episodes=5, render=True, save_video=True)

    print("\n可视化Q值热力图...")
    visualize_q_values(agent, env)