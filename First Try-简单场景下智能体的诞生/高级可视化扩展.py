"""
高级可视化扩展模块
提供Q值热力图等额外可视化功能

主要功能：
- 增强版评估：带轨迹渲染的视频生成
- Q值热力图：展示各位置下不同动作的Q值分布
"""

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

# 设置随机种子确保结果可复现
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# 创建输出目录
os.makedirs("models", exist_ok=True)
os.makedirs("videos", exist_ok=True)


def evaluate(agent, num_episodes=20, render=True, save_video=False):
    """
    评估智能体性能

    参数:
        agent: 待评估的智能体
        num_episodes: 评估回合数
        render: 是否实时渲染显示
        save_video: 是否保存评估视频

    返回:
        平均奖励值
    """
    env = Environment()
    rewards = []

    if save_video:
        # 视频保存模式
        fig, ax = plt.subplots(figsize=(10, 10))
        writer = animation.FFMpegWriter(fps=15)

        with writer.saving(fig, 'videos/evaluation_complex.mp4', dpi=100):
            for episode in tqdm(range(num_episodes), desc="Evaluating Episodes", unit="episode"):
                state = env.reset()
                episode_reward = 0
                step = 0
                trajectory = []  # 记录轨迹

                while True:
                    # 选择并执行动作
                    action = agent.select_action(state)
                    next_state, reward, done, _ = env.step(action)

                    # 记录位置
                    trajectory.append((env.car.x, env.car.y))

                    # 渲染环境并绘制轨迹
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
        # 非视频保存模式
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
                    # 实时渲染
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

    # 打印平均结果
    average_reward = np.mean(rewards)
    print(f"Average Reward over {num_episodes} episodes: {average_reward:.2f}")
    return average_reward


def visualize_q_values(agent, env):
    """
    Q值热力图可视化

    在2D热力图中展示各位置下不同动作的Q值分布
    可以直观理解智能体对不同位置的价值估计

    参数:
        agent: 智能体实例
        env: 环境实例
    """
    # 创建位置网格
    x_range = np.linspace(0, ENV_WIDTH, 20)
    y_range = np.linspace(0, ENV_HEIGHT, 20)
    q_values = np.zeros((len(x_range), len(y_range), ACTION_SPACE))

    # 计算各位置的Q值
    for i, x in enumerate(x_range):
        for j, y in enumerate(y_range):
            # 设置小车位置
            env.car.x, env.car.y = x, y
            state = env._get_state()

            # 获取各动作的Q值
            with torch.no_grad():
                q_values[i, j] = agent.policy_net(state).numpy()

    # 绘制三个动作的热力图
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
    # 初始化环境和智能体
    env = Environment()
    state_size = NUM_SENSORS + 2
    agent = Agent(state_size, ACTION_SPACE)

    # 加载预训练模型
    agent.load("models/best_model.pth")
    print("预训练模型已成功加载。")

    # 执行评估
    print("\n开始评估...")
    evaluate(agent, num_episodes=5, render=True, save_video=True)

    # 可视化Q值热力图
    print("\n可视化Q值热力图...")
    visualize_q_values(agent, env)