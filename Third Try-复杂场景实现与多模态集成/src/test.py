"""
第三阶段TD3智能体评估入口
用于评估已训练模型的性能，生成可视化结果
"""

import argparse
from copy import deepcopy
from datetime import datetime
import json
import os
import sys
from typing import Dict, List, Optional

import cv2
import numpy as np

# 设置项目路径
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.agent.agent import Agent
from src.config import AGENT_CONFIG, ENV_CONFIG, EVAL_CONFIG, RENDER_CONFIG
from src.environment.environment import CarEnvironment
from src.utils import MetricsRecorder


def test(
    model_path: str,
    env_config: Optional[Dict] = None,
    agent_config: Optional[Dict] = None,
    num_episodes: int = EVAL_CONFIG["episodes"],
    save_dir: str = "logs",
    curriculum_stage: int = 3,
    max_steps: int = EVAL_CONFIG["max_steps"],
    save_video: bool = True,
) -> Dict:
    """
    评估已训练模型

    评估流程：
    1. 加载模型并获取配置
    2. 在无探索噪声模式下运行多个回合
    3. 记录性能指标并生成可视化

    参数:
        model_path: 模型文件路径
        env_config: 环境配置（可选）
        agent_config: 智能体配置（可选）
        num_episodes: 评估回合数
        save_dir: 结果保存目录
        curriculum_stage: 课程学习阶段（评估难度）
        max_steps: 每回合最大步数
        save_video: 是否保存评估视频

    返回:
        包含评估结果的字典
    """
    # 合并配置
    env_config = deepcopy(env_config or ENV_CONFIG)
    env_config["render_config"] = {**deepcopy(RENDER_CONFIG), **env_config.get("render_config", {})}

    # 从模型文件中提取配置信息
    checkpoint_config = Agent.checkpoint_metadata(model_path)
    agent_config = {**AGENT_CONFIG, **checkpoint_config, **(agent_config or {})}

    # 创建结果保存目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(save_dir, f"test_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    # 保存评估配置
    with open(os.path.join(run_dir, "test_config.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "model_path": model_path,
                "env_config": env_config,
                "agent_config": agent_config,
                "num_episodes": num_episodes,
                "curriculum_stage": curriculum_stage,
                "max_steps": max_steps,
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )

    # 初始化环境和智能体
    env = CarEnvironment(env_config)
    agent = Agent(
        radar_dim=env.radar_rays,
        vector_dim=env.vector_dim,
        visual_shape=(3, env.observation_size[1], env.observation_size[0]),
        action_dim=env.action_space.shape[0],
        **agent_config,
    )

    # 加载模型参数
    agent.load(model_path)

    # 初始化指标记录器
    metrics = MetricsRecorder()
    episode_rewards: List[float] = []
    episode_lengths: List[int] = []
    collision_counts: List[int] = []
    target_reached: List[bool] = []

    # 执行评估回合
    for episode in range(num_episodes):
        metrics.start_episode()

        # 重置环境，设置评估难度
        state, info = env.reset(options={"curriculum_stage": curriculum_stage})
        episode_reward = 0.0
        frames = []
        final_info = info
        collisions = 0

        # 执行回合（评估模式，无探索噪声）
        for _ in range(max_steps):
            action = agent.select_action(state, evaluate=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            final_info = info
            episode_reward += reward
            metrics.add_step_reward(reward)

            # 如果需要保存视频，渲染当前帧
            if save_video:
                frame_info = dict(info)
                frame_info["episode_reward"] = episode_reward
                frames.append(env.render(frame_info))

            # 记录碰撞次数
            if info["collision"]:
                collisions += 1

            state = next_state
            if terminated or truncated:
                break

        # 记录回合数据
        metrics.add_episode_data(
            reward=episode_reward,
            length=final_info["step"],
            critic_loss=None,
            actor_loss=None,
            collisions=collisions,
            target_reached=final_info["target_reached"],
            final_distance=final_info["distance_to_target"],
            min_distance=final_info["min_distance_to_target"],
            stage_id=curriculum_stage,
        )

        episode_rewards.append(episode_reward)
        episode_lengths.append(final_info["step"])
        collision_counts.append(collisions)
        target_reached.append(final_info["target_reached"])

        # 保存回合视频
        if save_video and frames:
            save_video_file(frames, os.path.join(run_dir, f"episode_{episode + 1}.mp4"))

    # 保存和可视化指标
    metrics_payload = metrics.save_metrics(run_dir)
    metrics.plot_metrics(run_dir)

    # 生成轨迹热力图和回合总结
    env.visualizer.create_heatmap(os.path.join(run_dir, "heatmap.png"))
    env.visualizer.create_episode_summary(
        rewards=episode_rewards,
        lengths=episode_lengths,
        collision_counts=collision_counts,
        target_reached=target_reached,
        save_dir=run_dir,
    )

    # 计算并保存最终结果
    results = {
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "collision_counts": collision_counts,
        "target_reached": target_reached,
        "mean_reward": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
        "std_reward": float(np.std(episode_rewards)) if len(episode_rewards) > 1 else 0.0,
        "mean_length": float(np.mean(episode_lengths)) if episode_lengths else 0.0,
        "success_rate": float(np.mean(target_reached)) if target_reached else 0.0,
        "metrics_summary": metrics_payload["summary"],
    }

    with open(os.path.join(run_dir, "results.json"), "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False)

    write_summary(run_dir, results)

    return {"run_dir": run_dir, "results": results}


def save_video_file(frames: List[np.ndarray], path: str, fps: int = 30) -> None:
    """
    保存视频文件

    参数:
        frames: 视频帧列表
        path: 保存路径
        fps: 帧率
    """
    if not frames:
        return
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for frame in frames:
        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height))
        writer.write(frame.astype(np.uint8))
    writer.release()


def write_summary(save_dir: str, results: Dict) -> None:
    """
    写入评估总结文本文件

    参数:
        save_dir: 保存目录
        results: 结果数据字典
    """
    with open(os.path.join(save_dir, "summary.txt"), "w", encoding="utf-8") as handle:
        handle.write("评估总结\n")
        handle.write("==================\n\n")
        handle.write(f"平均奖励: {results['mean_reward']:.2f}\n")
        handle.write(f"奖励标准差: {results['std_reward']:.2f}\n")
        handle.write(f"平均长度: {results['mean_length']:.2f}\n")
        handle.write(f"成功率: {results['success_rate'] * 100:.1f}%\n")
        handle.write(f"总碰撞次数: {sum(results['collision_counts'])}\n")


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description="评估已训练的第三阶段模型")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--num_episodes", type=int, default=EVAL_CONFIG["episodes"])
    parser.add_argument("--curriculum_stage", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=EVAL_CONFIG["max_steps"])
    parser.add_argument("--save_dir", type=str, default="logs")
    parser.add_argument("--disable_visual", action="store_true")
    parser.add_argument("--no_video", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    agent_config = {"use_visual": not args.disable_visual}

    test(
        model_path=args.model_path,
        env_config=ENV_CONFIG,
        agent_config=agent_config,
        num_episodes=args.num_episodes,
        save_dir=args.save_dir,
        curriculum_stage=args.curriculum_stage,
        max_steps=args.max_steps,
        save_video=not args.no_video,
    )