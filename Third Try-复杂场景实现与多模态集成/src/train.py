"""
第三阶段TD3驾驶智能体训练入口
提供完整的训练流程，包括课程学习、模型保存、可视化等功能
"""

import argparse
from copy import deepcopy
from datetime import datetime
import json
import os
import random
import sys
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
from tqdm import tqdm

# 设置项目路径，确保模块可以正确导入
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.agent.agent import Agent
from src.config import AGENT_CONFIG, ENV_CONFIG, RENDER_CONFIG, TRAINING_CONFIG, clone_configs
from src.environment.environment import CarEnvironment
from src.utils import MetricsRecorder


def set_seed(seed: int) -> None:
    """
    设置全局随机种子，确保实验可复现

    参数:
        seed: 随机种子值
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(
    env_config: Optional[Dict] = None,
    agent_config: Optional[Dict] = None,
    training_config: Optional[Dict] = None,
    save_dir: str = "logs",
    visualize: bool = False,
    model_path: Optional[str] = None,
    seed: int = 42,
) -> Dict:
    """
    主训练函数

    训练流程：
    1. 初始化环境和智能体
    2. 进行课程学习（难度渐进增加）
    3. 定期评估和保存模型
    4. 记录训练指标并可视化

    参数:
        env_config: 环境配置（可选）
        agent_config: 智能体配置（可选）
        training_config: 训练配置（可选）
        save_dir: 日志和模型保存目录
        visualize: 是否生成训练视频
        model_path: 预训练模型路径（用于继续训练）
        seed: 随机种子

    返回:
        包含训练结果的字典
    """
    # 设置随机种子
    set_seed(seed)

    # 获取并合并配置
    configs = clone_configs()
    env_config = deepcopy(env_config or configs["env_config"])
    user_agent_config = agent_config or {}
    agent_config = {**configs["agent_config"], **user_agent_config}
    training_config = {**configs["training_config"], **(training_config or {})}
    render_config = {**configs["render_config"], **env_config.get("render_config", {})}
    env_config["render_config"] = render_config

    # 创建保存目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(save_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True)

    # 保存配置到文件，方便后续分析
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "env_config": env_config,
                "agent_config": agent_config,
                "training_config": training_config,
                "seed": seed,
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )

    # 初始化训练环境和评估环境（分开使用避免相互影响）
    env = CarEnvironment(env_config)
    eval_env = CarEnvironment(env_config)

    # 如果提供了预训练模型，加载并获取其配置
    if model_path:
        checkpoint_config = Agent.checkpoint_metadata(model_path)
        agent_config = {**configs["agent_config"], **checkpoint_config, **user_agent_config}

    # 初始化TD3智能体
    agent = Agent(
        radar_dim=env.radar_rays,
        vector_dim=env.vector_dim,
        visual_shape=(3, env.observation_size[1], env.observation_size[0]),
        action_dim=env.action_space.shape[0],
        warmup_steps=training_config["warmup_steps"],
        **agent_config,
    )

    # 如果提供了预训练模型，加载模型参数
    if model_path:
        agent.load(model_path)

    # 初始化指标记录器
    metrics = MetricsRecorder()
    log_path = os.path.join(run_dir, "train_log.csv")
    best_score = (-float("inf"), -float("inf"))  # (成功率, 平均奖励)

    # 创建日志文件并开始训练
    with open(log_path, "w", encoding="utf-8") as log_file:
        # 写入CSV头部
        log_file.write(
            "episode,stage,reward,length,critic_loss,actor_loss,collision,target_reached,final_distance,min_distance,"
            "eval_reward,eval_success_rate\n"
        )

        # 使用进度条显示训练进度
        progress_bar = tqdm(range(training_config["num_episodes"]), desc="训练中")

        for episode in progress_bar:
            metrics.start_episode()

            # 根据课程学习计划确定当前难度阶段
            stage_id = resolve_stage(episode, training_config["curriculum_schedule"])

            # 重置环境，设置对应的课程阶段
            state, info = env.reset(options={"curriculum_stage": stage_id})
            episode_reward = 0.0
            critic_losses: List[float] = []
            actor_losses: List[float] = []
            collision_count = 0

            # 判断是否需要录制本回合视频
            capture_video = visualize and ((episode + 1) % training_config["video_interval"] == 0)
            frames = []
            final_info = info

            # 执行回合
            for _ in range(training_config["max_steps"]):
                # 选择动作（训练模式下会添加探索噪声）
                action = agent.select_action(state)

                # 执行动作，获取下一状态和奖励
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                final_info = info

                # 记录奖励并存储经验到回放池
                metrics.add_step_reward(reward)
                agent.replay_buffer.push(state, action, reward, next_state, float(terminated))

                # 当回放池有足够数据且预热完成后，执行学习更新
                if len(agent.replay_buffer) >= agent.batch_size and agent.total_steps >= agent.warmup_steps:
                    for _ in range(training_config["updates_per_step"]):
                        critic_loss, actor_loss = agent.update()
                        if critic_loss is not None:
                            critic_losses.append(critic_loss)
                        if actor_loss is not None:
                            actor_losses.append(actor_loss)

                # 如果需要录制视频，保存当前帧
                if capture_video:
                    frame_info = dict(info)
                    frame_info["episode_reward"] = episode_reward + reward
                    frames.append(env.render(frame_info))

                # 更新累计奖励和碰撞计数
                episode_reward += reward
                if info["collision"]:
                    collision_count += 1

                # 更新状态，检查是否结束
                state = next_state
                if done:
                    break

            # 计算平均损失
            avg_critic_loss = float(np.mean(critic_losses)) if critic_losses else None
            avg_actor_loss = float(np.mean(actor_losses)) if actor_losses else None

            # 记录回合数据
            metrics.add_episode_data(
                reward=episode_reward,
                length=final_info["step"],
                critic_loss=avg_critic_loss,
                actor_loss=avg_actor_loss,
                collisions=collision_count,
                target_reached=final_info["target_reached"],
                final_distance=final_info["distance_to_target"],
                min_distance=final_info["min_distance_to_target"],
                stage_id=stage_id,
            )

            # 定期执行评估
            eval_stats = None
            if (episode + 1) % training_config["eval_interval"] == 0:
                eval_stats = evaluate(
                    env=eval_env,
                    agent=agent,
                    stage_id=stage_id,
                    num_episodes=training_config["eval_episodes"],
                    max_steps=training_config["max_steps"],
                )

                # 如果评估结果更好，保存为最佳模型
                candidate_score = (eval_stats["success_rate"], eval_stats["mean_reward"])
                if candidate_score > best_score:
                    best_score = candidate_score
                    agent.save(os.path.join(run_dir, "best_model.pt"))

            # 如果需要，保存训练视频
            if capture_video and frames:
                save_video(frames, os.path.join(run_dir, f"train_episode_{episode + 1}.mp4"))

            # 定期保存模型和指标
            if (episode + 1) % training_config["save_interval"] == 0:
                agent.save(os.path.join(run_dir, f"checkpoint_episode_{episode + 1}.pt"))
                metrics.save_metrics(os.path.join(run_dir, "metrics"))
                metrics.plot_metrics(os.path.join(run_dir, "plots"))

            # 更新进度条显示
            progress_bar.set_postfix(
                stage=stage_id,
                reward=f"{episode_reward:.1f}",
                success=int(final_info["target_reached"]),
                distance=f"{final_info['distance_to_target']:.1f}",
            )

            # 写入日志行
            eval_reward_text = ""
            eval_success_text = ""
            if eval_stats is not None:
                eval_reward_text = f"{eval_stats['mean_reward']:.4f}"
                eval_success_text = f"{eval_stats['success_rate']:.4f}"

            log_file.write(
                f"{episode + 1},{stage_id},{episode_reward:.4f},{final_info['step']},{avg_critic_loss or 0.0:.6f},"
                f"{avg_actor_loss or 0.0:.6f},{collision_count},{final_info['target_reached']},"
                f"{final_info['distance_to_target']:.4f},{final_info['min_distance_to_target']:.4f},"
                f"{eval_reward_text},{eval_success_text}\n"
            )
            log_file.flush()

    # 训练完成，保存最终模型和指标
    agent.save(os.path.join(run_dir, "final_model.pt"))
    metrics_payload = metrics.save_metrics(os.path.join(run_dir, "final_metrics"))
    metrics.plot_metrics(os.path.join(run_dir, "final_plots"))

    # 生成可视化总结
    env.visualizer.create_heatmap(os.path.join(run_dir, "final_heatmap.png"))
    env.visualizer.create_episode_summary(
        rewards=metrics.episode_rewards,
        lengths=metrics.episode_lengths,
        collision_counts=metrics.collision_counts,
        target_reached=metrics.target_reached,
        save_dir=run_dir,
    )
    write_summary(run_dir, metrics_payload["summary"])

    return {
        "run_dir": run_dir,
        "metrics": metrics_payload,
        "best_score": best_score,
    }


def evaluate(
    env: CarEnvironment,
    agent: Agent,
    stage_id: int,
    num_episodes: int,
    max_steps: int,
) -> Dict[str, float]:
    """
    评估智能体性能

    在无探索噪声的情况下测试智能体的表现

    参数:
        env: 评估环境
        agent: 待评估的智能体
        stage_id: 课程学习阶段
        num_episodes: 评估回合数
        max_steps: 每回合最大步数

    返回:
        包含评估指标的字典
    """
    rewards = []
    lengths = []
    successes = []
    final_distances = []

    for _ in range(num_episodes):
        state, info = env.reset(options={"curriculum_stage": stage_id})
        episode_reward = 0.0
        final_info = info

        # 执行回合，使用评估模式（无探索噪声）
        for _ in range(max_steps):
            action = agent.select_action(state, evaluate=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            final_info = info
            state = next_state
            if terminated or truncated:
                break

        # 记录结果
        rewards.append(episode_reward)
        lengths.append(final_info["step"])
        successes.append(bool(final_info["target_reached"]))
        final_distances.append(float(final_info["distance_to_target"]))

    # 计算统计指标
    return {
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "std_reward": float(np.std(rewards)) if len(rewards) > 1 else 0.0,
        "mean_length": float(np.mean(lengths)) if lengths else 0.0,
        "success_rate": float(np.mean(successes)) if successes else 0.0,
        "mean_final_distance": float(np.mean(final_distances)) if final_distances else 0.0,
    }


def resolve_stage(episode: int, schedule: List[Dict]) -> int:
    """
    根据回合数解析课程学习阶段

    课程学习：难度随训练进度逐渐增加
    从简单的固定目标、无障碍物场景逐步过渡到复杂的动态环境

    参数:
        episode: 当前回合数
        schedule: 课程学习计划

    返回:
        当前应该使用的难度阶段ID
    """
    stage_id = 0
    for item in schedule:
        if episode >= item["episode"]:
            stage_id = item["stage"]
    return stage_id


def save_video(frames: List[np.ndarray], path: str, fps: int = 30) -> None:
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


def write_summary(save_dir: str, summary: Dict) -> None:
    """
    写入训练总结文本文件

    参数:
        save_dir: 保存目录
        summary: 总结数据字典
    """
    with open(os.path.join(save_dir, "summary.txt"), "w", encoding="utf-8") as handle:
        handle.write("训练总结\n")
        handle.write("================\n\n")
        handle.write(f"平均奖励: {summary['mean_reward']:.2f}\n")
        handle.write(f"奖励标准差: {summary['std_reward']:.2f}\n")
        handle.write(f"平均回合长度: {summary['mean_length']:.2f}\n")
        handle.write(f"平均最终距离: {summary['mean_final_distance']:.2f}\n")
        handle.write(f"平均最小距离: {summary['mean_min_distance']:.2f}\n")
        handle.write(f"成功率: {summary['success_rate'] * 100:.1f}%\n")
        handle.write(f"总碰撞次数: {summary['total_collisions']}\n")


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数

    支持自定义训练参数，方便进行实验调参
    """
    parser = argparse.ArgumentParser(description="训练第三阶段TD3智能体")
    parser.add_argument("--num_episodes", type=int, default=TRAINING_CONFIG["num_episodes"])
    parser.add_argument("--max_steps", type=int, default=TRAINING_CONFIG["max_steps"])
    parser.add_argument("--eval_interval", type=int, default=TRAINING_CONFIG["eval_interval"])
    parser.add_argument("--eval_episodes", type=int, default=TRAINING_CONFIG["eval_episodes"])
    parser.add_argument("--save_interval", type=int, default=TRAINING_CONFIG["save_interval"])
    parser.add_argument("--video_interval", type=int, default=TRAINING_CONFIG["video_interval"])
    parser.add_argument("--warmup_steps", type=int, default=TRAINING_CONFIG["warmup_steps"])
    parser.add_argument("--updates_per_step", type=int, default=TRAINING_CONFIG["updates_per_step"])
    parser.add_argument("--save_dir", type=str, default="logs")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--disable_visual", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=AGENT_CONFIG["learning_rate"])
    parser.add_argument("--batch_size", type=int, default=AGENT_CONFIG["batch_size"])
    parser.add_argument("--hidden_dim", type=int, default=AGENT_CONFIG["hidden_dim"])
    parser.add_argument("--buffer_size", type=int, default=AGENT_CONFIG["buffer_size"])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # 构建配置字典
    env_config = deepcopy(ENV_CONFIG)
    env_config["render_config"] = deepcopy(RENDER_CONFIG)

    agent_config = {
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "hidden_dim": args.hidden_dim,
        "buffer_size": args.buffer_size,
        "use_visual": not args.disable_visual,
    }

    training_config = {
        "num_episodes": args.num_episodes,
        "max_steps": args.max_steps,
        "eval_interval": args.eval_interval,
        "eval_episodes": args.eval_episodes,
        "save_interval": args.save_interval,
        "video_interval": args.video_interval,
        "warmup_steps": args.warmup_steps,
        "updates_per_step": args.updates_per_step,
        "curriculum_schedule": deepcopy(TRAINING_CONFIG["curriculum_schedule"]),
    }

    # 开始训练
    train(
        env_config=env_config,
        agent_config=agent_config,
        training_config=training_config,
        save_dir=args.save_dir,
        visualize=args.render,
        model_path=args.model_path,
        seed=args.seed,
    )