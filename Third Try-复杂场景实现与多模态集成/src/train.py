import os
import numpy as np
import torch
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
import json
import argparse
import time
from src.environment.environment import CarEnvironment
from src.agent.agent import Agent
from src.config.environment_config import ENV_CONFIG, RENDER_CONFIG
from src.utils import MetricsRecorder

def train(env_config: dict = None,
          agent_config: dict = None,
          training_config: dict = None,
          save_dir: str = 'logs',
          visualize: bool = False,
          save_video_interval: int = 100,
          video_fps: int = 30,
          video_quality: int = 95,
          num_episodes: int = None,
          max_steps: int = None,
          eval_interval: int = None,
          eval_episodes: int = None,
          model_path: str = None,
          render_resolution: tuple = (800, 600)):
    
    # 创建保存目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(save_dir, timestamp)
    os.makedirs(save_dir, exist_ok=True)
    
    # 检查并更新渲染配置
    if env_config is None:
        env_config = ENV_CONFIG.copy()
    
    # 确保有RENDER_CONFIG
    if 'render_config' not in env_config:
        env_config['render_config'] = RENDER_CONFIG.copy()
    
    # 更新渲染分辨率
    if render_resolution:
        env_config['render_config'] = env_config.get('render_config', {}).copy()
        env_config['render_config']['visual_size'] = render_resolution
    
    # 保存配置
    config = {
        'env_config': env_config,
        'agent_config': agent_config or {},
        'training_config': training_config or {},
        'render_resolution': render_resolution
    }
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)
    
    # 创建环境
    env = CarEnvironment(env_config)
    
    # 创建智能体
    agent = Agent(
        radar_dim=env.radar_rays,
        visual_shape=(3, 256, 256),  # 更大的视觉输入
        action_dim=2,
        **agent_config
    )
    
    # 加载预训练模型（如果提供）
    if model_path:
        agent.load(model_path)
        print(f"Loaded pretrained model from: {model_path}")
    
    # 创建指标记录器
    metrics_recorder = MetricsRecorder()
    
    # 训练参数
    training_config = training_config or {}
    # 命令行参数优先级高于配置文件
    if num_episodes is not None:
        training_config['num_episodes'] = num_episodes
    if max_steps is not None:
        training_config['max_steps'] = max_steps
    if eval_interval is not None:
        training_config['eval_interval'] = eval_interval
    if eval_episodes is not None:
        training_config['eval_episodes'] = eval_episodes
        
    num_episodes = training_config.get('num_episodes', 1000)
    max_steps = training_config.get('max_steps', 200)
    eval_interval = training_config.get('eval_interval', 10)
    save_interval = training_config.get('save_interval', 100)
    
    # 可视化设置
    frames_buffer = []
    
    # 训练记录
    best_eval_reward = float('-inf')
    
    # 创建日志文件
    log_path = os.path.join(save_dir, 'train_log.txt')
    with open(log_path, 'w') as log_file:
        log_file.write("Episode,Reward,Length,ValueLoss,PolicyLoss,Steps,EvalReward,Collisions,TargetReached\n")
        
        # 训练循环
        for episode in tqdm(range(num_episodes), desc='训练进度'):
            state, _ = env.reset()
            episode_reward = 0
            episode_length = 0
            episode_value_loss = 0
            episode_policy_loss = 0
            update_count = 0
            collision_count = 0
            target_reached = False
            
            for step in range(max_steps):
                # 选择动作
                action = agent.select_action(state)
                
                # 执行动作
                next_state, reward, done, truncated, info = env.step(action)
                
                # 记录步级指标
                metrics_recorder.add_step_data(
                    reward=reward,
                    distance=info['distance_to_target'],
                    speed=info['car_speed'],
                    rotation=info['car_steering']
                )
                
                # 统计碰撞
                if info.get('collision', False):
                    collision_count += 1
                
                # 检查是否到达目标
                if info.get('target_reached', False):
                    target_reached = True
                
                # 保存可视化帧
                if visualize and episode % save_video_interval == 0:
                    frame = env.render()
                    # 添加信息显示
                    frame = add_info_to_frame(
                        frame, 
                        episode_reward, 
                        step + 1, 
                        action, 
                        info['distance_to_target'],
                        info.get('car_speed', 0),
                        info.get('car_steering', 0),
                        info.get('rotation_penalty', False)
                    )
                    
                    # 调整帧大小以匹配渲染分辨率
                    if render_resolution:
                        try:
                            if frame is not None and frame.size > 0:
                                frame = cv2.resize(frame, render_resolution, interpolation=cv2.INTER_AREA)
                        except Exception as e:
                            print(f"警告: 调整帧大小时出错: {e}")
                    
                    frames_buffer.append(frame)
                
                # 存储经验
                agent.replay_buffer.push(state, action, reward, next_state, done)
                
                # 更新网络
                value_loss, policy_loss = agent.update()
                
                # 更新状态
                state = next_state
                episode_reward += reward
                episode_length += 1
                
                # 记录损失
                if value_loss is not None:
                    episode_value_loss += value_loss
                    episode_policy_loss += policy_loss
                    update_count += 1
                
                if done or truncated:
                    break
            
            # 计算平均损失
            avg_value_loss = episode_value_loss / update_count if update_count > 0 else 0
            avg_policy_loss = episode_policy_loss / update_count if update_count > 0 else 0
            
            # 记录回合指标
            metrics_recorder.add_episode_data(
                reward=episode_reward,
                length=episode_length,
                value_loss=avg_value_loss,
                policy_loss=avg_policy_loss,
                collisions=collision_count,
                target_reached=target_reached
            )
            
            # 保存训练视频
            if visualize and episode % save_video_interval == 0 and frames_buffer:
                video_path = os.path.join(save_dir, f'train_episode_{episode + 1}.mp4')
                save_video(frames_buffer, video_path, video_fps, video_quality)
                frames_buffer = []  # 清空缓冲区
            
            # 控制台输出
            print(f"\n[Episode {episode+1:04d}/{num_episodes}]")
            print(f"  Reward: {episode_reward:.2f}")
            print(f"  Length: {episode_length}")
            print(f"  Steps: {step+1}/{max_steps}")
            print(f"  Value Loss: {avg_value_loss:.4f}")
            print(f"  Policy Loss: {avg_policy_loss:.4f}")
            print(f"  Collisions: {collision_count}")
            print(f"  Target Reached: {'Yes' if target_reached else 'No'}")
            print(f"  Buffer Size: {len(agent.replay_buffer)}")
            
            # 评估
            eval_reward = 0
            if (episode + 1) % eval_interval == 0:
                print(f"  [Eval] Starting evaluation...")
                eval_reward = evaluate(env, agent)
                print(f"  [Eval] Reward: {eval_reward:.2f}")
                
                # 保存最佳模型
                if eval_reward > best_eval_reward:
                    best_eval_reward = eval_reward
                    agent.save(os.path.join(save_dir, 'best_model.pt'))
                    print(f"  [New Best] Saved best model with eval reward: {eval_reward:.2f}")
            
            # 定期保存模型
            if (episode + 1) % save_interval == 0:
                agent.save(os.path.join(save_dir, f'model_episode_{episode + 1}.pt'))
                print(f"  [Save] Saved model at episode {episode + 1}")
                
                # 保存并绘制当前指标
                metrics_recorder.save_metrics(os.path.join(save_dir, 'metrics'))
                metrics_recorder.plot_metrics(os.path.join(save_dir, 'plots'))
                
                # 创建热图
                env.visualizer.create_heatmap(os.path.join(save_dir, f'heatmap_episode_{episode + 1}.png'))
            
            # 写入日志
            log_file.write(f"{episode+1},{episode_reward:.2f},{episode_length},{avg_value_loss:.4f},"
                          f"{avg_policy_loss:.4f},{step+1},{eval_reward if (episode + 1) % eval_interval == 0 else ''},"
                          f"{collision_count},{target_reached}\n")
            log_file.flush()
    
    # 保存最终模型和指标
    agent.save(os.path.join(save_dir, 'final_model.pt'))
    metrics_recorder.save_metrics(os.path.join(save_dir, 'final_metrics'))
    metrics_recorder.plot_metrics(os.path.join(save_dir, 'final_plots'))
    
    # 创建热图
    env.visualizer.create_heatmap(os.path.join(save_dir, 'final_heatmap.png'))
    
    # 创建训练总结
    env.visualizer.create_episode_summary(
        rewards=metrics_recorder.episode_rewards,
        lengths=metrics_recorder.episode_lengths,
        collision_counts=metrics_recorder.collision_counts,
        target_reached=metrics_recorder.target_reached,
        save_dir=save_dir
    )
    
    print(f"\nTraining completed. Best eval reward: {best_eval_reward:.2f}")
    print(f"Final metrics saved to: {os.path.join(save_dir, 'final_metrics')}")
    print(f"Final plots saved to: {os.path.join(save_dir, 'final_plots')}")
    
    return agent, best_eval_reward

def evaluate(env, agent, num_episodes=5, max_steps=500):
    eval_rewards = []
    print(f"    Running {num_episodes} evaluation episodes...")
    
    for i in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        step_count = 0
        
        while not done and step_count < max_steps:
            action = agent.select_action(state, evaluate=True)
            state, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
            done = done or truncated
            step_count += 1
        
        eval_rewards.append(episode_reward)
        print(f"    Eval episode {i+1}/{num_episodes}: Reward = {episode_reward:.2f}, Steps = {step_count}")
    
    mean_reward = np.mean(eval_rewards)
    std_reward = np.std(eval_rewards)
    print(f"    Evaluation complete. Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
    return mean_reward

def add_info_to_frame(frame, reward, length, action, distance, speed, steering, rotation_penalty):
    """在帧上添加信息显示"""
    try:
        # 确保帧有效
        if frame is None or frame.size == 0:
            # 创建一个空白帧
            frame = np.ones((600, 800, 3), dtype=np.uint8) * 220
            
        # 检查帧的形状和类型
        if len(frame.shape) != 3 or frame.shape[2] != 3:
            # 转换为三通道图像
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            else:
                # 创建一个空白帧
                frame = np.ones((600, 800, 3), dtype=np.uint8) * 220
                
        # 确保数据类型为uint8
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)
        
        # 创建信息显示区域
        frame_width = frame.shape[1]
        info_height = max(100, int(frame.shape[0] * 0.15))  # 根据帧高度动态调整信息区高度
        info_frame = np.ones((info_height, frame_width, 3), dtype=np.uint8) * 240
        
        # 计算字体大小和线宽（根据帧宽度调整）
        font_scale = max(0.6, min(1.0, frame_width / 800))
        line_thickness = max(1, int(font_scale * 2))
        
        # 添加文本信息 - 左边显示奖励和步数
        cv2.putText(info_frame, f"奖励: {reward:.2f}", (20, int(info_height*0.3)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), line_thickness)
        cv2.putText(info_frame, f"步数: {length}", (20, int(info_height*0.7)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), line_thickness)
        
        # 中间显示动作和距离
        mid_x = int(frame_width * 0.35)
        cv2.putText(info_frame, f"动作: [{action[0]:.2f}, {action[1]:.2f}]", (mid_x, int(info_height*0.3)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), line_thickness)
        cv2.putText(info_frame, f"目标距离: {distance:.2f}", (mid_x, int(info_height*0.7)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), line_thickness)
        
        # 右边显示速度和转向状态
        status_x = int(frame_width * 0.7)
        cv2.putText(info_frame, f"速度: {speed:.2f}", (status_x, int(info_height*0.3)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), line_thickness)
        
        # 旋转惩罚状态
        cv2.putText(info_frame, "转向:", (status_x, int(info_height*0.7)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), line_thickness)
        rotation_color = (0, 0, 255) if rotation_penalty else (0, 255, 0)
        indicator_radius = max(5, int(font_scale * 8))
        indicator_x = status_x + int(font_scale * 120)
        cv2.circle(info_frame, (indicator_x, int(info_height*0.65)), indicator_radius, rotation_color, -1)
        
        # 结合帧和信息区域
        result = np.vstack([frame, info_frame])
        
        # 添加边框
        cv2.rectangle(result, (0, 0), (result.shape[1]-1, result.shape[0]-1), (0, 0, 0), line_thickness)
        
        return result
    except Exception as e:
        print(f"  警告: 添加信息到帧时出错: {e}")
        # 返回原始帧
        return frame

def save_video(frames, path, fps=30, quality=95):
    """保存视频，使用高质量设置"""
    if not frames:
        print("  警告: 没有帧可以保存")
        return
    
    try:
        # 确保所有帧大小一致
        first_frame = frames[0]
        if first_frame is None or first_frame.size == 0:
            print("  警告: 第一帧无效，无法保存视频")
            return
            
        height, width = first_frame.shape[:2]
        
        # 确保所有帧的大小都匹配
        valid_frames = []
        for i, frame in enumerate(frames):
            if frame is None or frame.size == 0:
                print(f"  警告: 跳过第{i+1}帧 (无效)")
                continue
                
            if frame.shape[:2] != (height, width):
                try:
                    # 调整大小
                    frame = cv2.resize(frame, (width, height))
                except Exception as e:
                    print(f"  警告: 跳过第{i+1}帧 (无法调整大小: {e})")
                    continue
                    
            # 确保是3通道 uint8 类型
            if len(frame.shape) != 3 or frame.shape[2] != 3:
                try:
                    if len(frame.shape) == 2:
                        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                except Exception as e:
                    print(f"  警告: 跳过第{i+1}帧 (无法转换颜色: {e})")
                    continue
                    
            if frame.dtype != np.uint8:
                try:
                    frame = frame.astype(np.uint8)
                except Exception as e:
                    print(f"  警告: 跳过第{i+1}帧 (无法转换数据类型: {e})")
                    continue
                
            valid_frames.append(frame)
        
        if not valid_frames:
            print("  警告: 处理后没有有效帧，无法保存视频")
            return
            
        # 创建视频
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(path, fourcc, fps, (width, height))
        
        # 写入帧
        for frame in valid_frames:
            out.write(frame)
        
        # 释放资源
        out.release()
        
        print(f"  成功保存视频，包含 {len(valid_frames)} 帧，分辨率: {width}x{height}")
    except Exception as e:
        print(f"  错误: 保存视频时发生异常: {e}")

if __name__ == '__main__':
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='训练自动驾驶智能体')
    
    # 训练参数
    parser.add_argument('--num_episodes', type=int, default=1000,
                      help='训练的总回合数')
    parser.add_argument('--max_steps', type=int, default=500,
                      help='每个回合的最大步数')
    parser.add_argument('--eval_interval', type=int, default=10,
                      help='评估间隔（回合数）')
    parser.add_argument('--eval_episodes', type=int, default=5,
                      help='每次评估的回合数')
    parser.add_argument('--save_interval', type=int, default=100,
                      help='模型保存间隔（回合数）')
    
    # 模型参数
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                      help='学习率')
    parser.add_argument('--gamma', type=float, default=0.99,
                      help='折扣因子')
    parser.add_argument('--buffer_size', type=int, default=100000,
                      help='经验回放缓冲区大小')
    parser.add_argument('--batch_size', type=int, default=64,
                      help='批量大小')
    parser.add_argument('--tau', type=float, default=0.005,
                      help='目标网络软更新系数')
    parser.add_argument('--hidden_dim', type=int, default=64,
                      help='隐藏层维度')
                      
    # 其他参数
    parser.add_argument('--env_config', type=str, default=None,
                      help='环境配置文件路径')
    parser.add_argument('--save_dir', type=str, default='logs',
                      help='保存模型和日志的目录')
    parser.add_argument('--model_path', type=str, default=None,
                      help='预训练模型路径')
    parser.add_argument('--render', action='store_true',
                      help='是否渲染环境')
    parser.add_argument('--render_interval', type=int, default=10,
                      help='渲染间隔（回合数）')
    parser.add_argument('--save_video', action='store_true',
                      help='是否保存视频')
    parser.add_argument('--video_interval', type=int, default=100,
                      help='保存视频的间隔（回合数）')
    parser.add_argument('--render_width', type=int, default=800,
                      help='渲染宽度')
    parser.add_argument('--render_height', type=int, default=600,
                      help='渲染高度')
    
    # 解析参数
    args = parser.parse_args()
    
    # 加载环境配置
    env_config = ENV_CONFIG
    if args.env_config:
        with open(args.env_config, 'r') as f:
            env_config = json.load(f)
    
    # 智能体配置
    agent_config = {
        'learning_rate': args.learning_rate,
        'gamma': args.gamma,
        'buffer_size': args.buffer_size,
        'batch_size': args.batch_size,
        'tau': args.tau,
        'hidden_dim': args.hidden_dim
    }
    
    # 渲染配置
    render_config = {
        'render': args.render,
        'render_interval': args.render_interval,
        'save_video': args.save_video,
        'video_interval': args.video_interval,
        'render_resolution': (args.render_width, args.render_height)
    }
    
    # 运行训练
    train(
        env_config=env_config,
        agent_config=agent_config,
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        save_dir=args.save_dir,
        model_path=args.model_path,
        visualize=args.render,
        save_video_interval=args.video_interval,
        render_resolution=(args.render_width, args.render_height)
    ) 