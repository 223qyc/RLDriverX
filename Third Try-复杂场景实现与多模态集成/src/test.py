import os
import numpy as np
import torch
import cv2
from datetime import datetime
import json
import argparse
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt

# 添加当前目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment.environment import CarEnvironment
from src.agent.agent import Agent
from src.config.environment_config import ENV_CONFIG, RENDER_CONFIG
from src.utils import MetricsRecorder

def test(model_path: str,
         env_config: dict = None,
         agent_config: dict = None,
         num_episodes: int = 5,
         save_dir: str = 'logs',
         fps: int = 30,
         video_quality: int = 95,
         render_resolution: tuple = (800, 600),
         max_steps: int = 1000):
    
    # 创建保存目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(save_dir, f'test_{timestamp}')
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
        'render_resolution': render_resolution,
        'fps': fps,
        'video_quality': video_quality
    }
    
    # 保存测试配置
    with open(os.path.join(save_dir, 'test_config.json'), 'w') as f:
        json.dump(config, f, indent=4)
    
    # 创建环境
    env = CarEnvironment(env_config)
    
    # 创建智能体
    agent = Agent(
        radar_dim=env.radar_rays,
        visual_shape=(3, 256, 256),  # 使用更大的视觉输入
        action_dim=2,
        **agent_config
    )
    
    # 加载模型
    agent.load(model_path)
    
    # 创建指标记录器
    metrics = MetricsRecorder()
    
    # 测试记录
    episode_rewards = []
    episode_lengths = []
    collision_counts = []
    target_reached = []
    episode_frames = []
    
    print("\n=== 开始测试 ===")
    print(f"模型路径: {model_path}")
    print(f"测试回合数: {num_episodes}")
    print(f"视频帧率: {fps}")
    print(f"视频质量: {video_quality}")
    print(f"渲染分辨率: {render_resolution}")
    print(f"最大步数: {max_steps}")
    print("===============\n")
    
    # 测试循环
    for episode in range(num_episodes):
        print(f"\n[测试回合 {episode + 1}/{num_episodes}]")
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        frames = []
        current_collision_count = 0
        
        # 创建进度条
        progress_bar = tqdm(total=max_steps, desc="回合进度")
        
        for step in range(max_steps):
            # 选择动作
            action = agent.select_action(state, evaluate=True)
            
            # 执行动作
            next_state, reward, done, truncated, info = env.step(action)
            
            # 记录数据
            episode_reward += reward
            episode_length += 1
            
            # 更新指标记录器
            metrics.add_step_data(
                reward=reward,
                distance=info['distance_to_target'],
                speed=info['car_speed'],
                rotation=info['car_steering']
            )
            
            # 记录碰撞
            if info.get('collision', False):
                current_collision_count += 1
                done = True  # 碰撞时立即结束回合
            
            # 渲染环境 (每5帧保存一次，减少内存使用)
            if step % 5 == 0 or done or truncated:
                try:
                    frame = env.render()
                    
                    # 添加信息显示
                    frame = add_info_to_frame(
                        frame, 
                        episode_reward, 
                        episode_length, 
                        action, 
                        info['distance_to_target'],
                        info.get('car_speed', 0),
                        info.get('car_steering', 0),
                        info.get('rotation_penalty', False),
                        info.get('collision', False),
                        info.get('target_reached', False)
                    )
                    
                    # 调整大小 (安全调整)
                    if render_resolution:
                        try:
                            # 确保帧有效
                            if frame is not None and frame.size > 0:
                                frame = cv2.resize(frame, render_resolution, interpolation=cv2.INTER_AREA)
                        except Exception as e:
                            print(f"  警告: 调整帧大小时出错: {e}")
                    
                    frames.append(frame)
                except Exception as e:
                    print(f"  警告: 渲染帧时出错: {e}")
            
            # 更新状态
            state = next_state
            
            # 更新进度条
            progress_bar.update(1)
            
            # 如果回合结束则停止
            if done or truncated:
                break
                
        progress_bar.close()
        
        # 记录测试数据
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode_frames.append(frames)
        collision_counts.append(current_collision_count)
        target_reached.append(info.get('target_reached', False))
        
        # 添加回合数据到指标记录器
        metrics.add_episode_data(
            reward=episode_reward,
            length=episode_length,
            value_loss=0.0,  # 测试阶段没有损失
            policy_loss=0.0,
            collisions=current_collision_count,
            target_reached=info.get('target_reached', False)
        )
        
        # 保存视频 - 安全处理
        try:
            if frames:
                video_path = os.path.join(save_dir, f'episode_{episode + 1}.mp4')
                save_video(frames, video_path, fps, video_quality)
                print(f"  视频已保存: {video_path}")
            else:
                print("  没有帧可以保存为视频")
        except Exception as e:
            print(f"  警告: 保存视频时出错: {e}")
        
        print(f"  回合奖励: {episode_reward:.2f}")
        print(f"  回合长度: {episode_length}")
        print(f"  碰撞次数: {current_collision_count}")
        print(f"  目标达成: {'是' if info.get('target_reached', False) else '否'}")
    
    # 保存测试记录
    try:
        results = {
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'collision_counts': collision_counts,
            'target_reached': [bool(x) for x in target_reached],  # 确保可以序列化
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'std_length': np.std(episode_lengths),
            'success_rate': sum(target_reached) / len(target_reached) if target_reached else 0,
            'mean_collisions': np.mean(collision_counts)
        }
        
        results_path = os.path.join(save_dir, 'results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        # 保存并绘制指标
        metrics.save_metrics(save_dir)
        metrics.plot_metrics(save_dir)
        
        # 尝试创建热图和回合总结
        try:
            env.visualizer.create_heatmap(os.path.join(save_dir, 'heatmap.png'))
            env.visualizer.create_episode_summary(
                rewards=episode_rewards, 
                lengths=episode_lengths, 
                collision_counts=collision_counts, 
                target_reached=target_reached,
                save_dir=save_dir
            )
        except Exception as e:
            print(f"  警告: 创建可视化分析时出错: {e}")
        
        print("\n=== 测试完成 ===")
        print(f"平均奖励: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
        print(f"平均长度: {results['mean_length']:.2f} ± {results['std_length']:.2f}")
        print(f"成功率: {results['success_rate']*100:.1f}%")
        print(f"平均碰撞次数: {results['mean_collisions']:.2f}")
        print(f"详细结果已保存: {results_path}")
    except Exception as e:
        print(f"  警告: 保存结果时出错: {e}")
        
    print("===============\n")
    
    return episode_rewards

def add_info_to_frame(frame, reward, length, action, distance, speed, steering, rotation_penalty, collision, target_reached):
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
        
        # 中间显示目标距离和速度
        mid_x = int(frame_width * 0.35)
        cv2.putText(info_frame, f"目标距离: {distance:.2f}", (mid_x, int(info_height*0.3)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), line_thickness)
        cv2.putText(info_frame, f"速度: {speed:.2f}", (mid_x, int(info_height*0.7)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), line_thickness)
        
        # 右边显示碰撞和目标状态
        status_x = int(frame_width * 0.7)
        
        # 碰撞状态
        cv2.putText(info_frame, "碰撞:", (status_x, int(info_height*0.3)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), line_thickness)
        collision_color = (0, 0, 255) if collision else (0, 255, 0)
        indicator_radius = max(5, int(font_scale * 8))
        indicator_x = status_x + int(font_scale * 80)
        cv2.circle(info_frame, (indicator_x, int(info_height*0.25)), indicator_radius, collision_color, -1)
        
        # 目标达成状态
        cv2.putText(info_frame, "目标:", (status_x, int(info_height*0.7)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), line_thickness)
        target_color = (0, 255, 0) if target_reached else (0, 0, 255)
        cv2.circle(info_frame, (indicator_x, int(info_height*0.65)), indicator_radius, target_color, -1)
        
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
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='测试自动驾驶智能体')
    
    # 必需参数
    parser.add_argument('--model_path', type=str, required=True,
                      help='模型文件路径')
    
    # 测试参数
    parser.add_argument('--num_episodes', type=int, default=5,
                      help='测试的回合数')
    parser.add_argument('--fps', type=int, default=30,
                      help='视频帧率')
    parser.add_argument('--video_quality', type=int, default=95,
                      help='视频质量 (0-100)')
    parser.add_argument('--render_width', type=int, default=800,
                      help='渲染宽度')
    parser.add_argument('--render_height', type=int, default=600,
                      help='渲染高度')
    
    # 智能体参数
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
    
    # 环境参数
    parser.add_argument('--env_config', type=str, default=None,
                      help='环境配置文件路径（JSON格式）')
    
    # 其他参数
    parser.add_argument('--save_dir', type=str, default='logs',
                      help='测试结果保存目录')
    
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
    
    # 渲染分辨率
    render_resolution = (args.render_width, args.render_height)
    
    # 运行测试
    test(
        model_path=args.model_path,
        env_config=env_config,
        agent_config=agent_config,
        num_episodes=args.num_episodes,
        save_dir=args.save_dir,
        fps=args.fps,
        video_quality=args.video_quality,
        render_resolution=render_resolution
    ) 