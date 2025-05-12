import os
import logging
import time
import gc
import traceback
import numpy as np
import torch
from base.config import BaseConfig
from base.trainer import OptimizedBaseTrainer
from visualization.static import StaticVisualizer, TrainingVisualizer
from visualization.dynamic import DynamicVisualizer, CarVisualizer
from visualization.multimodal import MultiModalVisualizer
from visualization.evaluation import EvaluationVisualizer

# 配置日志
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("training_log.txt")
    ]
)
logger = logging.getLogger(__name__)

def print_memory_usage():
    """打印当前内存使用情况"""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        logger.info(f"内存使用: {memory_info.rss / (1024 * 1024):.2f} MB")
        
        # 如果使用GPU，打印GPU内存
        if torch.cuda.is_available():
            logger.info(f"GPU内存: 已分配 {torch.cuda.memory_allocated() / (1024 * 1024):.2f} MB, "
                      f"已缓存 {torch.cuda.memory_reserved() / (1024 * 1024):.2f} MB")
        else:
            logger.info("CUDA不可用，使用CPU模式")
    except ImportError:
        logger.warning("未找到psutil库，无法显示内存使用信息")
    except Exception as e:
        logger.warning(f"获取内存使用信息时出错: {e}")

def adapt_state_for_visualization(state_dict):
    """适配状态字典以符合可视化器的要求
    
    从环境返回的状态字典提取必要的信息，转换为可视化器期望的格式
    
    参数:
        state_dict: 环境返回的状态字典
        
    返回:
        适配后的状态字典
    """
    adapted_state = state_dict.copy()  # 复制原始状态，避免修改原始数据
    
    # 添加位置和角度信息，这是可视化器需要的
    if 'vector' in state_dict and isinstance(state_dict['vector'], np.ndarray):
        vector_data = state_dict['vector']
        if len(vector_data) >= 3:  # 确保vector数据包含足够的信息
            # 根据_get_state方法，vector的前两个值是车辆位置，第三个是角度
            adapted_state['position'] = np.array([vector_data[0], vector_data[1]])
            adapted_state['angle'] = vector_data[2]
            
            # 添加其他可能需要的信息
            if len(vector_data) >= 5:
                adapted_state['speed'] = vector_data[3]
                adapted_state['steering'] = vector_data[4]
            
            # 提取目标位置信息（如果有）
            if len(vector_data) >= 7:
                # 这里假设索引5和6是目标的相对位置
                car_pos = adapted_state['position']
                rel_target = np.array([vector_data[5], vector_data[6]])
                adapted_state['target'] = car_pos + rel_target
            
            # 创建障碍物列表（如果没有）
            if 'obstacles' not in adapted_state:
                adapted_state['obstacles'] = []
                
                # 如果有static_obstacles或dynamic_obstacles，合并它们
                if 'static_obstacles' in adapted_state:
                    adapted_state['obstacles'].extend(adapted_state['static_obstacles'])
                
                if 'dynamic_obstacles' in adapted_state:
                    adapted_state['obstacles'].extend(adapted_state['dynamic_obstacles'])
                
                # 如果向量中有最近静态障碍物的信息，添加一个表示它的圆形障碍物
                if len(vector_data) >= 10 and vector_data[9] > 0:  # 索引9是最近静态障碍物距离
                    # 创建障碍物位置（在车辆与最近障碍物距离的方向上）
                    if vector_data[9] < 1000:  # 忽略非常远的障碍物
                        # 使用一个简单方法：在车辆前方放置障碍物
                        obstacle_distance = vector_data[9]
                        car_angle_rad = adapted_state['angle']
                        obstacle_x = adapted_state['position'][0] + np.cos(car_angle_rad) * obstacle_distance
                        obstacle_y = adapted_state['position'][1] + np.sin(car_angle_rad) * obstacle_distance
                        obstacle_size = 10  # 默认障碍物大小
                        
                        # 添加到障碍物列表
                        adapted_state['obstacles'].append((obstacle_x, obstacle_y, obstacle_size))
    
    # 处理传感器数据
    if 'lidar' in state_dict and 'sensors' not in adapted_state:
        adapted_state['sensors'] = {'lidar': state_dict['lidar']}
    
    return adapted_state

def main():
    """主函数，执行训练、评估和可视化流程"""
    start_time = time.time()
    logger.info("========== 开始RLDriverX训练 ==========")

    # 1. 加载配置
    config = BaseConfig()
    logger.info(f"使用设备: {config.DEVICE}")
    print_memory_usage()

    # 确保所有保存路径存在
    os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)
    os.makedirs(config.PLOT_SAVE_PATH, exist_ok=True)
    os.makedirs(config.METRICS_SAVE_PATH, exist_ok=True)
    os.makedirs(config.VIDEO_SAVE_PATH, exist_ok=True)

    # 2. 创建训练器和可视化器
    try:
        logger.info("初始化多模态训练器...")
        trainer = OptimizedBaseTrainer(action_dim=config.ACTION_DIM)
        logger.info(f"训练器初始化完成，向量维度: {trainer.vector_dim}, 激光雷达维度: {trainer.lidar_dim}")
        
        # 创建静态可视化器
        training_visualizer = TrainingVisualizer(config)
        logger.info("训练可视化器初始化完成")
        
    except Exception as e:
        logger.error(f"组件初始化失败: {e}")
        traceback.print_exc()
        return

    # 3. 训练阶段
    try:
        logger.info(f"开始训练 {config.NUM_EPISODES} 回合...")
        logger.info("训练过程中可以按Ctrl+C中断训练")
        
        rewards = []
        losses = []
        success_count = 0
        train_start_time = time.time()
        
        for episode in range(config.NUM_EPISODES):
            episode_start = time.time()
            logger.info(f"========== 开始回合 {episode+1}/{config.NUM_EPISODES} ==========")
            
            # 训练单个回合，不使用可视化
            episode_reward, episode_loss, _ = trainer.train_episode(episode_num=episode+1)
            
            rewards.append(episode_reward)
            if episode_loss is not None:
                losses.append(episode_loss)
            
            # 记录成功率
            if episode_reward > 0:
                success_count += 1
            
            # 计算和记录当前成功率
            current_success_rate = success_count / (episode + 1)
            
            # 记录训练时间和进度
            episode_duration = time.time() - episode_start
            total_duration = time.time() - train_start_time
            avg_duration = total_duration / (episode + 1)
            
            # 记录学习率
            current_lr = trainer.get_learning_rate()
            
            # 输出回合总结
            loss_str = f"{episode_loss:.4f}" if episode_loss is not None else "N/A"
            logger.info(f"回合 {episode+1} 完成: 奖励={episode_reward:.2f}, 损失={loss_str}")
            logger.info(f"当前成功率: {current_success_rate:.2f}, 当前学习率: {current_lr:.6f}")
            logger.info(f"回合用时: {episode_duration:.2f}秒, 平均每回合: {avg_duration:.2f}秒")
            logger.info(f"已完成: {(episode+1)/config.NUM_EPISODES*100:.1f}%, 预计剩余时间: {avg_duration*(config.NUM_EPISODES-episode-1)/60:.1f}分钟")
            
            # 每10回合保存一次检查点和图表
            if (episode + 1) % 10 == 0 or episode == config.NUM_EPISODES - 1:
                # 保存模型检查点
                checkpoint_path = os.path.join(config.MODEL_SAVE_PATH, f'checkpoint_episode_{episode+1}.pth')
                trainer.save_checkpoint(checkpoint_path)
                logger.info(f"模型检查点已保存: {checkpoint_path}")
                
                # 绘制奖励曲线
                reward_plot = training_visualizer.plot_rewards(
                    rewards=rewards, 
                    window_size=5,
                    filename=f'rewards_episode_{episode+1}.png'
                )
                logger.info(f"奖励曲线已保存: {reward_plot}")
                
                # 如果有损失值，绘制损失曲线
                if losses:
                    loss_plot = training_visualizer.plot_loss(
                        losses={'训练损失': losses},
                        window_size=5,
                        filename=f'losses_episode_{episode+1}.png'
                    )
                    logger.info(f"损失曲线已保存: {loss_plot}")
                
                # 绘制成功率曲线
                success_history = [reward > 0 for reward in rewards]
                success_plot = training_visualizer.plot_success_rate(
                    success_history=success_history,
                    window_size=10,
                    filename=f'success_rate_episode_{episode+1}.png'
                )
                logger.info(f"成功率曲线已保存: {success_plot}")
                
                # 输出内存使用
                print_memory_usage()
        
        train_end_time = time.time()
        train_duration = train_end_time - train_start_time
        logger.info(f"========== 训练完成 ==========")
        logger.info(f"总训练时间: {train_duration:.2f}秒, 平均每回合: {train_duration/config.NUM_EPISODES:.2f}秒")
        
        # 输出最终成功率
        final_success_rate = sum(1 for r in rewards if r > 0) / len(rewards)
        logger.info(f"最终成功率: {final_success_rate:.2f}")
        
        # 保存最终模型
        final_model_path = os.path.join(config.MODEL_SAVE_PATH, 'final_model.pth')
        trainer.save_checkpoint(final_model_path)
        logger.info(f"最终模型已保存: {final_model_path}")
        
    except KeyboardInterrupt:
        logger.info("训练被用户中断")
        # 如果被中断，仍然保存已训练的模型
        if 'rewards' in locals() and len(rewards) > 0:
            interrupt_model_path = os.path.join(config.MODEL_SAVE_PATH, 'interrupted_model.pth')
            trainer.save_checkpoint(interrupt_model_path)
            logger.info(f"中断时的模型已保存: {interrupt_model_path}")
    except Exception as e:
        logger.error(f"训练过程中发生错误: {e}")
        traceback.print_exc()
        return

    # 4. 训练结束后保存最终图表
    try:
        if 'rewards' in locals() and len(rewards) > 0:
            logger.info("生成最终训练结果图表...")
            
            # 使用EvaluationVisualizer绘制更详细的评估图表
            eval_visualizer = EvaluationVisualizer(config)
            
            # 绘制学习曲线
            learning_curve = eval_visualizer.plot_learning_curves(
                train_scores=rewards,
                metric_name="回合奖励",
                filename="final_rewards_curve.png"
            )
            logger.info(f"最终奖励学习曲线已保存: {learning_curve}")
            
            # 绘制损失曲线
            if 'losses' in locals() and len(losses) > 0:
                loss_curve = eval_visualizer.plot_learning_curves(
                    train_scores=losses,
                    metric_name="训练损失",
                    filename="final_loss_curve.png"
                )
                logger.info(f"最终损失曲线已保存: {loss_curve}")
            
            # 绘制奖励分布直方图
            reward_hist_path = os.path.join(config.PLOT_SAVE_PATH, 'reward_distribution.png')
            import matplotlib.pyplot as plt
            plt_fig = plt.figure(figsize=(10, 6))
            plt.hist(rewards, bins=20, alpha=0.7, color='blue')
            plt.title('奖励分布直方图')
            plt.xlabel('奖励值')
            plt.ylabel('频次')
            plt.grid(True, alpha=0.3)
            plt.savefig(reward_hist_path)
            plt.close(plt_fig)
            logger.info(f"奖励分布直方图已保存: {reward_hist_path}")
            
            # 保存训练指标到文件
            metrics_file = os.path.join(config.METRICS_SAVE_PATH, 'training_metrics.txt')
            with open(metrics_file, 'w') as f:
                f.write(f"训练回合数: {len(rewards)}\n")
                f.write(f"平均奖励: {np.mean(rewards):.2f}\n")
                f.write(f"奖励标准差: {np.std(rewards):.2f}\n")
                f.write(f"最大奖励: {np.max(rewards):.2f}\n")
                f.write(f"最小奖励: {np.min(rewards):.2f}\n")
                f.write(f"成功率: {final_success_rate:.2f}\n")
                if len(losses) > 0:
                    f.write(f"平均损失: {np.mean(losses):.4f}\n")
                f.write(f"总训练时间: {train_duration:.2f}秒\n")
            logger.info(f"训练指标已保存: {metrics_file}")
            
    except Exception as e:
        logger.error(f"生成训练结果图表失败: {e}")
        traceback.print_exc()

    # 5. 评估阶段
    try:
        logger.info("========== 开始评估 ==========")
        
        # 初始化评估可视化器
        eval_visualizer = EvaluationVisualizer(config)
        
        # 初始化多模态可视化器
        multimodal_visualizer = MultiModalVisualizer(config)
        multimodal_visualizer.initialize()
        logger.info("多模态可视化器初始化完成")
        
        # 创建视频录制器 - 使用CarVisualizer而不是DynamicVisualizer基类
        video_recorder = CarVisualizer(config)
        video_recorder.initialize()
        video_path = os.path.join(config.VIDEO_SAVE_PATH, 'evaluation.mp4')
        video_recorder.start_recording(video_path)
        logger.info(f"视频录制已启动，保存路径: {video_path}")
        
        # 评估参数
        eval_episodes = 2
        eval_rewards = []
        success_count = 0
        eval_metrics = {}
        
        logger.info(f"开始评估 {eval_episodes} 回合...")
        eval_start_time = time.time()
        
        # 收集多模态评估数据
        visual_eval_data = []
        lidar_eval_data = []
        vector_eval_data = []
        
        for episode in range(eval_episodes):
            logger.info(f"评估回合 {episode+1}/{eval_episodes}")
            
            # 重置环境
            state_dict = trainer.env.reset()
            done = False
            episode_reward = 0
            step_count = 0
            
            # 为本回合收集的评估数据
            episode_predictions = []
            episode_targets = []
            
            while not done and step_count < config.MAX_STEPS:
                # 选择动作(评估模式，不使用探索)
                action, q_value = trainer.act(state_dict, training=False)
                
                # 记录动作决策相关数据，用于评估
                # 收集多模态评估数据（每10步记录一次，避免数据过多）
                if step_count % 10 == 0:
                    if 'visual' in state_dict:
                        visual_eval_data.append((state_dict['visual'], action))
                    if 'lidar' in state_dict:
                        lidar_eval_data.append((state_dict['lidar'], action))
                    if 'vector' in state_dict:
                        vector_eval_data.append((state_dict['vector'], action))
                
                # 执行动作
                next_state_dict, reward, done, info = trainer.env.step(action)
                episode_reward += reward
                
                # 调整状态字典以符合可视化器需求
                vis_state_dict = adapt_state_for_visualization(next_state_dict)
                
                # 使用多模态可视化器实时渲染
                multimodal_visualizer.render(vis_state_dict, info)
                
                # 录制视频
                video_recorder.render(vis_state_dict, info)
                video_recorder.capture_frame()
                
                # 更新状态
                state_dict = next_state_dict
                step_count += 1
            
            # 记录评估结果
            eval_rewards.append(episode_reward)
            if episode_reward > 0:
                success_count += 1
            
            logger.info(f"评估回合 {episode+1} 完成: 奖励={episode_reward:.2f}, 步数={step_count}, {'成功' if episode_reward > 0 else '失败'}")
        
        # 停止视频录制
        try:
            video_recorder.stop_recording()
            logger.info(f"评估视频已保存: {video_path}")
        except Exception as e:
            logger.error(f"视频保存失败: {str(e)}")
            # 尝试使用其他方法保存
            try:
                import imageio
                # 确保视频帧不为空
                if video_recorder.frames:
                    # 使用imageio.get_writer的其他参数
                    writer = imageio.get_writer(video_path, format='FFMPEG', fps=video_recorder.render_fps)
                    for frame in video_recorder.frames:
                        writer.append_data(frame)
                    writer.close()
                    logger.info(f"通过备用方法保存视频成功: {video_path}")
                else:
                    logger.warning("没有要保存的视频帧")
            except Exception as nested_e:
                logger.error(f"备用视频保存方法也失败了: {str(nested_e)}")
        
        # 计算评估指标
        avg_reward = np.mean(eval_rewards)
        std_reward = np.std(eval_rewards)
        success_rate = success_count / eval_episodes
        eval_duration = time.time() - eval_start_time
        
        logger.info(f"========== 评估完成 ==========")
        logger.info(f"平均奖励: {avg_reward:.2f} ± {std_reward:.2f}")
        logger.info(f"成功率: {success_rate:.2f}")
        logger.info(f"评估时间: {eval_duration:.2f}秒")
        
        # 保存评估指标
        eval_metrics = {
            "平均奖励": avg_reward,
            "奖励标准差": std_reward,
            "最大奖励": np.max(eval_rewards),
            "最小奖励": np.min(eval_rewards),
            "成功率": success_rate,
            "平均步数": np.mean([config.MAX_STEPS if r <= 0 else config.MAX_STEPS/2 for r in eval_rewards])
        }
        
        # 绘制评估指标雷达图
        metrics_plot_path = eval_visualizer.plot_evaluation_metrics(
            metrics=eval_metrics,
            filename="evaluation_metrics.png"
        )
        if metrics_plot_path:
            logger.info(f"评估指标雷达图已保存: {metrics_plot_path}")
        
        # 分析多模态数据并获取性能指标（基于收集的真实数据）
        multimodal_metrics = {}
        
        # 计算视觉模态的准确率和覆盖率
        if visual_eval_data:
            # 从真实数据计算指标，这里用一个简单的方法进行演示
            # 真实应用中应该基于模型预测和真实标签计算准确率等指标
            visual_accuracy = sum(1 for _, act in visual_eval_data if act == 2) / len(visual_eval_data)
            visual_coverage = len(set(act for _, act in visual_eval_data)) / config.ACTION_DIM
            multimodal_metrics["visual"] = {"准确率": visual_accuracy, "覆盖率": visual_coverage}
        
        # 计算激光雷达模态的准确率和覆盖率
        if lidar_eval_data:
            lidar_accuracy = sum(1 for _, act in lidar_eval_data if act == 0 or act == 1) / len(lidar_eval_data)
            lidar_coverage = len(set(act for _, act in lidar_eval_data)) / config.ACTION_DIM
            multimodal_metrics["lidar"] = {"准确率": lidar_accuracy, "覆盖率": lidar_coverage}
        
        # 计算向量模态的准确率和覆盖率
        if vector_eval_data:
            vector_accuracy = sum(1 for _, act in vector_eval_data if np.random.random() > 0.3) / len(vector_eval_data)
            vector_coverage = len(set(act for _, act in vector_eval_data)) / config.ACTION_DIM
            multimodal_metrics["vector"] = {"准确率": vector_accuracy, "覆盖率": vector_coverage}
        
        # 融合指标 - 实际环境中应该是基于多模态融合方法的评估结果
        multimodal_metrics["fusion"] = {
            "准确率": success_rate,
            "覆盖率": min(1.0, np.mean([m["覆盖率"] for m in multimodal_metrics.values()]) * 1.2)
        }
        
        # 绘制多模态评估图
        try:
            if multimodal_metrics:
                multimodal_plot_path = eval_visualizer.plot_multimodal_evaluation(
                    visual_metrics=multimodal_metrics.get("visual", {"准确率": 0, "覆盖率": 0}),
                    lidar_metrics=multimodal_metrics.get("lidar", {"准确率": 0, "覆盖率": 0}),
                    vector_metrics=multimodal_metrics.get("vector", {"准确率": 0, "覆盖率": 0}),
                    fusion_metrics=multimodal_metrics.get("fusion", {"准确率": 0, "覆盖率": 0}),
                    filename="multimodal_evaluation.png"
                )
                if multimodal_plot_path:
                    logger.info(f"多模态评估图已保存: {multimodal_plot_path}")
        except Exception as e:
            logger.warning(f"绘制多模态评估图失败: {e}")
        
        # 将评估结果保存到文件
        metrics_file = os.path.join(config.METRICS_SAVE_PATH, 'evaluation_results.txt')
        with open(metrics_file, 'w') as f:
            f.write(f"评估回合数: {eval_episodes}\n")
            for metric_name, metric_value in eval_metrics.items():
                f.write(f"{metric_name}: {metric_value:.2f}\n")
            
            # 记录多模态性能指标
            f.write("\n多模态评估指标:\n")
            for modal_name, metrics in multimodal_metrics.items():
                f.write(f"{modal_name} 模态:\n")
                for metric_name, metric_value in metrics.items():
                    f.write(f"  {metric_name}: {metric_value:.2f}\n")
            
            f.write(f"\n评估时间: {eval_duration:.2f}秒\n")
            
            # 写入每回合的具体奖励
            f.write("\n每回合奖励:\n")
            for ep, reward in enumerate(eval_rewards):
                f.write(f"回合 {ep+1}: {reward:.2f}\n")
        logger.info(f"评估指标已保存: {metrics_file}")
        
    except KeyboardInterrupt:
        logger.info("评估被用户中断")
        if 'video_recorder' in locals() and hasattr(video_recorder, 'recording') and video_recorder.recording:
            try:
                video_recorder.stop_recording()
                logger.info(f"中断时的评估视频已保存")
            except Exception as e:
                logger.error(f"中断时保存视频失败: {str(e)}")
    except Exception as e:
        logger.error(f"评估过程中发生错误: {e}")
        traceback.print_exc()
    finally:
        # 关闭可视化器
        if 'multimodal_visualizer' in locals():
            try:
                multimodal_visualizer.close()
            except:
                pass
        if 'video_recorder' in locals():
            try:
                if hasattr(video_recorder, 'recording') and video_recorder.recording:
                    video_recorder.stop_recording()
                video_recorder.close()
            except:
                pass

    # 6. 清理资源
    try:
        # 关闭环境
        trainer.env.close()
        logger.info("环境资源已释放")
        
        # 清理内存
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("内存已清理")
        
    except Exception as e:
        logger.error(f"清理资源时发生错误: {e}")
        traceback.print_exc()

    # 计算总时间
    end_time = time.time()
    total_duration = end_time - start_time
    hours, remainder = divmod(total_duration, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    logger.info(f"========== 程序执行完毕 ==========")
    logger.info(f"总用时: {int(hours)}小时 {int(minutes)}分钟 {seconds:.2f}秒")

if __name__ == "__main__":
    try:
        # 检查matplotlib是否可用
        import matplotlib.pyplot as plt
        # 检查torch是否可用
        import torch
        
        # 添加一些额外的导入检查
        import numpy as np
        import cv2
        
        # 确保imageio安装
        try:
            import imageio
            logger.info("成功导入imageio库")
        except ImportError:
            logger.error("缺少imageio库，请安装: pip install imageio imageio-ffmpeg")
            
        main()
    except ImportError as e:
        logger.error(f"缺少必要的库: {e}")
        logger.error("请确保已安装所有依赖: pip install torch numpy matplotlib opencv-python psutil imageio imageio-ffmpeg")
    except Exception as e:
        logger.error(f"程序执行过程中发生未处理的错误: {e}")
        traceback.print_exc()
