"""评估可视化模块

提供用于模型评估结果的可视化功能，包括评估指标、模型比较、学习曲线和多模态数据评估。
优化后的版本更好地支持base目录中的环境和模型，特别是多模态输入的评估。
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import cv2
from typing import List, Dict, Any, Tuple, Optional, Union
import logging
import traceback
from visualization.static import StaticVisualizer

# 使用与static模块相同的日志记录器
logger = logging.getLogger('visualization.evaluation')


class EvaluationVisualizer(StaticVisualizer):
    """评估可视化
    
    提供模型评估结果的可视化功能，包括评估指标雷达图、模型性能比较、学习曲线等。
    优化后的版本更好地支持多模态输入的评估。
    """

    def plot_evaluation_metrics(self, metrics: Dict[str, float],
                                filename: str = 'evaluation_metrics.png') -> str:
        """绘制评估指标雷达图
        
        参数:
            metrics: 评估指标字典，键为指标名称，值为对应的数值
            filename: 保存的文件名
            
        返回:
            str: 保存的文件路径
        """
        try:
            # 创建极坐标图表
            fig, ax = plt.subplots(figsize=self.plot_resolution,
                                   subplot_kw={'projection': 'polar'})
            
            if not metrics:
                logger.warning("无评估指标数据可绘制")
                # 在极坐标图中添加文本较为复杂，这里不添加
                return self.save_figure(fig, filename)

            # 提取数据
            categories = list(metrics.keys())
            values = list(metrics.values())

            # 检查数据有效性
            if any(not isinstance(v, (int, float)) for v in values):
                logger.error("评估指标包含非数值类型数据")
                # 尝试转换
                values = [float(v) if isinstance(v, (int, float)) else 0.0 for v in values]

            # 计算角度
            N = len(categories)
            angles = [n / float(N) * 2 * np.pi for n in range(N)]
            angles += angles[:1]  # 闭合图形

            # 添加值
            values += values[:1]  # 闭合图形

            # 绘制雷达图
            ax.plot(angles, values, 'o-', linewidth=2)
            ax.fill(angles, values, alpha=0.25)

            # 设置刻度标签
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories)

            # 设置y轴范围
            ax.set_ylim(0, max(values) * 1.1)

            # 设置图表属性
            ax.set_title('评估指标雷达图')
            ax.grid(True)

            # 保存图表
            return self.save_figure(fig, filename)
        except Exception as e:
            logger.error(f"绘制评估指标雷达图失败: {str(e)}")
            logger.debug(traceback.format_exc())
            return ""

    def plot_model_comparison(self, model_metrics: Dict[str, Dict[str, float]],
                              filename: str = 'model_comparison.png') -> str:
        """比较不同模型的评估指标
        
        参数:
            model_metrics: 模型指标嵌套字典，格式为 {模型名: {指标名: 指标值}}
            filename: 保存的文件名
            
        返回:
            str: 保存的文件路径
        """
        try:
            # 创建图表
            fig, ax = self.create_figure()
            
            if not model_metrics:
                logger.warning("无模型比较数据可绘制")
                ax.text(0.5, 0.5, '无数据', ha='center', va='center')
                return self.save_figure(fig, filename)

            # 提取数据
            models = list(model_metrics.keys())
            
            # 确保至少有一个模型
            if not models:
                logger.warning("模型比较字典为空")
                ax.text(0.5, 0.5, '无模型数据', ha='center', va='center')
                return self.save_figure(fig, filename)
                
            # 获取所有模型共有的指标
            common_metrics = set.intersection(*[set(model_metrics[model].keys()) for model in models])
            
            if not common_metrics:
                # 如果没有共有指标，则使用第一个模型的所有指标
                metrics = list(model_metrics[models[0]].keys())
                logger.warning("没有所有模型共有的指标，使用第一个模型的指标")
            else:
                metrics = list(common_metrics)

            # 设置x轴位置
            x = np.arange(len(metrics))
            width = 0.8 / len(models)  # 条形宽度

            # 绘制条形图
            for i, model in enumerate(models):
                # 获取当前模型的指标值
                model_data = model_metrics[model]
                values = []
                
                for metric in metrics:
                    if metric in model_data:
                        value = model_data[metric]
                        # 确保值是数值类型
                        if isinstance(value, (int, float)):
                            values.append(value)
                        else:
                            logger.warning(f"模型 {model} 的指标 {metric} 不是数值类型，使用0")
                            values.append(0)
                    else:
                        logger.warning(f"模型 {model} 缺少指标 {metric}，使用0")
                        values.append(0)
                
                offset = (i - len(models) / 2 + 0.5) * width
                ax.bar(x + offset, values, width, label=model)

            # 设置图表属性
            ax.set_xlabel('评估指标')
            ax.set_ylabel('得分')
            ax.set_title('模型性能比较')
            ax.set_xticks(x)
            ax.set_xticklabels(metrics)
            ax.legend()
            ax.grid(True, axis='y')

            # 保存图表
            return self.save_figure(fig, filename)
        except Exception as e:
            logger.error(f"绘制模型比较失败: {str(e)}")
            logger.debug(traceback.format_exc())
            return ""
            
    def plot_learning_curves(self, train_scores: List[float], val_scores: List[float] = None, 
                           epochs: List[int] = None, metric_name: str = "准确率",
                           filename: str = 'learning_curves.png') -> str:
        """绘制学习曲线
        
        参数:
            train_scores: 训练集评分列表
            val_scores: 验证集评分列表，如果为None或空列表则只绘制训练曲线
            epochs: 轮次列表，如果为None则使用索引
            metric_name: 指标名称
            filename: 保存的文件名
            
        返回:
            str: 保存的文件路径
        """
        try:
            # 创建图表
            fig, ax = self.create_figure()
            
            # 检查训练数据是否有效
            if not train_scores or not isinstance(train_scores, (list, np.ndarray)):
                logger.warning(f"无有效的训练数据可绘制，数据类型: {type(train_scores)}")
                ax.text(0.5, 0.5, '无训练数据', ha='center', va='center')
                return self.save_figure(fig, filename)
                
            # 确保所有数据是数值型
            try:
                train_scores = [float(score) for score in train_scores]
            except (ValueError, TypeError) as e:
                logger.error(f"训练数据包含非数值: {e}")
                ax.text(0.5, 0.5, '训练数据包含非数值', ha='center', va='center')
                return self.save_figure(fig, filename)
                
            # 如果没有提供epochs，则使用索引
            if epochs is None:
                epochs = list(range(1, len(train_scores) + 1))
            else:
                # 确保epochs长度匹配
                if len(epochs) != len(train_scores):
                    logger.warning(f"轮次数量({len(epochs)})与训练评分数量({len(train_scores)})不一致，使用索引替代")
                    epochs = list(range(1, len(train_scores) + 1))
                
            # 绘制训练集曲线
            ax.plot(epochs, train_scores, 'b-', label=f'训练集{metric_name}')
            
            # 如果提供了验证集数据，则绘制验证集曲线
            if val_scores and len(val_scores) > 0:
                try:
                    # 确保所有数据是数值型
                    val_scores = [float(score) for score in val_scores]
                    
                    # 调整验证集数据长度以匹配训练集，如果不一致
                    if len(val_scores) != len(train_scores):
                        logger.warning(f"训练集评分数量({len(train_scores)})与验证集评分数量({len(val_scores)})不一致")
                        # 使用较短的列表长度
                        min_len = min(len(train_scores), len(val_scores))
                        if min_len < len(train_scores):
                            train_scores_plot = train_scores[:min_len]
                            epochs_plot = epochs[:min_len]
                        else:
                            train_scores_plot = train_scores
                            epochs_plot = epochs
                            val_scores = val_scores[:min_len]
                    else:
                        train_scores_plot = train_scores
                        epochs_plot = epochs
                    
                    # 绘制验证集曲线
                    ax.plot(epochs_plot, val_scores, 'r-', label=f'验证集{metric_name}')
                except Exception as e:
                    logger.error(f"绘制验证集曲线失败: {str(e)}")
            
            # 设置图表属性
            ax.set_xlabel('训练轮次')
            ax.set_ylabel(metric_name)
            ax.set_title('学习曲线')
            ax.legend()
            ax.grid(True)
            
            # 保存图表
            return self.save_figure(fig, filename)
        except Exception as e:
            logger.error(f"绘制学习曲线失败: {str(e)}")
            logger.debug(traceback.format_exc())
            return ""

    def plot_multimodal_evaluation(self, visual_metrics: Dict[str, float] = None,
                                 lidar_metrics: Dict[str, float] = None,
                                 vector_metrics: Dict[str, float] = None,
                                 fusion_metrics: Dict[str, float] = None,
                                 filename: str = 'multimodal_evaluation.png') -> str:
        """绘制多模态评估结果
        
        参数:
            visual_metrics: 视觉模态评估指标
            lidar_metrics: 激光雷达模态评估指标
            vector_metrics: 向量模态评估指标
            fusion_metrics: 融合模型评估指标
            filename: 保存的文件名
            
        返回:
            str: 保存的文件路径
        """
        try:
            # 创建图表
            fig, ax = self.create_figure()
            
            # 检查是否有数据
            modalities = []
            if visual_metrics:
                modalities.append(('视觉', visual_metrics))
            if lidar_metrics:
                modalities.append(('激光雷达', lidar_metrics))
            if vector_metrics:
                modalities.append(('向量', vector_metrics))
            if fusion_metrics:
                modalities.append(('融合', fusion_metrics))
                
            if not modalities:
                logger.warning("无多模态评估数据可绘制")
                ax.text(0.5, 0.5, '无数据', ha='center', va='center')
                return self.save_figure(fig, filename)
            
            # 获取所有模态共有的指标
            common_metrics = set.intersection(*[set(metrics) for _, metrics in modalities])
            
            if not common_metrics:
                # 如果没有共有指标，则使用第一个模态的所有指标
                metrics = list(modalities[0][1].keys())
                logger.warning("没有所有模态共有的指标，使用第一个模态的指标")
            else:
                metrics = list(common_metrics)
                
            # 设置x轴位置
            x = np.arange(len(metrics))
            width = 0.8 / len(modalities)  # 条形宽度
            
            # 颜色映射
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
            
            # 绘制条形图
            for i, (modality_name, modality_metrics) in enumerate(modalities):
                values = []
                
                for metric in metrics:
                    if metric in modality_metrics:
                        value = modality_metrics[metric]
                        # 确保值是数值类型
                        if isinstance(value, (int, float)):
                            values.append(value)
                        else:
                            logger.warning(f"模态 {modality_name} 的指标 {metric} 不是数值类型，使用0")
                            values.append(0)
                    else:
                        logger.warning(f"模态 {modality_name} 缺少指标 {metric}，使用0")
                        values.append(0)
                
                offset = (i - len(modalities) / 2 + 0.5) * width
                ax.bar(x + offset, values, width, label=modality_name, color=colors[i % len(colors)])
            
            # 设置图表属性
            ax.set_xlabel('评估指标')
            ax.set_ylabel('得分')
            ax.set_title('多模态评估结果')
            ax.set_xticks(x)
            ax.set_xticklabels(metrics)
            ax.legend()
            ax.grid(True, axis='y')
            
            # 保存图表
            return self.save_figure(fig, filename)
        except Exception as e:
            logger.error(f"绘制多模态评估结果失败: {str(e)}")
            logger.debug(traceback.format_exc())
            return ""

    def plot_reward_history(self, rewards: List[float], smoothed: bool = True, window_size: int = 10,
                          filename: str = 'reward_history.png') -> str:
        """绘制奖励历史曲线
        
        参数:
            rewards: 奖励列表
            smoothed: 是否绘制平滑曲线
            window_size: 平滑窗口大小
            filename: 保存的文件名
            
        返回:
            str: 保存的文件路径
        """
        try:
            # 创建图表
            fig, ax = self.create_figure()
            
            # 检查奖励数据是否有效
            if not rewards or not isinstance(rewards, (list, np.ndarray)):
                logger.warning(f"无有效的奖励数据可绘制，数据类型: {type(rewards)}")
                ax.text(0.5, 0.5, '无奖励数据', ha='center', va='center')
                return self.save_figure(fig, filename)
                
            # 确保所有数据是数值型
            try:
                rewards = [float(r) for r in rewards]
            except (ValueError, TypeError) as e:
                logger.error(f"奖励数据包含非数值: {e}")
                ax.text(0.5, 0.5, '奖励数据包含非数值', ha='center', va='center')
                return self.save_figure(fig, filename)
                
            # 绘制原始奖励曲线
            episodes = list(range(1, len(rewards) + 1))
            ax.plot(episodes, rewards, 'b-', alpha=0.3, label='原始奖励')
            
            # 绘制平滑奖励曲线
            if smoothed and len(rewards) >= window_size:
                smoothed_rewards = []
                for i in range(len(rewards)):
                    if i < window_size - 1:
                        # 对于前window_size-1个点，使用所有可用点的平均值
                        smoothed_rewards.append(sum(rewards[:i+1]) / (i+1))
                    else:
                        # 对于其他点，使用窗口内的平均值
                        smoothed_rewards.append(sum(rewards[i-window_size+1:i+1]) / window_size)
                ax.plot(episodes, smoothed_rewards, 'r-', label=f'平滑奖励 (窗口={window_size})')
            
            # 设置图表属性
            ax.set_xlabel('训练回合')
            ax.set_ylabel('奖励')
            ax.set_title('训练奖励历史')
            ax.legend()
            ax.grid(True)
            
            # 保存图表
            return self.save_figure(fig, filename)
        except Exception as e:
            logger.error(f"绘制奖励历史失败: {str(e)}")
            logger.debug(traceback.format_exc())
            return ""

    def visualize_sample(self, visual_data=None, lidar_data=None, vector_data=None,
                        filename: str = 'sample_visualization.png') -> str:
        """可视化样本数据
        
        参数:
            visual_data: 视觉数据
            lidar_data: 激光雷达数据
            vector_data: 向量数据
            filename: 保存的文件名
            
        返回:
            str: 保存的文件路径
        """
        try:
            # 确定需要的子图数量
            n_plots = sum(x is not None for x in [visual_data, lidar_data, vector_data])
            
            if n_plots == 0:
                logger.warning("无样本数据可视化")
                fig, ax = self.create_figure()
                ax.text(0.5, 0.5, '无样本数据', ha='center', va='center')
                return self.save_figure(fig, filename)
            
            # 创建子图
            fig, axes = plt.subplots(1, n_plots, figsize=self.plot_resolution)
            if n_plots == 1:
                axes = [axes]  # 确保axes是列表
            
            plot_idx = 0
            
            # 绘制视觉数据
            if visual_data is not None:
                try:
                    ax = axes[plot_idx]
                    plot_idx += 1
                    
                    if isinstance(visual_data, np.ndarray):
                        # 处理不同格式的视觉数据
                        if visual_data.ndim == 3:
                            # 如果是 (C, H, W) 格式，转换为 (H, W, C)
                            if visual_data.shape[0] == 3 and visual_data.shape[2] != 3:
                                visual_data = np.transpose(visual_data, (1, 2, 0))
                            
                            # 如果数据是浮点型且范围在[0,1]，转换为[0,255]
                            if visual_data.dtype == np.float32 or visual_data.dtype == np.float64:
                                if np.max(visual_data) <= 1.0:
                                    visual_data = (visual_data * 255).astype(np.uint8)
                            
                            # 显示图像
                            ax.imshow(visual_data)
                        else:
                            ax.text(0.5, 0.5, '无效的视觉数据格式', ha='center', va='center')
                    else:
                        ax.text(0.5, 0.5, '无视觉数据', ha='center', va='center')
                    
                    ax.set_title('视觉数据')
                    ax.axis('off')
                except Exception as e:
                    logger.error(f"绘制视觉数据失败: {str(e)}")
                    ax.text(0.5, 0.5, '绘制视觉数据失败', ha='center', va='center')
            
            # 绘制激光雷达数据
            if lidar_data is not None:
                try:
                    ax = axes[plot_idx]
                    plot_idx += 1
                    
                    if isinstance(lidar_data, (list, np.ndarray)) and len(lidar_data) > 0:
                        # 绘制极坐标图
                        ax.set_projection('polar')
                        
                        # 计算角度
                        num_rays = len(lidar_data)
                        angles = np.linspace(0, 2*np.pi, num_rays, endpoint=False)
                        
                        # 绘制激光雷达数据
                        ax.plot(angles, lidar_data, 'ro-', markersize=3)
                        ax.fill(angles, lidar_data, alpha=0.2)
                        
                        # 设置极坐标属性
                        ax.set_theta_zero_location('N')  # 北方向为0度
                        ax.set_theta_direction(-1)  # 顺时针方向
                        
                        # 设置径向网格线
                        max_dist = max(lidar_data) * 1.1
                        ax.set_ylim(0, max_dist)
                        ax.set_rticks(np.linspace(0, max_dist, 5)[1:])  # 不显示原点刻度
                    else:
                        ax.text(0.5, 0.5, '无激光雷达数据', ha='center', va='center')
                    
                    ax.set_title('激光雷达数据')
                except Exception as e:
                    logger.error(f"绘制激光雷达数据失败: {str(e)}")
                    ax.text(0.5, 0.5, '绘制激光雷达数据失败', ha='center', va='center')
            
            # 绘制向量数据
            if vector_data is not None:
                try:
                    ax = axes[plot_idx]
                    plot_idx += 1
                    
                    if isinstance(vector_data, (list, np.ndarray)) and len(vector_data) > 0:
                        # 绘制条形图
                        x = np.arange(len(vector_data))
                        ax.bar(x, vector_data)
                        
                        # 设置x轴刻度
                        ax.set_xticks(x)
                        ax.set_xticklabels([f'{i}' for i in range(len(vector_data))], rotation=45)
                    else:
                        ax.text(0.5, 0.5, '无向量数据', ha='center', va='center')
                    
                    ax.set_title('向量数据')
                    ax.grid(True, axis='y')
                except Exception as e:
                    logger.error(f"绘制向量数据失败: {str(e)}")
                    ax.text(0.5, 0.5, '绘制向量数据失败', ha='center', va='center')
            
            # 调整布局
            plt.tight_layout()
            
            # 保存图表
            return self.save_figure(fig, filename)
        except Exception as e:
            logger.error(f"可视化样本数据失败: {str(e)}")
            logger.debug(traceback.format_exc())
            return ""