"""
静态可视化模块

提供用于训练指标、环境状态和算法比较的静态图表生成功能。
包含多种可视化器类，用于绘制不同类型的图表并保存到文件。
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import List, Dict, Any, Tuple, Optional, Union, Callable
import logging
import traceback

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger('visualization.static')


class StaticVisualizer:
    """静态可视化基类
    
    提供静态图表创建和保存的基本功能。
    
    属性:
        config: 配置对象，包含图表尺寸和保存路径等参数
        plot_path: 图表保存路径
        plot_resolution: 图表分辨率
    """

    def __init__(self, config):
        """初始化静态可视化器
        
        参数:
            config: 配置对象，必须包含 PLOT_SAVE_PATH，可选包含 PLOT_RESOLUTION
        """
        self.config = config
        
        # 获取配置参数，设置默认值
        self.plot_path = getattr(config, 'PLOT_SAVE_PATH', 'plots')
        
        # 默认图表分辨率为 (10, 8)
        default_resolution = (10, 8)
        if hasattr(config, 'PLOT_RESOLUTION'):
            try:
                # 验证格式
                if (isinstance(config.PLOT_RESOLUTION, tuple) and 
                    len(config.PLOT_RESOLUTION) == 2 and 
                    all(isinstance(x, (int, float)) for x in config.PLOT_RESOLUTION)):
                    self.plot_resolution = config.PLOT_RESOLUTION
                else:
                    logger.warning(f"无效的图表分辨率格式: {config.PLOT_RESOLUTION}，使用默认值 {default_resolution}")
                    self.plot_resolution = default_resolution
            except Exception as e:
                logger.warning(f"读取图表分辨率配置失败: {e}，使用默认值 {default_resolution}")
                self.plot_resolution = default_resolution
        else:
            self.plot_resolution = default_resolution
        
        # 确保保存路径存在
        try:
            if not os.path.exists(self.plot_path):
                os.makedirs(self.plot_path, exist_ok=True)
                logger.info(f"创建图表保存路径: {self.plot_path}")
            else:
                logger.info(f"图表保存路径: {self.plot_path}")
        except Exception as e:
            logger.error(f"创建保存目录失败: {str(e)}")
            logger.debug(traceback.format_exc())
            # 如果无法创建目录，尝试使用当前目录
            self.plot_path = "."
            logger.warning(f"使用当前目录作为备用保存路径: {os.path.abspath(self.plot_path)}")
        
        # 设置图表样式和默认参数
        try:
            # 使用 ggplot 样式，更美观
            plt.style.use('ggplot')
            # 设置中文字体支持（如果有）
            self._setup_fonts()
        except Exception as e:
            logger.warning(f"设置图表样式失败: {str(e)}")
    
    def _setup_fonts(self):
        """设置Matplotlib字体，支持中文显示"""
        try:
            from matplotlib import font_manager
            # 尝试查找系统中的中文字体
            chinese_fonts = [
                'SimHei', 'Microsoft YaHei', 'STHeiti', 'WenQuanYi Micro Hei',
                'PingFang SC', 'Hiragino Sans GB', 'Source Han Sans CN'
            ]
            
            # 查找是否有系统中文字体
            font_found = False
            for font in chinese_fonts:
                try:
                    if any(font.lower() in f.lower() for f in font_manager.findSystemFonts()):
                        plt.rcParams['font.family'] = [font, 'sans-serif']
                        font_found = True
                        logger.info(f"使用中文字体: {font}")
                        break
                except:
                    continue
            
            if not font_found:
                logger.warning("未找到合适的中文字体，图表中的中文可能显示为方块")
                
            # 修复负号显示问题
            plt.rcParams['axes.unicode_minus'] = False
            
        except Exception as e:
            logger.warning(f"设置中文字体失败: {str(e)}")

    def save_figure(self, fig: Figure, filename: str) -> str:
        """保存图表到文件
        
        参数:
            fig: matplotlib Figure 对象
            filename: 保存的文件名
            
        返回:
            str: 完整的文件路径，失败则返回空字符串
        """
        try:
            # 确保文件名有效
            filename = self._sanitize_filename(filename)
            
            # 确保文件名有正确的扩展名
            if not filename.endswith(('.png', '.jpg', '.jpeg', '.svg', '.pdf')):
                filename += '.png'

            # 构建完整路径
            filepath = os.path.join(self.plot_path, filename)

            # 创建目录结构（如果需要）
            dir_path = os.path.dirname(filepath)
            if dir_path and not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)

            # 保存图表，使用更高的DPI以提高质量
            fig.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close(fig)  # 关闭图表释放内存
            logger.info(f"图表已保存: {filepath}")

            return filepath
        except Exception as e:
            logger.error(f"保存图表失败: {str(e)}")
            logger.debug(traceback.format_exc())
            # 关闭图表避免内存泄漏
            try:
                plt.close(fig)
            except:
                pass
            return ""
    
    def _sanitize_filename(self, filename: str) -> str:
        """净化文件名，移除不合法字符"""
        # 移除不合法字符
        invalid_chars = r'<>:"/\|?*'
        for char in invalid_chars:
            filename = filename.replace(char, '_')
        # 确保文件名不以空格开始或结束
        filename = filename.strip()
        # 如果文件名为空，使用默认名称
        if not filename:
            filename = 'figure.png'
        return filename
            
    def create_figure(self, figsize=None, dpi=100) -> Tuple[Figure, plt.Axes]:
        """创建新的图表
        
        参数:
            figsize: 可选的图表大小，默认使用self.plot_resolution
            dpi: 图表DPI
        
        返回:
            Tuple[Figure, Axes]: 图表对象和坐标轴对象
        """
        try:
            if figsize is None:
                figsize = self.plot_resolution
                
            # 创建图表和轴对象
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
            
            # 设置默认的网格和边框
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # 返回创建的图表和轴
            return fig, ax
        except Exception as e:
            logger.error(f"创建图表失败: {str(e)}")
            logger.debug(traceback.format_exc())
            # 返回fallback的小尺寸图表
            try:
                return plt.subplots(figsize=(8, 6))
            except:
                # 如果还是失败，返回最小尺寸
                return plt.subplots(figsize=(5, 4))


class TrainingVisualizer(StaticVisualizer):
    """训练过程可视化
    
    提供各种训练指标的可视化功能，包括奖励曲线、成功率和损失等。
    """

    def plot_rewards(self, rewards: List[float], window_size: int = 10,
                     filename: str = 'rewards.png', title: str = '训练奖励曲线',
                     y_label: str = '奖励') -> str:
        """绘制奖励曲线
        
        参数:
            rewards: 奖励值列表
            window_size: 滑动平均窗口大小
            filename: 保存的文件名
            title: 图表标题
            y_label: y轴标签
            
        返回:
            str: 保存的文件路径
        """
        try:
            # 创建图表
            fig, ax = self.create_figure()

            if not rewards:
                logger.warning("无奖励数据可绘制")
                ax.text(0.5, 0.5, '无数据', ha='center', va='center')
                return self.save_figure(fig, filename)

            # 绘制原始奖励
            episodes = np.arange(1, len(rewards) + 1)
            ax.plot(episodes, rewards, 'b-', alpha=0.3, label='原始奖励')

            # 计算滑动平均
            if len(rewards) >= window_size:
                smoothed_rewards = np.convolve(rewards, np.ones(window_size) / window_size, mode='valid')
                # 调整x轴以对齐
                smoothed_episodes = np.arange(window_size, len(rewards) + 1)
                ax.plot(smoothed_episodes, smoothed_rewards, 'r-', label=f'滑动平均 (窗口={window_size})')

            # 设置图表属性
            ax.set_xlabel('训练回合')
            ax.set_ylabel(y_label)
            ax.set_title(title)
            ax.legend()
            ax.grid(True)

            # 保存图表
            return self.save_figure(fig, filename)
        except Exception as e:
            logger.error(f"绘制奖励曲线失败: {str(e)}")
            logger.debug(traceback.format_exc())
            return ""

    def plot_success_rate(self, success_history: List[bool], window_size: int = 100,
                          filename: str = 'success_rate.png') -> str:
        """绘制成功率曲线
        
        参数:
            success_history: 成功历史列表，True表示成功，False表示失败
            window_size: 滑动窗口大小
            filename: 保存的文件名
            
        返回:
            str: 保存的文件路径
        """
        try:
            # 创建图表
            fig, ax = self.create_figure()
            
            if not success_history:
                logger.warning("无成功率数据可绘制")
                ax.text(0.5, 0.5, '无数据', ha='center', va='center')
                return self.save_figure(fig, filename)

            # 转换为数值
            success_values = np.array(success_history, dtype=int)

            # 计算累积成功率
            cumulative_success_rate = np.cumsum(success_values) / np.arange(1, len(success_values) + 1)

            # 绘制累积成功率
            episodes = np.arange(1, len(success_values) + 1)
            ax.plot(episodes, cumulative_success_rate, 'g-', label='累积成功率')

            # 计算滑动窗口成功率
            if len(success_values) >= window_size:
                window_success_rate = np.convolve(success_values, np.ones(window_size) / window_size, mode='valid')
                window_episodes = np.arange(window_size, len(success_values) + 1)
                ax.plot(window_episodes, window_success_rate, 'b-', label=f'滑动成功率 (窗口={window_size})')

            # 设置图表属性
            ax.set_xlabel('训练回合')
            ax.set_ylabel('成功率')
            ax.set_title('训练成功率曲线')
            ax.set_ylim([0, 1.05])
            ax.legend()
            ax.grid(True)

            # 保存图表
            return self.save_figure(fig, filename)
        except Exception as e:
            logger.error(f"绘制成功率曲线失败: {str(e)}")
            logger.debug(traceback.format_exc())
            return ""

    def plot_metrics(self, metrics: Dict[str, List[float]], window_size: int = 10,
                     filename: str = 'metrics.png') -> str:
        """绘制多个指标曲线
        
        参数:
            metrics: 指标字典，键为指标名称，值为指标值列表
            window_size: 滑动平均窗口大小
            filename: 保存的文件名
            
        返回:
            str: 保存的文件路径
        """
        try:
            # 创建图表
            fig, ax = self.create_figure()
            
            if not metrics:
                logger.warning("无指标数据可绘制")
                ax.text(0.5, 0.5, '无数据', ha='center', va='center')
                return self.save_figure(fig, filename)

            # 检查是否所有指标长度都一致
            lengths = [len(values) for values in metrics.values()]
            if len(set(lengths)) > 1:
                logger.warning("指标长度不一致，可能导致图表异常")

            # 为每个指标绘制曲线
            for metric_name, values in metrics.items():
                if not values:
                    logger.warning(f"指标 {metric_name} 没有数据")
                    continue
                    
                episodes = np.arange(1, len(values) + 1)

                # 绘制原始值
                ax.plot(episodes, values, alpha=0.3)

                # 计算滑动平均
                if len(values) >= window_size:
                    smoothed_values = np.convolve(values, np.ones(window_size) / window_size, mode='valid')
                    smoothed_episodes = np.arange(window_size, len(values) + 1)
                    ax.plot(smoothed_episodes, smoothed_values, label=metric_name)
                else:
                    ax.plot(episodes, values, label=metric_name)

            # 设置图表属性
            ax.set_xlabel('训练回合')
            ax.set_ylabel('指标值')
            ax.set_title('训练指标曲线')
            ax.legend()
            ax.grid(True)

            # 保存图表
            return self.save_figure(fig, filename)
        except Exception as e:
            logger.error(f"绘制指标曲线失败: {str(e)}")
            logger.debug(traceback.format_exc())
            return ""

    def plot_loss(self, losses: Dict[str, List[float]], window_size: int = 100,
                  filename: str = 'loss.png', use_log_scale: bool = True) -> str:
        """绘制损失曲线
        
        参数:
            losses: 损失字典，键为损失名称，值为损失值列表
            window_size: 滑动平均窗口大小
            filename: 保存的文件名
            use_log_scale: 是否使用对数刻度
            
        返回:
            str: 保存的文件路径
        """
        try:
            # 创建图表
            fig, ax = self.create_figure()
            
            if not losses:
                logger.warning("无损失数据可绘制")
                ax.text(0.5, 0.5, '无数据', ha='center', va='center')
                return self.save_figure(fig, filename)

            # 为每种损失绘制曲线
            for loss_name, values in losses.items():
                if not values:
                    logger.warning(f"损失 {loss_name} 没有数据")
                    continue
                    
                iterations = np.arange(1, len(values) + 1)

                # 绘制原始损失
                ax.plot(iterations, values, alpha=0.2)

                # 计算滑动平均
                if len(values) >= window_size:
                    smoothed_values = np.convolve(values, np.ones(window_size) / window_size, mode='valid')
                    smoothed_iterations = np.arange(window_size, len(values) + 1)
                    ax.plot(smoothed_iterations, smoothed_values, label=loss_name)
                else:
                    ax.plot(iterations, values, label=loss_name)

            # 设置图表属性
            ax.set_xlabel('迭代次数')
            ax.set_ylabel('损失值')
            ax.set_title('训练损失曲线')
            ax.legend()
            ax.grid(True)

            # 使用对数刻度
            if use_log_scale:
                ax.set_yscale('log')

            # 保存图表
            return self.save_figure(fig, filename)
        except Exception as e:
            logger.error(f"绘制损失曲线失败: {str(e)}")
            logger.debug(traceback.format_exc())
            return ""


class ComparisonVisualizer(StaticVisualizer):
    """比较可视化
    
    提供不同算法和参数配置的比较功能，以及消融实验结果的可视化。
    """

    def plot_algorithm_comparison(self, results: Dict[str, List[float]], window_size: int = 10,
                                  filename: str = 'algorithm_comparison.png') -> str:
        """比较不同算法的性能
        
        参数:
            results: 算法结果字典，键为算法名称，值为奖励列表
            window_size: 滑动平均窗口大小
            filename: 保存的文件名
            
        返回:
            str: 保存的文件路径
        """
        try:
            # 创建图表
            fig, ax = self.create_figure()
            
            if not results:
                logger.warning("无算法比较数据可绘制")
                ax.text(0.5, 0.5, '无数据', ha='center', va='center')
                return self.save_figure(fig, filename)

            # 为每个算法绘制曲线
            for algo_name, rewards in results.items():
                if not rewards:
                    logger.warning(f"算法 {algo_name} 没有数据")
                    continue
                    
                episodes = np.arange(1, len(rewards) + 1)

                # 计算滑动平均
                if len(rewards) >= window_size:
                    smoothed_rewards = np.convolve(rewards, np.ones(window_size) / window_size, mode='valid')
                    smoothed_episodes = np.arange(window_size, len(rewards) + 1)
                    ax.plot(smoothed_episodes, smoothed_rewards, label=algo_name)
                else:
                    ax.plot(episodes, rewards, label=algo_name)

            # 设置图表属性
            ax.set_xlabel('训练回合')
            ax.set_ylabel('平均奖励')
            ax.set_title('算法性能比较')
            ax.legend()
            ax.grid(True)

            # 保存图表
            return self.save_figure(fig, filename)
        except Exception as e:
            logger.error(f"绘制算法比较失败: {str(e)}")
            logger.debug(traceback.format_exc())
            return ""

    def plot_parameter_comparison(self, parameter_values: List[Any], results: List[float],
                                  param_name: str = 'Parameter',
                                  filename: str = 'parameter_comparison.png') -> str:
        """比较不同参数值的性能
        
        参数:
            parameter_values: 参数值列表
            results: 对应的结果列表
            param_name: 参数名称
            filename: 保存的文件名
            
        返回:
            str: 保存的文件路径
        """
        try:
            # 创建图表
            fig, ax = self.create_figure()
            
            if not parameter_values or not results:
                logger.warning("无参数比较数据可绘制")
                ax.text(0.5, 0.5, '无数据', ha='center', va='center')
                return self.save_figure(fig, filename)
                
            if len(parameter_values) != len(results):
                logger.error(f"参数值数量({len(parameter_values)})与结果数量({len(results)})不一致")
                ax.text(0.5, 0.5, '参数值与结果数量不一致', ha='center', va='center')
                return self.save_figure(fig, filename)

            # 绘制参数比较曲线
            ax.plot(parameter_values, results, 'bo-')

            # 标记最佳值
            if results:
                best_idx = np.argmax(results)
                best_param = parameter_values[best_idx]
                best_result = results[best_idx]
                ax.plot(best_param, best_result, 'ro', markersize=10)
                ax.annotate(f'最佳: {best_param}',
                            (best_param, best_result),
                            xytext=(10, -20),
                            textcoords='offset points',
                            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))

            # 设置图表属性
            ax.set_xlabel(param_name)
            ax.set_ylabel('性能指标')
            ax.set_title(f'{param_name}参数调优')
            ax.grid(True)

            # 保存图表
            return self.save_figure(fig, filename)
        except Exception as e:
            logger.error(f"绘制参数比较失败: {str(e)}")
            logger.debug(traceback.format_exc())
            return ""

    def plot_ablation_study(self, components: List[str], results: List[float],
                            filename: str = 'ablation_study.png') -> str:
        """绘制消融实验结果
        
        参数:
            components: 组件名称列表
            results: 对应的结果列表
            filename: 保存的文件名
            
        返回:
            str: 保存的文件路径
        """
        try:
            # 创建图表
            fig, ax = self.create_figure()
            
            if not components or not results:
                logger.warning("无消融实验数据可绘制")
                ax.text(0.5, 0.5, '无数据', ha='center', va='center')
                return self.save_figure(fig, filename)
                
            if len(components) != len(results):
                logger.error(f"组件数量({len(components)})与结果数量({len(results)})不一致")
                ax.text(0.5, 0.5, '组件与结果数量不一致', ha='center', va='center')
                return self.save_figure(fig, filename)

            # 绘制条形图
            bars = ax.bar(components, results)

            # 添加数值标签
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.2f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3点垂直偏移
                            textcoords="offset points",
                            ha='center', va='bottom')

            # 设置图表属性
            ax.set_xlabel('组件')
            ax.set_ylabel('性能指标')
            ax.set_title('消融实验结果')
            ax.grid(True, axis='y')

            # 保存图表
            return self.save_figure(fig, filename)
        except Exception as e:
            logger.error(f"绘制消融实验结果失败: {str(e)}")
            logger.debug(traceback.format_exc())
            return ""


class EnvironmentVisualizer(StaticVisualizer):
    """环境可视化
    
    提供环境状态和轨迹的可视化功能，包括轨迹图、热力图、传感器数据图等。
    """

    def plot_trajectory(self, trajectory: List[Tuple[float, float]],
                        obstacles: List[Tuple] = None,
                        start_pos: Tuple[float, float] = None,
                        target_pos: Tuple[float, float] = None,
                        filename: str = 'trajectory.png') -> str:
        """绘制小车轨迹
        
        参数:
            trajectory: 轨迹坐标列表，每个元素为 (x, y) 坐标
            obstacles: 障碍物列表，每个元素为圆形 (x, y, radius) 或矩形 (x, y, width, height)
            start_pos: 起始位置，(x, y) 坐标
            target_pos: 目标位置，(x, y) 坐标
            filename: 保存的文件名
            
        返回:
            str: 保存的文件路径
        """
        try:
            # 创建图表
            fig, ax = self.create_figure()
            
            if not trajectory:
                logger.warning("无轨迹数据可绘制")
                ax.text(0.5, 0.5, '无轨迹数据', ha='center', va='center')
                return self.save_figure(fig, filename)

            # 提取轨迹坐标
            x_coords, y_coords = zip(*trajectory)

            # 绘制轨迹
            ax.plot(x_coords, y_coords, 'b-', alpha=0.7, label='轨迹')

            # 绘制起点和终点
            if start_pos:
                ax.plot(start_pos[0], start_pos[1], 'go', markersize=10, label='起点')
            if target_pos:
                ax.plot(target_pos[0], target_pos[1], 'ro', markersize=10, label='终点')

            # 绘制障碍物
            if obstacles:
                for obstacle in obstacles:
                    try:
                        if len(obstacle) == 3:  # 圆形障碍物 (x, y, radius)
                            x, y, radius = obstacle
                            circle = plt.Circle((x, y), radius, color='gray', alpha=0.5)
                            ax.add_patch(circle)
                        elif len(obstacle) == 4:  # 矩形障碍物 (x, y, width, height)
                            x, y, width, height = obstacle
                            rectangle = plt.Rectangle((x, y), width, height, color='gray', alpha=0.5)
                            ax.add_patch(rectangle)
                    except Exception as e:
                        logger.warning(f"绘制障碍物失败: {str(e)}")

            # 设置图表属性
            ax.set_xlabel('X 坐标')
            ax.set_ylabel('Y 坐标')
            ax.set_title('小车轨迹')
            ax.legend()
            ax.grid(True)
            ax.set_aspect('equal')  # 保持坐标轴比例一致

            # 保存图表
            return self.save_figure(fig, filename)
        except Exception as e:
            logger.error(f"绘制轨迹失败: {str(e)}")
            logger.debug(traceback.format_exc())
            return ""

    def plot_heatmap(self, data: np.ndarray, x_range: Tuple[float, float] = None,
                     y_range: Tuple[float, float] = None, title: str = '热力图',
                     filename: str = 'heatmap.png') -> str:
        """绘制环境热力图
        
        参数:
            data: 热力图数据，二维数组
            x_range: x轴范围 (min, max)
            y_range: y轴范围 (min, max)
            title: 图表标题
            filename: 保存的文件名
            
        返回:
            str: 保存的文件路径
        """
        try:
            # 创建图表
            fig, ax = self.create_figure()
            
            if data is None or data.size == 0:
                logger.warning("无热力图数据可绘制")
                ax.text(0.5, 0.5, '无数据', ha='center', va='center')
                return self.save_figure(fig, filename)

            # 检查数据维度
            if len(data.shape) != 2:
                logger.error(f"热力图数据必须是二维数组，当前维度: {data.shape}")
                ax.text(0.5, 0.5, '数据维度错误', ha='center', va='center')
                return self.save_figure(fig, filename)

            # 绘制热力图
            im = ax.imshow(data, cmap='viridis', origin='lower', aspect='auto')

            # 设置坐标轴范围
            if x_range and y_range:
                # 计算坐标轴刻度
                height, width = data.shape
                x_ticks = np.linspace(0, width - 1, 5)
                y_ticks = np.linspace(0, height - 1, 5)
                x_labels = np.linspace(x_range[0], x_range[1], 5)
                y_labels = np.linspace(y_range[0], y_range[1], 5)

                ax.set_xticks(x_ticks)
                ax.set_yticks(y_ticks)
                ax.set_xticklabels([f'{x:.1f}' for x in x_labels])
                ax.set_yticklabels([f'{y:.1f}' for y in y_labels])

            # 添加颜色条
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label('值')

            # 设置图表属性
            ax.set_xlabel('X 坐标')
            ax.set_ylabel('Y 坐标')
            ax.set_title(title)

            # 保存图表
            return self.save_figure(fig, filename)
        except Exception as e:
            logger.error(f"绘制热力图失败: {str(e)}")
            logger.debug(traceback.format_exc())
            return ""

    def plot_sensor_data(self, sensor_data: List[float], sensor_angles: List[float] = None,
                         filename: str = 'sensor_data.png') -> str:
        """绘制传感器数据
        
        参数:
            sensor_data: 传感器数据列表
            sensor_angles: 传感器角度列表，单位为弧度
            filename: 保存的文件名
            
        返回:
            str: 保存的文件路径
        """
        try:
            # 创建极坐标图表
            fig, ax = plt.subplots(figsize=self.plot_resolution, subplot_kw={'projection': 'polar'})
            
            if not sensor_data:
                logger.warning("无传感器数据可绘制")
                # 在极坐标图中添加文本较为复杂，这里不添加
                return self.save_figure(fig, filename)

            # 如果没有提供角度，则均匀分布在360度
            if sensor_angles is None:
                sensor_angles = np.linspace(0, 2 * np.pi, len(sensor_data), endpoint=False)
            elif len(sensor_angles) != len(sensor_data):
                logger.warning(f"传感器角度数量({len(sensor_angles)})与数据数量({len(sensor_data)})不一致")
                # 重新生成角度
                sensor_angles = np.linspace(0, 2 * np.pi, len(sensor_data), endpoint=False)

            # 绘制雷达图
            ax.plot(sensor_angles, sensor_data, 'b-')
            ax.fill(sensor_angles, sensor_data, 'b', alpha=0.2)

            # 设置图表属性
            ax.set_theta_zero_location('N')  # 设置0度在北方(上方)
            ax.set_theta_direction(-1)  # 设置角度顺时针增加
            ax.set_title('传感器数据')
            ax.grid(True)

            # 保存图表
            return self.save_figure(fig, filename)
        except Exception as e:
            logger.error(f"绘制传感器数据失败: {str(e)}")
            logger.debug(traceback.format_exc())
            return ""

    def plot_state_distribution(self, states: List[Tuple[float, float]],
                                filename: str = 'state_distribution.png') -> str:
        """绘制状态分布
        
        参数:
            states: 状态列表，每个元素为 (x, y) 坐标
            filename: 保存的文件名
            
        返回:
            str: 保存的文件路径
        """
        try:
            # 创建图表
            fig, ax = self.create_figure()
            
            if not states:
                logger.warning("无状态分布数据可绘制")
                ax.text(0.5, 0.5, '无数据', ha='center', va='center')
                return self.save_figure(fig, filename)

            # 提取坐标
            x_coords, y_coords = zip(*states)

            # 绘制散点图
            ax.scatter(x_coords, y_coords, alpha=0.5, s=10)

            # 尝试绘制密度等高线
            try:
                from scipy.stats import gaussian_kde
                # 计算核密度估计
                xy = np.vstack([x_coords, y_coords])
                z = gaussian_kde(xy)(xy)

                # 按密度排序点
                idx = z.argsort()
                x_sorted, y_sorted, z_sorted = np.array(x_coords)[idx], np.array(y_coords)[idx], z[idx]

                # 绘制带颜色的散点图
                scatter = ax.scatter(x_sorted, y_sorted, c=z_sorted, s=20, cmap='viridis')

                # 添加颜色条
                cbar = fig.colorbar(scatter, ax=ax)
                cbar.set_label('密度')
            except ImportError:
                logger.warning("未安装 scipy 库，无法绘制密度图")
            except Exception as e:
                logger.warning(f"绘制密度图失败: {str(e)}")

            # 设置图表属性
            ax.set_xlabel('X 坐标')
            ax.set_ylabel('Y 坐标')
            ax.set_title('状态分布')
            ax.grid(True)
            ax.set_aspect('equal')  # 保持坐标轴比例一致

            # 保存图表
            return self.save_figure(fig, filename)
        except Exception as e:
            logger.error(f"绘制状态分布失败: {str(e)}")
            logger.debug(traceback.format_exc())
            return ""