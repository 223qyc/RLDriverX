"""
动态可视化模块
提供了一系列可视化类，用于实时展示智能体行为和环境状态，
支持单智能体和多智能体场景，以及视频录制功能。
"""
import pygame
import numpy as np
import os
import time
import logging
import traceback
from typing import Dict, List, Tuple, Union, Optional, Any

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger('visualization')


class DynamicVisualizer:
    """动态可视化基类

    提供所有可视化器的基础功能，包括初始化、渲染、录制和关闭。

    属性:
        config: 配置对象，包含环境宽高和渲染参数
        width: 可视化窗口宽度
        height: 可视化窗口高度
        initialized: 是否已初始化
        screen: pygame 屏幕对象
        clock: pygame 时钟对象
        recording: 是否正在录制
        frames: 录制的帧列表
        paused: 是否暂停
        running: 是否正在运行
    """

    def __init__(self, config):
        """初始化可视化器

        参数:
            config: 配置对象，必须包含 ENV_WIDTH, ENV_HEIGHT 和 RENDER_FPS
        """
        self.config = config
        self.width = getattr(config, 'ENV_WIDTH', 800)
        self.height = getattr(config, 'ENV_HEIGHT', 600)
        self.render_fps = getattr(config, 'RENDER_FPS', 30)
        self.initialized = False
        self.screen = None
        self.clock = None
        self.recording = False
        self.frames = []
        self.output_filename = None  # 添加输出文件名属性
        self.paused = False
        self.running = True
        self.background_color = (255, 255, 255)  # 默认白色背景

    def initialize(self):
        """初始化可视化环境

        初始化 pygame 环境，创建屏幕和时钟对象
        """
        if not self.initialized:
            try:
                pygame.init()
                self.screen = pygame.display.set_mode((self.width, self.height))
                pygame.display.set_caption("可视化窗口")
                self.clock = pygame.time.Clock()
                self.initialized = True
                logger.info("可视化器初始化完成，分辨率: %dx%d", self.width, self.height)
            except Exception as e:
                logger.error("可视化器初始化失败: %s", str(e))
                logger.debug(traceback.format_exc())
                raise

    def start_recording(self, filename=None):
        """开始记录帧
        
        参数:
            filename: 录制完成后保存的文件路径
        """
        if not self.initialized:
            self.initialize()
        self.recording = True
        self.frames = []
        self.output_filename = filename
        logger.info(f"开始录制{' 保存至: ' + filename if filename else ''}")

    def stop_recording(self, filename=None):
        """停止记录并保存视频

        参数:
            filename: 保存的文件名，支持.gif和.mp4格式
        """
        if not self.recording or not self.frames:
            logger.warning("没有录制内容可保存")
            return

        self.recording = False

        # 优先使用传入的filename，其次使用start_recording时设置的filename，最后使用默认名称
        if not filename and hasattr(self, 'output_filename') and self.output_filename:
            filename = self.output_filename
        
        if not filename:
            timestamp = int(time.time())
            filename = f"recording_{timestamp}.mp4"

        try:
            import imageio
            # 确保目录存在
            os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)

            # 保存为GIF或MP4
            if filename.endswith('.gif'):
                imageio.mimsave(filename, self.frames, fps=self.render_fps)
            else:
                # 修复TiffWriter错误，使用正确的FFMPEG格式
                try:
                    # 尝试使用get_writer方式保存
                    writer = imageio.get_writer(filename, format='FFMPEG', fps=self.render_fps)
                    for frame in self.frames:
                        writer.append_data(frame)
                    writer.close()
                except Exception as e:
                    logger.warning(f"使用imageio.get_writer保存视频失败: {str(e)}，尝试其他方法")
                    # 备用方法：使用mimsave来保存mp4
                    try:
                        imageio.mimsave(filename, self.frames, fps=self.render_fps, format='FFMPEG')
                    except Exception as e2:
                        logger.warning(f"使用imageio.mimsave保存视频失败: {str(e2)}，尝试最后的备用方法")
                        # 最后的备用方法：保存为gif格式，然后转换
                        temp_filename = filename.replace('.mp4', '.gif')
                        imageio.mimsave(temp_filename, self.frames, fps=min(self.render_fps, 30))
                        logger.info(f"已保存为临时GIF文件: {temp_filename}")
                        logger.warning(f"无法保存为MP4格式。如需MP4，请手动转换GIF文件。")
                        
            logger.info(f"视频已保存到 {filename}")
        except ImportError:
            logger.error("需要安装imageio库来保存视频，请运行: pip install imageio imageio-ffmpeg")
        except Exception as e:
            logger.error(f"保存视频失败: {str(e)}")
            logger.debug(traceback.format_exc())

        self.frames = []

    def toggle_pause(self):
        """切换暂停状态"""
        self.paused = not self.paused
        status = "暂停" if self.paused else "继续"
        logger.info(f"可视化 {status}")
        return self.paused

    def handle_events(self):
        """处理pygame事件

        处理基本事件如退出和暂停

        返回:
            bool: 是否继续运行
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                self.close()
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                    self.close()
                    return False
                elif event.key == pygame.K_SPACE:
                    self.toggle_pause()
                elif event.key == pygame.K_r:
                    if self.recording:
                        self.stop_recording()
                    else:
                        self.start_recording()

        return True

    def render(self, state, info=None):
        """渲染当前状态

        参数:
            state: 当前状态
            info: 附加信息

        返回:
            bool: 是否继续运行
        """
        raise NotImplementedError("子类必须实现此方法")

    def capture_frame(self):
        """捕获当前帧用于录制"""
        if self.recording and self.initialized:
            try:
                pygame_surface = pygame.surfarray.array3d(self.screen)
                # 转换为RGB格式
                pygame_surface = pygame_surface.transpose([1, 0, 2])
                self.frames.append(pygame_surface)
            except Exception as e:
                logger.error(f"捕获帧失败: {str(e)}")
                logger.debug(traceback.format_exc())

    def close(self):
        """关闭可视化环境"""
        if self.recording:
            # 使用之前设置的文件名或生成新的文件名
            output_file = None
            if hasattr(self, 'output_filename') and self.output_filename:
                output_file = self.output_filename
            else:
                timestamp = int(time.time())
                output_file = f"recording_{timestamp}.mp4"
            self.stop_recording(output_file)

        if self.initialized:
            pygame.quit()
            self.initialized = False
            logger.info("可视化器已关闭")


class CarVisualizer(DynamicVisualizer):
    """小车可视化器

    用于单智能体场景的可视化，支持小车、目标、障碍物和传感器的绘制。

    属性:
        car_size: 小车尺寸
        car_color: 小车颜色
        target_color: 目标颜色
        obstacle_color: 障碍物颜色
        sensor_color: 传感器线颜色
        background_color: 背景颜色
        font: 字体对象
        show_sensors: 是否显示传感器
        show_info: 是否显示信息面板
        trail_points: 轨迹点列表
        max_trail_length: 最大轨迹长度
    """

    def __init__(self, config):
        """初始化小车可视化器

        参数:
            config: 配置对象
        """
        super().__init__(config)
        # 从配置中获取车辆尺寸，若不存在则使用默认值
        self.car_size = getattr(config, 'CAR_SIZE', 20)
        self.car_color = (255, 0, 0)  # 红色小车
        self.target_color = (0, 255, 0)  # 绿色目标
        self.obstacle_color = (0, 0, 255)  # 蓝色障碍物
        self.sensor_color = (255, 255, 0)  # 黄色传感器线
        self.background_color = (255, 255, 255)  # 白色背景
        self.font = None
        self.show_sensors = True
        self.show_info = True
        self.trail_points = []
        self.max_trail_length = 100
        self.trail_enabled = False
        self.trail_color = (200, 200, 200)  # 轨迹颜色

    def initialize(self):
        """初始化可视化环境"""
        super().initialize()
        if self.initialized:
            try:
                self.font = pygame.font.SysFont('Arial', 16)
                pygame.display.set_caption("小车控制可视化")
            except Exception as e:
                logger.error(f"小车可视化器初始化字体失败: {str(e)}")
                # 继续执行，只是没有字体

    def toggle_sensors(self):
        """切换传感器显示状态"""
        self.show_sensors = not self.show_sensors
        return self.show_sensors

    def toggle_info(self):
        """切换信息面板显示状态"""
        self.show_info = not self.show_info
        return self.show_info

    def toggle_trail(self):
        """切换轨迹显示状态"""
        self.trail_enabled = not self.trail_enabled
        if not self.trail_enabled:
            self.trail_points.clear()
        return self.trail_enabled

    def handle_events(self):
        """处理pygame事件，扩展基类功能"""
        running = super().handle_events()
        if not running:
            return False

        # 处理额外的事件
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_s:
                    self.toggle_sensors()
                    logger.info(f"传感器显示 {'开启' if self.show_sensors else '关闭'}")
                elif event.key == pygame.K_i:
                    self.toggle_info()
                    logger.info(f"信息面板 {'开启' if self.show_info else '关闭'}")
                elif event.key == pygame.K_t:
                    self.toggle_trail()
                    logger.info(f"轨迹显示 {'开启' if self.trail_enabled else '关闭'}")

        return True

    def render(self, state, info=None):
        """渲染当前状态

        参数:
            state: 包含位置、角度、目标和障碍物等信息的字典
            info: 额外的信息字典

        返回:
            bool: 是否继续运行
        """
        if not self.initialized:
            self.initialize()

        if self.paused:
            # 更新时钟但不渲染新帧
            self.clock.tick(self.config.RENDER_FPS)
            return self.running

        try:
            # 清空屏幕
            self.screen.fill(self.background_color)

            # 检查状态是否有效
            if not isinstance(state, dict):
                logger.warning("无效的状态数据类型：%s", type(state))
                pygame.display.flip()
                self.clock.tick(self.config.RENDER_FPS)
                return self.running

            # 确保状态包含必要的键
            if 'position' not in state or 'angle' not in state:
                logger.warning("状态缺少必要的键(position或angle)")
                pygame.display.flip()
                self.clock.tick(self.config.RENDER_FPS)
                return self.running

            # 提取小车状态
            car_x, car_y = state['position']
            car_angle = state['angle']

            # 更新轨迹
            if self.trail_enabled:
                self.trail_points.append((int(car_x), int(car_y)))
                if len(self.trail_points) > self.max_trail_length:
                    self.trail_points.pop(0)

                # 绘制轨迹
                if len(self.trail_points) > 1:
                    pygame.draw.lines(self.screen, self.trail_color, False, self.trail_points, 2)

            # 绘制小车
            self._draw_car(car_x, car_y, car_angle)

            # 绘制目标
            if 'target' in state:
                target_x, target_y = state['target']
                pygame.draw.circle(self.screen, self.target_color, (int(target_x), int(target_y)), 10)

            # 绘制障碍物
            if 'obstacles' in state:
                for obstacle in state['obstacles']:
                    if len(obstacle) == 3:  # 圆形障碍物 (x, y, radius)
                        x, y, radius = obstacle
                        pygame.draw.circle(self.screen, self.obstacle_color, (int(x), int(y)), int(radius))
                    elif len(obstacle) == 4:  # 矩形障碍物 (x, y, width, height)
                        x, y, width, height = obstacle
                        pygame.draw.rect(self.screen, self.obstacle_color, (int(x), int(y), int(width), int(height)))

            # 绘制传感器
            if self.show_sensors and 'sensors' in state:
                self._draw_sensors(car_x, car_y, car_angle, state['sensors'])

            # 绘制信息
            if self.show_info and info:
                self._draw_info(info)

            # 添加操作说明
            controls_text = "控制: [ESC] 退出 [空格] 暂停 [S] 传感器 [I] 信息 [T] 轨迹 [R] 录制"
            if self.font:
                controls_surface = self.font.render(controls_text, True, (50, 50, 50))
                self.screen.blit(controls_surface, (10, self.height - 30))

            # 更新屏幕
            pygame.display.flip()
            self.clock.tick(self.render_fps)

            # 处理事件
            if not self.handle_events():
                return False

            # 记录帧
            self.capture_frame()

            return self.running

        except Exception as e:
            logger.error(f"渲染过程出错: {str(e)}")
            logger.debug(traceback.format_exc())
            return self.running

    def _draw_car(self, x, y, angle):
        """绘制小车

        参数:
            x: x坐标
            y: y坐标
            angle: 角度(弧度)
        """
        try:
            # 创建小车表面
            car_surface = pygame.Surface((self.car_size * 2, self.car_size * 2), pygame.SRCALPHA)

            # 绘制小车主体(矩形)
            pygame.draw.rect(car_surface, self.car_color,
                            (self.car_size // 2, self.car_size // 2, self.car_size, self.car_size))

            # 绘制小车前方指示器(三角形)
            front_indicator = [
                (self.car_size + self.car_size // 2, self.car_size),
                (self.car_size, self.car_size - self.car_size // 4),
                (self.car_size, self.car_size + self.car_size // 4)
            ]
            pygame.draw.polygon(car_surface, (0, 0, 0), front_indicator)

            # 旋转小车
            rotated_car = pygame.transform.rotate(car_surface, -angle * 180 / np.pi)

            # 获取旋转后的矩形
            car_rect = rotated_car.get_rect(center=(int(x), int(y)))

            # 绘制到屏幕
            self.screen.blit(rotated_car, car_rect.topleft)
        except Exception as e:
            logger.error(f"绘制小车失败: {str(e)}")

    def _draw_sensors(self, x, y, angle, sensors):
        """绘制传感器

        参数:
            x: 小车x坐标
            y: 小车y坐标
            angle: 小车朝向角度
            sensors: 传感器数据
        """
        try:
            if isinstance(sensors, dict):
                # 处理不同类型的传感器
                if 'lidar' in sensors:
                    self._draw_lidar(x, y, angle, sensors['lidar'])
                # 其他类型传感器的处理可以在这里添加
            elif isinstance(sensors, (list, np.ndarray)):
                # 假设是激光雷达数据
                self._draw_lidar(x, y, angle, sensors)
        except Exception as e:
            logger.error(f"绘制传感器失败: {str(e)}")

    def _draw_lidar(self, x, y, angle, distances):
        """绘制激光雷达

        参数:
            x: 小车x坐标
            y: 小车y坐标
            angle: 小车朝向角度
            distances: 距离数组
        """
        try:
            if not isinstance(distances, (list, np.ndarray)):
                return

            num_rays = len(distances)
            for i, distance in enumerate(distances):
                # 计算射线角度
                ray_angle = angle + (i / num_rays) * 2 * np.pi - np.pi
                # 计算射线终点
                end_x = x + np.cos(ray_angle) * distance
                end_y = y + np.sin(ray_angle) * distance
                # 绘制射线
                pygame.draw.line(self.screen, self.sensor_color,
                                (int(x), int(y)), (int(end_x), int(end_y)), 1)
        except Exception as e:
            logger.error(f"绘制激光雷达失败: {str(e)}")

    def _draw_info(self, info):
        """绘制信息

        参数:
            info: 信息字典
        """
        try:
            if not self.font:
                return

            y_offset = 10
            for key, value in info.items():
                if isinstance(value, (int, float)):
                    # 格式化数值
                    text = f"{key}: {value:.2f}" if isinstance(value, float) else f"{key}: {value}"
                else:
                    text = f"{key}: {value}"

                # 渲染文本
                text_surface = self.font.render(text, True, (0, 0, 0))
                self.screen.blit(text_surface, (10, y_offset))
                y_offset += 20
        except Exception as e:
            logger.error(f"绘制信息失败: {str(e)}")


class MultiAgentVisualizer(DynamicVisualizer):
    """多智能体可视化器

    用于多智能体场景的可视化，支持不同队伍、目标、障碍物和传感器的绘制。

    属性:
        car_size: 智能体尺寸
        agent_colors: 不同智能体的颜色列表
        target_color: 默认目标颜色
        obstacle_color: 障碍物颜色
        sensor_color: 传感器线颜色
        background_color: 背景颜色
        font: 字体对象
        team_colors: 队伍颜色字典
        show_sensors: 是否显示传感器
        show_info: 是否显示信息面板
        show_ids: 是否显示智能体ID
        trail_enabled: 是否显示轨迹
        trail_points: 每个智能体的轨迹点字典
        max_trail_length: 最大轨迹长度
    """

    def __init__(self, config):
        """初始化多智能体可视化器

        参数:
            config: 配置对象
        """
        super().__init__(config)
        # 从配置中获取车辆尺寸，若不存在则使用默认值
        self.car_size = getattr(config, 'CAR_SIZE', 20)
        # 不同智能体使用不同颜色
        self.agent_colors = [
            (255, 0, 0),  # 红色
            (0, 0, 255),  # 蓝色
            (0, 255, 0),  # 绿色
            (255, 255, 0),  # 黄色
            (255, 0, 255),  # 紫色
            (0, 255, 255),  # 青色
            (128, 0, 0),  # 深红色
            (0, 128, 0),  # 深绿色
            (0, 0, 128),  # 深蓝色
            (128, 128, 0)  # 橄榄色
        ]
        self.target_color = (0, 255, 0)  # 绿色目标
        self.obstacle_color = (100, 100, 100)  # 灰色障碍物
        self.sensor_color = (200, 200, 200)  # 浅灰色传感器线
        self.background_color = (255, 255, 255)  # 白色背景
        self.font = None
        self.team_colors = {
            'team_1': (255, 0, 0),  # 红队
            'team_2': (0, 0, 255)   # 蓝队
        }
        # 显示选项
        self.show_sensors = True
        self.show_info = True
        self.show_ids = True
        # 轨迹设置
        self.trail_enabled = False
        self.trail_points = {}  # 每个智能体的轨迹点 {agent_id: [(x1, y1), (x2, y2), ...]}
        self.max_trail_length = 50

    def initialize(self):
        """初始化可视化环境"""
        super().initialize()
        if self.initialized:
            try:
                self.font = pygame.font.SysFont('Arial', 16)
                pygame.display.set_caption("多智能体可视化")
            except Exception as e:
                logger.error(f"多智能体可视化器初始化字体失败: {str(e)}")
                # 继续执行，只是没有字体

    def toggle_sensors(self):
        """切换传感器显示状态"""
        self.show_sensors = not self.show_sensors
        return self.show_sensors

    def toggle_info(self):
        """切换信息面板显示状态"""
        self.show_info = not self.show_info
        return self.show_info

    def toggle_ids(self):
        """切换智能体ID显示状态"""
        self.show_ids = not self.show_ids
        return self.show_ids

    def toggle_trail(self):
        """切换轨迹显示状态"""
        self.trail_enabled = not self.trail_enabled
        if not self.trail_enabled:
            self.trail_points.clear()
        return self.trail_enabled

    def handle_events(self):
        """处理pygame事件，扩展基类功能"""
        running = super().handle_events()
        if not running:
            return False

        # 处理额外的事件
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_s:
                    self.toggle_sensors()
                    logger.info(f"传感器显示 {'开启' if self.show_sensors else '关闭'}")
                elif event.key == pygame.K_i:
                    self.toggle_info()
                    logger.info(f"信息面板 {'开启' if self.show_info else '关闭'}")
                elif event.key == pygame.K_d:
                    self.toggle_ids()
                    logger.info(f"智能体ID显示 {'开启' if self.show_ids else '关闭'}")
                elif event.key == pygame.K_t:
                    self.toggle_trail()
                    logger.info(f"轨迹显示 {'开启' if self.trail_enabled else '关闭'}")

        return True

    def render(self, state, info=None):
        """渲染当前状态

        参数:
            state: 包含多个智能体状态的字典
            info: 额外信息字典

        返回:
            bool: 是否继续运行
        """
        if not self.initialized:
            self.initialize()

        if self.paused:
            # 更新时钟但不渲染新帧
            self.clock.tick(self.render_fps)
            return self.running

        try:
            # 清空屏幕
            self.screen.fill(self.background_color)

            # 检查状态是否有效
            if not isinstance(state, dict):
                logger.warning("无效的状态数据类型：%s", type(state))
                pygame.display.flip()
                self.clock.tick(self.render_fps)
                return self.running

            # 绘制障碍物
            if 'obstacles' in state:
                for obstacle in state['obstacles']:
                    if len(obstacle) == 3:  # 圆形障碍物 (x, y, radius)
                        x, y, radius = obstacle
                        pygame.draw.circle(self.screen, self.obstacle_color, (int(x), int(y)), int(radius))
                    elif len(obstacle) == 4:  # 矩形障碍物 (x, y, width, height)
                        x, y, width, height = obstacle
                        pygame.draw.rect(self.screen, self.obstacle_color, (int(x), int(y), int(width), int(height)))

            # 绘制目标
            if 'targets' in state:
                for target in state['targets']:
                    target_x, target_y = target[:2]
                    # 如果目标有队伍属性
                    if len(target) > 2 and target[2] in self.team_colors:
                        target_color = self.team_colors[target[2]]
                    else:
                        target_color = self.target_color
                    pygame.draw.circle(self.screen, target_color, (int(target_x), int(target_y)), 10)

            # 绘制智能体
            if 'agents' in state:
                for i, agent in enumerate(state['agents']):
                    # 检查智能体是否有效
                    if not isinstance(agent, dict) or 'position' not in agent or 'angle' not in agent:
                        logger.warning(f"智能体 {i} 数据无效或缺少必要字段")
                        continue

                    # 获取智能体颜色
                    if 'team' in agent and agent['team'] in self.team_colors:
                        agent_color = self.team_colors[agent['team']]
                    else:
                        agent_color = self.agent_colors[i % len(self.agent_colors)]

                    # 获取智能体ID
                    agent_id = agent.get('id', i)

                    # 更新轨迹
                    if self.trail_enabled:
                        position = agent['position']
                        if agent_id not in self.trail_points:
                            self.trail_points[agent_id] = []
                        self.trail_points[agent_id].append((int(position[0]), int(position[1])))
                        if len(self.trail_points[agent_id]) > self.max_trail_length:
                            self.trail_points[agent_id].pop(0)

                        # 绘制轨迹
                        if len(self.trail_points[agent_id]) > 1:
                            # 根据智能体颜色确定轨迹颜色（但透明度降低）
                            trail_color = tuple(min(c + 50, 255) for c in agent_color[:3])
                            pygame.draw.lines(self.screen, trail_color, False, self.trail_points[agent_id], 2)

                    # 绘制智能体
                    self._draw_agent(agent, agent_color, agent_id)

            # 绘制信息
            if self.show_info and info:
                self._draw_info(info)

            # 添加操作说明
            controls_text = "控制: [ESC] 退出 [空格] 暂停 [S] 传感器 [I] 信息 [D] ID [T] 轨迹 [R] 录制"
            if self.font:
                controls_surface = self.font.render(controls_text, True, (50, 50, 50))
                self.screen.blit(controls_surface, (10, self.height - 30))

            # 更新屏幕
            pygame.display.flip()
            self.clock.tick(self.render_fps)

            # 处理事件
            if not self.handle_events():
                return False

            # 记录帧
            self.capture_frame()

            return self.running

        except Exception as e:
            logger.error(f"渲染过程出错: {str(e)}")
            logger.debug(traceback.format_exc())
            return self.running

    def _draw_agent(self, agent, color, agent_id):
        """绘制单个智能体

        参数:
            agent: 智能体数据字典
            color: 智能体颜色
            agent_id: 智能体ID
        """
        try:
            x, y = agent['position']
            angle = agent['angle']

            # 创建智能体表面
            agent_surface = pygame.Surface((self.car_size * 2, self.car_size * 2), pygame.SRCALPHA)

            # 绘制智能体主体(矩形)
            pygame.draw.rect(agent_surface, color,
                            (self.car_size // 2, self.car_size // 2, self.car_size, self.car_size))

            # 绘制智能体前方指示器(三角形)
            front_indicator = [
                (self.car_size + self.car_size // 2, self.car_size),
                (self.car_size, self.car_size - self.car_size // 4),
                (self.car_size, self.car_size + self.car_size // 4)
            ]
            pygame.draw.polygon(agent_surface, (0, 0, 0), front_indicator)

            # 绘制智能体ID
            if self.show_ids:
                try:
                    id_font = pygame.font.SysFont('Arial', 12)
                    id_text = id_font.render(str(agent_id), True, (255, 255, 255))
                    id_rect = id_text.get_rect(center=(self.car_size, self.car_size))
                    agent_surface.blit(id_text, id_rect)
                except Exception as e:
                    logger.debug(f"绘制智能体ID失败: {str(e)}")

            # 旋转智能体
            rotated_agent = pygame.transform.rotate(agent_surface, -angle * 180 / np.pi)

            # 获取旋转后的矩形
            agent_rect = rotated_agent.get_rect(center=(int(x), int(y)))

            # 绘制到屏幕
            self.screen.blit(rotated_agent, agent_rect.topleft)

            # 绘制传感器
            if self.show_sensors and 'sensors' in agent:
                self._draw_agent_sensors(x, y, angle, agent['sensors'])
        except Exception as e:
            logger.error(f"绘制智能体失败: {str(e)}")

    def _draw_agent_sensors(self, x, y, angle, sensors):
        """绘制智能体传感器

        参数:
            x: 智能体x坐标
            y: 智能体y坐标
            angle: 智能体朝向角度
            sensors: 传感器数据
        """
        try:
            if isinstance(sensors, dict):
                # 处理不同类型的传感器
                if 'lidar' in sensors:
                    self._draw_lidar(x, y, angle, sensors['lidar'])
                # 可以添加其他类型传感器的绘制
            elif isinstance(sensors, (list, np.ndarray)):
                # 假设是激光雷达数据
                self._draw_lidar(x, y, angle, sensors)
        except Exception as e:
            logger.error(f"绘制传感器失败: {str(e)}")

    def _draw_lidar(self, x, y, angle, distances):
        """绘制激光雷达

        参数:
            x: 智能体x坐标
            y: 智能体y坐标
            angle: 智能体朝向角度
            distances: 距离数组
        """
        try:
            if not isinstance(distances, (list, np.ndarray)):
                return

            num_rays = len(distances)
            for i, distance in enumerate(distances):
                # 计算射线角度
                ray_angle = angle + (i / num_rays) * 2 * np.pi - np.pi
                # 计算射线终点
                end_x = x + np.cos(ray_angle) * distance
                end_y = y + np.sin(ray_angle) * distance
                # 绘制射线
                pygame.draw.line(self.screen, self.sensor_color,
                                (int(x), int(y)), (int(end_x), int(end_y)), 1)
        except Exception as e:
            logger.error(f"绘制激光雷达失败: {str(e)}")

    def _draw_info(self, info):
        """绘制信息

        参数:
            info: 信息字典
        """
        try:
            if not self.font:
                return

            y_offset = 10
            for key, value in info.items():
                if isinstance(value, (int, float)):
                    # 格式化数值
                    text = f"{key}: {value:.2f}" if isinstance(value, float) else f"{key}: {value}"
                else:
                    text = f"{key}: {value}"

                # 渲染文本
                text_surface = self.font.render(text, True, (0, 0, 0))
                self.screen.blit(text_surface, (10, y_offset))
                y_offset += 20
        except Exception as e:
            logger.error(f"绘制信息失败: {str(e)}")