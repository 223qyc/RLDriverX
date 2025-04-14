"""多模态可视化模块
提供用于多模态输入的可视化功能，支持视觉、激光雷达和向量数据的实时显示。
与base目录中的环境和模型紧密集成，提供更直观的训练和评估可视化。
"""
import pygame
import numpy as np
import os
import time
import logging
import traceback
import cv2
from typing import Dict, List, Tuple, Union, Optional, Any
from visualization.dynamic import DynamicVisualizer

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger('visualization.multimodal')


class MultiModalVisualizer(DynamicVisualizer):
    """多模态可视化器
    
    用于显示多模态输入的可视化器，支持视觉、激光雷达和向量数据的实时显示。
    与base目录中的环境和模型紧密集成，提供更直观的训练和评估可视化。
    
    属性:
        config: 配置对象
        car_size: 小车尺寸
        car_color: 小车颜色
        target_color: 目标颜色
        obstacle_color: 障碍物颜色
        sensor_color: 传感器线颜色
        background_color: 背景颜色
        font: 字体对象
        show_sensors: 是否显示传感器
        show_info: 是否显示信息面板
        show_visual: 是否显示视觉输入
        show_lidar: 是否显示激光雷达
        show_vector: 是否显示向量数据
        trail_points: 轨迹点列表
        max_trail_length: 最大轨迹长度
        visual_panel_size: 视觉面板大小
        visual_panel_pos: 视觉面板位置
        lidar_panel_size: 激光雷达面板大小
        lidar_panel_pos: 激光雷达面板位置
        vector_panel_size: 向量数据面板大小
        vector_panel_pos: 向量数据面板位置
    """

    def __init__(self, config):
        """初始化多模态可视化器
        
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
        
        # 显示选项
        self.show_sensors = True
        self.show_info = True
        self.show_visual = True
        self.show_lidar = True
        self.show_vector = True
        
        # 轨迹设置
        self.trail_points = []
        self.max_trail_length = 100
        self.trail_enabled = False
        self.trail_color = (200, 200, 200)  # 轨迹颜色
        
        # 多模态面板设置
        self.visual_panel_size = (200, 200)  # 视觉面板大小
        self.visual_panel_pos = (self.width - 220, 20)  # 视觉面板位置
        
        self.lidar_panel_size = (200, 200)  # 激光雷达面板大小
        self.lidar_panel_pos = (self.width - 220, 240)  # 激光雷达面板位置
        
        self.vector_panel_size = (200, 100)  # 向量数据面板大小
        self.vector_panel_pos = (self.width - 220, 460)  # 向量数据面板位置
        
        # 性能监控
        self.fps_history = []
        self.max_fps_history = 100
        self.last_time = time.time()
        self.frame_count = 0

    def initialize(self):
        """初始化可视化环境"""
        super().initialize()
        if self.initialized:
            try:
                self.font = pygame.font.SysFont('Arial', 16)
                pygame.display.set_caption("多模态可视化")
            except Exception as e:
                logger.error(f"多模态可视化器初始化字体失败: {str(e)}")
                # 继续执行，只是没有字体

    def toggle_sensors(self):
        """切换传感器显示状态"""
        self.show_sensors = not self.show_sensors
        return self.show_sensors
    
    def toggle_info(self):
        """切换信息面板显示状态"""
        self.show_info = not self.show_info
        return self.show_info
    
    def toggle_visual(self):
        """切换视觉输入显示状态"""
        self.show_visual = not self.show_visual
        return self.show_visual
    
    def toggle_lidar(self):
        """切换激光雷达显示状态"""
        self.show_lidar = not self.show_lidar
        return self.show_lidar
    
    def toggle_vector(self):
        """切换向量数据显示状态"""
        self.show_vector = not self.show_vector
        return self.show_vector
    
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
                elif event.key == pygame.K_v:
                    self.toggle_visual()
                    logger.info(f"视觉输入 {'开启' if self.show_visual else '关闭'}")
                elif event.key == pygame.K_l:
                    self.toggle_lidar()
                    logger.info(f"激光雷达 {'开启' if self.show_lidar else '关闭'}")
                elif event.key == pygame.K_d:
                    self.toggle_vector()
                    logger.info(f"向量数据 {'开启' if self.show_vector else '关闭'}")
                elif event.key == pygame.K_t:
                    self.toggle_trail()
                    logger.info(f"轨迹显示 {'开启' if self.trail_enabled else '关闭'}")
                    
        return True

    def render(self, state, info=None):
        """渲染当前状态
        
        参数:
            state: 包含位置、角度、目标和障碍物等信息的字典，以及多模态数据
            info: 额外的信息字典
            
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
                logger.warning(f"无效的状态数据类型：{type(state)}")
                pygame.display.flip()
                self.clock.tick(self.render_fps)
                return self.running
                
            # 尝试从state中获取位置和角度信息
            # 如果直接存在position和angle键，则直接使用
            # 否则尝试从vector数据中提取
            car_x, car_y, car_angle = None, None, None
            
            if 'position' in state and 'angle' in state:
                car_x, car_y = state['position']
                car_angle = state['angle']
            elif 'vector' in state and isinstance(state['vector'], np.ndarray) and len(state['vector']) >= 3:
                # 从vector数据中提取位置和角度
                vector_data = state['vector']
                car_x, car_y = vector_data[0], vector_data[1]
                car_angle = vector_data[2]
            else:
                logger.warning("状态缺少必要的键(position或angle)且无法从vector中提取")
                pygame.display.flip()
                self.clock.tick(self.render_fps)
                return self.running
            
            # 更新计数器
            self.frame_count += 1
            current_time = time.time()
            if current_time - self.last_time > 1.0:  # 每秒计算一次FPS
                fps = self.frame_count / (current_time - self.last_time)
                self.fps_history.append(fps)
                if len(self.fps_history) > self.max_fps_history:
                    self.fps_history.pop(0)
                self.last_time = current_time
                self.frame_count = 0
                
            # 更新轨迹
            if self.trail_enabled:
                self.trail_points.append((int(car_x), int(car_y)))
                if len(self.trail_points) > self.max_trail_length:
                    self.trail_points.pop(0)

                # 绘制轨迹
                if len(self.trail_points) > 1:
                    pygame.draw.lines(self.screen, self.trail_color, False, self.trail_points, 2)
                    
            # 绘制车辆主体
            self._draw_car(car_x, car_y, car_angle)
            
            # 绘制目标物体
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
            
            # 绘制传感器数据
            if self.show_sensors and 'sensors' in state:
                self._draw_sensors(car_x, car_y, car_angle, state['sensors'])
            elif self.show_sensors and 'lidar' in state:
                # 如果没有sensors键但有lidar键，使用lidar数据
                self._draw_lidar(car_x, car_y, car_angle, state['lidar'])
                
            # 绘制多模态数据面板
            self._draw_multimodal_data(state)
            
            # 绘制性能信息
            if self.show_info:
                # 创建信息面板
                info_panel = pygame.Surface((200, 80))
                info_panel.fill((240, 240, 240))  # 浅灰色背景
                
                if self.font:
                    # FPS
                    avg_fps = sum(self.fps_history) / max(len(self.fps_history), 1)
                    fps_text = self.font.render(f"FPS: {avg_fps:.1f}", True, (0, 0, 0))
                    info_panel.blit(fps_text, (10, 10))
                    
                    # 位置信息
                    pos_text = self.font.render(f"位置: ({car_x:.1f}, {car_y:.1f})", True, (0, 0, 0))
                    info_panel.blit(pos_text, (10, 30))
                    
                    # 角度信息
                    angle_text = self.font.render(f"角度: {np.degrees(car_angle):.1f}°", True, (0, 0, 0))
                    info_panel.blit(angle_text, (10, 50))
                    
                # 绘制信息面板
                self.screen.blit(info_panel, (10, 10))
                
            # 绘制额外信息
            if self.show_info and info:
                self._draw_info(info)
                
            # 添加操作说明
            controls_text = "控制: [ESC] 退出 [空格] 暂停 [S] 传感器 [I] 信息 [V] 视觉 [L] 雷达 [D] 向量 [T] 轨迹 [R] 录制"
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
            logger.error(f"多模态渲染过程出错: {str(e)}")
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

    def _draw_multimodal_data(self, state):
        """绘制多模态数据
        
        参数:
            state: 状态字典，包含多模态数据
        """
        try:
            # 绘制视觉输入
            if self.show_visual and 'visual' in state:
                self._draw_visual_panel(state['visual'])
                
            # 绘制激光雷达数据
            if self.show_lidar and 'lidar' in state:
                self._draw_lidar_panel(state['lidar'])
                
            # 绘制向量数据
            if self.show_vector and 'vector' in state:
                self._draw_vector_panel(state['vector'])
        except Exception as e:
            logger.error(f"绘制多模态数据失败: {str(e)}")
            logger.debug(traceback.format_exc())

    def _draw_visual_panel(self, visual_data):
        """绘制视觉输入面板
        
        参数:
            visual_data: 视觉输入数据
        """
        try:
            if visual_data is None:
                return
                
            # 创建面板
            panel = pygame.Surface(self.visual_panel_size)
            panel.fill((240, 240, 240))  # 浅灰色背景
            
            # 绘制标题
            if self.font:
                title = self.font.render("视觉输入", True, (0, 0, 0))
                panel.blit(title, (10, 5))
            
            # 处理视觉数据
            if isinstance(visual_data, np.ndarray):
                # 调整大小
                display_size = (self.visual_panel_size[0] - 20, self.visual_panel_size[1] - 30)
                try:
                    # 确保视觉数据是正确的格式
                    if visual_data.ndim == 3:
                        # 根据shape确定格式：(H, W, C) 或 (C, H, W)
                        if visual_data.shape[0] == 3 and visual_data.shape[2] != 3:
                            # 如果是 (C, H, W) 格式，转换为 (H, W, C)
                            visual_data = np.transpose(visual_data, (1, 2, 0))
                        
                        # 确保值范围是 0-255 的 uint8
                        if visual_data.dtype == np.float32 or visual_data.dtype == np.float64:
                            if np.max(visual_data) <= 1.0:
                                visual_data = (visual_data * 255).astype(np.uint8)
                        
                        # 调整大小
                        resized = cv2.resize(visual_data, (display_size[0], display_size[1]))
                        
                        # 确保是RGB格式(pygame需要)
                        if resized.shape[2] == 3:  # RGB
                            # pygame需要的格式是(width, height, 3)且BGR->RGB
                            pygame_surface = np.transpose(resized[:, :, ::-1], (1, 0, 2))
                            visual_surface = pygame.surfarray.make_surface(pygame_surface)
                        else:  # 如果不是3通道，转换为灰度图
                            # 如果不是3通道，可能是单通道或错误格式
                            if resized.shape[2] == 1:
                                # 单通道灰度图
                                gray = np.repeat(resized, 3, axis=2)  # 转为3通道
                                pygame_surface = np.transpose(gray, (1, 0, 2))
                                visual_surface = pygame.surfarray.make_surface(pygame_surface)
                            else:
                                # 未知格式
                                raise ValueError(f"不支持的通道数: {resized.shape[2]}")
                                
                        panel.blit(visual_surface, (10, 25))
                    else:
                        # 绘制错误信息
                        if self.font:
                            error_text = self.font.render(f"无效的视觉数据维度: {visual_data.ndim}D", True, (255, 0, 0))
                            panel.blit(error_text, (10, 50))
                except Exception as e:
                    logger.error(f"处理视觉数据失败: {str(e)}")
                    if self.font:
                        error_text = self.font.render(f"处理视觉数据失败: {type(e).__name__}", True, (255, 0, 0))
                        panel.blit(error_text, (10, 50))
                        detail_text = self.font.render(str(e)[:20], True, (255, 0, 0))
                        panel.blit(detail_text, (10, 70))
            else:
                # 绘制错误信息
                if self.font:
                    error_text = self.font.render(f"非数组视觉数据: {type(visual_data).__name__}", True, (255, 0, 0))
                    panel.blit(error_text, (10, 50))
            
            # 绘制面板到屏幕上
            self.screen.blit(panel, self.visual_panel_pos)
            
        except Exception as e:
            logger.error(f"绘制视觉面板失败: {str(e)}")
            logger.debug(traceback.format_exc())

    def _draw_lidar_panel(self, lidar_data):
        """绘制激光雷达面板
        
        参数:
            lidar_data: 激光雷达数据
        """
        try:
            if lidar_data is None:
                return
                
            # 创建面板
            panel = pygame.Surface(self.lidar_panel_size)
            panel.fill((240, 240, 240))  # 浅灰色背景
            
            # 绘制标题
            if self.font:
                title = self.font.render("激光雷达", True, (0, 0, 0))
                panel.blit(title, (10, 5))
            
            # 处理雷达数据
            if isinstance(lidar_data, (list, np.ndarray)):
                try:
                    # 转换为numpy数组方便处理
                    if isinstance(lidar_data, list):
                        lidar_data = np.array(lidar_data)
                        
                    # 如果是空数组，显示提示
                    if lidar_data.size == 0:
                        if self.font:
                            error_text = self.font.render("空雷达数据", True, (255, 0, 0))
                            panel.blit(error_text, (10, 50))
                        self.screen.blit(panel, self.lidar_panel_pos)
                        return
                        
                    # 检查数据维度
                    if lidar_data.ndim != 1:
                        # 如果不是一维的距离数组，尝试扁平化或取第一维
                        if lidar_data.ndim > 1:
                            try:
                                lidar_data = np.reshape(lidar_data, -1)  # 扁平化
                            except:
                                lidar_data = lidar_data[0]  # 取第一维
                    
                    # 绘制雷达图
                    # 中心点
                    center_x = self.lidar_panel_size[0] // 2
                    center_y = self.lidar_panel_size[1] // 2
                    
                    # 最大半径
                    max_radius = min(center_x, center_y) - 10
                    
                    # 数据归一化 - 通常雷达返回的是距离值，较大的值表示障碍物较远
                    max_dist = np.max(lidar_data) if np.max(lidar_data) > 0 else 1.0
                    
                    # 计算角度
                    num_points = len(lidar_data)
                    angles = np.linspace(0, 2*np.pi, num_points, endpoint=False)
                    
                    # 绘制网格
                    for r in [0.25, 0.5, 0.75, 1.0]:
                        pygame.draw.circle(panel, (200, 200, 200), (center_x, center_y), 
                                         int(max_radius * r), 1)
                    
                    # 绘制坐标轴
                    pygame.draw.line(panel, (150, 150, 150), 
                                   (center_x, center_y - max_radius), 
                                   (center_x, center_y + max_radius), 1)
                    pygame.draw.line(panel, (150, 150, 150), 
                                   (center_x - max_radius, center_y), 
                                   (center_x + max_radius, center_y), 1)
                    
                    # 绘制数据点
                    points = []
                    for i, (angle, dist) in enumerate(zip(angles, lidar_data)):
                        # 计算点的位置
                        norm_dist = dist / max_dist  # 归一化距离
                        radius = max_radius * (1 - norm_dist)  # 距离越远，圆越小
                        x = center_x + radius * np.cos(angle)
                        y = center_y + radius * np.sin(angle)
                        points.append((x, y))
                        
                        # 每隔几个点绘制距离值
                        if i % (num_points // 8) == 0 and self.font:
                            dist_text = self.font.render(f"{dist:.1f}", True, (100, 100, 100))
                            text_x = center_x + (max_radius + 15) * np.cos(angle) - dist_text.get_width() // 2
                            text_y = center_y + (max_radius + 15) * np.sin(angle) - dist_text.get_height() // 2
                            panel.blit(dist_text, (text_x, text_y))
                    
                    # 绘制连接线
                    if len(points) > 1:
                        points.append(points[0])  # 闭合多边形
                        pygame.draw.polygon(panel, (100, 180, 255, 128), points, 0)  # 填充
                        pygame.draw.lines(panel, (0, 120, 255), True, points, 2)  # 轮廓
                        
                except Exception as e:
                    logger.error(f"处理雷达数据失败: {str(e)}")
                    if self.font:
                        error_text = self.font.render(f"雷达数据处理错误: {type(e).__name__}", True, (255, 0, 0))
                        panel.blit(error_text, (10, 50))
                        detail_text = self.font.render(str(e)[:20], True, (255, 0, 0))
                        panel.blit(detail_text, (10, 70))
            else:
                # 绘制错误信息
                if self.font:
                    error_text = self.font.render(f"非数组雷达数据: {type(lidar_data).__name__}", True, (255, 0, 0))
                    panel.blit(error_text, (10, 50))
            
            # 绘制面板到屏幕上
            self.screen.blit(panel, self.lidar_panel_pos)
            
        except Exception as e:
            logger.error(f"绘制激光雷达面板失败: {str(e)}")
            logger.debug(traceback.format_exc())

    def _draw_vector_panel(self, vector_data):
        """绘制向量数据面板
        
        参数:
            vector_data: 向量数据
        """
        try:
            if vector_data is None:
                return
                
            # 创建面板
            panel = pygame.Surface(self.vector_panel_size)
            panel.fill((240, 240, 240))  # 浅灰色背景
            
            # 绘制标题
            if self.font:
                title = self.font.render("向量数据", True, (0, 0, 0))
                panel.blit(title, (10, 5))
            
            # 处理向量数据
            if isinstance(vector_data, (list, np.ndarray)):
                try:
                    # 将列表转换为numpy数组
                    if isinstance(vector_data, list):
                        vector_data = np.array(vector_data)
                    
                    # 确保是一维数组
                    if vector_data.ndim > 1:
                        # 如果是多维数组，扁平化
                        try:
                            flat_data = vector_data.flatten()
                        except:
                            flat_data = vector_data[0]  # 或取第一个元素
                    else:
                        flat_data = vector_data
                        
                    # 根据CarEnvironment._get_state方法的向量格式标识
                    labels = [
                        "位置X", "位置Y", "角度", "速度", "转向",
                        "目标相对X", "目标相对Y", "目标速度X", "目标速度Y",
                        "静态障碍物距离", "动态障碍物距离", "障碍物相对速度X", "障碍物相对速度Y"
                    ]
                    
                    # 如果向量长度与标签不匹配，进行调整
                    if len(flat_data) < len(labels):
                        # 向量元素少于标签
                        labels = labels[:len(flat_data)]
                    elif len(flat_data) > len(labels):
                        # 向量元素多于标签，为多出的元素创建通用标签
                        for i in range(len(labels), len(flat_data)):
                            labels.append(f"数据{i}")
                    
                    # 绘制向量数据
                    y_offset = 30
                    for i, (value, label) in enumerate(zip(flat_data, labels)):
                        # 每行显示两个数据项
                        col = i % 2
                        row = i // 2
                        x_offset = 10 + col * (self.vector_panel_size[0] // 2)
                        
                        # 格式化数值
                        try:
                            if abs(value) < 0.01 or abs(value) > 1000:
                                value_text = f"{value:.2e}"  # 科学计数法
                            else:
                                value_text = f"{value:.2f}"  # 固定小数点
                        except:
                            value_text = str(value)  # 转换失败时直接使用字符串
                            
                        # 绘制标签和值
                        if self.font:
                            text = self.font.render(f"{label}: {value_text}", True, (0, 0, 0))
                            panel.blit(text, (x_offset, y_offset + row * 20))
                            
                        # 确保不超出面板
                        if y_offset + row * 20 + 20 > self.vector_panel_size[1] - 10:
                            break
                    
                except Exception as e:
                    logger.error(f"处理向量数据失败: {str(e)}")
                    if self.font:
                        error_text = self.font.render(f"向量数据处理错误: {type(e).__name__}", True, (255, 0, 0))
                        panel.blit(error_text, (10, 50))
                        detail_text = self.font.render(str(e)[:20], True, (255, 0, 0))
                        panel.blit(detail_text, (10, 70))
            else:
                # 绘制错误信息
                if self.font:
                    error_text = self.font.render(f"非数组向量数据: {type(vector_data).__name__}", True, (255, 0, 0))
                    panel.blit(error_text, (10, 50))
            
            # 绘制面板到屏幕上
            self.screen.blit(panel, self.vector_panel_pos)
            
        except Exception as e:
            logger.error(f"绘制向量数据面板失败: {str(e)}")
            logger.debug(traceback.format_exc())

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
                
            # 绘制FPS信息
            if self.fps_history:
                avg_fps = sum(self.fps_history) / len(self.fps_history)
                fps_text = f"FPS: {avg_fps:.1f}"
                fps_surface = self.font.render(fps_text, True, (0, 0, 0))
                self.screen.blit(fps_surface, (10, y_offset))
        except Exception as e:
            logger.error(f"绘制信息失败: {str(e)}")
            
    def update(self, state_dict, action, reward, next_state_dict, done):
        """更新可视化，用于与训练器集成
        
        参数:
            state_dict: 当前状态字典
            action: 采取的动作
            reward: 获得的奖励
            next_state_dict: 下一状态字典
            done: 是否终止
        """
        try:
            # 构建信息字典
            info = {
                "动作": action,
                "奖励": reward,
                "完成": "是" if done else "否"
            }
            
            # 渲染当前状态
            self.render(state_dict, info)
            
            return self.running
        except Exception as e:
            logger.error(f"更新可视化失败: {str(e)}")
            logger.debug(traceback.format_exc())
            return self.running