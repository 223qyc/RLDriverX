import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import os
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import time

class Visualizer:
    def __init__(self, width: int, height: int, config: Dict):
        """
        初始化可视化工具
        
        Args:
            width: 环境宽度
            height: 环境高度
            config: 渲染配置
        """
        self.width = width
        self.height = height
        self.config = config
        self.car_trail = []
        self.heatmap_data = np.zeros((height, width), dtype=np.float32)
        self.last_update_time = time.time()
        self.fps_history = []
        
    def render_environment(self, 
                          car_pos: np.ndarray, 
                          car_angle: float,
                          car_length: float,
                          car_width: float,
                          car_speed: float,
                          car_steering: float,
                          radar_rays: int,
                          radar_length: float,
                          static_obstacles: List[np.ndarray],
                          dynamic_obstacles: List[np.ndarray],
                          target_pos: np.ndarray,
                          target_radius: float,
                          radar_data: np.ndarray,
                          episode_reward: float = 0.0,
                          step_count: int = 0,
                          action: Optional[np.ndarray] = None) -> np.ndarray:
        """
        渲染环境
        
        Returns:
            np.ndarray: 渲染的图像
        """
        # 计算FPS
        current_time = time.time()
        dt = current_time - self.last_update_time
        fps = 1.0 / dt if dt > 0 else 0
        self.last_update_time = current_time
        self.fps_history.append(fps)
        if len(self.fps_history) > 30:
            self.fps_history.pop(0)
        avg_fps = sum(self.fps_history) / len(self.fps_history)
        
        # 创建背景
        canvas = np.ones((self.height, self.width, 3), dtype=np.uint8) * self.config['background_color']
        
        # 绘制网格
        if self.config['grid_on']:
            self._draw_grid(canvas)
        
        # 更新并绘制小车轨迹
        if self.config['show_trails']:
            self._update_car_trail(car_pos)
            self._draw_car_trail(canvas)
        
        # 更新热图数据
        self._update_heatmap(car_pos)
        
        # 绘制目标
        # 使目标更加突出，添加辐射效果
        outer_radius = int(target_radius * 1.5)
        middle_radius = int(target_radius * 1.2)
        
        # 绘制目标外层和中层圈
        cv2.circle(canvas, (int(target_pos[0]), int(target_pos[1])), 
                  outer_radius, (0, 100, 0), 1)
        cv2.circle(canvas, (int(target_pos[0]), int(target_pos[1])), 
                  middle_radius, (0, 150, 0), 1)
        
        # 绘制目标圆
        cv2.circle(canvas, (int(target_pos[0]), int(target_pos[1])), 
                  int(target_radius), self.config['target_color'], -1)
        cv2.circle(canvas, (int(target_pos[0]), int(target_pos[1])), 
                  int(target_radius), self.config['target_border_color'], 2)
        
        # 绘制目标中心点
        cv2.circle(canvas, (int(target_pos[0]), int(target_pos[1])), 3, (255, 255, 255), -1)
        
        # 绘制静态障碍物
        for obs in static_obstacles:
            # 绘制障碍物主体
            cv2.circle(canvas, (int(obs[0]), int(obs[1])), int(obs[2]), self.config['static_color'], -1)
            
            # 添加纹理或边框使障碍物更明显
            cv2.circle(canvas, (int(obs[0]), int(obs[1])), int(obs[2]), (50, 50, 50), 2)
            
            # 添加内部圆以增加细节
            inner_radius = int(obs[2] * 0.7)
            if inner_radius > 5:
                cv2.circle(canvas, (int(obs[0]), int(obs[1])), inner_radius, (120, 120, 120), 1)
        
        # 绘制动态障碍物
        for obs in dynamic_obstacles:
            # 绘制动态障碍物主体
            cv2.circle(canvas, (int(obs[0]), int(obs[1])), int(obs[2]), self.config['dynamic_color'], -1)
            
            # 绘制边框
            cv2.circle(canvas, (int(obs[0]), int(obs[1])), int(obs[2]), (50, 0, 0), 2)
            
            # 添加内部图案以区分动态障碍物
            inner_radius = int(obs[2] * 0.6)
            if inner_radius > 3:
                cv2.circle(canvas, (int(obs[0]), int(obs[1])), inner_radius, (200, 100, 100), 1)
                # 添加"运动"标记
                cv2.line(canvas, 
                        (int(obs[0]) - inner_radius//2, int(obs[1])), 
                        (int(obs[0]) + inner_radius//2, int(obs[1])), 
                        (255, 200, 200), 1)
        
        # 绘制雷达线
        self._draw_radar(canvas, car_pos, car_angle, radar_rays, radar_length, radar_data)
        
        # 绘制小车
        self._draw_car(canvas, car_pos, car_angle, car_length, car_width)
        
        # 绘制信息面板
        info_panel = self._create_info_panel(
            car_speed=car_speed, 
            car_steering=car_steering, 
            episode_reward=episode_reward, 
            step_count=step_count,
            action=action,
            fps=avg_fps
        )
        
        # 合并画布和信息面板
        canvas_with_info = np.vstack([canvas, info_panel])
        
        # 绘制边框
        cv2.rectangle(canvas_with_info, (0, 0), (self.width-1, self.height+info_panel.shape[0]-1), (0, 0, 0), 2)
        
        return canvas_with_info
    
    def _draw_grid(self, canvas: np.ndarray):
        """绘制网格"""
        grid_size = self.config['grid_size']
        grid_color = self.config['grid_color']
        
        # 绘制垂直线
        for x in range(0, self.width, grid_size):
            cv2.line(canvas, (x, 0), (x, self.height), grid_color, 1)
        
        # 绘制水平线
        for y in range(0, self.height, grid_size):
            cv2.line(canvas, (0, y), (self.width, y), grid_color, 1)
    
    def _update_car_trail(self, car_pos: np.ndarray):
        """更新小车轨迹"""
        self.car_trail.append((int(car_pos[0]), int(car_pos[1])))
        if len(self.car_trail) > self.config['max_trail_length']:
            self.car_trail.pop(0)
    
    def _draw_car_trail(self, canvas: np.ndarray):
        """绘制小车轨迹"""
        if len(self.car_trail) < 2:
            return
        
        # 使用NumPy操作代替逐个创建对象，减少内存使用
        points = np.array(self.car_trail, dtype=np.int32)
        
        # 减少操作复杂度
        if len(points) > 10:
            # 创建透明叠加层
            overlay = canvas.copy()
            
            # 只保留前50个点以减少内存使用
            visible_points = min(50, len(points))
            for i in range(1, visible_points):
                alpha = i / visible_points  # 透明度随时间变化
                thickness = max(1, int(alpha * 3))
                cv2.line(overlay, tuple(points[i-1]), tuple(points[i]), 
                        self.config['trail_color'][:3], thickness)
            
            # 叠加轨迹
            alpha = self.config['trail_color'][3] / 255.0
            cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, canvas)
    
    def _update_heatmap(self, car_pos: np.ndarray):
        """更新热图数据"""
        x, y = int(car_pos[0]), int(car_pos[1])
        if 0 <= x < self.width and 0 <= y < self.height:
            # 在小车位置周围增加热度
            radius = 10
            for i in range(max(0, x-radius), min(self.width, x+radius+1)):
                for j in range(max(0, y-radius), min(self.height, y+radius+1)):
                    dist = np.sqrt((i-x)**2 + (j-y)**2)
                    if dist <= radius:
                        self.heatmap_data[j, i] += (1 - dist/radius) * 0.1
    
    def _draw_car(self, canvas: np.ndarray, car_pos: np.ndarray, car_angle: float, car_length: float, car_width: float):
        """绘制小车"""
        # 计算小车的四个角点
        corners = self._get_car_corners(car_pos, car_angle, car_length, car_width)
        
        # 绘制小车主体
        cv2.fillPoly(canvas, [corners.astype(np.int32)], self.config['car_color'])
        cv2.polylines(canvas, [corners.astype(np.int32)], True, self.config['car_border_color'], 2)
        
        # 计算车辆前部的中心点
        front_center = (corners[0] + corners[1]) // 2
        
        # 绘制小车方向指示
        cv2.circle(canvas, tuple(front_center.astype(np.int32)), 4, (255, 0, 0), -1)
        
        # 添加车辆细节，如灯光和车窗
        # 前灯
        headlight1 = corners[0] + (corners[1] - corners[0]) * 0.3
        headlight2 = corners[0] + (corners[1] - corners[0]) * 0.7
        cv2.circle(canvas, tuple(headlight1.astype(np.int32)), 2, (255, 255, 0), -1)
        cv2.circle(canvas, tuple(headlight2.astype(np.int32)), 2, (255, 255, 0), -1)
        
        # 车窗（简化为一条线）
        back_center = (corners[2] + corners[3]) // 2
        cv2.line(canvas, 
                tuple(front_center.astype(np.int32)), 
                tuple(back_center.astype(np.int32)), 
                (200, 200, 250), 1)
    
    def _draw_radar(self, canvas: np.ndarray, car_pos: np.ndarray, car_angle: float, radar_rays: int, radar_length: float, radar_data: np.ndarray):
        """绘制雷达线"""
        for i in range(radar_rays):
            angle = car_angle + (2 * np.pi * i / radar_rays)
            ray_length = radar_data[i]
            end_point = car_pos + ray_length * np.array([np.cos(angle), np.sin(angle)])
            
            # 使用不同颜色表示距离远近
            # 距离越近颜色越红，越远颜色越绿
            distance_ratio = ray_length / radar_length
            color = (
                int(255 * (1 - distance_ratio)),  # 蓝色分量
                int(255 * distance_ratio),        # 绿色分量
                50                               # 红色分量固定
            )
            
            # 绘制射线
            cv2.line(canvas, 
                    (int(car_pos[0]), int(car_pos[1])),
                    (int(end_point[0]), int(end_point[1])),
                    color, 1)
            
            # 在射线末端绘制小圆点
            cv2.circle(canvas, (int(end_point[0]), int(end_point[1])), 3, (0, 0, 255), -1)
    
    def _create_info_panel(self, car_speed: float, car_steering: float, episode_reward: float, step_count: int, action: Optional[np.ndarray], fps: float) -> np.ndarray:
        """创建信息面板"""
        panel_height = 150
        panel = np.ones((panel_height, self.width, 3), dtype=np.uint8) * 255
        
        # 绘制速度和转向指示器
        self._draw_speed_indicator(panel, car_speed, 200, panel_height//2, 150, 30)
        self._draw_steering_indicator(panel, car_steering, 500, panel_height//2, 100, 30)
        
        # 添加文本信息
        text_color = self.config['text_color']
        
        # 左侧信息
        cv2.putText(panel, f"Speed: {car_speed:.2f}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
        cv2.putText(panel, f"Steering: {car_steering:.2f}", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
        
        # 中间信息
        cv2.putText(panel, f"Reward: {episode_reward:.2f}", (350, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
        cv2.putText(panel, f"Steps: {step_count}", (350, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
        
        # 右侧信息
        if action is not None:
            cv2.putText(panel, f"Action: [{action[0]:.2f}, {action[1]:.2f}]", (650, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
        
        cv2.putText(panel, f"FPS: {fps:.1f}", (650, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
        
        return panel
    
    def _draw_speed_indicator(self, panel: np.ndarray, speed: float, x: int, y: int, width: int, height: int):
        """绘制速度指示器"""
        # 绘制指示器背景
        cv2.rectangle(panel, (x-width//2, y-height//2), (x+width//2, y+height//2), (220, 220, 220), -1)
        cv2.rectangle(panel, (x-width//2, y-height//2), (x+width//2, y+height//2), (0, 0, 0), 1)
        
        # 计算指针位置
        max_speed = 8.0  # 与环境配置中的最大速度一致
        min_speed = -3.0  # 与环境配置中的最小速度一致
        range_speed = max_speed - min_speed
        
        position = int((speed - min_speed) / range_speed * width) - width//2
        indicator_x = x + position
        
        # 绘制中心线
        cv2.line(panel, (x, y-height//2), (x, y+height//2), (150, 150, 150), 1)
        
        # 绘制指针
        cv2.circle(panel, (indicator_x, y), 8, (0, 0, 255), -1)
        
        # 标记最小和最大值
        cv2.putText(panel, f"{min_speed}", (x-width//2, y+height//2+15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(panel, f"{max_speed}", (x+width//2-20, y+height//2+15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    def _draw_steering_indicator(self, panel: np.ndarray, steering: float, x: int, y: int, width: int, height: int):
        """绘制转向指示器"""
        # 绘制指示器背景
        cv2.rectangle(panel, (x-width//2, y-height//2), (x+width//2, y+height//2), (220, 220, 220), -1)
        cv2.rectangle(panel, (x-width//2, y-height//2), (x+width//2, y+height//2), (0, 0, 0), 1)
        
        # 计算指针位置
        max_steering = 0.8  # 与环境配置中的最大转向角度一致
        position = int((steering + max_steering) / (2 * max_steering) * width) - width//2
        indicator_x = x + position
        
        # 绘制中心线
        cv2.line(panel, (x, y-height//2), (x, y+height//2), (150, 150, 150), 1)
        
        # 绘制指针
        cv2.circle(panel, (indicator_x, y), 8, (0, 0, 255), -1)
        
        # 标记最小和最大值
        cv2.putText(panel, f"-{max_steering}", (x-width//2, y+height//2+15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(panel, f"+{max_steering}", (x+width//2-20, y+height//2+15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    def _get_car_corners(self, car_pos: np.ndarray, car_angle: float, car_length: float, car_width: float) -> np.ndarray:
        """计算小车的四个角点"""
        # 计算偏移向量
        dx = 0.5 * car_length
        dy = 0.5 * car_width
        
        # 定义四个角的偏移量（前右、前左、后左、后右）
        offsets = np.array([
            [dx, -dy],  # 前右
            [dx, dy],   # 前左
            [-dx, dy],  # 后左
            [-dx, -dy]  # 后右
        ])
        
        # 旋转矩阵
        rot_matrix = np.array([
            [np.cos(car_angle), -np.sin(car_angle)],
            [np.sin(car_angle), np.cos(car_angle)]
        ])
        
        # 应用旋转
        rotated_offsets = np.dot(offsets, rot_matrix.T)
        
        # 应用平移
        corners = car_pos + rotated_offsets
        
        return corners
    
    def create_heatmap(self, save_path: str):
        """创建并保存热图"""
        plt.figure(figsize=(10, 8))
        plt.imshow(self.heatmap_data, cmap='hot', interpolation='gaussian')
        plt.colorbar(label='Visit Frequency')
        plt.title('Car Movement Heatmap')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    
    def create_episode_summary(self, 
                              rewards: List[float], 
                              lengths: List[float], 
                              collision_counts: List[int], 
                              target_reached: List[bool],
                              save_dir: str):
        """创建回合总结可视化"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 绘制奖励曲线
        ax1.plot(rewards, 'b-', label='Reward')
        ax1.set_title('Episode Rewards')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.grid(True)
        
        # 绘制回合长度曲线
        ax2.plot(lengths, 'g-', label='Length')
        ax2.set_title('Episode Lengths')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Length')
        ax2.grid(True)
        
        # 绘制碰撞统计
        ax3.bar(range(len(collision_counts)), collision_counts, color='r')
        ax3.set_title('Collision Counts')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Number of Collisions')
        ax3.grid(True)
        
        # 绘制成功率饼图
        success_count = sum(target_reached)
        labels = ['Success', 'Failure']
        sizes = [success_count, len(target_reached) - success_count]
        ax4.pie(sizes, labels=labels, autopct='%1.1f%%', colors=['g', 'r'])
        ax4.set_title('Success Rate')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'episode_summary.png'))
        plt.close()
        
        # 创建摘要信息文件
        with open(os.path.join(save_dir, 'summary.txt'), 'w') as f:
            f.write("Episode Summary\n")
            f.write("==============\n\n")
            f.write(f"Total Episodes: {len(rewards)}\n")
            f.write(f"Average Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}\n")
            f.write(f"Average Length: {np.mean(lengths):.2f} ± {np.std(lengths):.2f}\n")
            f.write(f"Total Collisions: {sum(collision_counts)}\n")
            f.write(f"Success Rate: {success_count/len(target_reached)*100:.1f}%\n") 