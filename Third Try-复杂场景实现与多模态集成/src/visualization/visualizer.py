"""
渲染辅助模块
用于生成观测图像和人可读的视频可视化

提供两种渲染模式：
1. render_observation: 生成干净的俯视图，用于策略网络的视觉输入
2. render_environment: 生成带信息面板的渲染图，用于视频和调试

还提供轨迹热力图和回合总结等分析可视化
"""

import os
import tempfile
from typing import Dict, Iterable, List, Optional, Tuple

# 设置matplotlib临时目录
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "rl_driverx_mpl"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(tempfile.gettempdir(), "rl_driverx_cache"))

import cv2
import matplotlib.pyplot as plt
import numpy as np


class Visualizer:
    """
    可视化器类

    负责环境状态的渲染和可视化分析

    主要功能：
    - 渲染观测图像（用于智能体输入）
    - 渲染带信息的可视化图像（用于视频）
    - 记录和可视化轨迹热力图
    - 生成回合总结图表
    """

    def __init__(self, width: int, height: int, config: Dict):
        """
        初始化可视化器

        参数:
            width: 环境宽度
            height: 环境高度
            config: 渲染配置字典
        """
        self.width = width
        self.height = height
        self.config = config
        self.trail: List[Tuple[int, int]] = []  # 小车轨迹记录
        self.heatmap = np.zeros((height, width), dtype=np.float32)  # 访问热力图

    def reset_episode(self) -> None:
        """开始新回合时清空轨迹"""
        self.trail = []

    def record_position(self, car_pos: np.ndarray) -> None:
        """
        记录小车位置，用于轨迹可视化和热力图

        参数:
            car_pos: 小车位置坐标
        """
        point = (int(car_pos[0]), int(car_pos[1]))
        self.trail.append(point)

        # 保持轨迹长度限制
        if len(self.trail) > self.config["max_trail_length"]:
            self.trail.pop(0)

        # 更新热力图
        self._update_heatmap(point)

    def render_observation(
        self,
        car_pos: np.ndarray,
        car_angle: float,
        car_length: float,
        car_width: float,
        static_obstacles: Iterable[np.ndarray],
        dynamic_obstacles: Iterable[np.ndarray],
        target_pos: np.ndarray,
        target_radius: float,
    ) -> np.ndarray:
        """
        渲染用于策略网络视觉输入的干净图像

        该图像不包含雷达、轨迹、信息面板等额外元素，
        仅展示核心环境状态，供CNN编码器处理

        参数:
            car_pos, car_angle: 小车位置和朝向
            car_length, car_width: 小车尺寸
            static_obstacles, dynamic_obstacles: 障碍物列表
            target_pos, target_radius: 目标位置和半径

        返回:
            归一化的图像数组（0-1范围），尺寸由visual_size配置决定
        """
        canvas = self._new_canvas()
        self._draw_scene(
            canvas=canvas,
            car_pos=car_pos,
            car_angle=car_angle,
            car_length=car_length,
            car_width=car_width,
            static_obstacles=static_obstacles,
            dynamic_obstacles=dynamic_obstacles,
            target_pos=target_pos,
            target_radius=target_radius,
            radar_data=None,
            radar_rays=0,
            radar_length=0.0,
            draw_trail=False,  # 不绘制轨迹
        )

        # 调整到视觉输入尺寸并归一化
        resized = cv2.resize(canvas, self.config["visual_size"], interpolation=cv2.INTER_AREA)
        return resized.astype(np.float32) / 255.0

    def render_environment(
        self,
        car_pos: np.ndarray,
        car_angle: float,
        car_length: float,
        car_width: float,
        car_speed: float,
        car_steering: float,
        radar_rays: int,
        radar_length: float,
        static_obstacles: Iterable[np.ndarray],
        dynamic_obstacles: Iterable[np.ndarray],
        target_pos: np.ndarray,
        target_radius: float,
        radar_data: np.ndarray,
        info: Optional[Dict] = None,
    ) -> np.ndarray:
        """
        渲染带完整信息的可视化图像

        包含雷达射线、轨迹、信息面板等元素，
        用于生成训练/评估视频或调试观察

        参数:
            car_pos, car_angle, car_speed, car_steering: 小车状态
            radar_rays, radar_length, radar_data: 雷达参数和数据
            其他参数同render_observation

        返回:
            RGB图像数组，底部附有信息面板
        """
        canvas = self._new_canvas()
        self._draw_scene(
            canvas=canvas,
            car_pos=car_pos,
            car_angle=car_angle,
            car_length=car_length,
            car_width=car_width,
            static_obstacles=static_obstacles,
            dynamic_obstacles=dynamic_obstacles,
            target_pos=target_pos,
            target_radius=target_radius,
            radar_data=radar_data,
            radar_rays=radar_rays,
            radar_length=radar_length,
            draw_trail=self.config["show_trail"],
        )

        # 添加信息面板
        panel = self._build_info_panel(car_speed, car_steering, info or {})
        return np.vstack([canvas, panel])

    def create_heatmap(self, save_path: str) -> None:
        """
        创建轨迹热力图

        显示小车在训练过程中的访问分布，
        可用于分析策略的行为模式

        参数:
            save_path: 保存路径
        """
        plt.figure(figsize=(10, 7))
        plt.imshow(self.heatmap, cmap="magma", interpolation="bilinear")
        plt.colorbar(label="Visit density")
        plt.title("Trajectory Heatmap")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()

    def create_episode_summary(
        self,
        rewards: List[float],
        lengths: List[float],
        collision_counts: List[int],
        target_reached: List[bool],
        save_dir: str,
    ) -> None:
        """
        创建回合总结可视化图表

        包含四个子图：奖励曲线、长度曲线、碰撞柱状图、成功率饼图

        参数:
            rewards: 奖励列表
            lengths: 长度列表
            collision_counts: 碰撞次数列表
            target_reached: 成功标志列表
            save_dir: 保存目录
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        # 奖励曲线
        axes[0].plot(rewards, color="#2f65e0")
        axes[0].set_title("Episode Reward")
        axes[0].set_xlabel("Episode")
        axes[0].set_ylabel("Reward")
        axes[0].grid(True, alpha=0.3)

        # 长度曲线
        axes[1].plot(lengths, color="#31aa70")
        axes[1].set_title("Episode Length")
        axes[1].set_xlabel("Episode")
        axes[1].set_ylabel("Steps")
        axes[1].grid(True, alpha=0.3)

        # 碰撞柱状图
        axes[2].bar(range(len(collision_counts)), collision_counts, color="#c1604d")
        axes[2].set_title("Collisions")
        axes[2].set_xlabel("Episode")
        axes[2].set_ylabel("Count")

        # 成功/失败饼图
        success_count = sum(target_reached)
        axes[3].pie(
            [success_count, max(len(target_reached) - success_count, 0)],
            labels=["Success", "Failure"],
            autopct="%1.1f%%",
            colors=["#31aa70", "#c1604d"],
        )
        axes[3].set_title("Success Rate")

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "episode_summary.png"), dpi=150)
        plt.close()

    def _new_canvas(self) -> np.ndarray:
        """
        创建新的画布

        根据配置设置背景色和网格
        """
        canvas = np.full((self.height, self.width, 3), self.config["background_color"], dtype=np.uint8)

        # 绘制网格线（如果启用）
        if self.config["show_grid"]:
            grid_size = self.config["grid_size"]
            for x in range(0, self.width, grid_size):
                cv2.line(canvas, (x, 0), (x, self.height), self.config["grid_color"], 1)
            for y in range(0, self.height, grid_size):
                cv2.line(canvas, (0, y), (self.width, y), self.config["grid_color"], 1)

        return canvas

    def _draw_scene(
        self,
        canvas: np.ndarray,
        car_pos: np.ndarray,
        car_angle: float,
        car_length: float,
        car_width: float,
        static_obstacles: Iterable[np.ndarray],
        dynamic_obstacles: Iterable[np.ndarray],
        target_pos: np.ndarray,
        target_radius: float,
        radar_data: Optional[np.ndarray],
        radar_rays: int,
        radar_length: float,
        draw_trail: bool,
    ) -> None:
        """
        绘制完整场景

        包括轨迹、障碍物、目标、雷达、小车等所有元素
        """
        # 绘制轨迹（如果启用且有数据）
        if draw_trail and len(self.trail) > 1:
            for index in range(1, len(self.trail)):
                alpha = index / len(self.trail)
                color = tuple(int(channel * alpha) for channel in self.config["trail_color"])
                cv2.line(canvas, self.trail[index - 1], self.trail[index], color, 2)

        # 绘制静态障碍物（深灰色圆形）
        for obstacle in static_obstacles:
            center = (int(obstacle[0]), int(obstacle[1]))
            radius = int(obstacle[2])
            cv2.circle(canvas, center, radius, self.config["static_color"], -1)
            cv2.circle(canvas, center, radius, (58, 69, 79), 2)  # 边框

        # 绘制动态障碍物（橙红色圆形）
        for obstacle in dynamic_obstacles:
            center = (int(obstacle[0]), int(obstacle[1]))
            radius = int(obstacle[2])
            cv2.circle(canvas, center, radius, self.config["dynamic_color"], -1)
            cv2.circle(canvas, center, radius, (120, 45, 35), 2)  # 边框

        # 绘制目标（绿色圆形，带外圈指示）
        cv2.circle(canvas, (int(target_pos[0]), int(target_pos[1])), int(target_radius * 1.5), (200, 230, 208), 2)
        cv2.circle(canvas, (int(target_pos[0]), int(target_pos[1])), int(target_radius), self.config["target_color"], -1)
        cv2.circle(canvas, (int(target_pos[0]), int(target_pos[1])), int(target_radius), self.config["target_border_color"], 2)

        # 绘制雷达射线（如果有数据）
        if radar_data is not None and radar_rays > 0:
            self._draw_radar(canvas, car_pos, car_angle, radar_data, radar_rays, radar_length)

        # 绘制小车
        self._draw_car(canvas, car_pos, car_angle, car_length, car_width)

    def _draw_radar(
        self,
        canvas: np.ndarray,
        car_pos: np.ndarray,
        car_angle: float,
        radar_data: np.ndarray,
        radar_rays: int,
        radar_length: float,
    ) -> None:
        """
        绘制雷达射线

        雷达颜色根据距离渐变：近距离为红色，远距离为绿色
        """
        for index in range(radar_rays):
            angle = car_angle + (2.0 * np.pi * index / radar_rays)
            ray_length = float(radar_data[index]) * radar_length
            end_point = car_pos + ray_length * np.array([np.cos(angle), np.sin(angle)])

            # 根据距离比例计算颜色
            distance_ratio = np.clip(ray_length / max(radar_length, 1e-6), 0.0, 1.0)
            color = tuple(
                int(
                    self.config["radar_near_color"][channel] * (1.0 - distance_ratio)
                    + self.config["radar_far_color"][channel] * distance_ratio
                )
                for channel in range(3)
            )

            cv2.line(
                canvas,
                (int(car_pos[0]), int(car_pos[1])),
                (int(end_point[0]), int(end_point[1])),
                color,
                1,
            )

    def _draw_car(
        self,
        canvas: np.ndarray,
        car_pos: np.ndarray,
        car_angle: float,
        car_length: float,
        car_width: float,
    ) -> None:
        """
        绘制小车

        小车为矩形，带有朝向指示线
        """
        corners = self._car_corners(car_pos, car_angle, car_length, car_width)

        # 绘制小车主体
        cv2.fillPoly(canvas, [corners.astype(np.int32)], self.config["car_color"])
        cv2.polylines(canvas, [corners.astype(np.int32)], True, self.config["car_border_color"], 2)

        # 绘制朝向指示线
        heading_end = car_pos + (car_length * 0.7) * np.array([np.cos(car_angle), np.sin(car_angle)])
        cv2.line(
            canvas,
            (int(car_pos[0]), int(car_pos[1])),
            (int(heading_end[0]), int(heading_end[1])),
            self.config["car_heading_color"],
            2,
        )

    def _build_info_panel(self, car_speed: float, car_steering: float, info: Dict) -> np.ndarray:
        """
        构建底部信息面板

        显示速度、转向、距离、状态等信息

        参数:
            car_speed, car_steering: 小车状态
            info: 环境返回的信息字典

        返回:
            信息面板图像
        """
        panel_height = 120
        panel = np.full((panel_height, self.width, 3), self.config["panel_background"], dtype=np.uint8)
        text_color = self.config["text_color"]

        # 左侧信息
        left_lines = [
            f"Speed: {car_speed:.2f}",
            f"Steering: {car_steering:.2f}",
            f"Distance: {info.get('distance_to_target', 0.0):.2f}",
        ]

        # 中间信息
        middle_lines = [
            f"Stage: {info.get('stage_name', '-')}",
            f"Reward: {info.get('episode_reward', 0.0):.2f}",
            f"Step: {info.get('step', 0)}",
        ]

        # 右侧信息
        right_lines = [
            f"Collision: {'Yes' if info.get('collision', False) else 'No'}",
            f"Success: {'Yes' if info.get('target_reached', False) else 'No'}",
            f"Min Dist: {info.get('min_distance_to_target', 0.0):.2f}",
        ]

        # 绘制各列信息文本
        for line_index, line in enumerate(left_lines):
            cv2.putText(panel, line, (20, 30 + line_index * 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, text_color, 2)
        for line_index, line in enumerate(middle_lines):
            cv2.putText(panel, line, (350, 30 + line_index * 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, text_color, 2)
        for line_index, line in enumerate(right_lines):
            cv2.putText(panel, line, (700, 30 + line_index * 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, text_color, 2)

        return panel

    def _update_heatmap(self, point: Tuple[int, int]) -> None:
        """
        更新热力图数据

        在小车位置周围添加访问密度

        参数:
            point: 小车位置坐标
        """
        x, y = point
        if not (0 <= x < self.width and 0 <= y < self.height):
            return

        radius = 8  # 热力影响半径
        for px in range(max(0, x - radius), min(self.width, x + radius + 1)):
            for py in range(max(0, y - radius), min(self.height, y + radius + 1)):
                distance = np.hypot(px - x, py - y)
                if distance <= radius:
                    # 距离越近，密度增加越多
                    self.heatmap[py, px] += 1.0 - distance / radius

    def _car_corners(
        self, car_pos: np.ndarray, car_angle: float, car_length: float, car_width: float
    ) -> np.ndarray:
        """
        计算小车四个角点的坐标

        参数:
            car_pos: 小车中心位置
            car_angle: 小车朝向角度
            car_length, car_width: 小车尺寸

        返回:
            四个角点的坐标数组
        """
        half_length = car_length / 2.0
        half_width = car_width / 2.0

        # 四个角点的相对位置
        offsets = np.array(
            [
                [half_length, -half_width],
                [half_length, half_width],
                [-half_length, half_width],
                [-half_length, -half_width],
            ]
        )

        # 旋转矩阵
        rotation = np.array(
            [
                [np.cos(car_angle), -np.sin(car_angle)],
                [np.sin(car_angle), np.cos(car_angle)],
            ]
        )

        # 应用旋转和平移
        return car_pos + offsets @ rotation.T