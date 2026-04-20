"""
第三阶段驾驶环境模块
用于TD3训练管道的Gymnasium兼容环境

实现了一个基于物理的2D驾驶模拟环境，包含：
- 自主控制的小车
- 静态和动态障碍物
- 可移动的目标点
- 多模态观测（雷达、向量状态、视觉）
"""

from copy import deepcopy
import math
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .geometry import check_car_obstacle_collision, get_car_corners, ray_circle_intersection
from ..config import CURRICULUM_CONFIG, ENV_CONFIG, OBSTACLE_CONFIG, RENDER_CONFIG, REWARD_CONFIG, TARGET_CONFIG
from ..visualization import Visualizer


class CarEnvironment(gym.Env):
    """
    自主驾驶模拟环境

    环境特点：
    - 使用自行车运动学模型模拟小车运动
    - 支持多模态观测：雷达传感器、向量状态、视觉输入
    - 包含静态和动态障碍物
    - 支持课程学习（难度渐进增加）
    - 可选移动目标
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, config: Optional[Dict] = None):
        """
        初始化驾驶环境

        参数:
            config: 自定义配置字典，会覆盖默认配置
        """
        super().__init__()

        # 合并配置：默认配置 + 用户自定义配置
        merged_config = deepcopy(ENV_CONFIG)
        if config:
            merged_config.update({key: value for key, value in config.items() if key != "render_config"})
        render_config = deepcopy(RENDER_CONFIG)
        render_config.update((config or {}).get("render_config", {}))

        # 存储各类配置
        self.config = merged_config
        self.render_config = render_config
        self.reward_config = deepcopy(REWARD_CONFIG)
        self.target_config = deepcopy(TARGET_CONFIG)
        self.obstacle_config = deepcopy(OBSTACLE_CONFIG)
        self.curriculum = deepcopy((config or {}).get("curriculum_config", CURRICULUM_CONFIG))

        # 解析环境参数
        self.width = int(self.config["width"])           # 环境宽度
        self.height = int(self.config["height"])         # 环境高度
        self.car_length = float(self.config["car_length"])   # 小车长度
        self.car_width = float(self.config["car_width"])     # 小车宽度
        self.wheel_base = float(self.config["wheel_base"])   # 轴距（用于自行车运动学模型）
        self.max_speed = float(self.config["max_speed"])     # 最大前进速度
        self.min_speed = float(self.config["min_speed"])     # 最大后退速度（负值）
        self.max_steering = float(self.config["max_steering"])   # 最大转向角
        self.steering_smoothness = float(self.config["steering_smoothness"])  # 转向平滑度
        self.acceleration = float(self.config["acceleration"])    # 加速度
        self.brake_deceleration = float(self.config["brake_deceleration"])  # 刹车减速度
        self.friction = float(self.config["friction"])            # 摩擦系数
        self.radar_rays = int(self.config["radar_rays"])          # 雷达射线数量
        self.radar_length = float(self.config["radar_length"])    # 雷达探测范围
        self.max_episode_steps = int(self.config["max_episode_steps"])  # 每回合最大步数
        self.diagonal = math.hypot(self.width, self.height)       # 对角线长度（用于距离归一化）
        self.observation_size = tuple(self.config.get("observation_size", self.render_config["visual_size"]))
        self.vector_dim = 8  # 向量状态维度

        # 定义动作空间：油门[0,1]和转向[-1,1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # 定义观测空间：雷达、向量状态、视觉输入
        self.observation_space = spaces.Dict(
            {
                "radar": spaces.Box(low=0.0, high=1.0, shape=(self.radar_rays,), dtype=np.float32),
                "vector": spaces.Box(low=-1.0, high=1.0, shape=(self.vector_dim,), dtype=np.float32),
                "visual": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(self.observation_size[1], self.observation_size[0], 3),
                    dtype=np.float32,
                ),
            }
        )

        # 初始化可视化器
        self.visualizer = Visualizer(self.width, self.height, self.render_config)

        # 初始化状态变量
        self.current_stage = 0
        self.current_stage_name = self.curriculum[0]["name"]
        self.car_pos = np.zeros(2, dtype=np.float32)      # 小车位置
        self.car_angle = 0.0                              # 小车朝向角度
        self.car_speed = 0.0                              # 小车速度
        self.car_steering = 0.0                           # 转向角
        self.last_car_pos = np.zeros(2, dtype=np.float32) # 上一步位置
        self.target = np.zeros(2, dtype=np.float32)       # 目标位置
        self.target_velocity = np.zeros(2, dtype=np.float32)  # 目标移动速度
        self.static_obstacles: List[np.ndarray] = []      # 静态障碍物列表
        self.dynamic_obstacles: List[np.ndarray] = []     # 动态障碍物列表
        self.dynamic_velocities: List[np.ndarray] = []    # 动态障碍物速度列表
        self.prev_action = np.zeros(2, dtype=np.float32)  # 上一步动作
        self.step_count = 0                               # 当前步数
        self.total_reward = 0.0                           # 累计奖励
        self.last_distance = 0.0                          # 上一步到目标的距离
        self.min_distance_to_target = 0.0                 # 最小距离记录
        self.steps_without_progress = 0                   # 无进展步数计数

        self.reset()

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict]:
        """
        重置环境，开始新回合

        参数:
            seed: 随机种子，用于结果复现
            options: 可选配置，如课程学习阶段

        返回:
            (observation, info)：初始观测和信息字典
        """
        super().reset(seed=seed)

        # 设置课程学习阶段
        if options and "curriculum_stage" in options:
            self.current_stage = int(options["curriculum_stage"])

        stage_config = self._get_stage_config(self.current_stage)
        self.current_stage_name = stage_config["name"]
        self.visualizer.reset_episode()

        # 重置计数器和状态
        self.step_count = 0
        self.total_reward = 0.0
        self.steps_without_progress = 0
        self.prev_action = np.zeros(2, dtype=np.float32)
        self.target_velocity = np.zeros(2, dtype=np.float32)

        # 初始化小车位置和目标位置
        self.car_pos = self._sample_start_position(stage_config)
        self.target = self._sample_target_position(stage_config, self.car_pos)

        # 设置初始朝向：大致朝向目标
        target_heading = math.atan2(self.target[1] - self.car_pos[1], self.target[0] - self.car_pos[0])
        self.car_angle = self._normalize_angle(target_heading + self.np_random.uniform(-0.20, 0.20))
        self.car_speed = 0.0
        self.car_steering = 0.0
        self.last_car_pos = self.car_pos.copy()

        # 生成障碍物
        excluded = [self.car_pos, self.target]
        self.static_obstacles = self._generate_static_obstacles(stage_config["static_count"], excluded)
        self.dynamic_obstacles, self.dynamic_velocities = self._generate_dynamic_obstacles(
            stage_config["dynamic_count"],
            excluded,
            self.static_obstacles,
        )

        # 记录初始距离
        self.last_distance = self._distance_to_target()
        self.min_distance_to_target = self.last_distance
        self.visualizer.record_position(self.car_pos)

        # 返回初始观测
        observation = self._get_observation()
        return observation, self._build_info(reward=0.0, collision=False, target_reached=False, truncated=False)

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict]:
        """
        执行一步动作，更新环境状态

        参数:
            action: 动作数组 [油门, 转向]，范围[-1, 1]

        返回:
            (observation, reward, terminated, truncated, info)
        """
        # 确保动作在合法范围内
        action = np.asarray(action, dtype=np.float32).clip(-1.0, 1.0)
        self.step_count += 1
        self.last_car_pos = self.car_pos.copy()
        previous_distance = self.last_distance

        # 更新小车状态（物理模拟）
        self._update_car_state(action)

        # 更新动态障碍物位置
        self._update_dynamic_obstacles()

        # 如果当前阶段配置了移动目标，更新目标位置
        if self._get_stage_config(self.current_stage)["moving_target"]:
            self._update_target()

        # 计算当前距离和进展
        current_distance = self._distance_to_target()
        progress = previous_distance - current_distance
        self.last_distance = current_distance
        self.min_distance_to_target = min(self.min_distance_to_target, current_distance)

        # 检测是否长时间无进展（停滞检测）
        if progress > 0.4:
            self.steps_without_progress = 0
        else:
            self.steps_without_progress += 1

        # 检测碰撞和到达目标
        collision = self._check_collision()
        target_reached = current_distance <= self.target_config["radius"] + self.car_length * 0.35
        terminated = collision or target_reached
        truncated = self.step_count >= self.max_episode_steps and not terminated

        # 计算奖励
        radar = self._get_radar_data()
        step_distance = float(np.linalg.norm(self.car_pos - self.last_car_pos))
        reward = self._calculate_reward(
            action=action,
            radar=radar,
            progress=progress,
            current_distance=current_distance,
            step_distance=step_distance,
            collision=collision,
            target_reached=target_reached,
        )

        # 更新状态记录
        self.total_reward += reward
        self.prev_action = action
        self.visualizer.record_position(self.car_pos)
        observation = self._get_observation(radar=radar)

        return (
            observation,
            reward,
            terminated,
            truncated,
            self._build_info(
                reward=reward,
                collision=collision,
                target_reached=target_reached,
                truncated=truncated,
            ),
        )

    def render(self, info: Optional[Dict] = None) -> np.ndarray:
        """
        渲染当前环境状态为图像

        参数:
            info: 可选的信息字典，会显示在信息面板中

        返回:
            RGB图像数组
        """
        radar = self._get_radar_data()
        render_info = self._build_info(reward=0.0, collision=False, target_reached=False, truncated=False)
        if info:
            render_info.update(info)
        return self.visualizer.render_environment(
            car_pos=self.car_pos,
            car_angle=self.car_angle,
            car_length=self.car_length,
            car_width=self.car_width,
            car_speed=self.car_speed,
            car_steering=self.car_steering,
            radar_rays=self.radar_rays,
            radar_length=self.radar_length,
            static_obstacles=self.static_obstacles,
            dynamic_obstacles=self.dynamic_obstacles,
            target_pos=self.target,
            target_radius=self.target_config["radius"],
            radar_data=radar,
            info=render_info,
        )

    def _get_observation(self, radar: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        构造完整观测字典

        观测包含三个部分：
        - radar: 雷达传感器读数（距离信息）
        - vector: 向量状态（距离、角度、速度等）
        - visual: 视觉输入（俯视图渲染）
        """
        radar_data = radar if radar is not None else self._get_radar_data()
        vector_state = self._get_vector_state(radar_data)
        visual = self.visualizer.render_observation(
            car_pos=self.car_pos,
            car_angle=self.car_angle,
            car_length=self.car_length,
            car_width=self.car_width,
            static_obstacles=self.static_obstacles,
            dynamic_obstacles=self.dynamic_obstacles,
            target_pos=self.target,
            target_radius=self.target_config["radius"],
        )
        return {
            "radar": radar_data.astype(np.float32),
            "vector": vector_state.astype(np.float32),
            "visual": visual.astype(np.float32),
        }

    def _get_vector_state(self, radar_data: np.ndarray) -> np.ndarray:
        """
        构造向量状态

        向量状态包含：
        - 归一化距离
        - 航向误差的sin/cos值
        - 归一化速度
        - 归一化转向角
        - 上一步的动作
        - 最小雷达读数

        这种表示方式提供了丰富的导航信息
        """
        distance_norm = np.clip(self.last_distance / self.diagonal, 0.0, 1.0)
        heading = math.atan2(self.target[1] - self.car_pos[1], self.target[0] - self.car_pos[0])
        heading_error = self._normalize_angle(heading - self.car_angle)
        speed_norm = np.clip(self.car_speed / max(self.max_speed, 1e-6), -1.0, 1.0)
        steering_norm = np.clip(self.car_steering / max(self.max_steering, 1e-6), -1.0, 1.0)
        min_radar = float(np.min(radar_data)) if radar_data.size else 1.0

        return np.array(
            [
                distance_norm,          # 到目标的归一化距离
                math.sin(heading_error),# 航向误差的sin值
                math.cos(heading_error),# 航向误差的cos值
                speed_norm,             # 归一化速度
                steering_norm,          # 归一化转向角
                self.prev_action[0],    # 上一步油门
                self.prev_action[1],    # 上一步转向
                min_radar,              # 最小雷达读数（最近障碍物）
            ],
            dtype=np.float32,
        )

    def _get_radar_data(self) -> np.ndarray:
        """
        获取雷达传感器读数

        雷达围绕小车均匀分布，检测到障碍物和边界的距离
        返回归一化的距离值（0-1）
        """
        readings = np.zeros(self.radar_rays, dtype=np.float32)
        for index in range(self.radar_rays):
            # 计算每条射线的角度
            angle = self.car_angle + (2.0 * math.pi * index / self.radar_rays)
            readings[index] = self._cast_ray(angle) / self.radar_length
        return readings

    def _cast_ray(self, angle: float) -> float:
        """
        执行射线投射，检测射线与障碍物和边界的最近交点

        参数:
            angle: 射线方向角度（相对于小车朝向）

        返回:
            到最近交点的距离
        """
        ray_end = self.car_pos + self.radar_length * np.array([math.cos(angle), math.sin(angle)], dtype=np.float32)
        min_distance = self.radar_length

        # 检测与静态障碍物的交点
        for obstacle in self.static_obstacles:
            min_distance = min(min_distance, ray_circle_intersection(self.car_pos, ray_end, obstacle))

        # 检测与动态障碍物的交点
        for obstacle in self.dynamic_obstacles:
            min_distance = min(min_distance, ray_circle_intersection(self.car_pos, ray_end, obstacle))

        # 检测与边界的交点
        cos_value = math.cos(angle)
        sin_value = math.sin(angle)
        if abs(cos_value) > 1e-6:
            for x_target in (0.0, float(self.width)):
                t_value = (x_target - self.car_pos[0]) / cos_value
                if t_value > 0:
                    y_intersection = self.car_pos[1] + t_value * sin_value
                    if 0.0 <= y_intersection <= self.height:
                        min_distance = min(min_distance, t_value)
        if abs(sin_value) > 1e-6:
            for y_target in (0.0, float(self.height)):
                t_value = (y_target - self.car_pos[1]) / sin_value
                if t_value > 0:
                    x_intersection = self.car_pos[0] + t_value * cos_value
                    if 0.0 <= x_intersection <= self.width:
                        min_distance = min(min_distance, t_value)

        return float(np.clip(min_distance, 0.0, self.radar_length))

    def _calculate_reward(
        self,
        action: np.ndarray,
        radar: np.ndarray,
        progress: float,
        current_distance: float,
        step_distance: float,
        collision: bool,
        target_reached: bool,
    ) -> float:
        """
        计算奖励函数

        奖励设计策略：
        - 碰撞：大惩罚
        - 到达目标：大奖励 + 时间奖励
        - 进展奖励：靠近目标的正向激励
        - 距离塑造：根据当前位置给予奖励
        - 安全奖励：靠近障碍物的惩罚
        - 平滑性奖励：转向和动作变化的惩罚
        - 停滞惩罚：长时间无进展的惩罚
        """
        # 碰撞惩罚
        if collision:
            return self.reward_config["collision_penalty"]

        # 成功到达目标的奖励
        if target_reached:
            remaining_steps = max(self.max_episode_steps - self.step_count, 0)
            return self.reward_config["target_reached_reward"] + remaining_steps * self.reward_config[
                "success_time_bonus_scale"
            ]

        # 基础奖励组件
        distance_shaping = (1.0 - current_distance / self.diagonal) * self.reward_config["distance_shaping_scale"]
        progress_reward = progress * self.reward_config["progress_reward_scale"]
        step_penalty = self.reward_config["step_penalty"]

        # 安全奖励：靠近障碍物的惩罚
        min_radar = float(np.min(radar)) if radar.size else 1.0
        near_obstacle_penalty = 0.0
        if min_radar < 0.18:
            near_obstacle_penalty = (0.18 - min_radar) * self.reward_config["near_obstacle_penalty_scale"]

        # 平滑性奖励
        steering_penalty = abs(self.car_steering) * self.reward_config["steering_penalty_scale"]
        action_change_penalty = (
            np.linalg.norm(action - self.prev_action) * self.reward_config["action_smoothness_penalty_scale"]
        )

        # 倒车惩罚
        reverse_penalty = max(-self.car_speed, 0.0) * self.reward_config["reverse_penalty_scale"]

        # 停滞惩罚
        stagnation_penalty = 0.0
        if self.steps_without_progress >= self.reward_config["stagnation_steps"] and step_distance < 0.35:
            stagnation_penalty = self.reward_config["stagnation_penalty"]

        return float(
            progress_reward
            + distance_shaping
            + step_penalty
            + near_obstacle_penalty
            + steering_penalty
            + action_change_penalty
            + reverse_penalty
            + stagnation_penalty
        )

    def _update_car_state(self, action: np.ndarray) -> None:
        """
        更新小车状态（物理模拟）

        使用自行车运动学模型：
        - 转向角影响朝向变化
        - 速度影响位置变化
        - 转向变化有平滑限制
        """
        throttle, steering = float(action[0]), float(action[1])

        # 平滑转向更新
        target_steering = steering * self.max_steering
        steering_delta = np.clip(
            target_steering - self.car_steering,
            -self.steering_smoothness,
            self.steering_smoothness,
        )
        self.car_steering += steering_delta

        # 速度更新：油门控制加速/刹车
        if throttle >= 0.0:
            self.car_speed += throttle * self.acceleration
        else:
            self.car_speed += throttle * self.brake_deceleration

        # 摩擦效果
        if abs(throttle) < 0.05:
            # 无输入时摩擦减速
            if abs(self.car_speed) <= self.friction:
                self.car_speed = 0.0
            else:
                self.car_speed -= math.copysign(self.friction, self.car_speed)
        else:
            # 有输入时减小摩擦效果
            self.car_speed -= math.copysign(self.friction * 0.5, self.car_speed)

        # 限制速度范围
        self.car_speed = float(np.clip(self.car_speed, self.min_speed, self.max_speed))

        # 自行车运动学模型：计算转向和位移
        turning_rate = math.tan(self.car_steering) * self.car_speed / max(self.wheel_base, 1e-6)
        self.car_angle = self._normalize_angle(self.car_angle + turning_rate)
        self.car_pos = self.car_pos + self.car_speed * np.array(
            [math.cos(self.car_angle), math.sin(self.car_angle)],
            dtype=np.float32,
        )

    def _update_dynamic_obstacles(self) -> None:
        """
        更新动态障碍物位置

        动态障碍物沿直线移动，碰到边界会反弹
        增加了随机方向变化以增加复杂性
        """
        for obstacle, velocity in zip(self.dynamic_obstacles, self.dynamic_velocities):
            obstacle[:2] += velocity

            # 边界反弹
            if obstacle[0] - obstacle[2] < 0 or obstacle[0] + obstacle[2] > self.width:
                velocity[0] *= -1.0
                obstacle[0] = np.clip(obstacle[0], obstacle[2], self.width - obstacle[2])
            if obstacle[1] - obstacle[2] < 0 or obstacle[1] + obstacle[2] > self.height:
                velocity[1] *= -1.0
                obstacle[1] = np.clip(obstacle[1], obstacle[2], self.height - obstacle[2])

            # 随机方向变化
            if self.np_random.random() < self.obstacle_config["dynamic"]["direction_change_prob"]:
                angle = math.atan2(velocity[1], velocity[0]) + self.np_random.uniform(-0.25, 0.25)
                speed = np.linalg.norm(velocity)
                velocity[:] = np.array([math.cos(angle) * speed, math.sin(angle) * speed], dtype=np.float32)

    def _update_target(self) -> None:
        """
        更新移动目标位置

        目标随机移动，增加任务难度
        使用平滑的速度更新避免突然跳跃
        """
        if self.np_random.random() < self.target_config["move_probability"]:
            direction = self.np_random.uniform(0.0, 2.0 * math.pi)
            desired_velocity = np.array(
                [
                    math.cos(direction) * self.target_config["move_speed"],
                    math.sin(direction) * self.target_config["move_speed"],
                ],
                dtype=np.float32,
            )
            smoothing = self.target_config["smoothing_factor"]
            self.target_velocity = smoothing * self.target_velocity + (1.0 - smoothing) * desired_velocity

        self.target += self.target_velocity
        radius = self.target_config["radius"]

        # 边界约束和反弹
        if self.target[0] - radius < 0:
            self.target[0] = radius
            self.target_velocity[0] = abs(self.target_velocity[0])
        elif self.target[0] + radius > self.width:
            self.target[0] = self.width - radius
            self.target_velocity[0] = -abs(self.target_velocity[0])

        if self.target[1] - radius < 0:
            self.target[1] = radius
            self.target_velocity[1] = abs(self.target_velocity[1])
        elif self.target[1] + radius > self.height:
            self.target[1] = self.height - radius
            self.target_velocity[1] = -abs(self.target_velocity[1])

        # 速度衰减
        self.target_velocity *= 0.985

    def _check_collision(self) -> bool:
        """
        检测碰撞

        检查小车是否与边界或障碍物发生碰撞
        使用精确的角点检测而非简单的圆形近似
        """
        corners = get_car_corners(self.car_pos, self.car_angle, self.car_length, self.car_width)

        # 边界碰撞检测
        if self._check_boundary_collision(corners):
            return True

        # 障碍物碰撞检测
        for obstacle in self.static_obstacles:
            if check_car_obstacle_collision(corners, obstacle):
                return True
        for obstacle in self.dynamic_obstacles:
            if check_car_obstacle_collision(corners, obstacle):
                return True

        return False

    def _check_boundary_collision(self, corners: np.ndarray) -> bool:
        """检查小车角点是否超出边界"""
        return bool(
            np.any(corners[:, 0] < 0)
            or np.any(corners[:, 0] > self.width)
            or np.any(corners[:, 1] < 0)
            or np.any(corners[:, 1] > self.height)
        )

    def _generate_static_obstacles(self, count: int, excluded_points: List[np.ndarray]) -> List[np.ndarray]:
        """
        生成静态障碍物

        参数:
            count: 目标数量
            excluded_points: 需要避开的位置（小车位置、目标位置）

        返回:
            障碍物列表，每个障碍物为[x, y, radius]
        """
        obstacles: List[np.ndarray] = []
        attempts = 0
        while len(obstacles) < count and attempts < count * 80:
            attempts += 1
            radius = self.np_random.uniform(
                self.obstacle_config["static"]["min_radius"],
                self.obstacle_config["static"]["max_radius"],
            )
            point = np.array(
                [
                    self.np_random.uniform(radius + 40.0, self.width - radius - 40.0),
                    self.np_random.uniform(radius + 40.0, self.height - radius - 40.0),
                ],
                dtype=np.float32,
            )
            if self._is_position_occupied(point, radius, obstacles, excluded_points):
                continue
            obstacles.append(np.array([point[0], point[1], radius], dtype=np.float32))
        return obstacles

    def _generate_dynamic_obstacles(
        self,
        count: int,
        excluded_points: List[np.ndarray],
        static_obstacles: List[np.ndarray],
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        生成动态障碍物及其初始速度

        参数:
            count: 目标数量
            excluded_points: 需要避开的位置
            static_obstacles: 已生成的静态障碍物

        返回:
            (障碍物列表, 速度列表)
        """
        obstacles: List[np.ndarray] = []
        velocities: List[np.ndarray] = []
        attempts = 0
        while len(obstacles) < count and attempts < count * 120:
            attempts += 1
            radius = self.np_random.uniform(
                self.obstacle_config["dynamic"]["min_radius"],
                self.obstacle_config["dynamic"]["max_radius"],
            )
            point = np.array(
                [
                    self.np_random.uniform(radius + 40.0, self.width - radius - 40.0),
                    self.np_random.uniform(radius + 40.0, self.height - radius - 40.0),
                ],
                dtype=np.float32,
            )
            if self._is_position_occupied(point, radius, static_obstacles + obstacles, excluded_points):
                continue

            # 生成随机速度
            speed = self.np_random.uniform(
                self.obstacle_config["dynamic"]["min_speed"],
                self.obstacle_config["dynamic"]["max_speed"],
            )
            angle = self.np_random.uniform(0.0, 2.0 * math.pi)
            velocity = np.array([math.cos(angle) * speed, math.sin(angle) * speed], dtype=np.float32)

            obstacles.append(np.array([point[0], point[1], radius], dtype=np.float32))
            velocities.append(velocity)
        return obstacles, velocities

    def _sample_start_position(self, stage_config: Dict) -> np.ndarray:
        """
        采样小车起始位置

        根据课程学习阶段配置，可能在固定位置或随机位置
        """
        base_position = np.array(self.config["start_position"], dtype=np.float32)
        jitter = stage_config.get("spawn_jitter", 0.0)
        if jitter <= 0.0:
            return base_position
        offset = np.array(
            [
                self.np_random.uniform(-jitter, jitter),
                self.np_random.uniform(-jitter, jitter),
            ],
            dtype=np.float32,
        )
        clipped = base_position + offset
        clipped[0] = np.clip(clipped[0], 80.0, self.width * 0.35)
        clipped[1] = np.clip(clipped[1], self.height * 0.55, self.height - 80.0)
        return clipped

    def _sample_target_position(self, stage_config: Dict, start_pos: np.ndarray) -> np.ndarray:
        """
        采样目标位置

        根据配置可能是固定位置或随机位置
        随机目标会增加任务难度
        """
        if stage_config.get("fixed_goal", False):
            return np.array(stage_config["goal_position"], dtype=np.float32)

        margin = self.target_config["goal_margin"]
        min_distance = self.target_config["min_goal_distance"]
        for _ in range(200):
            target = np.array(
                [
                    self.np_random.uniform(self.width * 0.45, self.width - margin),
                    self.np_random.uniform(margin, self.height * 0.55),
                ],
                dtype=np.float32,
            )
            if np.linalg.norm(target - start_pos) >= min_distance:
                return target
        return np.array([self.width - margin, margin], dtype=np.float32)

    def _is_position_occupied(
        self,
        point: np.ndarray,
        radius: float,
        obstacles: List[np.ndarray],
        excluded_points: List[np.ndarray],
    ) -> bool:
        """
        检查某位置是否已被占用

        用于避免障碍物生成时重叠
        """
        safe_margin = self.obstacle_config["static"]["safe_margin"]
        for position in excluded_points:
            if np.linalg.norm(point - position[:2]) < radius + safe_margin + 60.0:
                return True
        for obstacle in obstacles:
            if np.linalg.norm(point - obstacle[:2]) < radius + obstacle[2] + safe_margin:
                return True
        return False

    def _distance_to_target(self) -> float:
        """计算小车到目标的距离"""
        return float(np.linalg.norm(self.car_pos - self.target))

    def _get_stage_config(self, stage_index: int) -> Dict:
        """获取指定课程学习阶段的配置"""
        bounded_index = int(np.clip(stage_index, 0, len(self.curriculum) - 1))
        return self.curriculum[bounded_index]

    def _build_info(self, reward: float, collision: bool, target_reached: bool, truncated: bool) -> Dict:
        """
        构建信息字典

        包含回合的各种统计信息，用于调试和可视化
        """
        return {
            "stage": self.current_stage,
            "stage_name": self.current_stage_name,
            "step": self.step_count,
            "reward": reward,
            "episode_reward": self.total_reward,
            "collision": collision,
            "target_reached": target_reached,
            "truncated": truncated,
            "distance_to_target": self.last_distance,
            "min_distance_to_target": self.min_distance_to_target,
            "car_speed": self.car_speed,
            "car_steering": self.car_steering,
            "steps_without_progress": self.steps_without_progress,
        }

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        """
        将角度归一化到[-π, π]范围

        用于处理角度的连续性问题
        """
        return (angle + math.pi) % (2.0 * math.pi) - math.pi