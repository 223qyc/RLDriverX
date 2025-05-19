import numpy as np
import gymnasium as gym
from gymnasium import spaces
import cv2
from typing import Tuple, List, Dict, Optional
import math
import time
from .geometry import ray_circle_intersection, check_car_obstacle_collision, get_car_corners
from ..config.environment_config import ENV_CONFIG, OBSTACLE_CONFIG, TARGET_CONFIG, REWARD_CONFIG, RENDER_CONFIG
from ..visualization import Visualizer

class CarEnvironment(gym.Env):
    def __init__(self, config: Dict = None):
        super(CarEnvironment, self).__init__()
        
        # 合并配置
        self.config = ENV_CONFIG.copy()
        if config:
            self.config.update(config)
        
        # 环境配置
        self.width = self.config['width']
        self.height = self.config['height']
        self.car_length = self.config['car_length']
        self.car_width = self.config['car_width']
        self.max_speed = self.config['max_speed']
        self.min_speed = self.config['min_speed']
        self.max_steering = self.config['max_steering']
        self.steering_ratio = self.config['steering_ratio']
        self.acceleration = self.config['acceleration']
        self.deceleration = self.config['deceleration']
        self.friction = self.config['friction']
        self.radar_rays = self.config['radar_rays']
        self.radar_length = self.config['radar_length']
        
        # 动作空间：速度（前进/后退）和转向
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32
        )
        
        # 观察空间：雷达数据 + 视觉数据
        visual_shape = RENDER_CONFIG['visual_size']
        self.observation_space = spaces.Dict({
            'radar': spaces.Box(
                low=0,
                high=self.radar_length,
                shape=(self.radar_rays,),
                dtype=np.float32
            ),
            'visual': spaces.Box(
                low=0,
                high=255,
                shape=(visual_shape[1], visual_shape[0], 3),
                dtype=np.uint8
            )
        })
        
        # 创建可视化工具
        self.visualizer = Visualizer(self.width, self.height, RENDER_CONFIG)
        
        # 状态变量
        self.car_pos = None
        self.car_angle = None
        self.car_speed = None
        self.car_steering = None
        self.last_car_pos = None
        self.last_distance = None
        self.target = None
        self.target_velocity = np.zeros(2)
        self.static_obstacles = None
        self.dynamic_obstacles = None
        self.dynamic_velocities = None
        self.collision_count = 0
        self.steps_without_progress = 0
        self.last_rotation_direction = 0
        self.rotation_count = 0
        
        # 新增：位移跟踪变量
        self.position_history = []
        self.position_history_max_len = 10  # 保存最近10个位置
        self.total_distance_traveled = 0.0  # 总行驶距离
        self.net_displacement = 0.0         # 净位移
        
        # 初始化环境状态
        self.reset()
        
    def reset(self, seed: Optional[int] = None) -> Tuple[Dict, Dict]:
        super().reset(seed=seed)
        
        # 初始化小车位置和朝向
        self.car_pos = np.array(self.config['start_position'], dtype=np.float32)
        self.car_angle = self.config['start_angle']
        self.car_speed = 0.0
        self.car_steering = 0.0
        self.last_car_pos = self.car_pos.copy()
        
        # 初始化障碍物
        self.static_obstacles = self._generate_static_obstacles()
        self.dynamic_obstacles, self.dynamic_velocities = self._generate_dynamic_obstacles()
        
        # 初始化目标
        self.target = np.array(TARGET_CONFIG['start_position'], dtype=np.float32)
        self.target_velocity = np.zeros(2)
        
        # 重置状态变量
        self.last_distance = self._get_distance_to_target()
        self.collision_count = 0
        self.steps_without_progress = 0
        self.last_rotation_direction = 0
        self.rotation_count = 0
        
        # 新增：记录位置历史以计算真实位移
        self.position_history = [self.car_pos.copy()]
        self.position_history_max_len = 10  # 保存最近10个位置
        self.total_distance_traveled = 0.0  # 总行驶距离
        self.net_displacement = 0.0         # 净位移
        
        # 获取初始观察
        observation = self._get_observation()
        info = {
            'distance_to_target': self.last_distance,
            'car_speed': self.car_speed,
            'car_steering': self.car_steering
        }
        
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, bool, Dict]:
        # 保存上一帧状态
        self.last_car_pos = self.car_pos.copy()
        last_distance = self.last_distance
        last_car_speed = self.car_speed
        last_car_angle = self.car_angle
        
        # 解析动作
        speed_action, steering_action = action
        
        # 更新小车状态
        self._update_car_state(speed_action, steering_action)
        
        # 更新动态障碍物
        self._update_dynamic_obstacles()
        
        # 更新目标位置 - 平滑移动
        self._update_target()
        
        # 当前距离目标的距离
        current_distance = self._get_distance_to_target()
        self.last_distance = current_distance
        
        # 检查是否有进展（是否靠近目标）
        distance_progress = last_distance - current_distance
        if distance_progress <= 0.5:  # 如果没有明显靠近目标
            self.steps_without_progress += 1
        else:
            self.steps_without_progress = 0
        
        # 检测旋转行为 - 使用新的旋转阈值参数
        rotation_direction = 1 if self.car_angle > last_car_angle else -1 if self.car_angle < last_car_angle else 0
        rotation_threshold = REWARD_CONFIG.get('rotation_threshold', 10)
        if rotation_direction != 0 and rotation_direction == self.last_rotation_direction and abs(self.car_speed) < 0.5:
            self.rotation_count += 1
        else:
            self.rotation_count = 0
        self.last_rotation_direction = rotation_direction
        
        # 计算真实位移 - 更新位置历史
        self.position_history.append(self.car_pos.copy())
        if len(self.position_history) > self.position_history_max_len:
            self.position_history.pop(0)
        
        # 计算单步位移和总行驶距离
        step_distance = np.linalg.norm(self.car_pos - self.last_car_pos)
        self.total_distance_traveled += step_distance
        
        # 计算一段时间内的净位移（起点到当前点的直线距离）
        if len(self.position_history) >= 2:
            self.net_displacement = np.linalg.norm(
                self.position_history[-1] - self.position_history[0]
            )
        
        # 检查碰撞
        collision = self._check_collision()
        if collision:
            self.collision_count += 1
        
        # 计算奖励
        reward = self._calculate_reward(collision, distance_progress, current_distance, step_distance)
        
        # 检查是否到达目标或发生碰撞
        target_reached = self._check_target_reached()
        done = target_reached or collision
        
        # 获取新的观察
        observation = self._get_observation()
        info = {
            'collision': collision,
            'distance_to_target': current_distance,
            'car_speed': self.car_speed,
            'car_steering': self.car_steering,
            'rotation_penalty': True if self.rotation_count > rotation_threshold else False,
            'collision_count': self.collision_count,
            'target_reached': target_reached,
            'step_distance': step_distance,
            'net_displacement': self.net_displacement,
            'total_distance': self.total_distance_traveled
        }
        
        return observation, reward, done, False, info
    
    def _get_observation(self) -> Dict:
        # 获取雷达数据
        radar_data = self._get_radar_data()
        
        # 获取视觉数据
        visual_data = self._get_visual_data()
        
        return {
            'radar': radar_data,
            'visual': visual_data
        }
    
    def _get_radar_data(self) -> np.ndarray:
        radar_data = np.zeros(self.radar_rays)
        for i in range(self.radar_rays):
            angle = self.car_angle + (2 * np.pi * i / self.radar_rays)
            radar_data[i] = self._cast_ray(angle)
        return radar_data
    
    def _get_visual_data(self) -> np.ndarray:
        """获取视觉数据"""
        try:
            # 使用可视化器渲染环境
            frame = self.visualizer.render_environment(
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
                target_radius=TARGET_CONFIG['radius'],
                radar_data=self._get_radar_data()
            )
            
            # 检查渲染帧是否有效
            if frame is None or frame.size == 0:
                # 如果渲染帧无效，创建一个空白帧
                visual_shape = RENDER_CONFIG['visual_size']
                return np.ones((visual_shape[1], visual_shape[0], 3), dtype=np.uint8) * 220
            
            # 检查形状和类型
            if len(frame.shape) != 3 or frame.shape[2] != 3:
                # 如果不是3通道图像，转换为3通道
                if len(frame.shape) == 2:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                else:
                    visual_shape = RENDER_CONFIG['visual_size']
                    return np.ones((visual_shape[1], visual_shape[0], 3), dtype=np.uint8) * 220
            
            # 确保数据类型为uint8
            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)
            
            # 调整大小
            try:
                # 检查目标尺寸是否有效
                visual_shape = RENDER_CONFIG['visual_size']
                if not (isinstance(visual_shape, tuple) and len(visual_shape) == 2 and all(isinstance(x, int) and x > 0 for x in visual_shape)):
                    # 使用默认尺寸
                    visual_shape = (256, 256)
                
                # 安全地进行尺寸调整
                return cv2.resize(frame, visual_shape, interpolation=cv2.INTER_AREA)
            except Exception as e:
                print(f"调整视觉数据大小时出错: {e}")
                # 返回原始帧
                return frame
        
        except Exception as e:
            print(f"获取视觉数据时出错: {e}")
            # 返回一个默认的空白帧
            visual_shape = RENDER_CONFIG['visual_size']
            return np.ones((visual_shape[1], visual_shape[0], 3), dtype=np.uint8) * 220
    
    def render(self, mode='human'):
        # 渲染完整环境（不调整大小）
        frame = self.visualizer.render_environment(
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
            target_radius=TARGET_CONFIG['radius'],
            radar_data=self._get_radar_data()
        )
        
        return frame
    
    def _cast_ray(self, angle: float) -> float:
        # 射线检测实现
        ray_end = self.car_pos + self.radar_length * np.array([np.cos(angle), np.sin(angle)])
        
        # 检查与障碍物的碰撞
        min_distance = self.radar_length
        
        # 检查静态障碍物
        for obs in self.static_obstacles:
            distance = ray_circle_intersection(self.car_pos, ray_end, obs)
            if distance is not None and distance < min_distance:
                min_distance = distance
        
        # 检查动态障碍物
        for obs in self.dynamic_obstacles:
            distance = ray_circle_intersection(self.car_pos, ray_end, obs)
            if distance is not None and distance < min_distance:
                min_distance = distance
        
        # 检查边界碰撞
        # 上边界
        y_intersect = 0
        if abs(np.sin(angle)) > 1e-6:  # 避免除以零
            t = (y_intersect - self.car_pos[1]) / np.sin(angle)
            if t > 0:
                x_intersect = self.car_pos[0] + t * np.cos(angle)
                if 0 <= x_intersect <= self.width:
                    distance = t
                    if distance < min_distance:
                        min_distance = distance
        
        # 下边界
        y_intersect = self.height
        if abs(np.sin(angle)) > 1e-6:
            t = (y_intersect - self.car_pos[1]) / np.sin(angle)
            if t > 0:
                x_intersect = self.car_pos[0] + t * np.cos(angle)
                if 0 <= x_intersect <= self.width:
                    distance = t
                    if distance < min_distance:
                        min_distance = distance
        
        # 左边界
        x_intersect = 0
        if abs(np.cos(angle)) > 1e-6:
            t = (x_intersect - self.car_pos[0]) / np.cos(angle)
            if t > 0:
                y_intersect = self.car_pos[1] + t * np.sin(angle)
                if 0 <= y_intersect <= self.height:
                    distance = t
                    if distance < min_distance:
                        min_distance = distance
        
        # 右边界
        x_intersect = self.width
        if abs(np.cos(angle)) > 1e-6:
            t = (x_intersect - self.car_pos[0]) / np.cos(angle)
            if t > 0:
                y_intersect = self.car_pos[1] + t * np.sin(angle)
                if 0 <= y_intersect <= self.height:
                    distance = t
                    if distance < min_distance:
                        min_distance = distance
        
        return min_distance
    
    def _calculate_reward(self, collision: bool, distance_progress: float, current_distance: float, step_distance: float = 0) -> float:
        # 碰撞惩罚
        if collision:
            return REWARD_CONFIG['collision_penalty']
        
        # 到达目标奖励
        if self._check_target_reached():
            return REWARD_CONFIG['target_reached_reward']
        
        # 朝目标前进的奖励 - 使用增强的系数
        progress_reward = distance_progress * REWARD_CONFIG['distance_progress_factor']
        
        # 原地打转惩罚 - 使用更低的检测阈值和更高的惩罚系数
        rotation_penalty = 0
        rotation_threshold = REWARD_CONFIG.get('rotation_threshold', 10)
        if self.rotation_count > rotation_threshold:
            rotation_penalty = REWARD_CONFIG['rotation_penalty'] * (self.rotation_count - rotation_threshold)
            # 随着转圈次数增加，惩罚加剧
        
        # 基于当前距离的距离奖励（越近越好）
        distance_factor = 1.0 - min(current_distance / math.sqrt(self.width**2 + self.height**2), 1.0)
        distance_reward = distance_factor * REWARD_CONFIG['distance_factor']
        
        # 速度奖励 - 使用增强的系数
        speed_reward = abs(self.car_speed) * REWARD_CONFIG['speed_factor']
        
        # 位移奖励 - 新增：根据实际位移给予奖励，鼓励大范围移动
        movement_reward = 0
        if hasattr(self, 'position_history') and len(self.position_history) >= 2:
            # 计算位移与行驶距离的比值，鼓励直线移动
            # 如果车辆不动或原地旋转，这个比值会很小
            if self.total_distance_traveled > 0:
                efficiency = min(self.net_displacement / max(self.total_distance_traveled, 1e-6), 1.0)
                movement_reward = REWARD_CONFIG.get('movement_reward', 0) * step_distance * efficiency
        
        # 步数惩罚
        step_penalty = REWARD_CONFIG['step_penalty']
        
        # 无进展惩罚 - 使用增强的惩罚系数
        stagnation_penalty = 0
        if self.steps_without_progress > 20:  # 降低无进展检测阈值
            stagnation_penalty = REWARD_CONFIG['stagnation_penalty'] * (self.steps_without_progress - 20) / 10
            stagnation_penalty = max(stagnation_penalty, REWARD_CONFIG['stagnation_penalty'])  # 限制最大惩罚
        
        # 总奖励
        reward = (
            progress_reward +
            distance_reward +
            speed_reward +
            rotation_penalty +
            step_penalty +
            stagnation_penalty +
            movement_reward  # 新增位移奖励
        )
        
        return reward
    
    def _get_distance_to_target(self) -> float:
        return np.linalg.norm(self.car_pos - self.target)
    
    def _check_target_reached(self) -> bool:
        return self._get_distance_to_target() < TARGET_CONFIG['radius'] + self.car_length / 2
    
    def _check_collision(self) -> bool:
        # 获取小车的四个角落坐标
        car_corners = get_car_corners(self.car_pos, self.car_angle, self.car_length, self.car_width)
        
        # 检查与静态障碍物的碰撞
        for obs in self.static_obstacles:
            if check_car_obstacle_collision(car_corners, obs):
                return True
        
        # 检查与动态障碍物的碰撞
        for obs in self.dynamic_obstacles:
            if check_car_obstacle_collision(car_corners, obs):
                return True
        
        # 检查与边界的碰撞
        if self._check_boundary_collision(car_corners):
            return True
        
        return False
    
    def _check_boundary_collision(self, car_corners: np.ndarray) -> bool:
        # 检查小车是否与边界碰撞
        for corner in car_corners:
            if (corner[0] < 0 or corner[0] > self.width or
                corner[1] < 0 or corner[1] > self.height):
                return True
        return False
    
    def _generate_static_obstacles(self) -> List[np.ndarray]:
        # 生成静态障碍物
        obstacles = []
        config = OBSTACLE_CONFIG['static']
        
        # 避免在起点和终点附近生成障碍物
        start_area = np.array(self.config['start_position'])
        target_area = np.array(TARGET_CONFIG['start_position'])
        exclusion_radius = 150  # 排除区域半径
        
        for _ in range(config['count']):
            valid_position = False
            while not valid_position:
                x = np.random.uniform(100, self.width - 100)
                y = np.random.uniform(100, self.height - 100)
                pos = np.array([x, y])
                
                # 检查是否在排除区域内
                if (np.linalg.norm(pos - start_area) < exclusion_radius or
                    np.linalg.norm(pos - target_area) < exclusion_radius):
                    continue
                
                radius = np.random.uniform(config['min_radius'], config['max_radius'])
                valid_position = not self._is_position_occupied(pos, radius, obstacles)
                
                if valid_position:
                    obstacles.append(np.array([x, y, radius]))
        
        return obstacles
    
    def _generate_dynamic_obstacles(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        # 生成动态障碍物
        obstacles = []
        velocities = []
        config = OBSTACLE_CONFIG['dynamic']
        
        # 避免在起点和终点附近生成障碍物
        start_area = np.array(self.config['start_position'])
        target_area = np.array(TARGET_CONFIG['start_position'])
        exclusion_radius = 150  # 排除区域半径
        
        for _ in range(config['count']):
            valid_position = False
            while not valid_position:
                x = np.random.uniform(100, self.width - 100)
                y = np.random.uniform(100, self.height - 100)
                pos = np.array([x, y])
                
                # 检查是否在排除区域内
                if (np.linalg.norm(pos - start_area) < exclusion_radius or
                    np.linalg.norm(pos - target_area) < exclusion_radius):
                    continue
                
                radius = np.random.uniform(config['min_radius'], config['max_radius'])
                valid_position = not self._is_position_occupied(pos, radius, obstacles + self.static_obstacles)
                
                if valid_position:
                    obstacles.append(np.array([x, y, radius]))
                    
                    # 随机速度向量
                    speed = np.random.uniform(config['min_speed'], config['max_speed'])
                    angle = np.random.uniform(0, 2 * np.pi)
                    vx = speed * np.cos(angle)
                    vy = speed * np.sin(angle)
                    velocities.append(np.array([vx, vy]))
        
        return obstacles, velocities
    
    def _generate_target(self) -> np.ndarray:
        # 生成目标位置 - 使用配置中的初始位置
        return np.array(TARGET_CONFIG['start_position'], dtype=np.float32)
    
    def _is_position_occupied(self, pos: np.ndarray, radius: float, obstacles: List[np.ndarray]) -> bool:
        # 检查位置是否被障碍物占据
        for obs in obstacles:
            dist = np.linalg.norm(pos[:2] - obs[:2])
            if dist < (radius + obs[2]) * 1.2:  # 增加一点间距
                return True
        return False
    
    def _update_car_state(self, speed_action: float, steering_action: float):
        # 更新转向角度 - 使用转向比例
        target_steering = steering_action * self.max_steering
        # 平滑转向变化
        steering_diff = target_steering - self.car_steering
        self.car_steering += np.clip(steering_diff, -self.steering_ratio, self.steering_ratio)
        
        # 更新速度 - 加速度模型
        if speed_action > 0:  # 加速
            self.car_speed += speed_action * self.acceleration
        else:  # 减速/倒车
            self.car_speed += speed_action * self.deceleration
        
        # 应用摩擦力
        if abs(self.car_speed) < self.friction:
            self.car_speed = 0
        elif self.car_speed > 0:
            self.car_speed -= self.friction
        else:
            self.car_speed += self.friction
        
        # 限制速度
        self.car_speed = np.clip(self.car_speed, self.min_speed, self.max_speed)
        
        # 根据速度和转向角更新位置和朝向
        # 更新朝向（转向半径与速度成正比）
        turning_rate = self.car_steering * (abs(self.car_speed) / 5.0)
        self.car_angle += turning_rate
        
        # 标准化角度
        self.car_angle = self.car_angle % (2 * np.pi)
        
        # 更新位置
        self.car_pos[0] += self.car_speed * np.cos(self.car_angle)
        self.car_pos[1] += self.car_speed * np.sin(self.car_angle)
    
    def _update_dynamic_obstacles(self):
        # 更新动态障碍物位置
        for i, (obs, vel) in enumerate(zip(self.dynamic_obstacles, self.dynamic_velocities)):
            # 更新位置
            obs[0] += vel[0]
            obs[1] += vel[1]
            
            # 边界碰撞检测和反弹
            if obs[0] - obs[2] < 0 or obs[0] + obs[2] > self.width:
                vel[0] = -vel[0]
            if obs[1] - obs[2] < 0 or obs[1] + obs[2] > self.height:
                vel[1] = -vel[1]
            
            # 随机改变方向
            if np.random.random() < OBSTACLE_CONFIG['dynamic']['direction_change_prob']:
                # 轻微改变方向，而不是完全随机
                angle = np.arctan2(vel[1], vel[0])
                angle += np.random.uniform(-0.5, 0.5)  # 小幅度转向
                speed = np.linalg.norm(vel)
                vel[0] = speed * np.cos(angle)
                vel[1] = speed * np.sin(angle)
    
    def _update_target(self):
        # 目标有概率移动 - 平滑运动
        if np.random.random() < TARGET_CONFIG['move_probability']:
            # 生成目标新位置方向
            angle = np.random.uniform(0, 2 * np.pi)
            max_step = TARGET_CONFIG['max_step_size']
            
            # 计算目标速度
            target_speed = TARGET_CONFIG['move_speed']
            target_vx = target_speed * np.cos(angle)
            target_vy = target_speed * np.sin(angle)
            
            # 平滑速度变化
            smoothing = TARGET_CONFIG['smoothing_factor']
            self.target_velocity = smoothing * self.target_velocity + (1 - smoothing) * np.array([target_vx, target_vy])
        
        # 更新目标位置
        self.target += self.target_velocity
        
        # 确保目标在边界内
        target_radius = TARGET_CONFIG['radius']
        if self.target[0] - target_radius < 0:
            self.target[0] = target_radius
            self.target_velocity[0] = abs(self.target_velocity[0])
        elif self.target[0] + target_radius > self.width:
            self.target[0] = self.width - target_radius
            self.target_velocity[0] = -abs(self.target_velocity[0])
            
        if self.target[1] - target_radius < 0:
            self.target[1] = target_radius
            self.target_velocity[1] = abs(self.target_velocity[1])
        elif self.target[1] + target_radius > self.height:
            self.target[1] = self.height - target_radius
            self.target_velocity[1] = -abs(self.target_velocity[1])
        
        # 逐渐减小速度（摩擦）
        self.target_velocity *= 0.98 