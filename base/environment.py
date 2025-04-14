import cv2
import numpy as np
import pygame
from typing import Tuple, List, Dict, Union, Any
from base.config import BaseConfig


class Car:
    """小车类，负责小车的运动学模型、状态更新和绘制"""

    def __init__(self, init_pos=None):
        """
        初始化小车

        参数:
            init_pos: 初始位置，如果为None则使用默认位置
        """
        self.pos = np.array(init_pos, dtype=np.float64) if init_pos is not None else np.array(
            [BaseConfig.ENV_WIDTH // 4, BaseConfig.ENV_HEIGHT // 4], dtype=np.float64)
        self.angle = 0  # 角度，0表示向右
        self.speed = 0.0  # 当前速度
        self.steering = 0.0  # 当前转向角度

    def reset(self, init_pos=None):
        """重置小车状态"""
        self.pos = np.array(init_pos, dtype=np.float64) if init_pos is not None else np.array(
            [BaseConfig.ENV_WIDTH // 4, BaseConfig.ENV_HEIGHT // 4], dtype=np.float64)
        self.angle = 0
        self.speed = 3.0
        self.steering = 0.0

    def update(self, action: int):
        """
        根据动作更新小车状态

        参数:
            action: 动作ID（0-左转，1-右转，2-加速，3-减速/刹车）
        """
        # 更新小车状态
        if action == 0:  # 左转
            self.steering = min(self.steering + BaseConfig.CAR_STEERING_SPEED, BaseConfig.CAR_MAX_STEERING)
        elif action == 1:  # 右转
            self.steering = max(self.steering - BaseConfig.CAR_STEERING_SPEED, -BaseConfig.CAR_MAX_STEERING)
        elif action == 2:  # 加速
            self.speed = min(self.speed + BaseConfig.CAR_ACCELERATION, BaseConfig.CAR_MAX_SPEED)
        elif action == 3:  # 减速/刹车
            self.speed = max(self.speed - BaseConfig.CAR_DECELERATION, BaseConfig.CAR_MIN_SPEED)

        # 更新车辆角度
        self.angle += self.steering * (self.speed / BaseConfig.CAR_MAX_SPEED)

        # 根据当前角度和速度更新位置
        rad = np.radians(self.angle)
        self.pos += np.array([np.cos(rad), np.sin(rad)]) * self.speed

        # 边界检查
        self.pos = np.clip(self.pos, 0, [BaseConfig.ENV_WIDTH, BaseConfig.ENV_HEIGHT])

    def get_corners(self) -> np.ndarray:
        """获取车辆四个角点的坐标（考虑旋转）"""
        # 车辆中心点
        cx, cy = self.pos
        # 车辆长宽的一半
        half_length = BaseConfig.CAR_LENGTH / 2
        half_width = BaseConfig.CAR_WIDTH / 2

        # 计算旋转角度（弧度）
        angle_rad = np.radians(self.angle)
        cos_angle = np.cos(angle_rad)
        sin_angle = np.sin(angle_rad)

        # 计算四个角点（相对于中心点的偏移，然后旋转）
        corners = []
        for dx, dy in [(half_length, half_width), (half_length, -half_width),
                       (-half_length, -half_width), (-half_length, half_width)]:
            # 旋转偏移量
            rotated_dx = dx * cos_angle - dy * sin_angle
            rotated_dy = dx * sin_angle + dy * cos_angle
            # 计算实际坐标
            corners.append([cx + rotated_dx, cy + rotated_dy])

        return np.array(corners)

    def get_velocity_vector(self) -> np.ndarray:
        """获取车辆当前速度向量 (单位向量)"""
        rad = np.radians(self.angle)
        velocity = np.array([np.cos(rad), np.sin(rad)])
        norm = np.linalg.norm(velocity)
        return velocity / norm if norm > 0 else np.array([1.0, 0.0])  # 避免除以零

    def draw(self, surface):
        """
        绘制小车，用于后续的测试
        """
        light_color = (180, 180, 200)  # 浅灰色
        arrow_color = (80, 80, 255)  # 蓝色

        car_corners = self.get_corners()
        car_corners_int = [(int(x), int(y)) for x, y in car_corners]
        center_x, center_y = int(self.pos[0]), int(self.pos[1])

        # 绘制车身主体
        pygame.draw.polygon(surface, light_color, car_corners_int)

        # 绘制方向箭头 (在车头前方)
        arrow_length = BaseConfig.CAR_LENGTH / 1.5
        arrow_width = 3
        head_length = arrow_length / 3
        head_width = arrow_width * 2

        velocity_vector = self.get_velocity_vector()
        arrow_start_x = center_x + velocity_vector[0] * BaseConfig.CAR_LENGTH / 2.5
        arrow_start_y = center_y + velocity_vector[1] * BaseConfig.CAR_LENGTH / 2.5
        arrow_end_x = center_x + velocity_vector[0] * (BaseConfig.CAR_LENGTH / 2.5 + arrow_length)
        arrow_end_y = center_y + velocity_vector[1] * (BaseConfig.CAR_LENGTH / 2.5 + arrow_length)

        pygame.draw.line(surface, arrow_color, (int(arrow_start_x), int(arrow_start_y)),
                         (int(arrow_end_x), int(arrow_end_y)), arrow_width)

        # 绘制箭头头部
        angle = np.arctan2(velocity_vector[1], velocity_vector[0])
        points = [
            (arrow_end_x - head_length * np.cos(angle - np.pi / 6),
             arrow_end_y - head_length * np.sin(angle - np.pi / 6)),
            (arrow_end_x, arrow_end_y),
            (arrow_end_x - head_length * np.cos(angle + np.pi / 6),
             arrow_end_y - head_length * np.sin(angle + np.pi / 6)),
        ]
        pygame.draw.polygon(surface, arrow_color, [(int(p[0]), int(p[1])) for p in points])

        # 绘制车辆信息文本 (可以根据需要调整位置和样式)
        font = pygame.font.SysFont(None, 20)
        text_color = (0, 0, 0)
        speed_text = font.render(f'Speed: {self.speed:.1f}', True, text_color)
        angle_text = font.render(f'Angle: {self.steering:.1f}', True, text_color)
        surface.blit(speed_text, (10, 10))
        surface.blit(angle_text, (10, 30))

    def draw_cv2(self, image: np.ndarray):
        """在OpenCV图像上绘制小车（用于视觉观测）"""
        # 绘制小车（长方形，考虑旋转）
        car_corners = self.get_corners()
        car_corners_int = car_corners.astype(np.int32)
        cv2.fillPoly(image, [car_corners_int], (255, 0, 0))

        # 绘制车辆前进方向指示线
        front_center = (car_corners[0] + car_corners[1]) / 2
        front_center_int = front_center.astype(int)
        car_center_int = self.pos.astype(int)
        cv2.line(image, tuple(car_center_int), tuple(front_center_int), (255, 255, 255), 2)


class Target:
    """目标类，可以静止或移动"""

    def __init__(self):
        """初始化目标"""
        # 初始位置在屏幕右侧区域随机生成
        self.pos = np.array([
            np.random.randint(BaseConfig.TARGET_RANGE_X[0], BaseConfig.TARGET_RANGE_X[1]),
            np.random.randint(BaseConfig.TARGET_RANGE_Y[0], BaseConfig.TARGET_RANGE_Y[1])
        ], dtype=np.float64)
        self.size = BaseConfig.TARGET_SIZE

        # 如果目标需要移动，初始化速度向量
        if BaseConfig.TARGET_MOVING:
            speed = np.random.uniform(BaseConfig.TARGET_MIN_SPEED, BaseConfig.TARGET_MAX_SPEED)
            angle = np.random.uniform(0, 2 * np.pi)
            self.velocity = np.array([np.cos(angle), np.sin(angle)]) * speed
        else:
            self.velocity = np.array([0.0, 0.0])

    def reset(self):
        """重置目标位置和速度"""
        self.pos = np.array([
            np.random.randint(BaseConfig.TARGET_RANGE_X[0], BaseConfig.TARGET_RANGE_X[1]),
            np.random.randint(BaseConfig.TARGET_RANGE_Y[0], BaseConfig.TARGET_RANGE_Y[1])
        ], dtype=np.float64)

        if BaseConfig.TARGET_MOVING:
            speed = np.random.uniform(BaseConfig.TARGET_MIN_SPEED, BaseConfig.TARGET_MAX_SPEED)
            angle = np.random.uniform(0, 2 * np.pi)
            self.velocity = np.array([np.cos(angle), np.sin(angle)]) * speed
        else:
            self.velocity = np.array([0.0, 0.0])

    def update(self):
        """更新目标位置"""
        if BaseConfig.TARGET_MOVING:
            # 有一定概率随机改变方向
            if np.random.random() < BaseConfig.TARGET_CHANGE_DIR_PROB:
                speed = np.linalg.norm(self.velocity)
                angle = np.random.uniform(0, 2 * np.pi)
                self.velocity = np.array([np.cos(angle), np.sin(angle)]) * speed

            # 更新位置
            self.pos += self.velocity

            # 边界检查和反弹
            for i in range(2):
                if self.pos[i] < 0 or self.pos[i] > (BaseConfig.ENV_WIDTH if i == 0 else BaseConfig.ENV_HEIGHT):
                    self.velocity[i] *= -1  # 反向速度
                    # 确保在边界内
                    self.pos[i] = np.clip(self.pos[i], 0, BaseConfig.ENV_WIDTH if i == 0 else BaseConfig.ENV_HEIGHT)

            # 确保目标不会离开指定范围太远
            if (self.pos[0] < BaseConfig.TARGET_RANGE_X[0] - 100 or
                    self.pos[0] > BaseConfig.TARGET_RANGE_X[1] + 100):
                self.velocity[0] *= -1

            if (self.pos[1] < BaseConfig.TARGET_RANGE_Y[0] - 100 or
                    self.pos[1] > BaseConfig.TARGET_RANGE_Y[1] + 100):
                self.velocity[1] *= -1

    def draw(self, surface):
        """
        绘制，可用于测试
        """
        outer_color = (102, 255, 102)  # 浅绿色
        inner_color = (51, 153, 51)  # 深绿色
        center = (int(self.pos[0]), int(self.pos[1]))
        radius = int(self.size)

        # 绘制外圆
        pygame.draw.circle(surface, outer_color, center, radius)

        # 绘制内圆 (形成圆环)
        inner_radius = int(radius * 0.6)
        pygame.draw.circle(surface, inner_color, center, inner_radius)

        # 如果目标是移动的，绘制一个简洁的箭头
        if BaseConfig.TARGET_MOVING and np.linalg.norm(self.velocity) > 0:
            arrow_start = self.pos
            arrow_end = self.pos + self.velocity * 8
            pygame.draw.line(surface, (0, 100, 0), (int(arrow_start[0]), int(arrow_start[1])),
                             (int(arrow_end[0]), int(arrow_end[1])), 2)

    def draw_cv2(self, image: np.ndarray):
        """在OpenCV图像上绘制目标（用于视觉观测）"""
        cv2.circle(image, tuple(self.pos.astype(int)), self.size, (0, 255, 0), -1)

        # 如果目标是移动的，绘制其速度向量
        if BaseConfig.TARGET_MOVING and np.linalg.norm(self.velocity) > 0:
            end_point = self.pos + self.velocity * 5  # 放大速度向量以便可视化
            cv2.arrowedLine(image,
                            tuple(self.pos.astype(int)),
                            tuple(end_point.astype(int)),
                            (0, 200, 0), 2)


class Obstacle:
    """障碍物基类"""

    def __init__(self, pos=None, size=None):
        """初始化障碍物"""
        self.pos = np.array(pos, dtype=np.float64) if pos is not None else np.random.randint(0, BaseConfig.ENV_WIDTH,size=2).astype(float)
        self.size = size if size is not None else BaseConfig.STATIC_OBSTACLE_SIZE

    def reset(self, pos=None):
        """重置障碍物位置"""
        self.pos = np.array(pos, dtype=np.float64) if pos is not None else np.random.randint(0, BaseConfig.ENV_WIDTH,size=2).astype(float)

    def draw(self, surface):
        """在给定surface上绘制更美观的静态障碍物"""
        outer_color = (139, 69, 19)  # 棕色
        inner_color = (101, 67, 33)  # 深棕色
        center = (int(self.pos[0]), int(self.pos[1]))
        radius = int(self.size)

        # 绘制外圆
        pygame.draw.circle(surface, outer_color, center, radius)

        # 绘制内部小圆点
        pygame.draw.circle(surface, inner_color, center, int(radius * 0.4))

    def draw_cv2(self, image: np.ndarray):
        """在OpenCV图像上绘制障碍物（用于视觉观测）"""
        cv2.circle(image, tuple(self.pos.astype(int)), self.size, (0, 0, 255), -1)


class DynamicObstacle(Obstacle):
    """动态障碍物类，继承自Obstacle"""

    def __init__(self):
        """初始化动态障碍物"""
        # 随机位置
        pos = np.random.randint(0, BaseConfig.ENV_WIDTH, size=2).astype(float)
        # 随机大小
        size = np.random.randint(BaseConfig.DYNAMIC_OBSTACLE_MIN_SIZE, BaseConfig.DYNAMIC_OBSTACLE_MAX_SIZE)
        super().__init__(pos, size)

        # 随机速度向量
        speed = np.random.uniform(BaseConfig.DYNAMIC_OBSTACLE_MIN_SPEED, BaseConfig.DYNAMIC_OBSTACLE_MAX_SPEED)
        angle = np.random.uniform(0, 2 * np.pi)
        self.velocity = np.array([np.cos(angle), np.sin(angle)]) * speed

    def reset(self):
        """重置动态障碍物位置和速度"""
        super().reset()
        self.size = np.random.randint(BaseConfig.DYNAMIC_OBSTACLE_MIN_SIZE, BaseConfig.DYNAMIC_OBSTACLE_MAX_SIZE)

        # 随机速度向量
        speed = np.random.uniform(BaseConfig.DYNAMIC_OBSTACLE_MIN_SPEED, BaseConfig.DYNAMIC_OBSTACLE_MAX_SPEED)
        angle = np.random.uniform(0, 2 * np.pi)
        self.velocity = np.array([np.cos(angle), np.sin(angle)]) * speed

    def update(self):
        """更新动态障碍物位置"""
        # 有一定概率随机改变方向
        if np.random.random() < BaseConfig.DYNAMIC_OBSTACLE_CHANGE_DIR_PROB:
            speed = np.linalg.norm(self.velocity)
            angle = np.random.uniform(0, 2 * np.pi)
            self.velocity = np.array([np.cos(angle), np.sin(angle)]) * speed

        # 更新位置
        self.pos += self.velocity

        # 边界检查和反弹
        for i in range(2):
            if self.pos[i] < 0 or self.pos[i] > (BaseConfig.ENV_WIDTH if i == 0 else BaseConfig.ENV_HEIGHT):
                self.velocity[i] *= -1  # 反向速度
                # 确保在边界内
                self.pos[i] = np.clip(self.pos[i], 0, BaseConfig.ENV_WIDTH if i == 0 else BaseConfig.ENV_HEIGHT)

    def draw(self, surface):
        """
        绘制，可用于测试
        """
        obstacle_color = (255, 165, 0)  # 橙色
        arrow_color = (255, 255, 255)  # 白色
        center = (int(self.pos[0]), int(self.pos[1]))
        radius = int(self.size)

        # 绘制障碍物主体
        pygame.draw.circle(surface, obstacle_color, center, radius)

        # 绘制一个小的方向箭头
        if np.linalg.norm(self.velocity) > 0:
            velocity_vector_normalized = self.velocity / np.linalg.norm(self.velocity) if np.linalg.norm(
                self.velocity) > 0 else np.array([1.0, 0.0])
            arrow_length = radius * 1.2
            arrow_start_x = center[0] + velocity_vector_normalized[0] * radius * 0.3
            arrow_start_y = center[1] + velocity_vector_normalized[1] * radius * 0.3
            arrow_end_x = center[0] + velocity_vector_normalized[0] * arrow_length
            arrow_end_y = center[1] + velocity_vector_normalized[1] * arrow_length
            pygame.draw.line(surface, arrow_color, (int(arrow_start_x), int(arrow_start_y)),
                             (int(arrow_end_x), int(arrow_end_y)), 2)
            # 绘制箭头头部 (简化)
            head_size = 3
            pygame.draw.polygon(surface, arrow_color, [
                (int(arrow_end_x), int(arrow_end_y)),
                (int(arrow_end_x - velocity_vector_normalized[0] * head_size + velocity_vector_normalized[
                    1] * head_size),
                 int(arrow_end_y - velocity_vector_normalized[1] * head_size - velocity_vector_normalized[
                     0] * head_size)),
                (int(arrow_end_x - velocity_vector_normalized[0] * head_size - velocity_vector_normalized[
                    1] * head_size),
                 int(arrow_end_y - velocity_vector_normalized[1] * head_size + velocity_vector_normalized[
                     0] * head_size)),
            ])

    def draw_cv2(self, image: np.ndarray):
        """在OpenCV图像上绘制动态障碍物（用于视觉观测）"""
        cv2.circle(image, tuple(self.pos.astype(int)), self.size, (255, 0, 255), -1)

        # 绘制速度向量
        if np.linalg.norm(self.velocity) > 0:
            end_point = self.pos + self.velocity * 5  # 放大速度向量以便可视化
            cv2.arrowedLine(image,
                            tuple(self.pos.astype(int)),
                            tuple(end_point.astype(int)),
                            (255, 255, 0), 2)


class CarEnvironment:
    """小车环境类，管理所有对象和交互"""

    def __init__(self):
        """初始化环境，但不立即初始化Pygame"""
        # Pygame 相关变量延迟初始化
        self.screen = None
        self.clock = None
        self.pygame_initialized = False  # 添加一个标志位

        # 初始化小车
        self.car = Car()

        # 初始化目标
        self.target = Target()

        # 初始化静态障碍物
        self.static_obstacles = self._generate_static_obstacles()

        # 初始化动态障碍物
        self.dynamic_obstacles = self._generate_dynamic_obstacles()
        
    def close(self):
        """关闭环境并释放资源"""
        # 如果Pygame已初始化，则退出Pygame
        if self.pygame_initialized and pygame.get_init():
            pygame.quit()
            self.pygame_initialized = False
            self.screen = None
            self.clock = None

    def _generate_static_obstacles(self) -> List[Obstacle]:
        """生成静态随机障碍物"""
        return [Obstacle() for _ in range(BaseConfig.STATIC_OBSTACLES_COUNT)]

    def _generate_dynamic_obstacles(self) -> List[DynamicObstacle]:
        """生成动态障碍物"""
        return [DynamicObstacle() for _ in range(BaseConfig.DYNAMIC_OBSTACLES_COUNT)]

    def reset(self) -> Dict[str, Union[np.ndarray, float]]:
        """重置环境并返回初始状态"""
        # 重置小车
        self.car.reset()

        # 重置目标
        self.target.reset()

        # 重置障碍物
        self.static_obstacles = self._generate_static_obstacles()
        for dyn_obs in self.dynamic_obstacles:
            dyn_obs.reset()

        return self._get_state()

    def step(self, action: int) -> Tuple[Dict[str, Union[np.ndarray, float]], float, bool, Dict[str, Any]]:
        """
        执行动作并返回(next_state, reward, done, info)

        参数:
            action: 动作ID（0-左转，1-右转，2-加速，3-减速/刹车）

        返回:
            next_state: 下一个状态
            reward: 奖励值
            done: 是否终止
            info: 额外信息
        """
        # 更新小车状态
        self.car.update(action)

        # 更新目标位置
        self.target.update()

        # 更新动态障碍物位置
        for dyn_obs in self.dynamic_obstacles:
            dyn_obs.update()

        # 计算奖励和终止条件
        reward, done = self._calculate_reward()

        # 获取下一个状态
        next_state = self._get_state()

        # 返回额外信息
        info = {
            'car_speed': self.car.speed,
            'car_steering': self.car.steering,
            'target_pos': self.target.pos,
            'target_velocity': self.target.velocity
        }

        return next_state, reward, done, info

    def _calculate_reward(self) -> Tuple[float, bool]:
        """计算奖励和终止条件"""
        # 计算车辆到目标的距离
        distance_to_target = np.linalg.norm(self.car.pos - self.target.pos)

        # 基础奖励：距离目标越近奖励越高
        reward = -distance_to_target / 1000

        # 速度奖励：鼓励车辆保持适当速度
        speed_reward = 0.01 * self.car.speed if self.car.speed > 0 else -0.02 * abs(self.car.speed)
        reward += speed_reward

        # 朝向奖励：如果车辆朝向目标，给予额外奖励
        car_direction = np.array([np.cos(np.radians(self.car.angle)), np.sin(np.radians(self.car.angle))])
        target_direction = (self.target.pos - self.car.pos) / (distance_to_target + 1e-6)  # 防止除零
        direction_alignment = np.dot(car_direction, target_direction)
        direction_reward = 0.02 * max(0, direction_alignment)  # 只在正向时给予奖励
        reward += direction_reward

        # 获取车辆的四个角点坐标
        car_corners = self.car.get_corners()

        # 检查碰撞：静态障碍物
        for obs in self.static_obstacles:
            if self._check_collision_with_obstacle(car_corners, obs.pos, obs.size):
                return -100, True  # 与静态障碍物碰撞则终止并给予负奖励

        # 检查碰撞：动态障碍物
        for obs in self.dynamic_obstacles:
            if self._check_collision_with_obstacle(car_corners, obs.pos, obs.size):
                return -150, True  # 与动态障碍物碰撞惩罚更大

        # 检查是否到达目标
        if distance_to_target < (self.car.get_corners().max() + self.target.size) / 2:
            return 1000, True  # 到达目标给予大量奖励并终止

        # 边界惩罚
        if (self.car.pos[0] <= 0 or self.car.pos[0] >= BaseConfig.ENV_WIDTH or
                self.car.pos[1] <= 0 or self.car.pos[1] >= BaseConfig.ENV_HEIGHT):
            return -50, True

        return reward, False

    def _check_collision_with_obstacle(self, car_corners: np.ndarray, obstacle_pos: np.ndarray,
                                       obstacle_size: float) -> bool:
        """检查车辆是否与障碍物碰撞"""
        # 方法1：检查障碍物中心是否在车辆多边形内部
        if self._point_in_polygon(obstacle_pos, car_corners):
            return True

        # 方法2：检查车辆的任意边是否与障碍物圆相交
        for i in range(4):
            p1 = car_corners[i]
            p2 = car_corners[(i + 1) % 4]
            if self._line_circle_intersection(p1, p2, obstacle_pos, obstacle_size):
                return True

        return False

    def _point_in_polygon(self, point: np.ndarray, polygon: np.ndarray) -> bool:
        """判断点是否在多边形内部（射线法）"""
        n = len(polygon)
        inside = False
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if point[1] > min(p1y, p2y):
                if point[1] <= max(p1y, p2y):
                    if point[0] <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (point[1] - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or point[0] <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    def _line_circle_intersection(self, p1: np.ndarray, p2: np.ndarray,
                                  circle_center: np.ndarray, circle_radius: float) -> bool:
        """判断线段是否与圆相交"""
        # 线段向量
        line_vec = p2 - p1
        line_len = np.linalg.norm(line_vec)
        line_unit_vec = line_vec / line_len if line_len > 0 else line_vec

        # 圆心到线段起点的向量
        p1_to_center = circle_center - p1

        # 计算圆心到线段的投影长度
        proj_len = np.dot(p1_to_center, line_unit_vec)

        # 如果投影点不在线段上，则检查端点到圆心的距离
        if proj_len < 0:
            closest_point = p1
        elif proj_len > line_len:
            closest_point = p2
        else:
            # 计算投影点坐标
            closest_point = p1 + line_unit_vec * proj_len

        # 计算最近点到圆心的距离
        distance = np.linalg.norm(circle_center - closest_point)

        # 如果距离小于圆的半径，则相交
        return distance <= circle_radius

    def _get_state(self) -> Dict[str, Union[np.ndarray, float]]:
        """获取多模态状态数据"""
        # 计算相对目标位置
        relative_target = self.target.pos - self.car.pos

        # 计算最近的静态障碍物距离
        min_static_obs_dist = min(np.linalg.norm(self.car.pos - obs.pos) for obs in self.static_obstacles)

        # 计算最近的动态障碍物距离和相对速度
        min_dynamic_obs_dist = float('inf')
        min_dynamic_obs_rel_vel = np.array([0.0, 0.0])
        closest_dynamic_obs = None

        for obs in self.dynamic_obstacles:
            dist = np.linalg.norm(self.car.pos - obs.pos)
            if dist < min_dynamic_obs_dist:
                min_dynamic_obs_dist = dist
                closest_dynamic_obs = obs

        if closest_dynamic_obs is not None:
            # 计算车辆速度向量
            car_vel = self.car.get_velocity_vector()
            # 计算相对速度
            min_dynamic_obs_rel_vel = closest_dynamic_obs.velocity - car_vel
        else:
            min_dynamic_obs_dist = 1000  # 如果没有动态障碍物，设置一个大值

        # 获取视觉观测
        visual_obs = self._get_visual_observation()

        # 获取激光雷达观测
        lidar_obs = self._get_lidar_observation()

        return {
            'vector': np.array([
                self.car.pos[0], self.car.pos[1],  # 车辆位置
                self.car.angle,  # 车辆角度
                self.car.speed,  # 车辆速度
                self.car.steering,  # 车辆转向角度
                relative_target[0], relative_target[1],  # 目标相对位置
                self.target.velocity[0], self.target.velocity[1],  # 目标速度
                min_static_obs_dist,  # 最近静态障碍物距离
                min_dynamic_obs_dist,  # 最近动态障碍物距离
                min_dynamic_obs_rel_vel[0], min_dynamic_obs_rel_vel[1]  # 最近动态障碍物相对速度
            ]),
            'visual': visual_obs,
            'lidar': lidar_obs
        }

    def _get_visual_observation(self) -> np.ndarray:
        """获取视觉观测
        
        返回:
            np.ndarray: 环境的渲染图像，shape=(H, W, 3)，RGB格式，uint8类型
        """
        try:
            # 创建一个空白图像（RGB格式）
            image = np.zeros((BaseConfig.ENV_HEIGHT, BaseConfig.ENV_WIDTH, 3), dtype=np.uint8)
            image[:] = (255, 255, 255)  # 白色背景
    
            # 绘制静态障碍物
            for obs in self.static_obstacles:
                obs.draw_cv2(image)
    
            # 绘制动态障碍物
            for obs in self.dynamic_obstacles:
                obs.draw_cv2(image)
                
            # 绘制目标
            self.target.draw_cv2(image)
    
            # 最后绘制小车（确保小车在最上层）
            self.car.draw_cv2(image)
            
            # 可选：绘制车辆到目标的连线，辅助训练
            # 绿色线表示从车到目标的连线
            cv2.line(image, 
                     (int(self.car.pos[0]), int(self.car.pos[1])),
                     (int(self.target.pos[0]), int(self.target.pos[1])),
                     (0, 200, 0), 1)
            
            return image
            
        except Exception as e:
            print(f"生成视觉观测时出错: {str(e)}")
            # 如果出错，返回一个默认的黑色图像
            return np.zeros((BaseConfig.ENV_HEIGHT, BaseConfig.ENV_WIDTH, 3), dtype=np.uint8)

    def _get_lidar_observation(self) -> np.ndarray:
        """获取激光雷达观测
        
        模拟激光雷达扫描，测量不同角度方向上的障碍物距离
        
        返回:
            np.ndarray: 形状为(36,)的一维数组，表示36个角度上的距离观测
        """
        try:
            # 模拟激光雷达扫描
            angles = np.linspace(0, 360, 36, endpoint=False)
            distances = []

            for angle in angles:
                rad = np.radians(angle + self.car.angle)
                direction = np.array([np.cos(rad), np.sin(rad)])

                # 射线投射
                dist = self._cast_ray(self.car.pos, direction)
                distances.append(dist)

            return np.array(distances, dtype=np.float32)
            
        except Exception as e:
            print(f"生成激光雷达观测时出错: {str(e)}")
            # 如果出错，返回一个全部为最大距离的数组
            return np.ones(36, dtype=np.float32) * 1000.0

    def _cast_ray(self, origin: np.ndarray, direction: np.ndarray, max_dist: float = 1000.0) -> float:
        """从给定原点和方向投射射线，返回到最近障碍物的距离"""
        # 步长，每次移动的距离
        step_size = 1.0
        current_dist = 0.0

        while current_dist < max_dist:
            # 当前点
            point = origin + direction * current_dist

            # 边界检查
            if not (0 <= point[0] < BaseConfig.ENV_WIDTH and 0 <= point[1] < BaseConfig.ENV_HEIGHT):
                return current_dist

            # 检查是否与目标碰撞
            if np.linalg.norm(point - self.target.pos) < self.target.size:
                return current_dist

            # 检查是否与静态障碍物碰撞
            for obs in self.static_obstacles:
                if np.linalg.norm(point - obs.pos) < obs.size:
                    return current_dist

            # 检查是否与动态障碍物碰撞
            for obs in self.dynamic_obstacles:
                if np.linalg.norm(point - obs.pos) < obs.size:
                    return current_dist

            current_dist += step_size

        return max_dist





