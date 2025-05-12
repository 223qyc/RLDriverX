import os
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, Circle, Arrow
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_

# 确保结果可复现
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

os.makedirs("models", exist_ok=True)
os.makedirs("videos", exist_ok=True)

# ================== 环境参数 ==================
ENV_WIDTH = 800
ENV_HEIGHT = 800
SENSOR_RANGE = 50  # 增加传感器范围以适应更大环境
NUM_SENSORS = 16  # 增加传感器数量以获取更全面的环境信息
GOAL_RADIUS = 15
OBSTACLE_RADIUS = 15
NUM_OBSTACLES = 60  # 增加障碍物数量以提供更多学习样本
CAR_LENGTH = 5
CAR_WIDTH = 3

# ================== DQN参数 ==================
BATCH_SIZE = 128  # 增大批大小
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.05  # 保留少量探索
EPS_DECAY = 20000  # 延长探索衰减
TARGET_UPDATE = 200  # 减少目标网络更新频率
MEMORY_SIZE = 50000  # 增大经验回放池
LEARNING_RATE = 0.00025  # 降低学习率
NUM_EPISODES = 3000  # 增加训练轮数
GRAD_CLIP = 1.0  # 梯度裁剪

# 动作空间保持不变
ACTION_SPACE = 3


# ================== 改进的车类 ==================
class Car:
    def __init__(self):
        self.x = ENV_WIDTH // 4
        self.y = ENV_HEIGHT // 4
        self.theta = 0
        self.speed = 2.5  # 略微提高速度
        self.turn_angle = np.pi / 12  # 减小转向角度使控制更平滑
        self.length = CAR_LENGTH
        self.width = CAR_WIDTH
        self.crashed = False

    def reset(self):
        self.x = ENV_WIDTH // 4
        self.y = ENV_HEIGHT // 4
        self.theta = 0
        self.crashed = False

    def move(self, action):
        self.theta = self.theta % (2 * np.pi)

        # 添加速度控制使动作更平滑
        if action == 0:  # 前进
            self.x += self.speed * math.cos(self.theta)
            self.y += self.speed * math.sin(self.theta)
        elif action == 1:  # 左转
            self.theta += self.turn_angle
            self.x += (self.speed * 0.8) * math.cos(self.theta)  # 转向时略微减速
            self.y += (self.speed * 0.8) * math.sin(self.theta)
        elif action == 2:  # 右转
            self.theta -= self.turn_angle
            self.x += (self.speed * 0.8) * math.cos(self.theta)
            self.y += (self.speed * 0.8) * math.sin(self.theta)

    def get_corners(self):
        corners = []
        cos_theta = math.cos(self.theta)
        sin_theta = math.sin(self.theta)

        # 计算四个角点
        half_length = self.length / 2
        half_width = self.width / 2

        # 前左
        x = self.x - half_length * cos_theta - half_width * sin_theta
        y = self.y - half_length * sin_theta + half_width * cos_theta
        corners.append((x, y))

        # 前右
        x = self.x + half_length * cos_theta - half_width * sin_theta
        y = self.y + half_length * sin_theta + half_width * cos_theta
        corners.append((x, y))

        # 后左
        x = self.x - half_length * cos_theta + half_width * sin_theta
        y = self.y - half_length * sin_theta - half_width * cos_theta
        corners.append((x, y))

        # 后右
        x = self.x + half_length * cos_theta + half_width * sin_theta
        y = self.y + half_length * sin_theta - half_width * cos_theta
        corners.append((x, y))

        return corners


# ================== 环境类 ==================
class Environment():
    def __init__(self):
        self.car = Car()
        self.reset()

    def reset(self):
        self.car.reset()
        self.prev_x = self.car.x
        self.prev_y = self.car.y

        # 随机化目标位置
        if random.random() < 0.3:  # 30%概率目标在对角线另一侧
            self.goal_x = ENV_WIDTH * 3 // 4
            self.goal_y = ENV_HEIGHT * 3 // 4
        else:
            self.goal_x = random.randint(ENV_WIDTH // 2, ENV_WIDTH - 20)
            self.goal_y = random.randint(ENV_HEIGHT // 2, ENV_HEIGHT - 20)

        # 改进的障碍物生成算法
        self.obstacles = []
        min_dist = OBSTACLE_RADIUS * 2.5  # 增加障碍物间最小距离

        # 确保初始区域无障碍物
        safe_zone_radius = 100
        for _ in range(NUM_OBSTACLES * 2):  # 尝试更多次
            if len(self.obstacles) >= NUM_OBSTACLES:
                break

            x = random.uniform(OBSTACLE_RADIUS, ENV_WIDTH - OBSTACLE_RADIUS)
            y = random.uniform(OBSTACLE_RADIUS, ENV_HEIGHT - OBSTACLE_RADIUS)

            # 检查与车的距离
            car_dist = math.sqrt((x - self.car.x) ** 2 + (y - self.car.y) ** 2)
            if car_dist < safe_zone_radius:
                continue

            # 检查与目标的距离
            goal_dist = math.sqrt((x - self.goal_x) ** 2 + (y - self.goal_y) ** 2)
            if goal_dist < safe_zone_radius / 2:
                continue

            # 检查与其他障碍物的距离
            valid = True
            for (ox, oy) in self.obstacles:
                if math.sqrt((x - ox) ** 2 + (y - oy) ** 2) < min_dist:
                    valid = False
                    break

            if valid:
                self.obstacles.append((x, y))

        self.sensor_readings = self._get_sensor_readings()
        return self._get_state()

    def _get_sensor_readings(self):
        readings = []
        for i in range(NUM_SENSORS):
            angle = self.car.theta + (i - NUM_SENSORS // 2) * (2 * np.pi / NUM_SENSORS)
            reading = self._raycast(angle)
            readings.append(reading / SENSOR_RANGE)
        return readings

    def _raycast(self, angle):
        ray_end_x = self.car.x + SENSOR_RANGE * math.cos(angle)
        ray_end_y = self.car.y + SENSOR_RANGE * math.sin(angle)
        min_distance = SENSOR_RANGE

        # 边界检测
        border_intersections = [
            self._line_intersection((self.car.x, self.car.y), (ray_end_x, ray_end_y), (0, 0), (ENV_WIDTH, 0)),
            self._line_intersection((self.car.x, self.car.y), (ray_end_x, ray_end_y), (ENV_WIDTH, 0),
                                    (ENV_WIDTH, ENV_HEIGHT)),
            self._line_intersection((self.car.x, self.car.y), (ray_end_x, ray_end_y), (0, ENV_HEIGHT),
                                    (ENV_WIDTH, ENV_HEIGHT)),
            self._line_intersection((self.car.x, self.car.y), (ray_end_x, ray_end_y), (0, 0), (0, ENV_HEIGHT))
        ]

        for intersection in border_intersections:
            if intersection:
                dist = math.sqrt((intersection[0] - self.car.x) ** 2 + (intersection[1] - self.car.y) ** 2)
                min_distance = min(min_distance, dist)

        # 障碍物检测
        for ox, oy in self.obstacles:
            dist = self._ray_circle_intersection((self.car.x, self.car.y), angle, (ox, oy), OBSTACLE_RADIUS)
            if dist is not None:
                min_distance = min(min_distance, dist)

        return min_distance

    def _line_intersection(self, line1_start, line1_end, line2_start, line2_end):
        x1, y1 = line1_start
        x2, y2 = line1_end
        x3, y3 = line2_start
        x4, y4 = line2_end

        denominator = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)

        if denominator == 0:
            return None

        ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denominator
        ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / denominator

        if 0 <= ua <= 1 and 0 <= ub <= 1:
            x = x1 + ua * (x2 - x1)
            y = y1 + ua * (y2 - y1)
            return (x, y)

        return None

    def _ray_circle_intersection(self, ray_origin, ray_angle, circle_center, circle_radius):
        ox, oy = ray_origin
        cx, cy = circle_center

        dx = math.cos(ray_angle)
        dy = math.sin(ray_angle)

        cx -= ox
        cy -= oy

        a = dx * dx + dy * dy
        b = 2 * (dx * cx + dy * cy)
        c = cx * cx + cy * cy - circle_radius * circle_radius

        discriminant = b * b - 4 * a * c

        if discriminant < 0:
            return None

        t1 = (-b + math.sqrt(discriminant)) / (2 * a)
        t2 = (-b - math.sqrt(discriminant)) / (2 * a)

        if t1 > 0 and t2 > 0:
            t = min(t1, t2)
        elif t1 > 0:
            t = t1
        elif t2 > 0:
            t = t2
        else:
            return None

        return t

    def _check_collision(self):
        corners = self.car.get_corners()

        # 边界碰撞检测
        for x, y in corners:
            if x < 0 or x > ENV_WIDTH or y < 0 or y > ENV_HEIGHT:
                return True

        # 障碍物碰撞检测
        for ox, oy in self.obstacles:
            for x, y in corners:
                dist = math.sqrt((x - ox) ** 2 + (y - oy) ** 2)
                if dist < OBSTACLE_RADIUS + 2:  # 增加安全距离
                    return True

            # 额外检查车中心与障碍物的距离
            center_dist = math.sqrt((self.car.x - ox) ** 2 + (self.car.y - oy) ** 2)
            if center_dist < OBSTACLE_RADIUS + self.car.length / 2:
                return True

        return False

    def _get_state(self):
        # 状态表示
        goal_distance = math.sqrt((self.car.x - self.goal_x) ** 2 + (self.car.y - self.goal_y) ** 2)
        max_distance = math.sqrt(ENV_WIDTH ** 2 + ENV_HEIGHT ** 2)
        normalized_distance = goal_distance / max_distance

        goal_angle = math.atan2(self.goal_y - self.car.y, self.goal_x - self.car.x) - self.car.theta
        goal_angle = (goal_angle + np.pi) % (2 * np.pi) - np.pi
        normalized_angle = goal_angle / np.pi

        # 添加速度信息
        speed_info = self.car.speed / 5.0  # 归一化

        # 添加历史传感器读数
        state = self.sensor_readings + [normalized_distance, normalized_angle, speed_info]
        return torch.FloatTensor(state)

    def step(self, action):
        self.car.move(action)
        self.sensor_readings = self._get_sensor_readings()

        goal_distance = math.sqrt((self.car.x - self.goal_x) ** 2 + (self.car.y - self.goal_y) ** 2)
        reached_goal = goal_distance < GOAL_RADIUS
        collision = self._check_collision()
        self.car.crashed = collision

        # ========== 奖励函数 ==========
        if reached_goal:
            reward = 1000  # 增加到达目标的奖励
            done = True
        elif collision:
            reward = -800  # 调整碰撞惩罚
            done = True
        else:
            # 基于距离的奖励
            prev_distance = math.sqrt((self.prev_x - self.goal_x) ** 2 + (self.prev_y - self.goal_y) ** 2)
            distance_reward = (prev_distance - goal_distance) * 10  # 鼓励靠近目标

            # 传感器危险惩罚
            danger_penalty = 0
            min_sensor = min(self.sensor_readings)
            if min_sensor < 0.3:  # 太靠近障碍物
                danger_penalty = -5 * (1 - min_sensor)

            # 行动惩罚 (鼓励高效路径)
            action_penalty = -0.2

            # 角度奖励 (鼓励朝向目标)
            goal_angle = math.atan2(self.goal_y - self.car.y, self.goal_x - self.car.x)
            angle_diff = abs((self.car.theta - goal_angle + np.pi) % (2 * np.pi) - np.pi)
            angle_reward = -angle_diff / np.pi * 0.5

            reward = distance_reward + danger_penalty + action_penalty + angle_reward
            done = False

        self.prev_x, self.prev_y = self.car.x, self.car.y
        next_state = self._get_state()

        return next_state, reward, done, {}

    def render(self, ax=None, step=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
        ax.clear()

        ax.set_xlim(0, ENV_WIDTH)
        ax.set_ylim(0, ENV_HEIGHT)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("Autonomous Car Navigation (Optimized)", fontsize=14, fontweight='bold')

        # 绘制背景网格
        ax.grid(True, color='gray', linestyle='--', alpha=0.3)

        # 绘制目标
        goal_circle = Circle((self.goal_x, self.goal_y), GOAL_RADIUS, fill=True, color='limegreen', alpha=0.7)
        ax.add_patch(goal_circle)

        # 绘制障碍物
        for ox, oy in self.obstacles:
            obstacle_circle = Circle((ox, oy), OBSTACLE_RADIUS, fill=True, color='firebrick', alpha=0.8)
            ax.add_patch(obstacle_circle)

        # 绘制车
        car_corners = np.array(self.car.get_corners())
        car_color = 'deepskyblue' if not self.car.crashed else 'black'
        car_patch = plt.Polygon(car_corners, fill=True, edgecolor='navy', linewidth=1.5, color=car_color, alpha=0.9)
        ax.add_patch(car_patch)

        # 绘制方向箭头
        arrow_length = 8
        arrow_x = self.car.x + arrow_length * math.cos(self.car.theta)
        arrow_y = self.car.y + arrow_length * math.sin(self.car.theta)
        ax.arrow(self.car.x, self.car.y, arrow_x - self.car.x, arrow_y - self.car.y,
                 head_width=3, head_length=4, fc='gold', ec='darkgoldenrod', alpha=0.9)

        # 绘制传感器
        for i, reading in enumerate(self.sensor_readings):
            angle = self.car.theta + (i - NUM_SENSORS // 2) * (2 * np.pi / NUM_SENSORS)
            sensor_length = reading * SENSOR_RANGE
            sensor_x = self.car.x + sensor_length * math.cos(angle)
            sensor_y = self.car.y + sensor_length * math.sin(angle)
            ax.plot([self.car.x, sensor_x], [self.car.y, sensor_y], 'lime', linewidth=1.5, alpha=0.6)
            ax.add_patch(Circle((sensor_x, sensor_y), 1.5, color='lime', alpha=0.7))

        # 信息面板
        info_text = f"Step: {step if step is not None else 0}\n" \
                    f"Goal Distance: {math.sqrt((self.car.x - self.goal_x) ** 2 + (self.car.y - self.goal_y) ** 2):.1f}\n" \
                    f"Status: {'Crashed' if self.car.crashed else 'Running'}"
        ax.text(5, ENV_HEIGHT - 5, info_text, fontsize=10, verticalalignment='top',
                bbox=dict(facecolor='white', alpha=0.7))

        return ax


# ================== DQN网络 ==================
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


# ================== 经验回放 ==================
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)

    def push(self, state, action, next_state, reward, done):
        max_prio = self.priorities.max() if self.buffer else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, next_state, reward, done))
        else:
            self.buffer[self.pos] = (state, action, next_state, reward, done)

        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]

        probs = prios ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        batch = list(zip(*samples))
        states = torch.cat(batch[0])
        actions = torch.tensor(batch[1])
        next_states = torch.cat(batch[2])
        rewards = torch.tensor(batch[3], dtype=torch.float32)
        dones = torch.tensor(batch[4], dtype=torch.bool)

        return states, actions, next_states, rewards, dones, indices, torch.from_numpy(weights)

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio

    def __len__(self):
        return len(self.buffer)


# ================== 智能体 ==================
class Agent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        # 双DQN网络
        self.policy_net = DQN(state_size, action_size)
        self.target_net = DQN(state_size, action_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # 优化器
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
        self.loss_fn = nn.SmoothL1Loss(reduction='none')

        # 优先经验回放
        self.memory = PrioritizedReplayBuffer(MEMORY_SIZE)

        # 探索参数
        self.eps = EPS_START
        self.steps_done = 0
        self.beta = 0.4
        self.beta_increment = 0.001

    def select_action(self, state):
        sample = random.random()
        self.eps = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1

        if sample > self.eps:
            with torch.no_grad():
                # 修改这行，添加unsqueeze(0)使输入变为2D
                return self.policy_net(state.unsqueeze(0)).max(1)[1].item()
        else:
            return random.randrange(self.action_size)

    def learn(self):
        if len(self.memory) < BATCH_SIZE:
            return 0

        states, actions, next_states, rewards, dones, indices, weights = self.memory.sample(BATCH_SIZE, self.beta)
        self.beta = min(1.0, self.beta + self.beta_increment)

        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))

        with torch.no_grad():
            next_actions = self.policy_net(next_states).max(1)[1].unsqueeze(1)
            next_q = self.target_net(next_states).gather(1, next_actions)
            expected_q = rewards.unsqueeze(1) + GAMMA * next_q * (~dones).unsqueeze(1)

        losses = self.loss_fn(current_q, expected_q)
        new_priorities = (losses.detach().numpy() + 1e-5).flatten()
        self.memory.update_priorities(indices, new_priorities)

        # 修改这行代码
        loss = (losses * torch.FloatTensor(weights).unsqueeze(1)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.policy_net.parameters(), GRAD_CLIP)
        self.optimizer.step()

        return loss.item()

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, filename):
        torch.save({
            'policy_state_dict': self.policy_net.state_dict(),
            'target_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'steps_done': self.steps_done,
            'eps': self.eps
        }, filename)

    def load(self, filename):
        checkpoint = torch.load(filename)
        self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.steps_done = checkpoint['steps_done']
        self.eps = checkpoint['eps']


# ================== 训练过程 ==================
def train(num_episodes):
    env = Environment()
    state_size = NUM_SENSORS + 3  # 传感器 + 距离 + 角度 + 速度
    agent = Agent(state_size, ACTION_SPACE)

    rewards = []
    losses = []
    best_reward = -float('inf')

    # 课程学习 - 初始简单环境
    global NUM_OBSTACLES, OBSTACLE_RADIUS
    initial_obstacles = NUM_OBSTACLES
    initial_radius = OBSTACLE_RADIUS

    for episode in tqdm(range(num_episodes), desc="Training", unit="episode"):
        # 动态调整难度
        if episode > num_episodes * 0.7:  # 最后30%增加难度
            NUM_OBSTACLES = min(initial_obstacles * 2, 100)
            OBSTACLE_RADIUS = initial_radius * 1.5
        elif episode > num_episodes * 0.4:  # 中间30%中等难度
            NUM_OBSTACLES = initial_obstacles
            OBSTACLE_RADIUS = initial_radius

        state = env.reset()
        episode_reward = 0
        episode_loss = 0
        step_count = 0

        while True:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)

            agent.memory.push(state.unsqueeze(0), action, next_state.unsqueeze(0), reward, done)
            state = next_state
            episode_reward += reward
            step_count += 1

            loss = agent.learn()
            if loss:
                episode_loss += loss

            if done or step_count >= 1000:  # 防止无限循环
                break

        # 更新目标网络
        if episode % TARGET_UPDATE == 0:
            agent.update_target_net()

        rewards.append(episode_reward)
        if episode_loss > 0:
            losses.append(episode_loss / step_count)

        # 保存最佳模型
        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.save("models/best_model.pth")

        # 定期打印进度
        if episode % 100 == 0:
            avg_reward = np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards)
            avg_loss = np.mean(losses[-100:]) if len(losses) >= 100 else np.mean(losses) if losses else 0
            tqdm.write(
                f"Episode: {episode}, Avg Reward: {avg_reward:.2f}, Avg Loss: {avg_loss:.4f}, Epsilon: {agent.eps:.3f}")

    # 恢复原始环境参数
    NUM_OBSTACLES = initial_obstacles
    OBSTACLE_RADIUS = initial_radius

    # 绘制训练曲线
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(rewards)
    plt.title('Training Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')

    plt.subplot(1, 2, 2)
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Episode')
    plt.ylabel('Loss')

    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.close()

    return agent


# ================== 改进的评估函数 ==================
def evaluate(agent, num_episodes=10, render=True, save_video=False):
    env = Environment()
    rewards = []
    success_rate = 0

    if save_video:
        fig, ax = plt.subplots(figsize=(10, 10))
        writer = animation.FFMpegWriter(fps=15)
        with writer.saving(fig, 'videos/evaluation.mp4', dpi=100):
            for episode in tqdm(range(num_episodes), desc="Evaluating", unit="episode"):
                state = env.reset()
                episode_reward = 0
                step = 0

                while True:
                    with torch.no_grad():
                        action = agent.policy_net(state).max(0)[1].item()

                    next_state, reward, done, _ = env.step(action)

                    env.render(ax=ax, step=step)
                    writer.grab_frame()

                    state = next_state
                    episode_reward += reward
                    step += 1

                    if done or step >= 1000:
                        if reward > 100:  # 成功到达目标
                            success_rate += 1
                        break

                rewards.append(episode_reward)
        plt.close(fig)
    else:
        fig, ax = None, None
        if render:
            fig, ax = plt.subplots(figsize=(10, 10))

        for episode in tqdm(range(num_episodes), desc="Evaluating", unit="episode"):
            state = env.reset()
            episode_reward = 0
            step = 0

            while True:
                with torch.no_grad():
                    action = agent.policy_net(state).max(0)[1].item()

                next_state, reward, done, _ = env.step(action)

                if render:
                    env.render(ax=ax, step=step)
                    plt.pause(0.01)

                state = next_state
                episode_reward += reward
                step += 1

                if done or step >= 1000:
                    if reward > 100:  # 成功到达目标
                        success_rate += 1
                    break

            rewards.append(episode_reward)

        if fig is not None:
            plt.close(fig)

    avg_reward = np.mean(rewards)
    success_rate = success_rate / num_episodes * 100
    print(f"\nEvaluation Results:")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Success Rate: {success_rate:.1f}%")
    print(f"Min Reward: {min(rewards):.2f}, Max Reward: {max(rewards):.2f}")

    return avg_reward, success_rate


if __name__ == "__main__":
    # 训练智能体
    print("Starting training...")
    trained_agent = train(NUM_EPISODES)

    # 评估智能体
    print("\nStarting evaluation...")
    avg_reward, success_rate = evaluate(trained_agent, num_episodes=10, render=True, save_video=True)