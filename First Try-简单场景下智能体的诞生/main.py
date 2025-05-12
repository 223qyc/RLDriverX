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

# 需要确保结果可以复现
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

os.makedirs("models", exist_ok=True)
os.makedirs("videos", exist_ok=True)

# 定义环境的参数
'''
值得考虑的有：
- 环境的大小参数
- 车身上的传感器数量
- 传感器的探测范围
- 目标的半径以及障碍物的数量和半径
- 车本身的大小参数
'''
ENV_WIDTH = 800  # 环境宽度
ENV_HEIGHT = 800  # 环境高度
SENSOR_RANGE = 25  # 传感器范围
NUM_SENSORS = 8  # 传感器数量
GOAL_RADIUS = 10  # 目标半径
OBSTACLE_RADIUS = 12  # 障碍物半径
NUM_OBSTACLES = 40  # 障碍物数量
CAR_LENGTH = 4  # 车宽
CAR_WIDTH = 2  # 车高

# 定义DQN网络的参数
BATCH_SIZE = 64
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 10000
TARGET_UPDATE = 100
MEMORY_SIZE = 10000
LEARNING_RATE = 0.001
NUM_EPISODES = 2000

# 定义简化的动作空间,包括前进与左右转
ACTION_SPACE=3

# 定义车类
class Car:
    def __init__(self):
        self.x=ENV_WIDTH // 4
        self.y=ENV_HEIGHT // 4
        self.theta = 0  # 角度通过弧度表示
        self.speed = 2
        self.turn_angle = np.pi / 10  # 转向角采用简化表示
        self.length = CAR_LENGTH
        self.width = CAR_WIDTH
        self.crashed = False   # 碰撞情况

    def reset(self):
        # 重置状态
        self.x = ENV_WIDTH // 4
        self.y = ENV_HEIGHT // 4
        self.theta = 0
        self.crashed = False

    def move(self,action):
        # 进行角度控制，保持在[0,2pi]
        self.theta = self.theta % (2 * np.pi)

        # 通过传入动作参数进行移动更新，0: 前进，1: 左转，2: 右转
        if action == 0:
            self.x += self.speed * math.cos(self.theta)
            self.y += self.speed * math.sin(self.theta)
        elif action == 1:
            self.theta += self.turn_angle
            self.x += self.speed * math.cos(self.theta)
            self.y += self.speed * math.sin(self.theta)
        elif action == 2:
            self.theta -= self.turn_angle
            self.x += self.speed * math.cos(self.theta)
            self.y += self.speed * math.sin(self.theta)

    def get_corners(self):
        # 计算车子的四个点的坐标，先建立存放点的容器，并确定角度后进行计算
        '''这里定义了坐标轴左右为左右，纵坐标向下为负，与常识略反'''
        corners = []
        cos_theta = math.cos(self.theta)
        sin_theta = math.sin(self.theta)

        # Left-Front
        front_left_x = self.x - (self.length / 2) * cos_theta - (self.width / 2) * sin_theta
        front_left_y = self.y - (self.length / 2) * sin_theta + (self.width / 2) * cos_theta
        corners.append((front_left_x, front_left_y))

        # Right-Front
        front_right_x = self.x + (self.length / 2) * cos_theta - (self.width / 2) * sin_theta
        front_right_y = self.y + (self.length / 2) * sin_theta + (self.width / 2) * cos_theta
        corners.append((front_right_x, front_right_y))

        # Left-Back
        rear_left_x = self.x - (self.length / 2) * cos_theta + (self.width / 2) * sin_theta
        rear_left_y = self.y - (self.length / 2) * sin_theta - (self.width / 2) * cos_theta
        corners.append((rear_left_x, rear_left_y))

        # Right-Back
        rear_right_x = self.x + (self.length / 2) * cos_theta + (self.width / 2) * sin_theta
        rear_right_y = self.y + (self.length / 2) * sin_theta - (self.width / 2) * cos_theta
        corners.append((rear_right_x, rear_right_y))

        return corners


class Environment():
    def __init__(self):
        self.car = Car()
        self.reset()

    def reset(self):
        ''' 配置初始化环境'''
        self.car.reset()

        # 获取目标位置
        self.goal_x = ENV_WIDTH * 3 // 4
        self.goal_y = ENV_HEIGHT * 3 // 4

        # 生成障碍物
        self.obstacles = []
        for _ in range(NUM_OBSTACLES):
            valid = False
            while not valid:
                x = random.uniform(10, ENV_WIDTH - 10)
                y = random.uniform(10, ENV_HEIGHT - 10)

                # 确保障碍物不与车辆和目标重叠
                car_dist = math.sqrt((x - self.car.x) ** 2 + (y - self.car.y) ** 2)
                goal_dist = math.sqrt((x - self.goal_x) ** 2 + (y - self.goal_y) ** 2)

                valid = car_dist > OBSTACLE_RADIUS + 10 and goal_dist > OBSTACLE_RADIUS + GOAL_RADIUS + 5
                # 计算障碍物与车辆和目标的距离，通过valid确保障碍物不太靠近车辆，障碍物也不要太靠近目标

                if valid:
                    # 检查是否与已有障碍物重叠，这里是设置了完全不能重叠
                    for ox, oy in self.obstacles:
                        dist = math.sqrt((x - ox) ** 2 + (y - oy) ** 2)
                        if dist < 2 * OBSTACLE_RADIUS:
                            valid = False
                            break

                if valid:
                    self.obstacles.append((x, y))

        # 初始化传感器的读数
        self.sensor_readings = self._get_sensor_readings()
        # 返回环境的状态
        return self._get_state()


    def _get_sensor_readings(self):
        '''获取传感器的数据'''
        readings = []
        for i in range(NUM_SENSORS):
            angle = self.car.theta + (i - NUM_SENSORS // 2) * (np.pi / (NUM_SENSORS - 1))
            reading = self._raycast(angle)
            readings.append(reading / SENSOR_RANGE)  # 归一化到[0, 1]
        return readings


    def _raycast(self,angle):
        '''计算传感器射线的终点'''
        # 计算传感器射线终点
        ray_end_x = self.car.x + SENSOR_RANGE * math.cos(angle)
        ray_end_y = self.car.y + SENSOR_RANGE * math.sin(angle)

        min_distance = SENSOR_RANGE

        # 检测与边界的碰撞
        border_intersections = [
            self._line_intersection((self.car.x, self.car.y), (ray_end_x, ray_end_y), (0, 0), (ENV_WIDTH, 0)),  # 下边界
            self._line_intersection((self.car.x, self.car.y), (ray_end_x, ray_end_y), (ENV_WIDTH, 0),
                                    (ENV_WIDTH, ENV_HEIGHT)),  # 右边界
            self._line_intersection((self.car.x, self.car.y), (ray_end_x, ray_end_y), (0, ENV_HEIGHT),
                                    (ENV_WIDTH, ENV_HEIGHT)),  # 上边界
            self._line_intersection((self.car.x, self.car.y), (ray_end_x, ray_end_y), (0, 0), (0, ENV_HEIGHT))  # 左边界
        ]

        for intersection in border_intersections:
            if intersection:
                dist = math.sqrt((intersection[0] - self.car.x) ** 2 + (intersection[1] - self.car.y) ** 2)
                min_distance = min(min_distance, dist)

        # 检测与障碍物的碰撞
        for ox, oy in self.obstacles:
            # 简化为与圆的交点检测
            dist = self._ray_circle_intersection((self.car.x, self.car.y), angle, (ox, oy), OBSTACLE_RADIUS)
            if dist is not None:
                min_distance = min(min_distance, dist)

        return min_distance


    def _line_intersection(self, line1_start, line1_end, line2_start, line2_end):
        '''计算两条线段的交点'''
        x1, y1 = line1_start
        x2, y2 = line1_end
        x3, y3 = line2_start
        x4, y4 = line2_end

        denominator = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)

        if denominator == 0:  # 平行或共线
            return None

        ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denominator
        ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / denominator

        if 0 <= ua <= 1 and 0 <= ub <= 1:  # 交点在两条线段上
            x = x1 + ua * (x2 - x1)
            y = y1 + ua * (y2 - y1)
            return (x, y)

        return None


    def _ray_circle_intersection(self, ray_origin, ray_angle, circle_center, circle_radius):
        '''计算射线和圆的交点'''
        ox, oy = ray_origin
        cx, cy = circle_center

        # 射线方向向量
        dx = math.cos(ray_angle)
        dy = math.sin(ray_angle)

        # 平移坐标系，使射线起点为原点
        cx -= ox
        cy -= oy

        # 二次方程系数
        a = dx * dx + dy * dy
        b = 2 * (dx * cx + dy * cy)
        c = cx * cx + cy * cy - circle_radius * circle_radius

        discriminant = b * b - 4 * a * c

        if discriminant < 0:  # 无交点
            return None

        # 计算交点
        t1 = (-b + math.sqrt(discriminant)) / (2 * a)
        t2 = (-b - math.sqrt(discriminant)) / (2 * a)

        # 取较近的交点，且交点必须在射线前方(t > 0)
        if t1 > 0 and t2 > 0:
            t = min(t1, t2)
        elif t1 > 0:
            t = t1
        elif t2 > 0:
            t = t2
        else:
            return None

        return t  # 返回距离


    def _check_collision(self):
        '''判断小车是否和边界发生碰撞'''
        corners = self.car.get_corners()
        for x, y in corners:
            if x < 0 or x > ENV_WIDTH or y < 0 or y > ENV_HEIGHT:
                return True

        # 检查是否与障碍物碰撞（简化为点与圆的碰撞检测）
        for ox, oy in self.obstacles:
            for x, y in corners:
                dist = math.sqrt((x - ox) ** 2 + (y - oy) ** 2)
                if dist < OBSTACLE_RADIUS:
                    return True

        return False


    def _get_state(self):
        '''用于获取传感器读数，对于目标的相对距离和角度'''
        # 状态包括传感器读数、到目标的相对距离和角度
        goal_distance = math.sqrt((self.car.x - self.goal_x) ** 2 + (self.car.y - self.goal_y) ** 2) / math.sqrt(
            ENV_WIDTH ** 2 + ENV_HEIGHT ** 2)  # 归一化
        goal_angle = math.atan2(self.goal_y - self.car.y, self.goal_x - self.car.x) - self.car.theta
        goal_angle = (goal_angle + np.pi) % (2 * np.pi) - np.pi  # 归一化到[-π, π]
        goal_angle = goal_angle / np.pi  # 归一化到[-1, 1]

        state = self.sensor_readings + [goal_distance, goal_angle]
        return torch.FloatTensor(state)


    def step(self,action):
        '''核心：更新状态，发送奖励'''
        self.car.move(action)

        # 更新传感器读数
        self.sensor_readings = self._get_sensor_readings()

        # 检查是否到达目标
        goal_distance = math.sqrt((self.car.x - self.goal_x) ** 2 + (self.car.y - self.goal_y) ** 2)
        reached_goal = goal_distance < GOAL_RADIUS

        # 检查是否碰撞
        collision = self._check_collision()
        self.car.crashed = collision

        # 计算奖励
        if reached_goal:
            reward = 500
            done = True
        elif collision:
            reward = -500
            done = True
        else:
            # 距离目标越近奖励越高
            reward = -0.1 - goal_distance / 100
            done = False

        next_state = self._get_state()

        return next_state, reward, done, {}

    def render(self, ax=None, step=None):
        '''环境可视化'''
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
        ax.clear()

        # 设置画布
        ax.set_xlim(0, ENV_WIDTH)
        ax.set_ylim(0, ENV_HEIGHT)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("Autonomous Car Navigation", fontsize=14, fontweight='bold')

        # 绘制背景
        ax.add_patch(Rectangle((0, 0), ENV_WIDTH, ENV_HEIGHT, fill=True, color='lightgray', alpha=0.3))

        # 绘制目标区域
        goal_circle = Circle((self.goal_x, self.goal_y), GOAL_RADIUS, fill=True, color='limegreen', alpha=0.7)
        ax.add_patch(goal_circle)

        # 绘制障碍物
        for ox, oy in self.obstacles:
            obstacle_circle = Circle((ox, oy), OBSTACLE_RADIUS, fill=True, color='firebrick', alpha=0.8)
            ax.add_patch(obstacle_circle)

        # 绘制车辆
        car_corners = np.array(self.car.get_corners())
        car_color = 'deepskyblue' if not self.car.crashed else 'black'
        car_patch = plt.Polygon(car_corners, fill=True, edgecolor='navy', linewidth=1.5, color=car_color, alpha=0.9)
        ax.add_patch(car_patch)

        # 绘制方向箭头
        arrow_length = 6
        arrow_x = self.car.x + arrow_length * math.cos(self.car.theta)
        arrow_y = self.car.y + arrow_length * math.sin(self.car.theta)
        ax.arrow(self.car.x, self.car.y, arrow_x - self.car.x, arrow_y - self.car.y,
                 head_width=2, head_length=3, fc='gold', ec='darkgoldenrod', alpha=0.9)

        # 绘制传感器
        for i, reading in enumerate(self.sensor_readings):
            angle = self.car.theta + (i - NUM_SENSORS // 2) * (np.pi / (NUM_SENSORS - 1))
            sensor_length = reading * SENSOR_RANGE
            sensor_x = self.car.x + sensor_length * math.cos(angle)
            sensor_y = self.car.y + sensor_length * math.sin(angle)
            ax.plot([self.car.x, sensor_x], [self.car.y, sensor_y], 'lime', linewidth=1.5, alpha=0.6)
            ax.add_patch(Circle((sensor_x, sensor_y), 1, color='lime', alpha=0.5))  # 终点标识

        # 信息面板
        info_text = f"Step: {step if step is not None else 0}\n" \
                    f"Goal Distance: {math.sqrt((self.car.x - self.goal_x) ** 2 + (self.car.y - self.goal_y) ** 2):.1f}\n" \
                    f"Status: {'Crashed' if self.car.crashed else 'Running'}"
        ax.text(5, ENV_HEIGHT - 5, info_text, fontsize=10, verticalalignment='top',
                bbox=dict(facecolor='white', alpha=0.5))

        return ax



# DQN网络
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class ReplayBuffer:
    '''定义经验缓冲池'''
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, next_state, reward, done):
        self.memory.append((state, action, next_state, reward, done))

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        state, action, next_state, reward, done = zip(*batch)
        return (
            torch.cat(state),
            torch.tensor(action),
            torch.cat(next_state),
            torch.tensor(reward, dtype=torch.float32),
            torch.tensor(done, dtype=torch.bool)
        )

    def __len__(self):
        return len(self.memory)



class Agent:
    '''定义智能代理'''
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        # DQN网络
        self.policy_net = DQN(state_size, action_size)
        self.target_net = DQN(state_size, action_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # 优化器
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)

        # 经验回放
        self.memory = ReplayBuffer(MEMORY_SIZE)

        # 探索参数
        self.eps = EPS_START
        self.steps_done = 0

    def select_action(self, state):
        sample = random.random()
        self.eps = EPS_END + (EPS_START - EPS_END) * math.exp(-self.steps_done / EPS_DECAY)
        self.steps_done += 1

        if sample > self.eps:
            with torch.no_grad():
                return self.policy_net(state).max(0)[1].item()
        else:
            return random.randrange(self.action_size)

    def learn(self):
        if len(self.memory) < BATCH_SIZE:
            return

        states, actions, next_states, rewards, dones = self.memory.sample(BATCH_SIZE)

        # 计算当前Q值
        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))

        # 计算下一状态的最大Q值
        next_q_values = self.target_net(next_states).max(1)[0].detach()

        # 计算期望Q值
        expected_q_values = rewards + GAMMA * next_q_values * (~dones)

        # 计算损失
        loss = F.smooth_l1_loss(q_values, expected_q_values.unsqueeze(1))

        # 优化模型
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, filename):
        torch.save(self.policy_net.state_dict(), filename)

    def load(self, filename):
        self.policy_net.load_state_dict(torch.load(filename))
        self.target_net.load_state_dict(self.policy_net.state_dict())



def train(num_episodes):
    '''训练过程'''
    env = Environment()
    state_size = NUM_SENSORS + 2  # 传感器数量 + 到目标的距离和角度
    agent = Agent(state_size, ACTION_SPACE)

    rewards = []
    best_reward = -float('inf')

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0

        while True:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)

            agent.memory.push(state.unsqueeze(0), action, next_state.unsqueeze(0), reward, done)
            state = next_state
            episode_reward += reward

            loss = agent.learn()

            if done:
                break

        # 更新目标网络
        if episode % TARGET_UPDATE == 0:
            agent.update_target_net()

        rewards.append(episode_reward)

        # 保存最佳模型
        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.save("models/best_model.pth")

        print(f"Episode: {episode}, Reward: {episode_reward:.2f}, Epsilon: {agent.eps:.2f}")

    # 绘制奖励曲线
    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    plt.title('Training Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.savefig('training_rewards.png')
    plt.close()

    return agent


def evaluate(agent, num_episodes=20, render=True, save_video=False):
    env = Environment()
    rewards = []

    # 视频保存的设置
    if save_video:
        os.makedirs("videos", exist_ok=True)
        # 创建一个图形对象
        fig, ax = plt.subplots(figsize=(10, 10))
        writer = animation.FFMpegWriter(fps=15)
        with writer.saving(fig, 'videos/evaluation.mp4', dpi=100):
            for episode in tqdm(range(num_episodes), desc="Evaluating Episodes", unit="episode"):
                state = env.reset()
                episode_reward = 0
                step = 0

                while True:
                    action = agent.select_action(state)
                    next_state, reward, done, _ = env.step(action)

                    # 渲染当前帧并保存
                    env.render(ax=ax, step=step)
                    writer.grab_frame()

                    state = next_state
                    episode_reward += reward
                    step += 1

                    if done:
                        break

                rewards.append(episode_reward)
                print(f"Evaluation Episode: {episode + 1}, Reward: {episode_reward:.2f}")
        plt.close(fig)
    else:
        # 不保存视频，只进行评估
        fig, ax = None, None
        if render:
            fig, ax = plt.subplots(figsize=(10, 10))

        for episode in tqdm(range(num_episodes), desc="Evaluating Episodes", unit="episode"):
            state = env.reset()
            episode_reward = 0
            step = 0

            while True:
                action = agent.select_action(state)
                next_state, reward, done, _ = env.step(action)

                if render:
                    env.render(ax=ax, step=step)
                    plt.pause(0.01)  # 显示一小段时间

                state = next_state
                episode_reward += reward
                step += 1

                if done:
                    break

            rewards.append(episode_reward)
            print(f"Evaluation Episode: {episode + 1}, Reward: {episode_reward:.2f}")

        if fig is not None:
            plt.close(fig)

    # 返回所有回合的平均奖励
    average_reward = np.mean(rewards)
    print(f"Average Reward over {num_episodes} episodes: {average_reward:.2f}")
    return average_reward


if __name__ == "__main__":
    # 训练智能体
    print("开始训练...")
    agent = train(NUM_EPISODES)

    # 评估智能体
    print("\n开始评估...")
    evaluate(agent, num_episodes=6, render=True, save_video=True)

