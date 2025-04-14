"""
优化后的基础训练器模块
实现了基于优先经验回放(PER)的DQN算法，用于自动驾驶决策训练
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import os
import time
from .environment import CarEnvironment
from .config import BaseConfig
from .models import MultiModalDQN
import cv2



# --- 图像预处理函数 ---
def preprocess_visual(frame: np.ndarray, target_dim: tuple) -> np.ndarray:
    """
    预处理视觉帧：调整大小、归一化、转换通道顺序
    
    参数:
        frame: 原始图像，格式为(H, W, C)的RGB图像
        target_dim: 目标分辨率，(高度, 宽度)
        
    返回:
        处理后的图像，格式为(C, H, W)的张量格式，值范围[0,1]
    """
    if frame is None:
        # 如果环境没有提供视觉帧（例如配置关闭），返回一个零数组
        # 形状应为 (C, H, W)
        return np.zeros((BaseConfig.VISUAL_INPUT_CHANNELS, target_dim[0], target_dim[1]), dtype=np.float32)
    
    try:
        # 检查输入帧的有效性
        if not isinstance(frame, np.ndarray):
            print(f"警告：视觉帧类型无效，期望numpy数组，得到{type(frame)}")
            return np.zeros((BaseConfig.VISUAL_INPUT_CHANNELS, target_dim[0], target_dim[1]), dtype=np.float32)
            
        # 检查帧的维度
        if frame.ndim != 3:
            print(f"警告：视觉帧维度无效，期望3维(H,W,C)，得到{frame.ndim}维")
            if frame.ndim == 2:  # 灰度图
                # 扩展为3维
                frame = np.expand_dims(frame, axis=2)
                # 复制到3通道
                frame = np.repeat(frame, 3, axis=2)
            else:
                return np.zeros((BaseConfig.VISUAL_INPUT_CHANNELS, target_dim[0], target_dim[1]), dtype=np.float32)
                
        # 确保是uint8类型且范围为[0,255]
        if frame.dtype == np.float32 or frame.dtype == np.float64:
            if np.max(frame) <= 1.0:
                frame = (frame * 255).astype(np.uint8)
        
        # 调整大小 (使用 OpenCV)
        resized_frame = cv2.resize(frame, (target_dim[1], target_dim[0]), interpolation=cv2.INTER_AREA)
        
        # 归一化到 [0, 1]
        normalized_frame = resized_frame.astype(np.float32) / 255.0
        
        # 转换通道顺序 (H, W, C) -> (C, H, W)，PyTorch CNN 通常期望通道在前而cv不同
        transposed_frame = np.transpose(normalized_frame, (2, 0, 1))
        
        return transposed_frame
        
    except Exception as e:
        print(f"预处理视觉帧时出错: {str(e)}")
        # 出错时返回零张量
        return np.zeros((BaseConfig.VISUAL_INPUT_CHANNELS, target_dim[0], target_dim[1]), dtype=np.float32)

# --- Lidar 预处理函数 ---
def preprocess_lidar(lidar_data: np.ndarray, max_dist=1000.0) -> np.ndarray:
    """
    将Lidar距离归一化到[0,1]范围
    
    参数:
        lidar_data: 激光雷达距离数据，应该是一维数组
        max_dist: 最大距离值，用于归一化
        
    返回:
        归一化后的激光雷达数据，值范围[0,1]
    """
    try:
        if lidar_data is None:
            # 处理可能不存在的情况
            print("警告: Lidar数据为None")
            return np.array([], dtype=np.float32)
            
        # 检查数据类型和转换
        if isinstance(lidar_data, list):
            lidar_data = np.array(lidar_data, dtype=np.float32)
        elif not isinstance(lidar_data, np.ndarray):
            print(f"警告: Lidar数据类型无效，期望数组，得到{type(lidar_data)}")
            return np.array([], dtype=np.float32)
            
        # 检查数组维度
        if lidar_data.ndim > 1:
            # 尝试展平多维数组
            try:
                original_shape = lidar_data.shape
                lidar_data = lidar_data.flatten()
                print(f"警告: 将多维Lidar数据从{original_shape}展平为{lidar_data.shape}")
            except Exception as e:
                print(f"展平Lidar数据失败: {str(e)}")
                return np.array([], dtype=np.float32)
                
        # 确保是float32类型    
        lidar_data = lidar_data.astype(np.float32)
        
        # 归一化处理
        # 替换无效值（如无穷大或NaN）
        lidar_data = np.nan_to_num(lidar_data, nan=max_dist, posinf=max_dist, neginf=0)
        
        # 裁剪值范围并归一化
        lidar_data = np.clip(lidar_data, 0, max_dist) / max_dist
        
        return lidar_data
        
    except Exception as e:
        print(f"预处理Lidar数据时出错: {str(e)}")
        # 出错时返回空数组
        return np.array([], dtype=np.float32)


class OptimizedBaseTrainer:
    """使用多模态输入的优化基础训练器"""

    def __init__(self, action_dim=4):
        """
        初始化训练器
        :param action_dim: 动作维度，默认为4(左转、右转、加速、减速)
        """
        self.env = CarEnvironment()  # 创建汽车环境
        self.action_dim = action_dim
        
        # 获取环境状态维度信息 (从环境实例获取)
        # 需要 reset 一次来获取状态字典结构
        initial_state_dict = self.env.reset()
        self.vector_dim = initial_state_dict['vector'].shape[0]
        self.lidar_dim = initial_state_dict['lidar'].shape[0]
        # 视觉维度由配置决定 BaseConfig.VISUAL_RESIZE_DIM
        
        if self.lidar_dim == 0:
             print("警告：环境未提供有效的Lidar数据，Lidar分支将接收空输入。")
             # 如果 lidar_dim 为 0 (因为 preprocess_lidar 返回空)，需要处理
             # 我们需要在 MultiModalDQN 初始化前确定 EXPECTED_LIDAR_DIM
             # 检查环境的 _get_lidar_observation 确定维度
             # 假设它总是返回 36 个点
             self.lidar_dim = 36 # 硬编码一个预期维度，如果 reset 失败
             print(f"假设 Lidar 维度为 {self.lidar_dim}")
             
        print(f"状态维度 - Vector: {self.vector_dim}, Lidar: {self.lidar_dim}, Visual: {BaseConfig.VISUAL_INPUT_CHANNELS}x{BaseConfig.VISUAL_RESIZE_DIM}")


        # 初始化多模态DQN和目标网络
        self.model = MultiModalDQN(self.vector_dim, self.lidar_dim, self.action_dim).to(BaseConfig.DEVICE)
        self.target_model = MultiModalDQN(self.vector_dim, self.lidar_dim, self.action_dim).to(BaseConfig.DEVICE)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()  # 目标网络设为评估模式

        # 优化器和学习率调度器
        self.optimizer = optim.AdamW(self.model.parameters(), lr=BaseConfig.LR, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=5, verbose=True
        )

        # 优先经验回放缓冲区
        self.memory = PrioritizedReplayBuffer(capacity=BaseConfig.REPLAY_BUFFER_SIZE, alpha=0.6)
        self.beta_start = 0.4
        self.beta_frames = 10000
        self.batch_size = BaseConfig.BATCH_SIZE

        # 训练参数
        self.gamma = BaseConfig.GAMMA
        self.epsilon = BaseConfig.EPSILON_START
        self.epsilon_min = BaseConfig.EPSILON_END
        self.epsilon_decay = (BaseConfig.EPSILON_START - BaseConfig.EPSILON_END) / BaseConfig.EPSILON_DECAY_STEPS
        self.tau = BaseConfig.TAU
        self.learning_steps_per_update = BaseConfig.LEARNING_STEPS_PER_UPDATE
        self.train_step_counter = 0
        self.criterion = nn.SmoothL1Loss(reduction='none')  # 使用none以支持PER权重
        
        # 训练统计信息
        self.training_stats = {
            'episode_rewards': [],
            'episode_losses': [],
            'avg_q_values': [],
            'training_times': [],
            'td_errors': []
        }
        
        # 性能监控
        self.last_time = time.time()
        self.last_step = 0
        self.fps_history = deque(maxlen=100)

    def remember(self, state_dict, action, reward, next_state_dict, done):
        """
        将包含多模态数据的经验存入记忆缓冲区
        :param state_dict: 当前状态字典
        :param action: 采取的动作
        :param reward: 获得的奖励
        :param next_state_dict: 下一状态字典
        :param done: 是否终止
        """
        priority = self.memory.max_priority() if len(self.memory) > 0 else 1.0
        self.memory.add(priority, (state_dict, action, reward, next_state_dict, done))

    def act(self, state_dict, training=True):
        """
        ε-贪婪策略选择动作
        :param state_dict: 当前状态字典
        :param training: 是否处于训练模式（决定是否使用ε-贪婪策略）
        :return: 选择的动作和Q值
        """
        # 只有在训练模式下才应用ε-贪婪
        if training and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_dim), 0.0

        # --- 预处理状态 --- 
        visual_input = preprocess_visual(state_dict.get('visual'), BaseConfig.VISUAL_RESIZE_DIM)
        lidar_input = preprocess_lidar(state_dict.get('lidar'))
        vector_input = state_dict['vector'].astype(np.float32) # 确保向量是 float32
        
        # 如果 Lidar 处理后为空（因为 reset 时获取失败），创建一个零向量
        if lidar_input.size == 0:
            lidar_input = np.zeros(self.lidar_dim, dtype=np.float32)

        # 将处理后的数据转换为 PyTorch 张量并添加 batch 维度
        processed_state_dict = {
            'visual': torch.FloatTensor(visual_input).unsqueeze(0).to(BaseConfig.DEVICE),
            'lidar': torch.FloatTensor(lidar_input).unsqueeze(0).to(BaseConfig.DEVICE),
            'vector': torch.FloatTensor(vector_input).unsqueeze(0).to(BaseConfig.DEVICE)
        }
        
        # --- 使用模型预测 --- 
        self.model.eval()
        with torch.no_grad():
            act_values = self.model(processed_state_dict)
            best_action = torch.argmax(act_values).item()
            best_q_value = act_values[0, best_action].item()
        self.model.train()
        return best_action, best_q_value

    def update_epsilon(self):
        """更新探索率ε"""
        self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay)

    def update_target_network(self):
        """软更新目标网络"""
        for target_param, local_param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.mul_(1 - self.tau)
            target_param.data.add_(self.tau * local_param.data)

    def replay(self, frame_idx):
        """
        从记忆缓冲区采样并训练
        :param frame_idx: 当前帧索引(用于计算beta)
        :return: 损失值和平均Q值
        """
        # 如果内存缓冲区不足以形成一个批次，返回None
        if len(self.memory) < self.batch_size:
            return None, None

        try:
            # 计算PER的beta参数
            beta = min(1.0, self.beta_start + frame_idx * (1.0 - self.beta_start) / self.beta_frames)
            # 从经验回放缓冲区采样
            samples = self.memory.sample(self.batch_size, beta)

            # --- 批量预处理状态 --- 
            # 从样本中解包状态字典列表
            state_dicts = samples['states'] # 这现在是字典列表
            next_state_dicts = samples['next_states'] # 这现在是字典列表
            
            # 批量处理视觉、Lidar 和向量数据
            try:
                # 使用列表推导预处理数据，并处理可能的空值情况
                visual_batch = np.array([preprocess_visual(s.get('visual'), BaseConfig.VISUAL_RESIZE_DIM) for s in state_dicts])
                lidar_batch = np.array([preprocess_lidar(s.get('lidar')) for s in state_dicts])
                vector_batch = np.array([s['vector'].astype(np.float32) for s in state_dicts])
                
                next_visual_batch = np.array([preprocess_visual(ns.get('visual'), BaseConfig.VISUAL_RESIZE_DIM) for ns in next_state_dicts])
                next_lidar_batch = np.array([preprocess_lidar(ns.get('lidar')) for ns in next_state_dicts])
                next_vector_batch = np.array([ns['vector'].astype(np.float32) for ns in next_state_dicts])
            except Exception as e:
                print(f"批量预处理状态出错: {e}")
                return None, None

            # 处理可能的空 Lidar 数据
            if lidar_batch.ndim == 1 or lidar_batch.size == 0: # 处理空数组情况
                lidar_batch = np.zeros((self.batch_size, self.lidar_dim), dtype=np.float32)
            if next_lidar_batch.ndim == 1 or next_lidar_batch.size == 0:
                next_lidar_batch = np.zeros((self.batch_size, self.lidar_dim), dtype=np.float32)
            
            # --- 转换为 PyTorch 张量 --- 
            indices = samples['indices']
            weights = torch.FloatTensor(samples['weights']).unsqueeze(-1).to(BaseConfig.DEVICE)
            actions = torch.LongTensor(samples['actions']).unsqueeze(-1).to(BaseConfig.DEVICE)
            rewards = torch.FloatTensor(samples['rewards']).unsqueeze(-1).to(BaseConfig.DEVICE)
            dones = torch.FloatTensor(samples['dones']).unsqueeze(-1).to(BaseConfig.DEVICE)
            
            # 当前状态批处理字典
            current_states_batch_dict = {
                'visual': torch.FloatTensor(visual_batch).to(BaseConfig.DEVICE),
                'lidar': torch.FloatTensor(lidar_batch).to(BaseConfig.DEVICE),
                'vector': torch.FloatTensor(vector_batch).to(BaseConfig.DEVICE)
            }
            
            # 下一状态批处理字典
            next_states_batch_dict = {
                'visual': torch.FloatTensor(next_visual_batch).to(BaseConfig.DEVICE),
                'lidar': torch.FloatTensor(next_lidar_batch).to(BaseConfig.DEVICE),
                'vector': torch.FloatTensor(next_vector_batch).to(BaseConfig.DEVICE)
            }

            # --- 计算 Q 值和损失 --- 
            # 当前 Q 值 (来自在线网络)
            q_values = self.model(current_states_batch_dict)
            current_q = q_values.gather(1, actions)
            
            # 使用目标网络计算下一步最大Q值 (使用Double DQN策略)
            with torch.no_grad():  # 使用torch.no_grad()避免梯度计算，节省内存
                # 1. 使用在线网络选择下一状态的最佳动作
                next_q_online = self.model(next_states_batch_dict)
                next_actions = next_q_online.max(1)[1].unsqueeze(-1)
                # 2. 使用目标网络评估这些动作的 Q 值
                next_q = self.target_model(next_states_batch_dict).gather(1, next_actions)
                # 计算目标 Q 值
                target = rewards + (1 - dones) * self.gamma * next_q

            # 计算 TD 误差并更新优先级
            td_error = torch.abs(target - current_q).cpu().detach().numpy()
            self.memory.update_priorities(indices, td_error.flatten() + 1e-6) # 添加小值避免零优先级

            # 计算损失 (使用重要性采样权重)
            elementwise_loss = self.criterion(current_q, target)
            loss = (weights * elementwise_loss).mean()

            # --- 反向传播 --- 
            self.optimizer.zero_grad()
            loss.backward()
            # 梯度裁剪，防止梯度爆炸
            nn.utils.clip_grad_norm_(self.model.parameters(), BaseConfig.GRAD_CLIP)
            self.optimizer.step()
            
            # 记录统计数据
            avg_q = q_values.mean().item()
            self.training_stats['avg_q_values'].append(avg_q)
            self.training_stats['td_errors'].append(np.mean(td_error))
            
            return loss.item(), avg_q
            
        except Exception as e:
            print(f"训练过程中发生错误: {e}")
            import traceback
            traceback.print_exc()
            return None, None

    def train_episode(self, episode_num, visualizer=None):
        """
        训练单个回合并返回结果
        :param episode_num: 当前回合编号
        :param visualizer: 可选的可视化器
        :return: 回合总奖励, 平均损失, 动作历史
        """
        state_dict = self.env.reset()  # 重置环境，获取完整状态字典
        total_reward = 0
        episode_losses = []
        episode_q_values = []
        actions_history = []
        steps_per_second = []
        episode_start_time = time.time()
        
        # 每回合前输出内存使用情况
        if torch.cuda.is_available():
            print(f"回合 {episode_num} - GPU内存: 已分配 {torch.cuda.memory_allocated()/1024**2:.1f}MB, 缓存 {torch.cuda.memory_reserved()/1024**2:.1f}MB")

        for step in range(BaseConfig.MAX_STEPS):
            step_start_time = time.time()
            
            # 选择动作
            action, q_value = self.act(state_dict)
            actions_history.append(action)
            episode_q_values.append(q_value)
            
            # 执行动作
            next_state_dict, reward, done, _ = self.env.step(action)
            
            # 如果有可视化器，更新可视化
            if visualizer is not None and hasattr(visualizer, 'update'):
                try:
                    visualizer.update(state_dict, action, reward, next_state_dict, done)
                except Exception as e:
                    print(f"可视化更新失败: {e}")
            
            # 存储经验
            self.remember(state_dict, action, reward, next_state_dict, done)
            total_reward += reward
            state_dict = next_state_dict
            self.train_step_counter += 1

            # 更新探索率
            self.update_epsilon()

            # 定期训练网络
            if self.train_step_counter % self.learning_steps_per_update == 0 and len(self.memory) >= self.batch_size:
                # 尝试进行训练，并处理可能的错误
                try:
                    loss, avg_q = self.replay(self.train_step_counter)
                    if loss is not None:
                        episode_losses.append(loss)
                        
                    # 每隔一段时间更新目标网络
                    if self.train_step_counter % (self.learning_steps_per_update * 10) == 0:
                        self.update_target_network()
                except Exception as e:
                    print(f"训练步骤出错: {e}")
                    # 继续执行，不中断整个训练过程
                    pass
            
            # 计算FPS
            if step % 20 == 0 and step > 0:
                current_time = time.time()
                elapsed = current_time - self.last_time
                if elapsed > 0:
                    steps = step - self.last_step
                    fps = steps / elapsed
                    self.fps_history.append(fps)
                    steps_per_second.append(fps)
                    self.last_time = current_time
                    self.last_step = step
                    print(f"回合 {episode_num}, 步骤 {step}: {fps:.1f} 步/秒, 探索率: {self.epsilon:.3f}")
               
            # 如果回合结束，跳出步骤循环
            if done:
                break

        # 回合结束，计算平均损失和性能指标
        avg_loss = np.mean(episode_losses) if episode_losses else None
        avg_q_value = np.mean(episode_q_values) if episode_q_values else 0
        avg_fps = np.mean(steps_per_second) if steps_per_second else 0
        episode_duration = time.time() - episode_start_time
        
        # 更新训练统计信息
        self.training_stats['episode_rewards'].append(total_reward)
        if avg_loss is not None:
            self.training_stats['episode_losses'].append(avg_loss)
        self.training_stats['training_times'].append(episode_duration)
        
        # 更新学习率调度器 - 根据奖励更新学习率
        self.scheduler.step(total_reward)
        
        # 输出回合摘要
        loss_str = f"{avg_loss:.4f}" if avg_loss is not None else "N/A"
        print(f"回合 {episode_num} 完成 - 奖励: {total_reward:.2f}, 平均损失: {loss_str}, "
              f"平均Q值: {avg_q_value:.2f}, 步数: {step+1}, 平均FPS: {avg_fps:.1f}, 总用时: {episode_duration:.1f}秒")
        
        # 清理PyTorch缓存，防止内存泄漏
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return total_reward, avg_loss, np.array(actions_history)
        
    def get_learning_rate(self):
        """获取当前学习率"""
        return self.optimizer.param_groups[0]['lr']

    def save_checkpoint(self, path):
        """
        保存模型检查点
        :param path: 保存路径
        """
        # 创建目录（如果不存在）
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epsilon': self.epsilon,
            'train_step_counter': self.train_step_counter,
            'training_stats': self.training_stats
        }, path)
        print(f"模型检查点已保存到 {path}")

    def load_checkpoint(self, path):
        """
        加载模型检查点
        :param path: 检查点路径
        """
        if not os.path.exists(path):
             print(f"检查点文件未找到: {path}")
             return False
             
        try:
            checkpoint = torch.load(path, map_location=BaseConfig.DEVICE)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # 加载目标模型（如果存在）
            if 'target_model_state_dict' in checkpoint:
                self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
            else:
                # 否则从主模型复制
                self.target_model.load_state_dict(self.model.state_dict())
                
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # 加载调度器（如果存在）
            if 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                
            self.epsilon = checkpoint.get('epsilon', BaseConfig.EPSILON_END)
            self.train_step_counter = checkpoint.get('train_step_counter', 0)
            
            # 加载训练统计信息（如果存在）
            if 'training_stats' in checkpoint:
                self.training_stats = checkpoint['training_stats']
                
            self.target_model.eval()
            print(f"检查点已从 {path} 加载，当前探索率: {self.epsilon:.4f}, 训练步数: {self.train_step_counter}")
            return True
        except Exception as e:
            print(f"加载检查点时出错: {e}")
            import traceback
            traceback.print_exc()
            return False


class SumTree:
    """SumTree数据结构实现，用于优先经验回放"""

    def __init__(self, capacity):
        """
        初始化SumTree
        :param capacity: 容量
        """
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)  # 存储优先级和的二叉树
        self.data = np.zeros(capacity, dtype=object)  # 存储数据
        self.n_entries = 0  # 当前条目数
        self.ptr = 0  # 数据指针

    def _propagate(self, idx, change):
        """从叶子节点向上传播优先级变化"""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        """从根节点向下检索样本"""
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        """返回总优先级"""
        return self.tree[0]

    def add(self, p, data):
        """添加样本"""
        idx = self.ptr + self.capacity - 1
        self.data[self.ptr] = data
        self.update(idx, p)
        self.ptr = (self.ptr + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, idx, p):
        """更新样本优先级"""
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        """获取样本"""
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]


class PrioritizedReplayBuffer:
    """优先经验回放缓冲区"""

    def __init__(self, capacity, alpha=0.6):
        """
        初始化缓冲区
        :param capacity: 容量
        :param alpha: 优先级指数(0-1)
        """
        self.tree = SumTree(capacity)
        self.alpha = alpha  # 控制优先级的程度(0表示均匀采样)
        self.capacity = capacity

    def _get_priority(self, error):
        """根据TD误差计算优先级"""
        return (np.abs(error) + 1e-6) ** self.alpha

    def max_priority(self):
        """返回当前缓冲区中的最大优先级"""
        if self.tree.n_entries == 0:
            return 1.0
        # SumTree 的优先级存储在 tree 数组的后半部分 (从 capacity - 1 开始)
        # 我们只需要检查已填充的部分
        leaf_start_index = self.tree.capacity - 1
        valid_priorities = self.tree.tree[leaf_start_index : leaf_start_index + self.tree.n_entries]
        return np.max(valid_priorities) if self.tree.n_entries > 0 else 1.0

    def add(self, priority, sample):
        """添加样本，使用传入的优先级"""
        # 注意：priority 参数现在由调用者 (remember) 提供
        self.tree.add(priority, sample)

    def sample(self, n, beta):
        """
        采样n个样本
        :param n: 样本数
        :param beta: 重要性采样权重参数
        :return: 采样结果字典
        """
        segment = self.tree.total() / n  # 将优先级总和分成n段
        priorities = []
        indices = []
        batch_tuples = [] # 存储原始的 (state_dict, action, ...) 元组

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            idx, p, data_tuple = self.tree.get(s)
            priorities.append(p)
            indices.append(idx)
            batch_tuples.append(data_tuple)

        sampling_probabilities = np.array(priorities) / self.tree.total()
        is_weights = (self.tree.n_entries * sampling_probabilities) ** -beta
        is_weights /= is_weights.max()

        # --- 解包批量数据 --- 
        # batch_tuples 是 [(s1_dict, a1, r1, ns1_dict, d1), (s2_dict, ...), ...]
        states, actions, rewards, next_states, dones = zip(*batch_tuples)
        # states 是 (s1_dict, s2_dict, ...)
        # next_states 是 (ns1_dict, ns2_dict, ...)
        
        return {
            'states': list(states), # 传递状态字典列表
            'actions': np.array(actions),
            'rewards': np.array(rewards),
            'next_states': list(next_states), # 传递下一状态字典列表
            'dones': np.array(dones, dtype=np.float32),
            'weights': is_weights,
            'indices': indices
        }

    def update_priorities(self, indices, td_errors):
        """更新样本优先级"""
        for idx, error in zip(indices, td_errors):
            priority = self._get_priority(error)
            self.tree.update(idx, priority)

    def __len__(self):
        """返回当前缓冲区大小"""
        return self.tree.n_entries