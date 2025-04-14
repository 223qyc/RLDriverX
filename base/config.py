"""
基础配置模块
"""
import os
import torch
from dataclasses import dataclass
from typing import Tuple

@dataclass
class BaseConfig:
    # 训练参数
    NUM_EPISODES: int =3  # 减少训练回合数，避免长时间训练卡住
    MAX_STEPS: int = 400  # 每回合最大步数
    ACTION_DIM: int = 4  # 动作空间维度
    BATCH_SIZE: int = 64  # 批次大小
    GAMMA: float = 0.98  # 折扣因子
    LR: float = 0.0005  # 学习率


    DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 训练设备
    REPLAY_BUFFER_SIZE: int = 10000  # 减小经验回放缓冲区大小，减少内存占用
    EPSILON_START: float = 1.0  # 初始探索率
    EPSILON_END: float = 0.05  # 提高最终探索率，增加探索
    EPSILON_DECAY_STEPS: int = 20000  # 增加探索率衰减步数，使探索衰减更慢
    TAU: float = 0.005  # 增加目标网络软更新参数，加快目标网络更新
    LEARNING_STEPS_PER_UPDATE: int = 8  # 增加每更新一次目标网络的学习步数
    GRAD_CLIP: float = 5.0  # 降低梯度裁剪阈值，防止梯度爆炸
    
    # 路径配置
    MODEL_SAVE_PATH: str = os.path.join('models', 'base')
    VIDEO_SAVE_PATH: str = os.path.join('videos', 'base')
    PLOT_SAVE_PATH: str = os.path.join('plots', 'base')
    METRICS_SAVE_PATH: str = os.path.join('metrics', 'base')
    
    # 环境参数
    ENV_WIDTH: int = 1000
    ENV_HEIGHT: int = 800
    
    # 车辆参数
    CAR_LENGTH: int = 35  # 车辆长度
    CAR_WIDTH: int = 16   # 车辆宽度
    CAR_MAX_SPEED: float = 10.0  # 最大速度
    CAR_MIN_SPEED: float = -2.0  # 最小速度（倒车）
    CAR_ACCELERATION: float = 2  # 加速度
    CAR_DECELERATION: float = 1.0  # 减速度
    CAR_MAX_STEERING: float = 30.0  # 最大转向角度
    CAR_STEERING_SPEED: float = 3.0  # 转向速度
    
    # 静态障碍物参数
    STATIC_OBSTACLES_COUNT: int = 20  # 减少静态障碍物数量
    STATIC_OBSTACLE_SIZE: int = 14  # 静态障碍物大小
    
    # 动态障碍物参数
    DYNAMIC_OBSTACLES_COUNT: int = 5  # 减少动态障碍物数量
    DYNAMIC_OBSTACLE_MIN_SIZE: int = 15  # 动态障碍物最小尺寸
    DYNAMIC_OBSTACLE_MAX_SIZE: int = 30  # 动态障碍物最大尺寸
    DYNAMIC_OBSTACLE_MIN_SPEED: float = -2.0  # 降低动态障碍物最小速度
    DYNAMIC_OBSTACLE_MAX_SPEED: float = 2.0  # 降低动态障碍物最大速度
    DYNAMIC_OBSTACLE_CHANGE_DIR_PROB: float = 0.01  # 动态障碍物随机改变方向的概率

    # 目标参数
    TARGET_MOVING: bool = True  # 目标是否移动
    TARGET_MIN_SPEED: float = 0.5  # 目标最小速度
    TARGET_MAX_SPEED: float = 2.0  # 目标最大速度
    TARGET_SIZE: int = 20  # 目标大小
    TARGET_CHANGE_DIR_PROB: float = 0.02  # 目标随机改变方向的概率
    TARGET_RANGE_X: tuple = (int(ENV_WIDTH * 0.6), ENV_WIDTH)  # 目标X坐标范围
    TARGET_RANGE_Y: tuple = (int(ENV_HEIGHT * 0.2), int(ENV_HEIGHT * 0.8))  # 目标Y坐标范围
    
    # 可视化参数
    RENDER_FPS: int = 60  # 帧率
    # PLOT_RESOLUTION: tuple = (1000, 800) # 绘图分辨率似乎与EvaluationVisualizer相关，暂且保留

    # 多模态输入配置
    VISUAL_RESIZE_DIM: Tuple[int, int] = (128, 128)  # 视觉输入分辨率
    VISUAL_INPUT_CHANNELS: int = 3 # (RGB)，灰度则额外做出调整

    # 课程学习相关
    INITIAL_DIFFICULTY: float = 0.1
    DIFFICULTY_STEP: float = 0.05
    MAX_DIFFICULTY: float = 1.0