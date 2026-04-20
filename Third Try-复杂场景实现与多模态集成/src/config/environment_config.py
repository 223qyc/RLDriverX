"""
第三阶段训练管道的配置中心
包含环境、智能体、训练、奖励等所有配置参数

配置组织方式：
- 各类配置使用独立的字典常量定义
- clone_configs()函数返回深拷贝，避免修改原始配置
"""

from copy import deepcopy


# ==================== 环境配置 ====================
# 定义物理环境的各项参数
ENV_CONFIG = {
    "width": 1024,              # 环境宽度（像素）
    "height": 768,              # 环境高度（像素）
    "car_length": 36.0,         # 小车长度
    "car_width": 20.0,          # 小车宽度
    "wheel_base": 28.0,         # 轴距（用于自行车运动学模型）
    "max_speed": 7.5,           # 最大前进速度
    "min_speed": -2.0,          # 最大后退速度（负值）
    "max_steering": 0.55,       # 最大转向角（弧度）
    "steering_smoothness": 0.18,    # 转向变化率限制
    "acceleration": 0.28,       # 加速度
    "brake_deceleration": 0.35, # 刹车减速度
    "friction": 0.04,           # 摩擦系数
    "radar_rays": 24,           # 雷达射线数量
    "radar_length": 220.0,      # 雷达探测距离
    "observation_size": (96, 96),   # 视觉观测尺寸
    "max_episode_steps": 320,   # 每回合最大步数
    "start_position": [120, 648],   # 小车初始位置
    "start_angle": -0.35,       # 小车初始朝向
}


# ==================== 障碍物配置 ====================
# 定义静态和动态障碍物的生成规则
OBSTACLE_CONFIG = {
    "static": {
        "min_radius": 20.0,     # 静态障碍物最小半径
        "max_radius": 42.0,     # 静态障碍物最大半径
        "safe_margin": 28.0,    # 障碍物间的安全间距
    },
    "dynamic": {
        "min_radius": 16.0,     # 动态障碍物最小半径
        "max_radius": 28.0,     # 动态障碍物最大半径
        "min_speed": 0.7,       # 动态障碍物最小移动速度
        "max_speed": 1.9,       # 动态障碍物最大移动速度
        "direction_change_prob": 0.015,  # 方向随机变化的概率
    },
}


# ==================== 目标配置 ====================
# 定义目标点的属性和移动规则
TARGET_CONFIG = {
    "radius": 18.0,             # 目标点半径
    "move_probability": 0.04,   # 目标移动概率（用于移动目标场景）
    "move_speed": 1.5,          # 目标移动速度
    "smoothing_factor": 0.85,   # 速度平滑系数
    "goal_margin": 90.0,        # 目标生成区域的边距
    "min_goal_distance": 360.0, # 目标与小车的最小初始距离
}


# ==================== 课程学习配置 ====================
# 定义训练难度的渐进计划
# 从简单场景逐步过渡到复杂的多模态场景
CURRICULUM_CONFIG = [
    {
        "name": "stage_0_basic",     # 第0阶段：基础训练
        "static_count": 4,           # 静态障碍物数量
        "dynamic_count": 0,          # 无动态障碍物
        "moving_target": False,      # 固定目标
        "fixed_goal": True,          # 固定目标位置
        "goal_position": [860.0, 160.0],  # 目标坐标
        "spawn_jitter": 0.0,         # 无随机扰动
    },
    {
        "name": "stage_1_random_goal",   # 第1阶段：随机目标
        "static_count": 8,
        "dynamic_count": 0,
        "moving_target": False,
        "fixed_goal": False,         # 目标位置随机
        "spawn_jitter": 40.0,        # 小车起始位置有扰动
    },
    {
        "name": "stage_2_dynamic_obstacles",  # 第2阶段：动态障碍物
        "static_count": 10,
        "dynamic_count": 4,          # 添加动态障碍物
        "moving_target": False,
        "fixed_goal": False,
        "spawn_jitter": 55.0,
    },
    {
        "name": "stage_3_multimodal_full",  # 第3阶段：完整多模态场景
        "static_count": 12,
        "dynamic_count": 6,
        "moving_target": True,       # 目标可移动
        "fixed_goal": False,
        "spawn_jitter": 65.0,
    },
]


# ==================== 奖励配置 ====================
# 定义奖励函数的各个组件权重
REWARD_CONFIG = {
    "collision_penalty": -50.0,              # 碰撞惩罚
    "target_reached_reward": 160.0,          # 成功到达目标的奖励
    "progress_reward_scale": 10.0,           # 进展奖励权重（靠近目标）
    "distance_shaping_scale": 1.0,           # 距离塑造权重
    "step_penalty": -0.03,                   # 每步时间惩罚
    "near_obstacle_penalty_scale": -0.45,    # 靠近障碍物惩罚权重
    "steering_penalty_scale": -0.015,        # 转向惩罚权重（鼓励平稳驾驶）
    "action_smoothness_penalty_scale": -0.02, # 动作平滑度惩罚权重
    "reverse_penalty_scale": -0.03,          # 倒车惩罚权重
    "stagnation_penalty": -0.35,             # 停滞惩罚
    "stagnation_steps": 22,                  # 判断停滞的步数阈值
    "success_time_bonus_scale": 0.08,        # 成功时间奖励（越快越好）
}


# ==================== 智能体配置 ====================
# 定义TD3智能体的核心参数
AGENT_CONFIG = {
    "learning_rate": 3e-4,       # 学习率
    "gamma": 0.99,               # 折扣因子（未来奖励的权重）
    "buffer_size": 150_000,      # 经验回放池容量
    "batch_size": 128,           # 训练批大小
    "tau": 0.005,                # 目标网络软更新系数
    "policy_noise": 0.18,        # 目标策略噪声幅度
    "noise_clip": 0.35,          # 噪声裁剪上限
    "exploration_noise": 0.12,   # 探索噪声幅度
    "policy_delay": 2,           # 策略更新延迟（TD3核心特性）
    "hidden_dim": 192,           # 网络隐藏层维度
    "use_visual": True,          # 是否使用视觉输入分支
}


# ==================== 训练配置 ====================
# 定义训练流程的参数
TRAINING_CONFIG = {
    "num_episodes": 900,         # 总训练回合数
    "max_steps": 320,            # 每回合最大步数
    "eval_interval": 25,         # 评估间隔回合数
    "eval_episodes": 6,          # 每次评估的回合数
    "save_interval": 100,        # 模型保存间隔回合数
    "video_interval": 150,       # 视频保存间隔回合数
    "warmup_steps": 3_000,       # 预热步数（随机探索阶段）
    "updates_per_step": 1,       # 每步执行的更新次数
    "curriculum_schedule": [     # 课程学习时间表
        {"episode": 0, "stage": 0},     # 从第0回合开始使用阶段0
        {"episode": 180, "stage": 1},   # 第180回合进入阶段1
        {"episode": 420, "stage": 2},   # 第420回合进入阶段2
        {"episode": 680, "stage": 3},   # 第680回合进入阶段3（最终难度）
    ],
}


# ==================== 评估配置 ====================
# 定义模型评估的参数
EVAL_CONFIG = {
    "episodes": 8,               # 评估回合数
    "max_steps": 320,            # 每回合最大步数
}


# ==================== 渲染配置 ====================
# 定义可视化渲染的样式参数
RENDER_CONFIG = {
    "visual_size": (96, 96),     # 视觉观测尺寸
    "display_size": (1024, 768), # 显示尺寸
    "background_color": (241, 244, 247),   # 背景颜色（浅灰）
    "grid_color": (218, 222, 227),         # 网格线颜色
    "static_color": (94, 107, 120),        # 静态障碍物颜色（深灰）
    "dynamic_color": (193, 96, 77),        # 动态障碍物颜色（橙红）
    "target_color": (49, 170, 112),        # 目标点颜色（绿色）
    "target_border_color": (23, 122, 72),  # 目标边框颜色
    "car_color": (48, 103, 224),           # 小车颜色（蓝色）
    "car_border_color": (25, 62, 145),     # 小车边框颜色
    "car_heading_color": (255, 255, 255),  # 小车方向指示颜色（白色）
    "radar_far_color": (70, 170, 130),     # 雷达远距离颜色（绿色）
    "radar_near_color": (214, 92, 92),     # 雷达近距离颜色（红色）
    "trail_color": (57, 88, 155),          # 轨迹颜色
    "text_color": (35, 39, 42),            # 文字颜色
    "panel_background": (255, 255, 255),   # 信息面板背景色
    "show_grid": True,           # 是否显示网格
    "show_trail": True,          # 是否显示轨迹
    "max_trail_length": 160,     # 最大轨迹长度
    "grid_size": 48,             # 网格单元大小
}


def clone_configs():
    """
    返回所有配置的深拷贝

    使用深拷贝确保调用者可以安全修改配置副本，
    而不会影响原始配置常量

    返回:
        包含所有配置副本的字典
    """
    return {
        "env_config": deepcopy(ENV_CONFIG),
        "obstacle_config": deepcopy(OBSTACLE_CONFIG),
        "target_config": deepcopy(TARGET_CONFIG),
        "curriculum_config": deepcopy(CURRICULUM_CONFIG),
        "reward_config": deepcopy(REWARD_CONFIG),
        "agent_config": deepcopy(AGENT_CONFIG),
        "training_config": deepcopy(TRAINING_CONFIG),
        "eval_config": deepcopy(EVAL_CONFIG),
        "render_config": deepcopy(RENDER_CONFIG),
    }