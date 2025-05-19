"""
环境配置文件
"""

# 环境基本参数
ENV_CONFIG = {
    'width': 1024,         # 环境宽度
    'height': 768,         # 环境高度
    'car_length': 40,      # 小车长度
    'car_width': 20,       # 小车宽度
    'max_speed': 8.0,      # 最大速度
    'min_speed': -3.0,     # 最小速度（倒车）
    'max_steering': 0.8,   # 最大转向角度
    'steering_ratio': 0.2, # 转向比例系数
    'acceleration': 0.2,   # 加速度
    'deceleration': 0.3,   # 减速度
    'friction': 0.05,      # 摩擦系数
    'radar_rays': 16,      # 雷达射线数量
    'radar_length': 200,   # 雷达射线长度
    'start_position': [100, 668],  # 小车初始位置（左下角）
    'start_angle': 0.0,    # 小车初始角度
}

# 障碍物参数
OBSTACLE_CONFIG = {
    'static': {
        'count': 12,           # 静态障碍物数量
        'min_radius': 20,      # 最小半径
        'max_radius': 50,      # 最大半径
    },
    'dynamic': {
        'count': 8,            # 动态障碍物数量
        'min_radius': 15,      # 最小半径
        'max_radius': 30,      # 最大半径
        'min_speed': 1.0,      # 最小速度
        'max_speed': 3.0,      # 最大速度
        'direction_change_prob': 0.02,  # 方向改变概率
    }
}

# 目标参数
TARGET_CONFIG = {
    'radius': 15,              # 目标半径
    'move_probability': 0.02,  # 移动概率
    'move_speed': 2.0,         # 移动速度
    'start_position': [100, 100],  # 目标初始位置（左上角）
    'max_step_size': 20,       # 最大移动步长
    'smoothing_factor': 0.6,   # 移动平滑因子
}

# 奖励参数
REWARD_CONFIG = {
    'collision_penalty': -10.0,     # 碰撞惩罚
    'distance_factor': -0.01,       # 距离因子
    'target_reached_reward': 100.0,  # 到达目标奖励
    'target_threshold': 30.0,       # 到达目标阈值
    'distance_progress_factor': 0.3, # 靠近目标的奖励系数，增大到0.3激励向目标前进
    'rotation_penalty': -0.05,      # 原地旋转惩罚，增大到-0.05
    'speed_factor': 0.05,           # 保持速度奖励，增大到0.05鼓励移动
    'boundary_penalty': -5.0,       # 接近边界惩罚
    'boundary_threshold': 50,       # 边界阈值
    'step_penalty': -0.01,          # 步数惩罚
    'stagnation_penalty': -0.1,     # 无进展惩罚，增大到-0.1
    'movement_reward': 0.1,         # 新增：位移奖励，鼓励大幅移动
    'rotation_threshold': 5,        # 新增：旋转检测阈值，降低到5
}

# 渲染参数
RENDER_CONFIG = {
    'visual_size': (256, 256),     # 视觉输入大小
    'display_size': (1024, 768),   # 显示大小
    'background_color': (240, 240, 240),  # 背景颜色
    'static_color': (100, 100, 100),      # 静态障碍物颜色
    'dynamic_color': (150, 50, 50),       # 动态障碍物颜色
    'target_color': (0, 200, 0),          # 目标颜色
    'target_border_color': (0, 255, 0),   # 目标边框颜色
    'car_color': (0, 0, 255),             # 小车颜色
    'car_border_color': (50, 50, 200),    # 小车边框颜色
    'radar_color': (255, 255, 0),         # 雷达线颜色
    'trail_color': (0, 0, 150, 100),      # 小车轨迹颜色
    'text_color': (0, 0, 0),              # 文字颜色
    'show_trails': True,                  # 显示轨迹
    'max_trail_length': 100,              # 轨迹最大长度
    'grid_on': True,                      # 显示网格
    'grid_size': 50,                      # 网格大小
    'grid_color': (200, 200, 200),        # 网格颜色
} 