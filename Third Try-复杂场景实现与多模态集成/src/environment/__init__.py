"""
环境模块初始化文件
导出CarEnvironment类和几何计算函数
"""

from .environment import CarEnvironment
from .geometry import ray_circle_intersection, check_car_obstacle_collision, get_car_corners

__all__ = [
    'CarEnvironment',
    'ray_circle_intersection',
    'check_car_obstacle_collision',
    'get_car_corners'
]