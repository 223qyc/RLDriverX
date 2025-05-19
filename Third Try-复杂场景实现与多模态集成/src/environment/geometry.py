import numpy as np
from typing import Tuple, Optional

def ray_circle_intersection(start: np.ndarray, end: np.ndarray, circle: np.ndarray) -> float:
    """
    计算射线与圆的交点
    
    参数:
        start: 射线起点 [x, y]
        end: 射线终点 [x, y]
        circle: 圆的信息 [x, y, radius]
    
    返回:
        射线起点到交点的距离，如果没有交点则返回射线长度
    """
    # 提取圆的参数
    circle_center = circle[:2]
    radius = circle[2]
    
    # 计算射线方向向量
    ray_dir = end - start
    ray_length = np.linalg.norm(ray_dir)
    ray_dir = ray_dir / ray_length
    
    # 计算射线起点到圆心的向量
    to_circle = circle_center - start
    
    # 计算射线方向向量与到圆心向量的点积
    proj = np.dot(to_circle, ray_dir)
    
    # 计算射线起点到圆心的垂直距离
    perp_dist = np.linalg.norm(to_circle - proj * ray_dir)
    
    # 如果垂直距离大于半径，则没有交点
    if perp_dist > radius:
        return ray_length
    
    # 计算射线起点到交点的距离
    dist_to_intersection = proj - np.sqrt(radius**2 - perp_dist**2)
    
    # 如果交点在射线起点之后，则没有交点
    if dist_to_intersection < 0:
        return ray_length
    
    # 如果交点在射线终点之后，则没有交点
    if dist_to_intersection > ray_length:
        return ray_length
    
    return dist_to_intersection

def check_car_obstacle_collision(car_corners: np.ndarray, obstacle: np.ndarray) -> bool:
    """
    检查小车与障碍物的碰撞
    
    参数:
        car_corners: 小车四个角点的坐标 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        obstacle: 障碍物信息 [x, y, radius]
    
    返回:
        是否发生碰撞
    """
    # 提取障碍物参数
    circle_center = obstacle[:2]
    radius = obstacle[2]
    
    # 检查每个角点是否在圆内
    for corner in car_corners:
        if np.linalg.norm(corner - circle_center) < radius:
            return True
    
    # 检查每条边是否与圆相交
    for i in range(4):
        start = car_corners[i]
        end = car_corners[(i + 1) % 4]
        
        # 计算边到圆心的距离
        edge_vec = end - start
        edge_length = np.linalg.norm(edge_vec)
        edge_dir = edge_vec / edge_length
        
        # 计算起点到圆心的向量
        to_circle = circle_center - start
        
        # 计算投影
        proj = np.dot(to_circle, edge_dir)
        proj = np.clip(proj, 0, edge_length)
        
        # 计算最近点
        closest_point = start + proj * edge_dir
        
        # 检查最近点到圆心的距离
        if np.linalg.norm(closest_point - circle_center) < radius:
            return True
    
    return False

def get_car_corners(position: np.ndarray, angle: float, length: float, width: float) -> np.ndarray:
    """
    计算小车四个角点的坐标
    
    参数:
        position: 小车中心位置 [x, y]
        angle: 小车朝向角度（弧度）
        length: 小车长度
        width: 小车宽度
    
    返回:
        四个角点的坐标 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    """
    # 计算半长和半宽
    half_length = length / 2
    half_width = width / 2
    
    # 计算四个角点的相对位置
    corners = np.array([
        [-half_length, -half_width],
        [half_length, -half_width],
        [half_length, half_width],
        [-half_length, half_width]
    ])
    
    # 创建旋转矩阵
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])
    
    # 旋转角点
    rotated_corners = np.dot(corners, rotation_matrix.T)
    
    # 平移角点
    return rotated_corners + position 