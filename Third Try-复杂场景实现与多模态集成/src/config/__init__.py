"""
配置模块初始化文件
导出所有配置常量和clone_configs函数
"""

from .environment_config import (
    AGENT_CONFIG,
    CURRICULUM_CONFIG,
    ENV_CONFIG,
    EVAL_CONFIG,
    OBSTACLE_CONFIG,
    RENDER_CONFIG,
    REWARD_CONFIG,
    TARGET_CONFIG,
    TRAINING_CONFIG,
    clone_configs,
)

__all__ = [
    "AGENT_CONFIG",
    "CURRICULUM_CONFIG",
    "ENV_CONFIG",
    "EVAL_CONFIG",
    "OBSTACLE_CONFIG",
    "RENDER_CONFIG",
    "REWARD_CONFIG",
    "TARGET_CONFIG",
    "TRAINING_CONFIG",
    "clone_configs",
]