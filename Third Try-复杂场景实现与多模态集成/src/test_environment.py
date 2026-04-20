"""
第三阶段环境的简单冒烟测试
用于验证环境的基本功能是否正常工作

测试内容包括：
- 环境初始化和重置
- 观测空间形状验证
- 随机动作执行
- 基本输出验证
"""

import argparse
import os
import sys

import cv2
import numpy as np

# 设置项目路径
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.environment.environment import CarEnvironment


def test_environment(num_steps: int = 100, curriculum_stage: int = 0, show_window: bool = False) -> None:
    """
    测试环境基本功能

    参数:
        num_steps: 测试步数
        curriculum_stage: 课程学习阶段
        show_window: 是否显示实时渲染窗口
    """
    # 初始化环境
    env = CarEnvironment()
    observation, info = env.reset(options={"curriculum_stage": curriculum_stage})

    # 打印观测空间信息
    print("初始观测形状:")
    print(f"  radar:  {observation['radar'].shape}")    # 雷达数据形状
    print(f"  vector: {observation['vector'].shape}")   # 向量状态形状
    print(f"  visual: {observation['visual'].shape}")   # 视觉输入形状
    print(f"初始距离: {info['distance_to_target']:.2f}")

    # 执行随机动作测试
    for step in range(num_steps):
        # 生成随机动作：油门[0,1]，转向[-1,1]
        action = np.array(
            [
                np.random.uniform(0.0, 1.0),   # 随机油门
                np.random.uniform(-1.0, 1.0),  # 随机转向
            ],
            dtype=np.float32,
        )

        # 执行动作
        observation, reward, terminated, truncated, info = env.step(action)

        # 打印每步信息
        print(
            f"step={step + 1:03d} reward={reward:7.3f} "
            f"distance={info['distance_to_target']:7.2f} success={info['target_reached']} collision={info['collision']}"
        )

        # 如果需要显示窗口，渲染并显示
        if show_window:
            frame = env.render({"episode_reward": 0.0})
            cv2.imshow("Environment", frame)
            cv2.waitKey(30)

        # 如果回合结束，退出测试
        if terminated or truncated:
            break

    # 关闭显示窗口
    if show_window:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="第三阶段环境冒烟测试")
    parser.add_argument("--num_steps", type=int, default=100)
    parser.add_argument("--curriculum_stage", type=int, default=0)
    parser.add_argument("--show_window", action="store_true")
    args = parser.parse_args()

    # 执行测试
    test_environment(args.num_steps, args.curriculum_stage, args.show_window)