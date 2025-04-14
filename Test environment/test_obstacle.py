import pygame
import numpy as np
import sys
import os
from base.config import BaseConfig
from base.environment import Obstacle,DynamicObstacle


def main():
    # 初始化pygame
    pygame.init()

    # 设置窗口尺寸
    screen_width, screen_height = BaseConfig.ENV_WIDTH, BaseConfig.ENV_HEIGHT
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Obstacle Test")

    # 创建静态和动态障碍物
    obstacles = [Obstacle() for _ in range(5)]  # 5个静态障碍物
    dynamic_obstacles = [DynamicObstacle() for _ in range(3)]  # 3个动态障碍物

    clock = pygame.time.Clock()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # 更新动态障碍物
        for obstacle in dynamic_obstacles:
            obstacle.update()

        # 填充背景
        screen.fill((255, 255, 255))

        # 绘制所有障碍物
        for obstacle in obstacles:
            obstacle.draw(screen)

        for dynamic_obstacle in dynamic_obstacles:
            dynamic_obstacle.draw(screen)

        # 更新屏幕显示
        pygame.display.flip()

        # 设置帧率
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
