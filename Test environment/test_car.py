import pygame
from base.environment import Car  # 假设你把前面的类定义放在了 car.py 里
from base.config import BaseConfig

def main():
    pygame.init()
    screen = pygame.display.set_mode((BaseConfig.ENV_WIDTH, BaseConfig.ENV_HEIGHT))
    pygame.display.set_caption("RL Car Simulation")
    clock = pygame.time.Clock()

    # 实例化小车
    car = Car()

    running = True
    while running:
        screen.fill((200, 200, 200))  # 灰色背景

        # 处理事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # 获取按键控制小车（用于测试）
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            car.update(0)  # 左转
        elif keys[pygame.K_RIGHT]:
            car.update(1)  # 右转
        elif keys[pygame.K_UP]:
            car.update(2)  # 加速
        elif keys[pygame.K_DOWN]:
            car.update(3)  # 减速
        else:
            car.update(-1)  # 空动作不更新（你可以扩展为惩罚）

        # 绘制小车
        car.draw(screen)

        pygame.display.flip()
        clock.tick(100)  # 每秒30帧

    pygame.quit()

if __name__ == "__main__":
    main()
