import pygame
from base.environment import Target
from base.config import BaseConfig

def main():
    pygame.init()
    screen = pygame.display.set_mode((BaseConfig.ENV_WIDTH, BaseConfig.ENV_HEIGHT))
    pygame.display.set_caption("Target Test")
    clock = pygame.time.Clock()

    target = Target()

    running = True
    while running:
        screen.fill((230, 230, 230))  # 背景浅灰

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        target.update()
        target.draw(screen)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()
