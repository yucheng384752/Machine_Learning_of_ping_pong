import pygame
import random

pygame.init()

WINDOW_W, WINDOW_H = 160, 120
SCALE = 4
SCREEN_W, SCREEN_H = WINDOW_W * SCALE, WINDOW_H * SCALE

screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
clock = pygame.time.Clock()

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

PADDLE_W, PADDLE_H = 4, 20
BALL_SIZE = 3
PADDLE_SPEED = 2
BALL_SPEED = 1.5


def main():
    # 玩家板子在左邊
    player_x = 10
    player_y = WINDOW_H // 2 - PADDLE_H // 2

    # AI 板子在右邊
    ai_x = WINDOW_W - 10 - PADDLE_W
    ai_y = WINDOW_H // 2 - PADDLE_H // 2

    ball_x = WINDOW_W // 2
    ball_y = WINDOW_H // 2
    ball_vx = random.choice([-BALL_SPEED, BALL_SPEED])
    ball_vy = random.choice([-BALL_SPEED, BALL_SPEED])

    running = True
    while running:
        dt = clock.tick(60)  # 固定 60FPS

        # 事件處理
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # 玩家輸入（上 / 下）
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            player_y -= PADDLE_SPEED
        if keys[pygame.K_DOWN]:
            player_y += PADDLE_SPEED

        player_y = max(0, min(WINDOW_H - PADDLE_H, player_y))

        # 簡單 AI：盯著球 y
        if ball_y < ai_y:
            ai_y -= PADDLE_SPEED
        elif ball_y > ai_y + PADDLE_H:
            ai_y += PADDLE_SPEED
        ai_y = max(0, min(WINDOW_H - PADDLE_H, ai_y))

        # 球移動
        ball_x += ball_vx
        ball_y += ball_vy

        # 碰到上下邊界反彈
        if ball_y <= 0 or ball_y >= WINDOW_H - BALL_SIZE:
            ball_vy *= -1

        # 碰板子
        # 玩家板
        if (player_x < ball_x < player_x + PADDLE_W and
            player_y < ball_y < player_y + PADDLE_H):
            ball_vx *= -1

        # AI 板
        if (ai_x < ball_x + BALL_SIZE < ai_x + PADDLE_W and
            ai_y < ball_y < ai_y + PADDLE_H):
            ball_vx *= -1

        # 出界就重置（之後可以加分數）
        if ball_x < 0 or ball_x > WINDOW_W:
            ball_x = WINDOW_W // 2
            ball_y = WINDOW_H // 2
            ball_vx = random.choice([-BALL_SPEED, BALL_SPEED])
            ball_vy = random.choice([-BALL_SPEED, BALL_SPEED])

        # 繪圖（先畫在小畫面，再放大）
        small_surface = pygame.Surface((WINDOW_W, WINDOW_H))
        small_surface.fill(BLACK)

        # 球 & 板子
        pygame.draw.rect(small_surface, WHITE,
                         (player_x, player_y, PADDLE_W, PADDLE_H))
        pygame.draw.rect(small_surface, WHITE,
                         (ai_x, ai_y, PADDLE_W, PADDLE_H))
        pygame.draw.rect(small_surface, WHITE,
                         (ball_x, ball_y, BALL_SIZE, BALL_SIZE))

        # 放大
        pygame.transform.scale(small_surface, (SCREEN_W, SCREEN_H), screen)
        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()
