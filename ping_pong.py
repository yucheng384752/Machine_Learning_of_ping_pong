import pygame
import random

# 是否開啟困難模式（True 有障礙物、False 無）
HARD_MODE = True

# --- 初始化 pygame ---
pygame.init()

# --- 遊戲常數（照題目規格） ---
SCREEN_W, SCREEN_H = 200, 500

PADDLE_W, PADDLE_H = 40, 10
BALL_SIZE = 10

OBSTACLE_W, OBSTACLE_H = 30, 20

PADDLE_SPEED = 5
BALL_BASE_SPEED = 7
BALL_SPEED_UP_INTERVAL = 100      # 每 100 frame 速度 +1
SERVE_TIMEOUT_FRAMES = 150        # 150 frame 內沒發球就自動發

OBSTACLE_SPEED = 5

# 顏色
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED   = (255,   0,   0)
BLUE  = (0,   128, 255)
GREEN = (0,   255,   0)
YELLOW = (255, 255,   0)

# --- 視窗 ---
screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
pygame.display.set_caption("Two-Player Ping Pong (Spec Version)")
clock = pygame.time.Clock()

# 字型（用來顯示分數等）
font = pygame.font.SysFont("Courier", 16)


def create_obstacle():
    """依規格產生一個障礙物（只在困難模式用）。"""
    # x: 0~180 之間，每 20 一格
    possible_x = list(range(0, 181, 20))
    x = random.choice(possible_x)
    y = 240
    # 初始方向隨機：向左或向右
    dir_x = random.choice([-1, 1])
    return x, y, dir_x


def main():
    # --- 1P / 2P 板子初始位置 ---
    p1_x = 80
    p1_y = 420

    p2_x = 80
    p2_y = 70

    # 板子移動方向（-1:左, 0:不動, 1:右），用於切球機制
    p1_move_dir = 0
    p2_move_dir = 0

    # 分數與發球權
    p1_score = 0
    p2_score = 0

    server = 1  # 1P 先發
    ball_active = False  # False 代表目前在「準備發球」狀態
    frames_since_serve = 0

    # 球的位置 & 速度
    ball_x = 0
    ball_y = 0
    ball_vx = 0
    ball_vy = 0

    # 初始化第一次發球位置（先放在 1P 板子上）
    def reset_ball_on_paddle(current_server: int):
        nonlocal ball_x, ball_y, ball_vx, ball_vy, ball_active, frames_since_serve
        ball_active = False
        frames_since_serve = 0
        if current_server == 1:
            # 放在 1P 板子上方中央
            ball_x = p1_x + (PADDLE_W - BALL_SIZE) / 2
            ball_y = p1_y - BALL_SIZE
        else:
            # 放在 2P 板子下方中央
            ball_x = p2_x + (PADDLE_W - BALL_SIZE) / 2
            ball_y = p2_y + PADDLE_H
        ball_vx = 0
        ball_vy = 0

    reset_ball_on_paddle(server)

    # --- 困難模式：障礙物 ---
    if HARD_MODE:
        obs_x, obs_y, obs_dir = create_obstacle()
    else:
        obs_x = obs_y = obs_dir = None  # 只為了型別安全

    running = True
    while running:
        dt = clock.tick(60)  # 固定 60 FPS

        # ---------- 處理事件 ----------
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()

        # ESC 離開
        if keys[pygame.K_ESCAPE]:
            running = False

        # ---------- 玩家輸入：更新板子位置 ----------
        # 重置移動方向
        p1_move_dir = 0
        p2_move_dir = 0

        # 1P：左右鍵移動
        if keys[pygame.K_LEFT]:
            p1_x -= PADDLE_SPEED
            p1_move_dir = -1
        elif keys[pygame.K_RIGHT]:
            p1_x += PADDLE_SPEED
            p1_move_dir = 1

        # 2P：A / D 移動
        if keys[pygame.K_a]:
            p2_x -= PADDLE_SPEED
            p2_move_dir = -1
        elif keys[pygame.K_d]:
            p2_x += PADDLE_SPEED
            p2_move_dir = 1

        # 限制板子在畫面內
        p1_x = max(0, min(SCREEN_W - PADDLE_W, p1_x))
        p2_x = max(0, min(SCREEN_W - PADDLE_W, p2_x))

        # ---------- 發球邏輯 ----------
        if not ball_active:
            frames_since_serve += 1

            # 球跟著發球方的板子移動
            if server == 1:
                ball_x = p1_x + (PADDLE_W - BALL_SIZE) / 2
                ball_y = p1_y - BALL_SIZE
                # dot-key / dash-key 控制左 / 右發球
                if keys[pygame.K_PERIOD]:  # '.'
                    ball_vx = -BALL_BASE_SPEED
                    ball_vy = -BALL_BASE_SPEED
                    ball_active = True
                    frames_since_serve = 0
                elif keys[pygame.K_MINUS]:  # '-'
                    ball_vx = BALL_BASE_SPEED
                    ball_vy = -BALL_BASE_SPEED
                    ball_active = True
                    frames_since_serve = 0
            else:
                ball_x = p2_x + (PADDLE_W - BALL_SIZE) / 2
                ball_y = p2_y + PADDLE_H
                # q-key / e-key 控制左 / 右發球
                if keys[pygame.K_q]:
                    ball_vx = -BALL_BASE_SPEED
                    ball_vy = BALL_BASE_SPEED
                    ball_active = True
                    frames_since_serve = 0
                elif keys[pygame.K_e]:
                    ball_vx = BALL_BASE_SPEED
                    ball_vy = BALL_BASE_SPEED
                    ball_active = True
                    frames_since_serve = 0

            # 若超過 SERVE_TIMEOUT_FRAMES 還沒發，就隨機選一邊自動發球
            if not ball_active and frames_since_serve >= SERVE_TIMEOUT_FRAMES:
                dir_x = random.choice([-1, 1])
                if server == 1:
                    ball_vx = BALL_BASE_SPEED * dir_x
                    ball_vy = -BALL_BASE_SPEED
                else:
                    ball_vx = BALL_BASE_SPEED * dir_x
                    ball_vy = BALL_BASE_SPEED
                ball_active = True
                frames_since_serve = 0

        else:
            # ---------- 球在場上：移動 + 加速 ----------
            frames_since_serve += 1

            # 每 BALL_SPEED_UP_INTERVAL frame 速度 +1
            if frames_since_serve % BALL_SPEED_UP_INTERVAL == 0:
                if ball_vx > 0:
                    ball_vx += 1
                else:
                    ball_vx -= 1
                if ball_vy > 0:
                    ball_vy += 1
                else:
                    ball_vy -= 1

            ball_x += ball_vx
            ball_y += ball_vy

            # 左右牆反彈（只改 X 方向）
            if ball_x <= 0:
                ball_x = 0
                ball_vx *= -1
            elif ball_x + BALL_SIZE >= SCREEN_W:
                ball_x = SCREEN_W - BALL_SIZE
                ball_vx *= -1

            # ---------- 與板子碰撞 ----------
            ball_rect = pygame.Rect(ball_x, ball_y, BALL_SIZE, BALL_SIZE)
            p1_rect = pygame.Rect(p1_x, p1_y, PADDLE_W, PADDLE_H)
            p2_rect = pygame.Rect(p2_x, p2_y, PADDLE_W, PADDLE_H)

            # 1P 在下方，球往下撞到時反彈
            if ball_rect.colliderect(p1_rect) and ball_vy > 0:
                # 先把球拉出板子外，避免卡住
                ball_y = p1_y - BALL_SIZE
                ball_vy *= -1  # 垂直反彈

                # 切球機制（根據板子移動方向 vs 球 vx）
                if p1_move_dir == 0:
                    # 板子沒動，vx 不變
                    pass
                else:
                    if (p1_move_dir > 0 and ball_vx > 0) or (p1_move_dir < 0 and ball_vx < 0):
                        # 同向：vx 速度增加 3
                        if ball_vx > 0:
                            ball_vx += 3
                        else:
                            ball_vx -= 3
                    else:
                        # 反向：vx 反向，速度大小維持
                        ball_vx *= -1

            # 2P 在上方，球往上撞到時反彈
            if ball_rect.colliderect(p2_rect) and ball_vy < 0:
                ball_y = p2_y + PADDLE_H
                ball_vy *= -1

                if p2_move_dir == 0:
                    pass
                else:
                    if (p2_move_dir > 0 and ball_vx > 0) or (p2_move_dir < 0 and ball_vx < 0):
                        if ball_vx > 0:
                            ball_vx += 3
                        else:
                            ball_vx -= 3
                    else:
                        ball_vx *= -1

            # ---------- 障礙物（困難模式） ----------
            if HARD_MODE:
                obs_x += obs_dir * OBSTACLE_SPEED

                # 左右牆反彈
                if obs_x <= 0:
                    obs_x = 0
                    obs_dir = 1
                elif obs_x + OBSTACLE_W >= SCREEN_W:
                    obs_x = SCREEN_W - OBSTACLE_W
                    obs_dir = -1

                obs_rect = pygame.Rect(obs_x, obs_y, OBSTACLE_W, OBSTACLE_H)

                if ball_rect.colliderect(obs_rect):
                    # 題目說「保持球的速度」，所以只反彈方向、不改大小
                    # 這裡簡單做：優先改垂直方向
                    if ball_vy > 0:
                        ball_y = obs_y - BALL_SIZE
                    else:
                        ball_y = obs_y + OBSTACLE_H
                    ball_vy *= -1

            # ---------- 出界 & 得分 ----------
            # 球從下邊界出去：2P 得分
            if ball_y > SCREEN_H:
                p2_score += 1
                server = 2  # 下一球由 2P 發
                reset_ball_on_paddle(server)

            # 球從上邊界出去：1P 得分
            elif ball_y + BALL_SIZE < 0:
                p1_score += 1
                server = 1
                reset_ball_on_paddle(server)

        # ---------- 繪圖 ----------
        screen.fill(BLACK)

        # 畫板子
        pygame.draw.rect(screen, RED, (p1_x, p1_y, PADDLE_W, PADDLE_H))
        pygame.draw.rect(screen, BLUE, (p2_x, p2_y, PADDLE_W, PADDLE_H))

        # 畫球
        pygame.draw.rect(screen, GREEN, (ball_x, ball_y, BALL_SIZE, BALL_SIZE))

        # 畫障礙物
        if HARD_MODE:
            pygame.draw.rect(screen, YELLOW, (obs_x, obs_y, OBSTACLE_W, OBSTACLE_H))

        # 顯示分數與發球方
        score_text = font.render(
            f"P1: {p1_score}   P2: {p2_score}", True, WHITE
        )
        server_text = font.render(
            f"Server: {'P1' if server == 1 else 'P2'} ({'Hard' if HARD_MODE else 'Normal'})",
            True,
            WHITE,
        )
        screen.blit(score_text, (10, 10))
        screen.blit(server_text, (10, 30))

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
