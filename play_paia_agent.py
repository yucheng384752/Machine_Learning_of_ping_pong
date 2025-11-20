import pygame
import numpy as np
import torch
import torch.nn as nn

from train_dqn import DQN
from env_paia import PongEnvPAIA   # ✅ 使用 PAIA 版環境


def load_trained_model(env: PongEnvPAIA, device: torch.device, model_path: str):
    """從 .pt 檔案載入已訓練模型（根據 env 自動抓 state_dim / action_dim）"""
    dummy_state = env.reset()
    state_dim = dummy_state.shape[0]
    action_dim = env.action_space_n

    model = DQN(state_dim, action_dim).to(device)

    state_dict = torch.load(
        model_path,
        map_location=device,
        weights_only=True,  # 比較安全
    )
    model.load_state_dict(state_dict)
    model.eval()

    print(f"Loaded model from {model_path}")
    print(f"State dim = {state_dim}, Action dim = {action_dim}")
    return model


def select_action_greedy(model: nn.Module, state: np.ndarray, device: torch.device) -> int:
    """純 exploit：選 Q 值最大的 action。"""
    with torch.no_grad():
        x = torch.from_numpy(state).float().unsqueeze(0).to(device)
        q = model(x)
        return int(q.argmax(dim=1).item())


def main():
    # ✅ 使用 PAIA 版環境；先用 easy 模式
    env = PongEnvPAIA(mode="hard")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ✅ 載入你用 env_paia 訓練的模型
    model = load_trained_model(env, device, "models/dqn_pong_best.pt")

    # pygame 視覺化設定
    pygame.init()
    WINDOW_W, WINDOW_H = env.W, env.H
    SCALE = 2  # 放大倍率
    SCREEN_W, SCREEN_H = WINDOW_W * SCALE, WINDOW_H * SCALE

    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("PAIA Pong - DQN Agent (RED) vs Rule-based AI (BLUE)")
    clock = pygame.time.Clock()

    # 顏色
    BLACK = (0, 0, 0)
    RED = (255, 80, 80)     # ✅ 1P agent
    BLUE = (80, 80, 255)    # ✅ 2P AI
    GREEN = (80, 255, 80)   # 球
    YELLOW = (255, 255, 80) # 障礙物
    WHITE = (255, 255, 255)

    font = pygame.font.SysFont("consolas", 16)

    state = env.reset()
    running = True
    episode = 1
    win_count = 0
    lose_count = 0
    timeout_count = 0

    while running:
        # 處理離開事件
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_ESCAPE]:
            running = False

        # 讓 DQN 選動作（控制紅色 1P）
        action = select_action_greedy(model, state, device)

        # 與環境互動
        next_state, reward, done, info = env.step(action)
        state = next_state

        # 小畫面 (原始解析度)
        small = pygame.Surface((WINDOW_W, WINDOW_H))
        small.fill(BLACK)

        # ✅ 1P：底部紅色板子（agent）
        pygame.draw.rect(
            small,
            RED,
            (env.p1_x, env.p1_y, env.PADDLE_W, env.PADDLE_H),
        )

        # ✅ 2P：上方藍色板子（rule-based 對手）
        pygame.draw.rect(
            small,
            BLUE,
            (env.p2_x, env.p2_y, env.PADDLE_W, env.PADDLE_H),
        )

        # 球（綠色）
        pygame.draw.rect(
            small,
            GREEN,
            (env.ball_x, env.ball_y, env.BALL_SIZE, env.BALL_SIZE),
        )

        # 障礙物（hard 模式）
        if env.use_obstacle:
            pygame.draw.rect(
                small,
                YELLOW,
                (env.obstacle_x, env.obstacle_y, env.obstacle_w, env.obstacle_h),
            )

        # 放大到視窗
        pygame.transform.scale(small, (SCREEN_W, SCREEN_H), screen)

        # 顯示戰績資訊
        info_text = (
            f"Ep: {episode}  "
            f"Mode: {env.mode}  "
            f"Wins: {win_count}  "
            f"Losses: {lose_count}  "
            f"Timeouts: {timeout_count}"
        )
        text_surf = font.render(info_text, True, WHITE)
        screen.blit(text_surf, (10, 10))

        pygame.display.flip()
        clock.tick(60)

        # 一局結束，統計結果 & reset
        if done:
            result = info.get("result")
            if result == "win":
                win_count += 1
            elif result == "lose":
                lose_count += 1
            else:
                timeout_count += 1

            episode += 1
            state = env.reset()

    pygame.quit()


if __name__ == "__main__":
    main()
