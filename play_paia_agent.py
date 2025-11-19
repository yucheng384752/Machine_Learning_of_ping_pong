import pygame
import numpy as np
import torch
import torch.nn as nn

from env_paia import PongEnvPAIA
from train_dqn import DQN   # 共用你訓練時用的 DQN 結構


def load_trained_model(env: PongEnvPAIA, device: torch.device, model_path: str):
    """從 .pt 檔案載入已訓練模型，根據 env 自動抓 state_dim / action_dim。"""
    dummy_state = env.reset()
    state_dim = dummy_state.shape[0]
    action_dim = env.action_space_n

    model = DQN(state_dim, action_dim).to(device)

    # 建議使用 weights_only=True，比較安全
    state_dict = torch.load(
        model_path,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(state_dict)
    model.eval()

    print(f"Loaded model from {model_path}")
    print(f"State dim = {state_dim}, Action dim = {action_dim}")
    return model


def select_action_greedy(model: nn.Module, state: np.ndarray, device: torch.device) -> int:
    """不探索的 greedy 策略：直接選 Q 最大的 action。"""
    with torch.no_grad():
        x = torch.from_numpy(state).float().unsqueeze(0).to(device)
        q = model(x)
        return int(q.argmax(dim=1).item())


def main():
    # 1. 建立環境（先用 easy 模式，也可以改成 "hard" 看障礙物版本）
    env = PongEnvPAIA(mode="easy")

    # 2. 裝置（多半推論用 CPU 即可，有 GPU 也能自動用）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 3. 載入已訓練好的模型（請確認路徑與檔名正確）
    model = load_trained_model(env, device, "models/dqn_pong_best.pt")

    # 4. pygame 視覺化設定
    pygame.init()
    WINDOW_W, WINDOW_H = env.W, env.H
    SCALE = 2  # 放大倍率，避免畫面太小
    SCREEN_W, SCREEN_H = WINDOW_W * SCALE, WINDOW_H * SCALE

    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("PAIA Pong - DQN Agent vs Rule-based AI")
    clock = pygame.time.Clock()

    # 顏色
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    RED = (255, 80, 80)     # 1P
    BLUE = (80, 80, 255)    # 2P
    GREEN = (80, 255, 80)   # 球
    YELLOW = (255, 255, 80) # 障礙物

    font = pygame.font.SysFont("consolas", 16)

    # 5. 主回合 loop
    state = env.reset()
    running = True
    episode = 1
    win_count = 0
    lose_count = 0
    timeout_count = 0

    while running:
        # ---- 處理事件（關閉視窗 / ESC 離開） ----
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_ESCAPE]:
            running = False

        # ---- DQN 選動作 ----
        action = select_action_greedy(model, state, device)

        # ---- 與環境互動 ----
        next_state, reward, done, info = env.step(action)
        state = next_state

        # ---- 繪圖到小畫面 (原解析度) ----
        small = pygame.Surface((WINDOW_W, WINDOW_H))
        small.fill(BLACK)

        # 1P 板子（底部，紅色）
        pygame.draw.rect(
            small,
            RED,
            (env.p1_x, env.p1_y, env.PADDLE_W, env.PADDLE_H),
        )

        # 2P 板子（上方，藍色）
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

        # 障礙物（hard 模式才會動，easy 模式會在 (0,0) 但尺寸為 0）
        if env.use_obstacle:
            pygame.draw.rect(
                small,
                YELLOW,
                (env.obstacle_x, env.obstacle_y, env.obstacle_w, env.obstacle_h),
            )

        # ---- 放大顯示到主畫面 ----
        pygame.transform.scale(small, (SCREEN_W, SCREEN_H), screen)

        # ---- 顯示文字資訊（集數 / 戰績 / 模式）----
        info_text = (
            f"Ep: {episode}  "
            f"Mode: {env.mode}  "
            f"Wins: {win_count}  "
            f"Losses: {lose_count}  "
            f"Timeout: {timeout_count}"
        )
        text_surf = font.render(info_text, True, WHITE)
        screen.blit(text_surf, (10, 10))

        pygame.display.flip()
        clock.tick(60)  # 60 FPS

        # ---- 一局結束，統計結果 & 重置 ----
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
