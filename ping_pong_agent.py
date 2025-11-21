import pygame
import numpy as np
import torch
import torch.nn as nn

from env import PongEnv
from train_dqn import DQN   # 共用訓練的網路結構
from env_paia import PongEnvPAIA

def load_trained_model(env: PongEnv, device: torch.device, model_path: str):
    """從 .pt 檔案載入已訓練模型"""
    # 自動取得 state_dim / action_dim
    dummy_state = env.reset()
    state_dim = dummy_state.shape[0]       # <-- 這裡改成動態抓取
    action_dim = env.action_space_n

    model = DQN(state_dim, action_dim).to(device)

    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    return model


def select_action_greedy(model: DQN, state: np.ndarray, device: torch.device) -> int:
    with torch.no_grad():
        x = torch.from_numpy(state).float().unsqueeze(0).to(device)
        q = model(x)
        return int(q.argmax(dim=1).item())


def main():
    # 選擇 easy/hard 模式
    env = PongEnv(mode="easy")  # or "hard"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using:", device)

    # 載入模型（請確保 best 模型存在 models/ 資料夾）
    model = load_trained_model(env, device, "models/dqn_pong_best.pt")

    # pygame 繪圖設定
    WINDOW_W, WINDOW_H = env.W, env.H
    SCALE = 4
    SCREEN_W, SCREEN_H = WINDOW_W * SCALE, WINDOW_H * SCALE

    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    clock = pygame.time.Clock()

    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    font = pygame.font.SysFont("Courier", 12)

    state = env.reset()
    running = True

    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False

        # ESC 離開
        if pygame.key.get_pressed()[pygame.K_ESCAPE]:
            running = False

        # 讓 DQN 選動作
        action = select_action_greedy(model, state, device)

        # 環境更新（右側 AI 自動動）
        next_state, reward, done, info = env.step(action)
        state = next_state

        # --- 繪圖 ---
        small = pygame.Surface((WINDOW_W, WINDOW_H))
        small.fill(BLACK)

        # 玩家板子（左）
        pygame.draw.rect(small, WHITE,
                         (env.player_x, env.player_y, env.PADDLE_W, env.PADDLE_H))

        # AI 板子（右）
        pygame.draw.rect(small, WHITE,
                         (env.ai_x, env.ai_y, env.PADDLE_W, env.PADDLE_H))

        # 球
        pygame.draw.rect(small, WHITE,
                         (env.ball_x, env.ball_y, env.BALL_SIZE, env.BALL_SIZE))

        # 障礙物（hard 模式才有）
        if env.use_obstacle:
            pygame.draw.rect(small, WHITE,
                             (env.obstacle_x, env.obstacle_y,
                              env.obstacle_w, env.obstacle_h))

        # 放大到畫面
        pygame.transform.scale(small, (SCREEN_W, SCREEN_H), screen)
        pygame.display.flip()

        clock.tick(60)

        if done:
            state = env.reset()

    pygame.quit()


if __name__ == "__main__":
    main()
