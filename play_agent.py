# v0.0.2
# Changelog:
# - 在畫面上加入分數顯示（Player vs AI）。
# - 新增累計 win/loss/timeout 紀錄，顯示在畫面左上角。
# - 其餘遊戲流程與 DQN 結構保持不變。

import sys

import pygame
import numpy as np
import torch
import torch.nn as nn

from env import PongEnv


# ---------- DQN 架構（要跟 train_dqn.py 裡的一樣） ----------

class DQN(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_trained_model(env: PongEnv, model_path: str = "dqn_pong_best.pt") -> DQN:
    """載入訓練好的 DQN 模型（如果失敗則嘗試載入 dqn_pong_last.pt）。"""
    state_dim = env.reset().shape[0]
    action_dim = env.action_space_n

    device = torch.device("cpu")  # 玩的時候用 CPU 就好
    model = DQN(state_dim, action_dim).to(device)

    try:
        state_dict = torch.load(model_path, map_location=device)
    except FileNotFoundError:
        print(f"[WARN] 找不到 {model_path}，改載入 dqn_pong_last.pt")
        state_dict = torch.load("dqn_pong_last.pt", map_location=device)

    model.load_state_dict(state_dict)
    model.eval()
    return model


def select_action_greedy(model: DQN, state: np.ndarray) -> int:
    """純 exploitation：選擇 Q 值最大的 action。"""
    with torch.no_grad():
        x = torch.from_numpy(state).float().unsqueeze(0)
        q_values = model(x)
        return int(q_values.argmax(dim=1).item())


# ---------- pygame 顯示部分 ----------

def main():
    # 初始化環境與模型
    env = PongEnv()
    model = load_trained_model(env)

    # 顯示設定（像素風：用小畫面放大）
    WINDOW_W, WINDOW_H = env.W, env.H
    SCALE = 4
    SCREEN_W, SCREEN_H = WINDOW_W * SCALE, WINDOW_H * SCALE

    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("Pong - DQN Agent")
    clock = pygame.time.Clock()

    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)

    font = pygame.font.SysFont("Courier", 8)

    running = True
    state = env.reset()
    episode_reward = 0.0

    # 分數與戰績紀錄
    player_score = 0
    ai_score = 0
    wins = 0
    losses = 0
    timeouts = 0

    while running:
        # --- 處理事件（關閉視窗 / ESC 退出） ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_ESCAPE]:
            running = False

        # --- 用模型選擇動作（agent 控制左邊板子） ---
        action = select_action_greedy(model, state)

        # --- 與環境互動 ---
        next_state, reward, done, info = env.step(action)
        episode_reward += reward

        # --- 繪圖：先畫在小畫面，再放大 ---
        small_surface = pygame.Surface((WINDOW_W, WINDOW_H))
        small_surface.fill(BLACK)

        # 左側 agent 板子
        pygame.draw.rect(
            small_surface,
            WHITE,
            (env.player_x, env.player_y, env.PADDLE_W, env.PADDLE_H),
        )
        # 右側 AI 板子
        pygame.draw.rect(
            small_surface,
            WHITE,
            (env.ai_x, env.ai_y, env.PADDLE_W, env.PADDLE_H),
        )
        # 球
        pygame.draw.rect(
            small_surface,
            WHITE,
            (env.ball_x, env.ball_y, env.BALL_SIZE, env.BALL_SIZE),
        )

        # 顯示簡單資訊（本局 reward、累計分數、戰績）
        info_text1 = f"EpReward: {episode_reward:7.3f}"
        info_text2 = f"Score P:{player_score} - AI:{ai_score}"
        info_text3 = f"W/L/T: {wins}/{losses}/{timeouts}"

        text1 = font.render(info_text1, True, WHITE)
        text2 = font.render(info_text2, True, WHITE)
        text3 = font.render(info_text3, True, WHITE)

        small_surface.blit(text1, (2, 2))
        small_surface.blit(text2, (2, 12))
        small_surface.blit(text3, (2, 22))

        # 放大到實際視窗
        pygame.transform.scale(small_surface, (SCREEN_W, SCREEN_H), screen)
        pygame.display.flip()

        # 如果一局結束，更新分數與戰績，稍微停一下再重開下一局
        if done:
            result = info.get("result", "timeout")
            if result == "win":
                player_score += 1
                wins += 1
            elif result == "lose":
                ai_score += 1
                losses += 1
            else:
                timeouts += 1

            print(
                f"Episode finished. total_reward = {episode_reward:.3f}, "
                f"result = {result}, "
                f"Score P:{player_score} - AI:{ai_score}, "
                f"W/L/T={wins}/{losses}/{timeouts}"
            )

            pygame.time.delay(500)  # 停 0.5 秒
            state = env.reset()
            episode_reward = 0.0
        else:
            state = next_state

        clock.tick(60)  # 限制在 60 FPS

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
