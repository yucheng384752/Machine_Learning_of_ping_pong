# v0.1.0
# Changelog:
# - 統一整理 DQN 訓練流程（保留原本 hyperparameters 精神）
# - 加入 GPU / CPU 自動偵測與輸出
# - 自動建立 models/ 資料夾，統一儲存 best / last 權重
# - 保留 epsilon-greedy + replay buffer + target network 架構

import os
import random
from collections import deque
from typing import Deque, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from env import PongEnv
from env_paia import PongEnvPAIA


# --------- DQN 網路定義 ---------
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


# --------- epsilon-greedy 動作選擇 ---------
def select_action(
    policy_net: DQN,
    state: np.ndarray,
    action_dim: int,
    epsilon: float,
    device: torch.device,
) -> int:
    """epsilon-greedy 選擇動作。"""
    # 探索
    if random.random() < epsilon:
        return random.randrange(action_dim)

    # 利用：選 Q 最大的動作
    state_t = torch.from_numpy(state).float().unsqueeze(0).to(device)
    with torch.no_grad():
        q_values = policy_net(state_t)
    return int(q_values.argmax(dim=1).item())


# --------- 主訓練流程 ---------
def train():
    # 讓 models/ 一定存在
    os.makedirs("models", exist_ok=True)

    # ----------- 建立環境 -----------
    env = PongEnvPAIA(
        max_steps=1000,
        time_penalty=-0.005,
        hit_reward=0.5,
        win_reward=2.0,
        lose_penalty=-2.0,
        # shaping_coef=0.2,
    )

    init_state = env.reset()
    state_dim = init_state.shape[0]   # 目前是 6
    action_dim = env.action_space_n   # 目前是 3

    # ----------- 裝置（GPU / CPU） -----------
    device = torch.device("cpu")
    print("Using device:", device)
    if device.type == "cuda":
        print("GPU name:", torch.cuda.get_device_name(0))

    # ----------- 建立網路與 optimizer -----------
    policy_net = DQN(state_dim, action_dim).to(device)
    target_net = DQN(state_dim, action_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    # ----------- Replay buffer + 超參數 -----------
    replay_buffer: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(
        maxlen=50_000
    )

    gamma = 0.99
    batch_size = 64

    # epsilon 設定（隨步數 decay）
    epsilon_start = 1.0
    epsilon_end = 0.05
    epsilon_decay_steps = 10_000  # 步數越大，衰減越慢

    target_update_every = 1_000   # 每多少 steps 更新一次 target_net
    max_episodes = 500

    total_steps = 0
    best_episode_reward = -1e9

    print(f"State dim = {state_dim}, Action dim = {action_dim}")
    print(f"Max episodes = {max_episodes}")

    # ----------- 訓練迴圈 -----------
    for episode in range(1, max_episodes + 1):
        state = env.reset()
        episode_reward = 0.0

        while True:
            # 線性衰減 epsilon
            epsilon = max(
                epsilon_end,
                epsilon_start
                - (epsilon_start - epsilon_end) * (total_steps / epsilon_decay_steps),
            )

            # 1. 用 epsilon-greedy 選動作
            action = select_action(policy_net, state, action_dim, epsilon, device)

            # 2. 與環境互動
            next_state, reward, done, info = env.step(action)
            episode_reward += reward

            # 3. 存進 replay buffer
            replay_buffer.append((state, action, reward, next_state, done))
            state = next_state
            total_steps += 1

            # 4. 從 replay buffer 抽樣做一次梯度更新
            if len(replay_buffer) >= batch_size:
                batch = random.sample(replay_buffer, batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)

                states_t = torch.from_numpy(np.stack(states)).float().to(device)
                actions_t = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(device)
                rewards_t = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(device)
                next_states_t = torch.from_numpy(np.stack(next_states)).float().to(device)
                dones_t = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(device)

                # Q(s, a) from policy_net
                q_values = policy_net(states_t).gather(1, actions_t)

                # target: r + gamma * max_a' Q_target(s', a')
                with torch.no_grad():
                    max_next_q = target_net(next_states_t).max(dim=1, keepdim=True)[0]
                    target_q = rewards_t + gamma * (1.0 - dones_t) * max_next_q

                loss = criterion(q_values, target_q)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # 5. 定期更新 target network（student → teacher）
            if total_steps % target_update_every == 0:
                target_net.load_state_dict(policy_net.state_dict())

            if done:
                break

        # ----------- episode 結束，印出結果 -----------
        print(
            f"[Episode {episode:4d}] "
            f"steps = {total_steps:6d} | "
            f"reward = {episode_reward:7.3f} | "
            f"epsilon = {epsilon:5.3f}"
        )

        # 儲存「單集 reward 最佳」的模型
        if episode_reward > best_episode_reward:
            best_episode_reward = episode_reward
            torch.save(policy_net.state_dict(), "models/dqn_pong_best.pt")

    # 訓練結束，儲存最後一版
    torch.save(policy_net.state_dict(), "models/dqn_pong_last.pt")
    print("Training finished. Models saved as models/dqn_pong_best.pt and models/dqn_pong_last.pt")


if __name__ == "__main__":
    train()
