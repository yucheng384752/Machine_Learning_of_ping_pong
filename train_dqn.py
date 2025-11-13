# v0.0.2
# Changelog:
# - 使用新的 PongEnv reward 設定（較小 time penalty、較大 win/lose 獎勵）。
# - 保持 DQN 結構與訓練流程不變。

import random
from collections import deque
from typing import Deque, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from env import PongEnv


class DQN(nn.Module):
    """簡單三層全連接網路，輸入 state 吐出每個 action 的 Q 值。"""

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


def select_action(
    policy_net: DQN,
    state: np.ndarray,
    action_dim: int,
    epsilon: float,
    device: torch.device,
) -> int:
    """epsilon-greedy 選擇動作。"""
    if random.random() < epsilon:
        return random.randrange(action_dim)

    state_t = torch.from_numpy(state).float().unsqueeze(0).to(device)
    with torch.no_grad():
        q_values = policy_net(state_t)
    return int(q_values.argmax(dim=1).item())


def train():
    # ----------- 基本設定 -----------
    # 使用新的獎勵超參數（也可以自行微調玩玩看）
    env = PongEnv(
        max_steps=1000,
        time_penalty=-0.005,
        hit_reward=0.5,
        win_reward=2.0,
        lose_penalty=-2.0,
        shaping_coef=0.2,
    )

    init_state = env.reset()
    state_dim = init_state.shape[0]   # 6
    action_dim = env.action_space_n   # 3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    policy_net = DQN(state_dim, action_dim).to(device)
    target_net = DQN(state_dim, action_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    replay_buffer: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(
        maxlen=50_000
    )

    gamma = 0.99
    batch_size = 64

    # epsilon 設定（隨時間 decay）
    epsilon_start = 1.0
    epsilon_end = 0.05
    epsilon_decay_steps = 10_000  # 步數越大，衰減越慢

    target_update_every = 1_000  # 每多少 steps 更新一次 target_net
    max_episodes = 500

    total_steps = 0

    best_avg_reward = -1e9

    for episode in range(1, max_episodes + 1):
        state = env.reset()
        episode_reward = 0.0

        while True:
            total_steps += 1

            # 計算目前 epsilon（指數衰減）
            epsilon = epsilon_end + (epsilon_start - epsilon_end) * np.exp(
                -1.0 * total_steps / epsilon_decay_steps
            )

            # 1. 選動作
            action = select_action(policy_net, state, action_dim, epsilon, device)

            # 2. 與環境互動
            next_state, reward, done, info = env.step(action)
            episode_reward += reward

            # 3. 存經驗到 replay buffer
            replay_buffer.append((state, action, reward, next_state, done))

            state = next_state

            # 4. 從 replay buffer 抽樣訓練
            if len(replay_buffer) >= batch_size:
                batch = random.sample(replay_buffer, batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)

                states_t = torch.from_numpy(np.stack(states)).float().to(device)
                actions_t = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(device)
                rewards_t = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(device)
                next_states_t = torch.from_numpy(np.stack(next_states)).float().to(device)
                dones_t = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(device)

                # Q(s, a)
                q_values = policy_net(states_t).gather(1, actions_t)

                # max_a' Q_target(s', a')
                with torch.no_grad():
                    max_next_q = target_net(next_states_t).max(dim=1, keepdim=True)[0]
                    target_q = rewards_t + gamma * (1.0 - dones_t) * max_next_q

                loss = criterion(q_values, target_q)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # 5. 定期更新 target network
            if total_steps % target_update_every == 0:
                target_net.load_state_dict(policy_net.state_dict())

            if done:
                break

        # episode 結束，印出結果
        print(
            f"Episode {episode:04d} | "
            f"reward = {episode_reward:7.3f} | "
            f"epsilon = {epsilon:5.3f}"
        )

        # 簡單儲存最佳模型
        if episode_reward > best_avg_reward:
            best_avg_reward = episode_reward
            torch.save(policy_net.state_dict(), "dqn_pong_best.pt")

    # 訓練結束，儲存最後一版
    torch.save(policy_net.state_dict(), "dqn_pong_last.pt")
    print("Training finished. Models saved as dqn_pong_best.pt and dqn_pong_last.pt")


if __name__ == "__main__":
    train()
