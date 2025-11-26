import os
from collections import deque
from typing import Deque, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

try:
    # package 模式：Machine_Learning_of_ping_pong 作為專案根目錄時
    from env_paia import PongEnvPAIA
except ImportError:
    # 腳本模式：直接在 PAIA 資料夾內執行 train_dqn.py
    from env_paia import PongEnvPAIA

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------------ #
# DQN 網路
# ------------------------------------------------------------------ #

class DQN(nn.Module):
    def __init__(self, state_dim: int, action_dim: int) -> None:
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


# ------------------------------------------------------------------ #
# Replay Buffer (uniform)
# ------------------------------------------------------------------ #

class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.buffer: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(
            maxlen=capacity
        )

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(
            *(self.buffer[idx] for idx in indices)
        )
        return (
            np.stack(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.stack(next_states),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


# ------------------------------------------------------------------ #
# 訓練主程式
# ------------------------------------------------------------------ #

def train(
    num_episodes: int = 1500,
    max_steps_per_episode: int = 2000,
    gamma: float = 0.99,
    lr: float = 5e-4,
    batch_size: int = 128,
    buffer_capacity: int = 80000,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay_steps: int = 150000,
    target_update_interval: int = 2000,
    model_dir: str = f"C:\\Users\\Yucheng\\Desktop\\Machine_Learning_of_ping_pong\\models",
) -> None:
    os.makedirs(model_dir, exist_ok=True)

    env = PongEnvPAIA(mode="hard", max_steps=max_steps_per_episode)
    state = env.reset()
    state_dim = state.shape[0]
    action_dim = 5  # 0~4

    policy_net = DQN(state_dim, action_dim).to(DEVICE)
    target_net = DQN(state_dim, action_dim).to(DEVICE)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    replay_buffer = ReplayBuffer(buffer_capacity)

    steps_done = 0

    episode_rewards: List[float] = []
    episode_losses: List[float] = []
    episode_avg_max_q: List[float] = []
    theta_norm_list: List[float] = []

    for episode in range(1, num_episodes + 1):
        state = env.reset()
        total_reward = 0.0
        losses = []
        max_q_vals = []

        for t in range(max_steps_per_episode):
            steps_done += 1

            # epsilon-greedy
            epsilon = max(
                epsilon_end,
                epsilon_start
                - (epsilon_start - epsilon_end) * steps_done / epsilon_decay_steps,
            )

            if np.random.rand() < epsilon:
                action = np.random.randint(action_dim)
            else:
                with torch.no_grad():
                    s = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                    q_vals = policy_net(s)
                    action = int(torch.argmax(q_vals, dim=1).item())
                    max_q_vals.append(q_vals.max().item())

            next_state, reward, done, info = env.step(action)

            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            # ----------------- DQN 更新 -----------------
            if len(replay_buffer) >= batch_size:
                (
                    batch_states,
                    batch_actions,
                    batch_rewards,
                    batch_next_states,
                    batch_dones,
                ) = replay_buffer.sample(batch_size)

                bs = torch.tensor(batch_states, dtype=torch.float32, device=DEVICE)
                ba = torch.tensor(batch_actions, dtype=torch.int64, device=DEVICE).unsqueeze(-1)
                br = torch.tensor(batch_rewards, dtype=torch.float32, device=DEVICE).unsqueeze(-1)
                bns = torch.tensor(batch_next_states, dtype=torch.float32, device=DEVICE)
                bd = torch.tensor(batch_dones, dtype=torch.float32, device=DEVICE).unsqueeze(-1)

                # Q(s,a;θ)
                q_values = policy_net(bs).gather(1, ba)

                # max_a' Q_target(s',a')
                with torch.no_grad():
                    next_q_values = target_net(bns).max(dim=1, keepdim=True)[0]
                    target_q = br + gamma * (1.0 - bd) * next_q_values

                loss = nn.MSELoss()(q_values, target_q)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=5.0)
                optimizer.step()

                losses.append(loss.item())

            # 更新 target network
            if steps_done % target_update_interval == 0:
                target_net.load_state_dict(policy_net.state_dict())

            if done:
                break

        # episode 結束統計
        episode_rewards.append(total_reward)
        episode_losses.append(np.mean(losses) if losses else 0.0)
        episode_avg_max_q.append(np.mean(max_q_vals) if max_q_vals else 0.0)

        # θ 的 L2-norm
        with torch.no_grad():
            sq_sum = 0.0
            for p in policy_net.parameters():
                sq_sum += (p.detach() ** 2).sum().item()
            theta_norm_list.append(math.sqrt(sq_sum))

        print(
            f"[Episode {episode:4d}] steps = {steps_done:6d} | "
            f"reward = {total_reward:7.3f} | epsilon = {epsilon:0.3f}"
        )

    # 存模型
    torch.save(policy_net.state_dict(), os.path.join(model_dir, "dqn_pong_last.pt"))
    print("Saving model to:", os.path.abspath(model_dir))

    # 簡單把曲線存成 npy, 給 analysis/plot_training_curves.py 使用
    np.save(os.path.join(model_dir, "episode_rewards.npy"), np.array(episode_rewards))
    np.save(os.path.join(model_dir, "episode_losses.npy"), np.array(episode_losses))
    np.save(os.path.join(model_dir, "episode_avg_max_q.npy"), np.array(episode_avg_max_q))
    np.save(os.path.join(model_dir, "theta_norm.npy"), np.array(theta_norm_list))


if __name__ == "__main__":
    import math

    train()
