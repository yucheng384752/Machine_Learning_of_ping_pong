# v0.5.0
# Changelog:
# - 將 uniform replay buffer 改為 Prioritized Replay
# - 高速球 (speed_level 高) 的資料在訓練中加強 (high-speed augmentation)
# - reward / loss / avg Q / θ-norm 仍然記錄，並新增 plot_training_dashboard()
# - 保留：PAIA 版環境、Double DQN、左右鏡射 augmentation

import os
import random
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from env_paia import PongEnvPAIA


# ------------------ DQN 網路 ------------------
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


# ------------------ Prioritized Replay Buffer ------------------
class PrioritizedReplayBuffer:
    def __init__(self, capacity: int, alpha: float = 0.6):
        """
        capacity: buffer 最大長度
        alpha: 控制優先度的影響力 (0 = uniform, 1 = 完全依照 priority)
        """
        self.capacity = capacity
        self.alpha = alpha

        self.pos = 0
        self.full = False

        self.states: List[np.ndarray] = [None] * capacity
        self.actions: List[int] = [0] * capacity
        self.rewards: List[float] = [0.0] * capacity
        self.next_states: List[np.ndarray] = [None] * capacity
        self.dones: List[bool] = [False] * capacity

        # 儲存每筆 transition 的 priority
        self.priorities = np.zeros((capacity,), dtype=np.float32)

    def __len__(self):
        return self.capacity if self.full else self.pos

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        idx = self.pos
        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_states[idx] = next_state
        self.dones[idx] = done

        # 新資料的 priority 設為目前最大值，避免剛加入卻抽不到
        max_prio = self.priorities.max() if self.pos > 0 or self.full else 1.0
        self.priorities[idx] = max_prio

        self.pos = (self.pos + 1) % self.capacity
        self.full = self.full or self.pos == 0

    def sample(self, batch_size: int, beta: float = 0.4):
        """
        依照 priority^alpha 進行抽樣，並回傳 importance-sampling weights。
        beta: 越接近 1，對權重修正越強。
        """
        assert len(self) >= batch_size

        prios = self.priorities[: len(self)]
        probs = prios ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self), batch_size, p=probs)

        # IS weights
        N = len(self)
        weights = (N * probs[indices]) ** (-beta)
        weights /= weights.max()  # 正規化到 [0,1]

        states = [self.states[i] for i in indices]
        actions = [self.actions[i] for i in indices]
        rewards = [self.rewards[i] for i in indices]
        next_states = [self.next_states[i] for i in indices]
        dones = [self.dones[i] for i in indices]

        return (
            np.stack(states),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.stack(next_states),
            np.array(dones, dtype=np.float32),
            indices,
            weights.astype(np.float32),
        )

    def update_priorities(self, indices, new_priorities):
        # priority 不能是 0，避免抽不到
        self.priorities[indices] = np.maximum(new_priorities, 1e-6)


# ------------------ epsilon-greedy ------------------
def select_action(policy_net: DQN, state: np.ndarray,
                  action_dim: int, epsilon: float, device: torch.device) -> int:
    if random.random() < epsilon:
        return random.randrange(action_dim)
    state_t = torch.from_numpy(state).float().unsqueeze(0).to(device)
    with torch.no_grad():
        q = policy_net(state_t)
    return int(q.argmax(dim=1).item())


# ------------------ 左右鏡射 ------------------
def mirror_state(state: np.ndarray) -> np.ndarray:
    """
    PAIA state 結構 (12 維):
    0 bx, 1 by, 2 bvx, 3 bvy, 4 p1x, 5 p2x,
    6 ball_attached, 7 we_serving,
    8 ox, 9 oy, 10 speed_level, 11 landing_x
    """
    m = state.copy()
    # x 相關：鏡射 → 1 - x
    m[0] = 1.0 - state[0]   # bx
    m[4] = 1.0 - state[4]   # p1x
    m[5] = 1.0 - state[5]   # p2x
    m[8] = 1.0 - state[8]   # ox
    m[11] = 1.0 - state[11] # landing_x
    # 水平速度：bvx → 1 - bvx
    m[2] = 1.0 - state[2]
    return m


def mirror_action(action: int) -> int:
    # 0: 不動, 1: 左移, 2: 右移, 3: 發球左, 4: 發球右
    if action == 1:
        return 2
    if action == 2:
        return 1
    if action == 3:
        return 4
    if action == 4:
        return 3
    return action


# ------------------ 主訓練流程 ------------------
def train():
    os.makedirs("models", exist_ok=True)

    # 1) 建立環境
    env = PongEnvPAIA(
        mode="hard",           # easy / hard 自行切換
        max_steps=1000,
        time_penalty=-0.005,
        hit_reward=0.5,
        win_reward=2.0,
        lose_penalty=-2.0,
        fast_speed_level=2,    # >=2 視為「快速球」
        fast_hit_bonus=0.3,
        fast_win_bonus=0.5,
    )

    init_state = env.reset()
    state_dim = init_state.shape[0]
    action_dim = env.action_space_n

    # 2) 裝置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    # 3) 網路 & optimizer
    policy_net = DQN(state_dim, action_dim).to(device)
    target_net = DQN(state_dim, action_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)

    # 4) Prioritized Replay 超參數
    buffer = PrioritizedReplayBuffer(capacity=50_000, alpha=0.6)
    gamma = 0.99
    batch_size = 64

    # epsilon 以「總步數」衰減
    epsilon_start = 1.0
    epsilon_end = 0.05
    epsilon_decay_steps = 80_000  # 比之前大 → 衰減慢一點

    # beta（重要性權重）也隨步數從 0.4 漸進到 1.0
    beta_start = 0.4
    beta_frames = 100_000

    target_update_every = 1_000
    max_episodes = 1000

    total_steps = 0
    best_episode_reward = -1e9

    # 5) 紀錄訓練曲線
    episodes_log: List[int] = []
    rewards_log: List[float] = []
    losses_log: List[float] = []
    avgmaxq_log: List[float] = []
    theta_norm_log: List[float] = []

    print(f"State dim = {state_dim}, Action dim = {action_dim}")

    for episode in range(1, max_episodes + 1):
        state = env.reset()
        episode_reward = 0.0

        losses_this_ep: List[float] = []
        maxq_this_ep: List[float] = []

        while True:
            # epsilon / beta 更新
            epsilon = max(
                epsilon_end,
                epsilon_start - (epsilon_start - epsilon_end) * (total_steps / epsilon_decay_steps),
            )
            beta = min(
                1.0,
                beta_start + (1.0 - beta_start) * (total_steps / beta_frames),
            )

            # 1) 選動作
            action = select_action(policy_net, state, action_dim, epsilon, device)

            # 2) 與環境互動
            next_state, reward, done, info = env.step(action)
            episode_reward += reward

            # 3) 存進 prioritized buffer
            buffer.push(state, action, reward, next_state, done)

            state = next_state
            total_steps += 1

            # 4) 從 buffer 中 sample + high-speed augmentation
            if len(buffer) >= batch_size:
                (
                    states_np,
                    actions_np,
                    rewards_np,
                    next_states_np,
                    dones_np,
                    indices,
                    weights_np,
                ) = buffer.sample(batch_size, beta=beta)

                # ---- 基本 batch 轉 tensor ----
                states_list = list(states_np)
                actions_list = list(actions_np)
                rewards_list = list(rewards_np)
                next_states_list = list(next_states_np)
                dones_list = list(dones_np)

                # ---- 左右鏡射 (所有樣本) ----
                aug_states: List[np.ndarray] = []
                aug_actions: List[int] = []
                aug_rewards: List[float] = []
                aug_next_states: List[np.ndarray] = []
                aug_dones: List[bool] = []

                # 用於 high-speed augmentation 的額外列表
                hs_states: List[np.ndarray] = []
                hs_actions: List[int] = []
                hs_rewards: List[float] = []
                hs_next_states: List[np.ndarray] = []
                hs_dones: List[bool] = []

                for s, a, r, ns, d in zip(
                    states_list, actions_list, rewards_list, next_states_list, dones_list
                ):
                    # 原始
                    aug_states.append(s)
                    aug_actions.append(a)
                    aug_rewards.append(r)
                    aug_next_states.append(ns)
                    aug_dones.append(d)

                    # 鏡射
                    s_m = mirror_state(s)
                    ns_m = mirror_state(ns)
                    a_m = mirror_action(a)
                    aug_states.append(s_m)
                    aug_actions.append(a_m)
                    aug_rewards.append(r)
                    aug_next_states.append(ns_m)
                    aug_dones.append(d)

                    # ---- high-speed augmentation ----
                    # 如果 speed_level（state[10]）很高，就再多複製一次（原始+鏡射）
                    speed_level_norm = s[10]  # 0~1
                    if speed_level_norm >= 0.66:  # 例如最後 1/3 速度區間視為「快速」
                        hs_states.extend([s, s_m])
                        hs_actions.extend([a, a_m])
                        hs_rewards.extend([r, r])
                        hs_next_states.extend([ns, ns_m])
                        hs_dones.extend([d, d])

                # 把高速度樣本附加上去
                if hs_states:
                    aug_states.extend(hs_states)
                    aug_actions.extend(hs_actions)
                    aug_rewards.extend(hs_rewards)
                    aug_next_states.extend(hs_next_states)
                    aug_dones.extend(hs_dones)

                # 轉 tensor
                states_t = torch.from_numpy(np.stack(aug_states)).float().to(device)
                actions_arr = np.array(aug_actions, dtype=np.int64)
                actions_t = torch.from_numpy(actions_arr).unsqueeze(1).to(device)
                rewards_arr = np.array(aug_rewards, dtype=np.float32)
                rewards_t = torch.from_numpy(rewards_arr).unsqueeze(1).to(device)
                next_states_t = torch.from_numpy(np.stack(aug_next_states)).float().to(device)
                dones_arr = np.array(aug_dones, dtype=np.float32)
                dones_t = torch.from_numpy(dones_arr).unsqueeze(1).to(device)

                # importance weights 也要擴充到對應長度（簡單做法：重複）
                base_weights = torch.from_numpy(weights_np).float().to(device)
                repeat_factor = len(aug_states) // len(weights_np)
                extra = len(aug_states) - repeat_factor * len(weights_np)
                weights_list = base_weights.repeat(repeat_factor)
                if extra > 0:
                    weights_list = torch.cat([weights_list, base_weights[:extra]], dim=0)
                weights_t = weights_list.unsqueeze(1)  # (N,1)

                # Q(s,a)
                all_q = policy_net(states_t)
                q_values = all_q.gather(1, actions_t)

                # 平均 max Q (分析用)
                with torch.no_grad():
                    batch_max_q = all_q.max(dim=1)[0].mean().item()

                # Double DQN target
                with torch.no_grad():
                    next_q_policy = policy_net(next_states_t)
                    best_actions = next_q_policy.argmax(dim=1, keepdim=True)
                    next_q_target = target_net(next_states_t).gather(1, best_actions)
                    target_q = rewards_t + gamma * (1.0 - dones_t) * next_q_target

                td_error = target_q - q_values
                loss = (weights_t * td_error.pow(2)).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # 更新 priority（只用原始 batch 的 TD-error，取前 batch_size 個元素）
                with torch.no_grad():
                    td_abs = td_error.detach().abs().cpu().numpy().flatten()
                    # 因為我們擴充了樣本數，這裡取前 batch_size 個對應原 indices
                    new_prios = td_abs[: len(indices)]
                    buffer.update_priorities(indices, new_prios)

                losses_this_ep.append(loss.item())
                maxq_this_ep.append(batch_max_q)

            # 5) 更新 target_net
            if total_steps % target_update_every == 0:
                target_net.load_state_dict(policy_net.state_dict())

            if done:
                break

        # ---- 一集結束，紀錄 ----
        episodes_log.append(episode)
        rewards_log.append(episode_reward)

        mean_loss = float(sum(losses_this_ep)) / len(losses_this_ep) if losses_this_ep else 0.0
        losses_log.append(mean_loss)

        mean_maxq = float(sum(maxq_this_ep)) / len(maxq_this_ep) if maxq_this_ep else 0.0
        avgmaxq_log.append(mean_maxq)

        # θ 的 L2 norm
        with torch.no_grad():
            sq_sum = 0.0
            for p in policy_net.parameters():
                sq_sum += p.data.pow(2).sum().item()
            theta_norm = sq_sum ** 0.5
        theta_norm_log.append(theta_norm)

        print(
            f"[Episode {episode:4d}] "
            f"steps = {total_steps:6d} | "
            f"reward = {episode_reward:7.3f} | "
            f"epsilon = {epsilon:5.3f} | "
            f"mean_loss = {mean_loss:7.4f} | "
            f"avg_maxQ = {mean_maxq:7.4f}"
        )

        if episode_reward > best_episode_reward:
            best_episode_reward = episode_reward
            torch.save(policy_net.state_dict(), "models/dqn_pong_best.pt")

    torch.save(policy_net.state_dict(), "models/dqn_pong_last.pt")
    print("Training finished. Models saved as models/dqn_pong_best.pt and models/dqn_pong_last.pt")

    log_data = {
        "episodes": episodes_log,
        "episode_rewards": rewards_log,
        "episode_losses": losses_log,
        "episode_avgmaxq": avgmaxq_log,
        "episode_theta_norm": theta_norm_log,
    }
    torch.save(log_data, "training_logs_prio.pth")
    print("Training logs saved to training_logs_prio.pth")

    # 訓練完直接畫一次儀表板
    plot_training_dashboard("training_logs_prio.pth")


# ------------------ 訓練儀表板 ------------------
def plot_training_dashboard(log_path: str = "training_logs_prio.pth"):
    if not os.path.exists(log_path):
        print(f"[WARN] log file not found: {log_path}")
        return

    data = torch.load(log_path)
    episodes = data["episodes"]
    rewards = data["episode_rewards"]
    losses = data["episode_losses"]
    avgmaxq = data["episode_avgmaxq"]
    theta_norm = data["episode_theta_norm"]

    plt.figure(figsize=(12, 10))

    # 1) Reward
    plt.subplot(2, 2, 1)
    plt.plot(episodes, rewards)
    plt.title("Episode Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")

    # 2) Loss
    plt.subplot(2, 2, 2)
    plt.plot(episodes, losses)
    plt.title("Loss")
    plt.xlabel("Episode")
    plt.ylabel("Mean loss")

    # 3) Avg max Q
    plt.subplot(2, 2, 3)
    plt.plot(episodes, avgmaxq)
    plt.title("Avg Max Q")
    plt.xlabel("Episode")
    plt.ylabel("Avg max Q")

    # 4) ||θ||₂
    plt.subplot(2, 2, 4)
    plt.plot(episodes, theta_norm)
    plt.title("Parameter Norm ||θ||₂")
    plt.xlabel("Episode")
    plt.ylabel("Norm")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    train()
