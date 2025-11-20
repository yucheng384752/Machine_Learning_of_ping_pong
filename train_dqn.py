# v0.4.0
# Changelog:
# - 加入訓練過程紀錄：
#   - 每一集的 episode_reward
#   - 每一集的平均 loss
#   - 每一集的平均 max Q
#   - 每一集 policy_net 參數的 L2 norm（||θ||₂）
# - 訓練結束後會將紀錄存成 training_logs.pth，供後續畫圖使用
#
# 保留：
# - PAIA 版環境 env_paia.PongEnvPAIA
# - Double DQN 更新方式
# - 左右鏡射的 data augmentation
# - epsilon 以步數線性衰減 (50_000 steps)

import os
import random
from collections import deque
from typing import Deque, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from env_paia import PongEnvPAIA   # 使用 PAIA 版環境


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


# --------- 左右鏡射工具 ---------
def mirror_state(state: np.ndarray) -> np.ndarray:
    """
    對 PAIA state 做左右鏡射。

    state 結構（長度 12）：
    [0] bx              球 x（0~1）
    [1] by              球 y（0~1）
    [2] bvx             球 vx 正規化後（0~1，0.5=0，>0.5=向右）
    [3] bvy             球 vy 正規化後
    [4] p1x             1P 板子中心 x（0~1）
    [5] p2x             2P 板子中心 x（0~1）
    [6] ball_attached
    [7] we_serving
    [8] ox              障礙物 x（0~1）
    [9] oy              障礙物 y（0~1）
    [10] speed_level    球速等級正規化（0~1）
    [11] landing_x      預測落點 x（0~1）

    左右鏡射意義：
    - x 相關：bx, p1x, p2x, ox, landing_x → 1 - 原本值
    - 水平速度：vx → -vx；對應到正規化後的 bvx 就是 1 - bvx
    - 其他維度不變
    """
    mirrored = state.copy()

    # bx
    mirrored[0] = 1.0 - state[0]
    # by 不變: mirrored[1]
    # bvx -> 1 - bvx
    mirrored[2] = 1.0 - state[2]
    # bvy 不變: mirrored[3]

    # p1x, p2x
    mirrored[4] = 1.0 - state[4]
    mirrored[5] = 1.0 - state[5]

    # ball_attached, we_serving 不變: [6], [7]

    # ox, oy：x 鏡射、y 不變
    mirrored[8] = 1.0 - state[8]
    # mirrored[9] = state[9]

    # speed_level 不變: mirrored[10]

    # landing_x 鏡射
    mirrored[11] = 1.0 - state[11]

    return mirrored


def mirror_action(action: int) -> int:
    """
    對動作做左右鏡射。

    action 定義：
        0 = 不動
        1 = 左移
        2 = 右移
        3 = 發球向左
        4 = 發球向右
    鏡射後：
        0 -> 0
        1 <-> 2
        3 <-> 4
    """
    if action == 1:
        return 2
    if action == 2:
        return 1
    if action == 3:
        return 4
    if action == 4:
        return 3
    return action


# --------- 主訓練流程 ---------
def train():
    # 讓 models/ 一定存在
    os.makedirs("models", exist_ok=True)

    # ----------- 建立環境（PAIA 版）-----------
    env = PongEnvPAIA(
        mode="hard",          # 要練 hard 模式就改這裡；easy 就改成 "easy"
        max_steps=1000,
        time_penalty=-0.005,
        hit_reward=0.5,
        win_reward=2.0,
        lose_penalty=-2.0,
    )

    init_state = env.reset()
    state_dim = init_state.shape[0]   # 12
    action_dim = env.action_space_n   # 5

    # ----------- 裝置（GPU / CPU） -----------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    epsilon_decay_steps = 50_000  # 步數越大，衰減越慢

    target_update_every = 1_000   # 每多少 steps 更新一次 target_net
    max_episodes = 1000

    total_steps = 0
    best_episode_reward = -1e9

    print(f"State dim = {state_dim}, Action dim = {action_dim}")
    print(f"Max episodes = {max_episodes}")

    # ----------- 訓練過程紀錄用 ----------- 
    episode_indices: List[int] = []
    episode_rewards_log: List[float] = []
    episode_losses_log: List[float] = []       # 每集平均 loss
    episode_avgmaxq_log: List[float] = []      # 每集平均 max Q
    episode_theta_norm_log: List[float] = []   # 每集 ||θ||₂

    # ----------- 訓練迴圈 ----------- 
    for episode in range(1, max_episodes + 1):
        state = env.reset()
        episode_reward = 0.0

        # 這一集內用來計算平均的容器
        losses_this_ep: List[float] = []
        maxq_this_ep: List[float] = []

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

            # 4. 從 replay buffer 抽樣做一次梯度更新（含左右鏡射增強）
            if len(replay_buffer) >= batch_size:
                batch = random.sample(replay_buffer, batch_size)

                # 原始 & 鏡射資料容器
                aug_states: List[np.ndarray] = []
                aug_actions: List[int] = []
                aug_rewards: List[float] = []
                aug_next_states: List[np.ndarray] = []
                aug_dones: List[bool] = []

                for s, a, r, ns, d in batch:
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
                    aug_rewards.append(r)      # reward 在左右對稱下相同
                    aug_next_states.append(ns_m)
                    aug_dones.append(d)

                # 轉成 tensor
                states_t = torch.from_numpy(np.stack(aug_states)).float().to(device)
                actions_arr = np.array(aug_actions, dtype=np.int64)
                actions_t = torch.from_numpy(actions_arr).unsqueeze(1).to(device)
                rewards_arr = np.array(aug_rewards, dtype=np.float32)
                rewards_t = torch.from_numpy(rewards_arr).unsqueeze(1).to(device)
                next_states_t = torch.from_numpy(np.stack(aug_next_states)).float().to(device)
                dones_arr = np.array(aug_dones, dtype=np.float32)
                dones_t = torch.from_numpy(dones_arr).unsqueeze(1).to(device)

                # Q(s, a) from policy_net
                all_q_values = policy_net(states_t)
                q_values = all_q_values.gather(1, actions_t)

                # 這一個 batch 的平均 max Q（用來畫 Q-value 變化）
                with torch.no_grad():
                    batch_max_q = all_q_values.max(dim=1)[0].mean().item()

                # --------- Double DQN target 計算 ---------
                with torch.no_grad():
                    # 用 policy_net 在 next_state 上選動作（argmax）
                    next_q_policy = policy_net(next_states_t)
                    best_actions = next_q_policy.argmax(dim=1, keepdim=True)

                    # 用 target_net 估計這些動作的 Q 值
                    next_q_target = target_net(next_states_t).gather(1, best_actions)

                    # target: r + gamma * max_a' Q_target(s', a*)
                    target_q = rewards_t + gamma * (1.0 - dones_t) * next_q_target

                loss = criterion(q_values, target_q)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # 紀錄這個 batch 的 loss / maxQ
                losses_this_ep.append(loss.item())
                maxq_this_ep.append(batch_max_q)

            # 5. 定期更新 target network（student → teacher）
            if total_steps % target_update_every == 0:
                target_net.load_state_dict(policy_net.state_dict())

            if done:
                break

        # ----------- episode 結束，統計並紀錄 ----------- 
        episode_indices.append(episode)
        episode_rewards_log.append(episode_reward)

        mean_loss = (
            float(sum(losses_this_ep)) / len(losses_this_ep)
            if losses_this_ep else 0.0
        )
        episode_losses_log.append(mean_loss)

        mean_maxq = (
            float(sum(maxq_this_ep)) / len(maxq_this_ep)
            if maxq_this_ep else 0.0
        )
        episode_avgmaxq_log.append(mean_maxq)

        # θ 的 L2 norm
        with torch.no_grad():
            sq_sum = 0.0
            for p in policy_net.parameters():
                sq_sum += p.data.pow(2).sum().item()
            theta_norm = sq_sum ** 0.5
        episode_theta_norm_log.append(theta_norm)

        # 印出結果
        print(
            f"[Episode {episode:4d}] "
            f"steps = {total_steps:6d} | "
            f"reward = {episode_reward:7.3f} | "
            f"epsilon = {epsilon:5.3f} | "
            f"mean_loss = {mean_loss:7.4f} | "
            f"avg_maxQ = {mean_maxq:7.4f}"
        )

        # 儲存「單集 reward 最佳」的模型
        if episode_reward > best_episode_reward:
            best_episode_reward = episode_reward
            torch.save(policy_net.state_dict(), "models/dqn_pong_best.pt")

    # 訓練結束，儲存最後一版模型
    torch.save(policy_net.state_dict(), "models/dqn_pong_last.pt")
    print("Training finished. Models saved as models/dqn_pong_best.pt and models/dqn_pong_last.pt")

    # ----------- 存下訓練曲線資料，供後續畫圖 ----------- 
    log_data = {
        "episodes": episode_indices,
        "episode_rewards": episode_rewards_log,
        "episode_losses": episode_losses_log,
        "episode_avgmaxq": episode_avgmaxq_log,
        "episode_theta_norm": episode_theta_norm_log,
    }
    torch.save(log_data, "training_logs.pth")
    print("Training logs saved to training_logs.pth")


if __name__ == "__main__":
    train()
