import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from env_paia import PongEnvPAIA
from train_dqn import DQN  # 用你訓練時的同一個網路結構


# --------------------------------------------------
# 載入已訓練好的模型
# --------------------------------------------------
def load_trained_model(env: PongEnvPAIA, device: torch.device, model_path: str) -> nn.Module:
    dummy_state = env.reset()
    state_dim = dummy_state.shape[0]
    action_dim = env.action_space_n

    model = DQN(state_dim, action_dim).to(device)
    state_dict = torch.load(
        model_path,
        map_location=device,
        weights_only=True,   # 比較安全
    )
    model.load_state_dict(state_dict)
    model.eval()

    print(f"Loaded model from {model_path}")
    return model


def select_action_greedy(model: nn.Module, state: np.ndarray, device: torch.device) -> int:
    with torch.no_grad():
        x = torch.from_numpy(state).float().unsqueeze(0).to(device)
        q = model(x)
        return int(q.argmax(dim=1).item())


# --------------------------------------------------
# 收集「漏球位置」的資料
# --------------------------------------------------
def collect_miss_positions(
    env: PongEnvPAIA,
    model: nn.Module,
    device: torch.device,
    num_episodes: int = 1000,
):
    miss_x_norm_list = []  # 儲存 0~1 之間的 x（bx）

    for ep in range(1, num_episodes + 1):
        state = env.reset()
        done = False

        while not done:
            action = select_action_greedy(model, state, device)
            next_state, reward, done, info = env.step(action)

            if done and info.get("result") == "lose":
                # state 定義：[bx, by, bvx, bvy, p1x, p2x, ball_attached, we_serving, ox, oy]
                # 這裡用 next_state[0] = bx (0~1)，代表球中心相對於寬度的位置
                bx = float(next_state[0])
                miss_x_norm_list.append(bx)

            state = next_state

        if ep % 100 == 0:
            print(f"Simulated {ep} episodes...")

    miss_x_norm = np.array(miss_x_norm_list, dtype=np.float32)
    print(f"Total lose samples: {len(miss_x_norm)}")
    return miss_x_norm


# --------------------------------------------------
# 繪圖 & 簡單統計
# --------------------------------------------------
def plot_miss_histogram(miss_x_norm: np.ndarray, num_bins: int = 10):
    if len(miss_x_norm) == 0:
        print("No lose samples collected, nothing to plot.")
        return

    plt.figure(figsize=(8, 4))
    plt.hist(miss_x_norm, bins=num_bins, range=(0.0, 1.0), edgecolor="black")
    plt.xlabel("Normalized X position (0.0 = left, 1.0 = right)")
    plt.ylabel("Lose count")
    plt.title("Distribution of ball X when agent loses")
    plt.tight_layout()
    plt.show()


def print_left_mid_right_stats(miss_x_norm: np.ndarray):
    """
    簡單把場地切成 3 區：左中右，算各自漏球次數。
    你也可以改成 5 區 / 10 區都可以。
    """
    left = np.sum(miss_x_norm < 1.0 / 3.0)
    mid = np.sum((miss_x_norm >= 1.0 / 3.0) & (miss_x_norm < 2.0 / 3.0))
    right = np.sum(miss_x_norm >= 2.0 / 3.0)
    total = len(miss_x_norm) + 1e-9

    print("=== Lose position stats (3 zones) ===")
    print(f"Left  (0.0 ~ 0.33): {left:4d} ({left / total:5.2%})")
    print(f"Middle(0.33~ 0.66): {mid:4d} ({mid / total:5.2%})")
    print(f"Right (0.66~ 1.00): {right:4d} ({right / total:5.2%})")


# --------------------------------------------------
# main
# --------------------------------------------------
def main():
    # 1. 建立環境（建議用你實際測試的模式：easy 或 hard）
    env = PongEnvPAIA(mode="hard")  # 你現在想看 hard 模式的弱點，就用 "hard"

    # 2. 裝置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 3. 載入模型
    model = load_trained_model(env, device, "models/dqn_pong_best.pt")

    # 4. 收集漏球位置
    miss_x_norm = collect_miss_positions(env, model, device, num_episodes=1000)

    # 5. 印出簡單統計 + 畫 histogram
    print_left_mid_right_stats(miss_x_norm)
    plot_miss_histogram(miss_x_norm)


if __name__ == "__main__":
    main()
