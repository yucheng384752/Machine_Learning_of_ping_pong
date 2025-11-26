import torch
import matplotlib.pyplot as plt
import os
import numpy as np

def main():
    # 修改這裡 → 指到你 models 資料夾
    model_dir = r"C:\Users\Yucheng\Desktop\Machine_Learning_of_ping_pong\models"
    out_dir = os.path.join(model_dir, "output_img")
    os.makedirs(out_dir, exist_ok=True)

    # 1) 載入 npy 檔案
    rewards = np.load(os.path.join(model_dir, "episode_rewards.npy"))
    losses = np.load(os.path.join(model_dir, "episode_losses.npy"))
    avgmaxq = np.load(os.path.join(model_dir, "episode_avg_max_q.npy"))
    theta_norm = np.load(os.path.join(model_dir, "theta_norm.npy"))

    episodes = np.arange(1, len(rewards) + 1)

    # 2) Loss 曲線
    plt.figure()
    plt.plot(episodes, losses)
    plt.xlabel("Episode")
    plt.ylabel("Mean loss per episode")
    plt.title("DQN Training - Loss Curve")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "loss_curve.png"))

    # 3) 平均 max Q 變化
    plt.figure()
    plt.plot(episodes, avgmaxq)
    plt.xlabel("Episode")
    plt.ylabel("Average max Q")
    plt.title("DQN Training - Avg Max Q per Episode")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "avgmaxq_curve.png"))

    # 4) θ 的 L2 norm 變化
    plt.figure()
    plt.plot(episodes, theta_norm)
    plt.xlabel("Episode")
    plt.ylabel("||θ||₂")
    plt.title("DQN Training - Parameter Norm")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "theta_norm_curve.png"))

    # 5) Episode reward 曲線
    plt.figure()
    plt.plot(episodes, rewards)
    plt.xlabel("Episode")
    plt.ylabel("Episode reward")
    plt.title("DQN Training - Episode Reward")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "reward_curve.png"))

    print("Saved plots in:", out_dir)

if __name__ == "__main__":
    main()
