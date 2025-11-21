import torch
import matplotlib.pyplot as plt

def main():
    logs = torch.load("training_logs_prio.pth")

    episodes = logs["episodes"]
    rewards = logs["episode_rewards"]
    losses = logs["episode_losses"]
    avgmaxq = logs["episode_avgmaxq"]
    theta_norm = logs["episode_theta_norm"]

    # 1) Loss 曲線
    plt.figure()
    plt.plot(episodes, losses)
    plt.xlabel("Episode")
    plt.ylabel("Mean loss per episode")
    plt.title("DQN Training - Loss Curve")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("output_img\\loss_curve.png")

    # 2) 平均 max Q 變化
    plt.figure()
    plt.plot(episodes, avgmaxq)
    plt.xlabel("Episode")
    plt.ylabel("Average max Q")
    plt.title("DQN Training - Avg Max Q per Episode")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("output_img\\avgmaxq_curve.png")

    # 3) θ 的 L2 norm 變化
    plt.figure()
    plt.plot(episodes, theta_norm)
    plt.xlabel("Episode")
    plt.ylabel("||θ||₂")
    plt.title("DQN Training - Parameter Norm")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("output_img\\theta_norm_curve.png")

    # 4) 順便畫 reward 曲線
    plt.figure()
    plt.plot(episodes, rewards)
    plt.xlabel("Episode")
    plt.ylabel("Episode reward")
    plt.title("DQN Training - Episode Reward")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("output_img\\reward_curve.png")

    print("Saved plots: loss_curve.png, avgmaxq_curve.png, theta_norm_curve.png, reward_curve.png")

if __name__ == "__main__":
    main()
