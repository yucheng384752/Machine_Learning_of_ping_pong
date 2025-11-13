from env import PongEnv

env = PongEnv()
state = env.reset()
print("初始 state:", state)

total_reward = 0.0

for t in range(50):
    action = 0  # 先全部不動試試看
    state, reward, done, info = env.step(action)
    total_reward += reward

    print(f"t={t}, reward={reward:.3f}, done={done}, info={info}")

    if done:
        break

print("總 reward:", total_reward)
