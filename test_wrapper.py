import numpy as np

from r1_project.env_wrapper import SmartsSingleAgentEnv


# 创建包装后的环境对象
env = SmartsSingleAgentEnv(
    scenario_path="scenarios/mymap",  # 使用我在SUMO上创建的场景
    headless=False                    # 这里为 False，网页端才能看到
)


obs, info = env.reset()

# 打印 reset 成功信息
print("reset 成功")


print("obs shape =", obs.shape)


print("obs =", obs)

# 连续跑10000回合
for episode in range(10000):

   
    obs, info = env.reset()
    print(f"第 {episode + 1} 回合开始")

    # 每回合最多跑 10000 步
    for step in range(10000):

       
        # [throttle, brake, steering]
        action = np.array([0.2, 0.0, 0.0], dtype=np.float32)

       
        obs, reward, done, info = env.step(action)

        # 打印当前信息
        print(f"episode={episode + 1}, step={step}, obs_shape={obs.shape}, reward={reward:.4f}")

        # 如果当前回合结束，就跳到下一回合
        if done:
            print(f"第 {episode + 1} 回合结束")
            break

# 关闭环境
env.close()

# 打印关闭信息
print("环境关闭")
