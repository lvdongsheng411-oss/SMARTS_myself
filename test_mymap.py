import gymnasium as gym
import numpy as np

from smarts.core.agent_interface import AgentInterface, AgentType

agent_interfaces = {
    "Agent-001": AgentInterface.from_type(
        AgentType.Loner,
        max_episode_steps=300,   # 单回合最多300步
    )
}

env = gym.make(
    "smarts.env:hiway-v1",
    scenarios=["scenarios/mymap"],
    agent_interfaces=agent_interfaces,
    headless=False,
)

NUM_EPISODES = 200   # 连续跑200回合，可自行改大

for episode in range(NUM_EPISODES):
    obs, info = env.reset()
    print(f"第 {episode + 1} 回合开始")

    for step in range(300):
        action = {
            "Agent-001": np.array([0.2, 0.0, 0.0], dtype=np.float32)
        }

        obs, rewards, terminated, truncated, info = env.step(action)
        print(f"episode={episode + 1}, step={step}, reward={rewards}")

        if terminated["__all__"] or truncated["__all__"]:
            print(f"第 {episode + 1} 回合结束")
            break

env.close()
print("环境关闭")
