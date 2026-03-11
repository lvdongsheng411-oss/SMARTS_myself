#该文件用于演示训练的结果，即在train_ppo.py中进行小车训练后
#用该文件在网页端显示训练成果

from pathlib import Path

import numpy as np

# PyTorch
import torch
import torch.nn as nn

from r1_project.env_wrapper import SmartsSingleAgentEnv


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




# 观测维度
OBS_DIM = 10

# 动作维度
ACT_DIM = 3

# 隐藏层维度
HIDDEN_DIM = 128

# 单回合最大步数
MAX_EPISODE_STEPS = 300

# 测试多少个回合
NUM_EPISODES = 10



# 定义 Actor 网络


class Actor(nn.Module):


    def __init__(self, obs_dim, act_dim, hidden_dim):
        super().__init__()

        # 第一层全连接
        self.fc1 = nn.Linear(obs_dim, hidden_dim)

        # 第二层全连接
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        # 输出动作均值
        self.mean_head = nn.Linear(hidden_dim, act_dim)

        # 可学习的 log_std
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, obs):

        x = torch.relu(self.fc1(obs))

        x = torch.relu(self.fc2(x))

        # 输出动作均值
        mean = self.mean_head(x)


        mean = torch.tanh(mean)

        # 计算标准差
        std = torch.exp(self.log_std)

        return mean, std



# 动作缩放函数


def scale_action(action):
    """
  
    """


    a = np.array(action, dtype=np.float32)


    throttle = (a[0] + 1.0) / 2.0

    brake = (a[1] + 1.0) / 2.0


    steering = np.clip(a[2], -1.0, 1.0)

    throttle = np.clip(throttle, 0.0, 1.0)
    brake = np.clip(brake, 0.0, 1.0)


    return np.array([throttle, brake, steering], dtype=np.float32)



#  用 Actor 选择动作


def select_action(actor, obs):



    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)


    with torch.no_grad():
        mean, std = actor(obs_tensor)

    action = mean.squeeze(0).cpu().numpy()

    return action



# 主函数


def test():
    """
    加载训练好的 PPO Actor 模型，
    在网页端场景中测试运行。
    """


    # headless=False，这样网页端 Envision 才能看到，用True时网页端看不见
    env = SmartsSingleAgentEnv(
        scenario_path="scenarios/mymap",
        headless=False
    )


    actor = Actor(OBS_DIM, ACT_DIM, HIDDEN_DIM).to(device)

    #  定义模型路径 
    project_dir = Path(__file__).resolve().parent
    actor_path = project_dir / "models" / "ppo_actor.pth"

    print(f"准备加载 Actor 模型: {actor_path}")

    actor.load_state_dict(torch.load(actor_path, map_location=device))

    actor.eval()

    print("Actor 模型加载成功，开始测试。")

    # ---------- 7.5 测试多个回合 ----------
    for episode in range(NUM_EPISODES):

        # 重置环境
        obs, info = env.reset()

        # 当前回合累计奖励
        episode_reward = 0.0

        print(f"第 {episode + 1} 回合开始")

        for step in range(MAX_EPISODE_STEPS):

            # 用当前策略网络选择动作
            raw_action = select_action(actor, obs)

            env_action = scale_action(raw_action)

            # 与环境交互
            obs, reward, done, info = env.step(env_action)

            # 累积回合奖励
            episode_reward += reward


            print(
                f"episode={episode + 1}, "
                f"step={step}, "
                f"reward={reward:.4f}, "
                f"action={env_action}"
            )


            if done:
                print(f"第 {episode + 1} 回合结束，累计奖励 = {episode_reward:.4f}")
                break


    env.close()

    print("测试结束，环境已关闭。")


if __name__ == "__main__":
    test()
