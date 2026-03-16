import numpy as np
from pathlib import Path

import torch
import torch.nn as nn

from r1_project.env_wrapper import SmartsSingleAgentEnv


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 必须和 train_ppo.py 保持一致
OBS_DIM = 16
ACT_DIM = 3
HIDDEN_DIM = 128

NUM_EPISODES = 5
MAX_EPISODE_STEPS = 300


def orthogonal_init(layer, gain=np.sqrt(2)):
    if isinstance(layer, nn.Linear):
        nn.init.orthogonal_(layer.weight, gain=gain)
        nn.init.constant_(layer.bias, 0.0)


class Actor(nn.Module):
    """
    策略网络：
    输入观测
    输出高斯策略的 mean 和 std
    测试时采用贪心策略：直接取 mean 作为最大概率动作
    """

    def __init__(self, obs_dim, act_dim, hidden_dim):
        super().__init__()

        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_head = nn.Linear(hidden_dim, act_dim)
        self.log_std = nn.Parameter(torch.full((act_dim,), -1.0))

        self.apply(orthogonal_init)
        orthogonal_init(self.mean_head, gain=0.01)

        with torch.no_grad():
            self.mean_head.bias[0] = 0.2
            self.mean_head.bias[1] = -0.8
            self.mean_head.bias[2] = 0.0

    def forward(self, obs):
        x = torch.relu(self.fc1(obs))
        x = torch.relu(self.fc2(x))

        mean = self.mean_head(x)
        mean = torch.tanh(mean)

        log_std = torch.clamp(self.log_std, -2.5, 0.5)
        std = torch.exp(log_std)

        return mean, std


def scale_action(action, obs=None):
    """
    和训练阶段完全一致的动作映射：
    1. 贪心动作先映射为环境动作
    2. 使用动态 steering 限幅，保证网页端演示更稳
    """
    a = np.array(action, dtype=np.float32)
    a = np.clip(a, -1.0, 1.0)

    throttle = (a[0] + 1.0) / 2.0
    brake = (a[1] + 1.0) / 2.0

    # 避免同时大油门大刹车
    if throttle >= brake:
        brake *= 0.15
    else:
        throttle *= 0.15

    steer_limit = 0.12

    if obs is not None:
        ego_speed = float(np.clip(obs[0], -1.0, 1.0) * 20.0)
        heading_err = abs(float(np.clip(obs[2], -1.0, 1.0))) * np.pi

        if ego_speed < 3.0:
            steer_limit = 0.20
        elif ego_speed < 8.0:
            steer_limit = 0.16
        else:
            steer_limit = 0.12

        if heading_err > 0.18:
            steer_limit = min(0.22, steer_limit + 0.04)

    steering = np.clip(0.18 * a[2], -steer_limit, steer_limit)

    throttle = np.clip(throttle, 0.0, 1.0)
    brake = np.clip(brake, 0.0, 1.0)

    return np.array([throttle, brake, steering], dtype=np.float32)


def select_greedy_action(actor, obs):
    """
    贪心策略：
    对当前连续动作高斯策略来说，最大概率动作就是 mean。
    因此网页端演示每一步都直接取 mean，不采样。
    """
    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

    with torch.no_grad():
        mean, _ = actor(obs_tensor)

    raw_action = mean.squeeze(0).cpu().numpy()
    raw_action = np.clip(raw_action, -1.0, 1.0)
    return raw_action


def test():
    env = SmartsSingleAgentEnv(
        scenario_path="scenarios/mymap",
        headless=False,   # 网页端演示
    )

    actor = Actor(OBS_DIM, ACT_DIM, HIDDEN_DIM).to(device)

    project_dir = Path(__file__).resolve().parent
    actor_path = project_dir / "models" / "ppo_actor.pth"

    print(f"准备加载 Actor 模型: {actor_path}")
    actor.load_state_dict(torch.load(actor_path, map_location=device))
    actor.eval()
    print("Actor 模型加载成功，开始使用贪心策略测试。")

    for episode in range(1, NUM_EPISODES + 1):
        obs, info = env.reset()
        episode_reward = 0.0

        print(f"第 {episode} 回合开始")

        for step in range(MAX_EPISODE_STEPS):
            # 贪心：直接取 mean
            raw_action = select_greedy_action(actor, obs)

            # 再映射成环境动作
            env_action = scale_action(raw_action, obs)

            obs, reward, done, info = env.step(env_action)
            episode_reward += reward

            print(
                f"episode={episode}, "
                f"step={step}, "
                f"reward={reward:.6f}, "
                f"raw_action={raw_action}, "
                f"env_action={env_action}"
            )

            if done:
                print(f"第 {episode} 回合结束，累计奖励 = {episode_reward:.6f}")
                break

    env.close()
    print("测试结束，环境已关闭。")


if __name__ == "__main__":
    test()
