import numpy as np
from pathlib import Path

import torch
import torch.nn as nn

from r1_project.env_wrapper import SmartsSingleAgentEnv


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


def select_greedy_action(actor, obs):
    """
    贪心策略：
    测试演示时不采样，直接取高斯策略 mean。
    """
    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

    with torch.no_grad():
        mean, _ = actor(obs_tensor)

    raw_action = mean.squeeze(0).cpu().numpy()
    raw_action = np.clip(raw_action, -1.0, 1.0)

    return raw_action


def scale_action_for_demo(raw_action, obs, prev_env_action=None):
    """
    演示优化版动作映射：
    1. 油门/刹车互斥
    2. 动态转向限幅
    3. 转向低通滤波
    4. 油门低通滤波
    5. 起步阶段给最小油门，避免车不动
    """

    a = np.array(raw_action, dtype=np.float32)
    a = np.clip(a, -1.0, 1.0)

    # =========================
    # 1. 恢复状态信息
    # =========================
    ego_speed = float(np.clip(obs[0], -1.0, 1.0) * 20.0)
    heading_err = abs(float(np.clip(obs[2], -1.0, 1.0))) * np.pi
    lane_err = abs(float(np.clip(obs[1], -1.0, 1.0)) * 3.0)

    # =========================
    # 2. 油门/刹车映射
    # =========================
    throttle = (a[0] + 1.0) / 2.0
    brake = (a[1] + 1.0) / 2.0

    # 油门和刹车互斥
    if throttle >= brake:
        brake = 0.0
    else:
        throttle = 0.0

    # 起步阶段给一个最小油门，避免演示时卡住
    if ego_speed < 1.0 and brake < 0.2:
        throttle = max(throttle, 0.35)
        brake = 0.0

    # 巡航阶段限制过大油门，让车更稳
    if ego_speed > 8.0:
        throttle = min(throttle, 0.45)

    # =========================
    # 3. 动态 steering 限幅
    # =========================
    if ego_speed < 3.0:
        steer_limit = 0.18
    elif ego_speed < 8.0:
        steer_limit = 0.14
    else:
        steer_limit = 0.10

    # 如果偏航或偏离车道明显，允许多一点纠偏
    if heading_err > 0.20 or lane_err > 0.8:
        steer_limit = min(0.22, steer_limit + 0.04)

    steering = np.clip(0.16 * a[2], -steer_limit, steer_limit)

    # =========================
    # 4. 动作低通滤波
    # =========================
    if prev_env_action is not None:
        prev_throttle = float(prev_env_action[0])
        prev_brake = float(prev_env_action[1])
        prev_steering = float(prev_env_action[2])

        # 转向强平滑，减少左右抖动
        steering = 0.75 * prev_steering + 0.25 * steering

        # 油门轻平滑，避免速度突变
        throttle = 0.60 * prev_throttle + 0.40 * throttle

        # 刹车轻平滑
        brake = 0.60 * prev_brake + 0.40 * brake

    throttle = np.clip(throttle, 0.0, 1.0)
    brake = np.clip(brake, 0.0, 1.0)
    steering = np.clip(steering, -0.25, 0.25)

    return np.array([throttle, brake, steering], dtype=np.float32)


def test():
    env = SmartsSingleAgentEnv(
        scenario_path="scenarios/mymap",
        headless=False,
    )

    actor = Actor(OBS_DIM, ACT_DIM, HIDDEN_DIM).to(device)

    project_dir = Path(__file__).resolve().parent
    actor_path = project_dir / "models" / "ppo_actor.pth"

    print(f"准备加载 Actor 模型: {actor_path}")
    actor.load_state_dict(torch.load(actor_path, map_location=device))
    actor.eval()
    print("Actor 模型加载成功，开始使用演示优化版贪心策略测试。")

    for episode in range(1, NUM_EPISODES + 1):
        obs, info = env.reset()
        episode_reward = 0.0
        prev_env_action = None

        print(f"第 {episode} 回合开始")

        for step in range(MAX_EPISODE_STEPS):
            raw_action = select_greedy_action(actor, obs)

            env_action = scale_action_for_demo(
                raw_action=raw_action,
                obs=obs,
                prev_env_action=prev_env_action,
            )

            obs, reward, done, info = env.step(env_action)
            episode_reward += reward

            prev_env_action = env_action.copy()

            print(
                f"episode={episode}, "
                f"step={step}, "
                f"reward={reward:.3f}, "
                f"speed={obs[0] * 20.0:.2f}, "
                f"lane_err={obs[1] * 3.0:.2f}, "
                f"heading_err={obs[2] * np.pi:.2f}, "
                f"env_action={env_action}"
            )

            if done:
                print(f"第 {episode} 回合结束，累计奖励 = {episode_reward:.3f}")
                break

    env.close()
    print("测试结束，环境已关闭。")


if __name__ == "__main__":
    test()
