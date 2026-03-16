import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

from r1_project.env_wrapper import SmartsSingleAgentEnv


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 现在 obs_adapter.py 输出 16 维
OBS_DIM = 16
ACT_DIM = 3
HIDDEN_DIM = 128

LR_ACTOR = 3e-4
LR_CRITIC = 3e-4

GAMMA = 0.99
LAMBDA = 0.95
CLIP_EPS = 0.2

ROLLOUT_STEPS = 1024
UPDATE_EPOCHS = 10
MINI_BATCH_SIZE = 256
TOTAL_UPDATES = 150

ENTROPY_COEF = 0.005
VALUE_COEF = 0.5
MAX_GRAD_NORM = 0.5


def orthogonal_init(layer, gain=np.sqrt(2)):
    if isinstance(layer, nn.Linear):
        nn.init.orthogonal_(layer.weight, gain=gain)
        nn.init.constant_(layer.bias, 0.0)


class Actor(nn.Module):
    """策略网络：输入观测，输出动作分布参数 mean 和 std。"""

    def __init__(self, obs_dim, act_dim, hidden_dim):
        super().__init__()

        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_head = nn.Linear(hidden_dim, act_dim)

        # log_std 设小一些，初期探索更稳定
        self.log_std = nn.Parameter(torch.full((act_dim,), -1.0))

        self.apply(orthogonal_init)
        orthogonal_init(self.mean_head, gain=0.01)

        with torch.no_grad():
            # 初始策略稍微偏向：轻给油、少刹车、转向居中
            self.mean_head.bias[0] = 0.2
            self.mean_head.bias[1] = -0.8
            self.mean_head.bias[2] = 0.0

    def forward(self, obs):
        x = torch.relu(self.fc1(obs))
        x = torch.relu(self.fc2(x))

        mean = self.mean_head(x)
        mean = torch.tanh(mean)

        # 防止 std 过大或过小
        log_std = torch.clamp(self.log_std, -2.5, 0.5)
        std = torch.exp(log_std)

        return mean, std


class Critic(nn.Module):
    """价值网络：输入观测，输出状态价值。"""

    def __init__(self, obs_dim, hidden_dim):
        super().__init__()

        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

        self.apply(orthogonal_init)
        orthogonal_init(self.value_head, gain=1.0)

    def forward(self, obs):
        x = torch.relu(self.fc1(obs))
        x = torch.relu(self.fc2(x))
        value = self.value_head(x)
        return value


def scale_action(action, obs=None):
    """
    将 Actor 输出的 [-1, 1] 原始动作映射成环境动作 [throttle, brake, steering]。
    这里做“动态转向限幅”，速度越快转向越保守；偏航较大时允许略增大转向上限。
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
        # obs[0] 是归一化后的 ego_speed，按 20m/s 反推
        ego_speed = float(np.clip(obs[0], -1.0, 1.0) * 20.0)
        heading_err = abs(float(np.clip(obs[2], -1.0, 1.0))) * np.pi

        if ego_speed < 3.0:
            steer_limit = 0.20
        elif ego_speed < 8.0:
            steer_limit = 0.16
        else:
            steer_limit = 0.12

        # 如果偏航较大，允许额外增加一点转向能力
        if heading_err > 0.18:
            steer_limit = min(0.22, steer_limit + 0.04)

    steering = np.clip(0.18 * a[2], -steer_limit, steer_limit)

    throttle = np.clip(throttle, 0.0, 1.0)
    brake = np.clip(brake, 0.0, 1.0)

    return np.array([throttle, brake, steering], dtype=np.float32)


def sample_action(actor, obs):
    """
    训练阶段从高斯策略中采样动作。
    返回：
    - clipped_raw_action: PPO 内部使用的动作（已裁剪到 [-1, 1]）
    - log_prob: 对应对数概率
    """
    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

    mean, std = actor(obs_tensor)
    dist = Normal(mean, std)

    raw_action = dist.sample()
    clipped_action = torch.clamp(raw_action, -1.0, 1.0)

    log_prob = dist.log_prob(clipped_action).sum(dim=-1)

    action_np = clipped_action.squeeze(0).detach().cpu().numpy()
    return action_np, log_prob.item()


class RolloutBuffer:
    def __init__(self):
        self.obs = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def clear(self):
        self.obs.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()


def compute_gae(rewards, dones, values, last_value, gamma=0.99, lam=0.95):
    """
    GAE 计算，正确使用最后状态价值 bootstrap。
    """
    advantages = []
    gae = 0.0

    values = values + [last_value]

    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages.insert(0, gae)

    returns = [adv + v for adv, v in zip(advantages, values[:-1])]
    return advantages, returns


def ppo_update(actor, critic, actor_optimizer, critic_optimizer, buffer, last_obs, last_done):
    obs = torch.tensor(np.array(buffer.obs), dtype=torch.float32, device=device)
    actions = torch.tensor(np.array(buffer.actions), dtype=torch.float32, device=device)
    old_log_probs = torch.tensor(np.array(buffer.log_probs), dtype=torch.float32, device=device)

    rewards = buffer.rewards
    dones = buffer.dones
    values = buffer.values

    # 正确 bootstrap：只有 rollout 最后不是 done，才估计 last_value
    if last_done:
        last_value = 0.0
    else:
        with torch.no_grad():
            last_obs_tensor = torch.tensor(last_obs, dtype=torch.float32, device=device).unsqueeze(0)
            last_value = critic(last_obs_tensor).item()

    advantages, returns = compute_gae(rewards, dones, values, last_value, GAMMA, LAMBDA)

    advantages = torch.tensor(np.array(advantages), dtype=torch.float32, device=device)
    returns = torch.tensor(np.array(returns), dtype=torch.float32, device=device)

    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    data_size = obs.size(0)

    actor_loss_mean = 0.0
    critic_loss_mean = 0.0
    batch_count = 0

    for _ in range(UPDATE_EPOCHS):
        indices = np.arange(data_size)
        np.random.shuffle(indices)

        for start in range(0, data_size, MINI_BATCH_SIZE):
            end = start + MINI_BATCH_SIZE
            batch_idx = indices[start:end]

            batch_obs = obs[batch_idx]
            batch_actions = actions[batch_idx]
            batch_old_log_probs = old_log_probs[batch_idx]
            batch_advantages = advantages[batch_idx]
            batch_returns = returns[batch_idx]

            # ========== Actor ==========
            mean, std = actor(batch_obs)
            dist = Normal(mean, std)

            new_log_probs = dist.log_prob(batch_actions).sum(dim=-1)
            ratio = torch.exp(new_log_probs - batch_old_log_probs)

            surr1 = ratio * batch_advantages
            surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * batch_advantages

            entropy = dist.entropy().sum(dim=-1).mean()
            actor_loss = -torch.min(surr1, surr2).mean() - ENTROPY_COEF * entropy

            actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), MAX_GRAD_NORM)
            actor_optimizer.step()

            # ========== Critic ==========
            values_pred = critic(batch_obs).squeeze(-1)
            critic_loss = ((values_pred - batch_returns) ** 2).mean()

            critic_optimizer.zero_grad()
            (VALUE_COEF * critic_loss).backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), MAX_GRAD_NORM)
            critic_optimizer.step()

            actor_loss_mean += actor_loss.item()
            critic_loss_mean += critic_loss.item()
            batch_count += 1

    actor_loss_mean /= max(batch_count, 1)
    critic_loss_mean /= max(batch_count, 1)

    return actor_loss_mean, critic_loss_mean


def train():
    env = SmartsSingleAgentEnv(
        scenario_path="scenarios/mymap",
        headless=True,
    )

    actor = Actor(OBS_DIM, ACT_DIM, HIDDEN_DIM).to(device)
    critic = Critic(OBS_DIM, HIDDEN_DIM).to(device)

    actor_optimizer = optim.Adam(actor.parameters(), lr=LR_ACTOR)
    critic_optimizer = optim.Adam(critic.parameters(), lr=LR_CRITIC)

    buffer = RolloutBuffer()

    obs, _ = env.reset()
    episode_reward = 0.0
    episode_count = 0

    for update in range(TOTAL_UPDATES):
        buffer.clear()

        last_done = False

        for step in range(ROLLOUT_STEPS):
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                value = critic(obs_tensor).item()

            raw_action, log_prob = sample_action(actor, obs)
            env_action = scale_action(raw_action, obs)

            next_obs, reward, done, _ = env.step(env_action)

            buffer.obs.append(obs)
            buffer.actions.append(raw_action)
            buffer.log_probs.append(log_prob)
            buffer.rewards.append(float(reward))
            buffer.dones.append(float(done))
            buffer.values.append(float(value))

            episode_reward += reward
            obs = next_obs
            last_done = done

            if done:
                episode_count += 1
                print(
                    f"[Episode {episode_count}] "
                    f"update={update + 1}, episode_reward={episode_reward:.3f}"
                )
                obs, _ = env.reset()
                episode_reward = 0.0

        actor_loss, critic_loss = ppo_update(
            actor,
            critic,
            actor_optimizer,
            critic_optimizer,
            buffer,
            last_obs=obs,
            last_done=last_done,
        )

        print(
            f"PPO update {update + 1}/{TOTAL_UPDATES} 完成 | "
            f"actor_loss={actor_loss:.6f}, critic_loss={critic_loss:.6f}"
        )

    env.close()

    project_dir = Path(__file__).resolve().parent
    model_dir = project_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    actor_path = model_dir / "ppo_actor.pth"
    critic_path = model_dir / "ppo_critic.pth"

    torch.save(actor.state_dict(), actor_path)
    torch.save(critic.state_dict(), critic_path)

    print(f"训练结束，Actor 模型已保存到: {actor_path}")
    print(f"训练结束，Critic 模型已保存到: {critic_path}")
    print("训练结束，模型已保存。")


if __name__ == "__main__":
    train()
