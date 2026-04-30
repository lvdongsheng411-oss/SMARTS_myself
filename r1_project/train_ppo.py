import csv
import time
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

from r1_project.env_wrapper import SmartsSingleAgentEnv


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
TOTAL_UPDATES = 800
REWARD_LOG_FILENAME = "training_reward_log.csv"

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

    这里采用动态 steering 限幅：
    1. 低速时允许稍大转向，方便转弯和纠偏
    2. 高速时限制转向，减少左右摆动
    3. 航向误差较大时，适当放宽转向上限
    """
    a = np.array(action, dtype=np.float32)
    a = np.clip(a, -1.0, 1.0)

    throttle = (a[0] + 1.0) / 2.0
    brake = (a[1] + 1.0) / 2.0

    # 避免同时大油门和大刹车
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


def sample_action(actor, obs):
    """
    训练阶段从高斯策略中采样动作。
    返回：
    1. raw_action：PPO 内部动作，范围裁剪到 [-1, 1]
    2. log_prob：该动作在当前策略下的对数概率
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
    """PPO 采样缓存区。"""

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
    GAE 优势估计。
    advantages 用于更新 Actor。
    returns 用于训练 Critic。
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

    # 如果 rollout 最后不是终止状态，用 Critic 估计最后一个状态价值
    if last_done:
        last_value = 0.0
    else:
        with torch.no_grad():
            last_obs_tensor = torch.tensor(last_obs, dtype=torch.float32, device=device).unsqueeze(0)
            last_value = critic(last_obs_tensor).item()

    advantages, returns = compute_gae(
        rewards,
        dones,
        values,
        last_value,
        GAMMA,
        LAMBDA,
    )

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

            # =========================
            # 更新 Actor
            # =========================
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

            # =========================
            # 更新 Critic
            # =========================
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
    train_start_time = time.time()

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
    episode_steps = 0
    episode_count = 0
    episode_rewards = []

    project_dir = Path(__file__).resolve().parent
    reward_log_path = project_dir / REWARD_LOG_FILENAME

    with open(reward_log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "episode",
            "update",
            "episode_reward",
            "mean_reward_all",
            "mean_reward_20",
            "mean_reward_100",
            "episode_steps",
        ])

    print(f"训练奖励日志将保存到: {reward_log_path}")

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
            episode_steps += 1
            obs = next_obs
            last_done = done

            if done:
                episode_count += 1

                episode_rewards.append(float(episode_reward))

                mean_reward_all = float(np.mean(episode_rewards))
                mean_reward_20 = float(np.mean(episode_rewards[-20:]))
                mean_reward_100 = float(np.mean(episode_rewards[-100:]))

                with open(reward_log_path, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        episode_count,
                        update + 1,
                        float(episode_reward),
                        mean_reward_all,
                        mean_reward_20,
                        mean_reward_100,
                        episode_steps,
                    ])

                print(
                    f"[Episode {episode_count}] "
                    f"update={update + 1}/{TOTAL_UPDATES}, "
                    f"episode_reward={episode_reward:.3f}, "
                    f"mean_reward_all={mean_reward_all:.3f}, "
                    f"mean_reward_20={mean_reward_20:.3f}, "
                    f"mean_reward_100={mean_reward_100:.3f}, "
                    f"episode_steps={episode_steps}"
                )

                obs, _ = env.reset()
                episode_reward = 0.0
                episode_steps = 0

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
            f"已训练回合数={episode_count}, "
            f"actor_loss={actor_loss:.6f}, "
            f"critic_loss={critic_loss:.6f}"
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

    train_end_time = time.time()
    elapsed_seconds = int(train_end_time - train_start_time)

    elapsed_hours = elapsed_seconds // 3600
    elapsed_minutes = (elapsed_seconds % 3600) // 60

    print(f"训练总耗时: {elapsed_hours}h{elapsed_minutes}min")


if __name__ == "__main__":
    train()
