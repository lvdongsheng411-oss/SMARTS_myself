# 后面真正训练 PPO 用


import math
import numpy as np

from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

from r1_project.env_wrapper import SmartsSingleAgentEnv


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




# 观测维度
OBS_DIM = 10

# 动作维度
# 我的动作是 [throttle, brake, steering]
ACT_DIM = 3

# 隐藏层维度
HIDDEN_DIM = 128

# 学习率
LR_ACTOR = 3e-4
LR_CRITIC = 3e-4

# 折扣因子 gamma
GAMMA = 0.99#是否太大？可以考虑调小

# GAE 参数 lambda
LAMBDA = 0.95

# PPO clip 系数
CLIP_EPS = 0.2

# 每轮收集多少步数据
ROLLOUT_STEPS = 1024

# PPO 每轮更新多少次
UPDATE_EPOCHS = 10

# 每次更新的小批量大小
MINI_BATCH_SIZE = 256

# 总训练轮数
TOTAL_UPDATES = 50

# 单回合最大步数
MAX_EPISODE_STEPS = 300



# 构建 Actor 网络

class Actor(nn.Module):
    """
    策略网络：
    输入观测向量 obs
    输出动作分布参数（均值 mean 和对数标准差 log_std）
    """

    def __init__(self, obs_dim, act_dim, hidden_dim):
        super().__init__()


        self.fc1 = nn.Linear(obs_dim, hidden_dim)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        # 输出动作均值
        self.mean_head = nn.Linear(hidden_dim, act_dim)

        # 对数标准差，直接设为可学习参数
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, obs):

        x = torch.relu(self.fc1(obs))

        x = torch.relu(self.fc2(x))

        # 输出动作均值
        mean = self.mean_head(x)


        mean = torch.tanh(mean)


        std = torch.exp(self.log_std)

        return mean, std



# 构建 Critic 网络


class Critic(nn.Module):
    """
    价值网络：
    输入观测 obs
    输出状态价值 V(s)
    """

    def __init__(self, obs_dim, hidden_dim):
        super().__init__()

       
        self.fc1 = nn.Linear(obs_dim, hidden_dim)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        # 输出一个标量价值
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, obs):

        x = torch.relu(self.fc1(obs))

        x = torch.relu(self.fc2(x))

        # 输出状态价值
        value = self.value_head(x)

        return value



#  动作缩放函数

def scale_action(action):
    """
    把 Actor 输出的 [-1, 1] 动作，
    转换成 SMARTS 环境需要的动作格式：

    throttle: [0, 1]
    brake:    [0, 1]
    steering: [-1, 1]
    """


    a = np.array(action, dtype=np.float32)


    throttle = (a[0] + 1.0) / 2.0


    brake = (a[1] + 1.0) / 2.0

    steering = np.clip(a[2], -1.0, 1.0)

    throttle = np.clip(throttle, 0.0, 1.0)
    brake = np.clip(brake, 0.0, 1.0)

    return np.array([throttle, brake, steering], dtype=np.float32)



# 采样动作函数


def sample_action(actor, obs):
    """
    用当前策略网络采样动作
    """

    # 把观测转成 tensor，并加 batch 维
    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

    # 前向传播得到均值和标准差
    mean, std = actor(obs_tensor)

    # 构造高斯分布
    dist = Normal(mean, std)

    # 重参数采样动作
    action = dist.sample()

    # 计算动作对数概率
    log_prob = dist.log_prob(action).sum(dim=-1)

    # 去掉 batch 维，转回 numpy
    action_np = action.squeeze(0).detach().cpu().numpy()

    # 返回动作、对数概率
    return action_np, log_prob.item()


# 经验缓存


class RolloutBuffer:
    """
    用于存储一段 rollout 采样到的数据
    """

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



#  计算 GAE 优势函数


def compute_gae(rewards, dones, values, last_value, gamma=0.99, lam=0.95):
    """
    根据 rewards / dones / values 计算：
    1. advantages
    2. returns
    """

    advantages = []
    gae = 0.0

    # values 后面补一个最后状态价值
    values = values + [last_value]

    # 从后往前算
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages.insert(0, gae)

    # returns = advantages + values
    returns = [adv + v for adv, v in zip(advantages, values[:-1])]

    return advantages, returns



# PPO 更新函数

def ppo_update(actor, critic, actor_optimizer, critic_optimizer, buffer):
    """
    使用 buffer 中采样到的数据，对 actor / critic 做 PPO 更新
    """

    # 先把 list 转成 tensor
    obs = torch.tensor(np.array(buffer.obs), dtype=torch.float32, device=device)
    actions = torch.tensor(np.array(buffer.actions), dtype=torch.float32, device=device)
    old_log_probs = torch.tensor(np.array(buffer.log_probs), dtype=torch.float32, device=device)

    rewards = buffer.rewards
    dones = buffer.dones
    values = buffer.values

    # 计算最后一个状态的价值（这里简单设为 0，如果 done 了也合理）
    last_value = 0.0

    # 计算优势和回报
    advantages, returns = compute_gae(rewards, dones, values, last_value, GAMMA, LAMBDA)

    # 转 tensor
    advantages = torch.tensor(np.array(advantages), dtype=torch.float32, device=device)
    returns = torch.tensor(np.array(returns), dtype=torch.float32, device=device)

    # 标准化优势函数，PPO 常见技巧
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # 数据总量
    data_size = obs.size(0)

    # 做多轮 PPO 更新
    for _ in range(UPDATE_EPOCHS):

        # 打乱索引
        indices = np.arange(data_size)
        np.random.shuffle(indices)

        # 分小批量训练
        for start in range(0, data_size, MINI_BATCH_SIZE):
            end = start + MINI_BATCH_SIZE
            batch_idx = indices[start:end]

            # 取一个 mini-batch
            batch_obs = obs[batch_idx]
            batch_actions = actions[batch_idx]
            batch_old_log_probs = old_log_probs[batch_idx]
            batch_advantages = advantages[batch_idx]
            batch_returns = returns[batch_idx]

            # 更新 Actor 
            mean, std = actor(batch_obs)
            dist = Normal(mean, std)


            new_log_probs = dist.log_prob(batch_actions).sum(dim=-1)


            ratio = torch.exp(new_log_probs - batch_old_log_probs)

 
            surr1 = ratio * batch_advantages
            surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * batch_advantages

      
            actor_loss = -torch.min(surr1, surr2).mean()

 
            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            values_pred = critic(batch_obs).squeeze(-1)

            critic_loss = ((values_pred - batch_returns) ** 2).mean()

  
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()


#主函数


def train():
    """
    PPO 主训练流程
    """

    # 创建环境
    env = SmartsSingleAgentEnv(
        scenario_path="scenarios/mymap",
        headless=True,   # 训练时通常先关闭网页，跑得更稳
    )

    # 创建 Actor / Critic
    actor = Actor(OBS_DIM, ACT_DIM, HIDDEN_DIM).to(device)
    critic = Critic(OBS_DIM, HIDDEN_DIM).to(device)

    actor_optimizer = optim.Adam(actor.parameters(), lr=LR_ACTOR)
    critic_optimizer = optim.Adam(critic.parameters(), lr=LR_CRITIC)

    # 创建 rollout buffer
    buffer = RolloutBuffer()

    # 重置环境
    obs, _ = env.reset()

    # 记录当前 episode reward
    episode_reward = 0.0

    # 训练循环
    for update in range(TOTAL_UPDATES):
        buffer.clear()

        # 每轮先收集一批数据
        for step in range(ROLLOUT_STEPS):

            # 把当前 obs 转 tensor，送入 critic 估值
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            value = critic(obs_tensor).item()

            # 采样动作
            raw_action, log_prob = sample_action(actor, obs)

            # 把动作缩放到环境需要的格式
            env_action = scale_action(raw_action)

            # 与环境交互
            next_obs, reward, done, _ = env.step(env_action)

            # 存数据
            buffer.obs.append(obs)
            buffer.actions.append(raw_action)
            buffer.log_probs.append(log_prob)
            buffer.rewards.append(reward)
            buffer.dones.append(float(done))
            buffer.values.append(value)

            # 累积回合奖励
            episode_reward += reward

            # 状态更新
            obs = next_obs

            # 如果一回合结束，就 reset
            if done:
                print(f"update={update}, episode_reward={episode_reward:.3f}")
                obs, _ = env.reset()
                episode_reward = 0.0

        # 每收集完一批数据，做一次 PPO 更新
        ppo_update(actor, critic, actor_optimizer, critic_optimizer, buffer)

        print(f"PPO update {update + 1}/{TOTAL_UPDATES} 完成")

    env.close()

    # 保存模型
    # 训练结束后保存模型


    # 获取当前 train_ppo.py 文件所在目录
    project_dir = Path(__file__).resolve().parent

    # 在 r1_project 下创建 models 文件夹路径
    model_dir = project_dir / "models"

    # 如果 models 文件夹不存在，就自动创建
    model_dir.mkdir(parents=True, exist_ok=True)

    # 定义 actor 模型保存路径
    actor_path = model_dir / "ppo_actor.pth"

    # 定义 critic 模型保存路径
    critic_path = model_dir / "ppo_critic.pth"

    # 保存 actor 参数
    torch.save(actor.state_dict(), actor_path)

    # 保存 critic 参数
    torch.save(critic.state_dict(), critic_path)

    print(f"训练结束，Actor 模型已保存到: {actor_path}")
    print(f"训练结束，Critic 模型已保存到: {critic_path}")

    print("训练结束，模型已保存。")



if __name__ == "__main__":
    train()
