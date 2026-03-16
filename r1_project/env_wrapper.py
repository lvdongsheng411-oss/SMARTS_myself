import gymnasium as gym
import numpy as np

from smarts.core.agent_interface import AgentInterface, AgentType

from r1_project.obs_adapter import extract_obs
from r1_project.reward_adapter import compute_reward


def _extract_goal_position(agent_obs):
    """尽量从观测中解析目标点位置。"""
    candidates = []

    ego = agent_obs.get("ego_vehicle_state", None)
    if ego is not None:
        mission = getattr(ego, "mission", None)
        if mission is not None:
            candidates.append(getattr(mission, "goal", None))

    candidates.append(agent_obs.get("goal", None))
    candidates.append(agent_obs.get("mission", None))

    for obj in candidates:
        if obj is None:
            continue

        goal = getattr(obj, "goal", obj)

        for attr in ["position", "pos", "center", "centroid"]:
            p = getattr(goal, attr, None)
            if p is None:
                continue

            if isinstance(p, (list, tuple, np.ndarray)) and len(p) >= 2:
                return float(p[0]), float(p[1])

            x = getattr(p, "x", None)
            y = getattr(p, "y", None)
            if x is not None and y is not None:
                return float(x), float(y)

        x = getattr(goal, "x", None)
        y = getattr(goal, "y", None)
        if x is not None and y is not None:
            return float(x), float(y)

    return None


def _compute_goal_distance(agent_obs):
    """计算自车到终点的距离；无法获取则返回 None。"""
    ego = agent_obs.get("ego_vehicle_state", None)
    if ego is None:
        return None

    ego_pos = getattr(ego, "position", [0.0, 0.0, 0.0])
    ego_x = float(ego_pos[0])
    ego_y = float(ego_pos[1])

    goal_pos = _extract_goal_position(agent_obs)
    if goal_pos is None:
        return None

    gx, gy = goal_pos
    return float(np.hypot(gx - ego_x, gy - ego_y))


class SmartsSingleAgentEnv:
    """
    对 SMARTS 环境做二次封装：
    1. 原始 observation -> 固定长度向量
    2. 使用自定义 reward
    3. 统一 reset()/step() 输出
    4. 记录 prev_action
    5. 记录 prev_goal_dist，用于“距离终点进度奖励”
    """

    def __init__(self, scenario_path="scenarios/mymap", headless=False):
        self.agent_id = "Agent-001"
        self.scenario_path = scenario_path
        self.headless = headless

        self.prev_action = None
        self.prev_goal_dist = None
        self.last_agent_obs = None

        agent_interfaces = {
            self.agent_id: AgentInterface.from_type(
                AgentType.Loner,
                max_episode_steps=300,
            )
        }

        self.env = gym.make(
            "smarts.env:hiway-v1",
            scenarios=[self.scenario_path],
            agent_interfaces=agent_interfaces,
            headless=self.headless,
        )

    def reset(self):
        self.prev_action = None
        self.prev_goal_dist = None
        self.last_agent_obs = None

        raw_obs, info = self.env.reset()
        agent_raw_obs = raw_obs[self.agent_id]
        self.last_agent_obs = agent_raw_obs

        self.prev_goal_dist = _compute_goal_distance(agent_raw_obs)
        obs_vec = extract_obs(agent_raw_obs)

        return obs_vec, info

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)

        raw_obs, rewards, terminated, truncated, info = self.env.step(
            {self.agent_id: action}
        )

        # 某些终止情况下，agent 观测可能已经不在字典里了
        if self.agent_id in raw_obs:
            agent_obs = raw_obs[self.agent_id]
            self.last_agent_obs = agent_obs
        else:
            agent_obs = self.last_agent_obs

        if agent_obs is None:
            # 极端保护：若确实没有观测，就给一个全零向量
            obs_vec = np.zeros((16,), dtype=np.float32)
            reward = -10.0
            done = True
            return obs_vec, reward, done, info

        current_goal_dist = _compute_goal_distance(agent_obs)
        obs_vec = extract_obs(agent_obs)

        # rewards 可能也是 dict
        env_reward = rewards[self.agent_id] if isinstance(rewards, dict) else rewards

        reward = compute_reward(
            agent_obs=agent_obs,
            env_reward=env_reward,
            action=action,
            prev_action=self.prev_action,
            prev_goal_dist=self.prev_goal_dist,
            current_goal_dist=current_goal_dist,
        )

        done = terminated["__all__"] or truncated["__all__"]

        self.prev_action = action.copy()
        self.prev_goal_dist = current_goal_dist

        return obs_vec, reward, done, info

    def close(self):
        self.env.close()
