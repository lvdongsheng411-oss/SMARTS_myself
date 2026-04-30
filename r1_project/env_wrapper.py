import gymnasium as gym
import numpy as np

from smarts.core.agent_interface import AgentInterface, AgentType

from r1_project.obs_adapter import extract_obs
from r1_project.reward_adapter import compute_reward


def _get(obj, key, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _extract_goal_position(agent_obs):
    ego = _get(agent_obs, "ego_vehicle_state", {})
    mission = _get(ego, "mission", {})

    goal_position = _get(mission, "goal_position", None)
    if goal_position is not None:
        gp = np.asarray(goal_position)
        if gp.shape[0] >= 2:
            return float(gp[0]), float(gp[1])

    return None


def _compute_goal_distance(agent_obs):
    ego = _get(agent_obs, "ego_vehicle_state", {})
    ego_pos = np.asarray(_get(ego, "position", [0.0, 0.0, 0.0]))

    ego_x = float(ego_pos[0])
    ego_y = float(ego_pos[1])

    goal_pos = _extract_goal_position(agent_obs)
    if goal_pos is None:
        return None

    gx, gy = goal_pos
    return float(np.hypot(gx - ego_x, gy - ego_y))


class SmartsSingleAgentEnv:
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

        if self.agent_id in raw_obs:
            agent_obs = raw_obs[self.agent_id]
            self.last_agent_obs = agent_obs
        else:
            agent_obs = self.last_agent_obs

        if agent_obs is None:
            obs_vec = np.zeros((16,), dtype=np.float32)
            reward = -10.0
            done = True
            return obs_vec, reward, done, info

        current_goal_dist = _compute_goal_distance(agent_obs)
        obs_vec = extract_obs(agent_obs)

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
