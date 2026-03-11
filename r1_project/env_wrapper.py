#导入一些库，引入函数
import gymnasium as gym
import numpy as np
from smarts.core.agent_interface import AgentInterface, AgentType
from r1_project.obs_adapter import extract_obs
from r1_project.reward_adapter import compute_reward
class SmartsSingleAgentEnv:
    
    def __init__(self, scenario_path="scenarios/mymap", headless=True):

        # 单智能体的名字
        self.agent_id = "Agent-001"

        # 保存场景路径
        self.scenario_path = scenario_path
        self.headless = headless

        
        self.prev_action = None

        # 为当前智能体创建接口
        # 这里使用 SMARTS 自带的单车预设接口 Loner
        agent_interfaces = {
            self.agent_id: AgentInterface.from_type(
                AgentType.Loner,
                max_episode_steps=300,
            )
        }

        # 创建 SMARTS 环境
        self.env = gym.make(
            "smarts.env:hiway-v1",
            scenarios=[self.scenario_path],
            agent_interfaces=agent_interfaces,
            headless=self.headless,
        )

    def reset(self):
     

        # 每个新回合开始时，把上一动作清空
        self.prev_action = None

        # 调用底层环境 
        raw_obs, info = self.env.reset()


        agent_raw_obs = raw_obs[self.agent_id]

        obs_vec = extract_obs(agent_raw_obs)

        return obs_vec, info

    def step(self, action):
        ""      
            action : 当前动作，格式应为 [throttle, brake, steering]
        返回：
            obs_vec : 下一时刻观测向量
            reward  : 自定义奖励
            done    : 当前回合是否结束
            info    : 附加信息
        """


        action = np.asarray(action, dtype=np.float32)

        # 调用 SMARTS 原始环境 step
        # SMARTS 需要的动作格式是：{agent_id: action}
        raw_obs, rewards, terminated, truncated, info = self.env.step(
            {self.agent_id: action}
        )

        # 取当前智能体的原始观测
        agent_obs = raw_obs[self.agent_id]

        obs_vec = extract_obs(agent_obs)

        # 用 reward 函数计算 reward
        # 这里把当前动作 action 和上一时刻动作 self.prev_action 都传进去
        reward = compute_reward(
            agent_obs=agent_obs,
            env_reward=rewards[self.agent_id],
            action=action,
            prev_action=self.prev_action,
        )
        
        done = terminated["__all__"] or truncated["__all__"]

        # 当前 step 结束后，把当前动作保存为“上一时刻动作”
        # 供下一步计算平滑性惩罚使用
        self.prev_action = action.copy()

        return obs_vec, reward, done, info

    def close(self):
       
        self.env.close()
