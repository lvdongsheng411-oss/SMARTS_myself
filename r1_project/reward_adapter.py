#这里的奖励只是最初设计的奖励，后续要一直更新，
#建立多目标权衡机制，调整各维度权重，防止惩罚过重导致保守驾驶或奖励失衡引发激进行为
#最终引导智能体在复杂交互中形成稳健果断的决策风格，平衡安全、效率与舒适。
import math

def _clip(x, low, high):
    
    return max(low, min(high, x))


def _safe_first_waypoint(waypoint_paths):

    if waypoint_paths is None:
        return None

    
    if isinstance(waypoint_paths, dict):
        values = list(waypoint_paths.values())
        if not values:
            return None

        first_path = values[0]
        if hasattr(first_path, "__len__") and len(first_path) > 0:
            return first_path[0]
        return None

   
    if isinstance(waypoint_paths, (list, tuple)):
        if len(waypoint_paths) == 0:
            return None

        first_path = waypoint_paths[0]
        if hasattr(first_path, "__len__") and len(first_path) > 0:
            return first_path[0]
        return None

   
    try:
        paths = list(waypoint_paths)
        if len(paths) == 0:
            return None

        first_path = paths[0]
        if hasattr(first_path, "__len__") and len(first_path) > 0:
            return first_path[0]
    except Exception:
        pass

  
    return None


def _angle_diff(a, b):
   
    d = a - b

    while d > math.pi:
        d -= 2 * math.pi

    while d < -math.pi:
        d += 2 * math.pi

    return d


def compute_reward(agent_obs, env_reward, action=None, prev_action=None):
    """
    计算当前时刻的自定义奖励。

    参数说明：
        agent_obs   : 当前智能体的原始观测
        env_reward  : SMARTS 环境自带的 reward
        action      : 当前动作 [throttle, brake, steering]，即油门、刹车、转向
        prev_action : 上一时刻动作 [throttle, brake, steering]

    当前 reward 的设计目标：
        1. 鼓励车辆向前走
        2. 鼓励合理速度，避免停着不动
        3. 惩罚偏离车道中心
        4. 惩罚车头方向与道路方向差太大
        5. 惩罚动作过于剧烈，减少抖动
        6. 对碰撞、出路、逆行等严重事件做强惩罚
        7. 对到达目标给大额奖励
    """

   
    #  初始化总奖励

    reward = 0.0

    # 保留一部分环境默认 reward
   

    reward += 0.3 * float(env_reward)

  
    #  读取自车状态
    # 自车状态对象
    ego = agent_obs["ego_vehicle_state"]

    # 当前速度
    ego_speed = float(getattr(ego, "speed", 0.0))

    # 当前坐标
    ego_pos = getattr(ego, "position", [0.0, 0.0, 0.0])
    ego_x = float(ego_pos[0])
    ego_y = float(ego_pos[1])

    # 当前航向角
    ego_heading = float(getattr(ego, "heading", 0.0))

    
    # 读取 waypoint 信息

    waypoint_paths = agent_obs.get("waypoint_paths", None)

    # 取第一个 waypoint，作为最直接的道路参考
    wp = _safe_first_waypoint(waypoint_paths)

    if wp is not None:
        # waypoint 的坐标
        wp_pos = getattr(wp, "pos", [ego_x, ego_y])
        wp_x = float(wp_pos[0])
        wp_y = float(wp_pos[1])

        # waypoint 的航向角，可以近似看作“道路方向”
        wp_heading = float(getattr(wp, "heading", 0.0))

        # 当前道路限速
        speed_limit = float(getattr(wp, "speed_limit", 13.89))

        # ego 到 waypoint 的距离，当前先近似作“车道中心偏移”
        lane_offset = math.hypot(ego_x - wp_x, ego_y - wp_y)

        # 自车航向与道路航向的差值
        heading_error = _angle_diff(ego_heading, wp_heading)

    else:
        # 如果取不到 waypoint，就给默认值，防止程序报错
        speed_limit = 13.89
        lane_offset = 0.0
        heading_error = 0.0

    
    # 速度奖励
   

    # 给速度一个正奖励，鼓励车往前走，系数不能太大，否则会导致模型只顾加速
    reward += 0.03 * ego_speed

    # 如果速度过低，说明车可能停住，给惩罚
    if ego_speed < 1.0:
        reward -= 0.5

    # 如果速度明显超过限速，给惩罚
    # 超速惩罚也有
    if ego_speed > speed_limit * 1.1:
        reward -= 0.2 * (ego_speed - speed_limit * 1.1)

    
    #  车道偏移惩罚
    # 离车道中心越远，惩罚越大，这里主要用来减少压线、漂移、出道  
    reward -= 2 * lane_offset

    # 如果偏移已经明显过大，再额外加罚
    if lane_offset > 2.0:
        reward -= 2.0
      
    #  航向误差惩罚
  
    # 如果车头方向和道路方向差太大，就给惩罚
    reward -= 0.8 * abs(heading_error)

    # 如果航向误差已经比较大，再额外扣一点
    if abs(heading_error) > 0.5:
        reward -= 0.5

   
    # 动作平滑性惩罚
    if action is not None:
        # 当前油门
        throttle = float(action[0])

        # 当前刹车
        brake = float(action[1])

        # 当前转向
        steering = float(action[2])

        # 如果同时给了较大的油门和刹车，说明动作不合理
        if throttle > 0.2 and brake > 0.2:
            reward -= 0.5

        # 转向绝对值太大，也稍微惩罚一下
        # 避免模型学出“大幅乱打方向盘”
        reward -= 0.1 * abs(steering)

        # 如果上一时刻动作也存在，就进一步约束动作变化速度
        if prev_action is not None:
            prev_throttle = float(prev_action[0])
            prev_brake = float(prev_action[1])
            prev_steering = float(prev_action[2])

            # 惩罚方向盘变化过快，减少左右抖动
            reward -= 0.2 * abs(steering - prev_steering)

            # 惩罚油门变化过快
            reward -= 0.05 * abs(throttle - prev_throttle)

            # 惩罚刹车变化过快
            reward -= 0.05 * abs(brake - prev_brake)

  
    #事件类惩罚 / 奖励

    events = agent_obs.get("events", None)

    if events is not None:
        # 碰撞：大惩罚
        if getattr(events, "collisions", []):
            reward -= 100.0

        # 驶出道路：大惩罚
        if getattr(events, "off_road", False):
            reward -= 80.0

        # 到达目标：大奖励
        if getattr(events, "reached_goal", False):
            reward += 100.0

        # 逆行：惩罚
        if getattr(events, "wrong_way", False):
            reward -= 60.0

    # 防止奖励过大或过小，影响训练稳定性
    reward = _clip(reward, -200.0, 200.0)

    return reward
