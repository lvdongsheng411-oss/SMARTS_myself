import math


def _clip(x, low, high):
    return max(low, min(high, x))


def _angle_diff(a, b):
    d = a - b
    while d > math.pi:
        d -= 2 * math.pi
    while d < -math.pi:
        d += 2 * math.pi
    return d


def _safe_waypoint_paths(waypoint_paths):
    if waypoint_paths is None:
        return []

    if isinstance(waypoint_paths, dict):
        return list(waypoint_paths.values())

    if isinstance(waypoint_paths, (list, tuple)):
        return list(waypoint_paths)

    try:
        return list(waypoint_paths)
    except Exception:
        return []


def _get_waypoint(path, idx):
    if path is None:
        return None
    if not hasattr(path, "__len__") or len(path) == 0:
        return None
    if idx < len(path):
        return path[idx]
    return path[-1]


def compute_reward(
    agent_obs,
    env_reward,
    action=None,
    prev_action=None,
    prev_goal_dist=None,
    current_goal_dist=None,
):
    """
    强化版 reward：
    1. 保留少量 env_reward
    2. 速度奖励
    3. 车道中心奖励
    4. 航向一致性奖励
    5. 动作平滑惩罚
    6. 距离终点进度奖励（新增重点）
    7. 碰撞/越界/逆行惩罚
    8. 到达终点大奖励
    """
    reward = 0.0

    # =========================================================
    # 1. 自车状态
    # =========================================================
    ego = agent_obs["ego_vehicle_state"]

    ego_speed = float(getattr(ego, "speed", 0.0))
    ego_pos = getattr(ego, "position", [0.0, 0.0, 0.0])
    ego_x = float(ego_pos[0])
    ego_y = float(ego_pos[1])
    ego_heading = float(getattr(ego, "heading", 0.0))

    # =========================================================
    # 2. waypoint 信息
    # =========================================================
    waypoint_paths = agent_obs.get("waypoint_paths", None)
    paths = _safe_waypoint_paths(waypoint_paths)

    signed_lane_error = 0.0
    heading_error_now = 0.0
    speed_limit = 13.89

    if len(paths) > 0:
        path0 = paths[0]
        wp0 = _get_waypoint(path0, 0)

        if wp0 is not None:
            wp0_pos = getattr(wp0, "pos", [ego_x, ego_y])
            wp0_x = float(wp0_pos[0])
            wp0_y = float(wp0_pos[1])
            wp0_heading = float(getattr(wp0, "heading", 0.0))

            dx = ego_x - wp0_x
            dy = ego_y - wp0_y

            signed_lane_error = -math.sin(wp0_heading) * dx + math.cos(wp0_heading) * dy
            heading_error_now = _angle_diff(ego_heading, wp0_heading)
            speed_limit = float(getattr(wp0, "speed_limit", 13.89))

    abs_lane_error = abs(signed_lane_error)
    abs_heading_error = abs(heading_error_now)

    # =========================================================
    # 3. 少量保留环境原始奖励
    # =========================================================
    reward += 0.05 * float(env_reward)

    # =========================================================
    # 4. 速度奖励：鼓励接近合理巡航速度，不鼓励过慢也不鼓励盲目超速
    # =========================================================
    target_speed = min(0.75 * speed_limit, 10.0)

    if target_speed < 2.0:
        target_speed = 2.0

    speed_error = abs(ego_speed - target_speed)

    # 速度越接近目标越好
    reward += 1.2 * (1.0 - min(speed_error / max(target_speed, 1.0), 1.0))

    # 极低速轻微惩罚，避免停住不动
    if ego_speed < 0.3:
        reward -= 1.0

    # 超速轻罚
    if ego_speed > speed_limit + 1.5:
        reward -= 1.5

    # =========================================================
    # 5. 车道中心奖励：比“纯重罚”更稳定
    # =========================================================
    lane_center_reward = 1.5 * (1.0 - min(abs_lane_error / 1.5, 1.0))
    reward += lane_center_reward

    if abs_lane_error > 1.2:
        reward -= 2.0
    if abs_lane_error > 2.0:
        reward -= 6.0

    # =========================================================
    # 6. 航向一致性奖励
    # =========================================================
    heading_align_reward = 1.2 * (1.0 - min(abs_heading_error / 0.35, 1.0))
    reward += heading_align_reward

    if abs_heading_error > 0.30:
        reward -= 1.5
    if abs_heading_error > 0.60:
        reward -= 4.0

    # =========================================================
    # 7. 距离终点奖励（新增重点）
    #    progress = 上一步距离 - 当前距离
    #    越接近终点，奖励越高
    # =========================================================
    if prev_goal_dist is not None and current_goal_dist is not None:
        progress = prev_goal_dist - current_goal_dist

        # 每一步朝终点前进就给奖励，后退则扣分
        reward += 2.5 * _clip(progress, -2.0, 2.0)

        # 离终点越近，额外给一点“接近奖励”
        close_bonus = 1.2 * (1.0 - min(current_goal_dist / 80.0, 1.0))
        reward += close_bonus

    # =========================================================
    # 8. 动作平滑 + 更稳定 steering 控制
    # =========================================================
    if action is not None:
        throttle = float(action[0])
        brake = float(action[1])
        steering = float(action[2])

        # 大转向惩罚：速度越高，越不允许大打方向
        if ego_speed < 3.0:
            reward -= 0.30 * abs(steering)
        elif ego_speed < 8.0:
            reward -= 0.70 * abs(steering)
        else:
            reward -= 1.20 * abs(steering)

        # 油门和刹车同时较大，不合理
        if throttle > 0.2 and brake > 0.2:
            reward -= 1.5

        if prev_action is not None:
            prev_throttle = float(prev_action[0])
            prev_brake = float(prev_action[1])
            prev_steering = float(prev_action[2])

            # 转向变化过快，重点惩罚
            reward -= 1.6 * abs(steering - prev_steering)

            # 油门/刹车突变也轻罚
            reward -= 0.25 * abs(throttle - prev_throttle)
            reward -= 0.25 * abs(brake - prev_brake)

    # =========================================================
    # 9. 事件奖励 / 惩罚
    # =========================================================
    events = agent_obs.get("events", None)

    if events is not None:
        if getattr(events, "collisions", []):
            reward -= 150.0

        if getattr(events, "off_road", False):
            reward -= 120.0

        if getattr(events, "wrong_way", False):
            reward -= 40.0

        if getattr(events, "reached_goal", False):
            reward += 250.0

    # =========================================================
    # 10. 最终裁剪
    # =========================================================
    reward = _clip(reward, -200.0, 200.0)
    return reward
