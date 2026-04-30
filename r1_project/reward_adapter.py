import math
import numpy as np


def _get(obj, key, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _clip(x, low, high):
    return max(low, min(high, x))


def _angle_diff(a, b):
    d = float(a) - float(b)
    while d > math.pi:
        d -= 2 * math.pi
    while d < -math.pi:
        d += 2 * math.pi
    return d


def _get_waypoint_heading(waypoint_paths, path_idx=0, wp_idx=0, default=0.0):
    if not isinstance(waypoint_paths, dict):
        return default

    headings = waypoint_paths.get("heading", None)
    if headings is None:
        return default

    headings = np.asarray(headings)

    try:
        return float(headings[path_idx][wp_idx])
    except Exception:
        return default


def _event_true(events, key):
    if events is None:
        return False
    if isinstance(events, dict):
        return bool(events.get(key, 0))
    return bool(getattr(events, key, False))


def compute_reward(
    agent_obs,
    env_reward,
    action=None,
    prev_action=None,
    prev_goal_dist=None,
    current_goal_dist=None,
):
    reward = 0.0

    # 1. 保留 SMARTS 原始 route-progress reward
    reward += 1.0 * float(env_reward)

    ego = _get(agent_obs, "ego_vehicle_state", {})

    ego_speed = float(_get(ego, "speed", 0.0))
    ego_heading = float(_get(ego, "heading", 0.0))

    lane_position = _get(ego, "lane_position", None)
    lane_position = np.asarray(lane_position)

    if lane_position.ndim > 0 and lane_position.size >= 2:
        signed_lane_error = float(lane_position[1])
    else:
        signed_lane_error = 0.0

    waypoint_paths = _get(agent_obs, "waypoint_paths", {})
    wp0_heading = _get_waypoint_heading(waypoint_paths, 0, 0, ego_heading)

    heading_error_now = _angle_diff(ego_heading, wp0_heading)

    abs_lane_error = abs(signed_lane_error)
    abs_heading_error = abs(heading_error_now)

    speed_limit = 13.89

    # 2. 车道保持
    reward -= 0.3 * abs_lane_error

    if abs_lane_error > 1.5:
        reward -= 1.0

    if abs_lane_error > 2.5:
        reward -= 3.0

    # 3. 航向一致性
    reward -= 0.4 * abs_heading_error

    if abs_heading_error > 0.5:
        reward -= 1.0

    # 4. 速度 shaping
    target_speed = min(0.8 * speed_limit, 12.0)

    if target_speed < 2.0:
        target_speed = 2.0

    if ego_speed < 0.3:
        reward -= 5.0
    elif ego_speed < 1.0:
        reward -= 2.0
    else:
        speed_score = 1.0 - min(abs(ego_speed - target_speed) / max(target_speed, 1.0), 1.0)
        reward += 0.3 * speed_score

    if ego_speed > speed_limit + 2.0:
        reward -= 1.0

    # 5. 距离终点进度奖励
    # 只有车辆真的有速度时才给，避免“原地不动也奖励”
    if (
        prev_goal_dist is not None
        and current_goal_dist is not None
        and ego_speed > 0.2
    ):
        progress_to_goal = prev_goal_dist - current_goal_dist
        reward += 0.8 * _clip(progress_to_goal, -2.0, 2.0)

        close_bonus = 0.3 * (1.0 - min(current_goal_dist / 100.0, 1.0))
        reward += close_bonus

    # 6. 动作平滑
    if action is not None:
        throttle = float(action[0])
        brake = float(action[1])
        steering = float(action[2])

        reward -= 0.2 * abs(steering)

        if throttle > 0.2 and brake > 0.2:
            reward -= 0.8

        if prev_action is not None:
            prev_throttle = float(prev_action[0])
            prev_brake = float(prev_action[1])
            prev_steering = float(prev_action[2])

            reward -= 0.5 * abs(steering - prev_steering)
            reward -= 0.1 * abs(throttle - prev_throttle)
            reward -= 0.1 * abs(brake - prev_brake)

    # 7. 事件奖励/惩罚
    events = _get(agent_obs, "events", None)

    if _event_true(events, "collisions"):
        reward -= 100.0

    if _event_true(events, "off_road"):
        reward -= 80.0

    if _event_true(events, "off_route"):
        reward -= 40.0

    if _event_true(events, "wrong_way"):
        reward -= 30.0

    if _event_true(events, "not_moving"):
        reward -= 10.0

    if _event_true(events, "reached_goal"):
        reward += 120.0

    if _event_true(events, "reached_max_episode_steps"):
        reward -= 5.0

    reward = _clip(reward, -100.0, 100.0)

    return reward
