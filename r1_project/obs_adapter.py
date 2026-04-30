import math
import numpy as np


def _get(obj, key, default=None):
    """兼容 dict 和对象属性两种读取方式。"""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _clip_norm(x, scale, low=-1.0, high=1.0):
    if scale <= 1e-6:
        return 0.0
    return float(np.clip(float(x) / scale, low, high))


def _angle_diff(a, b):
    d = float(a) - float(b)
    while d > math.pi:
        d -= 2 * math.pi
    while d < -math.pi:
        d += 2 * math.pi
    return d


def _as_array(x, default=None):
    if x is None:
        return default
    return np.asarray(x)


def _extract_goal_position(agent_obs):
    """从 dict 格式 SMARTS observation 中提取目标点。"""
    ego = _get(agent_obs, "ego_vehicle_state", {})
    mission = _get(ego, "mission", {})

    goal_position = _get(mission, "goal_position", None)
    if goal_position is not None:
        gp = np.asarray(goal_position)
        if gp.shape[0] >= 2:
            return float(gp[0]), float(gp[1])

    goal = _get(mission, "goal", None)
    if goal is not None:
        for key in ["position", "pos", "center", "centroid"]:
            p = _get(goal, key, None)
            if p is not None:
                p = np.asarray(p)
                if p.shape[0] >= 2:
                    return float(p[0]), float(p[1])

    return None


def _compute_goal_distance(agent_obs):
    ego = _get(agent_obs, "ego_vehicle_state", {})
    ego_pos = _as_array(_get(ego, "position", [0.0, 0.0, 0.0]))
    ego_x = float(ego_pos[0])
    ego_y = float(ego_pos[1])

    goal_pos = _extract_goal_position(agent_obs)
    if goal_pos is None:
        return None

    gx, gy = goal_pos
    return math.hypot(gx - ego_x, gy - ego_y)


def _get_waypoint_heading(waypoint_paths, path_idx=0, wp_idx=0, default=0.0):
    """从 waypoint_paths['heading'] 这种数组结构中取 heading。"""
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


def _safe_neighbors(agent_obs):
    neighbors = _get(agent_obs, "neighborhood_vehicle_states", [])
    if neighbors is None:
        return []
    return neighbors


def extract_obs(agent_obs):
    """
    输出 16 维状态向量：

    1  ego_speed
    2  signed_lane_error
    3  heading_error_now
    4  heading_error_future_1
    5  heading_error_future_2
    6  speed_limit
    7  front_dist
    8  front_rel_speed
    9  left_dist
    10 left_rel_speed
    11 right_dist
    12 right_rel_speed
    13 route_curvature
    14 goal_distance
    15 reserved
    16 reserved
    """

    ego = _get(agent_obs, "ego_vehicle_state", {})

    ego_speed = float(_get(ego, "speed", 0.0))
    ego_pos = _as_array(_get(ego, "position", [0.0, 0.0, 0.0]))
    ego_x = float(ego_pos[0])
    ego_y = float(ego_pos[1])
    ego_heading = float(_get(ego, "heading", 0.0))

    # lane_position 通常格式为 [longitudinal, lateral, 0]
    lane_position = _as_array(_get(ego, "lane_position", None))
    if lane_position is not None and np.ndim(lane_position) > 0 and lane_position.size >= 2:
        signed_lane_error = float(lane_position[1])
    else:
        signed_lane_error = 0.0

    waypoint_paths = _get(agent_obs, "waypoint_paths", {})

    wp0_heading = _get_waypoint_heading(waypoint_paths, 0, 0, ego_heading)
    wp1_heading = _get_waypoint_heading(waypoint_paths, 0, 1, wp0_heading)
    wp2_heading = _get_waypoint_heading(waypoint_paths, 0, 2, wp1_heading)
    wp10_heading = _get_waypoint_heading(waypoint_paths, 0, 10, wp2_heading)

    heading_error_now = _angle_diff(ego_heading, wp0_heading)
    heading_error_future_1 = _angle_diff(ego_heading, wp1_heading)
    heading_error_future_2 = _angle_diff(ego_heading, wp2_heading)
    route_curvature = _angle_diff(wp10_heading, wp0_heading)

    speed_limit = 13.89

    neighbors = _safe_neighbors(agent_obs)

    front_dist = 50.0
    front_rel_speed = 0.0
    left_dist = 50.0
    left_rel_speed = 0.0
    right_dist = 50.0
    right_rel_speed = 0.0

    # 兼容邻车是 list[dict] 或 list[object] 的情况
    if isinstance(neighbors, (list, tuple)):
        for nv in neighbors:
            nv_pos = _as_array(_get(nv, "position", [ego_x, ego_y, 0.0]))
            dx = float(nv_pos[0]) - ego_x
            dy = float(nv_pos[1]) - ego_y
            dist = math.hypot(dx, dy)

            nv_speed = float(_get(nv, "speed", 0.0))
            rel_speed = nv_speed - ego_speed

            if dx > 0 and dist < front_dist:
                front_dist = dist
                front_rel_speed = rel_speed

            if dy > 0 and dist < left_dist:
                left_dist = dist
                left_rel_speed = rel_speed

            if dy < 0 and dist < right_dist:
                right_dist = dist
                right_rel_speed = rel_speed

    goal_distance = _compute_goal_distance(agent_obs)
    if goal_distance is None:
        goal_distance = 100.0

    obs_vec = np.array(
        [
            _clip_norm(ego_speed, 20.0),
            _clip_norm(signed_lane_error, 3.0),
            _clip_norm(heading_error_now, math.pi),
            _clip_norm(heading_error_future_1, math.pi),
            _clip_norm(heading_error_future_2, math.pi),
            _clip_norm(speed_limit, 20.0),
            _clip_norm(front_dist, 50.0),
            _clip_norm(front_rel_speed, 20.0),
            _clip_norm(left_dist, 50.0),
            _clip_norm(left_rel_speed, 20.0),
            _clip_norm(right_dist, 50.0),
            _clip_norm(right_rel_speed, 20.0),
            _clip_norm(route_curvature, math.pi),
            _clip_norm(goal_distance, 100.0),
            0.0,
            0.0,
        ],
        dtype=np.float32,
    )

    return obs_vec
