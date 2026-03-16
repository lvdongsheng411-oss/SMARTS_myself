import math
import numpy as np


def _clip_norm(x, scale, low=-1.0, high=1.0):
    """将数值按 scale 归一化并裁剪到 [low, high]。"""
    if scale <= 1e-6:
        return 0.0
    return float(np.clip(x / scale, low, high))


def _angle_diff(a, b):
    """角度差映射到 [-pi, pi]。"""
    d = a - b
    while d > math.pi:
        d -= 2 * math.pi
    while d < -math.pi:
        d += 2 * math.pi
    return d


def _safe_waypoint_paths(waypoint_paths):
    """尽量把 waypoint_paths 转成 list。"""
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
    """安全取 waypoint。"""
    if path is None:
        return None
    if not hasattr(path, "__len__") or len(path) == 0:
        return None
    if idx < len(path):
        return path[idx]
    return path[-1]


def _safe_neighbors(agent_obs):
    """安全读取邻车。"""
    neighbors = agent_obs.get("neighborhood_vehicle_states", [])
    if neighbors is None:
        return []
    return neighbors


def _extract_goal_position(agent_obs):
    """
    尽量从 SMARTS 观测中解析目标点坐标。
    这里采用“多种可能字段逐级尝试”的写法，兼容性更强。
    """
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

        # 如果传进来的是 mission，则再取 goal
        goal = getattr(obj, "goal", obj)

        # 常见位置字段
        for attr in ["position", "pos", "center", "centroid"]:
            p = getattr(goal, attr, None)
            if p is None:
                continue

            # list / tuple / np.ndarray
            if isinstance(p, (list, tuple, np.ndarray)) and len(p) >= 2:
                return float(p[0]), float(p[1])

            # 可能是带 x y 属性的对象
            x = getattr(p, "x", None)
            y = getattr(p, "y", None)
            if x is not None and y is not None:
                return float(x), float(y)

        # goal 自身可能直接带 x y
        x = getattr(goal, "x", None)
        y = getattr(goal, "y", None)
        if x is not None and y is not None:
            return float(x), float(y)

    return None


def _compute_goal_distance(agent_obs):
    """计算当前自车到终点的欧氏距离；取不到就返回 None。"""
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
    return math.hypot(gx - ego_x, gy - ego_y)


def extract_obs(agent_obs):
    """
    将 SMARTS 原始观测转换为 16 维状态向量。

    维度定义：
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
    ego = agent_obs["ego_vehicle_state"]

    ego_speed = float(getattr(ego, "speed", 0.0))
    ego_pos = getattr(ego, "position", [0.0, 0.0, 0.0])
    ego_x, ego_y = float(ego_pos[0]), float(ego_pos[1])
    ego_heading = float(getattr(ego, "heading", 0.0))

    waypoint_paths = agent_obs.get("waypoint_paths", None)
    paths = _safe_waypoint_paths(waypoint_paths)

    signed_lane_error = 0.0
    heading_error_now = 0.0
    heading_error_future_1 = 0.0
    heading_error_future_2 = 0.0
    speed_limit = 13.89
    route_curvature = 0.0

    if len(paths) > 0:
        path0 = paths[0]

        wp0 = _get_waypoint(path0, 0)
        wp1 = _get_waypoint(path0, 1)
        wp2 = _get_waypoint(path0, 2)

        if wp0 is not None:
            wp0_pos = getattr(wp0, "pos", [ego_x, ego_y])
            wp0_x, wp0_y = float(wp0_pos[0]), float(wp0_pos[1])
            wp0_heading = float(getattr(wp0, "heading", 0.0))

            dx = ego_x - wp0_x
            dy = ego_y - wp0_y

            signed_lane_error = -math.sin(wp0_heading) * dx + math.cos(wp0_heading) * dy
            heading_error_now = _angle_diff(ego_heading, wp0_heading)
            speed_limit = float(getattr(wp0, "speed_limit", 13.89))

        if wp1 is not None:
            wp1_heading = float(getattr(wp1, "heading", 0.0))
            heading_error_future_1 = _angle_diff(ego_heading, wp1_heading)
        else:
            heading_error_future_1 = heading_error_now

        if wp2 is not None:
            wp2_heading = float(getattr(wp2, "heading", 0.0))
            heading_error_future_2 = _angle_diff(ego_heading, wp2_heading)
        else:
            heading_error_future_2 = heading_error_future_1

        # 路线曲率：看前后 waypoint 航向变化
        if wp0 is not None and wp2 is not None:
            wp0_heading = float(getattr(wp0, "heading", 0.0))
            wp2_heading = float(getattr(wp2, "heading", 0.0))
            route_curvature = _angle_diff(wp2_heading, wp0_heading)

    neighbors = _safe_neighbors(agent_obs)

    front_dist = 50.0
    front_rel_speed = 0.0
    left_dist = 50.0
    left_rel_speed = 0.0
    right_dist = 50.0
    right_rel_speed = 0.0

    # 这里仍采用简单近似：按全局坐标粗略区分前/左/右
    for nv in neighbors:
        nv_pos = getattr(nv, "position", [ego_x, ego_y, 0.0])
        dx = float(nv_pos[0]) - ego_x
        dy = float(nv_pos[1]) - ego_y
        dist = math.hypot(dx, dy)

        nv_speed = float(getattr(nv, "speed", 0.0))
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
            _clip_norm(ego_speed, 20.0),                    # 1
            _clip_norm(signed_lane_error, 3.0),             # 2
            _clip_norm(heading_error_now, math.pi),         # 3
            _clip_norm(heading_error_future_1, math.pi),    # 4
            _clip_norm(heading_error_future_2, math.pi),    # 5
            _clip_norm(speed_limit, 20.0),                  # 6
            _clip_norm(front_dist, 50.0),                   # 7
            _clip_norm(front_rel_speed, 20.0),              # 8
            _clip_norm(left_dist, 50.0),                    # 9
            _clip_norm(left_rel_speed, 20.0),               # 10
            _clip_norm(right_dist, 50.0),                   # 11
            _clip_norm(right_rel_speed, 20.0),              # 12
            _clip_norm(route_curvature, math.pi),           # 13
            _clip_norm(goal_distance, 100.0),               # 14
            0.0,                                            # 15
            0.0,                                            # 16
        ],
        dtype=np.float32,
    )

    return obs_vec
