import numpy as np


def _clip_norm(x, scale, low=-1.0, high=1.0):
    return float(np.clip(x / scale, low, high))


def _safe_first_waypoint(waypoint_paths):
 
    if waypoint_paths is None:
        return None

    # 如果是 dict 
    if isinstance(waypoint_paths, dict):
        # 取字典的 values
        values = list(waypoint_paths.values())
        if len(values) == 0:
            return None

        first_path = values[0]

        # first_path 可能还是列表/元组
        if hasattr(first_path, "__len__") and len(first_path) > 0:
            return first_path[0]

        return None

    # 如果是普通 list / tuple
    if isinstance(waypoint_paths, (list, tuple)):
        if len(waypoint_paths) == 0:
            return None

        first_path = waypoint_paths[0]

        if hasattr(first_path, "__len__") and len(first_path) > 0:
            return first_path[0]

        return None

    # 如果是其他可迭代对象 
    try:
        paths = list(waypoint_paths)
        if len(paths) == 0:
            return None

        first_path = paths[0]

        if hasattr(first_path, "__len__") and len(first_path) > 0:
            return first_path[0]
    except Exception:
        pass

    # 实在取不到，就返回 None
    return None


def extract_obs(agent_obs):
   
    # 读取xiao车状态
    ego = agent_obs["ego_vehicle_state"]

    ego_speed = float(getattr(ego, "speed", 0.0))

    ego_pos = getattr(ego, "position", [0.0, 0.0, 0.0])
    ego_x = float(ego_pos[0])
    ego_y = float(ego_pos[1])

    ego_heading = float(getattr(ego, "heading", 0.0))

  
    # waypoint 信息
    waypoint_paths = agent_obs.get("waypoint_paths", None)

    # 用安全函数取第一个 waypoint
    wp = _safe_first_waypoint(waypoint_paths)

    if wp is not None:
        # 获取 waypoint 坐标
        wp_pos = getattr(wp, "pos", [ego_x, ego_y])
        wp_x = float(wp_pos[0])
        wp_y = float(wp_pos[1])

        # 航向
        wp_heading = float(getattr(wp, "heading", 0.0))

        #  限速
        speed_limit = float(getattr(wp, "speed_limit", 13.89))

        # 近似车道中心偏移
        lane_center_offset = np.hypot(ego_x - wp_x, ego_y - wp_y)

        # waypoint 航向误差
        next_wp_heading_error = wp_heading - ego_heading

    else:
        # 如果取不到 waypoint，就给默认值
        wp_x = ego_x
        wp_y = ego_y
        speed_limit = 13.89
        lane_center_offset = 0.0
        next_wp_heading_error = 0.0

    # 当前航向误差先直接等于 waypoint 航向误差
    heading_error = next_wp_heading_error

    # 到目标距离：当前先用到第一个 waypoint 的距离近似代替
    dist_to_goal = np.hypot(ego_x - wp_x, ego_y - wp_y)

    # =========================================================
    # 3. 周围车辆信息
    # =========================================================
    neighbors = agent_obs.get("neighborhood_vehicle_states", []) or []

    front_dist = 50.0
    front_rel_speed = 0.0
    left_dist = 50.0
    right_dist = 50.0

    for nv in neighbors:
        nv_pos = getattr(nv, "position", [ego_x, ego_y, 0.0])

        dx = float(nv_pos[0]) - ego_x
        dy = float(nv_pos[1]) - ego_y

        dist = np.hypot(dx, dy)

        nv_speed = float(getattr(nv, "speed", 0.0))
        rel_speed = nv_speed - ego_speed

        # 前方最近车辆
        if dx > 0 and dist < front_dist:
            front_dist = dist
            front_rel_speed = rel_speed

        # 左侧最近车辆
        if dy > 0 and dist < left_dist:
            left_dist = dist

        # 右侧最近车辆
        if dy < 0 and dist < right_dist:
            right_dist = dist

    #  拼成最终 10 维向量
    obs_vec = np.array(
        [
            _clip_norm(ego_speed, 20.0),                 # 1 自车速度
            _clip_norm(heading_error, np.pi),           # 2 自车航向误差
            _clip_norm(lane_center_offset, 5.0),        # 3 车道中心偏移
            _clip_norm(dist_to_goal, 50.0),             # 4 到目标距离
            _clip_norm(front_dist, 50.0),               # 5 前车距离
            _clip_norm(front_rel_speed, 20.0),          # 6 前车相对速度
            _clip_norm(left_dist, 50.0),                # 7 左侧最近车距离
            _clip_norm(right_dist, 50.0),               # 8 右侧最近车距离
            _clip_norm(next_wp_heading_error, np.pi),   # 9 下一 waypoint 航向误差
            _clip_norm(speed_limit, 20.0),              # 10 当前限速
        ],
        dtype=np.float32,
    )

    return obs_vec
