"""Vectorized oriented-rectangle separation and perception forecast sampling."""

import numpy as np


def rectangle_separation(a_xy, a_yaw, a_length, a_width, b_xy, b_yaw, b_length, b_width):
    """SAT separation in metres (positive: separated, <= 0: touching/overlap).

    This is a separating-axis gap, not Euclidean distance. Leading dimensions
    broadcast, so whole timed trajectories can be checked in one NumPy call.
    """
    a_yaw, b_yaw = np.broadcast_arrays(a_yaw, b_yaw)
    a_x = np.stack([np.cos(a_yaw), np.sin(a_yaw)], axis=-1)
    a_y = np.stack([-np.sin(a_yaw), np.cos(a_yaw)], axis=-1)
    b_x = np.stack([np.cos(b_yaw), np.sin(b_yaw)], axis=-1)
    b_y = np.stack([-np.sin(b_yaw), np.cos(b_yaw)], axis=-1)
    delta = np.asarray(b_xy) - np.asarray(a_xy)
    gaps = []
    for axis in (a_x, a_y, b_x, b_y):
        radius_a = (
            a_length * np.abs(np.sum(a_x * axis, axis=-1))
            + a_width * np.abs(np.sum(a_y * axis, axis=-1))
        ) / 2
        radius_b = (
            b_length * np.abs(np.sum(b_x * axis, axis=-1))
            + b_width * np.abs(np.sum(b_y * axis, axis=-1))
        ) / 2
        gaps.append(np.abs(np.sum(delta * axis, axis=-1)) - radius_a - radius_b)
    return np.max(gaps, axis=0)


def sample_obstacles(output, times):
    """Use all forecast modes; unmatched detections are static until tracked.

    Beyond a forecast's final sample use its last segment velocity. Stale-frame
    rejection is handled by LocalPlanner before this adapter is called.
    """
    obstacles = []
    predicted_ids = set()
    for prediction in output.predictions:
        box = prediction.current_box
        predicted_ids.add(prediction.track_id)
        for trajectory in prediction.trajectories or [None]:
            points = [] if trajectory is None else trajectory.points
            stamps = np.array([0.0] + [p.t for p in points])
            xy = np.array([[box.x, box.y]] + [[p.x, p.y] for p in points])
            yaw = np.array([box.yaw] + [p.yaw for p in points])
            if not np.isfinite(stamps).all() or np.any(np.diff(stamps) <= 0):
                raise ValueError("forecast times must be finite and increasing after zero")
            # A nearly stationary track has no reliable velocity heading.
            # Also, a rectangle is pi-periodic: opposite headings must not
            # interpolate through a spurious 90-degree rotation.
            if (
                len(stamps) > 1
                and np.max(np.linalg.norm(np.diff(xy, axis=0), axis=1) / np.diff(stamps)) < 0.5
            ):
                yaw[:] = box.yaw
            else:
                yaw = np.r_[
                    yaw[0], yaw[0] + np.cumsum((np.diff(yaw) + np.pi / 2) % np.pi - np.pi / 2)
                ]
            sampled = np.column_stack([np.interp(times, stamps, xy[:, k]) for k in range(2)])
            if len(stamps) > 1:
                velocity = (xy[-1] - xy[-2]) / (stamps[-1] - stamps[-2])
                sampled += np.maximum(0, times - stamps[-1])[:, None] * velocity
            obstacles.append((sampled, np.interp(times, stamps, yaw), box.l, box.w))
    for track in output.tracks:
        if track.track_id not in predicted_ids:
            box = track.box
            obstacles.append(
                (
                    np.array([box.x, box.y]) + times[:, None] * track.velocity,
                    np.full(len(times), box.yaw),
                    box.l,
                    box.w,
                )
            )
    centers = np.array([[t.box.x, t.box.y] for t in output.tracks]).reshape(-1, 2)
    for detection in output.detections:
        box = detection.box
        if len(centers) and np.min(np.linalg.norm(centers - [box.x, box.y], axis=1)) < 1.0:
            continue
        obstacles.append(
            (np.tile([box.x, box.y], (len(times), 1)), np.full(len(times), box.yaw), box.l, box.w)
        )
    for xy, yaw, length, width in obstacles:
        if (
            not np.isfinite(xy).all()
            or not np.isfinite(yaw).all()
            or not np.isfinite([length, width]).all()
            or min(length, width) <= 0
        ):
            raise ValueError("invalid obstacle geometry")
    return obstacles
