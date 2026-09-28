"""A* on a directed road graph and arc-length reference-path sampling."""

import heapq

import numpy as np
from scipy.interpolate import CubicSpline


class RoadGraph:
    """Directed edges cost their Euclidean length; blocked edges are excluded.

    Nodes are road waypoints, not occupied/free grid cells. Lane permissions
    belong in the caller's edge list. A* does not infer traffic rules.
    """

    def __init__(self, nodes, edges):
        self.nodes = {key: np.asarray(value, dtype=float) for key, value in nodes.items()}
        if any(p.shape != (2,) or not np.isfinite(p).all() for p in self.nodes.values()):
            raise ValueError("nodes must contain finite XY positions")
        self.edges = {key: [] for key in self.nodes}
        for a, b in edges:
            if a not in self.nodes or b not in self.nodes:
                raise ValueError("edge endpoint is missing")
            self.edges[a].append(b)

    def search(self, start, goal, blocked_edges=()):
        if start not in self.nodes or goal not in self.nodes:
            raise ValueError("start or goal is missing")
        blocked = set(blocked_edges)

        def heuristic(node):
            return float(np.linalg.norm(self.nodes[node] - self.nodes[goal]))

        counter = 0
        queue = [(heuristic(start), counter, 0.0, start)]
        costs, parents = {start: 0.0}, {}
        while queue:
            _, _, cost, current = heapq.heappop(queue)
            if cost > costs[current]:
                continue
            if current == goal:
                route = [goal]
                while route[-1] != start:
                    route.append(parents[route[-1]])
                return route[::-1]
            for nxt in self.edges[current]:
                if (current, nxt) in blocked:
                    continue
                candidate = cost + float(np.linalg.norm(self.nodes[nxt] - self.nodes[current]))
                if candidate < costs.get(nxt, np.inf):
                    costs[nxt], parents[nxt] = candidate, current
                    counter += 1
                    heapq.heappush(queue, (candidate + heuristic(nxt), counter, candidate, nxt))
        raise ValueError("no route to goal")


class ReferencePath:
    """Smooth a waypoint polyline, then parameterize it by measured arc length.

    Spline smoothing is geometric, not an obstacle/curvature feasibility proof.
    The local planner checks vehicle curvature and the declared road corridor.
    """

    def __init__(self, xy, spacing=0.25):
        xy = np.asarray(xy, dtype=float)
        if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) < 2 or not np.isfinite(xy).all():
            raise ValueError("reference path needs at least two finite XY points")
        if not np.isfinite(spacing) or spacing <= 0:
            raise ValueError("spacing must be positive")
        xy = xy[np.r_[True, np.linalg.norm(np.diff(xy, axis=0), axis=1) > 1e-6]]
        if len(xy) < 2:
            raise ValueError("reference path has zero length")
        knots = np.r_[0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
        spline = CubicSpline(knots, xy, bc_type="natural")
        self.xy = spline(np.linspace(0, knots[-1], max(2, int(knots[-1] / spacing) + 1)))
        self.s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(self.xy, axis=0), axis=1))]
        delta = np.gradient(self.xy, self.s, axis=0)
        self.yaw = np.unwrap(np.arctan2(delta[:, 1], delta[:, 0]))
        self.curvature = np.gradient(self.yaw, self.s)
        self.length = float(self.s[-1])

    def sample(self, s):
        s = np.clip(s, 0, self.length)
        xy = np.stack([np.interp(s, self.s, self.xy[:, k]) for k in range(2)], axis=-1)
        return xy, np.interp(s, self.s, self.yaw), np.interp(s, self.s, self.curvature)

    def project(self, x, y):
        segments = np.diff(self.xy, axis=0)
        u = np.clip(
            np.sum((np.array([x, y]) - self.xy[:-1]) * segments, axis=1)
            / np.sum(segments * segments, axis=1),
            0,
            1,
        )
        projected = self.xy[:-1] + segments * u[:, None]
        i = int(np.argmin(np.sum((projected - [x, y]) ** 2, axis=1)))
        s = self.s[i] + u[i] * (self.s[i + 1] - self.s[i])
        _, yaw, _ = self.sample(s)
        delta = np.array([x, y]) - projected[i]
        return float(s), float(-np.sin(yaw) * delta[0] + np.cos(yaw) * delta[1])
