"""Bird's-eye-view rendering of a :class:`PerceptionOutput` with Matplotlib.

Optional module (requires the ``viz`` extra). Produces publication-style BEV
frames showing the LiDAR sweep, ground-truth boxes, confirmed tracks (coloured
by ID, with velocity arrows and history trails) and predicted trajectories.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Polygon as MplPolygon  # noqa: E402

from perception_core.common.types import Frame, ObjectClass, PerceptionOutput  # noqa: E402

_TRACK_PALETTE = [
    "#e6194B", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4",
    "#f032e6", "#bfef45", "#fabed4", "#469990", "#dcbeff", "#9A6324",
]


def _track_color(track_id: int) -> str:
    return _TRACK_PALETTE[track_id % len(_TRACK_PALETTE)]


class BevRenderer:
    def __init__(
        self,
        xlim: Tuple[float, float] = (-20.0, 60.0),
        ylim: Tuple[float, float] = (-30.0, 30.0),
        figsize: Tuple[float, float] = (9.0, 7.0),
    ) -> None:
        self.xlim, self.ylim = xlim, ylim
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.fig.tight_layout()

    def draw(
        self,
        frame: Frame,
        output: PerceptionOutput,
        show_lidar: bool = True,
        show_gt: bool = True,
        max_points: int = 6000,
    ) -> np.ndarray:
        ax = self.ax
        ax.clear()
        ax.set_facecolor("#101418")
        ax.set_xlim(*self.xlim)
        ax.set_ylim(*self.ylim)
        ax.set_aspect("equal")
        ax.grid(color="#2a2f36", linewidth=0.5)
        ax.set_xlabel("x [m] (forward)")
        ax.set_ylabel("y [m] (left)")

        if show_lidar and frame.lidar is not None and len(frame.lidar):
            pts = frame.lidar
            if len(pts) > max_points:
                idx = np.random.default_rng(0).choice(len(pts), max_points, replace=False)
                pts = pts[idx]
            ax.scatter(pts[:, 0], pts[:, 1], s=0.4, c="#5a6473", alpha=0.6, linewidths=0)

        # ego vehicle marker at the origin
        ax.plot(0, 0, marker="^", color="white", markersize=10, zorder=5)

        if show_gt and frame.ground_truth:
            for gt in frame.ground_truth:
                ax.add_patch(MplPolygon(gt.box.bev_corners(), closed=True, fill=False,
                                        edgecolor="#00d000", linewidth=1.0, linestyle="--", alpha=0.7))

        for tr in output.tracks:
            self._draw_track(ax, tr)

        for pred in output.predictions:
            traj = pred.best
            if traj is None or not traj.points:
                continue
            xy = traj.as_array()
            xy = np.vstack([[pred.current_box.x, pred.current_box.y], xy])
            ax.plot(xy[:, 0], xy[:, 1], ":", color=_track_color(pred.track_id), linewidth=1.6, alpha=0.9)

        ax.set_title(f"t = {output.timestamp:5.2f} s   |   tracks: {len(output.tracks)}",
                     color="white")
        self.fig.canvas.draw()
        img = np.asarray(self.fig.canvas.buffer_rgba())[:, :, :3].copy()
        return img

    def _draw_track(self, ax, tr) -> None:
        color = _track_color(tr.track_id)
        ax.add_patch(MplPolygon(tr.box.bev_corners(), closed=True, fill=True,
                                facecolor=color, edgecolor="white", linewidth=1.2, alpha=0.45))
        # heading tick
        c, s = np.cos(tr.box.yaw), np.sin(tr.box.yaw)
        ax.plot([tr.box.x, tr.box.x + c * tr.box.l / 2], [tr.box.y, tr.box.y + s * tr.box.l / 2],
                color="white", linewidth=1.0)
        # velocity arrow (1 s look-ahead)
        vx, vy = tr.velocity
        if np.hypot(vx, vy) > 0.3:
            ax.arrow(tr.box.x, tr.box.y, vx, vy, head_width=0.6, head_length=0.8,
                     fc=color, ec=color, length_includes_head=True, alpha=0.9)
        # history trail
        if len(tr.history) > 1:
            h = np.array(tr.history)
            ax.plot(h[:, 0], h[:, 1], "-", color=color, linewidth=0.8, alpha=0.5)
        ax.text(tr.box.x, tr.box.y + tr.box.w / 2 + 0.6, f"#{tr.track_id} {tr.label.value}",
                color="white", fontsize=7, ha="center")

    def close(self) -> None:
        plt.close(self.fig)


def save_gif(frames_rgb: Sequence[np.ndarray], path: str, fps: int = 10) -> None:
    """Assemble RGB frames into an animated GIF using Pillow (no imageio dep)."""
    from PIL import Image

    imgs = [Image.fromarray(f) for f in frames_rgb]
    if not imgs:
        raise ValueError("no frames to write")
    imgs[0].save(path, save_all=True, append_images=imgs[1:],
                 duration=int(1000 / max(fps, 1)), loop=0, optimize=True)
