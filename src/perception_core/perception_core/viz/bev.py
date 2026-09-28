"""Bird's-eye-view rendering of a :class:`PerceptionOutput` with Matplotlib.

Optional module (requires the ``viz`` extra). Produces publication-style BEV
frames showing the LiDAR sweep, ground-truth boxes, confirmed tracks (coloured
by ID, with velocity arrows and history trails) and predicted trajectories.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Polygon as MplPolygon  # noqa: E402

from perception_core.common.types import Frame, PerceptionOutput  # noqa: E402
from perception_core.viz.palette import track_color_hex as _track_color  # noqa: E402


class BevRenderer:
    def __init__(
        self,
        xlim: Tuple[float, float] = (-20.0, 60.0),
        ylim: Tuple[float, float] = (-30.0, 30.0),
        figsize: Tuple[float, float] = (9.0, 7.0),
    ) -> None:
        self.xlim, self.ylim = xlim, ylim
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.fig.patch.set_facecolor("#101418")
        self.fig.subplots_adjust(left=0.10, right=0.98, bottom=0.10, top=0.91)

    def draw(
        self,
        frame: Frame,
        output: PerceptionOutput,
        show_lidar: bool = True,
        show_gt: bool = True,
        max_points: int = 6000,
        all_modes: bool = False,
        camera_fov: Optional[float] = None,
        title: Optional[str] = None,
    ) -> np.ndarray:
        """Render one frame. ``all_modes`` also draws the non-best forecast modes
        (fainter); ``camera_fov`` (degrees) outlines a forward camera's frustum."""
        ax = self.ax
        ax.clear()
        ax.set_facecolor("#101418")
        ax.tick_params(colors="#bac5d1", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#39434f")
        ax.set_xlim(*self.xlim)
        ax.set_ylim(*self.ylim)
        ax.set_aspect("equal")
        ax.grid(color="#2a2f36", linewidth=0.5)
        ax.set_xlabel("x [m] (forward)", color="#bac5d1")
        ax.set_ylabel("y [m] (left)", color="#bac5d1")

        if show_lidar and frame.lidar is not None and len(frame.lidar):
            pts = frame.lidar
            if len(pts) > max_points:
                idx = np.random.default_rng(0).choice(len(pts), max_points, replace=False)
                pts = pts[idx]
            ax.scatter(pts[:, 0], pts[:, 1], s=0.4, c="#5a6473", alpha=0.6, linewidths=0)

        if camera_fov is not None:
            reach = max(abs(v) for v in (*self.xlim, *self.ylim)) * 1.5
            half = np.radians(camera_fov) / 2.0
            for s in (-1.0, 1.0):
                ax.plot([0, reach * np.cos(half)], [0, s * reach * np.sin(half)],
                        color="#8b949e", linewidth=0.8, linestyle="-.", alpha=0.6)

        # ego vehicle marker at the origin
        ax.plot(0, 0, marker="^", color="white", markersize=10, zorder=5)

        if show_gt and frame.ground_truth:
            for gt in frame.ground_truth:
                ax.add_patch(MplPolygon(gt.box.bev_corners(), closed=True, fill=False,
                                        edgecolor="#00d000", linewidth=1.0, linestyle="--", alpha=0.7))

        for tr in output.tracks:
            self._draw_track(ax, tr)

        for pred in output.predictions:
            best = pred.best
            modes = pred.trajectories if all_modes else ([best] if best is not None else [])
            color = _track_color(pred.track_id)
            for traj in modes:
                if not traj.points:
                    continue
                xy = np.vstack([[pred.current_box.x, pred.current_box.y], traj.as_array()])
                if traj is best:
                    ax.plot(xy[:, 0], xy[:, 1], ":", color=color, linewidth=1.6, alpha=0.9)
                else:
                    ax.plot(xy[:, 0], xy[:, 1], "-", color=color, linewidth=0.8,
                            alpha=0.25 + 0.5 * traj.confidence)

        label = f"t = {output.timestamp:5.2f} s   |   tracks: {len(output.tracks)}"
        ax.set_title(f"{title}\n{label}" if title else label, color="white", fontsize=10)
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
