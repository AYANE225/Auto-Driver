"""Camera-view overlay: project 3D tracks and their forecasts into an image.

Optional module (needs Pillow, which ships with the ``viz`` extra through
Matplotlib). Boxes and forecasts are expected in the *sensor (LiDAR/ego) frame*
— use :func:`perception_core.common.ego_frame.output_to_frame` to bring
world-frame pipeline output there — and are mapped into the image with the
frame's :class:`~perception_core.common.types.SensorCalibration`. Segments that
cross behind the camera are clipped against a near plane before projection, so
partially visible boxes are drawn correctly instead of wrapping around.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from perception_core.common.geometry import project_to_image, transform_points
from perception_core.common.types import Box3D, PerceptionOutput, SensorCalibration
from perception_core.viz.palette import track_color_rgb as track_color

__all__ = ["CameraOverlay", "clip_segment_to_near_plane", "side_by_side"]

# Box3D.corners(): indices 0-3 are the top face, 4-7 the bottom face.
_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
          (0, 4), (1, 5), (2, 6), (3, 7)]


def clip_segment_to_near_plane(p: np.ndarray, q: np.ndarray,
                               near: float) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Clip a camera-frame segment to ``z >= near``; ``None`` if fully behind."""
    zp, zq = float(p[2]), float(q[2])
    if zp < near and zq < near:
        return None
    if zp >= near and zq >= near:
        return p, q
    r = p + (near - zp) / (zq - zp) * (q - p)
    return (r, q) if zp < near else (p, r)


def _load_font(size: int):
    from PIL import ImageFont
    try:
        return ImageFont.load_default(size=size)  # Pillow >= 10.1 (scalable)
    except TypeError:  # pragma: no cover - older Pillow
        return ImageFont.load_default()


class CameraOverlay:
    """Draw tracks (3D wireframes + ID labels) and forecasts on a camera image."""

    def __init__(self, calib: SensorCalibration, camera: str, near: float = 0.5,
                 max_depth: float = 80.0, line_width: int = 3, font_size: int = 15) -> None:
        if not calib.has_camera(camera):
            raise ValueError(f"calibration has no camera '{camera}'")
        self.T = calib.lidar_to_cam[camera]
        self.K = calib.intrinsics[camera]
        self.size = calib.image_size.get(camera)
        self.near = near
        self.max_depth = max_depth
        self.line_width = line_width
        self.font = _load_font(font_size)

    # -- projection helpers -----------------------------------------------------
    def _segments(self, pts_sensor: np.ndarray, pairs) -> List[Tuple[tuple, tuple]]:
        cam = transform_points(pts_sensor, self.T)
        out = []
        for i, j in pairs:
            seg = clip_segment_to_near_plane(cam[i], cam[j], self.near)
            if seg is None:
                continue
            uv, _ = project_to_image(np.vstack(seg), self.K)
            out.append((tuple(uv[0]), tuple(uv[1])))
        return out

    def _in_view(self, box: Box3D) -> bool:
        centre = transform_points(box.center[None, :], self.T)[0]
        if not (self.near < centre[2] < self.max_depth):
            return False
        if self.size is None:
            return True
        uv, _ = project_to_image(centre[None, :], self.K)
        w, h = self.size
        margin = 0.25 * w
        return -margin < uv[0, 0] < w + margin and -margin < uv[0, 1] < h + margin

    # -- drawing -----------------------------------------------------------------
    def draw(self, image: np.ndarray, output: PerceptionOutput,
             show_predictions: bool = True, show_labels: bool = True) -> np.ndarray:
        """Return a copy of ``image`` with ``output`` (sensor frame) drawn on it."""
        from PIL import Image, ImageDraw

        base = Image.fromarray(np.asarray(image, dtype=np.uint8)).convert("RGBA")
        layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)

        # Far-to-near so that nearer objects are painted on top.
        tracks = [t for t in output.tracks if self._in_view(t.box)]
        tracks.sort(key=lambda t: -float(np.hypot(t.box.x, t.box.y)))
        preds = {p.track_id: p for p in output.predictions}

        for tr in tracks:
            color = track_color(tr.track_id)
            if show_predictions and tr.track_id in preds:
                self._draw_forecast(draw, preds[tr.track_id], color)
            for a, b in self._segments(tr.box.corners(), _EDGES):
                draw.line([a, b], fill=color + (255,), width=self.line_width)
            if show_labels:
                self._draw_label(draw, tr, color)

        return np.asarray(Image.alpha_composite(base, layer).convert("RGB"))

    def _draw_forecast(self, draw, pred, color) -> None:
        z = pred.current_box.z - pred.current_box.h / 2.0  # forecasts ride on the ground
        best = pred.best
        for traj in sorted(pred.trajectories, key=lambda t: t.confidence):
            if not traj.points:
                continue
            xy = np.vstack([[pred.current_box.x, pred.current_box.y], traj.as_array()])
            pts = np.column_stack([xy, np.full(len(xy), z)])
            is_best = traj is best
            alpha = 235 if is_best else int(60 + 120 * traj.confidence)
            width = max(2, self.line_width - (0 if is_best else 1))
            segs = self._segments(pts, [(k, k + 1) for k in range(len(pts) - 1)])
            for a, b in segs:
                draw.line([a, b], fill=color + (alpha,), width=width)
            if is_best:
                for _, b in segs:  # waypoint dots on the most likely mode
                    r = width + 1
                    draw.ellipse([b[0] - r, b[1] - r, b[0] + r, b[1] + r], fill=color + (alpha,))

    def _draw_label(self, draw, tr, color) -> None:
        cam = transform_points(tr.box.corners(), self.T)
        cam = cam[cam[:, 2] > self.near]
        if not len(cam):
            return
        uv, _ = project_to_image(cam, self.K)
        u, v = float(uv[:, 0].mean()), float(uv[:, 1].min())
        text = f"#{tr.track_id} {tr.label.value} {tr.speed:.1f}m/s"
        x0, y0, x1, y1 = draw.textbbox((0, 0), text, font=self.font)
        tw, th = x1 - x0, y1 - y0
        left, top = u - tw / 2 - 4, v - th - 10
        draw.rectangle([left, top, left + tw + 8, top + th + 6], fill=color + (200,))
        draw.text((left + 4, top + 2 - y0), text, fill=(255, 255, 255, 255), font=self.font)


def side_by_side(panels: Sequence[np.ndarray], height: int, gap: int = 0,
                 background: Tuple[int, int, int] = (16, 20, 24)) -> np.ndarray:
    """Resize RGB panels to a common ``height`` and concatenate them horizontally."""
    from PIL import Image

    resized = []
    for p in panels:
        im = Image.fromarray(np.asarray(p, dtype=np.uint8))
        w = max(1, int(round(im.width * height / im.height)))
        resized.append(im.resize((w, height), Image.LANCZOS))
    total = sum(im.width for im in resized) + gap * (len(resized) - 1)
    canvas = Image.new("RGB", (total, height), background)
    x = 0
    for im in resized:
        canvas.paste(im, (x, 0))
        x += im.width + gap
    return np.asarray(canvas)
