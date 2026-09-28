"""Track colour palette shared by the bird's-eye-view and camera renderers."""
from __future__ import annotations

from typing import Tuple

TRACK_PALETTE = [
    "#e6194B", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4",
    "#f032e6", "#bfef45", "#fabed4", "#469990", "#dcbeff", "#9A6324",
]


def track_color_hex(track_id: int) -> str:
    return TRACK_PALETTE[track_id % len(TRACK_PALETTE)]


def track_color_rgb(track_id: int) -> Tuple[int, int, int]:
    h = track_color_hex(track_id).lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
