"""viz: optional Matplotlib bird's-eye-view rendering (requires the ``viz`` extra)."""

from perception_core.viz.bev import BevRenderer, save_gif

__all__ = ["BevRenderer", "save_gif"]
