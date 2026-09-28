"""viz: optional rendering helpers (requires the ``viz`` extra).

* :mod:`perception_core.viz.bev` — Matplotlib bird's-eye view + GIF writer.
* :mod:`perception_core.viz.camera` — Pillow camera overlay (3D boxes, forecasts).

Submodules are imported lazily so that using one backend never pulls in the
other's dependency (e.g. the camera overlay needs Pillow but not Matplotlib).
"""
from importlib import import_module

_EXPORTS = {
    "BevRenderer": "perception_core.viz.bev",
    "save_gif": "perception_core.viz.bev",
    "CameraOverlay": "perception_core.viz.camera",
    "side_by_side": "perception_core.viz.camera",
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name in _EXPORTS:
        return getattr(import_module(_EXPORTS[name]), name)
    raise AttributeError(f"module 'perception_core.viz' has no attribute {name!r}")
