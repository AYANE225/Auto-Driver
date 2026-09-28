"""Selectable batch BEV IoU, with an optional C++ implementation."""
import numpy as np

from perception_core.common.geometry import bev_iou

try:
    from perception_core._geometry import bev_iou_matrix as _native_iou
except ImportError:
    _native_iou = None


def resolve_iou_backend(backend="auto"):
    if backend not in ("auto", "python", "cpp"):
        raise ValueError("IoU backend must be auto, python or cpp")
    if backend == "cpp" and _native_iou is None:
        raise RuntimeError("C++ IoU is unavailable; reinstall perception_core with a C++14 compiler")
    return "cpp" if backend != "python" and _native_iou is not None else "python"


def bev_iou_matrix(boxes_a, boxes_b, backend="auto"):
    """Return an (N, M) IoU matrix for two sequences of Box3D in one frame.

    The Python backend is the scalar reference, including its numerical behavior.
    Native computation caches corners and rejects disjoint axis-aligned bounds.
    """
    backend = resolve_iou_backend(backend)
    a = np.asarray([box.to_array() for box in boxes_a], dtype=float).reshape(-1, 7)
    b = np.asarray([box.to_array() for box in boxes_b], dtype=float).reshape(-1, 7)
    for array in (a, b):
        if not np.isfinite(array).all() or np.any(array[:, 3:6] <= 0):
            raise ValueError("boxes must be finite with positive dimensions")
    if backend == "cpp":
        return _native_iou(a, b)
    return np.asarray([[bev_iou(a, b) for b in boxes_b] for a in boxes_a],
                      dtype=float).reshape(len(boxes_a), len(boxes_b))
