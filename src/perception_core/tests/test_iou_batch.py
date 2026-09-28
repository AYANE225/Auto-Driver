import numpy as np
import pytest

from perception_core.common import iou
from perception_core.common.geometry import bev_iou
from perception_core.common.types import Box3D


@pytest.fixture(params=["python", "cpp"])
def backend(request):
    if request.param == "cpp" and iou._native_iou is None:
        pytest.skip("optional C++ extension is not built")
    return request.param


def test_batch_matches_scalar_for_rotated_and_touching_boxes(backend):
    rng = np.random.default_rng(52)
    boxes = [Box3D(*rng.uniform(-10, 10, 3), *rng.uniform(.1, 8, 3), rng.uniform(-np.pi, np.pi))
             for _ in range(48)]
    boxes += [Box3D(0, 0, 0, 4, 2, 1, 0), Box3D(4, 0, 0, 4, 2, 1, 0),
              Box3D(0, 0, 0, .001, 8, 1, 1e-10), Box3D(0, 0, 0, 1, 1, 1, .4)]
    actual = iou.bev_iou_matrix(boxes, boxes, backend)
    expected = np.array([[bev_iou(a, b) for b in boxes] for a in boxes])
    np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-9)
    np.testing.assert_allclose(actual, actual.T, atol=1e-10)
    np.testing.assert_allclose(np.diag(actual), 1, atol=1e-10)


def test_batch_empty_and_invalid_inputs(backend):
    a = Box3D(0, 0, 0, 4, 2, 1)
    assert iou.bev_iou_matrix([], [a], backend).shape == (0, 1)
    assert iou.bev_iou_matrix([a], [], backend).shape == (1, 0)
    assert iou.bev_iou_matrix([], [], backend).shape == (0, 0)
    for bad in (Box3D(np.nan, 0, 0, 4, 2, 1), Box3D(0, 0, 0, -1, 2, 1)):
        with pytest.raises(ValueError):
            iou.bev_iou_matrix([bad], [a], backend)


def test_cpp_large_world_offsets_and_array_validation():
    native = pytest.importorskip("perception_core._geometry")
    boxes = np.array([[0, 0, 0, 4, 2, 1, .4], [2, 1, 0, 5, 2, 1, .6]])
    expected = native.bev_iou_matrix(boxes, boxes)
    boxes[:, :2] += 1e7
    np.testing.assert_allclose(native.bev_iou_matrix(boxes, boxes), expected, atol=2e-9)
    # Non-contiguous input must be converted safely; direct native calls validate too.
    np.testing.assert_allclose(native.bev_iou_matrix(boxes[::-1], boxes[::-1]), expected[::-1, ::-1], atol=2e-9)
    with pytest.raises(ValueError):
        native.bev_iou_matrix(np.zeros((2, 6)), boxes)
    with pytest.raises(ValueError):
        native.bev_iou_matrix(np.zeros((2, 7)), boxes)


def test_auto_falls_back_but_explicit_cpp_does_not(monkeypatch):
    monkeypatch.setattr(iou, "_native_iou", None)
    assert iou.resolve_iou_backend("auto") == "python"
    with pytest.raises(RuntimeError, match="C\\+\\+"):
        iou.resolve_iou_backend("cpp")
    with pytest.raises(ValueError):
        iou.resolve_iou_backend("invalid")
