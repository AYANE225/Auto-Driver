import numpy as np
import pytest

from perception_core.common.types import Box3D, ObjectClass


def test_box_corners_shape_and_center():
    box = Box3D(1, 2, 3, 4, 2, 1, 0.0)
    corners = box.corners()
    assert corners.shape == (8, 3)
    assert np.allclose(corners.mean(axis=0), [1, 2, 3])


def test_box_corners_extent_axis_aligned():
    box = Box3D(0, 0, 0, 4, 2, 1, 0.0)
    corners = box.corners()
    assert corners[:, 0].max() == pytest.approx(2.0)
    assert corners[:, 1].max() == pytest.approx(1.0)
    assert corners[:, 2].max() == pytest.approx(0.5)


def test_box_bev_corners_ccw_area():
    box = Box3D(0, 0, 0, 4, 2, 1, 0.0)
    bev = box.bev_corners()
    assert bev.shape == (4, 2)


def test_box_array_roundtrip():
    box = Box3D(1, 2, 3, 4, 5, 6, 0.7)
    assert np.allclose(Box3D.from_array(box.to_array()).to_array(), box.to_array())


def test_object_class_from_str():
    assert ObjectClass.from_str("Car") is ObjectClass.CAR
    assert ObjectClass.from_str("nonsense") is ObjectClass.UNKNOWN


def test_object_class_vru():
    assert ObjectClass.PEDESTRIAN.is_vru
    assert not ObjectClass.CAR.is_vru
