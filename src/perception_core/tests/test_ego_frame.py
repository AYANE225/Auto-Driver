import numpy as np

from perception_core.common.ego_frame import output_to_frame
from perception_core.common.geometry import invert_se3, make_se3, rot_z
from perception_core.common.types import Box3D, PerceptionOutput, Track
from perception_core.prediction.physics import MotionPredictor, PredictorConfig


def test_world_to_sensor_rotates_velocity_history_and_forecast_without_mutation():
    pose = make_se3(rot_z(np.pi / 2), [10, 20, 2])
    track = Track(8, Box3D(10, 25, 3, 4, 2, 1.5, np.pi / 2),
                  velocity=np.array([0., 2.]), history=[np.array([10., 23.])])
    prediction = MotionPredictor(PredictorConfig(mode='cv')).predict(track)
    original = PerceptionOutput(1.0, tracks=[track], predictions=[prediction])
    local = output_to_frame(original, invert_se3(pose))
    assert np.allclose(local.tracks[0].box.center, [5, 0, 1])
    assert np.allclose(local.tracks[0].velocity, [2, 0])
    assert np.allclose(local.tracks[0].history, [[3, 0]])
    assert np.allclose(local.predictions[0].best.as_array()[-1], [11, 0])
    assert abs(local.predictions[0].best.points[-1].yaw) < 1e-10
    assert np.allclose(track.box.center, [10, 25, 3])
    assert np.allclose(prediction.best.as_array()[-1], [10, 31])


def test_forecast_uses_box_height_for_tilted_transform():
    angle = 0.2
    rotation = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0],
                         [-np.sin(angle), 0, np.cos(angle)]])
    track = Track(0, Box3D(0, 0, 10, 4, 2, 2), velocity=np.zeros(2))
    original = PerceptionOutput(0, tracks=[track],
                                predictions=[MotionPredictor().predict(track)])
    local = output_to_frame(original, make_se3(rotation, [0, 0, 0]))
    assert np.allclose(local.predictions[0].best.as_array()[0],
                       local.tracks[0].box.center[:2])
