import numpy as np

from perception_core.common.types import ObjectClass
from perception_core.detection.mock import GroundTruthDetector
from perception_core.io.synthetic import generate_frames, make_default_scene
from perception_core.pipeline import PerceptionPipeline


def test_pipeline_full_stack_tracks_all_actors():
    frames = generate_frames(make_default_scene(), num_frames=30, dt=0.1, seed=2)
    pipe = PerceptionPipeline()
    out = None
    for f in frames:
        out = pipe.process(f)
    # 5 actors -> 5 stable confirmed tracks by the end of the sequence
    assert len(out.tracks) == 5
    assert len(out.predictions) == 5


def test_pipeline_velocity_matches_ground_truth():
    # Ground-truth detector isolates tracking accuracy from clustering noise.
    frames = generate_frames(make_default_scene(), num_frames=25, dt=0.1, seed=3)
    pipe = PerceptionPipeline(detector=GroundTruthDetector(position_noise=0.05, seed=0))
    out = None
    for f in frames:
        out = pipe.process(f)
    speeds = sorted(t.speed for t in out.tracks)
    expected = sorted(a.speed for a in make_default_scene())
    assert np.allclose(speeds, expected, atol=1.0)


def test_pipeline_predictions_have_expected_horizon():
    frames = generate_frames(make_default_scene(), num_frames=20, dt=0.1, seed=4)
    pipe = PerceptionPipeline()
    out = None
    for f in frames:
        out = pipe.process(f)
    for pred in out.predictions:
        assert pred.best is not None
        # default horizon 3.0 s at 0.5 s step -> 6 waypoints
        assert len(pred.best.points) == 6


def test_pipeline_is_deterministic():
    scene = make_default_scene()
    frames = generate_frames(scene, num_frames=15, dt=0.1, seed=5)

    def run():
        pipe = PerceptionPipeline(detector=GroundTruthDetector(seed=0))
        out = None
        for f in frames:
            out = pipe.process(f)
        return sorted((t.track_id, round(t.box.x, 3)) for t in out.tracks)

    assert run() == run()
