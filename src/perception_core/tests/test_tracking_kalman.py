import numpy as np

from perception_core.tracking.kalman import ConstantVelocityKF


def test_kf_recovers_constant_velocity():
    kf = ConstantVelocityKF(process_noise=1.0, measurement_noise=0.1)
    kf.init_state([0.0, 0.0])
    dt, vx, vy = 0.1, 3.0, -1.0
    rng = np.random.default_rng(0)
    for k in range(1, 60):
        kf.predict(dt)
        truth = np.array([vx * k * dt, vy * k * dt])
        kf.update(truth + rng.normal(0, 0.05, 2))
    assert np.allclose(kf.velocity, [vx, vy], atol=0.3)


def test_kf_predict_advances_position():
    kf = ConstantVelocityKF()
    kf.init_state([0.0, 0.0], [2.0, 0.0])
    kf.predict(1.0)
    assert np.allclose(kf.position, [2.0, 0.0], atol=1e-9)


def test_kf_covariance_shrinks_after_update():
    kf = ConstantVelocityKF(measurement_noise=0.1)
    kf.init_state([0.0, 0.0])
    p0 = np.trace(kf.P)
    kf.predict(0.1)
    kf.update([0.01, 0.0])
    assert np.trace(kf.P) < p0
