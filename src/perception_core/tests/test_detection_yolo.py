import numpy as np

from perception_core.detection.yolo import YoloCameraDetector, YoloConfig


def test_numpy_model_input_is_bgr_but_frame_stays_rgb():
    class Model:
        def predict(self, image, **kwargs):
            assert image.flags.c_contiguous
            assert image[0, 0].tolist() == [30, 20, 10]
            return []
    detector = object.__new__(YoloCameraDetector)
    detector.cfg = YoloConfig()
    detector._model = Model()
    detector._keep = None
    rgb = np.array([[[10, 20, 30]]], dtype=np.uint8)
    assert detector._detect_image(rgb, 'front') == []
    assert rgb[0, 0].tolist() == [10, 20, 30]
