import numpy
import cv2
from PIL import Image
import modules.globals
from modules.gpu_processing import gpu_cvt_color
from modules.typing import Frame

MAX_PROBABILITY = 0.85

_model = None
_opennsfw2 = None
_disabled = False


def _try_load():
    """Lazy-import opennsfw2/tensorflow. Returns False if unavailable.

    Why: tensorflow + tensorflow-metal can fail at import time on
    macOS due to plugin ABI mismatch. NSFW filter is off by default,
    so we shouldn't crash the whole app just to support it.
    """
    global _opennsfw2, _disabled
    if _disabled:
        return False
    if _opennsfw2 is not None:
        return True
    try:
        import opennsfw2
        _opennsfw2 = opennsfw2
        return True
    except Exception:
        _disabled = True
        return False


def predict_frame(target_frame: Frame) -> bool:
    if not _try_load():
        return False
    if modules.globals.color_correction:
        target_frame = gpu_cvt_color(target_frame, cv2.COLOR_BGR2RGB)

    image = Image.fromarray(target_frame)
    image = _opennsfw2.preprocess_image(image, _opennsfw2.Preprocessing.YAHOO)
    global _model
    if _model is None:
        _model = _opennsfw2.make_open_nsfw_model()

    views = numpy.expand_dims(image, axis=0)
    _, probability = _model.predict(views)[0]
    return probability > MAX_PROBABILITY


def predict_image(target_path: str) -> bool:
    if not _try_load():
        return False
    return _opennsfw2.predict_image(target_path) > MAX_PROBABILITY


def predict_video(target_path: str) -> bool:
    if not _try_load():
        return False
    _, probabilities = _opennsfw2.predict_video_frames(video_path=target_path, frame_interval=100)
    return any(probability > MAX_PROBABILITY for probability in probabilities)
