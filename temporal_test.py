"""Quantitative temporal stability test for kps_stabilize.

Apply small synthetic motion to a target frame across 30 frames; run swap
on each frame; measure inter-frame face-region pixel difference as a
proxy for flicker.  Compare with kps_stabilize on/off.
"""
import os
import sys
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import modules.globals
modules.globals.execution_providers = ['CoreMLExecutionProvider', 'CPUExecutionProvider']
modules.globals.execution_threads = 4
modules.globals.headless = True
modules.globals.hull_mask = True

from modules.face_analyser import (
    get_face_analyser, get_one_face, detect_one_face_fast)
from modules.processors.frame.face_swapper import (
    pre_check, pre_start, swap_face, _KPS_HISTORY)


def make_motion_sequence(base_target, n_frames=30, magnitude=1.0):
    """Subtle camera shake (sub-pixel translation, sub-degree rotation)."""
    h, w = base_target.shape[:2]
    rng = np.random.default_rng(42)
    frames = []
    for i in range(n_frames):
        # Reproducible noise resembling head micro-movements + camera shake
        theta = float(rng.normal(0, 0.4 * magnitude))  # degrees
        tx = float(rng.normal(0, 1.5 * magnitude))     # px
        ty = float(rng.normal(0, 1.5 * magnitude))     # px
        M = cv2.getRotationMatrix2D((w/2, h/2), theta, 1.0)
        M[0, 2] += tx
        M[1, 2] += ty
        frame = cv2.warpAffine(base_target, M, (w, h), flags=cv2.INTER_LANCZOS4)
        frames.append(frame)
    return frames


def measure_flicker(frames_swapped, face_box):
    """Mean absolute inter-frame difference INSIDE the face region.
    Lower = more temporally stable."""
    x1, y1, x2, y2 = face_box
    diffs = []
    for a, b in zip(frames_swapped[:-1], frames_swapped[1:]):
        crop_a = a[y1:y2, x1:x2]
        crop_b = b[y1:y2, x1:x2]
        diffs.append(float(cv2.absdiff(crop_a, crop_b).mean()))
    return float(np.mean(diffs)), float(np.std(diffs))


def main():
    pre_check(); pre_start()

    src = cv2.imread('test_data/face1.jpg')
    tgt = cv2.imread('test_data/face2.jpg')
    def pad(im):
        h, w = im.shape[:2]; ph, pw = h // 4, w // 4
        return cv2.copyMakeBorder(im, ph, ph, pw, pw, cv2.BORDER_REPLICATE)
    src = pad(src); tgt = pad(tgt)
    src_face = get_one_face(src)

    seq = make_motion_sequence(tgt, n_frames=30, magnitude=1.0)
    print(f'sequence: {len(seq)} frames at {seq[0].shape[1]}x{seq[0].shape[0]}')

    # First-frame face for the bbox crop
    f0_face = get_one_face(seq[len(seq) // 2])
    bx = f0_face.bbox.astype(int)
    pad_x = (bx[2] - bx[0]) // 2
    pad_y = (bx[3] - bx[1]) // 2
    h, w = seq[0].shape[:2]
    face_box = (max(0, bx[0] - pad_x), max(0, bx[1] - pad_y),
                min(w, bx[2] + pad_x), min(h, bx[3] + pad_y))

    for cfg_name, alpha in [
        ('no smoothing (alpha=1.0)', 1.0),
        ('mild     (alpha=0.8)',     0.8),
        ('default  (alpha=0.6)',     0.6),
        ('strong   (alpha=0.4)',     0.4),
        ('aggressive (alpha=0.25)',  0.25),
    ]:
        modules.globals.kps_stabilize = alpha
        _KPS_HISTORY.clear()
        out = []
        for f in seq:
            tf = detect_one_face_fast(f)
            if tf is None:
                out.append(f.copy())
                continue
            out.append(swap_face(src_face, tf, f.copy()))
        m, s = measure_flicker(out, face_box)
        print(f'  {cfg_name:30s} -> mean inter-frame diff {m:6.3f}  ± {s:5.3f}')


if __name__ == '__main__':
    main()
