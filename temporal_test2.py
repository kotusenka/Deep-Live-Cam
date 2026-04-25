"""Validate kps_stabilize on detector-jitter (not full-frame motion).

Take a static frame, perturb the detected kps with sub-pixel Gaussian
noise to model RetinaFace's per-frame detection wobble, run swap with
each kps, and measure inter-frame diff.  This isolates the stabilizer's
effect on detector noise.
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
    get_face_analyser, get_one_face)
from modules.processors.frame.face_swapper import (
    pre_check, pre_start, swap_face, _KPS_HISTORY)


def main():
    pre_check(); pre_start()
    src = cv2.imread('test_data/face1.jpg')
    tgt = cv2.imread('test_data/face2.jpg')
    def pad(im):
        h, w = im.shape[:2]; ph, pw = h // 4, w // 4
        return cv2.copyMakeBorder(im, ph, ph, pw, pw, cv2.BORDER_REPLICATE)
    src = pad(src); tgt = pad(tgt)
    src_face = get_one_face(src)
    tf_clean = get_one_face(tgt)
    print('clean kps:', tf_clean.kps.tolist())

    rng = np.random.default_rng(123)
    n_frames = 30
    jitter_sigma = 1.5  # sub-pixel detection wobble in px

    for cfg_name, alpha in [
        ('off  (alpha=1.0)', 1.0),
        ('mild (alpha=0.8)', 0.8),
        ('def  (alpha=0.6)', 0.6),
        ('agg  (alpha=0.4)', 0.4),
        ('hard (alpha=0.25)', 0.25),
    ]:
        modules.globals.kps_stabilize = alpha
        _KPS_HISTORY.clear()
        out = []
        for i in range(n_frames):
            # Make a fresh face copy with jittered kps
            from insightface.app.common import Face
            jit = rng.normal(0, jitter_sigma, tf_clean.kps.shape).astype(np.float32)
            jittered = Face(
                bbox=tf_clean.bbox,
                kps=tf_clean.kps + jit,
                det_score=tf_clean.det_score,
            )
            jittered.embedding = tf_clean.embedding
            if hasattr(tf_clean, 'landmark_2d_106'):
                jittered.landmark_2d_106 = tf_clean.landmark_2d_106
            out.append(swap_face(src_face, jittered, tgt.copy()))
        # Inter-frame diff in face crop
        bx = tf_clean.bbox.astype(int)
        pad_x = (bx[2] - bx[0]) // 3
        pad_y = (bx[3] - bx[1]) // 3
        h, w = tgt.shape[:2]
        crop = lambda im: im[max(0, bx[1]-pad_y):min(h, bx[3]+pad_y),
                             max(0, bx[0]-pad_x):min(w, bx[2]+pad_x)]
        diffs = [float(cv2.absdiff(crop(a), crop(b)).mean())
                 for a, b in zip(out[:-1], out[1:])]
        print(f'  {cfg_name:18s} -> inter-frame diff {np.mean(diffs):6.4f} '
              f'± {np.std(diffs):5.4f} (max {np.max(diffs):.4f})')


if __name__ == '__main__':
    main()
