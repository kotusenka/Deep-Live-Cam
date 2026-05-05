"""Compare EMA vs median-of-N kps filter for live face swap stability.

Hypothesis: median-of-N is robust to detector outliers (which produce
visible jerks) WITHOUT the lag-then-catch-up behavior of low-alpha EMA.
"""

import os
import sys
import time
from collections import deque
from typing import List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run  # noqa: F401

import cv2
import numpy as np

import modules.globals
modules.globals.execution_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

from modules.face_analyser import get_one_face, detect_one_face_fast
from modules.processors.frame import face_swapper as fs


REPO = os.path.dirname(os.path.abspath(__file__))


class MedianFilter:
    def __init__(self, window: int = 3):
        self.window = window
        self.history: deque = deque(maxlen=window)

    def reset(self):
        self.history.clear()

    def filter(self, kps: np.ndarray) -> np.ndarray:
        self.history.append(kps.astype(np.float32).copy())
        if len(self.history) < 2:
            return kps
        stack = np.stack(list(self.history))  # [N, 5, 2]
        return np.median(stack, axis=0).astype(np.float32)


class EMAFilter:
    def __init__(self, alpha: float, reset_threshold: float = 40.0):
        self.alpha = alpha
        self.reset_threshold = reset_threshold
        self.prev = None

    def reset(self):
        self.prev = None

    def filter(self, kps: np.ndarray) -> np.ndarray:
        if self.prev is None:
            self.prev = kps.astype(np.float32).copy()
            return kps
        delta = float(np.linalg.norm(self.prev - kps, axis=1).max())
        if delta > self.reset_threshold:
            self.prev = kps.astype(np.float32).copy()
            return kps
        smoothed = (self.alpha * kps + (1.0 - self.alpha) * self.prev).astype(np.float32)
        self.prev = smoothed
        return smoothed


def reset_caches():
    fs._SWAP_CACHE.clear()
    fs._KPS_HISTORY.clear()
    fs._paste_cache['soft_alpha'] = None
    fs._paste_cache['alpha_size'] = 0


def run_with_filter(frames, src_face, kps_filter, label):
    # Disable in-fs EMA so we drive smoothing externally
    modules.globals.kps_stabilize = 1.0
    modules.globals.swap_cache_threshold_px = 15.0
    modules.globals.swap_cache_max_age_ms = 1000.0
    reset_caches()
    kps_filter.reset()

    out_frames = []
    kps_raw_history = []
    kps_filtered_history = []
    bboxes = []
    swap_times = []

    # Warmup
    for f in frames[:5]:
        face = detect_one_face_fast(f)
        if face is not None:
            fs.swap_face(src_face, face, f.copy())
    reset_caches()
    kps_filter.reset()

    for f in frames:
        face = detect_one_face_fast(f)
        if face is None:
            out_frames.append(f.copy())
            kps_raw_history.append(None)
            kps_filtered_history.append(None)
            bboxes.append(None)
            continue
        kps_raw = face.kps.astype(np.float32).copy()
        kps_smoothed = kps_filter.filter(kps_raw)
        face.kps = kps_smoothed
        t0 = time.perf_counter()
        out = fs.swap_face(src_face, face, f.copy())
        swap_times.append((time.perf_counter() - t0) * 1000.0)
        out_frames.append(out)
        kps_raw_history.append(kps_raw)
        kps_filtered_history.append(kps_smoothed)
        bboxes.append(face.bbox.astype(np.float32).copy())

    # Metrics
    # 1) jerk on filtered kps (what gets passed to swap)
    jerks = []
    for i in range(1, len(kps_filtered_history) - 1):
        a, b, c = kps_filtered_history[i - 1], kps_filtered_history[i], kps_filtered_history[i + 1]
        if a is None or b is None or c is None:
            continue
        accel = c - 2.0 * b + a
        jerks.append(float(np.linalg.norm(accel, axis=1).mean()))
    jerks = np.array(jerks) if jerks else np.zeros(1)

    # 2) lag = mean L2 distance between filtered and raw — proxy for how
    #    much the filtered kps trails the noisy ground truth.
    lags = []
    for r, f in zip(kps_raw_history, kps_filtered_history):
        if r is None or f is None:
            continue
        lags.append(float(np.linalg.norm(r - f, axis=1).mean()))
    lags = np.array(lags) if lags else np.zeros(1)

    # 3) pixel diff in bbox
    pix_diffs = []
    for i in range(1, len(out_frames)):
        if bboxes[i] is None or bboxes[i - 1] is None:
            continue
        x1 = int(min(bboxes[i][0], bboxes[i - 1][0]))
        y1 = int(min(bboxes[i][1], bboxes[i - 1][1]))
        x2 = int(max(bboxes[i][2], bboxes[i - 1][2]))
        y2 = int(max(bboxes[i][3], bboxes[i - 1][3]))
        h, w = out_frames[i].shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        a = out_frames[i - 1][y1:y2, x1:x2].astype(np.int16)
        b = out_frames[i][y1:y2, x1:x2].astype(np.int16)
        pix_diffs.append(float(np.abs(a - b).mean()))
    pix_diffs = np.array(pix_diffs) if pix_diffs else np.zeros(1)

    print(f"  {label:>30s} | "
          f"jerk_mean={jerks.mean():.3f} jerk_p95={np.percentile(jerks, 95):.3f} jerk_max={jerks.max():.3f} | "
          f"lag={lags.mean():.3f} | "
          f"pix_mean={pix_diffs.mean():.3f} pix_p95={np.percentile(pix_diffs, 95):.3f} | "
          f"swap_p50={np.percentile(swap_times, 50):.2f}ms")
    return out_frames


def main():
    src_img = cv2.imread(os.path.join(REPO, "bench_source.jpg"))
    src_face = get_one_face(src_img)

    print("[loading moving video]")
    cap = cv2.VideoCapture(os.path.join(REPO, "bench_input.mp4"))
    moving = []
    while True:
        ok, f = cap.read()
        if not ok: break
        moving.append(f)
    cap.release()
    print(f"  loaded {len(moving)} frames")

    print("[loading frozen + sensor noise sigma=2]")
    frozen = cv2.imread(os.path.join(REPO, "bench_frozen.jpg"))
    rng = np.random.default_rng(42)
    noisy = []
    for _ in range(80):
        noise = rng.normal(0.0, 2.0, frozen.shape).astype(np.int16)
        noisy.append(np.clip(frozen.astype(np.int16) + noise, 0, 255).astype(np.uint8))

    configs = [
        ("baseline (no filter)",      EMAFilter(alpha=1.0)),
        ("EMA alpha=0.6 (default)",    EMAFilter(alpha=0.6)),
        ("EMA alpha=0.4",              EMAFilter(alpha=0.4)),
        ("EMA alpha=0.3",              EMAFilter(alpha=0.3)),
        ("EMA alpha=0.6, reset=20",    EMAFilter(alpha=0.6, reset_threshold=20.0)),
        ("EMA alpha=0.6, reset=80",    EMAFilter(alpha=0.6, reset_threshold=80.0)),
        ("Median window=3",            MedianFilter(window=3)),
        ("Median window=5",            MedianFilter(window=5)),
        ("Median window=7",            MedianFilter(window=7)),
    ]

    print()
    print("=" * 130)
    print("MOVING VIDEO TEST")
    print("=" * 130)
    for label, flt in configs:
        flt.reset()
        run_with_filter(moving, src_face, flt, label)

    print()
    print("=" * 130)
    print("FROZEN + SENSOR NOISE (sigma=2) TEST")
    print("=" * 130)
    for label, flt in configs:
        flt.reset()
        run_with_filter(noisy, src_face, flt, label)


if __name__ == "__main__":
    main()
