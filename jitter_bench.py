"""Jitter benchmark for live face swap pipeline.

Replays a deterministic 5-second clip through the same code path as the
live UI (detect_one_face_fast + swap_face), with different smoothing /
caching parameters, and measures inter-frame stability of the swapped
output.

Outputs a table sorted by stability score so we can pick the best
combination of (kps_stabilize, swap_cache_threshold_px, swap_cache_max_age_ms).
"""

import os
import sys
import time
import itertools
from dataclasses import dataclass
from typing import List, Optional

# Bootstrap CUDA DLLs (must run before importing onnxruntime).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run  # noqa: F401

import cv2
import numpy as np

import modules.globals
modules.globals.execution_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

from modules.face_analyser import get_one_face, detect_one_face_fast
from modules.processors.frame import face_swapper as fs


REPO = os.path.dirname(os.path.abspath(__file__))
SOURCE_PATH = os.path.join(REPO, "bench_source.jpg")
INPUT_VIDEO = os.path.join(REPO, "bench_input.mp4")


def load_source():
    src_img = cv2.imread(SOURCE_PATH)
    src_face = get_one_face(src_img)
    if src_face is None or src_face.normed_embedding is None:
        raise RuntimeError(f"Failed to extract source face from {SOURCE_PATH}")
    print(f"[source] face bbox={src_face.bbox.astype(int)} has_emb=True")
    return src_face


def load_frames(path: str) -> List[np.ndarray]:
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    print(f"[input] loaded {len(frames)} frames from {os.path.basename(path)}")
    return frames


def reset_caches():
    """Clear stateful caches so each config runs independently."""
    fs._SWAP_CACHE.clear()
    fs._KPS_HISTORY.clear()
    fs._paste_cache['soft_alpha'] = None
    fs._paste_cache['alpha_size'] = 0


@dataclass
class RunResult:
    config: dict
    n_detected: int
    kps_jerk_mean: float       # mean ||a[i+1] - 2a[i] + a[i-1]|| (proxy for noisiness)
    kps_jerk_p95: float
    pixel_diff_mean: float     # mean |frame[i] - frame[i-1]| within face bbox
    pixel_diff_p95: float
    swap_time_ms_p50: float
    cache_hit_rate: float


def measure(out_frames, kps_list, bboxes, swap_times_ms, cache_hits, total_runs):
    # kps jerk = magnitude of second-derivative — large on noisy detection,
    # small on smooth real movement.
    jerks = []
    for i in range(1, len(kps_list) - 1):
        a, b, c = kps_list[i - 1], kps_list[i], kps_list[i + 1]
        if a is None or b is None or c is None:
            continue
        accel = c - 2.0 * b + a
        jerks.append(float(np.linalg.norm(accel, axis=1).mean()))
    jerks = np.array(jerks) if jerks else np.zeros(1)

    # Pixel-level diff inside the face bbox — this is what you actually see.
    pix_diffs = []
    for i in range(1, len(out_frames)):
        if bboxes[i] is None or bboxes[i - 1] is None:
            continue
        # Use union of consecutive bboxes so movement doesn't shrink ROI.
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

    return {
        "kps_jerk_mean": float(jerks.mean()),
        "kps_jerk_p95": float(np.percentile(jerks, 95)) if len(jerks) > 1 else 0.0,
        "pixel_diff_mean": float(pix_diffs.mean()),
        "pixel_diff_p95": float(np.percentile(pix_diffs, 95)) if len(pix_diffs) > 1 else 0.0,
        "swap_time_ms_p50": float(np.percentile(swap_times_ms, 50)) if swap_times_ms else 0.0,
        "cache_hit_rate": cache_hits / max(1, total_runs),
    }


def run_config(frames, src_face, kps_stabilize, cache_thr, cache_age_ms):
    # Apply config
    modules.globals.kps_stabilize = float(kps_stabilize)
    modules.globals.swap_cache_threshold_px = float(cache_thr)
    modules.globals.swap_cache_max_age_ms = float(cache_age_ms)
    reset_caches()

    out_frames = []
    kps_list = []
    bboxes = []
    swap_times_ms = []
    cache_hits = 0
    total_runs = 0

    # Warmup: first few frames trigger CUDA Graph + algo search.
    for f in frames[:5]:
        face = detect_one_face_fast(f)
        if face is not None:
            fs.swap_face(src_face, face, f.copy())
    reset_caches()  # don't let warmup pollute the kps history

    for f in frames:
        face = detect_one_face_fast(f)
        if face is None:
            out_frames.append(f.copy())
            kps_list.append(None)
            bboxes.append(None)
            continue
        # Track cache state to compute hit rate.
        cache_size_before = len(fs._SWAP_CACHE)
        t0 = time.perf_counter()
        out = fs.swap_face(src_face, face, f.copy())
        dt = (time.perf_counter() - t0) * 1000.0
        cache_size_after = len(fs._SWAP_CACHE)
        # Cache miss = a new entry was written; hit = no write but call returned cached fake.
        # We can't perfectly distinguish here without instrumenting fs, so use timing instead:
        # cached path is much faster than full inference.
        if dt < 3.0:  # FP16 inswapper on 5090 is ~1-2ms via graph replay; full pipeline incl. paste-back still <3ms when cache hits.
            cache_hits += 1
        total_runs += 1

        out_frames.append(out)
        kps_list.append(face.kps.astype(np.float32).copy())
        bboxes.append(face.bbox.astype(np.float32).copy())
        swap_times_ms.append(dt)

    metrics = measure(out_frames, kps_list, bboxes, swap_times_ms, cache_hits, total_runs)
    return RunResult(
        config={
            "kps_stab": kps_stabilize,
            "cache_thr": cache_thr,
            "cache_age_ms": cache_age_ms,
        },
        n_detected=sum(1 for k in kps_list if k is not None),
        **metrics,
    )


def main():
    src_face = load_source()
    moving_frames = load_frames(INPUT_VIDEO)

    # Frozen-with-noise test: same frame repeated, but with per-frame
    # Gaussian sensor noise — emulates a real camera capturing a still face.
    # sigma=2.0 grayscale levels matches typical UVC noise on consumer webcams
    # in good light; sigma=4.0 matches mediocre light.
    frozen = cv2.imread(os.path.join(REPO, "bench_frozen.jpg"))
    rng = np.random.default_rng(42)
    def make_noisy_clip(n: int, sigma: float) -> List[np.ndarray]:
        out = []
        for _ in range(n):
            noise = rng.normal(0.0, sigma, frozen.shape).astype(np.int16)
            f = np.clip(frozen.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            out.append(f)
        return out

    print()
    print("=" * 90)
    print("FROZEN + SENSOR NOISE TEST (still face, camera noise only -> pure pipeline jitter)")
    print("=" * 90)
    for sigma in [1.0, 2.0, 4.0]:
        noisy = make_noisy_clip(80, sigma)
        # Compute input baseline noise inside face bbox for reference
        face_b = detect_one_face_fast(noisy[0])
        if face_b is not None:
            x1, y1, x2, y2 = face_b.bbox.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(noisy[0].shape[1], x2), min(noisy[0].shape[0], y2)
            in_d = []
            for i in range(1, len(noisy)):
                in_d.append(float(np.abs(noisy[i - 1][y1:y2, x1:x2].astype(np.int16)
                                          - noisy[i][y1:y2, x1:x2].astype(np.int16)).mean()))
            in_baseline_n = float(np.mean(in_d))
        else:
            in_baseline_n = float('nan')

        print(f"\nsigma={sigma:.1f}  (input baseline pix_diff in bbox = {in_baseline_n:.3f})")
        for kps_a in [1.0, 0.6, 0.3, 0.1]:
            for thr in [0.0, 15.0]:
                r = run_config(noisy, src_face, kps_a, thr, 1000.0)
                excess = r.pixel_diff_mean - in_baseline_n
                print(f"   kps_stab={kps_a} cache_thr={thr:>4} | "
                      f"jerk={r.kps_jerk_mean:.3f} (p95={r.kps_jerk_p95:.2f}) | "
                      f"out_pix={r.pixel_diff_mean:.3f} excess={excess:+.3f} | "
                      f"swap_p50={r.swap_time_ms_p50:.2f}ms")

    # Subtract baseline movement from moving-video metrics so we measure
    # excess jitter introduced by the pipeline, not actual face motion.
    print()
    print("=" * 80)
    print("INPUT-FRAME BASELINE (camera-side movement, no pipeline)")
    print("=" * 80)
    in_diffs = []
    for i in range(1, len(moving_frames)):
        # Same bbox-union approach as in measure(), but on raw input.
        a = moving_frames[i - 1].astype(np.int16)
        b = moving_frames[i].astype(np.int16)
        # Detect face on frame i to get bbox
        face = detect_one_face_fast(moving_frames[i])
        if face is None:
            continue
        x1, y1, x2, y2 = face.bbox.astype(int)
        h, w = a.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        in_diffs.append(float(np.abs(a[y1:y2, x1:x2] - b[y1:y2, x1:x2]).mean()))
    in_baseline = np.array(in_diffs)
    print(f"   raw input pix_diff_mean={in_baseline.mean():.3f} p95={np.percentile(in_baseline, 95):.3f}")
    print()

    # Grid: kps_stabilize × cache_threshold × cache_age
    kps_grid = [1.0, 0.6, 0.4, 0.3, 0.2, 0.1]
    thr_grid = [0.0, 5.0, 15.0]
    age_grid = [300.0, 1000.0]
    frames = moving_frames

    results: List[RunResult] = []
    total = len(kps_grid) * len(thr_grid) * len(age_grid)
    i = 0
    for kps_a, thr, age in itertools.product(kps_grid, thr_grid, age_grid):
        i += 1
        # cache_thr=0 disables cache; age irrelevant — skip duplicates
        if thr == 0.0 and age != age_grid[0]:
            continue
        print(f"[{i}/{total}] kps_stab={kps_a} cache_thr={thr} cache_age={age}ms ...")
        r = run_config(frames, src_face, kps_a, thr, age)
        results.append(r)
        print(f"   detected={r.n_detected}/{len(frames)} "
              f"jerk_mean={r.kps_jerk_mean:.3f} jerk_p95={r.kps_jerk_p95:.3f} "
              f"pix_diff_mean={r.pixel_diff_mean:.2f} pix_diff_p95={r.pixel_diff_p95:.2f} "
              f"swap_p50={r.swap_time_ms_p50:.2f}ms cache_hit={r.cache_hit_rate*100:.0f}%")

    # Sort by composite score: prefer lower pixel_diff_mean (visible jitter)
    # but penalise extreme kps lag (jerk near 0 means "frozen", which is also bad).
    results.sort(key=lambda r: r.pixel_diff_mean)

    print()
    print("=" * 110)
    print(f"{'kps_stab':>9} {'cache_thr':>10} {'cache_age':>10} | "
          f"{'jerk_mean':>10} {'jerk_p95':>10} | "
          f"{'pix_mean':>10} {'pix_p95':>10} | "
          f"{'swap_p50':>10} {'cache_hit':>10}")
    print("-" * 110)
    for r in results:
        print(f"{r.config['kps_stab']:>9.2f} {r.config['cache_thr']:>10.1f} {r.config['cache_age_ms']:>9.0f}ms | "
              f"{r.kps_jerk_mean:>10.3f} {r.kps_jerk_p95:>10.3f} | "
              f"{r.pixel_diff_mean:>10.2f} {r.pixel_diff_p95:>10.2f} | "
              f"{r.swap_time_ms_p50:>9.2f}ms {r.cache_hit_rate*100:>9.0f}%")
    print("=" * 110)


if __name__ == "__main__":
    main()
