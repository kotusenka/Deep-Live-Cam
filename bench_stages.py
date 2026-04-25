"""
Stage-by-stage profiling for deep-live-cam.

Goal: identify which stages dominate per-frame latency on Apple Silicon
so we know where quality changes can be added without losing FPS.
"""
import os
import sys
import time
import statistics
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np


def percentile(xs, p):
    if not xs:
        return 0.0
    s = sorted(xs)
    k = int(round((p / 100.0) * (len(s) - 1)))
    return s[k]


def fmt(label, samples_ms):
    if not samples_ms:
        return f"{label:>32s}: (no samples)"
    return (
        f"{label:>32s}: "
        f"avg={statistics.mean(samples_ms):6.2f}ms  "
        f"p50={percentile(samples_ms, 50):6.2f}  "
        f"p90={percentile(samples_ms, 90):6.2f}  "
        f"p99={percentile(samples_ms, 99):6.2f}  "
        f"min={min(samples_ms):6.2f}  max={max(samples_ms):6.2f}  "
        f"n={len(samples_ms)}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', default='test_data/source.jpg')
    ap.add_argument('--target', default='test_data/target.jpg')
    ap.add_argument('--frames', type=int, default=120)
    ap.add_argument('--input-size', default='960x540')
    ap.add_argument('--mouth-mask', action='store_true')
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--executor', default='coreml',
                    choices=['coreml', 'cpu', 'cuda'])
    args = ap.parse_args()

    import modules.globals
    if args.executor == 'coreml':
        modules.globals.execution_providers = [
            'CoreMLExecutionProvider', 'CPUExecutionProvider']
    elif args.executor == 'cuda':
        modules.globals.execution_providers = [
            'CUDAExecutionProvider', 'CPUExecutionProvider']
    else:
        modules.globals.execution_providers = ['CPUExecutionProvider']

    modules.globals.execution_threads = args.threads
    modules.globals.headless = True
    modules.globals.mouth_mask = args.mouth_mask
    modules.globals.color_correction = False
    modules.globals.face_enhancer_enabled = False
    modules.globals.show_mouth_mask_box = False
    modules.globals.use_seamless_clone = False

    print(f"=== bench_stages: {args.input_size}, executor={args.executor}, "
          f"threads={args.threads}, mouth_mask={args.mouth_mask} ===")

    width, height = [int(x) for x in args.input_size.split('x')]

    # Load test faces
    t0 = time.perf_counter()
    src = cv2.imread(args.source)
    tgt = cv2.imread(args.target)
    if src is None or tgt is None:
        print('FATAL: could not load test images')
        return 1
    print(f"Loaded source={src.shape}, target={tgt.shape} in "
          f"{(time.perf_counter()-t0)*1000:.1f}ms")

    # Preload analyser + swapper
    print('Loading models...')
    from modules.face_analyser import (
        get_face_analyser, get_one_face, get_many_faces,
        detect_one_face_fast, detect_many_faces_fast)
    fa = get_face_analyser()
    print(' face_analyser ok')

    from modules.processors.frame.face_swapper import (
        pre_check, pre_start, get_face_swapper, swap_face)
    pre_check()
    pre_start()
    sw = get_face_swapper()
    print(' face_swapper ok')

    # Detect source face once
    src_face = get_one_face(src)
    if src_face is None:
        print('FATAL: source face not detected')
        return 1
    print(f' source face: bbox={src_face.bbox.astype(int).tolist()}')

    # Build a simulated camera frame: scale target.jpg to webcam size
    base_frame = cv2.resize(tgt, (width, height), interpolation=cv2.INTER_AREA)

    # Pre-detect target face for quick paths
    target_face = get_one_face(base_frame)
    if target_face is None:
        print('FATAL: target face not detected in base_frame')
        return 1
    print(f' target face: bbox={target_face.bbox.astype(int).tolist()}')

    # Warmup
    print('Warming up (10 frames)...')
    for _ in range(10):
        f = base_frame.copy()
        _ = get_many_faces(f)
        _ = swap_face(src_face, target_face, f)

    # Stage timing
    detect_one = []
    detect_one_fast = []
    detect_many = []
    swap_only = []  # just inswapper inference + paste
    full_pipeline = []  # detect_fast + swap (live mode equivalent)

    print(f'Profiling {args.frames} frames...')
    for i in range(args.frames):
        f = base_frame.copy()

        t = time.perf_counter()
        face1 = get_one_face(f)
        detect_one.append((time.perf_counter() - t) * 1000)

        t = time.perf_counter()
        face_fast = detect_one_face_fast(f)
        detect_one_fast.append((time.perf_counter() - t) * 1000)

        t = time.perf_counter()
        many = get_many_faces(f)
        detect_many.append((time.perf_counter() - t) * 1000)

        t = time.perf_counter()
        _ = swap_face(src_face, target_face, f.copy())
        swap_only.append((time.perf_counter() - t) * 1000)

        t = time.perf_counter()
        f2 = base_frame.copy()
        face = detect_one_face_fast(f2)
        if face is not None:
            f2 = swap_face(src_face, face, f2)
        full_pipeline.append((time.perf_counter() - t) * 1000)

    # Native cv2 op baselines for reference
    print('Reference cv2 ops...')
    bgr2rgb = []
    resize_op = []
    for _ in range(50):
        t = time.perf_counter()
        _ = cv2.cvtColor(base_frame, cv2.COLOR_BGR2RGB)
        bgr2rgb.append((time.perf_counter() - t) * 1000)
        t = time.perf_counter()
        _ = cv2.resize(base_frame, (640, 360))
        resize_op.append((time.perf_counter() - t) * 1000)

    print()
    print('=== Per-stage timings ===')
    print(fmt('detect_one (det+rec)', detect_one))
    print(fmt('detect_one_fast (det only)', detect_one_fast))
    print(fmt('detect_many (det+rec)', detect_many))
    print(fmt('swap_face (with detect cached)', swap_only))
    print(fmt('detect+swap pipeline', full_pipeline))
    print(fmt('cv2.cvtColor BGR->RGB', bgr2rgb))
    print(fmt(f'cv2.resize {width}x{height}->640x360', resize_op))

    avg_pipe = statistics.mean(full_pipeline)
    print()
    print(f'theoretical FPS at avg pipeline = {1000/avg_pipe:.1f}')
    print(f'theoretical FPS without detection (cached) = '
          f'{1000/statistics.mean(swap_only):.1f}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
