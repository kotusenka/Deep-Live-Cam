"""Smoke test for RTX 50-series (Blackwell / sm_120) on this fork.

Steps:
  1. Confirm onnxruntime-gpu sees CUDAExecutionProvider.
  2. Build a minimal CUDA InferenceSession on inswapper_128_fp16.onnx and
     run one dummy forward pass — proves sm_120 kernels are present.
  3. Optionally load GFPGANv1.4 via the project's enhancer pipeline and
     run one frame through it.

Exit codes:
  0 — everything OK
  1 — CUDA EP missing or unusable
  2 — swap model failed
  3 — GFPGAN model failed
"""

import os
import sys
import time
import traceback

# Run.py's CUDA-DLL bootstrap (needed before importing onnxruntime on Win)
import run  # noqa: F401  (side-effect: PATH patched)

import numpy as np
import onnxruntime as ort


REPO = os.path.dirname(os.path.abspath(__file__))
SWAP_MODEL = os.path.join(REPO, "models", "inswapper_128_fp16.onnx")
GFPGAN_MODEL = os.path.join(REPO, "models", "gfpgan-1024.onnx")


def step1_providers():
    print("=" * 60)
    print("[1/3] onnxruntime providers")
    print("=" * 60)
    print(f"onnxruntime version: {ort.__version__}")
    providers = ort.get_available_providers()
    print(f"available providers: {providers}")
    if "CUDAExecutionProvider" not in providers:
        print("FAIL: CUDAExecutionProvider missing — onnxruntime-gpu didn't bind to CUDA.")
        return False
    device = ort.get_device()
    print(f"ort.get_device(): {device}")
    print("OK")
    return True


def step2_swap():
    print()
    print("=" * 60)
    print("[2/3] inswapper_128_fp16 forward pass on CUDA")
    print("=" * 60)
    if not os.path.isfile(SWAP_MODEL):
        print(f"FAIL: model not found at {SWAP_MODEL}")
        return False
    try:
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        t0 = time.time()
        session = ort.InferenceSession(
            SWAP_MODEL,
            sess_options=sess_opts,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        load_ms = (time.time() - t0) * 1000
        print(f"session loaded in {load_ms:.0f} ms")
        active = session.get_providers()
        print(f"session providers: {active}")
        if active[0] != "CUDAExecutionProvider":
            print("FAIL: session didn't pick CUDA as primary EP.")
            return False

        # Build dummy inputs matching inswapper_128's signature.
        feeds = {}
        for inp in session.get_inputs():
            shape = [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape]
            dtype = np.float32 if "float" in inp.type else np.int64
            feeds[inp.name] = np.zeros(shape, dtype=dtype)
            print(f"  input {inp.name}: shape={shape} dtype={dtype.__name__}")

        # Warm-up + timed run.
        session.run(None, feeds)
        runs = 10
        t0 = time.time()
        for _ in range(runs):
            session.run(None, feeds)
        avg_ms = (time.time() - t0) * 1000 / runs
        print(f"avg inference: {avg_ms:.1f} ms ({1000 / avg_ms:.1f} fps theoretical)")
        print("OK")
        return True
    except Exception as e:
        print(f"FAIL: {e}")
        traceback.print_exc()
        return False


def step3_gfpgan():
    print()
    print("=" * 60)
    print("[3/3] GFPGAN enhancer pipeline")
    print("=" * 60)
    if not os.path.isfile(GFPGAN_MODEL):
        print(f"SKIP: GFPGAN weights not found at {GFPGAN_MODEL}")
        return True  # not a failure — optional
    try:
        import modules.globals as g
        g.execution_providers = ["CUDAExecutionProvider"]

        from modules.processors.frame import face_enhancer
        session = face_enhancer.get_face_enhancer()
        active = session.get_providers()
        print(f"enhancer providers: {active}")
        if active[0] != "CUDAExecutionProvider":
            print("FAIL: enhancer didn't pick CUDA as primary EP.")
            return False

        # Read the model's expected input size and run a dummy pass.
        inp = session.get_inputs()[0]
        in_size = inp.shape[-1] if isinstance(inp.shape[-1], int) else 512
        print(f"enhancer input: {inp.name} shape={inp.shape}")
        feed = {inp.name: np.zeros(
            [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape],
            dtype=np.float32,
        )}
        # Warm-up + 3 timed runs.
        session.run(None, feed)
        runs = 3
        t0 = time.time()
        for _ in range(runs):
            session.run(None, feed)
        avg_ms = (time.time() - t0) * 1000 / runs
        print(f"avg enhance @ {in_size}px: {avg_ms:.0f} ms ({1000 / avg_ms:.1f} fps theoretical)")
        print("OK")
        return True
    except Exception as e:
        print(f"FAIL: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    if not step1_providers():
        sys.exit(1)
    if not step2_swap():
        sys.exit(2)
    if not step3_gfpgan():
        sys.exit(3)
    print()
    print("ALL CHECKS PASSED — RTX 5090 sm_120 path is live on this fork.")
