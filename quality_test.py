"""
Quality experiments — visualize side-by-side variants of the swap pipeline.

Each experiment renders one or more variants on the same target frame and
writes them to test_data/out/<experiment>.png so we can inspect visually.

Hypotheses to test:
  H1) Aligned-space soft-alpha is too symmetric — per-face hull mask better
  H2) Reinhard color transfer (LAB) has a brittleness for cool/warm light
  H3) Adding fine grain matched to source frame closes the "plastic" gap
  H4) Temporal smoothing of kps eliminates flicker in static scenes
  H5) Preserving target eyes/eyebrows improves liveness
  H6) Lanczos resampling on upscale beats default bilinear
"""

import os
import sys
import time
import numpy as np
import cv2
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import modules.globals
modules.globals.execution_providers = ['CoreMLExecutionProvider', 'CPUExecutionProvider']
modules.globals.execution_threads = 4
modules.globals.headless = True

from modules.face_analyser import (
    get_face_analyser, get_one_face, detect_one_face_fast)
from modules.processors.frame.face_swapper import (
    pre_check, pre_start, get_face_swapper, swap_face,
    _get_soft_alpha, _fast_paste_back)
from insightface.utils.face_align import norm_crop2


OUT_DIR = 'test_data/out'
os.makedirs(OUT_DIR, exist_ok=True)


def save(name, img):
    p = os.path.join(OUT_DIR, name)
    cv2.imwrite(p, img)
    print(f'  -> {p}  shape={img.shape}')


def hstack(images, gap=10, color=(40, 40, 40)):
    h = max(im.shape[0] for im in images)
    norm = []
    for im in images:
        if im.shape[0] != h:
            scale = h / im.shape[0]
            im = cv2.resize(im, (int(im.shape[1] * scale), h),
                            interpolation=cv2.INTER_LINEAR)
        norm.append(im)
    spacer = np.full((h, gap, 3), color, dtype=np.uint8)
    out = norm[0]
    for im in norm[1:]:
        out = np.concatenate([out, spacer, im], axis=1)
    return out


def label(img, text, scale=0.7):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(out, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (255, 255, 255), 1, cv2.LINE_AA)
    return out


# ----------------------------------------------------------------------------
# Variant 1: original swap_face baseline
# ----------------------------------------------------------------------------
def variant_baseline(src_face, target_frame, target_face):
    return swap_face(src_face, target_face, target_frame.copy())


# ----------------------------------------------------------------------------
# Variant 2: aligned-face inswapper with custom alpha mask
# ----------------------------------------------------------------------------
def get_aligned_face_and_M(target_face, target_frame, image_size=128):
    """Replicate insightface's INSwapper.forward path to get aligned + M."""
    aligned_face, M = norm_crop2(target_frame, target_face.kps, image_size)
    return aligned_face, M


def run_inswapper_aligned(swapper, src_face, target_face, target_frame, size=128):
    """Run inswapper and return (BGR aligned 128x128, M affine)."""
    fake_aligned, M = swapper.get(target_frame, target_face, src_face,
                                  paste_back=False)
    return fake_aligned, M  # ((128, 128, 3) BGR, (2,3) float32)


def variant_paste_default(swapper, src_face, target_face, frame):
    fake, M = run_inswapper_aligned(swapper, src_face, target_face, frame)
    return _fast_paste_back(frame, fake, M)


# ----------------------------------------------------------------------------
# Variant 3: paste with hull-based mask (per-face, not aligned-square)
# ----------------------------------------------------------------------------
def make_hull_alpha(face_landmarks_2d_106, output_size, feather_px):
    """Build a soft mask in OUTPUT space using the face hull from landmarks."""
    h, w = output_size
    mask = np.zeros((h, w), dtype=np.uint8)
    if face_landmarks_2d_106 is None:
        return None
    # First 33 = face contour, see InsightFace landmark spec
    pts = np.asarray(face_landmarks_2d_106[:33], dtype=np.int32)
    if pts.shape[0] < 6:
        return None
    hull = cv2.convexHull(pts)
    cv2.fillConvexPoly(mask, hull, 255)
    if feather_px > 0:
        k = feather_px * 2 + 1
        mask = cv2.GaussianBlur(mask, (k, k), feather_px / 2.5)
    return mask


# ----------------------------------------------------------------------------
# Variant 4: HSL/LAB color transfer with lighting decoupling
# ----------------------------------------------------------------------------
def color_transfer_lab(source_bgr, target_bgr, mask_uint8=None):
    """Reinhard in LAB. Source = swapped face, Target = surrounding skin.

    Operates only inside the mask if provided (better local match).
    """
    src = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    tgt = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    if mask_uint8 is not None:
        m = mask_uint8 > 127
        s_mean = src[m].mean(axis=0)
        s_std = src[m].std(axis=0) + 1e-3
        t_mean = tgt[m].mean(axis=0)
        t_std = tgt[m].std(axis=0) + 1e-3
    else:
        s_mean, s_std = src.mean((0, 1)), src.std((0, 1)) + 1e-3
        t_mean, t_std = tgt.mean((0, 1)), tgt.std((0, 1)) + 1e-3
    out = (src - s_mean) * (t_std / s_std) + t_mean
    out = np.clip(out, 0, 255).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_LAB2BGR)


# ----------------------------------------------------------------------------
# Variant 5: grain injection
# ----------------------------------------------------------------------------
def estimate_noise_sigma(gray):
    """Median-deviation estimator (Donoho).  Robust noise sigma in pixel units."""
    h, w = gray.shape[:2]
    # Use Y channel high-pass response with a Laplacian
    lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
    sigma = 0.6745 * np.median(np.abs(lap - np.median(lap)))
    return float(sigma) / 4.0  # Laplacian magnifies noise by ~4


def add_grain(img_bgr, sigma):
    """Add Gaussian luminance grain at a fixed sigma."""
    if sigma <= 0:
        return img_bgr
    yuv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb).astype(np.int16)
    noise = np.random.normal(0, sigma, yuv[:, :, 0].shape).astype(np.int16)
    yuv[:, :, 0] = np.clip(yuv[:, :, 0] + noise, 0, 255)
    return cv2.cvtColor(yuv.astype(np.uint8), cv2.COLOR_YCrCb2BGR)


# ----------------------------------------------------------------------------
# Variant 6: Lanczos vs default upsample on the inswapper output
# ----------------------------------------------------------------------------
def variant_lanczos_upsample(swapper, src_face, target_face, frame):
    """Same as paste_default but the warp uses LANCZOS."""
    fake, M = run_inswapper_aligned(swapper, src_face, target_face, frame)
    h, w = frame.shape[:2]
    inv_M = cv2.invertAffineTransform(M)
    warped = cv2.warpAffine(fake, inv_M, (w, h),
                            flags=cv2.INTER_LANCZOS4,
                            borderValue=0)
    # Default mask
    alpha = _get_soft_alpha(128)
    warped_alpha = cv2.warpAffine(alpha, inv_M, (w, h),
                                  flags=cv2.INTER_LINEAR, borderValue=0)
    a = warped_alpha[..., None].astype(np.float32) / 255.0
    out = (warped.astype(np.float32) * a +
           frame.astype(np.float32) * (1 - a)).astype(np.uint8)
    return out


# ----------------------------------------------------------------------------
# Variant 7: Hull-based blending in output space
# ----------------------------------------------------------------------------
def variant_hull_blend(swapper, src_face, target_face, frame, feather=15):
    """Use 2D-106 landmarks of the target to build a tight hull mask."""
    fake, M = run_inswapper_aligned(swapper, src_face, target_face, frame)
    h, w = frame.shape[:2]
    inv_M = cv2.invertAffineTransform(M)
    warped = cv2.warpAffine(fake, inv_M, (w, h),
                            flags=cv2.INTER_LANCZOS4, borderValue=0)
    # Try to get 106 landmarks
    fa = get_face_analyser()
    lmk_model = fa.models.get('landmark_2d_106')
    landmarks = None
    if lmk_model is not None:
        target_face_for_lmk = target_face
        lmk_model.get(frame, target_face_for_lmk)
        landmarks = getattr(target_face_for_lmk, 'landmark_2d_106', None)
    mask = make_hull_alpha(landmarks, (h, w), feather_px=feather)
    if mask is None:
        # fallback to default soft alpha
        alpha = _get_soft_alpha(128)
        mask = cv2.warpAffine(alpha, inv_M, (w, h),
                              flags=cv2.INTER_LINEAR, borderValue=0)
    a = mask[..., None].astype(np.float32) / 255.0
    out = (warped.astype(np.float32) * a +
           frame.astype(np.float32) * (1 - a)).astype(np.uint8)
    return out


# ----------------------------------------------------------------------------
# Variant 8: hull blend + local color transfer
# ----------------------------------------------------------------------------
def variant_hull_color(swapper, src_face, target_face, frame, feather=15):
    fake, M = run_inswapper_aligned(swapper, src_face, target_face, frame)
    h, w = frame.shape[:2]
    inv_M = cv2.invertAffineTransform(M)
    warped = cv2.warpAffine(fake, inv_M, (w, h),
                            flags=cv2.INTER_LANCZOS4, borderValue=0)
    fa = get_face_analyser()
    lmk = fa.models.get('landmark_2d_106')
    if lmk is not None:
        lmk.get(frame, target_face)
    landmarks = getattr(target_face, 'landmark_2d_106', None)
    mask = make_hull_alpha(landmarks, (h, w), feather_px=feather)
    if mask is None:
        alpha = _get_soft_alpha(128)
        mask = cv2.warpAffine(alpha, inv_M, (w, h),
                              flags=cv2.INTER_LINEAR, borderValue=0)
    # Color match swapped face to surrounding skin (locally)
    warped_matched = color_transfer_lab(warped, frame, mask)
    a = mask[..., None].astype(np.float32) / 255.0
    out = (warped_matched.astype(np.float32) * a +
           frame.astype(np.float32) * (1 - a)).astype(np.uint8)
    return out


# ----------------------------------------------------------------------------
# Variant 9: hull + color + grain matching
# ----------------------------------------------------------------------------
def variant_hull_color_grain(swapper, src_face, target_face, frame, feather=15):
    base = variant_hull_color(swapper, src_face, target_face, frame, feather)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sigma = estimate_noise_sigma(gray)
    fa = get_face_analyser()
    landmarks = getattr(target_face, 'landmark_2d_106', None)
    mask = make_hull_alpha(landmarks, frame.shape[:2], feather_px=feather)
    if mask is None:
        return base
    grained = add_grain(base, sigma)
    a = (mask[..., None].astype(np.float32) / 255.0)
    out = (grained.astype(np.float32) * a +
           base.astype(np.float32) * (1 - a)).astype(np.uint8)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', default='test_data/source.jpg')
    ap.add_argument('--target', default='test_data/target.jpg')
    args = ap.parse_args()

    print('Loading models...')
    pre_check(); pre_start()
    swapper = get_face_swapper()
    fa = get_face_analyser()

    src = cv2.imread(args.source)
    tgt = cv2.imread(args.target)
    src_face = get_one_face(src)
    tgt_face = get_one_face(tgt)
    if src_face is None or tgt_face is None:
        print('FATAL: faces not detected')
        return 1
    print('Source bbox:', src_face.bbox.astype(int).tolist())
    print('Target bbox:', tgt_face.bbox.astype(int).tolist())

    h, w = tgt.shape[:2]
    print(f'Target frame: {w}x{h}')

    # Force-load landmark model so we can use 2d106 in variants
    _ = get_one_face(tgt)  # already loaded landmarks via _analyse_faces

    # Run variants (each gets a fresh tgt copy)
    variants = []
    print('Running variants...')

    print(' baseline (default swap_face)')
    v1 = variant_baseline(src_face, tgt, tgt_face)
    variants.append(('baseline', v1))

    print(' lanczos upsample')
    # need to re-detect target_face since modifications could mutate it
    tgt_face2 = get_one_face(tgt)
    v2 = variant_lanczos_upsample(swapper, src_face, tgt_face2, tgt)
    variants.append(('lanczos_upsample', v2))

    print(' hull blend (2d106)')
    tgt_face3 = get_one_face(tgt)
    v3 = variant_hull_blend(swapper, src_face, tgt_face3, tgt)
    variants.append(('hull_blend', v3))

    print(' hull + color transfer')
    tgt_face4 = get_one_face(tgt)
    v4 = variant_hull_color(swapper, src_face, tgt_face4, tgt)
    variants.append(('hull_color', v4))

    print(' hull + color + grain match')
    tgt_face5 = get_one_face(tgt)
    v5 = variant_hull_color_grain(swapper, src_face, tgt_face5, tgt)
    variants.append(('hull_color_grain', v5))

    # Save individual + comparison
    for name, im in variants:
        save(f'{name}.png', label(im, name))

    # Side-by-side comparison strip
    strip = hstack([label(im, name, 0.6) for name, im in variants])
    save('all_strip.png', strip)

    # Crop comparison around face
    bbox = tgt_face.bbox.astype(int)
    pad_x = (bbox[2] - bbox[0]) // 2
    pad_y = (bbox[3] - bbox[1]) // 2
    x0 = max(0, bbox[0] - pad_x)
    y0 = max(0, bbox[1] - pad_y)
    x1 = min(w, bbox[2] + pad_x)
    y1 = min(h, bbox[3] + pad_y)
    crops = [(name, im[y0:y1, x0:x1]) for name, im in variants]
    crop_strip = hstack([label(im, name, 0.6) for name, im in crops])
    save('all_crop_strip.png', crop_strip)

    # Original target for reference
    save('original_target.png', label(tgt, 'original_target'))
    save('original_source.png', label(src, 'original_source'))

    print('done. inspect test_data/out/all_crop_strip.png')


if __name__ == '__main__':
    sys.exit(main() or 0)
