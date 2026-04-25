"""Quality experiments — motion stability + cross-lighting comparison."""
import os
import sys
import numpy as np
import cv2
import argparse
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import modules.globals
modules.globals.execution_providers = ['CoreMLExecutionProvider', 'CPUExecutionProvider']
modules.globals.execution_threads = 4
modules.globals.headless = True
# force-load landmark model
modules.globals.mouth_mask = True

from modules.face_analyser import (
    get_face_analyser, get_one_face, detect_one_face_fast)
from modules.processors.frame.face_swapper import (
    pre_check, pre_start, get_face_swapper, swap_face,
    _get_soft_alpha)


OUT_DIR = 'test_data/out2'
os.makedirs(OUT_DIR, exist_ok=True)


def save(name, img):
    p = os.path.join(OUT_DIR, name)
    cv2.imwrite(p, img)
    print(f'  -> {p}')


def label(img, text, scale=0.7):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(out, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def hstack(images, gap=10):
    h = max(im.shape[0] for im in images)
    norm = []
    for im in images:
        if im.shape[0] != h:
            scale = h / im.shape[0]
            im = cv2.resize(im, (int(im.shape[1] * scale), h),
                            interpolation=cv2.INTER_LINEAR)
        norm.append(im)
    spacer = np.full((h, gap, 3), 40, dtype=np.uint8)
    out = norm[0]
    for im in norm[1:]:
        out = np.concatenate([out, spacer, im], axis=1)
    return out


def vstack(images, gap=10):
    w = max(im.shape[1] for im in images)
    norm = []
    for im in images:
        if im.shape[1] != w:
            scale = w / im.shape[1]
            im = cv2.resize(im, (w, int(im.shape[0] * scale)),
                            interpolation=cv2.INTER_LINEAR)
        norm.append(im)
    spacer = np.full((gap, w, 3), 40, dtype=np.uint8)
    out = norm[0]
    for im in norm[1:]:
        out = np.concatenate([out, spacer, im], axis=0)
    return out


# ---- variants ------------------------------------------------------

from quality_test import (
    variant_baseline, variant_lanczos_upsample, variant_hull_blend,
    variant_hull_color, variant_hull_color_grain,
    color_transfer_lab, make_hull_alpha, add_grain, estimate_noise_sigma,
    run_inswapper_aligned)


# ---- temporal stability test ------------------------------------------------------
def temporal_test(swapper, src_face, base_target, n_frames=12):
    """Apply small synthetic camera motion (translation + rotation)
    and run swap independently per frame.  Measure inter-frame
    pixel difference inside the face region as a proxy for flicker.
    """
    h, w = base_target.shape[:2]
    diffs_baseline = []
    diffs_hull = []

    prev_baseline = None
    prev_hull = None

    for i in range(n_frames):
        # small jitter — like minor head movement
        theta = (i - n_frames / 2) * 0.4  # degrees
        tx = (i - n_frames / 2) * 1.2
        ty = (i - n_frames / 2) * 0.5
        M_aff = cv2.getRotationMatrix2D((w / 2, h / 2), theta, 1.0)
        M_aff[0, 2] += tx
        M_aff[1, 2] += ty
        moved = cv2.warpAffine(base_target, M_aff, (w, h), flags=cv2.INTER_LINEAR)
        target_face = get_one_face(moved)
        if target_face is None:
            print(f' frame {i}: no face')
            continue

        # baseline
        v_b = swap_face(src_face, target_face, moved.copy())
        # hull-color-grain
        target_face2 = get_one_face(moved)
        v_h = variant_hull_color_grain(swapper, src_face, target_face2,
                                       moved.copy())

        if prev_baseline is not None:
            d = cv2.absdiff(v_b, prev_baseline).mean()
            diffs_baseline.append(d)
        if prev_hull is not None:
            d = cv2.absdiff(v_h, prev_hull).mean()
            diffs_hull.append(d)
        prev_baseline = v_b
        prev_hull = v_h

        if i in (0, n_frames // 2, n_frames - 1):
            save(f'temporal_baseline_{i:02d}.png', label(v_b, f'baseline f{i}'))
            save(f'temporal_hull_{i:02d}.png', label(v_h, f'hull_color_grain f{i}'))

    print('Temporal mean inter-frame diff:')
    print(f'  baseline: {np.mean(diffs_baseline):.3f}')
    print(f'  hull_color_grain: {np.mean(diffs_hull):.3f}')


# ---- New variant: kps-stabilised swap ------------------------------------------------------
class KPSStabilizer:
    """Exponential moving average over face keypoints to suppress jitter."""
    def __init__(self, alpha=0.6):
        self.alpha = alpha
        self.prev_kps = None

    def __call__(self, face):
        if face is None:
            return face
        if self.prev_kps is None or self.prev_kps.shape != face.kps.shape:
            self.prev_kps = face.kps.astype(np.float32).copy()
            return face
        new = self.alpha * face.kps + (1 - self.alpha) * self.prev_kps
        face.kps = new.astype(np.float32)
        self.prev_kps = face.kps.copy()
        return face


# ---- Quality metrics ------------------------------------------------------
def measure_quality(orig_target_bgr, swapped_bgr, target_face):
    """Quantitative metrics:
      - skin color match: how close swapped face mean LAB matches surrounding skin
      - sharpness preservation: laplacian variance ratio
      - boundary continuity: gradient magnitude at face hull edge
    """
    h, w = orig_target_bgr.shape[:2]
    lmk = getattr(target_face, 'landmark_2d_106', None)
    if lmk is None:
        return None
    pts = np.asarray(lmk[:33], dtype=np.int32)
    hull = cv2.convexHull(pts)

    inside = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(inside, hull, 255)
    # surrounding skin: dilated ring outside face
    dilated = cv2.dilate(inside, np.ones((50, 50), np.uint8))
    ring = cv2.subtract(dilated, inside)

    swapped_lab = cv2.cvtColor(swapped_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    orig_lab = cv2.cvtColor(orig_target_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    inside_b = inside > 0
    ring_b = ring > 0
    # The "skin" reference — surrounding ring
    skin_mean = orig_lab[ring_b].mean(axis=0)
    swapped_mean = swapped_lab[inside_b].mean(axis=0)
    color_distance = np.linalg.norm(swapped_mean - skin_mean)

    # Sharpness inside face region (Laplacian variance)
    swapped_gray = cv2.cvtColor(swapped_bgr, cv2.COLOR_BGR2GRAY)
    orig_gray = cv2.cvtColor(orig_target_bgr, cv2.COLOR_BGR2GRAY)
    swapped_lap = cv2.Laplacian(swapped_gray, cv2.CV_64F)
    orig_lap = cv2.Laplacian(orig_gray, cv2.CV_64F)
    swapped_var = swapped_lap[inside_b].var()
    orig_var = orig_lap[inside_b].var()

    # Boundary discontinuity: gradient magnitude on the hull edge
    edge_mask = cv2.morphologyEx(inside, cv2.MORPH_GRADIENT,
                                 np.ones((5, 5), np.uint8))
    grad_x = cv2.Sobel(swapped_gray, cv2.CV_64F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(swapped_gray, cv2.CV_64F, 0, 1, ksize=5)
    grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
    boundary_grad = grad_mag[edge_mask > 0].mean()

    return dict(
        color_distance_LAB=float(color_distance),
        sharpness_face=float(swapped_var),
        sharpness_orig=float(orig_var),
        boundary_grad=float(boundary_grad),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', default='test_data/face1.jpg')
    ap.add_argument('--target', default='test_data/face2.jpg')
    args = ap.parse_args()

    print('Loading models...')
    pre_check(); pre_start()
    swapper = get_face_swapper()

    src = cv2.imread(args.source)
    tgt = cv2.imread(args.target)

    # pad to give detector context
    def pad(im):
        h, w = im.shape[:2]
        ph, pw = h // 4, w // 4
        return cv2.copyMakeBorder(im, ph, ph, pw, pw, cv2.BORDER_REPLICATE)

    src = pad(src)
    tgt = pad(tgt)

    src_face = get_one_face(src)
    tgt_face = get_one_face(tgt)
    print(f' source bbox: {src_face.bbox.astype(int).tolist()}')
    print(f' target bbox: {tgt_face.bbox.astype(int).tolist()}')

    # ---- Quality variants ----
    print('\n=== Static quality test ===')
    variants = []
    variants.append(('baseline',
                     variant_baseline(src_face, tgt, get_one_face(tgt))))
    variants.append(('lanczos',
                     variant_lanczos_upsample(swapper, src_face, get_one_face(tgt), tgt)))
    variants.append(('hull_blend',
                     variant_hull_blend(swapper, src_face, get_one_face(tgt), tgt)))
    variants.append(('hull_color',
                     variant_hull_color(swapper, src_face, get_one_face(tgt), tgt)))
    variants.append(('hull_color_grain',
                     variant_hull_color_grain(swapper, src_face, get_one_face(tgt), tgt)))

    print('\n=== Quality metrics (vs original target skin) ===')
    print(f'{"variant":>20s}  {"color_d":>10s}  {"sharp":>10s}  {"bnd_grad":>10s}')
    for name, im in variants:
        metrics = measure_quality(tgt, im, get_one_face(tgt))
        if metrics:
            print(f'{name:>20s}  '
                  f'{metrics["color_distance_LAB"]:10.3f}  '
                  f'{metrics["sharpness_face"]:10.1f}  '
                  f'{metrics["boundary_grad"]:10.1f}')

    # comparison strip
    cropped = []
    bbox = tgt_face.bbox.astype(int)
    pad_x = (bbox[2] - bbox[0]) // 3
    pad_y = (bbox[3] - bbox[1]) // 3
    h, w = tgt.shape[:2]
    x0 = max(0, bbox[0] - pad_x)
    y0 = max(0, bbox[1] - pad_y)
    x1 = min(w, bbox[2] + pad_x)
    y1 = min(h, bbox[3] + pad_y)
    for name, im in variants:
        cropped.append(label(im[y0:y1, x0:x1], name, 0.6))
    save('static_strip.png', hstack(cropped))
    save('original_target_static.png', label(tgt[y0:y1, x0:x1], 'original_target'))
    save('original_source_static.png', label(src, 'original_source'))

    # ---- Temporal test ----
    print('\n=== Temporal stability test ===')
    temporal_test(swapper, src_face, tgt, n_frames=8)

    print('\nDone. inspect test_data/out2/*.png')


if __name__ == '__main__':
    sys.exit(main() or 0)
