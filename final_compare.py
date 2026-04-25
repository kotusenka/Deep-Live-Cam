"""Final side-by-side comparison: baseline (square mask) vs improved (hull+kps)."""
import os, sys, cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import modules.globals
modules.globals.execution_providers = ['CoreMLExecutionProvider', 'CPUExecutionProvider']
modules.globals.headless = True

from modules.face_analyser import get_one_face
from modules.processors.frame.face_swapper import pre_check, pre_start, swap_face

OUT = 'test_data/out_final'
os.makedirs(OUT, exist_ok=True)


def label(img, text):
    o = img.copy()
    cv2.rectangle(o, (0, 0), (o.shape[1], 32), (0, 0, 0), -1)
    cv2.putText(o, text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 1, cv2.LINE_AA)
    return o


def hstack(images, gap=12):
    h = max(im.shape[0] for im in images)
    aligned = [im if im.shape[0] == h else
               cv2.resize(im, (int(im.shape[1] * h / im.shape[0]), h),
                          interpolation=cv2.INTER_LINEAR)
               for im in images]
    spacer = np.full((h, gap, 3), 50, dtype=np.uint8)
    out = aligned[0]
    for im in aligned[1:]:
        out = np.concatenate([out, spacer, im], axis=1)
    return out


def vstack(images, gap=12):
    w = max(im.shape[1] for im in images)
    aligned = [im if im.shape[1] == w else
               cv2.resize(im, (w, int(im.shape[0] * w / im.shape[1])),
                          interpolation=cv2.INTER_LINEAR)
               for im in images]
    spacer = np.full((gap, w, 3), 50, dtype=np.uint8)
    out = aligned[0]
    for im in aligned[1:]:
        out = np.concatenate([out, spacer, im], axis=0)
    return out


def pad(im, frac=0.25):
    h, w = im.shape[:2]
    ph, pw = int(h * frac), int(w * frac)
    return cv2.copyMakeBorder(im, ph, ph, pw, pw, cv2.BORDER_REPLICATE)


def run_for_pair(swapper_state_set, src_face, tgt):
    """swapper_state_set: dict mapping globals attr -> value."""
    for k, v in swapper_state_set.items():
        setattr(modules.globals, k, v)
    tf = get_one_face(tgt)
    return swap_face(src_face, tf, tgt.copy())


def main():
    pre_check(); pre_start()
    pairs = [
        ('test_data/face1.jpg', 'test_data/face2.jpg'),
        ('test_data/face3.jpg', 'test_data/face4.jpg'),
        ('test_data/face5.jpg', 'test_data/face6.jpg'),
    ]

    rows = []
    for src_p, tgt_p in pairs:
        src = pad(cv2.imread(src_p))
        tgt = pad(cv2.imread(tgt_p))
        sf = get_one_face(src)

        baseline = run_for_pair({
            'hull_mask': False, 'kps_stabilize': 1.0, 'grain_match': 0.0,
        }, sf, tgt)
        hull = run_for_pair({
            'hull_mask': True, 'kps_stabilize': 0.6, 'grain_match': 0.0,
        }, sf, tgt)
        hull_grain = run_for_pair({
            'hull_mask': True, 'kps_stabilize': 0.6, 'grain_match': 0.7,
        }, sf, tgt)

        # Crop around face
        bbox = get_one_face(tgt).bbox.astype(int)
        pad_x = (bbox[2] - bbox[0]) // 2
        pad_y = (bbox[3] - bbox[1]) // 2
        h, w = tgt.shape[:2]
        x0 = max(0, bbox[0] - pad_x)
        y0 = max(0, bbox[1] - pad_y)
        x1 = min(w, bbox[2] + pad_x)
        y1 = min(h, bbox[3] + pad_y)

        crops = [
            label(tgt[y0:y1, x0:x1], 'original target'),
            label(baseline[y0:y1, x0:x1], 'OLD: square soft-alpha'),
            label(hull[y0:y1, x0:x1], 'NEW: hull+kps_stabilize'),
            label(hull_grain[y0:y1, x0:x1], 'NEW + grain_match=0.7'),
        ]
        rows.append(hstack(crops))

    final = vstack(rows)
    cv2.imwrite(f'{OUT}/before_after_3pairs.png', final)
    print(f'wrote {OUT}/before_after_3pairs.png  shape={final.shape}')


if __name__ == '__main__':
    main()
