"""Remove the background from leaf images, keeping only the leaves.

Reads  image_data/<cycle-N>/*.jpg
Writes data_preprocessed/<cycle-N>/*.png  (leaves on a transparent background)
"""
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent          # project folder
SRC = ROOT / "image_data"
DST = ROOT / "data_preprocessed"

MIN_LEAF_AREA = 8000  # px, drops the sensor board, label and specks
MIN_PIECE_AREA = 500  # px, smallest leaf end worth keeping when it touches a leaf
MAX_HOLE_AREA = 2500  # px, holes up to this size (lesions) are filled; bigger ones are gaps between leaves


def _fill_small_holes(mask):
    inv = (mask == 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    out = mask.copy()
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        touches_border = x == 0 or y == 0 or x + w == mask.shape[1] or y + h == mask.shape[0]
        if not touches_border and area <= MAX_HOLE_AREA:
            out[labels == i] = 255
    return out


def _keep_leaves(mask):
    """Keep large components, plus smaller pieces (leaf ends cut off by a band) right next to one."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    big = np.zeros_like(mask)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= MIN_LEAF_AREA:
            big[labels == i] = 255
    near = cv2.dilate(big, _ellipse(41))
    keep = big.copy()
    for i in range(1, n):
        if MIN_PIECE_AREA <= stats[i, cv2.CC_STAT_AREA] < MIN_LEAF_AREA and near[labels == i].any():
            keep[labels == i] = 255
    return keep


def _ellipse(d):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d))


def _coarse_mask(hsv, smin=55, vmin=100, bright=256):
    # yellow-green to green, reasonably saturated and bright.
    # Excludes the dark board, orange rubber bands, purple sensor and white label.
    mask = cv2.inRange(hsv, (29, smin, vmin), (90, 255, 255))
    # Pale but bright leaf (midrib, sun-bleached areas) is low in saturation.
    mask |= cv2.inRange(hsv, (29, 30, bright), (90, 255, 255))

    # Rubber bands cut across the leaves: bridge the gaps they leave, but only where the band
    # itself is (closing everywhere would also fill the narrow gaps between neighbouring leaves).
    band = cv2.dilate(cv2.inRange(hsv, (3, 90, 90), (22, 255, 255)), _ellipse(3))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _ellipse(15))
    mask = mask | (closed & band)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _ellipse(5))

    keep = _keep_leaves(mask)

    # Diseased spots (brown/yellow lesions) fall outside the green range: fill them back in.
    return _fill_small_holes(keep)


def leaf_mask(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    first = _coarse_mask(hsv)
    # The first pass also lets in lit patches of board next to the leaves. Measure this image's
    # own leaf colour from the interior and demand pixels are close to it.
    interior = cv2.erode(first, _ellipse(25)) > 0
    med_s, med_v = np.median(hsv[..., 1][interior]), np.median(hsv[..., 2][interior])
    coarse = _coarse_mask(hsv, smin=max(55, 0.65 * med_s), vmin=max(85, 0.5 * med_v), bright=0.85 * med_v)

    # Board seen between/inside close leaves: thin non-leaf regions enclosed by leaf. Marking them
    # as certain background stops GrabCut from swallowing them (they are dark like a shaded leaf).
    enclosed = cv2.morphologyEx(coarse, cv2.MORPH_CLOSE, _ellipse(61)) & ~cv2.dilate(coarse, _ellipse(5))
    enclosed = cv2.morphologyEx(enclosed, cv2.MORPH_OPEN, _ellipse(5))

    # Refine the edges with GrabCut, seeded by the coarse mask. The narrow band around each leaf
    # is "probably background" so GrabCut learns the gray board/shadow colour and cuts it away.
    gc = np.full(coarse.shape, cv2.GC_BGD, np.uint8)
    gc[cv2.dilate(coarse, _ellipse(31)) > 0] = cv2.GC_PR_BGD
    gc[coarse > 0] = cv2.GC_PR_FGD
    gc[cv2.erode(coarse, _ellipse(11)) > 0] = cv2.GC_FGD
    gc[enclosed > 0] = cv2.GC_BGD
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    cv2.grabCut(bgr, gc, None, bgd, fgd, 4, cv2.GC_INIT_WITH_MASK)
    mask = ((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)).astype(np.uint8) * 255

    # Anything GrabCut added beyond the confident leaf pixels must still look like leaf
    # (saturated, not gray board) and lie close to them.
    added = (mask > 0) & (coarse == 0)
    bad = added & ((hsv[..., 1] < 0.6 * med_s) | (cv2.dilate(coarse, _ellipse(31)) == 0))
    mask[bad] = 0

    # Smooth the outline, drop leftovers, fill holes (lesions, rubber bands).
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _ellipse(7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _ellipse(9))
    keep = _fill_small_holes(_keep_leaves(mask))
    keep = cv2.GaussianBlur(keep, (0, 0), 1.5)
    return keep


PORTRAIT_W, PORTRAIT_H = 960, 1280


def to_portrait(bgra):
    """Make the figure portrait: leaves upright (taller than wide) on a 960x1280 canvas.

    Images that already are (portrait canvas, upright leaves) are returned untouched. Otherwise the
    leaf area is cropped, rotated 90 degrees counter-clockwise if it lies sideways, and centred on
    a transparent portrait canvas.
    """
    h, w = bgra.shape[:2]
    ys, xs = np.where(bgra[..., 3] > 0)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    upright = (y1 - y0) >= (x1 - x0)
    if upright and (w, h) == (PORTRAIT_W, PORTRAIT_H):
        return bgra
    crop = bgra[y0:y1, x0:x1]
    if not upright:
        crop = cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)
    ch, cw = crop.shape[:2]
    scale = min(1.0, PORTRAIT_W / cw, PORTRAIT_H / ch)
    if scale < 1.0:
        crop = cv2.resize(crop, (int(cw * scale), int(ch * scale)), interpolation=cv2.INTER_AREA)
        ch, cw = crop.shape[:2]
    out = np.zeros((PORTRAIT_H, PORTRAIT_W, 4), np.uint8)
    oy, ox = (PORTRAIT_H - ch) // 2, (PORTRAIT_W - cw) // 2
    out[oy:oy + ch, ox:ox + cw] = crop
    return out


def process(src, dst):
    bgr = cv2.imread(str(src))
    mask = leaf_mask(bgr)
    bgra = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
    bgra[..., 3] = mask
    bgra = to_portrait(bgra)
    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), bgra)


def _run(f):
    process(f, DST / f.relative_to(SRC).with_suffix(".png"))


def main():
    files = sorted(SRC.rglob("*.jpg"))
    with Pool(8) as pool:
        for i, _ in enumerate(pool.imap_unordered(_run, files), 1):
            if i % 25 == 0 or i == len(files):
                print(f"{i}/{len(files)}")


if __name__ == "__main__":
    main()
