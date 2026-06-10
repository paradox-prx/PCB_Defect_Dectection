"""
Generate a modality-randomized copy of a YOLO-format dataset.

All transforms are PHOTOMETRIC (grayscale, Otsu / adaptive binarization,
polarity inversion, Fourier low-frequency amplitude swap between SOURCE
images). Geometry is untouched -> label .txt files are copied verbatim.
DeepPCB is never used -> strict zero-shot domain generalization.

Usage:
    python make_modrand_dataset.py \
        --images path/to/train/images \
        --labels path/to/train/labels \
        --out    path/to/train_modrand \
        --k 3

Output: out/images and out/labels containing every original image PLUS
k randomized variants per image ({stem}__mr{i}.jpg). Point a new data.yaml
at out/ for the +ModRand training rows.
"""

import argparse
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

random.seed(0)
np.random.seed(0)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


# ----------------------------------------------------------- transforms
def gray3(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


def otsu_bin(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.cvtColor(b, cv2.COLOR_GRAY2BGR)


def adaptive_bin(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    b = cv2.adaptiveThreshold(
        g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 5
    )
    return cv2.cvtColor(b, cv2.COLOR_GRAY2BGR)


def invert(img):
    return 255 - img


def fda_amp_swap(img, ref, beta=0.03):
    """Swap the low-frequency amplitude spectrum of img with that of ref
    (another SOURCE-domain image). Phase (= structure) is preserved."""
    ref = cv2.resize(ref, (img.shape[1], img.shape[0]))
    img_f = np.fft.fft2(img.astype(np.float32), axes=(0, 1))
    ref_f = np.fft.fft2(ref.astype(np.float32), axes=(0, 1))
    amp = np.fft.fftshift(np.abs(img_f), axes=(0, 1))
    amp_ref = np.fft.fftshift(np.abs(ref_f), axes=(0, 1))
    h, w = img.shape[:2]
    b = max(1, int(min(h, w) * beta))
    cy, cx = h // 2, w // 2
    amp[cy - b:cy + b, cx - b:cx + b] = amp_ref[cy - b:cy + b, cx - b:cx + b]
    amp = np.fft.ifftshift(amp, axes=(0, 1))
    out = np.real(np.fft.ifft2(amp * np.exp(1j * np.angle(img_f)), axes=(0, 1)))
    return np.clip(out, 0, 255).astype(np.uint8)


def random_variant(img, pool):
    """Sample one randomized photometric variant."""
    choice = random.choice(["gray", "otsu", "adapt", "otsu_inv", "adapt_inv", "fda", "fda_bin"])
    if choice == "gray":
        return gray3(img)
    if choice == "otsu":
        return otsu_bin(img)
    if choice == "adapt":
        return adaptive_bin(img)
    if choice == "otsu_inv":
        return invert(otsu_bin(img))
    if choice == "adapt_inv":
        return invert(adaptive_bin(img))
    ref = cv2.imread(str(random.choice(pool)))
    out = fda_amp_swap(img, ref, beta=random.uniform(0.01, 0.05))
    if choice == "fda_bin":
        out = otsu_bin(out)
    return out


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=3, help="variants per image")
    args = ap.parse_args()

    src_img, src_lbl = Path(args.images), Path(args.labels)
    out_img = Path(args.out) / "images"
    out_lbl = Path(args.out) / "labels"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    pool = sorted(p for p in src_img.iterdir() if p.suffix.lower() in IMG_EXTS)
    print(f"{len(pool)} source images -> {len(pool) * (args.k + 1)} total")

    for p in pool:
        lbl = src_lbl / (p.stem + ".txt")
        img = cv2.imread(str(p))
        if img is None:
            print(f"skip unreadable: {p.name}")
            continue
        # original
        shutil.copy2(p, out_img / p.name)
        if lbl.exists():
            shutil.copy2(lbl, out_lbl / lbl.name)
        # variants (photometric -> identical labels)
        for i in range(args.k):
            name = f"{p.stem}__mr{i}"
            cv2.imwrite(str(out_img / f"{name}.jpg"), random_variant(img, pool))
            if lbl.exists():
                shutil.copy2(lbl, out_lbl / f"{name}.txt")

    print("done:", Path(args.out).resolve())


if __name__ == "__main__":
    main()
