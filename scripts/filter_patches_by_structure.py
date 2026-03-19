#!/usr/bin/env python3
import os, glob, math, random, argparse
import numpy as np
from PIL import Image
from scipy.ndimage import sobel, label

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")

def list_images(root):
    files = []
    for ext in IMG_EXTS:
        files += glob.glob(os.path.join(root, "**", f"*{ext}"), recursive=True)
        files += glob.glob(os.path.join(root, "**", f"*{ext.upper()}"), recursive=True)
    return sorted(set(files))

def load_gray(path):
    im = Image.open(path).convert("L")
    x = np.array(im, dtype=np.float32) / 255.0
    return x

def structure_score(gray, edge_quantile=0.90):
    # Sobel gradients
    gx = sobel(gray, axis=1, mode="reflect")
    gy = sobel(gray, axis=0, mode="reflect")
    mag = np.sqrt(gx*gx + gy*gy) + 1e-12

    # Use high-quantile as robust "edge strength"
    grad_q = np.quantile(mag, edge_quantile)

    # Orientation coherence (use only strong-gradient pixels)
    thr = np.quantile(mag, 0.85)
    mask = mag >= thr
    if mask.sum() < 200:  # too few edges
        return 0.0, {"grad_q": float(grad_q), "R": 0.0, "largest": 0}

    theta = np.arctan2(gy[mask], gx[mask])  # [-pi,pi]
    c = np.mean(np.cos(2.0*theta))
    s = np.mean(np.sin(2.0*theta))
    R = float(np.sqrt(c*c + s*s))  # [0,1]

    # Connectivity: threshold mag to make a binary edge map, compute largest connected component
    edge_map = mag >= thr
    cc, ncc = label(edge_map)  # 4-connectivity by default
    if ncc == 0:
        largest = 0
    else:
        # counts of each component id (skip 0)
        counts = np.bincount(cc.ravel())
        largest = int(counts[1:].max()) if len(counts) > 1 else 0

    # final score: edge strength * orientation coherence * connectivity
    score = float(grad_q * (0.5 + 0.5*R) * math.log1p(largest))

    meta = {"grad_q": float(grad_q), "R": float(R), "largest": int(largest)}
    return score, meta

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch_root", required=True, help="folder containing patch images")
    ap.add_argument("--keep_ratio", type=float, default=0.70, help="keep top ratio by score")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_list", default="filtered_list.txt")
    ap.add_argument("--out_dir", default="filtered_patches")  # optional copy
    ap.add_argument("--copy", action="store_true", help="copy kept patches into out_dir")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    paths = list_images(os.path.expanduser(args.patch_root))
    if not paths:
        raise SystemExit(f"No images found under {args.patch_root}")

    scores = []
    metas = []
    for i, p in enumerate(paths, 1):
        g = load_gray(p)
        sc, meta = structure_score(g)
        scores.append(sc)
        metas.append(meta)
        if i % 200 == 0:
            print(f"[SCORE] {i}/{len(paths)}")

    scores = np.array(scores, dtype=np.float32)
    k = int(math.ceil(len(paths) * args.keep_ratio))
    idx = np.argsort(-scores)[:k]  # top-k
    kept = [paths[i] for i in idx]

    print(f"[DONE] total={len(paths)} keep={len(kept)} keep_ratio={args.keep_ratio}")
    print(f"[STATS] score min/median/max kept = {scores[idx].min():.6f} / {np.median(scores[idx]):.6f} / {scores[idx].max():.6f}")

    # write list
    with open(args.out_list, "w") as f:
        for p in kept:
            f.write(p + "\n")
    print(f"[SAVE] list -> {args.out_list}")

    # optional copy
    if args.copy:
        out_dir = os.path.expanduser(args.out_dir)
        os.makedirs(out_dir, exist_ok=True)
        for p in kept:
            name = os.path.basename(p)
            Image.open(p).save(os.path.join(out_dir, name))
        print(f"[SAVE] copied kept patches -> {out_dir}")

if __name__ == "__main__":
    main()
