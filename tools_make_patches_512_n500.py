import os
import glob
import argparse
import hashlib
from pathlib import Path
from multiprocessing import Pool, cpu_count
from PIL import Image

def stable_int_seed(s: str) -> int:
    h = hashlib.sha256(s.encode("utf-8")).hexdigest()
    return int(h[:8], 16)

def worker(task):
    img_path, out_dir, ps, k, base_seed = task
    img_path = Path(img_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    if w < ps or h < ps:
        return (img_path.name, 0, f"SKIP small {w}x{h}")

    max_x = w - ps
    max_y = h - ps

    # reproducible per-image RNG (LCG)
    state = (base_seed + stable_int_seed(str(img_path))) & 0xFFFFFFFF
    def rand_u32():
        nonlocal state
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        return state

    stem = img_path.stem
    saved = 0
    for i in range(k):
        x = 0 if max_x == 0 else (rand_u32() % (max_x + 1))
        y = 0 if max_y == 0 else (rand_u32() % (max_y + 1))
        patch = img.crop((x, y, x + ps, y + ps))
        out_name = f"{stem}_p{i:04d}_x{x:04d}_y{y:04d}.png"
        patch.save(out_dir / out_name)
        saved += 1
    return (img_path.name, saved, "OK")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--patch_size", type=int, default=512)
    ap.add_argument("--patches_per_image", type=int, default=500)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260224)
    args = ap.parse_args()

    imgs = sorted(glob.glob(os.path.join(args.in_dir, "*.png")))
    if not imgs:
        raise SystemExit(f"No PNGs found in {args.in_dir}")

    tasks = [(p, args.out_dir, args.patch_size, args.patches_per_image, args.seed) for p in imgs]
    w = max(1, min(args.workers, cpu_count()))
    print(f"Images: {len(imgs)} | patch={args.patch_size} | per_img={args.patches_per_image} | workers={w}")
    print(f"IN : {args.in_dir}")
    print(f"OUT: {args.out_dir}")

    total = 0
    with Pool(w) as pool:
        for name, saved, status in pool.imap_unordered(worker, tasks):
            total += saved
            print(f"{name}: {status}, saved={saved}")

    print(f"\nDONE. Total patches saved: {total}")

if __name__ == "__main__":
    main()
