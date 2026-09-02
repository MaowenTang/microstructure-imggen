import os
import glob
import argparse
import hashlib
from pathlib import Path
from multiprocessing import Pool, cpu_count
from PIL import Image

IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")

def stable_int_seed(s: str) -> int:
    h = hashlib.sha256(s.encode("utf-8")).hexdigest()
    return int(h[:8], 16)

def worker(task):
    img_path, out_dir, ps, k, base_seed, grid, border, min_dist, max_tries = task
    img_path = Path(img_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    if w < ps + 2*border or h < ps + 2*border:
        return (img_path.name, 0, f"SKIP small {w}x{h} (need >= {ps+2*border})")

    # valid top-left range
    x0_min, y0_min = border, border
    x0_max = w - ps - border
    y0_max = h - ps - border

    # reproducible per-image RNG (LCG)
    state = (base_seed + stable_int_seed(str(img_path))) & 0xFFFFFFFF
    def rand_u32():
        nonlocal state
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        return state

    def rand_int(a, b):
        # inclusive
        if a == b:
            return a
        return a + (rand_u32() % (b - a + 1))

    # stratified grid on the VALID top-left coordinate space
    valid_w = (x0_max - x0_min + 1)
    valid_h = (y0_max - y0_min + 1)
    cell_w = max(1, valid_w // grid)
    cell_h = max(1, valid_h // grid)

    # distribute k across grid^2 cells
    n_cells = grid * grid
    base = k // n_cells
    rem  = k % n_cells

    # shuffle cell order deterministically
    cell_ids = list(range(n_cells))
    for i in range(n_cells - 1, 0, -1):
        j = rand_u32() % (i + 1)
        cell_ids[i], cell_ids[j] = cell_ids[j], cell_ids[i]

    # pick which cells get +1
    extra_cells = set(cell_ids[:rem])

    chosen = []  # store (x,y) top-left
    stem = img_path.stem
    saved = 0

    def ok_far_enough(x, y):
        # min_dist on top-left coords; cheap and works well enough
        for (xx, yy) in chosen:
            dx = x - xx
            dy = y - yy
            if dx*dx + dy*dy < (min_dist * min_dist):
                return False
        return True

    for cid in range(n_cells):
        r = cid // grid
        c = cid % grid

        n_pick = base + (1 if cid in extra_cells else 0)
        if n_pick == 0:
            continue

        # cell bounds in valid top-left coord
        cx_min = x0_min + c * cell_w
        cy_min = y0_min + r * cell_h
        cx_max = x0_min + min(valid_w - 1, (c + 1) * cell_w - 1)
        cy_max = y0_min + min(valid_h - 1, (r + 1) * cell_h - 1)

        # make sure cell range is legal
        cx_min = max(cx_min, x0_min); cy_min = max(cy_min, y0_min)
        cx_max = min(cx_max, x0_max); cy_max = min(cy_max, y0_max)
        if cx_min > cx_max or cy_min > cy_max:
            continue

        for _ in range(n_pick):
            found = False
            x = y = None

            for _try in range(max_tries):
                x_try = rand_int(cx_min, cx_max)
                y_try = rand_int(cy_min, cy_max)
                if ok_far_enough(x_try, y_try):
                    x, y = x_try, y_try
                    found = True
                    break

            # fallback: if cannot satisfy min_dist, relax once (don’t stall)
            if not found:
                x = rand_int(cx_min, cx_max)
                y = rand_int(cy_min, cy_max)

            patch = img.crop((x, y, x + ps, y + ps))
            out_name = f"{stem}_p{saved:04d}_x{x:04d}_y{y:04d}.png"
            patch.save(out_dir / out_name)
            chosen.append((x, y))
            saved += 1

    return (img_path.name, saved, "OK")

def list_images(in_dir):
    files = []
    for ext in IMG_EXTS:
        files += glob.glob(os.path.join(in_dir, f"*{ext}"))
        files += glob.glob(os.path.join(in_dir, f"*{ext.upper()}"))
    return sorted(set(files))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--patch_size", type=int, default=512)
    ap.add_argument("--patches_per_image", type=int, default=500)
    ap.add_argument("--grid", type=int, default=4, help="stratified grid (4 => 4x4)")
    ap.add_argument("--border", type=int, default=10)
    ap.add_argument("--min_dist", type=int, default=0, help="0 => auto patch_size//3")
    ap.add_argument("--max_tries", type=int, default=80)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260224)
    args = ap.parse_args()

    imgs = list_images(args.in_dir)
    if not imgs:
        raise SystemExit(f"No images found in {args.in_dir} with extensions {IMG_EXTS}")

    min_dist = args.min_dist if args.min_dist > 0 else max(1, args.patch_size // 3)

    tasks = [
        (p, args.out_dir, args.patch_size, args.patches_per_image, args.seed,
         args.grid, args.border, min_dist, args.max_tries)
        for p in imgs
    ]

    w = max(1, min(args.workers, cpu_count()))
    print(f"Images: {len(imgs)} | patch={args.patch_size} | per_img={args.patches_per_image} | workers={w}")
    print(f"grid={args.grid}x{args.grid} | border={args.border} | min_dist={min_dist} | max_tries={args.max_tries}")
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