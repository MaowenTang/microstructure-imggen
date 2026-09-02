import os
import glob
import argparse
from multiprocessing import Pool, cpu_count
from pathlib import Path
from PIL import Image

def process_image(task):
    img_path, out_dir = task
    img_path = Path(img_path)
    out_dir = Path(out_dir)

    img = Image.open(img_path).convert("L")  # 灰度
    stem = img_path.stem

    variants = {
        "orig": img,
        "hflip": img.transpose(Image.FLIP_LEFT_RIGHT),
        "vflip": img.transpose(Image.FLIP_TOP_BOTTOM),
        "rot90": img.rotate(90, expand=False),
        "rot180": img.rotate(180, expand=False),
        "rot270": img.rotate(270, expand=False),
    }

    saved = 0
    for tag, im in variants.items():
        out_name = f"{stem}_{tag}.png"
        im.save(out_dir / out_name)
        saved += 1

    return saved

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(glob.glob(str(in_dir / "*.png")))
    print(f"Input patches: {len(files)}")

    tasks = [(f, out_dir) for f in files]
    w = max(1, min(args.workers, cpu_count()))

    total = 0
    with Pool(w) as pool:
        for n in pool.imap_unordered(process_image, tasks):
            total += n

    print(f"\nDONE. Total augmented images: {total}")

if __name__ == "__main__":
    main()