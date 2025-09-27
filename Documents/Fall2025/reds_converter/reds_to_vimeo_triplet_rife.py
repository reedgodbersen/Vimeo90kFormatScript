#!/usr/bin/env python3
"""
REDS → Vimeo-90K Triplet (RIFE-ready)

This writes the exact folder & list structure RIFE expects:

<out_root>/
  vimeo_triplet/
    sequences/
      <clip_id>/<center_idx>/im1.png
                             /im2.png
                             /im3.png
    tri_trainlist.txt
    tri_testlist.txt
"""

import argparse
import re
from pathlib import Path
from PIL import Image
from tqdm import tqdm

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--reds_root", type=str, required=True,
                   help="Path to REDS root (contains train/ test/ ...)")
    p.add_argument("--subsets", type=str, nargs="+", default=["train"],
                   help="Which subsets to convert (train, test)")
    p.add_argument("--out_root", type=str, required=True,
                   help="Where to write vimeo_triplet/")
    p.add_argument("--stride", type=int, default=1,
                   help="Step size between triplets")
    p.add_argument("--start", type=int, default=1,
                   help="First valid center index (>=1 for triplet)")
    p.add_argument("--end_offset", type=int, default=1,
                   help="Exclude last end_offset frames from being centers")
    p.add_argument("--mode", choices=["resize", "center_crop", "none"], default="none",
                   help="Resize/crop to fixed size, or leave as-is")
    p.add_argument("--size", type=int, nargs=2, metavar=("W","H"), default=[1280,720],
                   help="Target size W H (used if resize/crop)")
    p.add_argument("--format", choices=["png","jpg"], default="png",
                   help="Output image format")
    p.add_argument("--jpg_quality", type=int, default=90,
                   help="JPEG quality if format=jpg")
    return p.parse_args()

def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', str(s))]

def load_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")

def transform(img: Image.Image, mode: str, target_wh):
    if mode == "none":
        return img
    W, H = target_wh
    if mode == "resize":
        return img.resize((W,H), Image.BICUBIC)
    if mode == "center_crop":
        w,h = img.size
        left = max(0,(w-W)//2)
        top  = max(0,(h-H)//2)
        right = left+W
        bottom = top+H
        if right>w or bottom>h:
            scale = max(W/w, H/h)
            new_w, new_h = int(round(w*scale)), int(round(h*scale))
            img = img.resize((new_w,new_h), Image.BICUBIC)
            w,h = img.size
            left = max(0,(w-W)//2)
            top  = max(0,(h-H)//2)
            right = left+W
            bottom = top+H
        return img.crop((left,top,right,bottom))
    raise ValueError

def main():
    args = parse_args()
    reds_root = Path(args.reds_root)
    vimeo_root = Path(args.out_root) / "vimeo_triplet"
    seq_root = vimeo_root / "sequences"
    seq_root.mkdir(parents=True, exist_ok=True)

    train_entries, test_entries = [], []

    for subset in args.subsets:
        src = reds_root / subset
        if not src.exists():
            print(f"[WARN] Subset not found: {src}")
            continue

        clip_ids = sorted([d.name for d in src.iterdir() if d.is_dir()], key=natural_key)
        print(f"[INFO] Subset {subset}: {len(clip_ids)} clips")

        for clip_id in tqdm(clip_ids, desc=f"Processing {subset}"):
            frames = sorted(list((src/clip_id).glob("*.png")), key=lambda p: natural_key(p.name))
            if len(frames) < 3: 
                continue

            for c in range(args.start, len(frames)-args.end_offset, args.stride):
                i1,i2,i3 = c-1,c,c+1
                f1,f2,f3 = frames[i1],frames[i2],frames[i3]
                center_stem = f2.stem

                out_triplet = seq_root/clip_id/center_stem
                out_triplet.mkdir(parents=True, exist_ok=True)

                imgs = [load_image(f) for f in (f1,f2,f3)]
                imgs = [transform(im,args.mode,tuple(args.size)) for im in imgs]

                if args.format=="png":
                    for j,im in enumerate(imgs,1):
                        im.save(out_triplet/f"im{j}.png", compress_level=0)
                else:
                    for j,im in enumerate(imgs,1):
                        im.save(out_triplet/f"im{j}.jpg", quality=args.jpg_quality, optimize=True)

                rel_entry = f"{clip_id}/{center_stem}"
                if subset=="train":
                    train_entries.append(rel_entry)
                else:
                    test_entries.append(rel_entry)

    if train_entries:
        with open(vimeo_root/"tri_trainlist.txt","w") as f:
            for r in sorted(train_entries, key=natural_key):
                f.write(r+"\n")
    if test_entries:
        with open(vimeo_root/"tri_testlist.txt","w") as f:
            for r in sorted(test_entries, key=natural_key):
                f.write(r+"\n")

    print(f"[DONE] Wrote to {vimeo_root}")

if __name__=="__main__":
    main()

