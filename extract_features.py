"""DINOv2 features for every kept tile.

Writes one .npz per slide to out/features with the 768-d CLS
embeddings and the tile coordinates.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import openslide
import pandas as pd
import torch
from PIL import Image

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def read_tile(slide, x, y, size):
    reg = slide.read_region((int(x), int(y)), 0, (size, size))
    reg = reg.convert("RGB").resize((224, 224), Image.BILINEAR)
    return np.asarray(reg)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slides", default="data/slides")
    ap.add_argument("--tiles", default="out/tiles")
    ap.add_argument("--out", default="out/features")
    ap.add_argument("--model", default="dinov2_vitb14")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.hub.load("facebookresearch/dinov2", args.model)
    model = model.to(device).eval()
    use_amp = device == "cuda"

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    slide_dirs = [d for d in sorted(Path(args.tiles).iterdir())
                  if (d / "tiles.csv").exists()]

    for d in slide_dirs:
        out = out_dir / f"{d.name}.npz"
        if out.exists() and not args.overwrite:
            continue
        t = pd.read_csv(d / "tiles.csv")
        t = t[t.keep].reset_index(drop=True)
        slide = openslide.OpenSlide(str(Path(args.slides)
                                        / f"{d.name}.svs"))
        size = int(t.tile_l0.iloc[0])
        feats = []
        t0 = time.time()
        # reading from the .svs is the bottleneck here, not the GPU
        for i in range(0, len(t), args.batch):
            b = t.iloc[i:i + args.batch]
            imgs = [read_tile(slide, r.x, r.y, size)
                    for r in b.itertuples()]
            x = torch.from_numpy(np.stack(imgs))
            x = x.permute(0, 3, 1, 2).float().div(255)
            x = ((x - MEAN) / STD).to(device)
            with torch.autocast(device_type=device,
                                dtype=torch.float16,
                                enabled=use_amp):
                f = model(x)
            feats.append(f.float().cpu().numpy())
        slide.close()
        F = np.concatenate(feats).astype(np.float16)
        np.savez(out, feats=F, x=t.x.values, y=t.y.values,
                 tile_l0=size)
        rate = len(t) / (time.time() - t0)
        print(f"{d.name[:23]}  {len(t)} tiles  {rate:.0f} tiles/s")


if __name__ == "__main__":
    main()
