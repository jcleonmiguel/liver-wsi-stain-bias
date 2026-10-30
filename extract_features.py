"""DINOv2 features for every kept tile, optionally after Macenko
stain normalisation (--macenko).

Writes one .npz per slide with the 768-d CLS embeddings and the tile
coordinates, to out/features or out/features_macenko.
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

# Reference stain matrix and max concentrations commonly used with
# Macenko et al. (2009); every slide is mapped onto these.
IO = 240.0
HE_REF = np.array([[0.5626, 0.2159],
                   [0.7201, 0.8012],
                   [0.4062, 0.5581]])
MAXC_REF = np.array([1.9705, 1.0308])


def to_od(rgb):
    rgb = rgb.reshape(-1, 3).astype(np.float64)
    return -np.log((rgb + 1) / IO)


def fit_macenko(pixels, alpha=1.0, beta=0.15, max_px=300000):
    od = to_od(pixels)
    od = od[(od > beta).all(axis=1)]      # ignore near-white pixels
    if len(od) < 1000:
        return None
    if len(od) > max_px:
        rng = np.random.default_rng(0)
        od = od[rng.choice(len(od), max_px, replace=False)]
    # the two stains span the plane of the two largest eigenvectors;
    # their directions are the extreme angles within that plane
    _, vecs = np.linalg.eigh(np.cov(od.T))
    v = vecs[:, 1:3]
    proj = od @ v
    phi = np.arctan2(proj[:, 1], proj[:, 0])
    lo = np.percentile(phi, alpha)
    hi = np.percentile(phi, 100 - alpha)
    a = v @ np.array([np.cos(lo), np.sin(lo)])
    b = v @ np.array([np.cos(hi), np.sin(hi)])
    he = np.stack([a, b], axis=1)
    he *= np.sign(he.sum(axis=0))
    if he[0, 0] < he[0, 1]:               # hematoxylin first
        he = he[:, ::-1]
    he /= np.linalg.norm(he, axis=0)
    conc = np.linalg.pinv(he) @ od.T
    maxc = np.percentile(conc, 99, axis=1)
    return he, maxc


def normalize(rgb, he, maxc):
    h, w, _ = rgb.shape
    conc = np.linalg.pinv(he) @ to_od(rgb).T
    conc *= (MAXC_REF / maxc)[:, None]
    out = IO * np.exp(-HE_REF @ conc)
    out = np.clip(out.T.reshape(h, w, 3), 0, 255)
    return out.astype(np.uint8)


def read_tile(slide, x, y, size):
    reg = slide.read_region((int(x), int(y)), 0, (size, size))
    reg = reg.convert("RGB").resize((224, 224), Image.BILINEAR)
    return np.asarray(reg)


def slide_stain(slide, t, size, n=64, seed=0):
    # one stain estimate per slide (not per tile), so that all tiles of
    # a slide get the same correction
    sub = t.sample(min(n, len(t)), random_state=seed)
    px = np.concatenate([read_tile(slide, r.x, r.y, size)
                         for r in sub.itertuples()])
    return fit_macenko(px)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slides", default="data/slides")
    ap.add_argument("--tiles", default="out/tiles")
    ap.add_argument("--out", default=None)
    ap.add_argument("--model", default="dinov2_vitb14")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--macenko", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    if args.out is None:
        args.out = ("out/features_macenko" if args.macenko
                    else "out/features")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.hub.load("facebookresearch/dinov2", args.model)
    model = model.to(device).eval()
    use_amp = device == "cuda"

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    slide_dirs = [d for d in sorted(Path(args.tiles).iterdir())
                  if (d / "tiles.csv").exists()]
    examples = []

    for d in slide_dirs:
        out = out_dir / f"{d.name}.npz"
        if out.exists() and not args.overwrite:
            continue
        t = pd.read_csv(d / "tiles.csv")
        t = t[t.keep].reset_index(drop=True)
        slide = openslide.OpenSlide(str(Path(args.slides)
                                        / f"{d.name}.svs"))
        size = int(t.tile_l0.iloc[0])
        stain = slide_stain(slide, t, size) if args.macenko else None
        if args.macenko and stain is None:
            print("stain fit failed, using raw tiles:", d.name[:23])
        feats = []
        t0 = time.time()
        for i in range(0, len(t), args.batch):
            b = t.iloc[i:i + args.batch]
            imgs = [read_tile(slide, r.x, r.y, size)
                    for r in b.itertuples()]
            if stain is not None:
                if i == 0:
                    examples.append((imgs[0], normalize(imgs[0],
                                                        *stain)))
                imgs = [normalize(im, *stain) for im in imgs]
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

    if examples:  # one tile per slide: raw on top, normalised below
        n = len(examples)
        grid = Image.new("RGB", (n * 224, 2 * 224 + 8), "white")
        for i, (raw, norm) in enumerate(examples):
            grid.paste(Image.fromarray(raw), (i * 224, 0))
            grid.paste(Image.fromarray(norm), (i * 224, 232))
        grid.save(out_dir / "stain_examples.jpg", quality=85)


if __name__ == "__main__":
    main()
