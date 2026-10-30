"""Tissue detection, tiling and tile QC for the downloaded slides.

Tiles are 224 x 224 px at 0.5 um/px (~20x). For each slide it writes
out/tiles/<slide>/tiles.csv plus two images to check the result by eye:
qc.png (kept tiles green, rejected red) and examples.png.
"""
import argparse
from pathlib import Path

import numpy as np
import openslide
import pandas as pd
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from skimage import color, filters

TILE = 224
TARGET_MPP = 0.5


def slide_mpp(slide):
    mpp = slide.properties.get(openslide.PROPERTY_NAME_MPP_X)
    if mpp:
        return float(mpp)
    power = slide.properties.get(
        openslide.PROPERTY_NAME_OBJECTIVE_POWER)
    return 0.25 if power and float(power) >= 40 else 0.5


def drop_small(mask, min_size):
    lab, n = ndi.label(mask)
    if n == 0:
        return mask
    sizes = np.bincount(lab.ravel())
    keep = sizes >= min_size
    keep[0] = False
    return keep[lab]


def tissue_mask(slide, size=2048):
    thumb = slide.get_thumbnail((size, size)).convert("RGB")
    rgb = np.asarray(thumb)
    hsv = color.rgb2hsv(rgb)
    sat, val, hue = hsv[..., 1], hsv[..., 2], hsv[..., 0]
    # glass is grey/white, tissue is coloured: Otsu on saturation
    t = filters.threshold_otsu(sat)
    mask = (sat > max(t, 0.05)) & (val > 0.15)
    # pen marks around the tissue are saturated too, cut them out
    pen = (sat > 0.35) & (hue > 0.30) & (hue < 0.70)
    mask &= ~ndi.binary_dilation(pen, iterations=5)
    mask = ndi.binary_closing(mask, iterations=3)
    min_obj = max(64, mask.size // 4000)
    mask = drop_small(mask, min_obj)        # dust and debris
    mask = ~drop_small(~mask, min_obj)      # small holes
    scale = slide.dimensions[0] / rgb.shape[1]
    return thumb, mask, scale


def tile_qc(img):
    arr = np.asarray(img.convert("RGB")).astype(np.float32)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    white = float((arr.min(axis=2) > 220).mean())
    # marker ink: clearly bluer or greener than the pink/purple H&E
    ink = ((b > r + 40) & (b > g + 20)) | ((g > r + 20) & (g > b + 20))
    pen = float(ink.mean())
    # variance of the Laplacian as a focus measure
    gray = arr.mean(axis=2)
    lap = (-4 * gray[1:-1, 1:-1] + gray[:-2, 1:-1] + gray[2:, 1:-1]
           + gray[1:-1, :-2] + gray[1:-1, 2:])
    return white, pen, float(lap.var())


def process(path, out_root, max_tiles, min_tissue, blur_thr,
            max_white, seed):
    slide = openslide.OpenSlide(str(path))
    mpp = slide_mpp(slide)
    # a few slides were scanned at 20x, so the tile size at level 0
    # has to come from the mpp rather than being fixed
    tile_l0 = int(round(TILE * TARGET_MPP / mpp))
    thumb, mask, scale = tissue_mask(slide)
    W, H = slide.dimensions

    rows = []
    for y in range(0, H - tile_l0 + 1, tile_l0):
        for x in range(0, W - tile_l0 + 1, tile_l0):
            x0, y0 = int(x / scale), int(y / scale)
            x1 = max(x0 + 1, int((x + tile_l0) / scale))
            y1 = max(y0 + 1, int((y + tile_l0) / scale))
            frac = float(mask[y0:y1, x0:x1].mean())
            if frac >= min_tissue:
                rows.append((x, y, frac))
    cand = pd.DataFrame(rows, columns=["x", "y", "tissue_frac"])
    if len(cand) > max_tiles:
        cand = cand.sample(max_tiles, random_state=seed)
    cand = cand.reset_index(drop=True)

    whites, pens, blurs = [], [], []
    for r in cand.itertuples():
        reg = slide.read_region((r.x, r.y), 0, (tile_l0, tile_l0))
        reg = reg.convert("RGB").resize((TILE, TILE), Image.BILINEAR)
        w, pn, b = tile_qc(reg)
        whites.append(w)
        pens.append(pn)
        blurs.append(b)
    cand["white_frac"] = whites
    cand["pen_frac"] = pens
    cand["blur_var"] = blurs
    # sharpness varies a lot between scanners, so the blur cut-off is
    # relative to each slide's median, with a fixed floor
    blur_cut = max(blur_thr, 0.3 * float(cand.blur_var.median()))
    cand["keep"] = ((cand.white_frac <= max_white)
                    & (cand.pen_frac <= 0.02)
                    & (cand.blur_var >= blur_cut))
    cand["tile_l0"] = tile_l0
    cand["slide"] = path.stem

    out = out_root / path.stem
    out.mkdir(parents=True, exist_ok=True)
    cand.to_csv(out / "tiles.csv", index=False)

    qc = thumb.copy()
    draw = ImageDraw.Draw(qc, "RGBA")
    s = tile_l0 / scale
    for r in cand.itertuples():
        box = [r.x / scale, r.y / scale,
               r.x / scale + s, r.y / scale + s]
        fill = (0, 170, 0, 90) if r.keep else (220, 0, 0, 110)
        draw.rectangle(box, fill=fill)
    qc.save(out / "qc.png")

    kept = cand[cand.keep]
    sample = kept.sample(min(16, len(kept)), random_state=seed)
    mosaic = Image.new("RGB", (4 * TILE, 4 * TILE), "white")
    for i, r in enumerate(sample.itertuples()):
        reg = slide.read_region((r.x, r.y), 0, (tile_l0, tile_l0))
        reg = reg.convert("RGB").resize((TILE, TILE))
        mosaic.paste(reg, ((i % 4) * TILE, (i // 4) * TILE))
    mosaic.save(out / "examples.png")
    slide.close()
    return mpp, tile_l0, len(rows), len(cand), int(cand.keep.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slides", default="data/slides")
    ap.add_argument("--out", default="out/tiles")
    ap.add_argument("--max-tiles", type=int, default=2000)
    ap.add_argument("--min-tissue", type=float, default=0.5)
    ap.add_argument("--blur", type=float, default=40.0)
    ap.add_argument("--max-white", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    slides = sorted(Path(args.slides).glob("*.svs"))
    if not slides:
        print("no .svs files in", args.slides)
        return
    out_root = Path(args.out)
    summary = []
    for p in slides:
        mpp, t0, n_tis, n_qc, n_keep = process(
            p, out_root, args.max_tiles, args.min_tissue,
            args.blur, args.max_white, args.seed)
        print(f"{p.name[:23]}  {mpp:.3f} um/px  tile {t0}px  "
              f"tissue {n_tis}  checked {n_qc}  kept {n_keep}")
        summary.append({"slide": p.stem, "mpp": mpp,
                        "tissue_tiles": n_tis, "checked": n_qc,
                        "kept": n_keep})
    pd.DataFrame(summary).to_csv(out_root / "summary.csv",
                                 index=False)


if __name__ == "__main__":
    main()
