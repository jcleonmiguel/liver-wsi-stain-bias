"""Unsupervised tissue maps from the tile features.

All slides are clustered together (PCA + k-means), so cluster C3 means
the same thing in every slide. Writes a map per slide, a gallery of
example tiles per cluster, and the cluster composition of each slide.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import openslide
import pandas as pd
from PIL import Image, ImageDraw
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

COLORS = plt.get_cmap("tab10").colors + plt.get_cmap("Set2").colors


def rgb(i, alpha=255):
    c = COLORS[i % len(COLORS)]
    return tuple(int(255 * v) for v in c) + (alpha,)


def load(feat_dir):
    feats, meta = [], []
    for f in sorted(Path(feat_dir).glob("*.npz")):
        d = np.load(f)
        feats.append(d["feats"].astype(np.float32))
        meta.append(pd.DataFrame({
            "slide": f.stem, "x": d["x"], "y": d["y"],
            "tile_l0": int(d["tile_l0"])}))
    return np.concatenate(feats), pd.concat(meta, ignore_index=True)


def slide_map(slide_path, tiles, k, out_path, size=1400):
    slide = openslide.OpenSlide(str(slide_path))
    thumb = slide.get_thumbnail((size, size)).convert("RGB")
    s = thumb.width / slide.dimensions[0]
    over = Image.new("RGBA", thumb.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(over)
    for r in tiles.itertuples():
        x0, y0 = r.x * s, r.y * s
        x1, y1 = x0 + r.tile_l0 * s, y0 + r.tile_l0 * s
        draw.rectangle([x0, y0, x1, y1], fill=rgb(r.cluster, 170))
    painted = Image.alpha_composite(thumb.convert("RGBA"), over)
    w, h = thumb.size
    canvas = Image.new("RGB", (2 * w + 20, h + 40), "white")
    canvas.paste(thumb, (0, 40))
    canvas.paste(painted.convert("RGB"), (w + 20, 40))
    d = ImageDraw.Draw(canvas)
    for c in range(k):
        d.rectangle([10 + 70 * c, 10, 30 + 70 * c, 30], fill=rgb(c))
        d.text((35 + 70 * c, 14), f"C{c}", fill=(0, 0, 0))
    canvas.save(out_path, quality=85)
    slide.close()


def gallery(df, slides_dir, k, out_path, n=8, px=112, seed=42):
    rows = []
    for c in range(k):
        sub = df[df.cluster == c]
        sub = sub.sample(min(n, len(sub)), random_state=seed)
        row = Image.new("RGB", (n * px + 60, px), "white")
        d = ImageDraw.Draw(row)
        d.rectangle([0, 0, 50, px], fill=rgb(c))
        d.text((12, px // 2 - 6), f"C{c}", fill=(255, 255, 255))
        for i, r in enumerate(sub.itertuples()):
            sl = openslide.OpenSlide(str(slides_dir
                                         / f"{r.slide}.svs"))
            t = sl.read_region((int(r.x), int(r.y)), 0,
                               (r.tile_l0, r.tile_l0))
            sl.close()
            t = t.convert("RGB").resize((px, px))
            row.paste(t, (60 + i * px, 0))
        rows.append(row)
    out = Image.new("RGB", (rows[0].width, k * (px + 6)), "white")
    for c, row in enumerate(rows):
        out.paste(row, (0, c * (px + 6)))
    out.save(out_path, quality=85)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slides", default="data/slides")
    ap.add_argument("--features", default="out/features")
    ap.add_argument("--out", default="out/maps")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--pcs", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    X, df = load(args.features)
    print(f"{len(df)} tiles from {df.slide.nunique()} slides")

    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
    pcs = min(args.pcs, X.shape[1], len(X) - 1)
    pca = PCA(n_components=pcs, random_state=args.seed)
    Z = pca.fit_transform(X)
    km = KMeans(n_clusters=args.k, n_init=10, random_state=args.seed)
    df["cluster"] = km.fit_predict(Z)
    df.to_csv(out / "clusters.csv", index=False)

    # A cluster made almost entirely of one slide is more likely to be
    # that slide's staining or scanner than a tissue type.
    comp = pd.crosstab(df.slide, df.cluster, normalize="index")
    counts = pd.crosstab(df.cluster, df.slide)
    summary = pd.DataFrame({
        "tiles": counts.sum(axis=1),
        "slides_with_5pct": (counts.div(counts.sum(axis=1), axis=0)
                             >= 0.05).sum(axis=1),
        "top_slide_share": counts.max(axis=1) / counts.sum(axis=1),
    }).round(2)
    summary.to_csv(out / "cluster_summary.csv")
    print(summary.to_string())

    fig, ax = plt.subplots(figsize=(9, 0.45 * len(comp) + 1.5))
    left = np.zeros(len(comp))
    names = [s[:12] for s in comp.index]
    for c in comp.columns:
        ax.barh(names, comp[c], left=left, color=COLORS[c],
                label=f"C{c}")
        left += comp[c].values
    ax.set_xlabel("Fraction of tiles")
    ax.legend(ncol=min(args.k, 10), bbox_to_anchor=(0.5, 1.02),
              loc="lower center", frameon=False)
    plt.tight_layout()
    plt.savefig(out / "composition.png", dpi=150)
    plt.close()

    slides_dir = Path(args.slides)
    for s, tiles in df.groupby("slide"):
        slide_map(slides_dir / f"{s}.svs", tiles, args.k,
                  out / f"{s[:12]}_map.jpg")
    gallery(df, slides_dir, args.k, out / "cluster_gallery.jpg",
            seed=args.seed)


if __name__ == "__main__":
    main()
