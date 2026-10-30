"""How much do tiles group by slide or hospital rather than by tissue?

Compares two feature sets (by default raw vs Macenko-normalised) with
four numbers each:

  slide_specific_clusters   k-means clusters with >= 75% of their
                            tiles from one slide
  knn_same_slide            share of each tile's 10 nearest neighbours
                            that come from its own slide
  cross_slide_same_site     of the neighbours from *other* slides, the
                            share from the same hospital (TCGA tissue
                            source site)
  ..._chance                what the last two would be if tiles were
                            mixed at random

The site metric only uses hospitals with more than one slide, so it
looks past patient-level differences.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


def cluster_metrics(maps_dir):
    df = pd.read_csv(Path(maps_dir) / "clusters.csv")
    counts = pd.crosstab(df.cluster, df.slide)
    top = counts.max(axis=1) / counts.sum(axis=1)
    return int((top >= 0.75).sum()), float(top.mean())


def site_of(slide):
    # TCGA-<site>-<patient>-...
    return slide.split("-")[1]


def knn_metrics(feat_dir, k=10, n_query=5000, seed=42):
    feats, slides = [], []
    for f in sorted(Path(feat_dir).glob("*.npz")):
        d = np.load(f)
        feats.append(d["feats"].astype(np.float32))
        slides += [f.stem] * len(d["feats"])
    X = np.concatenate(feats)
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
    Z = PCA(n_components=50, random_state=seed).fit_transform(X)
    s = np.array(slides)
    site = np.array([site_of(x) for x in s])
    rng = np.random.default_rng(seed)
    q = rng.choice(len(Z), min(n_query, len(Z)), replace=False)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(Z)
    _, idx = nn.kneighbors(Z[q])
    nb = idx[:, 1:]                       # drop the tile itself
    same = (s[nb] == s[q][:, None]).mean()
    p = pd.Series(s).value_counts(normalize=True).values
    chance = (p ** 2).sum()

    n_site = pd.Series(site).value_counts()
    n_slide = pd.Series(s).value_counts()
    obs, exp = [], []
    for qi, row in zip(q, nb):
        if n_site[site[qi]] == n_slide[s[qi]]:
            continue                      # only slide from that site
        other = row[s[row] != s[qi]]
        if len(other) == 0:
            continue
        obs.append((site[other] == site[qi]).mean())
        pool = len(s) - n_slide[s[qi]]
        exp.append((n_site[site[qi]] - n_slide[s[qi]]) / pool)
    site_obs = float(np.mean(obs)) if obs else float("nan")
    site_exp = float(np.mean(exp)) if exp else float("nan")
    return float(same), float(chance), site_obs, site_exp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps", nargs=2,
                    default=["out/maps", "out/maps_macenko"])
    ap.add_argument("--features", nargs=2,
                    default=["out/features", "out/features_macenko"])
    ap.add_argument("--names", nargs=2, default=["raw", "macenko"])
    ap.add_argument("--out", default="out/comparison.csv")
    args = ap.parse_args()

    rows = []
    for name, m, f in zip(args.names, args.maps, args.features):
        n_spec, mean_top = cluster_metrics(m)
        same, chance, site_obs, site_exp = knn_metrics(f)
        rows.append({"condition": name,
                     "slide_specific_clusters": n_spec,
                     "mean_top_slide_share": round(mean_top, 2),
                     "knn_same_slide": round(same, 2),
                     "knn_chance": round(chance, 2),
                     "cross_slide_same_site": round(site_obs, 2),
                     "cross_slide_site_chance": round(site_exp, 2)})
    res = pd.DataFrame(rows)
    res.to_csv(args.out, index=False)
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
