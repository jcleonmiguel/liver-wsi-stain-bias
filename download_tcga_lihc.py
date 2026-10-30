"""Download a random subset of TCGA-LIHC diagnostic slides from the GDC.

One slide per patient, within a size range, so the whole set fits on
an ordinary disk. Re-running the script resumes where it stopped.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

FILES_URL = "https://api.gdc.cancer.gov/files"
DATA_URL = "https://api.gdc.cancer.gov/data/{}"
# the GDC drops long downloads every now and then, so these are retried
NET_ERRORS = (requests.exceptions.ConnectionError,
              requests.exceptions.ChunkedEncodingError,
              requests.exceptions.Timeout)


def flt(field, value):
    return {"op": "in", "content": {"field": field, "value": [value]}}


def query_gdc():
    filters = {
        "op": "and",
        "content": [
            flt("cases.project.project_id", "TCGA-LIHC"),
            flt("files.data_format", "SVS"),
            flt("files.experimental_strategy", "Diagnostic Slide"),
            flt("files.access", "open"),
        ],
    }
    fields = "file_id,file_name,file_size,md5sum,cases.submitter_id"
    params = {
        "filters": json.dumps(filters),
        "fields": fields,
        "format": "JSON",
        "size": "2000",
    }
    r = requests.get(FILES_URL, params=params, timeout=60)
    r.raise_for_status()
    hits = r.json()["data"]["hits"]
    rows = []
    for h in hits:
        cases = h.get("cases") or [{}]
        patient = cases[0].get("submitter_id", h["file_name"][:12])
        rows.append({
            "file_id": h["file_id"],
            "file_name": h["file_name"],
            "size_mb": round(h["file_size"] / 1e6, 1),
            "md5sum": h.get("md5sum"),
            "patient": patient,
        })
    return pd.DataFrame(rows)


def md5_of(path, chunk=8 * 1024 * 1024):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def fetch(url, part):
    # resume from whatever is already in the .part file
    start = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={start}-"} if start else {}
    with requests.get(url, headers=headers, stream=True,
                      timeout=(30, 120)) as r:
        if start and r.status_code != 206:
            start = 0
            part.unlink(missing_ok=True)
        r.raise_for_status()
        size = int(r.headers.get("Content-Length", 0))
        mode = "ab" if start else "wb"
        bar = tqdm(total=size + start, initial=start, unit="B",
                   unit_scale=True, desc=part.name[:23], leave=False)
        with open(part, mode) as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
                bar.update(len(chunk))
        bar.close()


def download(row, out_dir, retries=6):
    dest = out_dir / row.file_name
    expected = int(row.size_mb * 1e6)
    if dest.exists() and dest.stat().st_size >= expected * 0.99:
        if row.md5sum and md5_of(dest) == row.md5sum:
            print("  already there:", row.file_name)
            return True

    part = dest.with_suffix(dest.suffix + ".part")
    url = DATA_URL.format(row.file_id)
    for attempt in range(1, retries + 1):
        try:
            fetch(url, part)
            break
        except NET_ERRORS as e:
            wait = 15 * attempt
            print(f"  connection lost ({type(e).__name__}), "
                  f"retry {attempt}/{retries} in {wait} s")
            time.sleep(wait)
    else:
        print("  giving up for now:", row.file_name)
        return False

    if row.md5sum and md5_of(part) != row.md5sum:
        print("  MD5 mismatch, deleting:", row.file_name)
        part.unlink(missing_ok=True)
        return False
    part.replace(dest)
    print(f"  ok {row.file_name} ({row.size_mb} MB)")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--min-mb", type=float, default=150)
    ap.add_argument("--max-mb", type=float, default=700)
    ap.add_argument("--out", default="data")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.out)
    slides_dir = out / "slides"
    slides_dir.mkdir(parents=True, exist_ok=True)

    df = query_gdc()
    df.to_csv(out / "manifest_tcga_lihc_dx.csv", index=False)
    print(f"{len(df)} diagnostic slides from {df.patient.nunique()} "
          f"patients ({df.size_mb.sum() / 1000:.0f} GB in total)")

    keep = (df.size_mb >= args.min_mb) & (df.size_mb <= args.max_mb)
    pool = df[keep].sample(frac=1, random_state=args.seed)
    chosen = pool.drop_duplicates("patient").head(args.n)
    chosen = chosen.reset_index(drop=True)
    chosen.to_csv(out / "selected_slides.csv", index=False)
    print(f"downloading {len(chosen)} slides "
          f"({chosen.size_mb.sum() / 1000:.1f} GB)")

    ok = sum(download(row, slides_dir) for row in chosen.itertuples())
    print(f"{ok}/{len(chosen)} slides in {slides_dir.resolve()}")
    if ok < len(chosen):
        print("run it again to pick up the missing ones")


if __name__ == "__main__":
    main()
