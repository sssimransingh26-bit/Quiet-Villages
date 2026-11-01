"""Cut the 2001 village-level SHRUG tables (keyed by shrid2) down to the Uttarakhand villages we use.

Run from the folder that holds the SHRUG zips and uk_pc11r_shrid_key.csv:
    python cut_by_shrid.py

Reads  : shrug-pca01-dta.zip and shrug-vd01-dta.zip (the ones WITHOUT "(2)" or "(3)" in the name)
Writes : uk_vil_<table>.csv.gz   (small, rows = Uttarakhand villages only)
Needs  : pandas
"""
from __future__ import annotations

import sys
import tempfile
import zipfile
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    sys.exit("pandas is missing. Run:  pip install pandas   then run this script again.")

CHUNK = 50_000
ID_COLS = ("shrid2", "shrid")


def uk_ids(folder: Path) -> set[str]:
    ids: set[str] = set()
    for name in ("uk_pc11r_shrid_key.csv", "uk_pc01r_shrid_key.csv"):
        p = folder / name
        if p.exists():
            ids |= set(pd.read_csv(p, usecols=["shrid2"], dtype=str).shrid2)
    if not ids:
        sys.exit("Could not find uk_pc11r_shrid_key.csv in this folder.")
    return ids


def cut(dta: Path, out: Path, ids: set[str]) -> None:
    kept = []
    total = 0
    idcol = None
    for chunk in pd.read_stata(dta, chunksize=CHUNK, convert_categoricals=False):
        if idcol is None:
            idcol = next((c for c in ID_COLS if c in chunk.columns), None)
            if idcol is None:
                print(f"    skipped (no shrid column): {dta.name}")
                return
        total += len(chunk)
        sub = chunk[chunk[idcol].astype(str).isin(ids)]
        if len(sub):
            kept.append(sub)
    if not kept:
        print(f"    no matching villages in {dta.name} ({total} rows)")
        return
    df = pd.concat(kept, ignore_index=True)
    df.to_csv(out, index=False, compression="gzip")
    print(f"    kept {len(df)} of {total} rows, {df.shape[1]} columns -> {out.name} ({out.stat().st_size/1e6:.1f} MB)")


def main() -> None:
    here = Path.cwd()
    ids = uk_ids(here)
    print(f"{len(ids)} Uttarakhand village ids loaded")
    zips = [z for z in sorted(here.glob("shrug-*01-dta*.zip")) if "(" not in z.name and "pc-keys" not in z.name]
    if not zips:
        sys.exit("No shrug-pca01-dta.zip / shrug-vd01-dta.zip found here.")
    for z in zips:
        print(f"\n{z.name}")
        with zipfile.ZipFile(z) as zf, tempfile.TemporaryDirectory() as tmp:
            for name in zf.namelist():
                low = name.lower()
                if not low.endswith(".dta") or "dist" in low or "subdist" in low:
                    continue   # we only want the village-level tables
                print("  ", name)
                zf.extract(name, tmp)
                cut(Path(tmp) / name, here / f"uk_vil_{Path(name).stem}.csv.gz", ids)
    print("\nDone. Attach the uk_vil_*.csv.gz files here.")


if __name__ == "__main__":
    main()
