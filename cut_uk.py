"""Cut big SHRUG .dta files (or .zip files containing .dta) down to Uttarakhand (state id 05).

Usage (from the folder that holds your downloads):
    python cut_uk.py

It reads every .zip and .dta in the current folder, keeps only rows whose state id is 05,
and writes uk_<name>.csv next to the original. Originals are never changed.
Needs: pandas (pip install pandas)
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

STATE = "05"
CHUNK = 50_000


def state_col(columns):
    hits = [c for c in columns if c.lower().endswith("state_id") or c.lower() == "state_id"]
    return hits


def is_uk(series):
    s = series.astype(str).str.strip()
    return (s == STATE) | (s == "5") | (s == "5.0") | (s == "05.0")


def cut_dta(path: Path, out: Path) -> int:
    kept = []
    total = 0
    reader = pd.read_stata(path, chunksize=CHUNK, convert_categoricals=False)
    cols_checked = False
    cols = []
    for chunk in reader:
        if not cols_checked:
            cols = state_col(chunk.columns)
            cols_checked = True
            if not cols:
                print(f"  !! no state id column found in {path.name}.")
                print("  Column names are:", ", ".join(map(str, chunk.columns[:60])))
                return -1
            print("  using state column:", cols[0])
        total += len(chunk)
        mask = is_uk(chunk[cols[0]])
        if mask.any():
            kept.append(chunk[mask])
    if not kept:
        print(f"  no Uttarakhand rows found in {path.name} ({total} rows read)")
        return 0
    df = pd.concat(kept, ignore_index=True)
    df.to_csv(out, index=False)
    print(f"  kept {len(df)} of {total} rows -> {out.name}  ({out.stat().st_size/1e6:.1f} MB)")
    return len(df)


def main() -> None:
    here = Path.cwd()
    jobs = []
    for z in sorted(here.glob("*.zip")):
        jobs.append(("zip", z))
    for d in sorted(here.glob("*.dta")):
        jobs.append(("dta", d))
    if not jobs:
        sys.exit("No .zip or .dta files in this folder. cd into the folder with your downloads first.")
    for kind, p in jobs:
        print(f"\n{p.name}")
        if kind == "dta":
            cut_dta(p, here / f"uk_{p.stem}.csv")
            continue
        with zipfile.ZipFile(p) as zf, tempfile.TemporaryDirectory() as tmp:
            for name in zf.namelist():
                if not name.lower().endswith(".dta"):
                    continue
                print(" ", name)
                zf.extract(name, tmp)
                cut_dta(Path(tmp) / name, here / f"uk_{Path(name).stem}.csv")
    print("\nDone. Send me the uk_*.csv files (they should be small).")


if __name__ == "__main__":
    main()
