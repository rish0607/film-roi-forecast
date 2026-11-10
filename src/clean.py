"""
src/clean.py
Reads a parquet file from data/raw, cleans it, builds targets and features,
and saves data/processed/titles.csv.

Run from any directory:
    python src/clean.py
    python src/clean.py path/to/file.parquet
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------- 1. find and load the parquet file ----------
project_dir = Path(__file__).resolve().parent.parent
if len(sys.argv) > 1:
    path = Path(sys.argv[1]).resolve()
else:
    raw_dir = project_dir / "data" / "raw"
    found = sorted(raw_dir.glob("*.parquet"))
    if not found:
        sys.exit(f"No .parquet file found in {raw_dir}")
    if len(found) > 1:
        sys.exit(f"Multiple .parquet files found in {raw_dir}; pass one explicitly.")
    path = found[0]

print(f"Loading: {path}")
df = pd.read_parquet(path)
print(f"Raw shape: {df.shape}")
print("Columns:", df.columns.tolist())


# ---------- helpers: handle JSON strings OR already-parsed lists ----------
def parse(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []
    if isinstance(x, str):
        try:
            return json.loads(x)
        except json.JSONDecodeError:
            return []
    return list(x)


def names(x, key="name", limit=None):
    items = parse(x)
    out = [i[key] for i in items if isinstance(i, dict) and key in i]
    return out[:limit] if limit else out


# ---------- 2. basic cleaning ----------
for col in ["budget", "revenue", "runtime"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

if "status" in df.columns:
    df = df[df["status"] == "Released"]
df = df[(df["budget"] >= 1e6) & (df["revenue"] >= 1e6) & (df["runtime"] > 0)].copy()

df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
df = df.dropna(subset=["release_date"])
df["year"] = df["release_date"].dt.year
df["quarter"] = df["release_date"].dt.quarter
print(f"After cleaning: {len(df)} rows")

# ---------- 3. targets ----------
df["roi"] = (df["revenue"] - df["budget"]) / df["budget"]
df["log_rev"] = np.log(df["revenue"])
df["tier"] = pd.qcut(df["revenue"], [0, 0.4, 0.75, 0.95, 1], labels=["Low", "Medium", "High", "Breakout"])

# ---------- 4. features ----------
df["genre_list"] = df["genres"].apply(names)
df["genre"] = df["genre_list"].apply(lambda g: g[0] if g else "Unknown")
df["log_budget"] = np.log(df["budget"])

overview = df["overview"].fillna("") if "overview" in df.columns else ""
tagline = df["tagline"].fillna("") if "tagline" in df.columns else ""
df["text"] = (overview + " " + tagline).str.strip()

# star-power proxy: how many films in the data feature the top-3 billed actors
if "cast" in df.columns:
    top_cast = df["cast"].apply(lambda s: names(s, limit=3))
    counts = pd.Series([n for lst in top_cast for n in lst]).value_counts()
    df["star_power"] = top_cast.apply(
        lambda lst: float(np.mean([np.log1p(counts[n]) for n in lst])) if lst else 0.0
    )
    print("star_power computed from the cast column")
else:
    df["star_power"] = 0.0
    print("WARNING: no 'cast' column in this file, so star_power is set to 0 for now.")

# ---------- 5. save ----------
df["genre_list"] = df["genre_list"].apply(lambda g: "|".join(g))  # CSV-friendly
keep = [c for c in ["title", "year", "quarter", "genre", "genre_list", "runtime", "budget", "revenue",
                    "roi", "log_rev", "log_budget", "star_power", "text", "tier"] if c in df.columns]
processed_dir = project_dir / "data" / "processed"
processed_dir.mkdir(parents=True, exist_ok=True)
output_path = processed_dir / "titles.csv"
df[keep].to_csv(output_path, index=False)

print(f"\nSaved {output_path}  ({len(df)} rows)")
print(df[["budget", "revenue", "roi"]].describe().round(2))
print("\nTier counts:\n", df["tier"].value_counts())