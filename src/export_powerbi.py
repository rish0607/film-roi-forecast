"""
src/export_powerbi.py
Exports clean, model-scored tables for Power BI into the powerbi/ folder.

Run from the project root (after clean.py and train.py):
    python src/export_powerbi.py

Tables created:
  films.csv               one row per film, with out-of-fold model predictions
  film_genres.csv         bridge table (film_id, genre) for multi-genre analysis
  genre_year_summary.csv  films, revenue and median ROI by genre and year
  scenario_grid.csv       model predictions across genre x budget x quarter x star power
  feature_importance.csv  what drives the model
  model_metrics.csv       accuracy numbers from training
Re-run this script whenever you retrain the model, then click Refresh in Power BI.
"""
import itertools
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import KFold, cross_val_predict

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed" / "titles.csv"
FEATURES = ["log_budget", "runtime", "quarter", "year", "star_power", "genre_list", "text"]
OUT = ROOT / "powerbi"
OUT.mkdir(parents=True, exist_ok=True)


def save(frame, name):
    frame.to_csv(OUT / name, index=False, encoding="utf-8-sig")   # utf-8-sig keeps accents intact in Excel/Power BI
    print(f"  {name}: {len(frame)} rows")


# ---------------- load ----------------
df = pd.read_csv(DATA)
df["text"] = df["text"].fillna("")
df["genre"] = df["genre"].fillna("Unknown")
if "genre_list" not in df.columns:
    df["genre_list"] = df["genre"]
df["genre_list"] = df["genre_list"].fillna("").replace("", "Unknown")
df = df.dropna(subset=["log_budget", "runtime", "quarter", "year", "star_power", "log_rev", "revenue"])
df = df.reset_index(drop=True)
df["film_id"] = np.arange(1, len(df) + 1)

model = joblib.load(ROOT / "models" / "rf_model.joblib")
with (ROOT / "models" / "tier_cutoffs.json").open(encoding="utf-8") as f:
    cutoffs = json.load(f)


def tier_of(rev_usd):
    if rev_usd <= cutoffs["low_max"]:
        return "Low"
    if rev_usd <= cutoffs["medium_max"]:
        return "Medium"
    if rev_usd <= cutoffs["high_max"]:
        return "High"
    return "Breakout"


# ---------------- films (honest out-of-fold predictions) ----------------
print("Scoring films with 5-fold out-of-fold predictions (this takes a minute)...")
oof_log = cross_val_predict(clone(model), df[FEATURES], df["log_rev"], cv=KFold(5, shuffle=True, random_state=42))
pred_rev = np.exp(oof_log)

films = pd.DataFrame({
    "film_id": df["film_id"],
    "title": df["title"] if "title" in df.columns else df["film_id"].astype(str),
    "release_year": df["year"].astype(int),
    "release_quarter": df["quarter"].astype(int),
    "primary_genre": df["genre"],
    "all_genres": df["genre_list"].str.replace("|", ", ", regex=False),
    "runtime_min": df["runtime"].astype(int),
    "budget_musd": (df["budget"] / 1e6).round(2),
    "revenue_musd": (df["revenue"] / 1e6).round(2),
    "roi_pct": (df["roi"] * 100).round(1),
    "star_power_percentile": (df["star_power"].rank(pct=True) * 100).round(0),
    "actual_revenue_tier": df["revenue"].apply(tier_of),
    "predicted_revenue_musd": (pred_rev / 1e6).round(2),
})
films["predicted_roi_pct"] = ((films["predicted_revenue_musd"] / films["budget_musd"] - 1) * 100).round(1)
films["prediction_error_musd"] = (films["predicted_revenue_musd"] - films["revenue_musd"]).round(2)
films["abs_pct_error"] = (films["prediction_error_musd"].abs() / films["revenue_musd"] * 100).round(1)
films["profitable"] = np.where(films["roi_pct"] > 0, "Profitable", "Loss")

# ---------------- bridge + genre/year summary ----------------
bridge = (df[["film_id", "genre_list"]].assign(genre=lambda d: d["genre_list"].str.split("|"))
          .explode("genre")[["film_id", "genre"]])
bridge = bridge[bridge["genre"].str.len() > 0]

gy = bridge.merge(films[["film_id", "release_year", "budget_musd", "revenue_musd", "roi_pct"]], on="film_id")
films_per_year = films.groupby("release_year")["film_id"].nunique().rename("films_that_year")
genre_year = (gy.groupby(["release_year", "genre"])
              .agg(films=("film_id", "nunique"), total_revenue_musd=("revenue_musd", "sum"),
                   avg_budget_musd=("budget_musd", "mean"), median_roi_pct=("roi_pct", "median"))
              .reset_index().merge(films_per_year, on="release_year"))
genre_year["share_of_releases_pct"] = (genre_year["films"] / genre_year["films_that_year"] * 100).round(1)
genre_year = genre_year.drop(columns="films_that_year").round(2)

# ---------------- scenario grid (lets Power BI act as a what-if explorer) ----------------
latest_year = int(df["year"].max())
genre_counts = bridge["genre"].value_counts()
grid_genres = genre_counts[genre_counts >= 20].index.tolist() or genre_counts.index.tolist()
b_lo, b_hi = df["budget"].min() / 1e6, df["budget"].max() / 1e6
budgets = [b for b in [5, 10, 20, 30, 40, 60, 80, 100, 130, 160, 200, 250, 300] if b_lo <= b <= b_hi]
star_pcts = [25, 50, 75, 90]
runtime_med = int(df["runtime"].median())

combos = pd.DataFrame(list(itertools.product(grid_genres, budgets, [1, 2, 3, 4], star_pcts)),
                      columns=["genre", "budget_musd", "release_quarter", "star_power_percentile"])
star_lookup = {p: float(np.quantile(df["star_power"], p / 100)) for p in star_pcts}
X = pd.DataFrame({
    "log_budget": np.log(combos["budget_musd"] * 1e6), "runtime": runtime_med,
    "quarter": combos["release_quarter"], "year": latest_year,
    "star_power": combos["star_power_percentile"].map(star_lookup),
    "genre_list": combos["genre"], "text": "",
})
combos["predicted_revenue_musd"] = (np.exp(model.predict(X)) / 1e6).round(2)
combos["predicted_profit_musd"] = (combos["predicted_revenue_musd"] - combos["budget_musd"]).round(2)
combos["predicted_roi_pct"] = ((combos["predicted_revenue_musd"] / combos["budget_musd"] - 1) * 100).round(1)
combos["predicted_tier"] = (combos["predicted_revenue_musd"] * 1e6).apply(tier_of)

# ---------------- feature importance + metrics ----------------
nice = {"log_budget": "Budget", "runtime": "Runtime", "year": "Release year",
        "quarter": "Release quarter", "star_power": "Star power"}
fi_path = ROOT / "models" / "feature_importances.csv"
if fi_path.exists():
    fi = pd.read_csv(fi_path)
    fi.columns = ["raw_feature", "importance"]
    raw = fi["raw_feature"].str.replace(r"^(remainder|genres|txt)__", "", regex=True)
    fi["feature"] = [nice.get(r, r) for r in raw]
    fi["group"] = np.where(fi["raw_feature"].str.startswith("genres__"), "Genre",
                   np.where(fi["raw_feature"].str.startswith("txt__"), "Logline words", "Project attribute"))
    fi = fi[["feature", "group", "importance"]]
else:
    fi = pd.DataFrame(columns=["feature", "group", "importance"])

metrics_path = ROOT / "models" / "metrics.json"
if metrics_path.exists():
    with metrics_path.open(encoding="utf-8") as f:
        metrics = json.load(f)
else:
    metrics = {}
metrics_df = pd.DataFrame({"metric": list(metrics.keys()), "value": list(metrics.values())})

# ---------------- save ----------------
print(f"\nWriting CSV files to {OUT.resolve()}")
save(films, "films.csv")
save(bridge, "film_genres.csv")
save(genre_year, "genre_year_summary.csv")
save(combos, "scenario_grid.csv")
save(fi, "feature_importance.csv")
save(metrics_df, "model_metrics.csv")
print("\nDone. Next: open Power BI Desktop and follow POWERBI_GUIDE.md")
print(f"Out-of-fold median abs % error: {films['abs_pct_error'].median():.0f}%")