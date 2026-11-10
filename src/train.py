"""
src/train.py  (v2)
Trains the Random Forest on log(revenue) from data/processed/titles.csv.

Improvements over v1:
  - adds release year (captures ticket-price inflation / market growth)
  - multi-hot genres instead of only the first genre
  - fewer, cleaner TF-IDF terms
  - hyperparameter tuning with RandomizedSearchCV
  - prints a v1-vs-v2 comparison so you can quote real before/after numbers

Run from the project root:  python src/train.py
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

DATA = r"C:\Users\dalal\OneDrive\Desktop\vibecode\streaming_roi\data\processed\titles.csv"
TARGET = "log_rev"
V1_FEATURES = ["log_budget", "runtime", "quarter", "star_power", "genre", "text"]
V2_FEATURES = ["log_budget", "runtime", "quarter", "year", "star_power", "genre_list", "text"]

# ---------------- load ----------------
df = pd.read_csv(DATA)
df["text"] = df["text"].fillna("")
df["genre"] = df["genre"].fillna("Unknown")
if "genre_list" not in df.columns:
    df["genre_list"] = df["genre"]
df["genre_list"] = df["genre_list"].fillna("").replace("", "Unknown")
df = df.dropna(subset=["log_budget", "runtime", "quarter", "year", "star_power", TARGET, "revenue"])
print(f"Training rows: {len(df)}")

y = df[TARGET]
idx_tr, idx_te = train_test_split(df.index, test_size=0.2, random_state=42)
kf = KFold(5, shuffle=True, random_state=42)

# ---------------- v1 baseline (for comparison) ----------------
pre_v1 = ColumnTransformer(
    [("cat", OneHotEncoder(handle_unknown="ignore"), ["genre"]),
     ("txt", TfidfVectorizer(max_features=100, stop_words="english"), "text")],
    remainder="passthrough",
)
v1 = Pipeline([("pre", pre_v1),
               ("rf", RandomForestRegressor(300, min_samples_leaf=3, n_jobs=-1, random_state=42))])
v1.fit(df.loc[idx_tr, V1_FEATURES], y[idx_tr])
v1_test = r2_score(y[idx_te], v1.predict(df.loc[idx_te, V1_FEATURES]))
v1_cv = cross_val_score(v1, df[V1_FEATURES], y, cv=kf, scoring="r2").mean()

# ---------------- v2 model ----------------
pre_v2 = ColumnTransformer(
    [
        # multi-hot: "Action|Adventure" -> Action=1, Adventure=1
        ("genres", CountVectorizer(token_pattern=r"[^|]+", lowercase=False, binary=True), "genre_list"),
        ("txt", TfidfVectorizer(max_features=30, stop_words="english", min_df=5), "text"),
    ],
    remainder="passthrough",
)
pipe = Pipeline([("pre", pre_v2), ("rf", RandomForestRegressor(random_state=42, n_jobs=1))])

param_dist = {
    "rf__n_estimators": [200, 300, 400],
    "rf__min_samples_leaf": [1, 2, 3, 5, 8],
    "rf__max_features": [0.3, 0.5, 0.7, 1.0],
    "rf__max_depth": [None, 10, 15, 20],
}
search = RandomizedSearchCV(pipe, param_dist, n_iter=20, cv=kf, scoring="r2",
                            n_jobs=-1, random_state=42, verbose=1)
search.fit(df.loc[idx_tr, V2_FEATURES], y[idx_tr])
best = search.best_estimator_
print("\nBest params:", search.best_params_)

Xte, yte = df.loc[idx_te, V2_FEATURES], y[idx_te]
pred = best.predict(Xte)
budget_te = np.exp(Xte["log_budget"])
roi_true, roi_pred = np.exp(yte) / budget_te - 1, np.exp(pred) / budget_te - 1

metrics = {
    "test_r2_log_revenue": round(r2_score(yte, pred), 3),
    "cv_r2_log_revenue": round(search.best_score_, 3),   # slightly optimistic: params were chosen on these folds
    "mae_log_revenue": round(mean_absolute_error(yte, pred), 3),
    "median_abs_error_roi": round(float(np.median(np.abs(roi_true - roi_pred))), 3),
    "v1_test_r2": round(v1_test, 3),
    "v1_cv_r2": round(v1_cv, 3),
    "latest_year_in_data": int(df["year"].max()),
}
print("\n=== v1 (baseline) vs v2 ===")
print(f"v1  test R2 {v1_test:.3f} | CV R2 {v1_cv:.3f}")
print(f"v2  test R2 {metrics['test_r2_log_revenue']:.3f} | CV R2 {metrics['cv_r2_log_revenue']:.3f} (tuned)")
print(json.dumps(metrics, indent=2))

# ---------------- feature importances ----------------
names = best.named_steps["pre"].get_feature_names_out()
imp = pd.Series(best.named_steps["rf"].feature_importances_, index=names).sort_values(ascending=False)
print("\nTop 15 features:\n", imp.head(15).round(3))

# ---------------- refit on all data and save ----------------
final = Pipeline(best.steps)
final.set_params(rf__n_jobs=-1)
final.fit(df[V2_FEATURES], y)

os.makedirs("models", exist_ok=True)
joblib.dump(final, "models/rf_model.joblib")
cut = df["revenue"].quantile([0.40, 0.75, 0.95]).tolist()
with open("models/tier_cutoffs.json", "w") as f:
    json.dump({"low_max": cut[0], "medium_max": cut[1], "high_max": cut[2]}, f, indent=2)
with open("models/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
imp.head(20).to_csv("models/feature_importances.csv")
print("\nSaved models/rf_model.joblib, tier_cutoffs.json, metrics.json, feature_importances.csv")