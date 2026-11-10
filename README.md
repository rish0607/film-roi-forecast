# 🎬 Film Revenue & ROI Forecast

A decision-support app that estimates a film's **box-office revenue** and **return on investment** before it is greenlit. A Random Forest model powers an interactive Streamlit app, and the same model's outputs feed a four-page Power BI report.

**[Live demo](https://YOUR-APP-NAME.streamlit.app)** · **[Power BI guide](docs/POWERBI_GUIDE.md)**

![App overview](assets/app-overview.png)

## What it does

- **Forecast:** enter a budget, genres, release quarter, runtime, star power and an optional logline. Get estimated revenue, ROI, profit or loss, a revenue tier, and a plain-English verdict.
- **Uncertainty:** shows a typical range and the share of the model's internal estimates that cover the budget, based on how much the forest's trees disagree.
- **What moves the forecast:** a sensitivity chart showing how much budget, star power, runtime and release quarter each change revenue.
- **Scenario lab:** test a different budget or release window, compare quarters and budget levels, and pin up to six forecasts side by side.
- **Similar films:** real historical films in the same genre with a comparable budget.
- **Market context:** budget vs ROI for past films with year and genre filters, genre trends, and median ROI by genre.
- **Break-even:** the largest budget the model expects the film to carry before revenue falls below cost.
- **Power BI report:** portfolio overview, genre trends, model accuracy, and a scenario explorer.

## Screenshots

| Forecast | Scenario lab |
|---|---|
| ![Forecast](assets/forecast.png) | ![Scenario lab](assets/scenario-lab.png) |

| Market context | Power BI report |
|---|---|
| ![Market context](assets/market-context.png) | ![Power BI](assets/powerbi-overview.png) |

## How it works

```
TMDB 5000 (parquet)
      │  src/clean.py
      ▼
data/processed/titles.csv ──► src/train.py ──► models/rf_model.joblib
                                                   │
                         ┌─────────────────────────┤
                         ▼                         ▼
                      app.py              src/export_powerbi.py ──► powerbi/*.csv ──► Power BI
                  (Streamlit app)
```

**Target:** log of box-office revenue. ROI is derived as `revenue / budget - 1`.

**Features:** log budget, runtime, release quarter, release year, a cast-experience score, multi-hot genres, and 30 TF-IDF terms from the overview and tagline.

**Model:** `RandomForestRegressor` in a scikit-learn `Pipeline`, tuned with `RandomizedSearchCV` (5-fold CV).

**Results** (log-revenue R², from `models/metrics.json`):

| Model | Test R² | 5-fold CV R² |
|---|---|---|
| Baseline (single genre, 100 text terms, default forest) | _fill from metrics.json_ | _fill from metrics.json_ |
| Tuned (year + multi-hot genres + 30 text terms) | 0.477 | 0.467 |

The CV score for the tuned model is slightly optimistic because the hyperparameters were chosen on the same folds.

## Quickstart

```bash
git clone https://github.com/YOUR-USERNAME/film-roi-forecast.git
cd film-roi-forecast

python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

streamlit run app.py
```

The trained model and processed data are committed, so the app runs straight away. To rebuild everything from scratch:

```bash
# put the TMDB 5000 parquet file in data/raw/ (see "Data" below)
python src/clean.py
python src/train.py
python src/export_powerbi.py    # optional: refreshes the Power BI tables
streamlit run app.py
```

Run all scripts from the project root; they use relative paths.

## Project structure

```
├── app.py                    Streamlit app
├── src/
│   ├── clean.py              load parquet, clean, build targets and features
│   ├── train.py              train, tune, evaluate, save model
│   ├── charts.py             Plotly chart builders and break-even finder
│   └── export_powerbi.py     model-scored tables for Power BI
├── data/processed/titles.csv cleaned dataset used by the app
├── models/                   saved model, tier cutoffs, metrics, feature importances
├── powerbi/                  exported CSV tables and the .pbix report
├── docs/POWERBI_GUIDE.md     report layout, relationships and DAX measures
├── assets/                   screenshots
└── .streamlit/config.toml    app theme
```

## Power BI

`src/export_powerbi.py` writes six tables (films with out-of-fold predictions, a genre bridge table, genre-by-year trends, a scenario grid, feature importance, and metrics). The report is in `powerbi/streaming_roi.pbix`; the layout and DAX measures are in [`docs/POWERBI_GUIDE.md`](docs/POWERBI_GUIDE.md). The `.pbix` stores file paths from the author's machine, so to refresh it, point the data sources at your local `powerbi/` folder (Transform data → Data source settings).

## Limitations

- **Box office, not streaming.** The model forecasts theatrical gross. It does not predict streaming viewership.
- **Survivorship bias.** Films with missing or tiny revenue were dropped, which likely removes real flops. The surviving data, and therefore the forecasts, lean optimistic.
- **Gross is not profit.** Revenue is total ticket sales, not what the studio keeps, and budgets exclude marketing.
- **Dated data.** TMDB 5000 ends around 2017, so forecasts are made at that year's market levels.
- **Cast score is a proxy.** "Star power" counts how often top-billed actors appear in the dataset. It is not social media reach.
- **No extrapolation.** A Random Forest cannot predict beyond the budgets it has seen.
- **Uncertainty is indicative.** The typical range and coverage figure come from tree disagreement. They are not a statistical confidence interval.

## Roadmap

- Merge Netflix's public engagement reports for real viewership hours
- Add maturity ratings and audience data via the TMDB API
- Per-prediction explanations with SHAP

## Data

[TMDB 5000 Movie Dataset](https://huggingface.co/datasets/AiresPucrs/tmdb-5000-movies), also on [Kaggle](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata). The raw file is not included in this repository; download it into `data/raw/`. Check the dataset's terms before redistributing it.

## License

MIT. See [LICENSE](LICENSE).