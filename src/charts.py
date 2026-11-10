"""
src/charts.py
Plotly figure builders and the break-even finder used by app.py.
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def bubble_chart(df: pd.DataFrame, user: dict | None = None):
    """Budget vs ROI for historical films. Bubble size = revenue, colour = genre.
    `user` = {"budget_m": float, "roi": float} adds a star for the film being evaluated."""
    d = df.copy()
    d["budget_m"] = d["budget"] / 1e6
    d["revenue_m"] = d["revenue"] / 1e6
    cap = d["roi"].quantile(0.97)                       # ROI is heavy-tailed: cap for display
    d["roi_shown"] = d["roi"].clip(upper=cap)
    fig = px.scatter(
        d, x="budget_m", y="roi_shown", size="revenue_m", color="genre",
        hover_name="title" if "title" in d.columns else None,
        hover_data={"budget_m": ":.1f", "revenue_m": ":.1f", "roi": ":.2f", "roi_shown": False},
        log_x=True, size_max=45, opacity=0.65,
        labels={"budget_m": "Budget ($M, log scale)", "roi_shown": "ROI (capped at 97th pct)"},
    )
    fig.add_hline(y=0, line_dash="dash", line_color="grey", annotation_text="Break-even (ROI = 0)")
    if user:
        fig.add_trace(go.Scatter(
            x=[user["budget_m"]], y=[min(user["roi"], cap)], mode="markers+text",
            marker=dict(symbol="star", size=24, color="black", line=dict(width=2, color="white")),
            text=["Your project"], textposition="top center", name="Your project",
        ))
    fig.update_layout(height=520, legend_title_text="Genre")
    return fig


def genre_trend_chart(df: pd.DataFrame, years: int = 5):
    """Share of releases per genre over the most recent `years` years in the data."""
    last = int(df["year"].max())
    d = df[df["year"] > last - years]
    share = d.groupby(["year", "genre"]).size().rename("n").reset_index()
    share["share_pct"] = share["n"] / share.groupby("year")["n"].transform("sum") * 100
    fig = px.line(share, x="year", y="share_pct", color="genre", markers=True,
                  labels={"share_pct": "Share of releases (%)", "year": "Release year"})
    fig.update_xaxes(dtick=1)
    fig.update_layout(height=450, legend_title_text="Genre")
    return fig, last - years + 1, last


def genre_roi_chart(df: pd.DataFrame):
    """Median ROI per genre (bar chart)."""
    g = df.groupby("genre")["roi"].median().sort_values().reset_index()
    fig = px.bar(g, x="roi", y="genre", orientation="h",
                 labels={"roi": "Median ROI", "genre": ""})
    fig.update_layout(height=450)
    return fig


def find_break_even(budgets_m: np.ndarray, revenues_m: np.ndarray):
    """Largest budget at which predicted revenue still covers cost.
    Returns (budget_m or None, status) where status is 'crossover', 'always_profitable' or 'never'."""
    diff = np.asarray(revenues_m) - np.asarray(budgets_m)
    if (diff >= 0).all():
        return None, "always_profitable"
    if (diff < 0).all():
        return None, "never"
    # last point where we go from covering cost to not covering it
    idx = np.where((diff[:-1] >= 0) & (diff[1:] < 0))[0]
    if len(idx) == 0:                                    # only crosses upward (loss at low budgets)
        idx = np.where((diff[:-1] < 0) & (diff[1:] >= 0))[0]
        i = idx[0]
    else:
        i = idx[-1]
    x0, x1, d0, d1 = budgets_m[i], budgets_m[i + 1], diff[i], diff[i + 1]
    return float(x0 + (x1 - x0) * (d0 / (d0 - d1))), "crossover"


def breakeven_chart(budgets_m, revenues_m, current_budget_m: float, break_even_m: float | None):
    """Predicted revenue vs budget, with the cost line y = x."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=budgets_m, y=revenues_m, mode="lines", name="Predicted revenue",
                             line=dict(width=3)))
    fig.add_trace(go.Scatter(x=budgets_m, y=budgets_m, mode="lines", name="Cost (budget)",
                             line=dict(dash="dash", color="red")))
    fig.add_vline(x=current_budget_m, line_dash="dot", line_color="grey",
                  annotation_text="Your budget", annotation_position="top")
    if break_even_m is not None:
        fig.add_trace(go.Scatter(x=[break_even_m], y=[break_even_m], mode="markers",
                                 marker=dict(size=14, color="black", symbol="diamond"),
                                 name=f"Break-even ≈ ${break_even_m:,.0f}M"))
    fig.update_layout(height=480, xaxis_title="Production budget ($M)", yaxis_title="$M",
                      legend=dict(orientation="h", y=-0.2))
    return fig