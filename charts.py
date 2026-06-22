"""
Chart generation for the Autonomous Data Analyst Agent.
Auto-selects relevant chart types based on column data types.
Uses Plotly (free, open-source) for interactive charts.
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd


def make_missing_data_chart(profile: dict):
    missing = {k: v for k, v in profile["missing_pct_per_col"].items() if v > 0}
    if not missing:
        return None
    df_missing = pd.DataFrame(
        {"column": list(missing.keys()), "missing_pct": list(missing.values())}
    ).sort_values("missing_pct", ascending=True)
    fig = px.bar(df_missing, x="missing_pct", y="column", orientation="h",
                 title="Missing Data by Column (%)", text="missing_pct")
    fig.update_layout(yaxis_title="", xaxis_title="% Missing")
    return fig


def make_numeric_distributions(df: pd.DataFrame, numeric_cols: list, max_charts: int = 6):
    """Histogram for each numeric column (capped to avoid huge dashboards)."""
    figs = []
    for col in numeric_cols[:max_charts]:
        fig = px.histogram(df, x=col, marginal="box", title=f"Distribution of {col}", nbins=40)
        figs.append((col, fig))
    return figs


def make_categorical_bar_charts(df: pd.DataFrame, categorical_cols: list, cat_summary: dict, max_charts: int = 6):
    """Bar chart of top categories for each categorical column (skips ID-like columns)."""
    figs = []
    count = 0
    for col in categorical_cols:
        if count >= max_charts:
            break
        info = cat_summary.get(col, {})
        if info.get("is_likely_id") or info.get("n_unique", 0) > 50:
            continue
        vc = df[col].value_counts(dropna=True).head(15).reset_index()
        vc.columns = [col, "count"]
        fig = px.bar(vc, x=col, y="count", title=f"Top categories in {col}")
        figs.append((col, fig))
        count += 1
    return figs


def make_correlation_heatmap(corr_df: pd.DataFrame):
    if corr_df.empty:
        return None
    fig = px.imshow(corr_df, text_auto=True, aspect="auto", color_continuous_scale="RdBu_r",
                     title="Correlation Heatmap", zmin=-1, zmax=1)
    return fig


def make_scatter_for_top_correlation(df: pd.DataFrame, corr_pairs: list):
    if not corr_pairs:
        return None
    col1, col2, val = corr_pairs[0]
    fig = px.scatter(df, x=col1, y=col2, trendline="ols",
                      title=f"{col1} vs {col2} (correlation = {val})")
    return fig


def make_time_series_chart(df: pd.DataFrame, datetime_cols: list, numeric_cols: list):
    """If a datetime column and a numeric column exist, plot a trend line."""
    if not datetime_cols or not numeric_cols:
        return None
    date_col = datetime_cols[0]
    value_col = numeric_cols[0]
    try:
        temp = df[[date_col, value_col]].copy()
        temp[date_col] = pd.to_datetime(temp[date_col], errors="coerce")
        temp = temp.dropna(subset=[date_col]).sort_values(date_col)
        if temp.empty:
            return None
        fig = px.line(temp, x=date_col, y=value_col, title=f"{value_col} over {date_col}")
        return fig
    except Exception:
        return None
