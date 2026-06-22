"""
Core analysis engine for the Autonomous Data Analyst Agent.
Profiles a dataframe and produces structured, rule-based insights.
No external API or token required for this layer.
"""

import pandas as pd
import numpy as np
from scipy import stats


def load_file(file_path_or_buffer, filename: str) -> pd.DataFrame:
    """Load CSV or Excel into a DataFrame."""
    if filename.lower().endswith(".csv"):
        df = pd.read_csv(file_path_or_buffer)
    elif filename.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(file_path_or_buffer)
    else:
        raise ValueError("Unsupported file type. Please upload a .csv or .xlsx/.xls file.")

    # Light cleanup: strip whitespace from column names
    df.columns = [str(c).strip() for c in df.columns]
    return df


def basic_profile(df: pd.DataFrame) -> dict:
    """High level shape and health of the dataset."""
    n_rows, n_cols = df.shape
    missing = df.isnull().sum()
    missing_pct = (missing / n_rows * 100).round(2)
    dupes = int(df.duplicated().sum())

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime64[ns]", "datetime64"]).columns.tolist()

    # Try to detect date-like string columns that weren't parsed
    for col in categorical_cols[:]:
        if df[col].dropna().empty:
            continue
        sample = df[col].dropna().astype(str).head(20)
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                parsed = pd.to_datetime(sample, errors="coerce")
            if parsed.notna().mean() > 0.8:
                datetime_cols.append(col)
                categorical_cols.remove(col)
        except Exception:
            pass

    return {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "missing_per_col": missing.to_dict(),
        "missing_pct_per_col": missing_pct.to_dict(),
        "total_missing_cells": int(missing.sum()),
        "duplicate_rows": dupes,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "datetime_cols": datetime_cols,
        "memory_mb": round(df.memory_usage(deep=True).sum() / (1024 ** 2), 2),
    }


def numeric_summary(df: pd.DataFrame, numeric_cols: list) -> pd.DataFrame:
    """Descriptive stats + skew/kurtosis + outlier counts for numeric columns."""
    rows = []
    for col in numeric_cols:
        series = df[col].dropna()
        if series.empty:
            continue
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outliers = series[(series < lower) | (series > upper)]

        rows.append({
            "column": col,
            "mean": round(series.mean(), 3),
            "median": round(series.median(), 3),
            "std": round(series.std(), 3),
            "min": round(series.min(), 3),
            "max": round(series.max(), 3),
            "skew": round(stats.skew(series), 3) if len(series) > 2 else np.nan,
            "outlier_count": len(outliers),
            "outlier_pct": round(len(outliers) / len(series) * 100, 2),
        })
    return pd.DataFrame(rows)


def categorical_summary(df: pd.DataFrame, categorical_cols: list, top_n: int = 5) -> dict:
    """Top categories and cardinality per categorical column."""
    summary = {}
    for col in categorical_cols:
        vc = df[col].value_counts(dropna=True)
        summary[col] = {
            "n_unique": int(df[col].nunique()),
            "top_values": vc.head(top_n).to_dict(),
            "is_likely_id": df[col].nunique() == len(df[col].dropna()) and len(df[col].dropna()) > 0,
        }
    return summary


def correlation_matrix(df: pd.DataFrame, numeric_cols: list) -> pd.DataFrame:
    if len(numeric_cols) < 2:
        return pd.DataFrame()
    return df[numeric_cols].corr(numeric_only=True).round(3)


def strong_correlations(corr_df: pd.DataFrame, threshold: float = 0.6) -> list:
    """Pull out pairs of columns with |correlation| above threshold."""
    pairs = []
    if corr_df.empty:
        return pairs
    cols = corr_df.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            val = corr_df.iloc[i, j]
            if pd.notna(val) and abs(val) >= threshold:
                pairs.append((cols[i], cols[j], round(float(val), 3)))
    pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    return pairs


def generate_rule_based_insights(df: pd.DataFrame, profile: dict, num_summary: pd.DataFrame,
                                   cat_summary: dict, corr_pairs: list) -> list:
    """
    Turns the statistical profile into a list of plain-English insight strings.
    This is deterministic and requires no LLM/API call.
    """
    insights = []
    n_rows = profile["n_rows"]

    # Shape / size
    insights.append(f"Dataset contains {n_rows:,} rows and {profile['n_cols']} columns "
                     f"({profile['memory_mb']} MB in memory).")

    # Missing data
    missing_cols = {k: v for k, v in profile["missing_pct_per_col"].items() if v > 0}
    if missing_cols:
        worst = sorted(missing_cols.items(), key=lambda x: x[1], reverse=True)[:3]
        worst_str = ", ".join([f"{k} ({v}% missing)" for k, v in worst])
        insights.append(f"Missing data detected. Most affected columns: {worst_str}.")
    else:
        insights.append("No missing values detected — the dataset is complete.")

    # Duplicates
    if profile["duplicate_rows"] > 0:
        pct = round(profile["duplicate_rows"] / n_rows * 100, 2)
        insights.append(f"Found {profile['duplicate_rows']:,} duplicate rows ({pct}% of the dataset). "
                         f"Consider deduplicating before downstream analysis.")

    # Numeric column insights
    if not num_summary.empty:
        for _, row in num_summary.iterrows():
            if row["outlier_pct"] > 5:
                insights.append(f"'{row['column']}' has notable outliers: {row['outlier_count']} values "
                                 f"({row['outlier_pct']}%) fall outside the typical IQR range.")
            if abs(row["skew"]) > 1 and not np.isnan(row["skew"]):
                direction = "right" if row["skew"] > 0 else "left"
                insights.append(f"'{row['column']}' is heavily skewed to the {direction} "
                                 f"(skew={row['skew']}), so the mean may be a misleading summary — "
                                 f"consider using the median instead.")

    # Categorical insights
    for col, info in cat_summary.items():
        if info["is_likely_id"]:
            insights.append(f"'{col}' looks like a unique identifier (every value is distinct) — "
                             f"likely not useful for grouping or correlation analysis.")
        elif info["n_unique"] == 1:
            insights.append(f"'{col}' has only one unique value across the whole dataset — "
                             f"it carries no information and can be dropped.")
        elif info["top_values"]:
            top_cat, top_count = next(iter(info["top_values"].items()))
            pct = round(top_count / n_rows * 100, 1)
            if pct > 50:
                insights.append(f"'{col}' is dominated by a single category ('{top_cat}', {pct}% of rows) — "
                                 f"check for class imbalance if this is used as a target.")

    # Correlations
    if corr_pairs:
        for col1, col2, val in corr_pairs[:5]:
            direction = "positive" if val > 0 else "negative"
            strength = "very strong" if abs(val) >= 0.85 else "strong"
            insights.append(f"{strength.capitalize()} {direction} correlation ({val}) between "
                             f"'{col1}' and '{col2}'.")
    elif len(profile["numeric_cols"]) >= 2:
        insights.append("No strong correlations (|r| ≥ 0.6) found among numeric columns.")

    return insights
