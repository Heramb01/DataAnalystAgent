"""
Autonomous Data Analyst Agent — Streamlit App
Upload a CSV/Excel file -> get automatic profiling, insights, and charts.

Run locally with:
    streamlit run app/main.py
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

import streamlit as st
import pandas as pd

from analyzer import (
    load_file, basic_profile, numeric_summary, categorical_summary,
    correlation_matrix, strong_correlations, generate_rule_based_insights
)
from charts import (
    make_missing_data_chart, make_numeric_distributions, make_categorical_bar_charts,
    make_correlation_heatmap, make_scatter_for_top_correlation, make_time_series_chart
)
from narrate import narrate

st.set_page_config(page_title="Autonomous Data Analyst Agent", layout="wide", page_icon="📊")

st.title("📊 Autonomous Data Analyst Agent")
st.caption("Upload a CSV or Excel file. The agent profiles your data, finds insights, "
           "and generates charts automatically — no API key required.")

with st.sidebar:
    st.header("⚙️ Settings")
    narration_mode = st.radio(
        "Narrative summary engine",
        options=["Local model (free, offline, no token)", "HuggingFace Hosted API (needs free token)"],
        index=0,
    )
    hf_token = None
    hf_model = None
    if "HuggingFace" in narration_mode:
        hf_token = st.text_input("HuggingFace API token", type="password",
                                  help="Get a free one at https://huggingface.co/settings/tokens")
        hf_model = st.text_input("Model name", value="HuggingFaceH4/zephyr-7b-beta",
                                  help="Must be an exact, case-sensitive model ID that supports "
                                       "the free Inference API.")
    st.divider()
    st.markdown("**Tip:** the local model downloads once (~80MB) the first time you "
                "generate a narrative, then runs fully offline.")

uploaded_file = st.file_uploader("Upload your data file", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    try:
        df = load_file(uploaded_file, uploaded_file.name)
    except Exception as e:
        st.error(f"Could not read the file: {e}")
        st.stop()

    st.success(f"Loaded **{uploaded_file.name}** — {df.shape[0]:,} rows x {df.shape[1]} columns")

    with st.expander("🔍 Preview raw data", expanded=False):
        st.dataframe(df.head(50), use_container_width=True)

    # ---- Run analysis ----
    profile = basic_profile(df)
    num_summary = numeric_summary(df, profile["numeric_cols"])
    cat_summary = categorical_summary(df, profile["categorical_cols"])
    corr_df = correlation_matrix(df, profile["numeric_cols"])
    corr_pairs = strong_correlations(corr_df)
    insights = generate_rule_based_insights(df, profile, num_summary, cat_summary, corr_pairs)

    # ---- Tabs ----
    tab1, tab2, tab3, tab4 = st.tabs(["🧠 Insights", "📈 Charts", "🔢 Numeric Stats", "📋 Raw Profile"])

    with tab1:
        st.subheader("Key Findings")
        for i in insights:
            st.markdown(f"- {i}")

        st.divider()
        st.subheader("AI-Generated Narrative Summary")
        if st.button("Generate narrative summary"):
            with st.spinner("Generating..."):
                try:
                    if "HuggingFace" in narration_mode:
                        text = narrate(insights, mode="hf_api", hf_token=hf_token, hf_model=hf_model)
                    else:
                        text = narrate(insights, mode="local")
                    st.write(text)
                except Exception as e:
                    st.error(f"Narration failed: {e}")
                    st.info("Falling back to local model...")
                    st.write(narrate(insights, mode="local"))

    with tab2:
        st.subheader("Automatically Generated Charts")

        missing_chart = make_missing_data_chart(profile)
        if missing_chart:
            st.plotly_chart(missing_chart, use_container_width=True)

        corr_heatmap = make_correlation_heatmap(corr_df)
        if corr_heatmap:
            st.plotly_chart(corr_heatmap, use_container_width=True)

        scatter = make_scatter_for_top_correlation(df, corr_pairs)
        if scatter:
            st.plotly_chart(scatter, use_container_width=True)

        ts_chart = make_time_series_chart(df, profile["datetime_cols"], profile["numeric_cols"])
        if ts_chart:
            st.plotly_chart(ts_chart, use_container_width=True)

        st.markdown("**Numeric distributions**")
        for col, fig in make_numeric_distributions(df, profile["numeric_cols"]):
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Categorical breakdowns**")
        for col, fig in make_categorical_bar_charts(df, profile["categorical_cols"], cat_summary):
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Numeric Column Statistics")
        if not num_summary.empty:
            st.dataframe(num_summary, use_container_width=True)
        else:
            st.info("No numeric columns found.")

        if not corr_df.empty:
            st.subheader("Correlation Matrix")
            st.dataframe(corr_df, use_container_width=True)

    with tab4:
        st.subheader("Full Profile (JSON)")
        st.json({
            "n_rows": profile["n_rows"],
            "n_cols": profile["n_cols"],
            "duplicate_rows": profile["duplicate_rows"],
            "numeric_cols": profile["numeric_cols"],
            "categorical_cols": profile["categorical_cols"],
            "datetime_cols": profile["datetime_cols"],
            "missing_pct_per_col": profile["missing_pct_per_col"],
        })

else:
    st.info("👆 Upload a CSV or Excel file to get started. Don't have one? "
            "Try the sample data generator below.")
    if st.button("Generate sample sales dataset"):
        import numpy as np
        rng = np.random.default_rng(42)
        n = 500
        sample = pd.DataFrame({
            "order_date": pd.date_range("2024-01-01", periods=n, freq="D"),
            "region": rng.choice(["North", "South", "East", "West"], n),
            "product": rng.choice(["Widget A", "Widget B", "Widget C"], n),
            "units_sold": rng.poisson(20, n),
            "unit_price": rng.normal(25, 5, n).round(2),
            "customer_rating": rng.normal(4.2, 0.6, n).clip(1, 5).round(1),
        })
        sample["revenue"] = (sample["units_sold"] * sample["unit_price"]).round(2)
        sample.to_csv("/tmp/sample_sales.csv", index=False)
        st.success("Sample created! Download it below, then upload it above.")
        st.download_button("Download sample_sales.csv", sample.to_csv(index=False),
                            file_name="sample_sales.csv", mime="text/csv")
