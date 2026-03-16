from __future__ import annotations

from io import BytesIO

import pandas as pd
import plotly.express as px
import streamlit as st

from src.cluster import find_best_k, fit_kmeans, run_pca
from src.explain import assign_segment_names, build_cluster_summary, generate_recommendations
from src.preprocess import load_data, prepare_features, scale_features
from src.utils import (
    CLUSTER_SUMMARY_PATH,
    CUSTOMER_SEGMENTS_PATH,
    DATA_PATH,
    ensure_reports_dir,
)


st.set_page_config(
    page_title="Transaction Behavior Clustering",
    page_icon=":bar_chart:",
    layout="wide",
)


@st.cache_data
def run_pipeline() -> dict[str, pd.DataFrame | int]:
    raw_df = load_data(DATA_PATH)
    _, numeric_df = prepare_features(raw_df)
    scaled_df, _ = scale_features(numeric_df)

    best_k, silhouette_scores = find_best_k(scaled_df)
    _, labels = fit_kmeans(scaled_df, best_k)
    pca_df = run_pca(scaled_df)

    customer_segments = raw_df.copy()
    customer_segments["cluster"] = labels.values
    customer_segments = pd.concat([customer_segments, pca_df], axis=1)

    cluster_summary = build_cluster_summary(numeric_df, labels)
    cluster_summary = assign_segment_names(cluster_summary)
    cluster_summary = generate_recommendations(cluster_summary)

    ensure_reports_dir()
    customer_segments.to_csv(CUSTOMER_SEGMENTS_PATH, index=False)
    cluster_summary.to_csv(CLUSTER_SUMMARY_PATH, index=False)

    return {
        "raw_df": raw_df,
        "numeric_df": numeric_df,
        "silhouette_scores": silhouette_scores,
        "best_k": best_k,
        "customer_segments": customer_segments,
        "cluster_summary": cluster_summary,
    }


def csv_bytes(df: pd.DataFrame) -> bytes:
    buffer = BytesIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue()


results = run_pipeline()
raw_df = results["raw_df"]
numeric_df = results["numeric_df"]
silhouette_scores = results["silhouette_scores"]
best_k = results["best_k"]
customer_segments = results["customer_segments"]
cluster_summary = results["cluster_summary"]

st.title("Transaction Behavior Clustering")
st.caption(
    "Customer segmentation for credit card portfolios using unsupervised learning, "
    "PCA visualization, and business-ready cluster interpretation."
)

metric_col_1, metric_col_2, metric_col_3 = st.columns(3)
metric_col_1.metric("Customers", f"{len(raw_df):,}")
metric_col_2.metric("Numeric Features", f"{numeric_df.shape[1]:,}")
metric_col_3.metric("Best Cluster Count", best_k)

st.subheader("Dataset Overview")
st.dataframe(raw_df.head(10), width="stretch")

st.subheader("Clustering Evaluation")
score_chart = px.line(
    silhouette_scores,
    x="k",
    y="silhouette_score",
    markers=True,
    title="Silhouette Score by Cluster Count",
)
score_chart.update_layout(xaxis_title="Number of Clusters", yaxis_title="Silhouette Score")
st.plotly_chart(score_chart, width="stretch")

left_col, right_col = st.columns(2)

with left_col:
    st.subheader("PCA Visualization")
    pca_chart = px.scatter(
        customer_segments,
        x="pca_1",
        y="pca_2",
        color=customer_segments["cluster"].astype(str),
        hover_data=["CUST_ID"] if "CUST_ID" in customer_segments.columns else None,
        title="Customer Clusters in PCA Space",
    )
    pca_chart.update_layout(legend_title="Cluster")
    st.plotly_chart(pca_chart, width="stretch")

with right_col:
    st.subheader("Cluster Distribution")
    cluster_counts = customer_segments["cluster"].value_counts().sort_index().reset_index()
    cluster_counts.columns = ["cluster", "customer_count"]
    bar_chart = px.bar(
        cluster_counts,
        x="cluster",
        y="customer_count",
        color=cluster_counts["cluster"].astype(str),
        title="Customers per Cluster",
    )
    bar_chart.update_layout(showlegend=False)
    st.plotly_chart(bar_chart, width="stretch")

st.subheader("Segment Profiles")
st.dataframe(cluster_summary, width="stretch")

st.subheader("Business Insights")
for row in cluster_summary.itertuples(index=False):
    st.markdown(f"**Cluster {row.cluster}: {row.segment_name}**")
    st.write(row.business_summary)
    st.write(f"Recommended action: {row.recommended_action}")

st.subheader("Download Reports")
download_col_1, download_col_2 = st.columns(2)

with download_col_1:
    st.download_button(
        label="Download customer_segments.csv",
        data=csv_bytes(customer_segments),
        file_name="customer_segments.csv",
        mime="text/csv",
    )

with download_col_2:
    st.download_button(
        label="Download cluster_summary.csv",
        data=csv_bytes(cluster_summary),
        file_name="cluster_summary.csv",
        mime="text/csv",
    )
