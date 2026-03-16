from __future__ import annotations

from typing import Dict

import pandas as pd


SEGMENT_LIBRARY: Dict[str, str] = {
    "high_balance_revolvers": "High Balance Revolvers",
    "frequent_full_payers": "Frequent Full Payers",
    "low_activity_customers": "Low Activity Customers",
    "premium_high_limit_users": "Premium High Limit Users",
    "installment_focused_spenders": "Installment Focused Spenders",
}


def build_cluster_summary(df: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    """Calculate average feature values and cluster size."""
    labelled_df = df.copy()
    labelled_df["cluster"] = labels.values

    summary_df = labelled_df.groupby("cluster").mean(numeric_only=True)
    summary_df["customer_count"] = labelled_df.groupby("cluster").size()
    summary_df = summary_df.reset_index()
    return summary_df


def assign_segment_names(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Attach a human-friendly segment label using simple business heuristics."""
    named_df = summary_df.copy()
    benchmarks = {
        "balance_high": named_df["BALANCE"].quantile(0.75) if "BALANCE" in named_df else 0.0,
        "credit_limit_high": named_df["CREDIT_LIMIT"].quantile(0.75) if "CREDIT_LIMIT" in named_df else 0.0,
        "cash_advance_high": named_df["CASH_ADVANCE"].quantile(0.75) if "CASH_ADVANCE" in named_df else 0.0,
        "purchases_low": named_df["PURCHASES"].quantile(0.25) if "PURCHASES" in named_df else 0.0,
    }
    named_df["segment_name"] = named_df.apply(
        lambda row: _segment_name_for_row(row, benchmarks),
        axis=1,
    )
    return named_df


def generate_recommendations(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Generate business-focused explanations for each cluster."""
    enriched_df = summary_df.copy()
    enriched_df["business_summary"] = enriched_df.apply(_build_business_summary, axis=1)
    enriched_df["recommended_action"] = enriched_df.apply(_build_recommendation, axis=1)
    return enriched_df


def _segment_name_for_row(row: pd.Series, benchmarks: dict[str, float]) -> str:
    balance = row.get("BALANCE", 0.0)
    purchases = row.get("PURCHASES", 0.0)
    full_payment = row.get("PRC_FULL_PAYMENT", 0.0)
    credit_limit = row.get("CREDIT_LIMIT", 0.0)
    cash_advance = row.get("CASH_ADVANCE", 0.0)
    installments = row.get("INSTALLMENTS_PURCHASES", 0.0)
    purchase_frequency = row.get("PURCHASES_FREQUENCY", 0.0)

    if (
        credit_limit >= benchmarks["credit_limit_high"]
        and balance >= benchmarks["balance_high"]
    ):
        return SEGMENT_LIBRARY["premium_high_limit_users"]
    if full_payment >= 0.5 and purchase_frequency >= 0.5:
        return SEGMENT_LIBRARY["frequent_full_payers"]
    if cash_advance > purchases and cash_advance >= benchmarks["cash_advance_high"]:
        return SEGMENT_LIBRARY["high_balance_revolvers"]
    if purchases <= benchmarks["purchases_low"] and balance < benchmarks["balance_high"] and purchase_frequency < 0.3:
        return SEGMENT_LIBRARY["low_activity_customers"]
    if installments >= cash_advance and installments > 300:
        return SEGMENT_LIBRARY["installment_focused_spenders"]
    return "Mixed Usage Customers"


def _build_business_summary(row: pd.Series) -> str:
    name = row.get("segment_name", "Customer Segment")
    balance = row.get("BALANCE", 0.0)
    purchases = row.get("PURCHASES", 0.0)
    cash_advance = row.get("CASH_ADVANCE", 0.0)
    credit_limit = row.get("CREDIT_LIMIT", 0.0)

    return (
        f"{name} carry an average balance of {balance:,.0f}, spend about {purchases:,.0f} "
        f"through purchases, use {cash_advance:,.0f} in cash advances, and have an average "
        f"credit limit near {credit_limit:,.0f}."
    )


def _build_recommendation(row: pd.Series) -> str:
    name = row.get("segment_name", "")

    if name == SEGMENT_LIBRARY["premium_high_limit_users"]:
        return "Offer premium rewards, concierge-style servicing, and retention campaigns focused on long-term value."
    if name == SEGMENT_LIBRARY["frequent_full_payers"]:
        return "Promote loyalty rewards and spend-based offers because these customers show disciplined repayment behavior."
    if name == SEGMENT_LIBRARY["high_balance_revolvers"]:
        return "Focus on credit-risk monitoring, balance transfer offers, and personalized debt management outreach."
    if name == SEGMENT_LIBRARY["low_activity_customers"]:
        return "Use reactivation campaigns, low-friction offers, and education on card benefits to increase engagement."
    if name == SEGMENT_LIBRARY["installment_focused_spenders"]:
        return "Highlight installment plans, merchant financing partnerships, and planned-purchase promotions."
    return "Monitor this mixed segment for cross-sell opportunities and refine messaging based on product usage patterns."
