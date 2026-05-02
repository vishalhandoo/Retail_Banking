import argparse
import html
import json
import os
import shutil
from http import HTTPStatus
from datetime import datetime
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlencode, urlparse
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
FINAL_OUTPUT_FILE = BASE_DIR / "final_project_output.csv"
MODEL_METRICS_FILE = BASE_DIR / "summary_model_metrics.csv"
SERVQUAL_CORRELATION_FILE = BASE_DIR / "summary_servqual_correlations.csv"
FEATURE_IMPORTANCE_FILE = BASE_DIR / "summary_feature_importance.csv"
THRESHOLD_FILE = BASE_DIR / "summary_thresholds.csv"
RELATIONSHIP_MANAGER_FILE = BASE_DIR / "summary_relationship_manager_effect.csv"
RETENTION_KPI_FILE = BASE_DIR / "summary_retention_kpis.csv"
RETENTION_FRAMEWORK_FILE = BASE_DIR / "summary_customer_retention_framework.csv"
CREDIT_RISK_INTEGRATION_FILE = BASE_DIR / "summary_credit_risk_integration.csv"
HIGH_RISK_CUSTOMERS_FILE = BASE_DIR / "likely_to_leave_customers.csv"
REPORT_FILE = BASE_DIR / "SIP_Report_Full_Draft.md"
ENV_FILE = BASE_DIR / ".env"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
CHATBOT_HISTORY_LIMIT = 6
CHATBOT_EXAMPLE_PROMPTS = [
    "Why are customers likely to leave in this portfolio?",
    "Which factors most affect retention and why?",
    "What operational improvements should be prioritized first?",
    "Which customers need urgent intervention right now?",
    "What does the complaint-resolution chart mean?",
    "How can the bank improve retention rate based on this dashboard?",
]
CHART_CONFIG = [
    {
        "file": "chart_servqual_retention_heatmap.png",
        "title": "Service experience and loyalty",
        "description": (
            "Shows which parts of the customer experience move most closely with stronger loyalty."
        ),
    },
    {
        "file": "chart_sqi_by_target.png",
        "title": "Service score for lower-risk vs higher-risk customers",
        "description": (
            "Compares the overall service experience score for customers with lower and higher leave risk."
        ),
    },
    {
        "file": "chart_servqual_feature_importance.png",
        "title": "What matters most in the risk estimate",
        "description": (
            "Highlights which customer and service factors matter most in the estimate."
        ),
    },
    {
        "file": "chart_app_rating_leave_risk.png",
        "title": "Low app ratings and higher risk",
        "description": "Shows where weaker app ratings start to line up with more leave risk.",
    },
    {
        "file": "chart_resolution_days_leave_risk.png",
        "title": "Slow complaint resolution and higher risk",
        "description": "Shows where longer complaint-resolution times start to line up with more leave risk.",
    },
    {
        "file": "chart_relationship_manager_retention.png",
        "title": "Effect of a dedicated banker",
        "description": (
            "Compares customers with and without a dedicated banker."
        ),
    },
]
CHART_FILES = [item["file"] for item in CHART_CONFIG]
SERVQUAL_DIMENSION_COLUMNS = [
    "servqual_reliability",
    "servqual_responsiveness",
    "servqual_assurance",
    "servqual_empathy",
    "servqual_tangibles",
]
SERVQUAL_LABELS = {
    "servqual_reliability": "Reliability",
    "servqual_responsiveness": "Responsiveness",
    "servqual_assurance": "Assurance",
    "servqual_empathy": "Personal care",
    "servqual_tangibles": "Digital experience",
}

COLOR_PRIMARY = "#0f4c5c"
COLOR_SECONDARY = "#d96c06"
COLOR_ACCENT = "#6c8a5d"
COLOR_SURFACE = "#f5efe3"
COLOR_TEXT = "#14232d"
COLOR_ALERT = "#9f2f28"

DOWNLOAD_FILES = {
    "final_project_output.csv": FINAL_OUTPUT_FILE,
    "summary_model_metrics.csv": MODEL_METRICS_FILE,
    "summary_thresholds.csv": THRESHOLD_FILE,
    "summary_relationship_manager_effect.csv": RELATIONSHIP_MANAGER_FILE,
    "summary_retention_kpis.csv": RETENTION_KPI_FILE,
    "summary_customer_retention_framework.csv": RETENTION_FRAMEWORK_FILE,
    "summary_credit_risk_integration.csv": CREDIT_RISK_INTEGRATION_FILE,
    "likely_to_leave_customers.csv": HIGH_RISK_CUSTOMERS_FILE,
}
PUBLIC_DOWNLOAD_FILES = {
    file_name: path
    for file_name, path in DOWNLOAD_FILES.items()
    if file_name not in {"final_project_output.csv", "likely_to_leave_customers.csv"}
}
PUBLIC_FILE_ROUTES = {
    **{f"/downloads/{name}": path for name, path in DOWNLOAD_FILES.items()},
    **{f"/charts/{name}": BASE_DIR / name for name in CHART_FILES},
}
RISK_SEGMENT_ORDER = ["Critical", "High", "Moderate", "Low"]
DISPLAY_LABELS = {
    "loan_type": "Loan Type",
    "leave_risk_segment": "Attention Level",
    "relationship_manager": "Dedicated Banker",
    "Has Relationship Manager": "Has Dedicated Banker",
    "No Relationship Manager": "No Dedicated Banker",
    "Critical": "Urgent",
    "High": "High",
    "Moderate": "Medium",
    "Low": "Low",
    "servqual_reliability": "Accuracy and first-time fixes",
    "servqual_responsiveness": "Speed of help",
    "servqual_assurance": "Trust and confidence",
    "servqual_empathy": "Personal care",
    "servqual_tangibles": "App experience",
}
PLAIN_KPI_LABELS = {
    "retention_rate": "Customers in safer shape",
    "high_critical_risk_share": "Need close attention",
    "reactive_retention_share": "Need direct follow-up now",
    "priority_retention_share": "Top-priority follow-up queue",
    "credit_review_required_rate": "Need approval before offers",
    "relationship_manager_coverage_rate": "Have a dedicated banker",
    "resolution_within_threshold_rate": "Complaints solved on time",
    "app_rating_above_threshold_rate": "Healthy app ratings",
    "average_product_count": "Average products per customer",
    "five_plus_products_rate": "Customers with 5+ products",
    "proactive_retention_share": "Handled early, before problems grow",
}
PLAIN_KPI_DETAILS = {
    "retention_rate": "Share of customers currently outside the main warning group.",
    "high_critical_risk_share": "Customers who may need closer follow-up.",
    "reactive_retention_share": "Customers already showing strong warning signs.",
    "priority_retention_share": "Customers who should be reviewed first.",
    "credit_review_required_rate": "Customers who need approval before credit-related offers or changes.",
    "relationship_manager_coverage_rate": "Customers who already have a dedicated banker.",
    "resolution_within_threshold_rate": "Customers whose complaints are being solved quickly enough.",
    "app_rating_above_threshold_rate": "Customers whose app experience still looks healthy.",
    "average_product_count": "A simple sign of how many banking relationships a customer has with us.",
    "five_plus_products_rate": "Customers with deeper relationships across the bank.",
    "proactive_retention_share": "Customers being managed early, before issues become urgent.",
}
PAGE_CONFIG = [
    ("overview", "Overview", "Snapshot of customer volume, service quality, and risk."),
    ("health", "Health Check", "Portfolio checkpoints and KPI definitions."),
    ("drivers", "Key Issues", "Warning signs that appear to push risk higher."),
    ("charts", "Charts", "Visual diagnostics behind the main findings."),
    ("actions", "Action List", "Customers to review first."),
    ("copilot", "Decipher AI", "Ask grounded questions about the dashboard findings."),
    ("details", "Details", "Analyst tables and supporting exports."),
    ("downloads", "Files", "Latest generated CSV downloads."),
]
PAGE_LABELS = {page_id: label for page_id, label, _ in PAGE_CONFIG}
PAGE_ROUTES = {"/": "overview", "/dashboard": "overview"}
PAGE_ROUTES.update({f"/{page_id}": page_id for page_id, _, _ in PAGE_CONFIG})
STATIC_EXPORT_DIR = BASE_DIR / "docs"


def load_csv(path):
    resolved_path = resolve_output_path(path)
    if resolved_path.exists():
        return pd.read_csv(resolved_path)
    return pd.DataFrame()


def resolve_output_path(path):
    fallback_path = path.with_name(f"{path.stem}_latest{path.suffix}")
    candidates = [candidate for candidate in [path, fallback_path] if candidate.exists()]
    if not candidates:
        return path
    return max(candidates, key=lambda candidate: candidate.stat().st_mtime)


def load_data():
    return {
        "final": load_csv(FINAL_OUTPUT_FILE),
        "metrics": load_csv(MODEL_METRICS_FILE),
        "correlations": load_csv(SERVQUAL_CORRELATION_FILE),
        "feature_importance": load_csv(FEATURE_IMPORTANCE_FILE),
        "thresholds": load_csv(THRESHOLD_FILE),
        "relationship_manager": load_csv(RELATIONSHIP_MANAGER_FILE),
        "retention_kpis": load_csv(RETENTION_KPI_FILE),
        "retention_framework": load_csv(RETENTION_FRAMEWORK_FILE),
        "credit_risk_integration": load_csv(CREDIT_RISK_INTEGRATION_FILE),
        "high_risk": load_csv(HIGH_RISK_CUSTOMERS_FILE),
    }


def get_last_updated():
    candidate_paths = [
        FINAL_OUTPUT_FILE,
        MODEL_METRICS_FILE,
        SERVQUAL_CORRELATION_FILE,
        FEATURE_IMPORTANCE_FILE,
        THRESHOLD_FILE,
        RELATIONSHIP_MANAGER_FILE,
        RETENTION_KPI_FILE,
        RETENTION_FRAMEWORK_FILE,
        CREDIT_RISK_INTEGRATION_FILE,
        HIGH_RISK_CUSTOMERS_FILE,
    ] + [BASE_DIR / file_name for file_name in CHART_FILES]
    existing_paths = []
    for path in candidate_paths:
        resolved_path = resolve_output_path(path)
        if resolved_path.exists():
            existing_paths.append(resolved_path)

    if not existing_paths:
        return "No analysis outputs found"

    latest_timestamp = max(path.stat().st_mtime for path in existing_paths)
    return datetime.fromtimestamp(latest_timestamp).strftime("%Y-%m-%d %H:%M:%S")


def format_number(value, decimals=0):
    if pd.isna(value):
        return "NA"
    return f"{value:,.{decimals}f}"


def format_percent(value, decimals=2):
    if pd.isna(value):
        return "NA"
    return f"{value * 100:.{decimals}f}%"


def format_currency(value):
    numeric_value = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric_value):
        return "NA"
    return f"Rs {numeric_value:,.0f}"


def format_label(value):
    if pd.isna(value):
        return "NA"
    text = str(value)
    return DISPLAY_LABELS.get(text, text.replace("servqual_", "").replace("_", " ").title())


def safe_numeric(value):
    return pd.to_numeric(value, errors="coerce")


def safe_flag_state(value):
    numeric_value = safe_numeric(value)
    if pd.isna(numeric_value):
        return None
    return bool(numeric_value == 1)


def safe_flag(value):
    return safe_flag_state(value) is True


def format_flag_status(value):
    state = safe_flag_state(value)
    if state is True:
        return "Yes"
    if state is False:
        return "No"
    return "NA"


def plain_target_note(target_strategy, fallback_note):
    if target_strategy == "observed_churn":
        return "This estimate is based on actual customer exits recorded in the data."
    if target_strategy == "retention_proxy":
        return (
            "This dataset does not show confirmed customer exits, so the dashboard treats "
            "customers with the weakest loyalty scores as the warning group."
        )
    return fallback_note


def plain_kpi_label(metric_name, fallback_label):
    return PLAIN_KPI_LABELS.get(str(metric_name), fallback_label)


def plain_kpi_detail(metric_name, fallback_detail):
    return PLAIN_KPI_DETAILS.get(str(metric_name), fallback_detail)


def get_filter_value(params, key):
    value = params.get(key, ["All"])[0]
    return value if value else "All"


def normalize_page(page_id):
    return page_id if page_id in PAGE_LABELS else "overview"


def build_page_url(page_id, params, static_site=False):
    if static_site:
        return "index.html" if page_id == "overview" else f"{page_id}.html"

    filtered_params = {}
    for key, values in params.items():
        if key == "page":
            continue
        if not values:
            continue
        value = values[0]
        if value not in ("", "All"):
            filtered_params[key] = value

    query = urlencode(filtered_params)
    path = "/" if page_id == "overview" else f"/{page_id}"
    return f"{path}?{query}" if query else path


def build_page_nav(active_page, params, static_site=False):
    links = []
    active_page = normalize_page(active_page)
    for page_id, label, detail in PAGE_CONFIG:
        active_class = " page-link-active" if page_id == active_page else ""
        page_url = build_page_url(page_id, params, static_site)
        links.append(
            f"""
            <a class="page-link{active_class}" href="{html.escape(page_url)}">
                <span>{html.escape(label)}</span>
                <small>{html.escape(detail)}</small>
            </a>
            """
        )
    return f'<nav class="page-nav" aria-label="Dashboard pages">{"".join(links)}</nav>'


def build_select(name, selected_value, options):
    option_html = ['<option value="All">All</option>']
    for option in options:
        option_text = str(option)
        selected_attr = ' selected="selected"' if option_text == selected_value else ""
        option_html.append(
            f'<option value="{html.escape(option_text)}"{selected_attr}>{html.escape(format_label(option_text))}</option>'
        )
    return f"""
    <label class="filter">
        <span>{html.escape(format_label(name))}</span>
        <select name="{html.escape(name)}">
            {''.join(option_html)}
        </select>
    </label>
    """


def dataframe_to_html(frame):
    if frame.empty:
        return '<div class="empty-state">No rows match the current filter.</div>'

    display_frame = frame.copy()
    numeric_columns = display_frame.select_dtypes(include="number").columns
    display_frame[numeric_columns] = display_frame[numeric_columns].round(4)
    return display_frame.to_html(index=False, classes="data-table", border=0)


def filter_dataframe(df, params):
    if df.empty:
        return df

    filtered = df.copy()
    loan_type_filter = get_filter_value(params, "loan_type")
    leave_risk_filter = get_filter_value(params, "leave_risk_segment")
    rm_filter = get_filter_value(params, "relationship_manager")

    if loan_type_filter != "All":
        filtered = filtered[filtered["loan_type"] == loan_type_filter]

    if leave_risk_filter != "All" and "leave_risk_segment" in filtered.columns:
        filtered = filtered[filtered["leave_risk_segment"] == leave_risk_filter]

    if rm_filter != "All":
        rm_value = 1 if rm_filter == "Has Relationship Manager" else 0
        filtered = filtered[filtered["has_relationship_manager"] == rm_value]

    return filtered


def sort_priority_customers(df):
    if df.empty:
        return df

    sort_columns = [
        column
        for column in [
            "risk_adjusted_retention_score",
            "predicted_leave_probability",
            "clv",
        ]
        if column in df.columns
    ]
    if not sort_columns:
        return df

    return df.sort_values(
        sort_columns,
        ascending=[False] * len(sort_columns),
        na_position="last",
    )


def ensure_dashboard_columns(df):
    if df.empty:
        return df

    normalized_df = df.copy()
    default_values = {
        "risk_adjusted_retention_score": np.nan,
        "retention_strategy_lane": "Proactive Retention",
        "credit_review_required": 0,
        "risk_band": "NA",
        "recommended_action": "Review account",
    }
    for column, default_value in default_values.items():
        if column not in normalized_df.columns:
            normalized_df[column] = default_value

    return normalized_df


def summarize_portfolio(filtered_df):
    if filtered_df.empty:
        return {
            "customers": 0,
            "avg_leave_probability": np.nan,
            "avg_risk_adjusted_retention_score": np.nan,
            "high_risk_customers": 0,
            "critical_customers": 0,
            "high_risk_share": np.nan,
            "avg_service_quality_index": np.nan,
            "avg_retention_score": np.nan,
            "avg_app_rating_score": np.nan,
            "avg_resolution_days": np.nan,
            "relationship_manager_coverage": np.nan,
            "reactive_retention_customers": 0,
            "credit_review_required_customers": 0,
            "priority_without_rm": 0,
            "weakest_dimension": "NA",
            "strongest_dimension": "NA",
            "dominant_high_risk_loan_type": "NA",
        }

    dimension_mean = filtered_df[SERVQUAL_DIMENSION_COLUMNS].mean().sort_values()
    high_risk_mask = filtered_df["leave_risk_segment"].isin(["High", "Critical"])
    critical_mask = filtered_df["leave_risk_segment"].eq("Critical")
    high_risk_df = filtered_df[high_risk_mask]
    dominant_loan_type = "NA"
    if not high_risk_df.empty:
        dominant_loan_type = str(high_risk_df["loan_type"].mode().iloc[0])

    return {
        "customers": int(len(filtered_df)),
        "avg_leave_probability": float(filtered_df["predicted_leave_probability"].mean()),
        "avg_risk_adjusted_retention_score": float(
            filtered_df["risk_adjusted_retention_score"].mean()
        ),
        "high_risk_customers": int(high_risk_mask.sum()),
        "critical_customers": int(critical_mask.sum()),
        "high_risk_share": float(high_risk_mask.mean()),
        "avg_service_quality_index": float(filtered_df["service_quality_index"].mean()),
        "avg_retention_score": float(filtered_df["retention_score"].mean()),
        "avg_app_rating_score": float(filtered_df["app_rating_score"].mean()),
        "avg_resolution_days": float(filtered_df["resolution_days"].mean()),
        "relationship_manager_coverage": float(filtered_df["has_relationship_manager"].mean()),
        "reactive_retention_customers": int(
            filtered_df["retention_strategy_lane"].eq("Reactive Retention").sum()
        ),
        "credit_review_required_customers": int(
            pd.to_numeric(
                filtered_df["credit_review_required"], errors="coerce"
            ).fillna(0).sum()
        ),
        "priority_without_rm": int(
            filtered_df.loc[high_risk_mask, "has_relationship_manager"].eq(0).sum()
        ),
        "weakest_dimension": dimension_mean.index[0],
        "strongest_dimension": dimension_mean.index[-1],
        "dominant_high_risk_loan_type": dominant_loan_type,
    }


def load_local_env_file():
    if not ENV_FILE.exists():
        return

    for raw_line in ENV_FILE.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue

        if value.startswith(("'", '"')) and value.endswith(("'", '"')) and len(value) >= 2:
            value = value[1:-1]
        os.environ.setdefault(key, value)


load_local_env_file()


def get_gemini_api_key():
    return os.environ.get("GEMINI_API_KEY", "").strip()


def get_gemini_model():
    return os.environ.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL).strip() or DEFAULT_GEMINI_MODEL


def build_filter_summary(params):
    labels = []
    for key in ["loan_type", "leave_risk_segment", "relationship_manager"]:
        value = get_filter_value(params, key)
        if value != "All":
            labels.append(f"{format_label(key)}: {format_label(value)}")
    return ", ".join(labels) if labels else "All customers in the dashboard"


def load_report_excerpt():
    resolved_path = resolve_output_path(REPORT_FILE)
    if not resolved_path.exists():
        return ""

    report_text = resolved_path.read_text(encoding="utf-8", errors="ignore")
    start_marker = "## 4. Executive Summary"
    end_marker = "## 5. Body"
    start_index = report_text.find(start_marker)
    end_index = report_text.find(end_marker)
    if start_index != -1 and end_index != -1 and end_index > start_index:
        report_text = report_text[start_index:end_index]
    report_text = " ".join(report_text.split())
    return report_text[:2500]


def build_top_urgent_customer_lines(filtered_df, threshold_df):
    if filtered_df.empty:
        return ["No customer rows are available for the current filter."]

    resolution_threshold = get_threshold_record(threshold_df, "resolution_days")
    app_threshold = get_threshold_record(threshold_df, "app_rating_score")
    resolution_cutoff = (
        pd.to_numeric(resolution_threshold.get("threshold"), errors="coerce")
        if resolution_threshold is not None
        else np.nan
    )
    app_cutoff = (
        pd.to_numeric(app_threshold.get("threshold"), errors="coerce")
        if app_threshold is not None
        else np.nan
    )

    top_rows = sort_priority_customers(filtered_df).head(5)
    lines = []
    for _, row in top_rows.iterrows():
        reasons = []
        resolution_days = pd.to_numeric(row.get("resolution_days"), errors="coerce")
        app_rating = pd.to_numeric(row.get("app_rating_score"), errors="coerce")
        if pd.notna(resolution_cutoff) and pd.notna(resolution_days) and resolution_days > resolution_cutoff:
            reasons.append(
                f"complaint resolution is above the {format_number(resolution_cutoff, 1)} day threshold"
            )
        if pd.notna(app_cutoff) and pd.notna(app_rating) and app_rating <= app_cutoff:
            reasons.append(
                f"app rating is at or below the {format_number(app_cutoff, 1)} warning point"
            )
        if safe_flag_state(row.get("has_relationship_manager")) is False:
            reasons.append("no relationship manager is assigned")
        if not reasons:
            reasons.append("overall priority score and predicted leave probability are both elevated")

        lines.append(
            "Customer {customer_id}: {segment}, leave probability {leave_prob}, priority score {priority}, "
            "loan type {loan_type}, recommended action {action}, reasons: {reasons}.".format(
                customer_id=row.get("customer_id", "NA"),
                segment=format_label(row.get("leave_risk_segment", "NA")),
                leave_prob=format_percent(row.get("predicted_leave_probability"), 1),
                priority=format_number(row.get("risk_adjusted_retention_score"), 2),
                loan_type=row.get("loan_type", "NA"),
                action=row.get("recommended_action", "Review account"),
                reasons="; ".join(reasons),
            )
        )
    return lines


def build_dashboard_context(data, filtered_df, params, active_page):
    metrics_df = data["metrics"]
    if filtered_df.empty or metrics_df.empty:
        return "Dashboard context is unavailable because the required analysis files are missing."

    metrics = metrics_df.iloc[0]
    portfolio_summary = summarize_portfolio(filtered_df)
    threshold_df = data["thresholds"].copy()
    relationship_manager_df = data["relationship_manager"].copy()
    retention_kpi_df = data["retention_kpis"].copy()
    feature_importance_df = data["feature_importance"].copy()
    correlations_df = data["correlations"].copy()
    retention_framework_df = data["retention_framework"].copy()
    credit_risk_integration_df = data["credit_risk_integration"].copy()

    threshold_lines = []
    for _, row in threshold_df.iterrows():
        threshold_lines.append(
            "{feature}: {direction} risk threshold at {threshold}; risky group leave rate {risky_rate}; "
            "other group leave rate {other_rate}; lift {lift}x.".format(
                feature=row.get("feature", "NA"),
                direction=row.get("risk_direction", "NA"),
                threshold=row.get("threshold", "NA"),
                risky_rate=format_percent(pd.to_numeric(row.get("risky_group_leave_rate"), errors="coerce"), 1),
                other_rate=format_percent(pd.to_numeric(row.get("other_group_leave_rate"), errors="coerce"), 1),
                lift=format_number(pd.to_numeric(row.get("risk_lift_multiple"), errors="coerce"), 2),
            )
        )

    relationship_lines = []
    for _, row in relationship_manager_df.iterrows():
        relationship_lines.append(
            "{status}: customer count {count}, avg predicted leave probability {leave_prob}, "
            "retention proxy {retention}, avg service quality {service_quality}.".format(
                status=row.get("relationship_manager_status", "NA"),
                count=format_number(pd.to_numeric(row.get("customer_count"), errors="coerce")),
                leave_prob=format_percent(pd.to_numeric(row.get("avg_predicted_leave_probability"), errors="coerce"), 1),
                retention=format_percent(pd.to_numeric(row.get("retention_rate_proxy"), errors="coerce"), 1),
                service_quality=format_percent(pd.to_numeric(row.get("avg_service_quality_index"), errors="coerce"), 1),
            )
        )

    kpi_lines = []
    for _, row in retention_kpi_df.iterrows():
        kpi_lines.append(
            "{label}: {value}. Interpretation: {meaning}".format(
                label=row.get("metric_label", row.get("metric_name", "NA")),
                value=format_kpi_value(row.get("metric_value"), row.get("unit")),
                meaning=plain_kpi_detail(row.get("metric_name"), str(row.get("interpretation", ""))),
            )
        )

    feature_lines = []
    for _, row in feature_importance_df.head(5).iterrows():
        feature_lines.append(
            f"{format_label(row.get('feature', 'NA'))}: importance {format_number(pd.to_numeric(row.get('importance'), errors='coerce'), 3)}"
        )

    correlation_lines = []
    for _, row in correlations_df.head(5).iterrows():
        correlation_lines.append(
            f"{format_label(row.get('servqual_dimension', 'NA'))}: correlation with retention score {format_number(pd.to_numeric(row.get('correlation_with_retention_score'), errors='coerce'), 3)}"
        )

    framework_lines = []
    for _, row in retention_framework_df.head(6).iterrows():
        framework_lines.append(
            "{segment}: {subsegment}, customers {count}, avg leave probability {leave_prob}, RM coverage {rm_coverage}.".format(
                segment=row.get("segment_label", "NA"),
                subsegment=row.get("subsegment_label", "NA"),
                count=format_number(pd.to_numeric(row.get("customer_count"), errors="coerce")),
                leave_prob=format_percent(pd.to_numeric(row.get("avg_predicted_leave_probability"), errors="coerce"), 1),
                rm_coverage=format_percent(pd.to_numeric(row.get("relationship_manager_coverage_rate"), errors="coerce"), 1),
            )
        )

    credit_lines = []
    for _, row in credit_risk_integration_df.head(5).iterrows():
        credit_lines.append(
            "{band}: avg leave probability {leave_prob}, credit review required {credit_review}, reactive retention rate {reactive_rate}.".format(
                band=row.get("risk_band", "NA"),
                leave_prob=format_percent(pd.to_numeric(row.get("avg_predicted_leave_probability"), errors="coerce"), 1),
                credit_review=format_percent(pd.to_numeric(row.get("credit_review_required_rate"), errors="coerce"), 1),
                reactive_rate=format_percent(pd.to_numeric(row.get("reactive_retention_rate"), errors="coerce"), 1),
            )
        )

    chart_lines = [
        f"{item['title']}: {item['description']}"
        for item in CHART_CONFIG
    ]

    current_view_lines = [
        f"Active page: {PAGE_LABELS.get(active_page, active_page)}.",
        f"Current filter scope: {build_filter_summary(params)}.",
        f"Customers shown: {format_number(portfolio_summary['customers'])}.",
        f"High and critical risk customers: {format_number(portfolio_summary['high_risk_customers'])} ({format_percent(portfolio_summary['high_risk_share'], 1)}).",
        f"Critical customers: {format_number(portfolio_summary['critical_customers'])}.",
        f"Average predicted leave probability: {format_percent(portfolio_summary['avg_leave_probability'], 1)}.",
        f"Average risk-adjusted retention score: {format_number(portfolio_summary['avg_risk_adjusted_retention_score'], 2)}.",
        f"Average app rating: {format_number(portfolio_summary['avg_app_rating_score'], 2)}.",
        f"Average complaint resolution days: {format_number(portfolio_summary['avg_resolution_days'], 2)}.",
        f"Relationship-manager coverage in this view: {format_percent(portfolio_summary['relationship_manager_coverage'], 1)}.",
        f"Weakest service dimension in this view: {format_label(portfolio_summary['weakest_dimension'])}.",
        f"Strongest service dimension in this view: {format_label(portfolio_summary['strongest_dimension'])}.",
        f"Loan type with the most urgent cases: {portfolio_summary['dominant_high_risk_loan_type']}.",
        f"Higher-risk customers without a relationship manager: {format_number(portfolio_summary['priority_without_rm'])}.",
        f"Customers needing approval before credit-related offers: {format_number(portfolio_summary['credit_review_required_customers'])}.",
    ]

    report_excerpt = load_report_excerpt()

    context_sections = [
        "PROJECT SUMMARY",
        f"Dataset rows loaded: {format_number(metrics.get('rows_loaded'))}.",
        f"Target strategy: {plain_target_note(str(metrics.get('target_strategy', '')), str(metrics.get('target_note', '')))}",
        f"Model performance: accuracy {format_percent(pd.to_numeric(metrics.get('accuracy'), errors='coerce'), 2)}, precision {format_percent(pd.to_numeric(metrics.get('precision'), errors='coerce'), 2)}, recall {format_percent(pd.to_numeric(metrics.get('recall'), errors='coerce'), 2)}.",
        f"Strongest SERVQUAL driver in the model: {format_label(metrics.get('strongest_servqual_dimension', 'NA'))}.",
        "",
        "CURRENT DASHBOARD VIEW",
        *current_view_lines,
        "",
        "RETENTION KPI SUMMARY",
        *kpi_lines,
        "",
        "TOP FEATURE IMPORTANCE DRIVERS",
        *feature_lines,
        "",
        "SERVQUAL CORRELATIONS WITH RETENTION",
        *correlation_lines,
        "",
        "KEY OPERATIONAL THRESHOLDS",
        *threshold_lines,
        "",
        "RELATIONSHIP MANAGER EFFECT",
        *relationship_lines,
        "",
        "RETENTION FRAMEWORK SNAPSHOT",
        *framework_lines,
        "",
        "CREDIT-RISK INTEGRATION SNAPSHOT",
        *credit_lines,
        "",
        "CHART GUIDE",
        *chart_lines,
        "",
        "TOP URGENT CUSTOMERS IN THE CURRENT VIEW",
        *build_top_urgent_customer_lines(filtered_df, threshold_df),
    ]
    if report_excerpt:
        context_sections.extend(["", "REPORT EXECUTIVE SUMMARY EXCERPT", report_excerpt])

    return "\n".join(str(item) for item in context_sections if item is not None)


def build_chatbot_system_prompt():
    return """
You are Decipher AI, the dashboard assistant for a retail banking retention and leave-risk dashboard.
Your audience is the dashboard handler, faculty reviewer, or manager, not end customers.

Always follow these rules:
- Ground every answer in the dashboard context provided with the request.
- Do not invent metrics, counts, thresholds, or causal claims that are not supported by the provided context.
- If the data needed for a direct answer is unavailable, say that clearly and then give a best-practice recommendation.
- Use simple business language, not technical jargon.
- Be concise, clear, and action-oriented.
- When possible, explain the answer using the project's observed findings:
  complaint-resolution days above threshold are strongly associated with higher leave risk;
  lower app ratings are associated with higher leave risk;
  relationship-manager coverage improves retention outcomes;
  responsiveness, reliability, and assurance are major drivers.
- Prefer recommendations such as improving complaint resolution time, improving app experience,
  improving responsiveness and reliability, targeting high-risk customers, and expanding relationship-manager
  support where it is useful.
- If you need to refer to yourself by name, use Decipher AI.
- Never describe yourself as a generic chatbot.

Use this response structure and complete every section:
Insight:
Why it matters:
Recommended action:

Keep answers complete and practical. Prefer 120-220 words unless the user asks for more detail.
""".strip()


def normalize_chat_history(history):
    cleaned_history = []
    if not isinstance(history, list):
        return cleaned_history

    for item in history[-CHATBOT_HISTORY_LIMIT:]:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = str(item.get("content", "")).strip()
        if role not in {"user", "assistant"} or not content:
            continue
        cleaned_history.append({"role": role, "content": content[:3000]})
    return cleaned_history


def extract_gemini_text(response_payload):
    candidates = response_payload.get("candidates") or []
    complete_texts = []
    partial_texts = []
    for candidate in candidates:
        content = candidate.get("content") or {}
        parts = content.get("parts") or []
        text_parts = [part.get("text", "") for part in parts if isinstance(part, dict)]
        combined = "".join(text_parts).strip()
        if combined:
            finish_reason = str(candidate.get("finishReason", ""))
            if finish_reason == "MAX_TOKENS":
                partial_texts.append(combined)
            else:
                complete_texts.append(combined)

    if complete_texts:
        return max(complete_texts, key=len)
    if partial_texts:
        longest_partial = max(partial_texts, key=len)
        return (
            longest_partial.rstrip()
            + "\n\nRecommended action:\nAsk the question again with a narrower scope if you need a longer version; "
            "the model response reached its output limit."
        )

    prompt_feedback = response_payload.get("promptFeedback") or {}
    block_reason = prompt_feedback.get("blockReason")
    if block_reason:
        return (
            "Insight:\nThe assistant could not answer because the model blocked this request.\n\n"
            "Why it matters:\nThis usually happens when the API did not return a usable completion.\n\n"
            "Recommended action:\nRephrase the question more specifically around the dashboard findings and try again."
        )
    return ""


def ask_gemini_dashboard_assistant(question, dashboard_context, history=None):
    api_key = get_gemini_api_key()
    if not api_key:
        return {
            "ok": False,
            "status": HTTPStatus.SERVICE_UNAVAILABLE,
            "message": (
                "Decipher AI is not configured yet because `GEMINI_API_KEY` is missing. "
                "Add the environment variable, restart the dashboard, and try again."
            ),
        }

    model_name = get_gemini_model()
    conversation_contents = []
    for item in normalize_chat_history(history):
        conversation_contents.append(
            {
                "role": "model" if item["role"] == "assistant" else "user",
                "parts": [{"text": item["content"]}],
            }
        )

    user_prompt = (
        "Use the dashboard context below to answer the user's question.\n\n"
        f"{dashboard_context}\n\n"
        f"USER QUESTION:\n{question.strip()}"
    )
    conversation_contents.append({"role": "user", "parts": [{"text": user_prompt}]})
    payload = {
        "systemInstruction": {"parts": [{"text": build_chatbot_system_prompt()}]},
        "contents": conversation_contents,
        "generationConfig": {
            "temperature": 0.25,
            "topP": 0.9,
            "maxOutputTokens": 1800,
        },
    }

    request = Request(
        GEMINI_API_URL.format(model=model_name),
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=30) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="ignore")
        error_message = "The Gemini API returned an error."
        if error_body:
            try:
                parsed_error = json.loads(error_body)
                error_message = (
                    parsed_error.get("error", {}).get("message")
                    or error_message
                )
            except json.JSONDecodeError:
                error_message = error_body[:500]
        return {
            "ok": False,
            "status": exc.code,
            "message": f"Decipher AI could not answer right now: {error_message}",
        }
    except URLError as exc:
        return {
            "ok": False,
            "status": HTTPStatus.BAD_GATEWAY,
            "message": (
                "Decipher AI could not reach the Gemini API. Check internet access, firewall settings, "
                "and your API key configuration, then try again."
            ),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "status": HTTPStatus.INTERNAL_SERVER_ERROR,
            "message": f"Decipher AI hit an unexpected error: {exc}",
        }

    answer_text = extract_gemini_text(response_payload)
    if not answer_text:
        return {
            "ok": False,
            "status": HTTPStatus.BAD_GATEWAY,
            "message": (
                "Decipher AI did not receive a usable answer from Gemini. "
                "Please try a more specific dashboard question."
            ),
        }

    return {
        "ok": True,
        "status": HTTPStatus.OK,
        "message": answer_text,
        "model": model_name,
    }


def chart_html(file_name, cache_key, static_site=False):
    chart_path = resolve_output_path(BASE_DIR / file_name)
    if not chart_path.exists():
        return (
            '<div class="empty-state">Chart not found. Run '
            '<code>python retail_banking_analysis.py</code> to regenerate outputs.</div>'
        )
    src = f"assets/charts/{file_name}" if static_site else f"/charts/{file_name}?v={cache_key}"
    return f'<img src="{html.escape(src)}" alt="{html.escape(chart_path.stem.replace("_", " ").title())}">'


def risk_tone(segment):
    return {
        "Critical": "critical",
        "High": "high",
        "Moderate": "moderate",
        "Low": "low",
    }.get(str(segment), "neutral")


def build_filter_context(params, refresh_seconds):
    labels = [
        ("loan_type", format_label("loan_type")),
        ("leave_risk_segment", format_label("leave_risk_segment")),
        ("relationship_manager", format_label("relationship_manager")),
    ]
    pills = []
    for key, label in labels:
        value = get_filter_value(params, key)
        if value != "All":
            pills.append(
                f'<span class="context-pill"><strong>{html.escape(label)}:</strong> '
                f"{html.escape(format_label(value))}</span>"
            )

    if not pills:
        pills.append('<span class="context-pill context-pill-muted">Showing all customers</span>')

    pills.append(
        f'<span class="context-pill context-pill-muted">Refreshes every '
        f"{html.escape(str(refresh_seconds))} sec</span>"
    )
    return "".join(pills)


def build_metric_card(label, value, detail, tone="neutral"):
    return f"""
    <article class="metric-card metric-{html.escape(tone)}">
        <span class="metric-label">{html.escape(label)}</span>
        <strong class="metric-value">{html.escape(value)}</strong>
        <span class="metric-detail">{html.escape(detail)}</span>
    </article>
    """


def build_signal_card(title, metric, detail, tone="neutral"):
    return f"""
    <article class="signal-card signal-{html.escape(tone)}">
        <h3>{html.escape(title)}</h3>
        <strong class="signal-value">{html.escape(metric)}</strong>
        <p>{html.escape(detail)}</p>
    </article>
    """


def build_chart_card(chart_config, cache_key, static_site=False):
    return f"""
    <article class="chart-card">
        <div class="chart-copy">
            <span class="section-tag">Chart</span>
            <h3>{html.escape(chart_config['title'])}</h3>
            <p>{html.escape(chart_config['description'])}</p>
        </div>
        <div class="chart-frame">{chart_html(chart_config['file'], cache_key, static_site)}</div>
    </article>
    """


def build_download_card(file_name, static_site=False):
    descriptions = {
        "final_project_output.csv": "Main customer-level file used by this dashboard.",
        "summary_model_metrics.csv": "Simple summary of how the estimate was built and checked.",
        "summary_thresholds.csv": "Simple warning points where risk starts rising faster.",
        "summary_relationship_manager_effect.csv": "Comparison of customers with and without a dedicated banker.",
        "summary_retention_kpis.csv": "Simple portfolio-wide health checks used in the dashboard.",
        "summary_customer_retention_framework.csv": "How customers are grouped for early follow-up or direct follow-up.",
        "summary_credit_risk_integration.csv": "Where approval is needed before credit-related offers or changes.",
        "likely_to_leave_customers.csv": "High-priority customer list for quick follow-up.",
    }
    title = file_name.replace("_", " ").replace(".csv", "").title()
    return f"""
    <article class="download-card">
        <span class="section-tag">Export</span>
        <h3>{html.escape(title)}</h3>
        <p>{html.escape(descriptions.get(file_name, 'Latest generated dashboard export.'))}</p>
        <a class="download-link" href="{html.escape('assets/downloads/' + file_name if static_site else '/downloads/' + file_name)}">Download CSV</a>
    </article>
    """


def build_risk_mix_bar(filtered_df):
    if filtered_df.empty:
        return '<div class="empty-state">Risk mix appears here when the current view has data.</div>'

    counts = filtered_df["leave_risk_segment"].value_counts()
    total = int(len(filtered_df))
    segments = []
    legend = []
    for label in RISK_SEGMENT_ORDER:
        count = int(counts.get(label, 0))
        share = count / total if total else 0
        tone = risk_tone(label)
        if count > 0:
            segments.append(
                f'<div class="mix-segment mix-{html.escape(tone)}" style="width:{share * 100:.2f}%"></div>'
            )
        legend.append(
            f'<span class="mix-pill mix-pill-{html.escape(tone)}">{html.escape(format_label(label))} '
            f"{format_number(count)} ({format_percent(share, 1)})</span>"
        )
    return f'<div class="mix-bar">{"".join(segments)}</div><div class="mix-legend">{"".join(legend)}</div>'


def build_driver_chips(customer):
    chips = []
    credit_review_required = safe_flag_state(customer.get("credit_review_required"))
    has_relationship_manager = safe_flag_state(customer.get("has_relationship_manager"))
    complaint_count = safe_numeric(customer.get("complaint_count"))
    app_rating = safe_numeric(customer.get("app_rating_score"))
    resolution_days = safe_numeric(customer.get("resolution_days"))
    credit_score = safe_numeric(customer.get("credit_score"))
    service_quality = safe_numeric(customer.get("service_quality_index"))

    if credit_review_required is True:
        chips.append("Credit review required before offers")
    if has_relationship_manager is False:
        chips.append("No RM assigned")
    if pd.notna(complaint_count) and complaint_count > 0:
        chips.append(f"{format_number(complaint_count)} complaint(s) recorded")
    if pd.notna(resolution_days) and resolution_days >= 7:
        chips.append("Slow complaint resolution")
    if pd.notna(app_rating) and app_rating <= 3:
        chips.append("Weak app rating")
    if pd.notna(service_quality) and service_quality < 0.45:
        chips.append("Low service quality")
    if pd.notna(credit_score) and credit_score < 600:
        chips.append("Weak credit score")

    return chips or ["Review relationship and confirm current concern"]


def build_frontline_guidance(customer):
    focus_points = []
    safe_offers = []
    avoid_offers = []

    risk_label = str(customer.get("leave_risk_segment", "NA"))
    service_quality = safe_numeric(customer.get("service_quality_index"))
    resolution_days = safe_numeric(customer.get("resolution_days"))
    complaint_count = safe_numeric(customer.get("complaint_count"))
    app_rating = safe_numeric(customer.get("app_rating_score"))
    credit_score = safe_numeric(customer.get("credit_score"))
    product_count = safe_numeric(customer.get("product_count"))
    has_relationship_manager = safe_flag_state(customer.get("has_relationship_manager"))
    credit_review_required = safe_flag_state(customer.get("credit_review_required"))
    loan_type = str(customer.get("loan_type", "loan relationship")).strip() or "loan relationship"

    if risk_label == "Critical":
        focus_points.append("Contact the customer the same working day and confirm ownership.")
    if has_relationship_manager is False:
        focus_points.append("Assign a named RM or branch owner before discussing offers.")
    if (pd.notna(complaint_count) and complaint_count > 0) or (pd.notna(resolution_days) and resolution_days >= 7):
        focus_points.append("Start with service recovery and close any complaint loop first.")
    if pd.notna(app_rating) and app_rating <= 3:
        focus_points.append("Offer digital banking support for app, login, alerts, or payment friction.")
    if pd.notna(service_quality) and service_quality < 0.45:
        focus_points.append("Repair reliability and responsiveness before any product conversation.")
    if credit_review_required is True or (pd.notna(credit_score) and credit_score < 600):
        focus_points.append("Avoid aggressive credit offers until credit review is cleared.")
    if pd.notna(product_count) and product_count >= 5:
        focus_points.append("Protect the wider relationship and check whether one product is creating dissatisfaction.")

    safe_offers.extend(
        [
            "Service recovery callback with a committed owner.",
            "EMI date alignment or payment assistance discussion if policy allows.",
            "Digital banking support for app, alerts, login, and payment setup.",
            "Auto-pay setup to reduce repayment friction.",
        ]
    )

    if credit_review_required is True or (pd.notna(credit_score) and credit_score < 600):
        safe_offers.append("Account servicing and repayment support before any new credit discussion.")
        avoid_offers.extend(
            [
                "Fresh unsecured lending while credit profile is weak or under review.",
                "Aggressive top-up or cross-sell before credit review is complete.",
            ]
        )
    else:
        safe_offers.append("Rate review or tenure optimization if the customer is eligible.")
        safe_offers.append("Relevant bundled products only after service issues are resolved.")

    if "personal" in loan_type.lower():
        avoid_offers.append("Extra unsecured exposure unless affordability is clearly healthy.")

    avoid_offers.append("Generic product pitching before trust is repaired.")

    if not focus_points:
        focus_points.append("Start with a relationship review and confirm the customer's immediate concern.")

    return {
        "primary_focus": focus_points[0],
        "focus_points": focus_points[:6],
        "safe_offers": list(dict.fromkeys(safe_offers))[:6],
        "avoid_offers": list(dict.fromkeys(avoid_offers))[:4],
    }


def build_customer_ai_context(customer):
    guidance = build_frontline_guidance(customer)
    service_lines = [
        f"{SERVQUAL_LABELS[column]}: {format_number(safe_numeric(customer.get(column)), 2)}"
        for column in SERVQUAL_DIMENSION_COLUMNS
    ]
    return "\n".join(
        [
            "CUSTOMER 360 CONTEXT",
            f"Customer ID: {customer.get('customer_id', 'NA')}",
            f"Risk segment: {customer.get('leave_risk_segment', 'NA')}",
            f"Predicted leave probability: {format_percent(safe_numeric(customer.get('predicted_leave_probability')), 1)}",
            f"Retention score: {format_number(safe_numeric(customer.get('retention_score')), 2)}",
            f"Service quality index: {format_number(safe_numeric(customer.get('service_quality_index')), 2)}",
            f"App rating: {format_number(safe_numeric(customer.get('app_rating_score')), 1)}",
            f"Complaint count: {format_number(safe_numeric(customer.get('complaint_count')), 0)}",
            f"Resolution days: {format_number(safe_numeric(customer.get('resolution_days')), 1)}",
            f"Credit score: {format_number(safe_numeric(customer.get('credit_score')), 0)}",
            f"Product count: {format_number(safe_numeric(customer.get('product_count')), 0)}",
            f"Relationship manager assigned: {format_flag_status(customer.get('has_relationship_manager'))}",
            f"Credit review required: {format_flag_status(customer.get('credit_review_required'))}",
            "",
            "SERVICE DIAGNOSTICS",
            *service_lines,
            "",
            "FRONTLINE GUIDANCE",
            f"Primary focus: {guidance['primary_focus']}",
            "RM next steps: " + "; ".join(guidance["focus_points"]),
            "Safe offers/support: " + "; ".join(guidance["safe_offers"]),
            "Avoid: " + "; ".join(guidance["avoid_offers"]),
        ]
    )


def build_customer_360_payload(frame):
    payload = {}
    for _, customer in frame.head(15).iterrows():
        customer_id = str(customer.get("customer_id", "NA"))
        risk_label = str(customer.get("leave_risk_segment", "NA"))
        guidance = build_frontline_guidance(customer)
        driver_chips = build_driver_chips(customer)
        has_relationship_manager = safe_flag_state(customer.get("has_relationship_manager"))
        credit_review_required = safe_flag_state(customer.get("credit_review_required"))
        risk_explanation = (
            f"This customer is marked {format_label(risk_label)} because the estimated leave probability is "
            f"{format_percent(safe_numeric(customer.get('predicted_leave_probability')), 1)} and the priority score is "
            f"{format_number(safe_numeric(customer.get('risk_adjusted_retention_score')), 2)}. "
            f"Main warning signals: {', '.join(driver_chips)}."
        )
        payload[customer_id] = {
            "customer_id": customer_id,
            "risk_label": risk_label,
            "risk_tone": risk_tone(risk_label),
            "risk_explanation": risk_explanation,
            "driver_chips": driver_chips,
            "guidance": guidance,
            "ai_context": build_customer_ai_context(customer),
            "metrics": {
                "Predicted leave probability": format_percent(safe_numeric(customer.get("predicted_leave_probability")), 1),
                "Retention score": format_number(safe_numeric(customer.get("retention_score")), 2),
                "Service quality index": format_number(safe_numeric(customer.get("service_quality_index")), 2),
                "App rating": format_number(safe_numeric(customer.get("app_rating_score")), 1),
                "Complaint count": format_number(safe_numeric(customer.get("complaint_count")), 0),
                "Resolution days": format_number(safe_numeric(customer.get("resolution_days")), 1),
                "Credit score": format_number(safe_numeric(customer.get("credit_score")), 0),
                "Product count": format_number(safe_numeric(customer.get("product_count")), 0),
                "Relationship manager": "Assigned" if has_relationship_manager is True else ("Not assigned" if has_relationship_manager is False else "NA"),
                "Credit review required": "Yes" if credit_review_required is True else ("No" if credit_review_required is False else "NA"),
            },
            "summary": {
                "Loan type": str(customer.get("loan_type", "NA")),
                "Loan amount": format_currency(customer.get("loan_amount")),
                "EMI": format_currency(customer.get("emi")),
                "Income": format_currency(customer.get("income")),
                "Customer value tier": str(customer.get("customer_value_tier", "NA")),
                "Primary channel": str(customer.get("primary_channel", "NA")),
                "Retention lane": str(customer.get("retention_strategy_lane", "NA")),
                "Risk profile": str(customer.get("customer_risk_profile", "NA")),
                "Action owner": str(customer.get("retention_action_owner", "NA")),
                "Dashboard action": str(customer.get("recommended_action", "Review account")),
            },
        }
    return json.dumps(payload).replace("</", "<\\/")


def build_action_table(frame):
    if frame.empty:
        return '<div class="empty-state">No customers match the current filter.</div>'

    rows = []
    for _, row in frame.head(15).iterrows():
        risk_segment = str(row.get("leave_risk_segment", "NA"))
        relationship_manager_state = safe_flag_state(row.get("has_relationship_manager"))
        credit_review_state = safe_flag_state(row.get("credit_review_required"))
        coverage_state = "assigned" if relationship_manager_state is True else ("unassigned" if relationship_manager_state is False else "unknown")
        coverage_text = "Yes" if relationship_manager_state is True else ("No" if relationship_manager_state is False else "NA")
        approval_needed = "Yes" if credit_review_state is True else ("No" if credit_review_state is False else "NA")
        rows.append(
            f"""
            <tr>
                <td data-label="Customer"><button type="button" class="customer-link" data-customer-id="{html.escape(str(row.get("customer_id", "NA")))}">#{html.escape(str(row.get("customer_id", "NA")))}</button></td>
                <td data-label="Attention Level"><span class="risk-pill risk-{html.escape(risk_tone(risk_segment))}">{html.escape(format_label(risk_segment))}</span></td>
                <td data-label="Priority Score">{format_number(row.get("risk_adjusted_retention_score"), 2)}</td>
                <td data-label="Chance Of Leaving">{format_percent(row.get("predicted_leave_probability"), 1)}</td>
                <td data-label="Loan Type">{html.escape(str(row.get("loan_type", "NA")))}</td>
                <td data-label="Days To Solve Issue">{format_number(row.get("resolution_days"), 1)}</td>
                <td data-label="Dedicated Banker"><span class="coverage-pill coverage-{html.escape(coverage_state)}">{html.escape(coverage_text)}</span></td>
                <td data-label="Approval Needed">{html.escape(approval_needed)}</td>
                <td data-label="Suggested Next Step">{html.escape(str(row.get("recommended_action", "Review account")))}</td>
            </tr>
            """
        )

    return f"""
    <table class="action-table">
        <thead>
            <tr>
                <th>Customer</th>
                <th>Attention Level</th>
                <th>Priority Score</th>
                <th>Chance Of Leaving</th>
                <th>Loan Type</th>
                <th>Days To Solve Issue</th>
                <th>Dedicated Banker</th>
                <th>Approval Needed</th>
                <th>Suggested Next Step</th>
            </tr>
        </thead>
        <tbody>{''.join(rows)}</tbody>
    </table>
    """


def build_chatbot_panel(active_page, params, static_site=False):
    current_scope = build_filter_summary(params)
    if static_site:
        status_text = (
            "Decipher AI is available only in the live local dashboard because the static export does not "
            "include the Gemini API endpoint."
        )
        availability_class = "assistant-status-warning"
        disabled_attr = ' disabled="disabled"'
    elif get_gemini_api_key():
        status_text = f"Decipher AI is connected to Gemini via {get_gemini_model()} when you ask a question."
        availability_class = "assistant-status-ready"
        disabled_attr = ""
    else:
        status_text = (
            "Gemini is not configured yet. Set GEMINI_API_KEY, restart the dashboard, and then test Decipher AI."
        )
        availability_class = "assistant-status-warning"
        disabled_attr = ""

    example_buttons_html = "".join(
        f'<button class="prompt-chip" type="button" data-prompt="{html.escape(prompt)}"{disabled_attr}>{html.escape(prompt)}</button>'
        for prompt in CHATBOT_EXAMPLE_PROMPTS
    )

    return f"""
    <section class="panel dashboard-section dashboard-section-copilot" id="copilot">
        <div class="panel-head">
            <div>
                <span class="panel-label">Decipher AI</span>
                <h2>Ask the dashboard for grounded explanations</h2>
                <p>Use Decipher AI to explain KPIs and charts, summarize risk patterns, identify urgent intervention cases, and suggest retention actions based on this project's actual findings.</p>
            </div>
            <div class="panel-note">Current chat scope: <strong>{html.escape(current_scope)}</strong></div>
        </div>
        <div class="assistant-layout" data-active-page="{html.escape(active_page)}">
            <div class="assistant-intro">
                <span class="section-tag">Decipher AI Scope</span>
                <h3>What it knows</h3>
                <p>The assistant is grounded in the dashboard summaries, KPI files, threshold analysis, feature importance, relationship-manager comparisons, retention framework, credit-risk guardrails, and the report’s executive summary.</p>
                <div class="assistant-status {html.escape(availability_class)}">{html.escape(status_text)}</div>
                <div class="prompt-chip-grid">
                    {example_buttons_html}
                </div>
            </div>
            <div class="assistant-console">
                <div class="chat-log" id="chat-log">
                    <article class="chat-message chat-message-assistant">
                        <span class="chat-role">Decipher AI</span>
                        <p>Ask a question about leave risk, key drivers, chart meaning, urgent customers, or retention actions. I will answer from the dashboard context and keep the response business-focused.</p>
                    </article>
                </div>
                <form id="chat-form" class="chat-form" action="javascript:void(0)" method="post">
                    <label class="chat-label" for="chat-input">Your question</label>
                    <textarea id="chat-input" name="question" rows="4" placeholder="Example: Which customers need urgent intervention and why?"{disabled_attr}></textarea>
                    <div class="chat-actions">
                        <button class="button button-primary" id="chat-submit" type="button"{disabled_attr}>Ask Decipher AI</button>
                        <span class="chat-help">Preferred answer format: Insight, Why it matters, Recommended action.</span>
                    </div>
                </form>
            </div>
        </div>
    </section>
    """


def build_loan_type_summary(final_df):
    required_columns = {
        "loan_type",
        "customer_id",
        "retention_score",
        "predicted_leave_probability",
        "service_quality_index",
        "resolution_days",
        "app_rating_score",
        "has_relationship_manager",
        "leave_risk_segment",
    }
    if final_df.empty or not required_columns.issubset(final_df.columns):
        return pd.DataFrame()

    summary_df = (
        final_df.groupby("loan_type", dropna=False)
        .agg(
            customers=("customer_id", "count"),
            avg_service_quality_index=("service_quality_index", "mean"),
            avg_retention_score=("retention_score", "mean"),
            avg_predicted_leave_probability=("predicted_leave_probability", "mean"),
            avg_resolution_days=("resolution_days", "mean"),
            avg_app_rating_score=("app_rating_score", "mean"),
            relationship_manager_coverage=("has_relationship_manager", "mean"),
            high_critical_risk_share=(
                "leave_risk_segment",
                lambda values: values.isin(["High", "Critical"]).mean(),
            ),
        )
        .reset_index()
        .sort_values("avg_predicted_leave_probability", ascending=False)
    )
    return summary_df


def build_loan_type_story(loan_type_summary_df):
    if loan_type_summary_df.empty:
        return {
            "highest_risk_loan_type": "Waiting for latest file",
            "lowest_risk_loan_type": "Waiting for latest file",
            "risk_gap": np.nan,
            "highest_risk_share": np.nan,
            "lowest_risk_share": np.nan,
        }

    highest_risk = loan_type_summary_df.iloc[0]
    lowest_risk = loan_type_summary_df.iloc[-1]
    return {
        "highest_risk_loan_type": str(highest_risk["loan_type"]),
        "lowest_risk_loan_type": str(lowest_risk["loan_type"]),
        "risk_gap": float(
            pd.to_numeric(
                highest_risk["avg_predicted_leave_probability"], errors="coerce"
            )
            - pd.to_numeric(
                lowest_risk["avg_predicted_leave_probability"], errors="coerce"
            )
        ),
        "highest_risk_share": float(
            pd.to_numeric(highest_risk["high_critical_risk_share"], errors="coerce")
        ),
        "lowest_risk_share": float(
            pd.to_numeric(lowest_risk["high_critical_risk_share"], errors="coerce")
        ),
    }


def build_loan_type_table(loan_type_summary_df):
    if loan_type_summary_df.empty:
        return '<div class="empty-state">Loan-type comparisons appear here after the latest export is generated.</div>'

    display_df = loan_type_summary_df.copy()
    display_df["Loan Type"] = display_df["loan_type"].apply(format_label)
    display_df["Customers"] = display_df["customers"].apply(format_number)
    display_df["Avg Leave Probability"] = display_df[
        "avg_predicted_leave_probability"
    ].apply(lambda value: format_percent(value, 1))
    display_df["High And Critical Share"] = display_df[
        "high_critical_risk_share"
    ].apply(lambda value: format_percent(value, 1))
    display_df["Avg Retention Score"] = display_df["avg_retention_score"].apply(
        lambda value: format_number(value, 2)
    )
    display_df["Avg Service Quality"] = display_df[
        "avg_service_quality_index"
    ].apply(lambda value: format_percent(value, 1))
    display_df["Avg Resolution Days"] = display_df["avg_resolution_days"].apply(
        lambda value: format_number(value, 2)
    )
    display_df["Avg App Rating"] = display_df["avg_app_rating_score"].apply(
        lambda value: format_number(value, 2)
    )
    display_df["RM Coverage"] = display_df["relationship_manager_coverage"].apply(
        lambda value: format_percent(value, 1)
    )

    return dataframe_to_html(
        display_df[
            [
                "Loan Type",
                "Customers",
                "Avg Leave Probability",
                "High And Critical Share",
                "Avg Retention Score",
                "Avg Service Quality",
                "Avg Resolution Days",
                "Avg App Rating",
                "RM Coverage",
            ]
        ]
    )


def get_threshold_record(df, feature_name):
    if df.empty or "feature" not in df.columns:
        return None
    match = df[df["feature"] == feature_name]
    return None if match.empty else match.iloc[0]


def get_relationship_manager_record(df, status):
    if df.empty or "relationship_manager_status" not in df.columns:
        return None
    match = df[df["relationship_manager_status"] == status]
    return None if match.empty else match.iloc[0]


def get_kpi_record(df, metric_name):
    if df.empty or "metric_name" not in df.columns:
        return None
    match = df[df["metric_name"] == metric_name]
    return None if match.empty else match.iloc[0]


def format_kpi_value(value, unit):
    normalized_unit = str(unit).lower() if not pd.isna(unit) else ""
    if normalized_unit == "rate":
        return format_percent(value, 1)
    if normalized_unit == "count":
        return format_number(value, 2)
    if normalized_unit == "days":
        return format_number(value, 2)
    return format_number(value, 2)


def build_retention_kpi_cards(retention_kpi_df):
    if retention_kpi_df.empty:
        return '<div class="empty-state">Portfolio health cards appear here after the latest export is generated.</div>'

    selected_metrics = [
        ("retention_rate", "low"),
        ("high_critical_risk_share", "critical"),
        ("reactive_retention_share", "critical"),
        ("priority_retention_share", "high"),
        ("credit_review_required_rate", "high"),
        ("relationship_manager_coverage_rate", "low"),
        ("resolution_within_threshold_rate", "high"),
        ("app_rating_above_threshold_rate", "high"),
        ("average_product_count", "low"),
        ("five_plus_products_rate", "low"),
        ("proactive_retention_share", "low"),
    ]

    cards = []
    for metric_name, tone in selected_metrics:
        record = get_kpi_record(retention_kpi_df, metric_name)
        if record is None:
            continue
        cards.append(
            build_metric_card(
                plain_kpi_label(metric_name, str(record.get("metric_label", metric_name))),
                format_kpi_value(record.get("metric_value"), record.get("unit")),
                plain_kpi_detail(metric_name, str(record.get("interpretation", "Portfolio KPI summary."))),
                tone,
            )
        )

    return "".join(cards) if cards else '<div class="empty-state">Portfolio health cards appear here after the latest export is generated.</div>'


def build_kpi_reference_table(retention_kpi_df):
    if retention_kpi_df.empty:
        return dataframe_to_html(retention_kpi_df)

    display_df = retention_kpi_df.copy()
    display_df["Group"] = display_df["metric_group"]
    display_df["Measure"] = display_df.apply(
        lambda row: plain_kpi_label(
            row.get("metric_name"), str(row.get("metric_label", row.get("metric_name", "Metric")))
        ),
        axis=1,
    )
    display_df["Value"] = display_df.apply(
        lambda row: format_kpi_value(row.get("metric_value"), row.get("unit")),
        axis=1,
    )
    display_df["What It Means"] = display_df.apply(
        lambda row: plain_kpi_detail(
            row.get("metric_name"), str(row.get("interpretation", ""))
        ),
        axis=1,
    )
    return dataframe_to_html(display_df[["Group", "Measure", "Value", "What It Means"]])


def build_dashboard_css():
    css = """
    :root {
        --bg: __COLOR_SURFACE__;
        --panel: rgba(255, 252, 247, 0.92);
        --ink: __COLOR_TEXT__;
        --muted: #64707a;
        --line: rgba(20, 35, 45, 0.12);
        --primary: __COLOR_PRIMARY__;
        --secondary: __COLOR_SECONDARY__;
        --accent: __COLOR_ACCENT__;
        --alert: __COLOR_ALERT__;
        --gold: #cfaa46;
        --display-font: Georgia, "Times New Roman", serif;
        --body-font: "Trebuchet MS", "Aptos", Verdana, sans-serif;
    }
    * { box-sizing: border-box; }
    html { scroll-behavior: smooth; }
    body {
        margin: 0;
        color: var(--ink);
        font-family: var(--body-font);
        background:
            radial-gradient(circle at top left, rgba(217, 108, 6, 0.14), transparent 24%),
            radial-gradient(circle at top right, rgba(108, 138, 93, 0.12), transparent 26%),
            linear-gradient(180deg, #fbf6ed 0%, var(--bg) 46%, #f8efe1 100%);
    }
    .page {
        max-width: 1500px;
        margin: 0 auto;
        padding: 28px 28px 40px;
    }
    .hero {
        display: grid;
        grid-template-columns: minmax(0, 1.45fr) minmax(320px, 0.95fr);
        gap: 20px;
        padding: 32px;
        border-radius: 30px;
        color: #fff;
        background:
            radial-gradient(circle at 18% 18%, rgba(255,255,255,0.12), transparent 22%),
            linear-gradient(135deg, #143a4b 0%, #1d5567 46%, #c55e1b 120%);
        box-shadow: 0 24px 60px rgba(20, 35, 45, 0.18);
    }
    .hero h1, .panel h2, .panel h3, .download-card h3 {
        font-family: var(--display-font);
        letter-spacing: -0.03em;
    }
    .hero h1 {
        margin: 12px 0 14px;
        max-width: 10ch;
        font-size: clamp(2.4rem, 5vw, 4.2rem);
        line-height: 0.95;
    }
    .hero p {
        margin: 0;
        max-width: 760px;
        color: rgba(255,255,255,0.86);
        line-height: 1.7;
        font-size: 15px;
    }
    .hero p + p, .hero-meta, .jump-links, .context-pills, .download-grid, .mix-legend, .action-summary {
        margin-top: 16px;
    }
    .hero-meta, .jump-links, .context-pills, .download-grid, .mix-legend, .action-summary {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
    }
    .page-nav {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 10px;
        margin-top: 18px;
    }
    .page-link {
        min-height: 82px;
        padding: 12px;
        border-radius: 16px;
        border: 1px solid rgba(255,255,255,0.18);
        background: rgba(255,255,255,0.11);
        color: #fff;
        text-decoration: none;
        transition: background 0.18s ease, transform 0.18s ease, border-color 0.18s ease;
    }
    .page-link span {
        display: block;
        font-weight: 900;
        line-height: 1.2;
    }
    .page-link small {
        display: block;
        margin-top: 7px;
        color: rgba(255,255,255,0.72);
        font-size: 11px;
        line-height: 1.35;
    }
    .page-link:hover, .page-link-active {
        transform: translateY(-2px);
        background: rgba(255,255,255,0.22);
        border-color: rgba(255,255,255,0.36);
    }
    .page-link-active small { color: rgba(255,255,255,0.88); }
    .hero-panel, .panel, .table-card, .download-card, .metric-card, .signal-card, .chart-card {
        border: 1px solid var(--line);
        border-radius: 22px;
        background: var(--panel);
    }
    .hero-panel {
        padding: 20px;
        background: rgba(255,255,255,0.12);
        color: #fff;
    }
    .hero-panel p { color: rgba(255,255,255,0.78); }
    .hero-panel h2 { margin: 10px 0 6px; font-size: 28px; }
    .spotlight-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 12px;
        margin-top: 16px;
    }
    .spotlight-card {
        padding: 14px;
        border-radius: 18px;
        background: rgba(255,255,255,0.10);
        border: 1px solid rgba(255,255,255,0.12);
    }
    .spotlight-label, .section-tag, .metric-label, .panel-label {
        display: block;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        font-size: 11px;
        font-weight: 800;
    }
    .spotlight-label { color: rgba(255,255,255,0.70); }
    .spotlight-value {
        display: block;
        margin-top: 8px;
        font-size: 28px;
        font-weight: 800;
        line-height: 1;
    }
    .spotlight-detail { display: block; margin-top: 8px; color: rgba(255,255,255,0.76); font-size: 12px; }
    .section-tag, .panel-label { color: var(--muted); }
    .panel {
        margin-top: 18px;
        padding: 22px;
        box-shadow: 0 14px 34px rgba(20, 35, 45, 0.06);
    }
    .dashboard-section { display: none; }
    .page-overview .dashboard-section-overview,
    .page-health .dashboard-section-health,
    .page-drivers .dashboard-section-drivers,
    .page-charts .dashboard-section-charts,
    .page-actions .dashboard-section-actions,
    .page-copilot .dashboard-section-copilot,
    .page-details .dashboard-section-details,
    .page-downloads .dashboard-section-downloads {
        display: block;
    }
    .toolbar {
        position: sticky;
        top: 16px;
        z-index: 10;
        background: rgba(255, 252, 247, 0.97);
    }
    .panel-head {
        display: flex;
        justify-content: space-between;
        align-items: end;
        gap: 18px;
        margin-bottom: 18px;
    }
    .panel h2 { margin: 8px 0 0; font-size: 32px; }
    .panel p, .signal-card p, .chart-card p, .download-card p {
        margin: 10px 0 0;
        color: var(--muted);
        line-height: 1.6;
        font-size: 14px;
    }
    .panel-note {
        max-width: 320px;
        padding: 14px 16px;
        border-radius: 16px;
        background: #fff4e7;
        border: 1px solid rgba(217, 108, 6, 0.18);
        color: #77431a;
        line-height: 1.55;
        font-size: 13px;
    }
    .hero-chip, .jump-link, .context-pill, .mix-pill, .action-pill, .risk-pill, .coverage-pill {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        min-height: 34px;
        padding: 8px 12px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 800;
        text-decoration: none;
        white-space: nowrap;
    }
    .hero-chip, .jump-link {
        color: #fff;
        background: rgba(255,255,255,0.12);
        border: 1px solid rgba(255,255,255,0.16);
    }
    .context-pill, .mix-pill, .action-pill {
        color: var(--primary);
        background: rgba(15, 76, 92, 0.08);
        border: 1px solid rgba(15, 76, 92, 0.10);
    }
    .context-pill-muted, .mix-pill-moderate {
        color: var(--ink);
        background: rgba(20, 35, 45, 0.06);
        border-color: rgba(20, 35, 45, 0.08);
    }
    .filters, .metric-grid, .signal-grid, .chart-grid, .support-grid, .prompt-chip-grid {
        display: grid;
        gap: 14px;
    }
    .filters { grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); }
    .metric-grid { grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); }
    .signal-grid { grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); }
    .chart-grid { grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); }
    .support-grid { grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); }
    .prompt-chip-grid { grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); margin-top: 16px; }
    .summary-grid {
        display: grid;
        grid-template-columns: minmax(0, 1.5fr) minmax(300px, 0.9fr);
        gap: 16px;
        align-items: start;
    }
    .filter span {
        display: block;
        margin-bottom: 8px;
        color: var(--muted);
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 700;
    }
    .filter select, .filter input {
        width: 100%;
        min-height: 46px;
        padding: 11px 13px;
        border-radius: 14px;
        border: 1px solid var(--line);
        background: rgba(255,255,255,0.94);
        color: var(--ink);
        font: inherit;
    }
    .filter-actions {
        display: flex;
        align-items: end;
        gap: 10px;
    }
    .button, .download-link {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        min-height: 46px;
        padding: 11px 16px;
        border: 0;
        border-radius: 14px;
        cursor: pointer;
        text-decoration: none;
        font-size: 14px;
        font-weight: 800;
    }
    .button-primary, .download-link {
        color: #fff;
        background: linear-gradient(135deg, var(--primary), #1c6176);
    }
    .button-secondary { background: rgba(20, 35, 45, 0.07); color: var(--ink); }
    .prompt-chip {
        min-height: 52px;
        padding: 12px 14px;
        border-radius: 16px;
        border: 1px solid rgba(15, 76, 92, 0.14);
        background: rgba(15, 76, 92, 0.05);
        color: var(--primary);
        font: inherit;
        font-size: 13px;
        font-weight: 800;
        text-align: left;
        cursor: pointer;
    }
    .metric-card, .signal-card, .table-card, .download-card, .chart-card, .summary-card {
        position: relative;
        overflow: hidden;
        padding: 18px;
    }
    .metric-card::before, .signal-card::before {
        content: "";
        position: absolute;
        inset: 0 auto auto 0;
        width: 100%;
        height: 4px;
        background: rgba(15, 76, 92, 0.18);
    }
    .metric-critical::before, .signal-critical::before { background: linear-gradient(90deg, var(--alert), #cb736a); }
    .metric-high::before, .signal-high::before { background: linear-gradient(90deg, var(--secondary), #ebaf77); }
    .metric-low::before, .signal-low::before { background: linear-gradient(90deg, var(--accent), #8baa7a); }
    .metric-value, .signal-value {
        display: block;
        margin-top: 14px;
        font-family: var(--display-font);
        font-size: 32px;
        line-height: 0.96;
        letter-spacing: -0.04em;
        color: var(--primary);
    }
    .metric-detail { display: block; margin-top: 10px; color: var(--muted); line-height: 1.5; font-size: 13px; }
    .summary-card {
        border-radius: 22px;
        border: 1px solid var(--line);
        background: linear-gradient(180deg, rgba(15, 76, 92, 0.04), rgba(255,255,255,0.92));
    }
    .summary-card h3 { margin: 8px 0 0; font-size: 28px; }
    .mix-bar {
        display: flex;
        overflow: hidden;
        height: 16px;
        margin-top: 16px;
        border-radius: 999px;
        background: rgba(20, 35, 45, 0.08);
    }
    .mix-segment { height: 100%; }
    .mix-critical { background: var(--alert); }
    .mix-high { background: var(--secondary); }
    .mix-moderate { background: var(--gold); }
    .mix-low { background: var(--accent); }
    .chart-frame, .table-wrap {
        margin-top: 14px;
        overflow: auto;
        border-radius: 16px;
        border: 1px solid var(--line);
        background: rgba(255,255,255,0.84);
    }
    .chart-frame img { display: block; width: 100%; height: auto; }
    .data-table, .action-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 13px;
    }
    .data-table th, .data-table td, .action-table th, .action-table td {
        padding: 11px 12px;
        text-align: left;
        border-bottom: 1px solid var(--line);
        vertical-align: top;
    }
    .data-table th, .action-table th {
        position: sticky;
        top: 0;
        background: #f3ede3;
        font-size: 12px;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    .customer-link {
        border: 0;
        padding: 0;
        background: transparent;
        color: var(--primary);
        cursor: pointer;
        font: inherit;
        font-weight: 900;
        text-decoration: underline;
        text-underline-offset: 3px;
    }
    .customer-link:hover, .customer-link:focus {
        color: var(--secondary);
        outline: none;
    }
    .risk-critical, .coverage-unassigned { color: var(--alert); background: rgba(159, 47, 40, 0.14); }
    .risk-high { color: #9a4d08; background: rgba(217, 108, 6, 0.14); }
    .risk-moderate, .coverage-unknown { color: #80610c; background: rgba(207, 170, 70, 0.18); }
    .risk-low, .coverage-assigned { color: #426645; background: rgba(108, 138, 93, 0.14); }
    .customer-modal {
        position: fixed;
        inset: 0;
        z-index: 100;
        display: none;
        align-items: center;
        justify-content: center;
        padding: 22px;
        background: rgba(10, 22, 28, 0.58);
        backdrop-filter: blur(7px);
    }
    .customer-modal.open { display: flex; }
    .customer-modal-card {
        width: min(1120px, 100%);
        max-height: min(90vh, 920px);
        overflow: auto;
        border-radius: 26px;
        border: 1px solid rgba(255,255,255,0.42);
        background:
            radial-gradient(circle at top right, rgba(217, 108, 6, 0.10), transparent 30%),
            linear-gradient(180deg, #fffdf8, #f8efe3);
        box-shadow: 0 34px 90px rgba(10, 22, 28, 0.35);
    }
    .customer-modal-head {
        position: sticky;
        top: 0;
        z-index: 2;
        display: flex;
        justify-content: space-between;
        gap: 18px;
        padding: 22px;
        border-bottom: 1px solid var(--line);
        background: rgba(255, 252, 247, 0.96);
    }
    .customer-modal-head h2 {
        margin: 6px 0 0;
        font-family: var(--display-font);
        font-size: clamp(2rem, 4vw, 3rem);
        letter-spacing: -0.04em;
    }
    .modal-close {
        align-self: flex-start;
        min-height: 42px;
        border: 0;
        border-radius: 999px;
        padding: 10px 15px;
        background: rgba(20, 35, 45, 0.08);
        color: var(--ink);
        cursor: pointer;
        font: inherit;
        font-weight: 900;
    }
    .customer-modal-body {
        display: grid;
        grid-template-columns: minmax(0, 0.95fr) minmax(0, 1.15fr);
        gap: 16px;
        padding: 18px 22px 24px;
    }
    .customer-360-section {
        border: 1px solid var(--line);
        border-radius: 20px;
        padding: 16px;
        background: rgba(255,255,255,0.78);
    }
    .customer-360-section h3 {
        margin: 0 0 10px;
        font-family: var(--display-font);
        font-size: 24px;
        letter-spacing: -0.03em;
    }
    .customer-360-section p {
        margin: 0;
        color: var(--muted);
        line-height: 1.6;
        font-size: 14px;
    }
    .customer-360-full { grid-column: 1 / -1; }
    .customer-360-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 10px;
    }
    .customer-360-stat {
        padding: 12px;
        border-radius: 15px;
        background: rgba(15, 76, 92, 0.055);
        border: 1px solid rgba(15, 76, 92, 0.08);
    }
    .customer-360-stat span {
        display: block;
        color: var(--muted);
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }
    .customer-360-stat strong {
        display: block;
        margin-top: 6px;
        font-size: 16px;
        color: var(--ink);
    }
    .driver-chip-row {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
    }
    .driver-chip {
        display: inline-flex;
        align-items: center;
        min-height: 32px;
        padding: 7px 10px;
        border-radius: 999px;
        color: var(--primary);
        background: rgba(15, 76, 92, 0.08);
        border: 1px solid rgba(15, 76, 92, 0.10);
        font-size: 12px;
        font-weight: 850;
    }
    .customer-360-list {
        margin: 0;
        padding-left: 18px;
        color: var(--muted);
        line-height: 1.55;
        font-size: 14px;
    }
    .customer-360-list li + li { margin-top: 8px; }
    .risk-modal-badge {
        display: inline-flex;
        width: fit-content;
        margin-top: 10px;
    }
    .download-grid { gap: 14px; }
    .download-card { flex: 1 1 240px; min-width: 240px; }
    .empty-state {
        padding: 18px;
        border-radius: 16px;
        border: 1px dashed rgba(20, 35, 45, 0.20);
        background: rgba(255,255,255,0.58);
        color: var(--muted);
    }
    .assistant-layout {
        display: grid;
        grid-template-columns: minmax(280px, 0.9fr) minmax(0, 1.3fr);
        gap: 18px;
        align-items: start;
    }
    .assistant-intro, .assistant-console {
        border-radius: 22px;
        border: 1px solid var(--line);
        background: linear-gradient(180deg, rgba(255,255,255,0.96), rgba(245, 239, 227, 0.9));
        padding: 18px;
    }
    .assistant-intro h3 {
        margin: 8px 0 0;
        font-size: 28px;
        font-family: var(--display-font);
    }
    .assistant-status {
        margin-top: 14px;
        padding: 12px 14px;
        border-radius: 16px;
        font-size: 13px;
        line-height: 1.55;
        font-weight: 700;
    }
    .assistant-status-ready {
        color: #215f37;
        background: rgba(108, 138, 93, 0.14);
        border: 1px solid rgba(108, 138, 93, 0.2);
    }
    .assistant-status-warning {
        color: #77431a;
        background: #fff4e7;
        border: 1px solid rgba(217, 108, 6, 0.18);
    }
    .chat-log {
        min-height: 380px;
        overflow: visible;
        padding-right: 4px;
    }
    .chat-message {
        margin-bottom: 14px;
        padding: 14px 16px;
        border-radius: 18px;
        border: 1px solid var(--line);
        overflow: visible;
        white-space: pre-wrap;
        overflow-wrap: anywhere;
        line-height: 1.6;
        font-size: 14px;
    }
    .chat-message p {
        margin: 8px 0 0;
        white-space: pre-wrap;
        overflow-wrap: anywhere;
    }
    .chat-message .chat-section-title {
        display: block;
        margin-top: 12px;
        color: var(--primary);
        font-weight: 900;
    }
    .chat-message .chat-section-title:first-of-type { margin-top: 8px; }
    .chat-message ul {
        margin: 8px 0 0;
        padding-left: 20px;
        color: var(--ink);
        line-height: 1.6;
    }
    .chat-message li + li { margin-top: 5px; }
    .chat-message-user {
        background: rgba(15, 76, 92, 0.07);
        margin-left: 42px;
    }
    .chat-message-assistant {
        background: rgba(255,255,255,0.96);
        margin-right: 42px;
    }
    .chat-message-system {
        background: rgba(217, 108, 6, 0.08);
        border-style: dashed;
    }
    .chat-role {
        display: block;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        font-size: 11px;
        font-weight: 800;
        color: var(--muted);
    }
    .chat-form { margin-top: 16px; }
    .chat-label {
        display: block;
        margin-bottom: 8px;
        color: var(--muted);
        font-size: 12px;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        font-weight: 700;
    }
    .chat-form textarea {
        width: 100%;
        resize: vertical;
        min-height: 128px;
        padding: 14px;
        border-radius: 16px;
        border: 1px solid var(--line);
        background: rgba(255,255,255,0.98);
        color: var(--ink);
        font: inherit;
    }
    .chat-actions {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 12px;
        margin-top: 12px;
    }
    .chat-help {
        color: var(--muted);
        font-size: 12px;
        line-height: 1.5;
    }
    code {
        padding: 2px 6px;
        border-radius: 8px;
        background: rgba(20, 35, 45, 0.08);
        font-family: Consolas, "Courier New", monospace;
    }
    @media (max-width: 1100px) {
        .hero, .summary-grid, .assistant-layout, .customer-modal-body { grid-template-columns: 1fr; }
        .customer-360-full { grid-column: auto; }
        .toolbar { position: static; }
    }
    @media (max-width: 760px) {
        .page { padding: 16px 16px 32px; }
        .hero, .panel { padding: 18px; }
        .panel-head { flex-direction: column; align-items: start; }
        .spotlight-grid { grid-template-columns: 1fr; }
        .filter-actions { grid-column: 1 / -1; display: grid; grid-template-columns: 1fr 1fr; }
        .button, .download-link { width: 100%; }
        .chat-actions { align-items: stretch; flex-direction: column; }
        .chat-message-user, .chat-message-assistant { margin-left: 0; margin-right: 0; }
        .customer-modal { padding: 10px; align-items: stretch; }
        .customer-modal-card { max-height: 96vh; }
        .customer-modal-head { flex-direction: column; padding: 16px; }
        .customer-modal-body { padding: 14px; }
        .customer-360-grid { grid-template-columns: 1fr; }
        .action-table thead { display: none; }
        .action-table, .action-table tbody, .action-table tr, .action-table td {
            display: block;
            width: 100%;
        }
        .action-table tr { padding: 12px; border-bottom: 1px solid var(--line); }
        .action-table td { padding: 8px 0; border-bottom: 0; }
        .action-table td::before {
            content: attr(data-label);
            display: block;
            margin-bottom: 4px;
            color: var(--muted);
            font-size: 11px;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            font-weight: 700;
        }
    }
    /* === INTERACTIONS & ANIMATIONS === */
    .metric-card, .signal-card, .chart-card, .download-card {
        transition: transform 0.22s ease, box-shadow 0.22s ease;
    }
    .metric-card:hover, .signal-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 24px 48px rgba(20, 35, 45, 0.14);
    }
    .chart-card:hover, .download-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 18px 36px rgba(20, 35, 45, 0.11);
    }
    @keyframes _slideUp {
        from { opacity: 0; transform: translateY(14px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    .metric-grid .metric-card { animation: _slideUp 0.45s ease both; }
    .metric-grid .metric-card:nth-child(1)  { animation-delay: 0.05s; }
    .metric-grid .metric-card:nth-child(2)  { animation-delay: 0.10s; }
    .metric-grid .metric-card:nth-child(3)  { animation-delay: 0.15s; }
    .metric-grid .metric-card:nth-child(4)  { animation-delay: 0.20s; }
    .metric-grid .metric-card:nth-child(5)  { animation-delay: 0.25s; }
    .metric-grid .metric-card:nth-child(6)  { animation-delay: 0.30s; }
    .metric-grid .metric-card:nth-child(7)  { animation-delay: 0.35s; }
    .metric-grid .metric-card:nth-child(8)  { animation-delay: 0.40s; }
    .metric-grid .metric-card:nth-child(9)  { animation-delay: 0.45s; }
    .metric-grid .metric-card:nth-child(10) { animation-delay: 0.50s; }
    .metric-grid .metric-card:nth-child(11) { animation-delay: 0.55s; }
    .signal-grid .signal-card { animation: _slideUp 0.45s ease both; }
    .signal-grid .signal-card:nth-child(1) { animation-delay: 0.05s; }
    .signal-grid .signal-card:nth-child(2) { animation-delay: 0.12s; }
    .signal-grid .signal-card:nth-child(3) { animation-delay: 0.19s; }
    .signal-grid .signal-card:nth-child(4) { animation-delay: 0.26s; }
    .signal-grid .signal-card:nth-child(5) { animation-delay: 0.33s; }
    .signal-grid .signal-card:nth-child(6) { animation-delay: 0.40s; }
    @keyframes _critPulse {
        0%, 100% { box-shadow: 0 0 0 0 rgba(159, 47, 40, 0); }
        50%       { box-shadow: 0 0 0 8px rgba(159, 47, 40, 0.16); }
    }
    .metric-critical, .signal-critical {
        animation: _critPulse 2.6s ease-in-out infinite, _slideUp 0.45s ease both;
    }
    .button, .download-link {
        transition: transform 0.18s ease, box-shadow 0.18s ease, filter 0.18s ease;
    }
    .button-primary:hover, .download-link:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 22px rgba(15, 76, 92, 0.30);
        filter: brightness(1.08);
    }
    .button-secondary:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 14px rgba(20, 35, 45, 0.10);
    }
    .prompt-chip:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 24px rgba(15, 76, 92, 0.12);
        background: rgba(15, 76, 92, 0.08);
    }
    .prompt-chip:disabled, .chat-form textarea:disabled, .button:disabled {
        cursor: not-allowed;
        opacity: 0.65;
    }
    .jump-link { transition: background 0.18s ease, transform 0.18s ease; }
    .jump-link:hover { background: rgba(255,255,255,0.22); transform: translateY(-2px); }
    .spotlight-card { transition: background 0.20s ease, transform 0.20s ease; }
    .spotlight-card:hover { background: rgba(255,255,255,0.20); transform: translateY(-2px); }
    .action-table tbody tr { transition: background 0.14s ease; }
    .action-table tbody tr:hover { background: rgba(15, 76, 92, 0.05); }
    @keyframes _barGrow {
        from { transform: scaleX(0); }
        to   { transform: scaleX(1); }
    }
    .mix-segment { transform-origin: left center; animation: _barGrow 1.1s cubic-bezier(0.22,1,0.36,1) both; }
    .mix-segment:nth-child(1) { animation-delay: 0.10s; }
    .mix-segment:nth-child(2) { animation-delay: 0.30s; }
    .mix-segment:nth-child(3) { animation-delay: 0.50s; }
    .mix-segment:nth-child(4) { animation-delay: 0.70s; }
    @keyframes _livePulse {
        0%, 100% { opacity: 1; box-shadow: 0 0 0 0 rgba(74,222,128,0.7); }
        50%       { opacity: 0.7; box-shadow: 0 0 0 5px rgba(74,222,128,0); }
    }
    .live-dot {
        display: inline-block; width: 7px; height: 7px; border-radius: 50%;
        background: #4ade80; margin-right: 5px; vertical-align: middle;
        animation: _livePulse 1.8s ease-in-out infinite;
    }
    .hero-chip { transition: background 0.18s ease; }
    .hero-chip:hover { background: rgba(255,255,255,0.20); }
    .filter select:focus, .filter input:focus { outline: 2px solid var(--primary); outline-offset: 2px; }
    .download-link::before { content: "↓  "; font-weight: 900; }
    .risk-pill, .coverage-pill { transition: transform 0.15s ease; }
    .risk-pill:hover, .coverage-pill:hover { transform: scale(1.06); }
    """
    replacements = {
        "__COLOR_SURFACE__": COLOR_SURFACE,
        "__COLOR_TEXT__": COLOR_TEXT,
        "__COLOR_PRIMARY__": COLOR_PRIMARY,
        "__COLOR_SECONDARY__": COLOR_SECONDARY,
        "__COLOR_ACCENT__": COLOR_ACCENT,
        "__COLOR_ALERT__": COLOR_ALERT,
    }
    for token, value in replacements.items():
        css = css.replace(token, value)
    return css


def build_dashboard_js():
    return """
(function(){
    var chatHistory = [];
    var CHAT_STORAGE_KEY = 'retailBankingDecipherAIHistory';

    function countUp(el){
        if(el.dataset.counted) return;
        el.dataset.counted='1';
        var raw=el.textContent.trim();
        var m=raw.match(/^([\d,]+(?:\.\d+)?)(\s*%|\s*[^%\d].*)?\s*$/);
        if(!m) return;
        var numStr=m[1].replace(/,/g,'');
        var suffix=m[2]||'';
        var target=parseFloat(numStr);
        if(isNaN(target)||target<=0) return;
        var decimals=(numStr.split('.')[1]||'').length;
        var hasComma=m[1].indexOf(',')>=0;
        var dur=900, t0=null;
        function step(ts){
            if(!t0) t0=ts;
            var p=Math.min((ts-t0)/dur,1);
            var ease=1-Math.pow(1-p,3);
            var val=ease*target;
            var str=val.toFixed(decimals);
            if(hasComma&&val>=1000) str=str.replace(/\\B(?=(\\d{3})+(?!\\d))/g,',');
            el.textContent=str+suffix;
            if(p<1) requestAnimationFrame(step);
        }
        requestAnimationFrame(step);
    }
    function revealOnScroll(){
        var panels=document.querySelectorAll('.panel:not(.toolbar)');
        if(!('IntersectionObserver' in window)){
            panels.forEach(function(p){p.style.opacity='1';p.style.transform='none';});
            return;
        }
        var io=new IntersectionObserver(function(entries){
            entries.forEach(function(e){
                if(e.isIntersecting){
                    e.target.style.opacity='1';
                    e.target.style.transform='none';
                    io.unobserve(e.target);
                    e.target.querySelectorAll('.metric-value').forEach(countUp);
                }
            });
        },{threshold:0.06});
        panels.forEach(function(p){
            p.style.opacity='0';
            p.style.transform='translateY(18px)';
            p.style.transition='opacity 0.55s ease, transform 0.55s ease';
            io.observe(p);
        });
    }
    function escapeHtml(text){
        return String(text)
            .replace(/&/g,'&amp;')
            .replace(/</g,'&lt;')
            .replace(/>/g,'&gt;')
            .replace(/\"/g,'&quot;')
            .replace(/'/g,'&#39;');
    }
    function formatChatContent(content){
        var lines=String(content || '').replace(/\\r\\n/g,'\\n').split('\\n');
        var html='';
        var listOpen=false;
        function closeList(){
            if(listOpen){
                html+='</ul>';
                listOpen=false;
            }
        }
        lines.forEach(function(rawLine){
            var line=rawLine.trim();
            if(!line){
                closeList();
                return;
            }
            if(/^(Insight|Why it matters|Recommended action|Suggested approach|What to say|What to offer|What to avoid):$/i.test(line)){
                closeList();
                html+='<span class=\"chat-section-title\">'+escapeHtml(line)+'</span>';
                return;
            }
            if(/^[-*]\\s+/.test(line)){
                if(!listOpen){
                    html+='<ul>';
                    listOpen=true;
                }
                html+='<li>'+escapeHtml(line.replace(/^[-*]\\s+/,''))+'</li>';
                return;
            }
            closeList();
            html+='<p>'+escapeHtml(line)+'</p>';
        });
        closeList();
        return html || '<p>NA</p>';
    }
    function renderMessage(role, content){
        var log=document.getElementById('chat-log');
        if(!log) return;
        var article=document.createElement('article');
        article.className='chat-message chat-message-'+role;
        var label=role==='user' ? 'You' : (role==='assistant' ? 'Decipher AI' : 'System');
        article.innerHTML='<span class=\"chat-role\">'+escapeHtml(label)+'</span>'+formatChatContent(content);
        log.appendChild(article);
        log.scrollTop=log.scrollHeight;
    }
    function saveChatHistory(){
        try{
            window.sessionStorage.setItem(CHAT_STORAGE_KEY, JSON.stringify(chatHistory.slice(-12)));
        }catch(err){}
    }
    function restoreChatHistory(){
        var log=document.getElementById('chat-log');
        if(!log || !window.sessionStorage) return;
        try{
            var saved=JSON.parse(window.sessionStorage.getItem(CHAT_STORAGE_KEY) || '[]');
            if(!Array.isArray(saved) || !saved.length) return;
            chatHistory=saved.filter(function(item){
                return item && (item.role==='user' || item.role==='assistant' || item.role==='system') && item.content;
            }).slice(-12);
            if(!chatHistory.length) return;
            log.innerHTML='';
            chatHistory.forEach(function(item){
                renderMessage(item.role,item.content);
            });
        }catch(err){}
    }
    function setSubmitting(isSubmitting){
        var form=document.getElementById('chat-form');
        if(!form) return;
        var button=document.getElementById('chat-submit');
        var textarea=document.getElementById('chat-input');
        if(button){
            button.disabled=isSubmitting || button.hasAttribute('data-hard-disabled');
            button.textContent=isSubmitting ? 'Thinking...' : 'Ask Decipher AI';
        }
        if(textarea && !textarea.hasAttribute('data-hard-disabled')){
            textarea.disabled=isSubmitting;
        }
    }
    function extractFilters(){
        var params={};
        var search=new URLSearchParams(window.location.search);
        search.forEach(function(value,key){ params[key]=value; });
        return params;
    }
    function submitQuestion(question){
        if(!question) return;
        renderMessage('user',question);
        chatHistory.push({role:'user',content:question});
        saveChatHistory();
        setSubmitting(true);
        fetch('/api/chat',{
            method:'POST',
            headers:{'Content-Type':'application/json'},
            body:JSON.stringify({
                question:question,
                history:chatHistory.slice(-6),
                params:extractFilters(),
                path:window.location.pathname
            })
        }).then(function(response){
            return response.json().catch(function(){ return {}; }).then(function(payload){
                return {ok:response.ok,payload:payload};
            });
        }).then(function(result){
            var message=result.payload && result.payload.message ? result.payload.message : 'Decipher AI did not return a response.';
            var role=result.ok ? 'assistant' : 'system';
            renderMessage(role,message);
            if(result.ok){
                chatHistory.push({role:'assistant',content:message});
            }else{
                chatHistory.push({role:'system',content:message});
            }
            saveChatHistory();
        }).catch(function(){
            var message='Decipher AI could not be reached from the dashboard. Please try again.';
            renderMessage('system',message);
            chatHistory.push({role:'system',content:message});
            saveChatHistory();
        }).finally(function(){
            setSubmitting(false);
        });
    }
    function setupChatbot(){
        var form=document.getElementById('chat-form');
        var textarea=document.getElementById('chat-input');
        if(!form || !textarea) return;
        if(textarea.disabled){
            textarea.setAttribute('data-hard-disabled','1');
        }
        var submitButton=document.getElementById('chat-submit');
        if(submitButton && submitButton.disabled){
            submitButton.setAttribute('data-hard-disabled','1');
        }
        function handleChatSubmit(event){
            if(event){
                event.preventDefault();
            }
            var question=textarea.value.trim();
            if(!question) return;
            textarea.value='';
            submitQuestion(question);
        }
        form.addEventListener('submit',handleChatSubmit);
        if(submitButton){
            submitButton.addEventListener('click',handleChatSubmit);
        }
        document.querySelectorAll('.prompt-chip').forEach(function(button){
            button.addEventListener('click',function(){
                var prompt=button.getAttribute('data-prompt') || '';
                if(!prompt) return;
                textarea.value=prompt;
                textarea.focus();
            });
        });
    }
    function parseCustomer360Data(){
        var dataNode=document.getElementById('customer-360-data');
        if(!dataNode) return {};
        try{
            return JSON.parse(dataNode.textContent || '{}');
        }catch(err){
            return {};
        }
    }
    function renderKeyValueGrid(id, values){
        var host=document.getElementById(id);
        if(!host) return;
        var html='';
        Object.keys(values || {}).forEach(function(label){
            html+='<div class=\"customer-360-stat\"><span>'+escapeHtml(label)+'</span><strong>'+escapeHtml(values[label] || 'NA')+'</strong></div>';
        });
        host.innerHTML=html || '<div class=\"empty-state\">No details available.</div>';
    }
    function renderCustomerList(id, items){
        var host=document.getElementById(id);
        if(!host) return;
        if(!items || !items.length){
            host.innerHTML='<li>NA</li>';
            return;
        }
        host.innerHTML=items.map(function(item){ return '<li>'+escapeHtml(item)+'</li>'; }).join('');
    }
    function renderDriverChips(items){
        var host=document.getElementById('customer-360-drivers');
        if(!host) return;
        if(!items || !items.length){
            host.innerHTML='<span class=\"driver-chip\">Review relationship</span>';
            return;
        }
        host.innerHTML=items.map(function(item){ return '<span class=\"driver-chip\">'+escapeHtml(item)+'</span>'; }).join('');
    }
    function setupCustomer360(){
        var modal=document.getElementById('customer-360-modal');
        var closeButton=document.getElementById('customer-360-close');
        var customerData=parseCustomer360Data();
        if(!modal) return;

        function setText(id, value){
            var el=document.getElementById(id);
            if(el) el.textContent=value || 'NA';
        }
        function openCustomer(customerId){
            var customer=customerData[String(customerId)];
            if(!customer){
                return;
            }
            setText('customer-360-title','Customer #'+customer.customer_id);
            setText('customer-360-risk-copy',customer.risk_explanation);
            var badge=document.getElementById('customer-360-risk-badge');
            if(badge){
                badge.className='risk-pill risk-modal-badge risk-'+(customer.risk_tone || 'neutral');
                badge.textContent=customer.risk_label || 'NA';
            }
            renderKeyValueGrid('customer-360-summary',customer.summary);
            renderKeyValueGrid('customer-360-metrics',customer.metrics);
            renderDriverChips(customer.driver_chips);
            renderCustomerList('customer-360-next-steps',customer.guidance && customer.guidance.focus_points);
            renderCustomerList('customer-360-safe-offers',customer.guidance && customer.guidance.safe_offers);
            renderCustomerList('customer-360-avoid-offers',customer.guidance && customer.guidance.avoid_offers);
            modal.classList.add('open');
            modal.setAttribute('aria-hidden','false');
            document.body.style.overflow='hidden';
            if(closeButton) closeButton.focus();
        }
        function closeCustomer(){
            modal.classList.remove('open');
            modal.setAttribute('aria-hidden','true');
            document.body.style.overflow='';
        }

        document.querySelectorAll('.customer-link').forEach(function(button){
            button.addEventListener('click',function(event){
                event.preventDefault();
                openCustomer(button.getAttribute('data-customer-id'));
            });
        });
        if(closeButton){
            closeButton.addEventListener('click',closeCustomer);
        }
        modal.addEventListener('click',function(event){
            if(event.target===modal){
                closeCustomer();
            }
        });
        document.addEventListener('keydown',function(event){
            if(event.key==='Escape' && modal.classList.contains('open')){
                closeCustomer();
            }
        });
    }
    document.addEventListener('DOMContentLoaded',function(){
        document.querySelectorAll('.spotlight-value').forEach(countUp);
        revealOnScroll();
        restoreChatHistory();
        setupChatbot();
        setupCustomer360();
    });
})();
"""


def build_waiting_html(refresh_seconds):
    return f"""
    <html>
    <head>
        <meta charset="utf-8">
        <meta http-equiv="refresh" content="{refresh_seconds}">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>Retail Banking Customer Dashboard</title>
        <style>{build_dashboard_css()}</style>
    </head>
    <body>
        <div class="page">
            <section class="hero">
                <div class="hero-copy">
                    <span class="section-tag">Dashboard Status</span>
                    <h1>Data Files Needed</h1>
                    <p>The page is ready, but the generated CSV files and charts have not been found yet.</p>
                    <p>Run <code>python retail_banking_analysis.py</code> first, then refresh the page.</p>
                </div>
                <aside class="hero-panel">
                    <span class="section-tag">Next Step</span>
                    <h2>Create the latest files</h2>
                    <p>The dashboard will read the newest files as soon as they are available.</p>
                </aside>
            </section>
        </div>
    </body>
    </html>
    """


def build_dashboard_html(params, active_page="overview", static_site=False):
    active_page = normalize_page(active_page)
    try:
        refresh_seconds = max(5, int(params.get("refresh", ["300"])[0]))
    except (TypeError, ValueError):
        refresh_seconds = 300
    refresh_meta_tag = (
        "" if active_page == "copilot" else f'<meta http-equiv="refresh" content="{refresh_seconds}">'
    )

    data = load_data()
    final_df = ensure_dashboard_columns(data["final"])
    metrics_df = data["metrics"]

    if final_df.empty or metrics_df.empty:
        return build_waiting_html(refresh_seconds)

    metrics = metrics_df.iloc[0]
    filtered_df = filter_dataframe(final_df, params)
    portfolio_summary = summarize_portfolio(filtered_df)
    final_output_path = resolve_output_path(FINAL_OUTPUT_FILE)
    cache_key = int(final_output_path.stat().st_mtime if final_output_path.exists() else 0)

    loan_options = sorted(final_df["loan_type"].dropna().unique().tolist())
    risk_options = sorted(final_df["leave_risk_segment"].dropna().unique().tolist())
    rm_options = ["Has Relationship Manager", "No Relationship Manager"]

    threshold_df = data["thresholds"].copy()
    relationship_manager_df = data["relationship_manager"].copy()
    retention_kpi_df = data["retention_kpis"].copy()
    retention_framework_df = data["retention_framework"].copy()
    credit_risk_integration_df = data["credit_risk_integration"].copy()
    correlations_df = data["correlations"].copy()
    feature_importance_df = data["feature_importance"].copy()
    loan_type_summary_df = build_loan_type_summary(final_df)
    loan_type_story = build_loan_type_story(loan_type_summary_df)

    top_customers = sort_priority_customers(filtered_df)
    customer_360_payload = build_customer_360_payload(top_customers)

    strongest_dimension_label = format_label(metrics["strongest_servqual_dimension"])
    weakest_dimension_label = format_label(portfolio_summary["weakest_dimension"])
    strongest_view_dimension = format_label(portfolio_summary["strongest_dimension"])
    target_strategy = str(metrics["target_strategy"])
    target_note = html.escape(
        plain_target_note(target_strategy, str(metrics["target_note"]))
    )

    resolution_threshold = get_threshold_record(threshold_df, "resolution_days")
    app_threshold = get_threshold_record(threshold_df, "app_rating_score")
    rm_has = get_relationship_manager_record(
        relationship_manager_df, "Has Relationship Manager"
    )
    rm_no = get_relationship_manager_record(
        relationship_manager_df, "No Relationship Manager"
    )

    resolution_metric = "Waiting for latest file"
    resolution_detail = "Run the latest analysis export to show the main complaint-delay warning point."
    if resolution_threshold is not None:
        resolution_cutoff = pd.to_numeric(
            resolution_threshold.get("threshold"), errors="coerce"
        )
        resolution_metric = f"More than {format_number(resolution_cutoff, 1)} days"
        resolution_detail = (
            "Complaints that take longer than this are linked to much higher leave risk."
        )

    app_metric = "Waiting for latest file"
    app_detail = "Run the latest analysis export to show the main app-rating warning point."
    if app_threshold is not None:
        app_cutoff = pd.to_numeric(app_threshold.get("threshold"), errors="coerce")
        app_metric = f"{format_number(app_cutoff, 1)} or lower"
        app_detail = (
            "App ratings at or below this level are linked to higher leave risk."
        )

    rm_metric = "Waiting for latest file"
    rm_detail = "Dedicated-banker comparisons appear when the summary file is available."
    if rm_has is not None and rm_no is not None:
        rm_gap = (
            pd.to_numeric(rm_no["avg_predicted_leave_probability"], errors="coerce")
            - pd.to_numeric(rm_has["avg_predicted_leave_probability"], errors="coerce")
        )
        direction = "higher" if rm_gap >= 0 else "lower"
        rm_metric = f"{format_percent(abs(rm_gap), 1)} {direction} risk without one"
        rm_detail = (
            "Customers with a dedicated banker are less likely to leave in this data."
        )

    correlation_metric = "Waiting for latest file"
    correlation_detail = "Service-to-loyalty links appear after the latest export is generated."
    if not correlations_df.empty:
        leader = correlations_df.sort_values(
            "correlation_with_retention_score", ascending=False
        ).iloc[0]
        correlation_metric = format_label(leader["servqual_dimension"])
        correlation_detail = (
            "This service area lines up most closely with stronger loyalty scores."
        )

    loan_type_risk_metric = "Waiting for latest file"
    loan_type_risk_detail = "Loan-type comparisons appear when the latest export is available."
    if not loan_type_summary_df.empty:
        loan_type_risk_metric = format_label(loan_type_story["highest_risk_loan_type"])
        loan_type_risk_detail = (
            f"This loan portfolio has the highest average leave probability at "
            f"{format_percent(loan_type_summary_df.iloc[0]['avg_predicted_leave_probability'], 1)}. "
            "It is a segment where service friction appears to matter more."
        )

    loan_type_gap_metric = "Waiting for latest file"
    loan_type_gap_detail = "Run the latest export to compare higher-risk and lower-risk loan portfolios."
    if not loan_type_summary_df.empty:
        loan_type_gap_metric = format_percent(loan_type_story["risk_gap"], 1)
        loan_type_gap_detail = (
            f"{format_label(loan_type_story['highest_risk_loan_type'])} loans show a much higher average leave probability "
            f"than {format_label(loan_type_story['lowest_risk_loan_type'])} loans in this portfolio."
        )

    chart_cards_html = "".join(
        build_chart_card(chart_config, cache_key, static_site)
        for chart_config in CHART_CONFIG
    )
    chatbot_panel_html = build_chatbot_panel(active_page, params, static_site)
    retention_kpi_cards_html = build_retention_kpi_cards(retention_kpi_df)
    download_files = PUBLIC_DOWNLOAD_FILES if static_site else DOWNLOAD_FILES
    download_cards_html = "".join(
        build_download_card(file_name, static_site) for file_name in download_files
    )
    page_nav_html = build_page_nav(active_page, params, static_site)
    current_page_path = (
        build_page_url(active_page, params, static_site)
        if static_site
        else ("/" if active_page == "overview" else f"/{active_page}")
    )
    reset_page_url = current_page_path
    support_tables_html = "".join(
        [
            f"""
            <article class="table-card">
                <span class="section-tag">Optional Details</span>
                <h3>How customers are grouped</h3>
                <p>Shows who is in early follow-up and who needs direct follow-up.</p>
                <div class="table-wrap">{dataframe_to_html(retention_framework_df)}</div>
            </article>
            """,
            f"""
            <article class="table-card">
                <span class="section-tag">Optional Details</span>
                <h3>Credit and approval summary</h3>
                <p>Shows where extra approval is needed before credit-related offers or changes.</p>
                <div class="table-wrap">{dataframe_to_html(credit_risk_integration_df)}</div>
            </article>
            """,
            f"""
            <article class="table-card">
                <span class="section-tag">Optional Details</span>
                <h3>Risk warning points</h3>
                <p>Shows the simple cutoffs where risk starts rising faster.</p>
                <div class="table-wrap">{dataframe_to_html(threshold_df)}</div>
            </article>
            """,
            f"""
            <article class="table-card">
                <span class="section-tag">Optional Details</span>
                <h3>Dedicated banker comparison</h3>
                <p>Compares customers with and without a dedicated banker.</p>
                <div class="table-wrap">{dataframe_to_html(relationship_manager_df)}</div>
            </article>
            """,
            f"""
            <article class="table-card">
                <span class="section-tag">Optional Details</span>
                <h3>Service areas linked to loyalty</h3>
                <p>Shows which parts of the customer experience move most closely with loyalty.</p>
                <div class="table-wrap">{dataframe_to_html(correlations_df)}</div>
            </article>
            """,
            f"""
            <article class="table-card">
                <span class="section-tag">Optional Details</span>
                <h3>Loan type and retention</h3>
                <p>Compares leave risk, retention strength, service quality, and relationship coverage across loan portfolios.</p>
                <div class="table-wrap">{build_loan_type_table(loan_type_summary_df)}</div>
            </article>
            """,
            f"""
            <article class="table-card">
                <span class="section-tag">Optional Details</span>
                <h3>What the estimate relies on</h3>
                <p>Shows which inputs matter most in the estimate.</p>
                <div class="table-wrap">{dataframe_to_html(feature_importance_df.head(10))}</div>
            </article>
            """,
        ]
    )

    return f"""
    <html>
    <head>
        <meta charset="utf-8">
        {refresh_meta_tag}
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>Retail Banking Customer Dashboard</title>
        <style>{build_dashboard_css()}</style>
    </head>
    <body class="page-{html.escape(active_page)}">
        <div class="page">
            <section class="hero">
                <div class="hero-copy">
                    <span class="section-tag">Customer Retention Dashboard</span>
                    <h1>Customers Who May Need Attention</h1>
                    <p>This page highlights which customers may need follow-up, what service issues are hurting the relationship, and who should be contacted first.</p>
                    <p><strong>How the estimate works:</strong> {target_note}</p>
                    <div class="hero-meta">
                        <span class="hero-chip"><span class="live-dot"></span>Last update: {get_last_updated()}</span>
                        <span class="hero-chip">Customers loaded: {format_number(metrics['rows_loaded'])}</span>
                        <span class="hero-chip">Customers shown: {format_number(portfolio_summary['customers'])}</span>
                        <span class="hero-chip">Need attention soon: {format_number(portfolio_summary['high_risk_customers'])}</span>
                        <span class="hero-chip">Average chance of leaving: {format_percent(portfolio_summary['avg_leave_probability'], 1)}</span>
                        <span class="hero-chip">Average priority score: {format_number(portfolio_summary['avg_risk_adjusted_retention_score'], 2)}</span>
                    </div>
                    {page_nav_html}
                </div>
                <aside class="hero-panel">
                    <span class="section-tag">Current View</span>
                    <h2>{format_number(portfolio_summary['high_risk_customers'])} customers may need quick follow-up</h2>
                    <p>{format_percent(portfolio_summary['high_risk_share'], 1)} of this view is in the high-attention group. The biggest weak spot is <strong>{html.escape(weakest_dimension_label)}</strong>, and <strong>{format_number(portfolio_summary['credit_review_required_customers'])}</strong> customers need approval before a credit-related offer or change.</p>
                    <div class="spotlight-grid">
                        <div class="spotlight-card">
                            <span class="spotlight-label">Average Chance Of Leaving</span>
                            <span class="spotlight-value">{format_percent(portfolio_summary['avg_leave_probability'], 1)}</span>
                            <span class="spotlight-detail">Estimated risk in this filtered group</span>
                        </div>
                        <div class="spotlight-card">
                            <span class="spotlight-label">Average Priority Score</span>
                            <span class="spotlight-value">{format_number(portfolio_summary['avg_risk_adjusted_retention_score'], 2)}</span>
                            <span class="spotlight-detail">Higher scores should be reviewed sooner</span>
                        </div>
                    </div>
                    {build_risk_mix_bar(filtered_df)}
                </aside>
            </section>

            <section class="panel toolbar">
                <div class="panel-head">
                    <div>
                        <span class="panel-label">View Controls</span>
                        <h2>Choose what to look at</h2>
                        <p>Filter customers by loan type, attention level, and whether they already have a dedicated banker.</p>
                    </div>
                    <div class="panel-note">The page refreshes automatically when the latest export files change.</div>
                </div>
                <form method="get" action="{html.escape(current_page_path)}">
                    <div class="filters">
                        {build_select("loan_type", get_filter_value(params, "loan_type"), loan_options)}
                        {build_select("leave_risk_segment", get_filter_value(params, "leave_risk_segment"), risk_options)}
                        {build_select("relationship_manager", get_filter_value(params, "relationship_manager"), rm_options)}
                        <label class="filter">
                            <span>Refresh Every (Seconds)</span>
                            <input type="number" name="refresh" min="5" step="1" value="{refresh_seconds}">
                        </label>
                        <div class="filter-actions">
                            <button class="button button-primary" type="submit">Apply Filters</button>
                            <a class="button button-secondary" href="{html.escape(reset_page_url)}">Show All</a>
                        </div>
                    </div>
                </form>
                <div class="context-pills">{build_filter_context(params, refresh_seconds)}</div>
            </section>

            <section class="panel dashboard-section dashboard-section-overview" id="overview">
                <div class="panel-head">
                    <div>
                        <span class="panel-label">Quick Summary</span>
                        <h2>What this view shows</h2>
                        <p>These cards give a simple snapshot of customer volume, risk, service quality, and follow-up workload.</p>
                    </div>
                    <div class="panel-note">Best service area in this view: <strong>{html.escape(strongest_view_dimension)}</strong></div>
                </div>
                <div class="summary-grid">
                    <div class="metric-grid">
                        {build_metric_card("Customers Shown", format_number(portfolio_summary['customers']), "Customers in the current filter", "low")}
                        {build_metric_card("Need Attention Soon", format_number(portfolio_summary['high_risk_customers']), "Customers who may leave soon", "critical")}
                        {build_metric_card("Need Attention Now", format_number(portfolio_summary['critical_customers']), "Most urgent follow-up cases", "critical")}
                        {build_metric_card("Average Chance Of Leaving", format_percent(portfolio_summary['avg_leave_probability'], 1), "Estimated risk in this view", "high")}
                        {build_metric_card("Average Priority Score", format_number(portfolio_summary['avg_risk_adjusted_retention_score'], 2), "Higher means review sooner", "high")}
                        {build_metric_card("Average Service Experience Score", format_percent(portfolio_summary['avg_service_quality_index'], 1), "Overall service experience", "low")}
                        {build_metric_card("Average Loyalty Score", format_number(portfolio_summary['avg_retention_score'], 2), "Current relationship strength", "low")}
                        {build_metric_card("Need Direct Follow-Up", format_number(portfolio_summary['reactive_retention_customers']), "Customers already showing warning signs", "critical")}
                        {build_metric_card("Need Approval Before Offers", format_number(portfolio_summary['credit_review_required_customers']), "Approval needed before credit-related offers or changes", "high")}
                        {build_metric_card("Average Days To Solve Issues", format_number(portfolio_summary['avg_resolution_days'], 2), "How long complaints take to close", "high")}
                        {build_metric_card("Average App Rating", format_number(portfolio_summary['avg_app_rating_score'], 2), "How customers rate the app", "high")}
                    </div>
                    <aside class="summary-card">
                        <span class="section-tag">In Plain English</span>
                        <h3>Main story</h3>
                        <p>The weakest part of the customer experience in this view is <strong>{html.escape(weakest_dimension_label)}</strong>, and most urgent cases are concentrated in <strong>{html.escape(portfolio_summary['dominant_high_risk_loan_type'])}</strong> loans.</p>
                        <p><strong>{format_number(portfolio_summary['priority_without_rm'])}</strong> higher-risk customers still do not have a dedicated banker, and <strong>{format_number(portfolio_summary['credit_review_required_customers'])}</strong> customers need approval before a credit-related offer or change.</p>
                    </aside>
                </div>
            </section>

            <section class="panel dashboard-section dashboard-section-health" id="benchmarks">
                <div class="panel-head">
                    <div>
                        <span class="panel-label">Health Check</span>
                        <h2>Simple portfolio checkpoints</h2>
                        <p>These cards show a plain-English health check for the latest full customer file.</p>
                    </div>
                    <div class="panel-note">These are estimate-based measures because the dataset does not track confirmed customer exits.</div>
                </div>
                <div class="metric-grid">{retention_kpi_cards_html}</div>
                <div class="support-grid" style="margin-top: 14px;">
                    <article class="table-card">
                        <span class="section-tag">Simple Reference</span>
                        <h3>What the health checks mean</h3>
                        <p>A simple reference table for the health-check cards shown above.</p>
                        <div class="table-wrap">{build_kpi_reference_table(retention_kpi_df)}</div>
                    </article>
                </div>
            </section>

            <section class="panel dashboard-section dashboard-section-drivers" id="drivers">
                <div class="panel-head">
                    <div>
                        <span class="panel-label">Key Issues</span>
                        <h2>What seems to push risk higher</h2>
                        <p>These cards highlight the clearest warning signs in plain language.</p>
                    </div>
                    <div class="panel-note">Service area with the biggest effect: <strong>{html.escape(strongest_dimension_label)}</strong></div>
                </div>
                <div class="signal-grid">
                    {build_signal_card("Most important service area", strongest_dimension_label, "This service area has the biggest effect on the risk estimate.", "high")}
                    {build_signal_card("Complaint delay warning", resolution_metric, resolution_detail, "critical")}
                    {build_signal_card("App rating warning", app_metric, app_detail, "high")}
                    {build_signal_card("Dedicated banker effect", rm_metric, rm_detail, "low")}
                    {build_signal_card("Best link to loyalty", correlation_metric, correlation_detail, "low")}
                    {build_signal_card("Loan type under most pressure", loan_type_risk_metric, loan_type_risk_detail, "high")}
                    {build_signal_card("Loan type risk gap", loan_type_gap_metric, loan_type_gap_detail, "high")}
                    {build_signal_card("Weakest area in this view", weakest_dimension_label, "This is the lowest average service score in the current filter, so it is the clearest weakness to fix first.", "critical")}
                </div>
                <div class="support-grid" style="margin-top: 14px;">
                    <article class="table-card">
                        <span class="section-tag">Loan Type View</span>
                        <h3>How product segments link with retention</h3>
                        <p><strong>{html.escape(format_label(loan_type_story['highest_risk_loan_type']))}</strong> loans currently show the highest average leave probability, while <strong>{html.escape(format_label(loan_type_story['lowest_risk_loan_type']))}</strong> loans show the lowest. This suggests loan type works mainly as a portfolio segment indicator of where service friction and follow-up pressure are concentrated.</p>
                        <div class="table-wrap">{build_loan_type_table(loan_type_summary_df)}</div>
                    </article>
                </div>
            </section>

            <section class="panel dashboard-section dashboard-section-charts" id="diagnostics">
                <div class="panel-head">
                    <div>
                        <span class="panel-label">Charts</span>
                        <h2>See the patterns</h2>
                        <p>These charts give visual proof behind the main findings.</p>
                    </div>
                </div>
                <div class="chart-grid">{chart_cards_html}</div>
            </section>

            <section class="panel dashboard-section dashboard-section-actions" id="actions">
                <div class="panel-head">
                    <div>
                        <span class="panel-label">Action List</span>
                        <h2>Customers to review first</h2>
                        <p>Customers below are sorted by priority score so the list brings the most urgent and appropriate follow-up cases to the top.</p>
                    </div>
                    <div class="panel-note">Start with higher-risk customers without a dedicated banker, then review the cases that need approval before a credit-related offer or change.</div>
                </div>
                <div class="action-summary">
                    <span class="action-pill">Need attention now: {format_number(portfolio_summary['critical_customers'])}</span>
                    <span class="action-pill">Higher-risk customers without a dedicated banker: {format_number(portfolio_summary['priority_without_rm'])}</span>
                    <span class="action-pill">Need approval before offers: {format_number(portfolio_summary['credit_review_required_customers'])}</span>
                    <span class="action-pill">Loan type with the most urgent cases: {html.escape(portfolio_summary['dominant_high_risk_loan_type'])}</span>
                </div>
                <div class="table-wrap">{build_action_table(top_customers)}</div>
            </section>

            {chatbot_panel_html}

            <section class="panel dashboard-section dashboard-section-details" id="details">
                <div class="panel-head">
                    <div>
                        <span class="panel-label">Detailed Tables</span>
                        <h2>Optional deeper details</h2>
                        <p>These tables keep the full exported details available if someone needs the analyst view.</p>
                    </div>
                </div>
                <div class="support-grid">{support_tables_html}</div>
            </section>

            <section class="panel dashboard-section dashboard-section-downloads" id="downloads">
                <div class="panel-head">
                    <div>
                        <span class="panel-label">Downloads</span>
                        <h2>Latest files</h2>
                        <p>Download the latest CSV files directly from the dashboard.</p>
                    </div>
                </div>
                <div class="download-grid">{download_cards_html}</div>
            </section>
        </div>
        <div class="customer-modal" id="customer-360-modal" aria-hidden="true">
            <section class="customer-modal-card" role="dialog" aria-modal="true" aria-labelledby="customer-360-title">
                <div class="customer-modal-head">
                    <div>
                        <span class="panel-label">Customer 360 Review</span>
                        <h2 id="customer-360-title">Customer</h2>
                        <span id="customer-360-risk-badge" class="risk-pill risk-modal-badge">NA</span>
                    </div>
                    <button type="button" class="modal-close" id="customer-360-close">Close</button>
                </div>
                <div class="customer-modal-body">
                    <article class="customer-360-section">
                        <h3>Customer Summary</h3>
                        <div class="customer-360-grid" id="customer-360-summary"></div>
                    </article>
                    <article class="customer-360-section">
                        <h3>Risk Explanation</h3>
                        <p id="customer-360-risk-copy">Select a Customer ID to view details.</p>
                        <div class="customer-360-grid" id="customer-360-metrics" style="margin-top: 12px;"></div>
                    </article>
                    <article class="customer-360-section customer-360-full">
                        <h3>Key Drivers / Warning Signals</h3>
                        <div class="driver-chip-row" id="customer-360-drivers"></div>
                    </article>
                    <article class="customer-360-section">
                        <h3>RM Next Steps</h3>
                        <ul class="customer-360-list" id="customer-360-next-steps"></ul>
                    </article>
                    <article class="customer-360-section">
                        <h3>Recommended Offers / Support</h3>
                        <ul class="customer-360-list" id="customer-360-safe-offers"></ul>
                    </article>
                    <article class="customer-360-section customer-360-full">
                        <h3>Offers To Avoid</h3>
                        <ul class="customer-360-list" id="customer-360-avoid-offers"></ul>
                    </article>
                </div>
            </section>
        </div>
        <script id="customer-360-data" type="application/json">{customer_360_payload}</script>
        <script>{build_dashboard_js()}</script>
    </body>
    </html>
    """


class DashboardHandler(SimpleHTTPRequestHandler):
    def _send_json(self, status, payload):
        response_bytes = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(response_bytes)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(response_bytes)

    def _handle_chat_request(self):
        parsed = urlparse(self.path)
        if parsed.path != "/api/chat":
            self.send_error(HTTPStatus.NOT_FOUND, "This API route is not available.")
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            content_length = 0
        if content_length <= 0:
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                {"message": "Please enter a question for Decipher AI."},
            )
            return

        try:
            request_payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
        except json.JSONDecodeError:
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                {"message": "The dashboard sent an invalid chat request."},
            )
            return

        question = str(request_payload.get("question", "")).strip()
        if not question:
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                {"message": "Please enter a question for Decipher AI."},
            )
            return

        raw_params = request_payload.get("params") or {}
        normalized_params = {}
        if isinstance(raw_params, dict):
            for key, value in raw_params.items():
                if isinstance(value, list):
                    normalized_params[key] = [str(item) for item in value]
                else:
                    normalized_params[key] = [str(value)]

        page_path = str(request_payload.get("path", "/"))
        active_page = PAGE_ROUTES.get(page_path, "overview")
        data = load_data()
        final_df = ensure_dashboard_columns(data["final"])
        metrics_df = data["metrics"]
        if final_df.empty or metrics_df.empty:
            self._send_json(
                HTTPStatus.SERVICE_UNAVAILABLE,
                {
                    "message": (
                        "Decipher AI cannot answer yet because the latest dashboard outputs are missing. "
                        "Run python retail_banking_analysis.py first."
                    )
                },
            )
            return

        filtered_df = filter_dataframe(final_df, normalized_params)
        dashboard_context = build_dashboard_context(
            data, filtered_df, normalized_params, active_page
        )
        result = ask_gemini_dashboard_assistant(
            question,
            dashboard_context,
            history=request_payload.get("history"),
        )
        status = result.pop("status", HTTPStatus.OK)
        self._send_json(status, result)

    def _handle_request(self, include_body):
        parsed = urlparse(self.path)
        active_page = PAGE_ROUTES.get(parsed.path)
        if active_page is not None:
            params = parse_qs(parsed.query)
            html_content = build_dashboard_html(params, active_page).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html_content)))
            self.end_headers()
            if include_body:
                self.wfile.write(html_content)
            return

        if parsed.path == "/favicon.ico":
            self.send_response(HTTPStatus.NO_CONTENT)
            self.end_headers()
            return

        preferred_path = PUBLIC_FILE_ROUTES.get(parsed.path)
        if preferred_path is None:
            self.send_error(HTTPStatus.NOT_FOUND, "This dashboard route is not available.")
            return

        resolved_path = resolve_output_path(preferred_path)
        if not resolved_path.exists():
            self.send_error(
                HTTPStatus.NOT_FOUND,
                "Requested dashboard asset not found. Run python retail_banking_analysis.py to regenerate outputs.",
            )
            return

        file_bytes = resolved_path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", self.guess_type(resolved_path.name))
        self.send_header("Content-Length", str(len(file_bytes)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        if include_body:
            self.wfile.write(file_bytes)

    def do_GET(self):
        self._handle_request(include_body=True)

    def do_HEAD(self):
        self._handle_request(include_body=False)

    def do_POST(self):
        self._handle_chat_request()


def copy_static_asset(source_path, destination_path):
    resolved_source = resolve_output_path(source_path)
    if not resolved_source.exists():
        return False
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(resolved_source, destination_path)
    return True


def export_static_site(output_dir=STATIC_EXPORT_DIR):
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / ".nojekyll").write_text("", encoding="utf-8")

    exported_pages = []
    for page_id, _, _ in PAGE_CONFIG:
        file_name = "index.html" if page_id == "overview" else f"{page_id}.html"
        page_html = build_dashboard_html({}, page_id, static_site=True)
        (output_dir / file_name).write_text(page_html, encoding="utf-8")
        exported_pages.append(file_name)

    copied_assets = []
    for file_name in CHART_FILES:
        destination = output_dir / "assets" / "charts" / file_name
        if copy_static_asset(BASE_DIR / file_name, destination):
            copied_assets.append(destination.relative_to(output_dir).as_posix())

    for file_name, source_path in PUBLIC_DOWNLOAD_FILES.items():
        destination = output_dir / "assets" / "downloads" / file_name
        if copy_static_asset(source_path, destination):
            copied_assets.append(destination.relative_to(output_dir).as_posix())

    print(f"Exported GitHub Pages site to {output_dir}")
    print(f"Pages: {', '.join(exported_pages)}")
    print(f"Assets copied: {len(copied_assets)}")


def main():
    parser = argparse.ArgumentParser(
        description="Run the retail banking customer dashboard."
    )
    parser.add_argument(
        "--export-static",
        action="store_true",
        help="Write a static GitHub Pages site to the docs folder.",
    )
    parser.add_argument(
        "--static-dir",
        default=str(STATIC_EXPORT_DIR),
        help="Output folder for --export-static.",
    )
    parser.add_argument(
        "--host", default="127.0.0.1", help="Host to bind the dashboard server to."
    )
    parser.add_argument(
        "--port", type=int, default=8501, help="Port to bind the dashboard server to."
    )
    args = parser.parse_args()

    if args.export_static:
        export_static_site(Path(args.static_dir))
        return

    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    print(f"Dashboard running at http://{args.host}:{args.port}")
    print("The dashboard reloads automatically and reads the latest analysis outputs.")
    print("Press Ctrl+C to stop the server.")
    server.serve_forever()


if __name__ == "__main__":
    main()
