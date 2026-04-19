import argparse
import html
from http import HTTPStatus
from datetime import datetime
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

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


def format_label(value):
    if pd.isna(value):
        return "NA"
    text = str(value)
    return DISPLAY_LABELS.get(text, text.replace("servqual_", "").replace("_", " ").title())


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


def chart_html(file_name, cache_key):
    chart_path = resolve_output_path(BASE_DIR / file_name)
    if not chart_path.exists():
        return (
            '<div class="empty-state">Chart not found. Run '
            '<code>python retail_banking_analysis.py</code> to regenerate outputs.</div>'
        )
    return f'<img src="/charts/{file_name}?v={cache_key}" alt="{html.escape(chart_path.stem.replace("_", " ").title())}">'


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


def build_chart_card(chart_config, cache_key):
    return f"""
    <article class="chart-card">
        <div class="chart-copy">
            <span class="section-tag">Chart</span>
            <h3>{html.escape(chart_config['title'])}</h3>
            <p>{html.escape(chart_config['description'])}</p>
        </div>
        <div class="chart-frame">{chart_html(chart_config['file'], cache_key)}</div>
    </article>
    """


def build_download_card(file_name):
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
        <a class="download-link" href="/downloads/{html.escape(file_name)}">Download CSV</a>
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


def build_action_table(frame):
    if frame.empty:
        return '<div class="empty-state">No customers match the current filter.</div>'

    rows = []
    for _, row in frame.head(15).iterrows():
        risk_segment = str(row.get("leave_risk_segment", "NA"))
        has_dedicated_banker = (
            pd.to_numeric(row.get("has_relationship_manager"), errors="coerce") == 1
        )
        coverage_state = "assigned" if has_dedicated_banker else "unassigned"
        coverage_text = "Yes" if has_dedicated_banker else "No"
        approval_needed = (
            "Yes"
            if pd.to_numeric(row.get("credit_review_required"), errors="coerce") == 1
            else "No"
        )
        rows.append(
            f"""
            <tr>
                <td data-label="Customer">{html.escape(str(row.get("customer_id", "NA")))}</td>
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
    .filters, .metric-grid, .signal-grid, .chart-grid, .support-grid {
        display: grid;
        gap: 14px;
    }
    .filters { grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); }
    .metric-grid { grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); }
    .signal-grid { grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); }
    .chart-grid { grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); }
    .support-grid { grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); }
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
    .risk-critical, .coverage-unassigned { color: var(--alert); background: rgba(159, 47, 40, 0.14); }
    .risk-high { color: #9a4d08; background: rgba(217, 108, 6, 0.14); }
    .risk-moderate { color: #80610c; background: rgba(207, 170, 70, 0.18); }
    .risk-low, .coverage-assigned { color: #426645; background: rgba(108, 138, 93, 0.14); }
    .download-grid { gap: 14px; }
    .download-card { flex: 1 1 240px; min-width: 240px; }
    .empty-state {
        padding: 18px;
        border-radius: 16px;
        border: 1px dashed rgba(20, 35, 45, 0.20);
        background: rgba(255,255,255,0.58);
        color: var(--muted);
    }
    code {
        padding: 2px 6px;
        border-radius: 8px;
        background: rgba(20, 35, 45, 0.08);
        font-family: Consolas, "Courier New", monospace;
    }
    @media (max-width: 1100px) {
        .hero, .summary-grid { grid-template-columns: 1fr; }
        .toolbar { position: static; }
    }
    @media (max-width: 760px) {
        .page { padding: 16px 16px 32px; }
        .hero, .panel { padding: 18px; }
        .panel-head { flex-direction: column; align-items: start; }
        .spotlight-grid { grid-template-columns: 1fr; }
        .filter-actions { grid-column: 1 / -1; display: grid; grid-template-columns: 1fr 1fr; }
        .button, .download-link { width: 100%; }
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


def build_dashboard_html(params):
    try:
        refresh_seconds = max(5, int(params.get("refresh", ["300"])[0]))
    except (TypeError, ValueError):
        refresh_seconds = 300

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

    top_customers = filtered_df.sort_values(
        ["risk_adjusted_retention_score", "predicted_leave_probability", "clv"],
        ascending=[False, False, False],
    )

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

    chart_cards_html = "".join(
        build_chart_card(chart_config, cache_key) for chart_config in CHART_CONFIG
    )
    retention_kpi_cards_html = build_retention_kpi_cards(retention_kpi_df)
    download_cards_html = "".join(
        build_download_card(file_name) for file_name in DOWNLOAD_FILES
    )
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
        <meta http-equiv="refresh" content="{refresh_seconds}">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>Retail Banking Customer Dashboard</title>
        <style>{build_dashboard_css()}</style>
    </head>
    <body>
        <div class="page">
            <section class="hero">
                <div class="hero-copy">
                    <span class="section-tag">Customer Retention Dashboard</span>
                    <h1>Customers Who May Need Attention</h1>
                    <p>This page highlights which customers may need follow-up, what service issues are hurting the relationship, and who should be contacted first.</p>
                    <p><strong>How the estimate works:</strong> {target_note}</p>
                    <div class="hero-meta">
                        <span class="hero-chip">Last update: {get_last_updated()}</span>
                        <span class="hero-chip">Customers loaded: {format_number(metrics['rows_loaded'])}</span>
                        <span class="hero-chip">Customers shown: {format_number(portfolio_summary['customers'])}</span>
                        <span class="hero-chip">Need attention soon: {format_number(portfolio_summary['high_risk_customers'])}</span>
                        <span class="hero-chip">Average chance of leaving: {format_percent(portfolio_summary['avg_leave_probability'], 1)}</span>
                        <span class="hero-chip">Average priority score: {format_number(portfolio_summary['avg_risk_adjusted_retention_score'], 2)}</span>
                    </div>
                    <div class="jump-links">
                        <a class="jump-link" href="#overview">Overview</a>
                        <a class="jump-link" href="#benchmarks">Health Check</a>
                        <a class="jump-link" href="#drivers">Key Issues</a>
                        <a class="jump-link" href="#diagnostics">Charts</a>
                        <a class="jump-link" href="#actions">Action List</a>
                        <a class="jump-link" href="#downloads">Files</a>
                    </div>
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
                <form method="get">
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
                            <a class="button button-secondary" href="/">Show All</a>
                        </div>
                    </div>
                </form>
                <div class="context-pills">{build_filter_context(params, refresh_seconds)}</div>
            </section>

            <section class="panel" id="overview">
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

            <section class="panel" id="benchmarks">
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

            <section class="panel" id="drivers">
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
                    {build_signal_card("Weakest area in this view", weakest_dimension_label, "This is the lowest average service score in the current filter, so it is the clearest weakness to fix first.", "critical")}
                </div>
            </section>

            <section class="panel" id="diagnostics">
                <div class="panel-head">
                    <div>
                        <span class="panel-label">Charts</span>
                        <h2>See the patterns</h2>
                        <p>These charts give visual proof behind the main findings.</p>
                    </div>
                </div>
                <div class="chart-grid">{chart_cards_html}</div>
            </section>

            <section class="panel" id="actions">
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

            <section class="panel">
                <div class="panel-head">
                    <div>
                        <span class="panel-label">Detailed Tables</span>
                        <h2>Optional deeper details</h2>
                        <p>These tables keep the full exported details available if someone needs the analyst view.</p>
                    </div>
                </div>
                <div class="support-grid">{support_tables_html}</div>
            </section>

            <section class="panel" id="downloads">
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
    </body>
    </html>
    """


class DashboardHandler(SimpleHTTPRequestHandler):
    def _handle_request(self, include_body):
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/dashboard"):
            params = parse_qs(parsed.query)
            html_content = build_dashboard_html(params).encode("utf-8")
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


def main():
    parser = argparse.ArgumentParser(
        description="Run the retail banking customer dashboard."
    )
    parser.add_argument(
        "--host", default="127.0.0.1", help="Host to bind the dashboard server to."
    )
    parser.add_argument(
        "--port", type=int, default=8501, help="Port to bind the dashboard server to."
    )
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    print(f"Dashboard running at http://{args.host}:{args.port}")
    print("The dashboard reloads automatically and reads the latest analysis outputs.")
    print("Press Ctrl+C to stop the server.")
    server.serve_forever()


if __name__ == "__main__":
    main()
