import argparse
import html
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
HIGH_RISK_CUSTOMERS_FILE = BASE_DIR / "likely_to_leave_customers.csv"
CHART_FILES = [
    "chart_servqual_retention_heatmap.png",
    "chart_sqi_by_target.png",
    "chart_servqual_feature_importance.png",
    "chart_app_rating_leave_risk.png",
    "chart_resolution_days_leave_risk.png",
    "chart_relationship_manager_retention.png",
]
SERVQUAL_DIMENSION_COLUMNS = [
    "servqual_reliability",
    "servqual_responsiveness",
    "servqual_assurance",
    "servqual_empathy",
    "servqual_tangibles",
]

COLOR_PRIMARY = "#1d3557"
COLOR_SECONDARY = "#457b9d"
COLOR_ACCENT = "#a8dadc"
COLOR_SURFACE = "#f5fbfd"
COLOR_TEXT = "#17324d"
COLOR_ALERT = "#d62828"


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


def get_filter_value(params, key):
    value = params.get(key, ["All"])[0]
    return value if value else "All"


def build_select(name, selected_value, options):
    option_html = ['<option value="All">All</option>']
    for option in options:
        option_text = str(option)
        selected_attr = ' selected="selected"' if option_text == selected_value else ""
        option_html.append(
            f'<option value="{html.escape(option_text)}"{selected_attr}>{html.escape(option_text)}</option>'
        )
    return f"""
    <label class="filter">
        <span>{html.escape(name.replace("_", " ").title())}</span>
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


def summarize_portfolio(filtered_df):
    if filtered_df.empty:
        return {
            "customers": 0,
            "avg_leave_probability": np.nan,
            "high_risk_customers": 0,
            "avg_service_quality_index": np.nan,
            "avg_retention_score": np.nan,
            "avg_app_rating_score": np.nan,
            "avg_resolution_days": np.nan,
            "weakest_dimension": "NA",
        }

    dimension_mean = filtered_df[SERVQUAL_DIMENSION_COLUMNS].mean().sort_values()

    return {
        "customers": int(len(filtered_df)),
        "avg_leave_probability": float(filtered_df["predicted_leave_probability"].mean()),
        "high_risk_customers": int(
            filtered_df["leave_risk_segment"].isin(["High", "Critical"]).sum()
        ),
        "avg_service_quality_index": float(filtered_df["service_quality_index"].mean()),
        "avg_retention_score": float(filtered_df["retention_score"].mean()),
        "avg_app_rating_score": float(filtered_df["app_rating_score"].mean()),
        "avg_resolution_days": float(filtered_df["resolution_days"].mean()),
        "weakest_dimension": dimension_mean.index[0],
    }


def chart_html(file_name, cache_key):
    chart_path = resolve_output_path(BASE_DIR / file_name)
    if not chart_path.exists():
        return (
            '<div class="empty-state">Chart not found. Run '
            '<code>python retail_banking_analysis.py</code> to regenerate outputs.</div>'
        )
    return f'<img src="/{chart_path.name}?v={cache_key}" alt="{html.escape(chart_path.name)}">'


def build_dashboard_html(params):
    refresh_seconds = max(5, int(params.get("refresh", ["15"])[0]))
    data = load_data()
    final_df = data["final"]
    metrics_df = data["metrics"]

    if final_df.empty or metrics_df.empty:
        return f"""
        <html>
        <head>
            <meta charset="utf-8">
            <meta http-equiv="refresh" content="{refresh_seconds}">
            <title>Retail Banking Leave Propensity Dashboard</title>
            <style>
                body {{ font-family: Segoe UI, sans-serif; background: {COLOR_SURFACE}; color: {COLOR_TEXT}; margin: 32px; }}
                .note {{ background: white; padding: 22px; border-radius: 16px; max-width: 900px; box-shadow: 0 8px 24px rgba(23, 50, 77, 0.08); }}
                code {{ background: #eef5f7; padding: 2px 6px; border-radius: 6px; }}
            </style>
        </head>
        <body>
            <div class="note">
                <h1>Retail Banking Leave Propensity Dashboard</h1>
                <p>The dashboard is waiting for analysis outputs.</p>
                <p>Run <code>python retail_banking_analysis.py</code> first, then reload this page.</p>
            </div>
        </body>
        </html>
        """

    metrics = metrics_df.iloc[0]
    filtered_df = filter_dataframe(final_df, params)
    portfolio_summary = summarize_portfolio(filtered_df)
    final_output_path = resolve_output_path(FINAL_OUTPUT_FILE)
    cache_key = int((final_output_path.stat().st_mtime if final_output_path.exists() else 0))

    loan_options = sorted(final_df["loan_type"].dropna().unique().tolist())
    risk_options = sorted(final_df["leave_risk_segment"].dropna().unique().tolist())
    rm_options = ["Has Relationship Manager", "No Relationship Manager"]

    threshold_df = data["thresholds"].copy()
    relationship_manager_df = data["relationship_manager"].copy()
    correlations_df = data["correlations"].copy()
    feature_importance_df = data["feature_importance"].copy()

    top_customers = filtered_df.sort_values(
        ["predicted_leave_probability", "clv"],
        ascending=[False, False],
    )
    top_customer_columns = [
        "customer_id",
        "loan_type",
        "clv",
        "retention_score",
        "service_quality_index",
        "predicted_leave_probability",
        "leave_risk_segment",
        "app_rating_score",
        "resolution_days",
        "recommended_action",
    ]
    top_customers = top_customers[top_customer_columns].head(20)

    weakest_dimension_label = portfolio_summary["weakest_dimension"].replace("servqual_", "").title()
    target_strategy = str(metrics["target_strategy"])
    target_note = html.escape(str(metrics["target_note"]))

    return f"""
    <html>
    <head>
        <meta charset="utf-8">
        <meta http-equiv="refresh" content="{refresh_seconds}">
        <title>Retail Banking Leave Propensity Dashboard</title>
        <style>
            :root {{
                --bg: {COLOR_SURFACE};
                --panel: #ffffff;
                --ink: {COLOR_TEXT};
                --muted: #5a7184;
                --line: #d8e5ea;
                --primary: {COLOR_PRIMARY};
                --secondary: {COLOR_SECONDARY};
                --accent: {COLOR_ACCENT};
                --alert: {COLOR_ALERT};
            }}
            * {{ box-sizing: border-box; }}
            body {{
                margin: 0;
                background:
                    radial-gradient(circle at top right, rgba(168, 218, 220, 0.45), transparent 28%),
                    linear-gradient(180deg, #f7fcfd 0%, #edf6f8 100%);
                color: var(--ink);
                font-family: "Segoe UI", Tahoma, sans-serif;
            }}
            .page {{
                max-width: 1480px;
                margin: 0 auto;
                padding: 28px;
            }}
            .hero {{
                background: linear-gradient(135deg, rgba(29,53,87,0.98), rgba(69,123,157,0.95));
                color: white;
                border-radius: 24px;
                padding: 28px;
                box-shadow: 0 20px 40px rgba(29, 53, 87, 0.18);
            }}
            .hero h1 {{
                margin: 0 0 10px;
                font-size: 32px;
            }}
            .hero p {{
                margin: 6px 0;
                color: rgba(255,255,255,0.90);
                line-height: 1.5;
                max-width: 980px;
            }}
            .meta {{
                display: flex;
                flex-wrap: wrap;
                gap: 12px;
                margin-top: 18px;
                font-size: 14px;
            }}
            .chip {{
                background: rgba(255,255,255,0.14);
                border: 1px solid rgba(255,255,255,0.2);
                border-radius: 999px;
                padding: 8px 12px;
            }}
            .panel {{
                background: var(--panel);
                border: 1px solid var(--line);
                border-radius: 18px;
                padding: 20px;
                margin-top: 18px;
                box-shadow: 0 8px 24px rgba(23, 50, 77, 0.05);
            }}
            .panel h2 {{
                margin: 0 0 12px;
                font-size: 20px;
            }}
            .panel p {{
                margin: 0;
                color: var(--muted);
                line-height: 1.45;
            }}
            .filters {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
                gap: 14px;
                margin-top: 14px;
            }}
            .filter span {{
                display: block;
                margin-bottom: 6px;
                color: var(--muted);
                font-size: 13px;
            }}
            .filter select,
            .filter input {{
                width: 100%;
                padding: 10px 12px;
                border-radius: 12px;
                border: 1px solid var(--line);
                background: #fbfdfe;
                color: var(--ink);
            }}
            .filter-actions {{
                display: flex;
                align-items: end;
                gap: 10px;
            }}
            .button {{
                display: inline-flex;
                align-items: center;
                justify-content: center;
                padding: 10px 14px;
                border-radius: 12px;
                text-decoration: none;
                border: 0;
                cursor: pointer;
                font-weight: 700;
            }}
            .button-primary {{
                background: var(--primary);
                color: white;
            }}
            .button-secondary {{
                background: #ebf3f6;
                color: var(--ink);
            }}
            .kpis {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
                gap: 14px;
            }}
            .kpi {{
                background: linear-gradient(180deg, #ffffff 0%, #f7fbfc 100%);
                border: 1px solid var(--line);
                border-radius: 16px;
                padding: 18px;
            }}
            .kpi .label {{
                font-size: 13px;
                color: var(--muted);
                margin-bottom: 8px;
            }}
            .kpi .value {{
                font-size: 28px;
                font-weight: 800;
                color: var(--primary);
                margin-bottom: 4px;
            }}
            .kpi .detail {{
                font-size: 13px;
                color: var(--muted);
            }}
            .insight-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
                gap: 14px;
            }}
            .insight-card {{
                border: 1px solid var(--line);
                border-radius: 16px;
                padding: 16px;
                background: #fcfeff;
            }}
            .charts {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
                gap: 16px;
            }}
            .chart-card {{
                background: #ffffff;
                border: 1px solid var(--line);
                border-radius: 18px;
                padding: 14px;
            }}
            .chart-card img {{
                width: 100%;
                display: block;
                border-radius: 12px;
            }}
            .tables {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
                gap: 16px;
            }}
            .table-wrap {{
                max-height: 380px;
                overflow: auto;
                border: 1px solid var(--line);
                border-radius: 14px;
            }}
            .data-table {{
                width: 100%;
                border-collapse: collapse;
                font-size: 13px;
            }}
            .data-table th,
            .data-table td {{
                border-bottom: 1px solid var(--line);
                padding: 8px 10px;
                text-align: left;
            }}
            .data-table th {{
                background: #f1f7f8;
                position: sticky;
                top: 0;
            }}
            .links {{
                display: flex;
                flex-wrap: wrap;
                gap: 10px;
                margin-top: 12px;
            }}
            .links a {{
                color: var(--primary);
                text-decoration: none;
                font-weight: 700;
            }}
            .empty-state {{
                padding: 18px;
                border: 1px dashed var(--line);
                border-radius: 14px;
                background: #fbfdfe;
                color: var(--muted);
            }}
            .highlight {{
                color: var(--alert);
                font-weight: 700;
            }}
            @media (max-width: 720px) {{
                .page {{ padding: 16px; }}
                .hero h1 {{ font-size: 26px; }}
                .kpi .value {{ font-size: 24px; }}
            }}
        </style>
    </head>
    <body>
        <div class="page">
            <section class="hero">
                <h1>Retail Banking Leave Propensity Dashboard</h1>
                <p>
                    Analytical framework for evaluating how SERVQUAL service dimensions relate to
                    customer retention and which customers are most likely to leave the bank.
                </p>
                <p>
                    Target mode: <strong>{html.escape(target_strategy)}</strong>. {target_note}
                </p>
                <div class="meta">
                    <div class="chip">Last analysis update: {get_last_updated()}</div>
                    <div class="chip">Auto-refresh: {refresh_seconds} sec</div>
                    <div class="chip">Accuracy: {format_number(metrics['accuracy'], 4)}</div>
                    <div class="chip">Precision: {format_number(metrics['precision'], 4)}</div>
                    <div class="chip">Recall: {format_number(metrics['recall'], 4)}</div>
                </div>
            </section>

            <section class="panel">
                <h2>Filters</h2>
                <form method="get">
                    <div class="filters">
                        {build_select("loan_type", get_filter_value(params, "loan_type"), loan_options)}
                        {build_select("leave_risk_segment", get_filter_value(params, "leave_risk_segment"), risk_options)}
                        {build_select("relationship_manager", get_filter_value(params, "relationship_manager"), rm_options)}
                        <label class="filter">
                            <span>Refresh Seconds</span>
                            <input type="number" name="refresh" min="5" step="1" value="{refresh_seconds}">
                        </label>
                        <div class="filter-actions">
                            <button class="button button-primary" type="submit">Apply View</button>
                            <a class="button button-secondary" href="/">Reset</a>
                        </div>
                    </div>
                </form>
                <div class="links">
                    <a href="/final_project_output.csv">Final Output CSV</a>
                    <a href="/summary_model_metrics.csv">Model Metrics CSV</a>
                    <a href="/summary_thresholds.csv">Threshold Summary CSV</a>
                    <a href="/summary_relationship_manager_effect.csv">Relationship Manager Summary CSV</a>
                    <a href="/likely_to_leave_customers.csv">High-Risk Customers CSV</a>
                </div>
            </section>

            <section class="panel">
                <h2>Portfolio Snapshot</h2>
                <div class="kpis">
                    <div class="kpi">
                        <div class="label">Customers in View</div>
                        <div class="value">{format_number(portfolio_summary['customers'])}</div>
                        <div class="detail">Filtered customer universe</div>
                    </div>
                    <div class="kpi">
                        <div class="label">Average Leave Probability</div>
                        <div class="value">{format_percent(portfolio_summary['avg_leave_probability'])}</div>
                        <div class="detail">Model-estimated attrition propensity</div>
                    </div>
                    <div class="kpi">
                        <div class="label">High / Critical Risk Customers</div>
                        <div class="value">{format_number(portfolio_summary['high_risk_customers'])}</div>
                        <div class="detail">Immediate review queue</div>
                    </div>
                    <div class="kpi">
                        <div class="label">Average Service Quality Index</div>
                        <div class="value">{format_number(portfolio_summary['avg_service_quality_index'], 3)}</div>
                        <div class="detail">Composite SERVQUAL score</div>
                    </div>
                    <div class="kpi">
                        <div class="label">Average Retention Score</div>
                        <div class="value">{format_number(portfolio_summary['avg_retention_score'], 2)}</div>
                        <div class="detail">Observed internal retention signal</div>
                    </div>
                    <div class="kpi">
                        <div class="label">Average App Rating</div>
                        <div class="value">{format_number(portfolio_summary['avg_app_rating_score'], 2)}</div>
                        <div class="detail">Digital service quality signal</div>
                    </div>
                    <div class="kpi">
                        <div class="label">Average Resolution Days</div>
                        <div class="value">{format_number(portfolio_summary['avg_resolution_days'], 2)}</div>
                        <div class="detail">Responsiveness pressure point</div>
                    </div>
                    <div class="kpi">
                        <div class="label">Weakest SERVQUAL Dimension</div>
                        <div class="value">{html.escape(weakest_dimension_label)}</div>
                        <div class="detail">Lowest average score in this filtered view</div>
                    </div>
                </div>
            </section>

            <section class="panel">
                <h2>Business Interpretation</h2>
                <div class="insight-grid">
                    <div class="insight-card">
                        <strong>Strongest SERVQUAL predictor</strong>
                        <p>{html.escape(str(metrics['strongest_servqual_dimension']).replace('servqual_', '').title())} carries the highest model importance among the five service dimensions.</p>
                    </div>
                    <div class="insight-card">
                        <strong>Critical threshold</strong>
                        <p>The largest jump in leave risk appears when customers wait too long for issue resolution or rate the app poorly. The detailed cutoffs are listed below for service recovery action.</p>
                    </div>
                    <div class="insight-card">
                        <strong>Relationship-manager effect</strong>
                        <p>Dedicated relationship coverage is associated with stronger retention outcomes and lower leave propensity, which supports a targeted human-intervention strategy for valuable accounts.</p>
                    </div>
                </div>
            </section>

            <section class="panel">
                <h2>Visual Diagnostics</h2>
                <div class="charts">
                    <div class="chart-card">{chart_html("chart_servqual_retention_heatmap.png", cache_key)}</div>
                    <div class="chart-card">{chart_html("chart_sqi_by_target.png", cache_key)}</div>
                    <div class="chart-card">{chart_html("chart_servqual_feature_importance.png", cache_key)}</div>
                    <div class="chart-card">{chart_html("chart_app_rating_leave_risk.png", cache_key)}</div>
                    <div class="chart-card">{chart_html("chart_resolution_days_leave_risk.png", cache_key)}</div>
                    <div class="chart-card">{chart_html("chart_relationship_manager_retention.png", cache_key)}</div>
                </div>
            </section>

            <section class="panel">
                <h2>Operational Tables</h2>
                <div class="tables">
                    <div>
                        <h3>Threshold Summary</h3>
                        <div class="table-wrap">{dataframe_to_html(threshold_df)}</div>
                    </div>
                    <div>
                        <h3>Relationship Manager Effect</h3>
                        <div class="table-wrap">{dataframe_to_html(relationship_manager_df)}</div>
                    </div>
                    <div>
                        <h3>SERVQUAL Correlations</h3>
                        <div class="table-wrap">{dataframe_to_html(correlations_df)}</div>
                    </div>
                    <div>
                        <h3>Feature Importance</h3>
                        <div class="table-wrap">{dataframe_to_html(feature_importance_df.head(10))}</div>
                    </div>
                </div>
            </section>

            <section class="panel">
                <h2>Highest-Risk Customers in Current View</h2>
                <p>
                    These customers combine weak retention signals with the highest model-estimated
                    leave propensity, which makes them the best candidates for immediate service recovery.
                </p>
                <div class="table-wrap">{dataframe_to_html(top_customers)}</div>
            </section>
        </div>
    </body>
    </html>
    """


class DashboardHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/dashboard"):
            params = parse_qs(parsed.query)
            html_content = build_dashboard_html(params).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html_content)))
            self.end_headers()
            self.wfile.write(html_content)
            return

        super().do_GET()


def main():
    parser = argparse.ArgumentParser(
        description="Run the retail banking leave propensity dashboard."
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
