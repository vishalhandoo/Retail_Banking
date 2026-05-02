import argparse
import html
import json
import os
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlencode, urlparse
from urllib.request import Request, urlopen

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "final_project_output.csv"
ENV_FILE = BASE_DIR / ".env"
RISK_ORDER = ["Critical", "High"]
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
CHAT_HISTORY_LIMIT = 6
ROLE_MODES = {
    "branch_staff": {
        "label": "Branch Staff",
        "focus": (
            "Focus on in-person service recovery, branch ownership, documentation clarity, complaint follow-up, "
            "and practical account servicing actions the branch can control."
        ),
    },
    "relationship_manager": {
        "label": "Relationship Manager",
        "focus": (
            "Focus on relationship deepening, trust repair, high-value retention, coordinated follow-up, "
            "and commercially sensible offers that fit the customer's risk guardrails."
        ),
    },
    "call_center": {
        "label": "Call Center",
        "focus": (
            "Focus on short phone scripts, empathy, issue triage, objection handling, escalation triggers, "
            "and actions that can be completed or promised during a call."
        ),
    },
}
SERVQUAL_COLUMNS = [
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
    "servqual_empathy": "Empathy",
    "servqual_tangibles": "Digital experience",
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


def load_priority_customers():
    if not DATA_FILE.exists():
        return pd.DataFrame()

    df = pd.read_csv(DATA_FILE)
    if "leave_risk_segment" not in df.columns:
        return pd.DataFrame()

    df = df[df["leave_risk_segment"].isin(RISK_ORDER)].copy()
    if df.empty:
        return df

    for column in [
        "customer_id",
        "predicted_leave_probability",
        "retention_score",
        "service_quality_index",
        "app_rating_score",
        "resolution_days",
        "loan_amount",
        "emi",
        "income",
        "credit_score",
        "product_count",
        "complaint_count",
        "has_relationship_manager",
        "credit_review_required",
    ] + SERVQUAL_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    df["priority_rank"] = df["leave_risk_segment"].map({"Critical": 0, "High": 1}).fillna(9)
    df = df.sort_values(
        by=["priority_rank", "predicted_leave_probability", "risk_adjusted_retention_score"],
        ascending=[True, False, False],
        na_position="last",
    )
    return df


def format_number(value, decimals=0):
    if pd.isna(value):
        return "NA"
    return f"{value:,.{decimals}f}"


def format_percent(value, decimals=1):
    if pd.isna(value):
        return "NA"
    return f"{value * 100:.{decimals}f}%"


def format_currency(value):
    if pd.isna(value):
        return "NA"
    return f"Rs {value:,.0f}"


def get_param(params, key, default=""):
    return params.get(key, [default])[0].strip()


def build_query(customer_id="", risk_filter="All", search_text=""):
    payload = {}
    if customer_id:
        payload["customer_id"] = customer_id
    if risk_filter and risk_filter != "All":
        payload["risk"] = risk_filter
    if search_text:
        payload["q"] = search_text
    query = urlencode(payload)
    return f"/?{query}" if query else "/"


def filter_priority_customers(df, risk_filter, search_text):
    filtered = df.copy()
    if risk_filter in RISK_ORDER:
        filtered = filtered[filtered["leave_risk_segment"] == risk_filter]

    if search_text:
        term = search_text.lower()
        searchable = filtered[
            [
                "customer_id",
                "loan_type",
                "recommended_action",
                "customer_value_tier",
                "customer_risk_profile",
                "retention_strategy_lane",
                "primary_channel",
            ]
        ].fillna("")
        mask = searchable.astype(str).apply(
            lambda column: column.str.lower().str.contains(term, regex=False)
        )
        filtered = filtered[mask.any(axis=1)]

    return filtered


def choose_selected_customer(filtered_df, customer_id):
    if filtered_df.empty:
        return None

    if customer_id:
        matched = filtered_df[filtered_df["customer_id"].astype(str) == customer_id]
        if not matched.empty:
            return matched.iloc[0]

    return filtered_df.iloc[0]


def build_driver_chips(customer):
    chips = []
    if customer.get("credit_review_required", 0) == 1:
        chips.append("Credit approval required before offers")
    if customer.get("has_relationship_manager", 0) == 0:
        chips.append("No relationship manager assigned")
    if not pd.isna(customer.get("complaint_count")) and customer["complaint_count"] > 0:
        chips.append(f"{format_number(customer['complaint_count'])} complaint(s) recorded")
    if not pd.isna(customer.get("app_rating_score")) and customer["app_rating_score"] <= 3:
        chips.append("Weak app rating")
    if not pd.isna(customer.get("resolution_days")) and customer["resolution_days"] >= 7:
        chips.append("Complaint resolution is slow")
    if not pd.isna(customer.get("credit_score")) and customer["credit_score"] < 600:
        chips.append("Weak credit score")
    if not pd.isna(customer.get("service_quality_index")) and customer["service_quality_index"] < 0.45:
        chips.append("Low service quality index")
    return chips or ["Urgent relationship review recommended"]


def build_frontline_guidance(customer):
    focus_points = []
    safe_offers = []
    avoid_offers = []

    credit_score = customer.get("credit_score")
    service_quality = customer.get("service_quality_index")
    resolution_days = customer.get("resolution_days")
    app_rating = customer.get("app_rating_score")
    has_rm = customer.get("has_relationship_manager", 0)
    credit_review_required = customer.get("credit_review_required", 0)
    product_count = customer.get("product_count")
    loan_type = str(customer.get("loan_type", "loan relationship")).strip() or "loan relationship"
    risk_label = str(customer.get("leave_risk_segment", "High"))

    if not pd.isna(service_quality) and service_quality < 0.45:
        focus_points.append("Repair the service experience first before discussing sales.")
    if not pd.isna(resolution_days) and resolution_days >= 7:
        focus_points.append("Acknowledge complaint delays and commit to a faster resolution path.")
    if not pd.isna(app_rating) and app_rating <= 3:
        focus_points.append("Check digital friction points and help with app usage or login issues.")
    if has_rm == 0:
        focus_points.append("Create named ownership so the customer knows exactly who will follow up.")
    if not pd.isna(credit_score) and credit_score < 600:
        focus_points.append("Keep the conversation trust-based and support-led because the credit profile is weak.")
    if risk_label == "Critical":
        focus_points.append("Prioritize immediate contact within the same working day.")

    if credit_review_required == 1 or (not pd.isna(credit_score) and credit_score < 600):
        safe_offers.append("Service recovery callback with a committed resolution owner.")
        safe_offers.append("EMI date alignment or payment assistance discussion if policy allows.")
        safe_offers.append("Basic digital help, alerts setup, and proactive account servicing.")
        avoid_offers.append("Fresh unsecured lending or aggressive top-up offers.")
        avoid_offers.append("Instant cross-sell pitches before credit review is cleared.")
    else:
        safe_offers.append("Relationship deepening conversation linked to the current banking need.")
        safe_offers.append("Rate review, balance transfer check, or tenure optimisation discussion.")
        safe_offers.append("Relevant bundled products only after service concerns are addressed.")

    if not pd.isna(product_count) and product_count <= 2:
        safe_offers.append("Simple savings or auto-pay setup to improve stickiness without pressure.")
    elif not pd.isna(product_count) and product_count >= 5:
        focus_points.append("Protect the full relationship value and check if any linked product is causing dissatisfaction.")

    if loan_type.lower() in {"home", "home loan"}:
        safe_offers.append("Home-loan servicing review, rate-check, and tenure support discussion.")
    elif "car" in loan_type.lower():
        safe_offers.append("Vehicle-loan service review and repayment convenience discussion.")
    elif loan_type.lower() == "personal":
        avoid_offers.append("Extra unsecured exposure unless the affordability picture is clearly healthy.")

    if not focus_points:
        focus_points.append("Start with a quick relationship review and confirm the customer's immediate concern.")
    if not avoid_offers:
        avoid_offers.append("Avoid generic product pitching until the reason for dissatisfaction is clear.")

    return {
        "primary_focus": focus_points[0],
        "focus_points": focus_points[:4],
        "safe_offers": safe_offers[:4],
        "avoid_offers": avoid_offers[:3],
    }


def normalize_chat_history(history):
    cleaned_history = []
    if not isinstance(history, list):
        return cleaned_history

    for item in history[-CHAT_HISTORY_LIMIT:]:
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
    for candidate in candidates:
        content = candidate.get("content") or {}
        parts = content.get("parts") or []
        text_parts = [part.get("text", "") for part in parts if isinstance(part, dict)]
        combined = "".join(text_parts).strip()
        if combined:
            return combined
    return ""


def build_frontline_ai_system_prompt():
    return """
You are Decipher AI, the frontline retention assistant for an internal Indian retail banking dashboard.
Your audience is branch staff, call-center staff, and relationship managers handling at-risk customers.

Always follow these rules:
- Ground every answer in the provided customer context and dashboard signals.
- Do not invent facts, balances, complaints, or approvals that are not in the context.
- Be practical, concise, and action-oriented.
- Focus on retention, service recovery, responsible selling, and next-best action.
- When credit review is required or the credit profile is weak, avoid aggressive selling.
- Prefer plain business language that a frontline worker can use immediately.
- If the user asks what to say, give a short talk track.
- If the data is insufficient, say that clearly and give a safe best-practice next step.
- Never advise policy violations, hidden fees, or misleading sales tactics.

Use this response structure:
Suggested approach:
What to say:
What to offer:
What to avoid:
""".strip()


def normalize_role_mode(role_mode):
    return role_mode if role_mode in ROLE_MODES else "branch_staff"


def build_role_mode_context(role_mode):
    resolved_role_mode = normalize_role_mode(role_mode)
    role_meta = ROLE_MODES[resolved_role_mode]
    return (
        f"ACTIVE ROLE MODE: {role_meta['label']}\n"
        f"ROLE-SPECIFIC GUIDANCE: {role_meta['focus']}"
    )


def build_customer_ai_context(customer):
    guidance = build_frontline_guidance(customer)
    service_lines = []
    for column in SERVQUAL_COLUMNS:
        service_lines.append(
            f"{SERVQUAL_LABELS[column]}: {format_number(customer.get(column), 2)}"
        )

    return "\n".join(
        [
            "FRONTLINE CUSTOMER CONTEXT",
            f"Customer ID: {customer.get('customer_id', 'NA')}",
            f"Risk segment: {customer.get('leave_risk_segment', 'NA')}",
            f"Predicted leave probability: {format_percent(customer.get('predicted_leave_probability'), 1)}",
            f"Retention score: {format_number(customer.get('retention_score'), 2)}",
            f"Service quality index: {format_number(customer.get('service_quality_index'), 2)}",
            f"Loan type: {customer.get('loan_type', 'NA')}",
            f"Loan amount: {format_currency(customer.get('loan_amount'))}",
            f"EMI: {format_currency(customer.get('emi'))}",
            f"Income: {format_currency(customer.get('income'))}",
            f"Credit score: {format_number(customer.get('credit_score'), 0)}",
            f"Complaint count: {format_number(customer.get('complaint_count'), 0)}",
            f"Resolution days: {format_number(customer.get('resolution_days'), 0)}",
            f"App rating: {format_number(customer.get('app_rating_score'), 0)}",
            f"Products held: {format_number(customer.get('product_count'), 0)}",
            f"Primary channel: {customer.get('primary_channel', 'NA')}",
            f"Relationship manager assigned: {'Yes' if customer.get('has_relationship_manager', 0) == 1 else 'No'}",
            f"Credit review required: {'Yes' if customer.get('credit_review_required', 0) == 1 else 'No'}",
            f"Recommended action from dashboard: {customer.get('recommended_action', 'Review account')}",
            f"Retention strategy lane: {customer.get('retention_strategy_lane', 'NA')}",
            f"Risk profile: {customer.get('customer_risk_profile', 'NA')}",
            f"Action owner: {customer.get('retention_action_owner', 'NA')}",
            "",
            "SERVICE DIAGNOSTICS",
            *service_lines,
            "",
            "FRONTLINE GUIDANCE",
            f"Primary focus: {guidance['primary_focus']}",
            "Focus points: " + "; ".join(guidance["focus_points"]),
            "Safe offers: " + "; ".join(guidance["safe_offers"]),
            "Avoid offers: " + "; ".join(guidance["avoid_offers"]),
        ]
    )


def looks_like_four_day_plan_request(question):
    question_text = str(question).lower()
    return (
        "4 day" in question_text
        or "4-day" in question_text
        or "four day" in question_text
        or "four-day" in question_text
    )


def build_four_day_retention_plan(customer, role_mode):
    guidance = build_frontline_guidance(customer)
    role_label = ROLE_MODES[normalize_role_mode(role_mode)]["label"]
    safe_offer = guidance["safe_offers"][0] if guidance["safe_offers"] else "Service recovery callback with named ownership."
    avoid_offer = guidance["avoid_offers"][0] if guidance["avoid_offers"] else "Generic product pitching before trust is restored."
    first_focus = guidance["focus_points"][0] if guidance["focus_points"] else guidance["primary_focus"]
    second_focus = guidance["focus_points"][1] if len(guidance["focus_points"]) > 1 else "Confirm the customer's immediate concern and remove friction."
    recommended_action = str(customer.get("recommended_action", "Review account"))
    loan_type = str(customer.get("loan_type", "loan relationship"))
    rm_status = "relationship manager assigned" if customer.get("has_relationship_manager", 0) == 1 else "no relationship manager assigned"

    return "\n\n".join(
        [
            f"Suggested approach:\nUse a structured 4-day retention plan for the selected customer in {role_label} mode, centered on service recovery, trust rebuilding, and controlled offers.",
            (
                "Day 1:\n"
                f"Make contact and acknowledge the current risk context. Focus first on {first_focus.lower()} "
                f"and explain that the bank is reviewing the {loan_type.lower()} relationship carefully. "
                f"Confirm whether there are unresolved concerns and note that there is currently {rm_status}."
            ),
            (
                "Day 2:\n"
                f"Act on the main service issue. Focus on {second_focus.lower()} and make sure there is a named owner "
                f"for follow-up. Reinforce the dashboard recommendation: {recommended_action.lower()}."
            ),
            (
                "Day 3:\n"
                f"Offer one safe retention step only: {safe_offer.lower()} "
                "Keep the conversation support-led rather than sales-led, especially if the customer is frustrated or risk-sensitive."
            ),
            (
                "Day 4:\n"
                "Close the loop with a follow-up call or message, confirm whether the customer feels the issue is improving, "
                "and document the next checkpoint. If the customer still sounds disengaged, escalate to a senior owner or RM immediately."
            ),
            (
                "What to avoid:\n"
                f"{avoid_offer} Do not push multiple offers in the same interaction before the service issue is stabilized."
            ),
        ]
    )


def ask_gemini_frontline_assistant(question, customer_context, role_mode, history=None):
    api_key = get_gemini_api_key()
    if not api_key:
        return {
            "ok": False,
            "status": HTTPStatus.SERVICE_UNAVAILABLE,
            "message": (
                "Decipher AI is not configured yet because `GEMINI_API_KEY` is missing. "
                "Add it to `.env`, restart the preview, and try again."
            ),
        }

    contents = []
    for item in normalize_chat_history(history):
        contents.append(
            {
                "role": "model" if item["role"] == "assistant" else "user",
                "parts": [{"text": item["content"]}],
            }
        )

    user_prompt = (
        "Use the role mode and customer context below to answer the frontline worker's question.\n\n"
        f"{build_role_mode_context(role_mode)}\n\n"
        f"{customer_context}\n\n"
        f"FRONTLINE QUESTION:\n{question.strip()}"
    )
    contents.append({"role": "user", "parts": [{"text": user_prompt}]})
    payload = {
        "systemInstruction": {"parts": [{"text": build_frontline_ai_system_prompt()}]},
        "contents": contents,
        "generationConfig": {
            "temperature": 0.3,
            "topP": 0.9,
            "maxOutputTokens": 1100,
        },
    }

    request = Request(
        GEMINI_API_URL.format(model=get_gemini_model()),
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
        message = "The Gemini API returned an error."
        if error_body:
            try:
                parsed_error = json.loads(error_body)
                message = parsed_error.get("error", {}).get("message") or message
            except json.JSONDecodeError:
                message = error_body[:500]
        return {
            "ok": False,
            "status": exc.code,
            "message": f"Decipher AI could not answer right now: {message}",
        }
    except URLError:
        return {
            "ok": False,
            "status": HTTPStatus.BAD_GATEWAY,
            "message": (
                "Decipher AI could not reach Gemini. Check internet access, firewall settings, "
                "and the API key, then try again."
            ),
        }
    except Exception as exc:
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
            "message": "Decipher AI did not return a usable answer. Please try a more specific question.",
        }

    return {"ok": True, "status": HTTPStatus.OK, "message": answer_text}


def build_service_bars(customer):
    blocks = []
    for column in SERVQUAL_COLUMNS:
        value = customer.get(column)
        width = 0 if pd.isna(value) else max(0, min(100, round(float(value) * 100)))
        blocks.append(
            f"""
            <div class="service-card">
                <div class="mini-label">{html.escape(SERVQUAL_LABELS[column])}</div>
                <div class="mini-value">{format_number(value, 2)}</div>
                <div class="bar"><div class="bar-fill" style="width: {width}%;"></div></div>
            </div>
            """
        )
    return "".join(blocks)


def build_table_rows(filtered_df, selected_customer_id, risk_filter, search_text):
    rows = []
    for _, customer in filtered_df.iterrows():
        customer_id = str(customer["customer_id"])
        row_class = "selected" if customer_id == selected_customer_id else ""
        rm_status = "Yes" if customer.get("has_relationship_manager", 0) == 1 else "No"
        row_url = build_query(customer_id, risk_filter, search_text)
        rows.append(
            f"""
            <tr class="customer-row {row_class}" data-customer-id="{html.escape(customer_id)}" tabindex="0" role="button" aria-label="Open details for customer {html.escape(customer_id)}" onclick="window.location.href='{html.escape(row_url)}'">
                <td><a class="row-link" href="{html.escape(row_url)}">#{html.escape(customer_id)}</a></td>
                <td>{html.escape(str(customer.get('leave_risk_segment', 'NA')))}</td>
                <td>{format_percent(customer.get('predicted_leave_probability'), 1)}</td>
                <td>{html.escape(str(customer.get('loan_type', 'NA')))}</td>
                <td>{format_number(customer.get('retention_score'), 2)}</td>
                <td>{format_number(customer.get('service_quality_index'), 2)}</td>
                <td>{rm_status}</td>
                <td>{html.escape(str(customer.get('recommended_action', 'Review account')))}</td>
            </tr>
            """
        )
    return "".join(rows)


def build_customer_payload(filtered_df):
    payload = {}
    for _, customer in filtered_df.iterrows():
        customer_id = str(customer["customer_id"])
        risk_label = str(customer.get("leave_risk_segment", "NA"))
        rm_status = "Assigned" if customer.get("has_relationship_manager", 0) == 1 else "Not assigned"
        review_status = "Required" if customer.get("credit_review_required", 0) == 1 else "Not required"
        payload[customer_id] = {
            "customer_id": customer_id,
            "risk_label": risk_label,
            "rm_status": rm_status,
            "review_status": review_status,
            "loan_type": str(customer.get("loan_type", "NA")),
            "loan_amount": format_currency(customer.get("loan_amount")),
            "emi": format_currency(customer.get("emi")),
            "leave_probability": format_percent(customer.get("predicted_leave_probability"), 1),
            "service_quality": format_number(customer.get("service_quality_index"), 2),
            "product_count": format_number(customer.get("product_count"), 0),
            "customer_value_tier": str(customer.get("customer_value_tier", "NA")),
            "primary_channel": str(customer.get("primary_channel", "NA")),
            "age": format_number(customer.get("age"), 0),
            "income": format_currency(customer.get("income")),
            "credit_score": format_number(customer.get("credit_score"), 0),
            "complaint_count": format_number(customer.get("complaint_count"), 0),
            "resolution_days": format_number(customer.get("resolution_days"), 0),
            "retention_score": format_number(customer.get("retention_score"), 2),
            "recommended_action": str(customer.get("recommended_action", "Review account")),
            "retention_strategy_lane": str(customer.get("retention_strategy_lane", "NA")),
            "customer_risk_profile": str(customer.get("customer_risk_profile", "NA")),
            "retention_action_owner": str(customer.get("retention_action_owner", "NA")),
            "driver_chips": build_driver_chips(customer),
            "guidance": build_frontline_guidance(customer),
            "service_bars": [
                {
                    "label": SERVQUAL_LABELS[column],
                    "value": format_number(customer.get(column), 2),
                    "width": 0
                    if pd.isna(customer.get(column))
                    else max(0, min(100, round(float(customer.get(column)) * 100))),
                }
                for column in SERVQUAL_COLUMNS
            ],
        }
    return json.dumps(payload).replace("</", "<\\/")


def summary_card(label, value, detail):
    return f"""
    <article class="summary-card">
        <span>{html.escape(label)}</span>
        <strong>{html.escape(value)}</strong>
        <small>{html.escape(detail)}</small>
    </article>
    """


def build_working_panel_html(selected_customer):
    if selected_customer is None:
        return """
        <section class="panel table-panel" id="working-panel">
            <div class="table-subtle">How it works</div>
            <h2>Popup customer detail</h2>
            <p class="helper-copy">Single click updates this working panel so the frontline user can stay in the queue. Double click opens the full Customer 360 overlay.</p>
            <p class="helper-copy">Use this panel for retention guidance: what to focus on, what to offer, and what to avoid for the selected customer.</p>
        </section>
        """

    guidance = build_frontline_guidance(selected_customer)
    risk_label = str(selected_customer.get("leave_risk_segment", "NA"))
    review_status = "Required" if selected_customer.get("credit_review_required", 0) == 1 else "Not required"
    rm_status = "Assigned" if selected_customer.get("has_relationship_manager", 0) == 1 else "Not assigned"
    safe_offer_items = "".join(
        f"<li>{html.escape(item)}</li>" for item in guidance["safe_offers"]
    )
    avoid_offer_items = "".join(
        f"<li>{html.escape(item)}</li>" for item in guidance["avoid_offers"]
    )

    return f"""
    <section class="panel table-panel" id="working-panel">
        <div class="table-subtle">Selected customer</div>
        <h2>Customer #{html.escape(str(selected_customer.get("customer_id", "NA")))}</h2>
        <p class="helper-copy">{html.escape(risk_label)} risk, {format_percent(selected_customer.get("predicted_leave_probability"), 1)} leave probability, {html.escape(rm_status.lower())} RM.</p>
        <div class="chip-row" style="margin-top: 12px;">
            <span class="badge badge-risk">{html.escape(risk_label)}</span>
            <span class="badge">{html.escape(review_status)} credit review</span>
            <span class="badge">{html.escape(rm_status)} RM</span>
        </div>
        <div class="detail-grid" style="margin-top: 16px;">
            <article class="info-card">
                <span class="mini-label">Loan context</span>
                <strong>{html.escape(str(selected_customer.get("loan_type", "NA")))}</strong>
                <p>{format_currency(selected_customer.get("loan_amount"))} outstanding with EMI {format_currency(selected_customer.get("emi"))}.</p>
            </article>
            <article class="info-card">
                <span class="mini-label">Service quality</span>
                <strong>{format_number(selected_customer.get("service_quality_index"), 2)}</strong>
                <p>Current relationship quality score for this customer.</p>
            </article>
        </div>
        <div class="action-callout" style="margin-top: 14px;">{html.escape(str(selected_customer.get("recommended_action", "Review account")))}</div>
        <div class="guidance-grid">
            <div class="guidance-card">
                <div class="mini-label">Primary focus</div>
                <p class="helper-copy" style="margin-top: 8px;">{html.escape(guidance["primary_focus"])}</p>
            </div>
            <div class="guidance-card">
                <div class="mini-label">What to offer</div>
                <ul>{safe_offer_items}</ul>
            </div>
            <div class="guidance-card">
                <div class="mini-label">What to avoid</div>
                <ul>{avoid_offer_items}</ul>
            </div>
        </div>
        <p class="helper-copy" style="margin-top: 12px;">Single click selects this customer. Double click opens the full popup overlay.</p>
    </section>
    """


def build_ai_query(customer_id, risk_filter, search_text, role_mode="", ai_question=""):
    payload = {}
    if customer_id:
        payload["customer_id"] = customer_id
    if risk_filter and risk_filter != "All":
        payload["risk"] = risk_filter
    if search_text:
        payload["q"] = search_text
    if role_mode:
        payload["role_mode"] = role_mode
    if ai_question:
        payload["ai_question"] = ai_question
    query = urlencode(payload)
    return f"/?{query}" if query else "/"


def build_ai_panel_html(selected_customer, risk_filter, search_text, role_mode, ai_question, ai_response):
    selected_role_mode = normalize_role_mode(role_mode)
    current_customer_label = (
        f"#{html.escape(str(selected_customer.get('customer_id', 'NA')))}"
        if selected_customer is not None
        else "None"
    )
    prompt_options = [
        "What should I say to retain this customer?",
        "What is the safest offer for this customer right now?",
        "What should I avoid saying or offering to this customer?",
        "Summarize the top reasons this customer may leave and what I should fix first.",
    ]
    prompt_links = []
    for prompt in prompt_options:
        prompt_links.append(
            f'<a class="prompt-chip" href="{html.escape(build_ai_query("" if selected_customer is None else str(selected_customer.get("customer_id", "")), risk_filter, search_text, selected_role_mode, prompt))}">{html.escape(prompt)}</a>'
        )

    response_html = ""
    if ai_response:
        response_html = build_ai_response_html(ai_response)

    return f"""
    <section class="panel copilot-panel">
        <div class="table-subtle">Decipher AI</div>
        <h2>Retention guidance for the selected customer</h2>
        <p class="copilot-status">
            Ask for a talk track, a safe offer suggestion, a complaint-handling approach, or what the worker should avoid.
            Current selected customer: <strong id="selected-customer-label">{current_customer_label}</strong>.
        </p>
        <form method="get" action="/" class="chat-form" id="frontline-chat-form">
            <input type="hidden" name="customer_id" value="{html.escape("" if selected_customer is None else str(selected_customer.get("customer_id", "")))}">
            <input type="hidden" name="risk" value="{html.escape(risk_filter if risk_filter != "All" else "")}">
            <input type="hidden" name="q" value="{html.escape(search_text)}">
            <label style="min-width: 220px;">
                <span class="table-subtle">Role mode</span>
                <select id="role-mode-select" name="role_mode">
                    <option value="branch_staff" {"selected" if selected_role_mode == "branch_staff" else ""}>Branch Staff</option>
                    <option value="relationship_manager" {"selected" if selected_role_mode == "relationship_manager" else ""}>Relationship Manager</option>
                    <option value="call_center" {"selected" if selected_role_mode == "call_center" else ""}>Call Center</option>
                </select>
            </label>
            <div class="prompt-row">
                {''.join(prompt_links)}
            </div>
            <label for="frontline-chat-input" class="table-subtle">Ask about the selected customer</label>
            <textarea id="frontline-chat-input" name="ai_question" placeholder="Example: Give me a short script for calling this customer today.">{html.escape(ai_question)}</textarea>
            <div class="chat-actions">
                <span class="copilot-status">Preferred format: practical steps the worker can use immediately.</span>
                <button type="submit">Ask Decipher AI</button>
            </div>
        </form>
        <div class="chat-log" id="frontline-chat-log">
            {response_html or '<article class="chat-message chat-message-assistant">Select a customer, then ask Decipher AI for a retention suggestion.</article>'}
        </div>
    </section>
    """


def build_ai_response_html(ai_response):
    sections = []
    for block in str(ai_response).strip().split("\n\n"):
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines:
            continue
        rendered_lines = []
        for line in lines:
            escaped_line = html.escape(line)
            if escaped_line.startswith("**") and escaped_line.endswith("**") and len(escaped_line) > 4:
                escaped_line = f"<strong>{escaped_line[2:-2]}</strong>"
            elif ": " in escaped_line and not escaped_line.startswith("- "):
                label, value = escaped_line.split(": ", 1)
                escaped_line = f"<strong>{label}:</strong> {value}"
            rendered_lines.append(f"<p>{escaped_line}</p>")
        sections.append("".join(rendered_lines))

    return '<article class="chat-message chat-message-assistant">{}</article>'.format(
        "".join(sections)
    )


def build_dashboard_html(params):
    df = load_priority_customers()
    risk_filter = get_param(params, "risk", "All")
    search_text = get_param(params, "q", "")
    requested_customer_id = get_param(params, "customer_id", "")
    role_mode = normalize_role_mode(get_param(params, "role_mode", "branch_staff"))
    ai_question = get_param(params, "ai_question", "")

    if df.empty:
        return """
        <!DOCTYPE html>
        <html lang="en">
        <head><meta charset="utf-8"><title>Customer 360 Priority Preview</title></head>
        <body style="font-family: Arial, sans-serif; padding: 24px;">
            <h1>Customer 360 Priority Preview</h1>
            <p>No priority customer data is available. Make sure final_project_output.csv exists and contains High/Critical customer rows.</p>
        </body>
        </html>
        """

    filtered_df = filter_priority_customers(df, risk_filter, search_text)
    selected_customer = choose_selected_customer(filtered_df, requested_customer_id)
    selected_customer_id = "" if selected_customer is None else str(selected_customer["customer_id"])

    critical_count = int((df["leave_risk_segment"] == "Critical").sum())
    high_count = int((df["leave_risk_segment"] == "High").sum())
    no_rm_count = int((df["has_relationship_manager"] == 0).sum())
    credit_review_count = int((df["credit_review_required"] == 1).sum())

    table_rows = build_table_rows(filtered_df, selected_customer_id, risk_filter, search_text)
    customer_payload = build_customer_payload(filtered_df)
    working_panel_html = build_working_panel_html(selected_customer)
    ai_response = ""
    if ai_question and selected_customer is not None:
        ai_result = ask_gemini_frontline_assistant(
            ai_question,
            build_customer_ai_context(selected_customer),
            role_mode,
            history=None,
        )
        ai_response = ai_result.get("message", "")
        if looks_like_four_day_plan_request(ai_question):
            normalized_response = ai_response.lower()
            if "day 2" not in normalized_response or "day 3" not in normalized_response or "day 4" not in normalized_response:
                ai_response = build_four_day_retention_plan(selected_customer, role_mode)
    ai_panel_html = build_ai_panel_html(
        selected_customer,
        risk_filter,
        search_text,
        role_mode,
        ai_question,
        ai_response,
    )

    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Customer 360 Priority Preview</title>
        <style>
            :root {{
                --bg: #f3eee4;
                --panel: rgba(255, 251, 245, 0.96);
                --ink: #162126;
                --muted: #5e6d73;
                --line: rgba(22, 33, 38, 0.1);
                --brand: #0b5e52;
                --brand-dark: #083e36;
                --accent: #ca8b2c;
                --alert: #b04636;
                --shadow: 0 18px 40px rgba(20, 31, 36, 0.12);
            }}

            * {{
                box-sizing: border-box;
            }}

            body {{
                margin: 0;
                font-family: Georgia, "Times New Roman", serif;
                color: var(--ink);
                background:
                    radial-gradient(circle at top left, rgba(202, 139, 44, 0.17), transparent 28%),
                    radial-gradient(circle at right top, rgba(11, 94, 82, 0.14), transparent 24%),
                    linear-gradient(180deg, #fbf7f0, var(--bg));
            }}

            .shell {{
                max-width: 1500px;
                margin: 0 auto;
                padding: 28px 18px 42px;
            }}

            .hero {{
                background: linear-gradient(150deg, rgba(11, 94, 82, 0.96), rgba(8, 62, 54, 0.94));
                color: #f9f5ee;
                border-radius: 28px;
                padding: 28px;
                box-shadow: var(--shadow);
            }}

            .eyebrow {{
                display: inline-block;
                padding: 7px 12px;
                border-radius: 999px;
                background: rgba(255, 255, 255, 0.14);
                font-size: 0.82rem;
                letter-spacing: 0.05em;
                text-transform: uppercase;
            }}

            h1, h2, h3, p {{
                margin: 0;
            }}

            h1 {{
                margin-top: 14px;
                font-size: clamp(2rem, 4vw, 3.4rem);
                line-height: 0.96;
                max-width: 10ch;
            }}

            .hero p {{
                margin-top: 14px;
                max-width: 68ch;
                color: rgba(249, 245, 238, 0.82);
                line-height: 1.55;
            }}

            .summary-grid {{
                display: grid;
                grid-template-columns: repeat(4, minmax(0, 1fr));
                gap: 12px;
                margin-top: 22px;
            }}

            .summary-card,
            .panel,
            .table-panel {{
                background: var(--panel);
                border: 1px solid var(--line);
                border-radius: 22px;
                box-shadow: var(--shadow);
            }}

            .summary-card {{
                padding: 16px;
                color: var(--ink);
            }}

            .summary-card span,
            .mini-label,
            .table-subtle {{
                color: var(--muted);
                font-size: 0.82rem;
                letter-spacing: 0.04em;
                text-transform: uppercase;
            }}

            .summary-card strong {{
                display: block;
                margin-top: 8px;
                font-size: 1.5rem;
            }}

            .summary-card small {{
                display: block;
                margin-top: 6px;
                color: var(--muted);
                line-height: 1.45;
            }}

            .controls {{
                margin-top: 18px;
                padding: 18px;
            }}

            .controls form {{
                display: grid;
                grid-template-columns: 180px 1fr auto;
                gap: 12px;
                align-items: end;
            }}

            label {{
                display: grid;
                gap: 6px;
                font-size: 0.92rem;
            }}

            input, select, button {{
                font: inherit;
                padding: 12px 14px;
                border-radius: 14px;
                border: 1px solid var(--line);
                background: #fffdfa;
            }}

            button {{
                background: var(--brand);
                color: white;
                border: none;
                cursor: pointer;
            }}

            .content {{
                display: grid;
                grid-template-columns: 0.95fr 1.35fr;
                gap: 18px;
                margin-top: 18px;
            }}

            .table-panel {{
                padding: 18px;
                min-width: 0;
            }}

            .table-wrap {{
                margin-top: 14px;
                max-height: 980px;
                overflow: auto;
                border-radius: 18px;
                border: 1px solid var(--line);
            }}

            table {{
                width: 100%;
                border-collapse: collapse;
                background: rgba(255, 255, 255, 0.72);
            }}

            th, td {{
                padding: 12px 10px;
                border-bottom: 1px solid rgba(22, 33, 38, 0.08);
                text-align: left;
                vertical-align: top;
                font-size: 0.95rem;
            }}

            th {{
                position: sticky;
                top: 0;
                background: #f7f1e8;
                z-index: 1;
            }}

            td a {{
                color: var(--brand-dark);
                text-decoration: none;
                font-weight: 700;
            }}

            .customer-row {{
                cursor: pointer;
            }}

            .customer-row:hover,
            .customer-row:focus {{
                background: rgba(11, 94, 82, 0.06);
                outline: none;
            }}

            tr.selected {{
                background: rgba(11, 94, 82, 0.08);
            }}

            .row-link {{
                padding: 0;
                border: 0;
                background: transparent;
                color: var(--brand-dark);
                font: inherit;
                font-weight: 700;
                cursor: pointer;
            }}

            .popup-hint {{
                margin-top: 10px;
                color: var(--muted);
                line-height: 1.45;
            }}

            .modal-shell {{
                position: fixed;
                inset: 0;
                display: none;
                align-items: center;
                justify-content: center;
                padding: 24px;
                background: rgba(10, 18, 22, 0.56);
                backdrop-filter: blur(8px);
                z-index: 999;
            }}

            .modal-shell.open {{
                display: flex;
            }}

            .modal-card {{
                width: min(1120px, 100%);
                max-height: 90vh;
                overflow: auto;
                padding: 22px;
                background: var(--panel);
                border: 1px solid var(--line);
                border-radius: 26px;
                box-shadow: 0 28px 70px rgba(10, 18, 22, 0.28);
            }}

            .detail-hero,
            .hero-badges,
            .chip-row {{
                display: flex;
            }}

            .detail-hero,
            .hero-badges,
            .chip-row {{
                flex-wrap: wrap;
                gap: 8px;
            }}

            .detail-hero {{
                justify-content: space-between;
                align-items: start;
                gap: 16px;
            }}

            .detail-hero p {{
                margin-top: 10px;
                color: var(--muted);
                line-height: 1.5;
                max-width: 60ch;
            }}

            .badge,
            .chip {{
                padding: 8px 12px;
                border-radius: 999px;
                background: #efe3cf;
                font-size: 0.9rem;
            }}

            .badge-risk {{
                background: rgba(176, 70, 54, 0.14);
                color: #8c2f24;
            }}

            .detail-grid,
            .snapshot-grid,
            .service-grid {{
                display: grid;
                gap: 12px;
            }}

            .detail-grid {{
                grid-template-columns: repeat(4, minmax(0, 1fr));
                margin-top: 18px;
            }}

            .info-card,
            .inset,
            .service-card {{
                background: rgba(255, 255, 255, 0.76);
                border: 1px solid var(--line);
                border-radius: 18px;
                padding: 16px;
            }}

            .info-card strong,
            .snapshot-grid strong,
            .service-card .mini-value {{
                display: block;
                margin-top: 8px;
                font-size: 1.2rem;
            }}

            .info-card p,
            .helper-copy {{
                margin-top: 8px;
                color: var(--muted);
                line-height: 1.45;
            }}

            .two-column {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 12px;
                margin-top: 12px;
            }}

            .inset h3 {{
                margin-bottom: 14px;
            }}

            .snapshot-grid {{
                grid-template-columns: repeat(3, minmax(0, 1fr));
            }}

            .snapshot-grid div {{
                padding: 12px;
                border-radius: 14px;
                background: rgba(243, 238, 228, 0.8);
            }}

            .action-callout {{
                padding: 14px 16px;
                border-left: 5px solid var(--brand);
                border-radius: 14px;
                background: rgba(11, 94, 82, 0.08);
                font-size: 1.05rem;
                line-height: 1.5;
            }}

            .guidance-grid {{
                display: grid;
                grid-template-columns: repeat(3, minmax(0, 1fr));
                gap: 12px;
                margin-top: 14px;
            }}

            .guidance-card {{
                padding: 16px;
                border-radius: 18px;
                border: 1px solid var(--line);
                background: rgba(255, 255, 255, 0.76);
            }}

            .guidance-card ul {{
                margin: 10px 0 0;
                padding-left: 18px;
                color: var(--muted);
                line-height: 1.5;
            }}

            .guidance-card li + li {{
                margin-top: 6px;
            }}

            .copilot-panel {{
                margin-top: 18px;
                padding: 18px;
            }}

            .copilot-status {{
                margin-top: 8px;
                color: var(--muted);
                line-height: 1.45;
            }}

            .prompt-row {{
                display: flex;
                flex-wrap: wrap;
                gap: 8px;
                margin-top: 14px;
            }}

            .prompt-chip {{
                padding: 8px 12px;
                border-radius: 999px;
                border: 1px solid var(--line);
                background: #fffdfa;
                color: var(--ink);
                cursor: pointer;
            }}

            .chat-log {{
                margin-top: 14px;
                display: grid;
                gap: 10px;
                min-height: 220px;
                max-height: 720px;
                overflow: auto;
                padding-right: 6px;
            }}

            .chat-message {{
                padding: 14px 16px;
                border-radius: 16px;
                border: 1px solid var(--line);
                line-height: 1.5;
                overflow-wrap: anywhere;
                word-break: break-word;
            }}

            .chat-message strong {{
                font-weight: 700;
            }}

            .chat-message p {{
                margin: 0;
            }}

            .chat-message p + p {{
                margin-top: 10px;
            }}

            .chat-message-user {{
                background: rgba(11, 94, 82, 0.08);
            }}

            .chat-message-assistant {{
                background: rgba(255, 255, 255, 0.78);
            }}

            .chat-message-system {{
                background: rgba(176, 70, 54, 0.08);
            }}

            .chat-form {{
                margin-top: 14px;
                display: grid;
                gap: 10px;
            }}

            .chat-form textarea {{
                min-height: 110px;
                resize: vertical;
                font: inherit;
                padding: 12px 14px;
                border-radius: 14px;
                border: 1px solid var(--line);
                background: #fffdfa;
            }}

            .chat-actions {{
                display: flex;
                align-items: center;
                justify-content: space-between;
                gap: 12px;
            }}

            .service-grid {{
                grid-template-columns: repeat(5, minmax(0, 1fr));
                margin-top: 12px;
            }}

            .modal-header-actions {{
                display: flex;
                align-items: center;
                gap: 10px;
            }}

            .close-button {{
                padding: 10px 14px;
                border-radius: 999px;
                border: 1px solid var(--line);
                background: #fffdfa;
                color: var(--ink);
                cursor: pointer;
            }}

            .bar {{
                height: 10px;
                margin-top: 12px;
                border-radius: 999px;
                background: #e8ddd1;
                overflow: hidden;
            }}

            .bar-fill {{
                height: 100%;
                border-radius: inherit;
                background: linear-gradient(90deg, var(--alert), var(--accent), var(--brand));
            }}

            @media (max-width: 1220px) {{
                .content {{
                    grid-template-columns: 1fr;
                }}

                .summary-grid,
                .detail-grid,
                .service-grid,
                .guidance-grid {{
                    grid-template-columns: repeat(2, minmax(0, 1fr));
                }}
            }}

            @media (max-width: 760px) {{
                .shell {{
                    padding: 18px 12px 30px;
                }}

                .controls form,
                .two-column,
                .detail-hero,
                .summary-grid,
                .detail-grid,
                .snapshot-grid,
                .service-grid,
                .guidance-grid {{
                    grid-template-columns: 1fr;
                    display: grid;
                }}
            }}
        </style>
    </head>
    <body>
        <main class="shell">
            <section class="hero">
                <div class="eyebrow">Standalone Preview</div>
                <h1>Customer 360 for every urgent and high-risk customer</h1>
                <p>
                    This separate preview turns the one-customer concept into a working priority portfolio. It keeps all High and Critical customers in one queue,
                    while showing a full Customer 360 detail pane for the selected relationship.
                </p>
                <div class="summary-grid">
                    {summary_card("Priority customers", format_number(len(df)), "All High and Critical relationships")}
                    {summary_card("Critical customers", format_number(critical_count), "Most urgent cases in the file")}
                    {summary_card("No RM assigned", format_number(no_rm_count), "Priority customers without relationship coverage")}
                    {summary_card("Credit review required", format_number(credit_review_count), "Cases with offer or action guardrails")}
                </div>
            </section>

            <section class="panel controls">
                <form method="get" action="/">
                    <label>
                        <span class="table-subtle">Risk filter</span>
                        <select name="risk">
                            <option value="All" {"selected" if risk_filter == "All" else ""}>All priority customers</option>
                            <option value="Critical" {"selected" if risk_filter == "Critical" else ""}>Critical only</option>
                            <option value="High" {"selected" if risk_filter == "High" else ""}>High only</option>
                        </select>
                    </label>
                    <label>
                        <span class="table-subtle">Search</span>
                        <input type="text" name="q" value="{html.escape(search_text)}" placeholder="customer id, loan type, action, lane...">
                    </label>
                    <button type="submit">Apply</button>
                </form>
            </section>

            <section class="content">
                <section class="table-panel">
                    <div class="table-subtle">Priority queue</div>
                    <h2>All urgent and high-risk customers</h2>
                    <p class="helper-copy">Showing {format_number(len(filtered_df))} customers. Single click loads the working panel. Double click opens the full Customer 360 popup.</p>
                    <p class="popup-hint">Current quick-select customer: <strong>#{html.escape(selected_customer_id) if selected_customer_id else "None"}</strong></p>
                    <div class="table-wrap">
                        <table>
                            <thead>
                                <tr>
                                    <th>Customer</th>
                                    <th>Risk</th>
                                    <th>Leave Prob.</th>
                                    <th>Loan</th>
                                    <th>Retention</th>
                                    <th>Service</th>
                                    <th>RM</th>
                                    <th>Recommended Action</th>
                                </tr>
                            </thead>
                            <tbody>{table_rows}</tbody>
                        </table>
                    </div>
                </section>
                {working_panel_html}
            </section>

            {ai_panel_html}
        </main>
        <div class="modal-shell" id="customer-modal" aria-hidden="true">
            <section class="modal-card" role="dialog" aria-modal="true" aria-labelledby="modal-customer-title">
                <div class="detail-hero">
                    <div>
                        <div class="eyebrow">Customer 360 Detail</div>
                        <h2 id="modal-customer-title">Customer</h2>
                        <p>This banker-facing view combines service friction, risk guardrails, relationship coverage, and the next action.</p>
                    </div>
                    <div class="modal-header-actions">
                        <div class="hero-badges" id="modal-badges"></div>
                        <button type="button" class="close-button" id="close-modal">Close</button>
                    </div>
                </div>

                <div class="detail-grid">
                    <article class="info-card">
                        <span class="mini-label">Loan context</span>
                        <strong id="modal-loan-type">NA</strong>
                        <p id="modal-loan-copy">NA</p>
                    </article>
                    <article class="info-card">
                        <span class="mini-label">Risk estimate</span>
                        <strong id="modal-leave-probability">NA</strong>
                        <p>Predicted probability that this relationship needs urgent retention attention.</p>
                    </article>
                    <article class="info-card">
                        <span class="mini-label">Service quality</span>
                        <strong id="modal-service-quality">NA</strong>
                        <p>Overall service experience score across reliability, responsiveness, assurance, empathy, and digital experience.</p>
                    </article>
                    <article class="info-card">
                        <span class="mini-label">Relationship depth</span>
                        <strong id="modal-product-count">NA</strong>
                        <p id="modal-relationship-copy">NA</p>
                    </article>
                </div>

                <div class="two-column">
                    <article class="panel inset">
                        <h3>Customer snapshot</h3>
                        <div class="snapshot-grid">
                            <div><span class="mini-label">Age</span><strong id="modal-age">NA</strong></div>
                            <div><span class="mini-label">Income</span><strong id="modal-income">NA</strong></div>
                            <div><span class="mini-label">Credit score</span><strong id="modal-credit-score">NA</strong></div>
                            <div><span class="mini-label">Complaints</span><strong id="modal-complaints">NA</strong></div>
                            <div><span class="mini-label">Resolution days</span><strong id="modal-resolution-days">NA</strong></div>
                            <div><span class="mini-label">Retention score</span><strong id="modal-retention-score">NA</strong></div>
                        </div>
                        <div class="chip-row" id="modal-driver-chips"></div>
                    </article>

                    <article class="panel inset">
                        <h3>Next-best action</h3>
                        <div class="action-callout" id="modal-recommended-action">Review account</div>
                        <p class="helper-copy">Retention lane: <strong id="modal-retention-lane">NA</strong></p>
                        <p class="helper-copy">Risk profile: <strong id="modal-risk-profile">NA</strong></p>
                        <p class="helper-copy">Action owner: <strong id="modal-action-owner">NA</strong></p>
                    </article>
                </div>

                <article class="panel inset">
                    <h3>Frontline retention guidance</h3>
                    <div class="action-callout" id="modal-primary-focus">NA</div>
                    <div class="guidance-grid">
                        <div class="guidance-card">
                            <div class="mini-label">Focus first</div>
                            <ul id="modal-focus-list"></ul>
                        </div>
                        <div class="guidance-card">
                            <div class="mini-label">Offer</div>
                            <ul id="modal-safe-offers"></ul>
                        </div>
                        <div class="guidance-card">
                            <div class="mini-label">Avoid</div>
                            <ul id="modal-avoid-offers"></ul>
                        </div>
                    </div>
                </article>

                <article class="panel inset">
                    <h3>Service diagnostics</h3>
                    <div class="service-grid" id="modal-service-grid"></div>
                </article>
            </section>
        </div>
        <script>
            const customerData = {customer_payload};
            const modal = document.getElementById("customer-modal");
            const closeModalButton = document.getElementById("close-modal");
            const rowElements = Array.from(document.querySelectorAll(".customer-row"));
            const chatLog = document.getElementById("frontline-chat-log");
            const chatForm = document.getElementById("frontline-chat-form");
            const chatInput = document.getElementById("frontline-chat-input");
            const selectedCustomerLabel = document.getElementById("selected-customer-label");
            const roleModeSelect = document.getElementById("role-mode-select");
            const promptChips = Array.from(document.querySelectorAll(".prompt-chip"));
            let lastFocusedRow = null;
            let selectedCustomerId = "{html.escape(selected_customer_id)}";
            let frontlineChatHistory = [];

            function setText(id, value) {{
                const element = document.getElementById(id);
                if (element) {{
                    element.textContent = value;
                }}
            }}

            function renderChatMessage(role, content) {{
                const article = document.createElement("article");
                article.className = `chat-message chat-message-${{role}}`;
                if (role === "assistant") {{
                    const escaped = content
                        .replace(/&/g, "&amp;")
                        .replace(/</g, "&lt;")
                        .replace(/>/g, "&gt;");
                    const withBold = escaped.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
                    const paragraphs = withBold
                        .split(/\n\s*\n/)
                        .map((block) => "<p>" + block.split("\\n").join("<br>") + "</p>")
                        .join("");
                    article.innerHTML = paragraphs;
                }} else {{
                    article.textContent = content;
                }}
                chatLog.appendChild(article);
                chatLog.scrollTop = chatLog.scrollHeight;
            }}

            function renderBadges(customer) {{
                const badgeHost = document.getElementById("modal-badges");
                badgeHost.innerHTML = "";
                [
                    {{ label: customer.risk_label, className: "badge badge-risk" }},
                    {{ label: `${{customer.review_status}} credit review`, className: "badge" }},
                    {{ label: `${{customer.rm_status}} RM`, className: "badge" }},
                ].forEach((badge) => {{
                    const span = document.createElement("span");
                    span.className = badge.className;
                    span.textContent = badge.label;
                    badgeHost.appendChild(span);
                }});
            }}

            function renderDriverChips(customer) {{
                const chipHost = document.getElementById("modal-driver-chips");
                chipHost.innerHTML = "";
                customer.driver_chips.forEach((chip) => {{
                    const span = document.createElement("span");
                    span.className = "chip";
                    span.textContent = chip;
                    chipHost.appendChild(span);
                }});
            }}

            function renderServiceGrid(customer) {{
                const serviceGrid = document.getElementById("modal-service-grid");
                serviceGrid.innerHTML = "";
                customer.service_bars.forEach((service) => {{
                    const card = document.createElement("div");
                    card.className = "service-card";
                    card.innerHTML = `
                        <div class="mini-label">${{service.label}}</div>
                        <div class="mini-value">${{service.value}}</div>
                        <div class="bar"><div class="bar-fill" style="width: ${{service.width}}%;"></div></div>
                    `;
                    serviceGrid.appendChild(card);
                }});
            }}

            function renderList(id, items) {{
                const host = document.getElementById(id);
                if (!host) {{
                    return;
                }}
                host.innerHTML = "";
                items.forEach((item) => {{
                    const li = document.createElement("li");
                    li.textContent = item;
                    host.appendChild(li);
                }});
            }}

            function updateSidePreview(customer) {{
                const host = document.getElementById("working-panel");
                if (!host || !customer) {{
                    return;
                }}
                const safeOffersHtml = customer.guidance.safe_offers
                    .map((item) => "<li>" + item + "</li>")
                    .join("");
                const avoidOffersHtml = customer.guidance.avoid_offers
                    .map((item) => "<li>" + item + "</li>")
                    .join("");
                selectedCustomerId = customer.customer_id;
                if (selectedCustomerLabel) {{
                    selectedCustomerLabel.textContent = `#${{customer.customer_id}}`;
                }}
                host.innerHTML = `
                    <div class="table-subtle">Selected customer</div>
                    <h2>Customer #${{customer.customer_id}}</h2>
                    <p class="helper-copy">${{customer.risk_label}} risk, ${{customer.leave_probability}} leave probability, ${{customer.rm_status.toLowerCase()}} RM.</p>
                    <div class="chip-row" style="margin-top: 12px;">
                        <span class="badge badge-risk">${{customer.risk_label}}</span>
                        <span class="badge">${{customer.review_status}} credit review</span>
                        <span class="badge">${{customer.rm_status}} RM</span>
                    </div>
                    <div class="detail-grid" style="margin-top: 16px;">
                        <article class="info-card">
                            <span class="mini-label">Loan context</span>
                            <strong>${{customer.loan_type}}</strong>
                            <p>${{customer.loan_amount}} outstanding with EMI ${{customer.emi}}.</p>
                        </article>
                        <article class="info-card">
                            <span class="mini-label">Service quality</span>
                            <strong>${{customer.service_quality}}</strong>
                            <p>Current relationship quality score for this customer.</p>
                        </article>
                    </div>
                    <div class="action-callout" style="margin-top: 14px;">${{customer.recommended_action}}</div>
                    <div class="guidance-grid">
                        <div class="guidance-card">
                            <div class="mini-label">Primary focus</div>
                            <p class="helper-copy" style="margin-top: 8px;">${{customer.guidance.primary_focus}}</p>
                        </div>
                        <div class="guidance-card">
                            <div class="mini-label">What to offer</div>
                            <ul>${{safeOffersHtml}}</ul>
                        </div>
                        <div class="guidance-card">
                            <div class="mini-label">What to avoid</div>
                            <ul>${{avoidOffersHtml}}</ul>
                        </div>
                    </div>
                    <p class="helper-copy" style="margin-top: 12px;">Single click updates this panel. Double click opens the full popup overlay.</p>
                `;
            }}

            function highlightRow(customerId) {{
                rowElements.forEach((row) => {{
                    row.classList.toggle("selected", row.dataset.customerId === customerId);
                }});
            }}

            function openCustomerModal(customerId) {{
                const customer = customerData[customerId];
                if (!customer) {{
                    return;
                }}

                setText("modal-customer-title", `Customer #${{customer.customer_id}}`);
                setText("modal-loan-type", customer.loan_type);
                setText("modal-loan-copy", `${{customer.loan_amount}} outstanding relationship with EMI ${{customer.emi}}.`);
                setText("modal-leave-probability", customer.leave_probability);
                setText("modal-service-quality", customer.service_quality);
                setText("modal-product-count", `${{customer.product_count}} products`);
                setText("modal-relationship-copy", `${{customer.customer_value_tier}} via ${{customer.primary_channel}} channel.`);
                setText("modal-age", customer.age);
                setText("modal-income", customer.income);
                setText("modal-credit-score", customer.credit_score);
                setText("modal-complaints", customer.complaint_count);
                setText("modal-resolution-days", customer.resolution_days);
                setText("modal-retention-score", customer.retention_score);
                setText("modal-recommended-action", customer.recommended_action);
                setText("modal-retention-lane", customer.retention_strategy_lane);
                setText("modal-risk-profile", customer.customer_risk_profile);
                setText("modal-action-owner", customer.retention_action_owner);
                setText("modal-primary-focus", customer.guidance.primary_focus);

                renderBadges(customer);
                renderDriverChips(customer);
                renderServiceGrid(customer);
                renderList("modal-focus-list", customer.guidance.focus_points);
                renderList("modal-safe-offers", customer.guidance.safe_offers);
                renderList("modal-avoid-offers", customer.guidance.avoid_offers);
                highlightRow(customerId);
                updateSidePreview(customer);

                modal.classList.add("open");
                modal.setAttribute("aria-hidden", "false");
            }}

            function closeCustomerModal() {{
                modal.classList.remove("open");
                modal.setAttribute("aria-hidden", "true");
                if (lastFocusedRow) {{
                    lastFocusedRow.focus();
                }}
            }}

            rowElements.forEach((row) => {{
                row.addEventListener("click", () => {{
                    lastFocusedRow = row;
                    const customer = customerData[row.dataset.customerId];
                    if (customer) {{
                        highlightRow(row.dataset.customerId);
                        updateSidePreview(customer);
                    }}
                }});

                row.addEventListener("dblclick", () => {{
                    lastFocusedRow = row;
                    openCustomerModal(row.dataset.customerId);
                }});

                row.addEventListener("keydown", (event) => {{
                    if (event.key === "Enter" || event.key === " ") {{
                        event.preventDefault();
                        lastFocusedRow = row;
                        openCustomerModal(row.dataset.customerId);
                    }}
                }});
            }});

            closeModalButton.addEventListener("click", closeCustomerModal);

            promptChips.forEach((chip) => {{
                chip.addEventListener("click", () => {{
                    if (chatInput && chip.dataset.prompt) {{
                        chatInput.value = chip.dataset.prompt;
                    }}
                }});
            }});

            if (chatForm && chatInput && chatLog) {{
                chatForm.addEventListener("submit", async (event) => {{
                    event.preventDefault();
                    const question = chatInput.value.trim();
                    if (!question) {{
                        return;
                    }}
                    if (!selectedCustomerId || !customerData[selectedCustomerId]) {{
                        renderChatMessage("system", "Select a customer first so the AI can answer in the right context.");
                        return;
                    }}

                    renderChatMessage("user", question);
                    frontlineChatHistory.push({{ role: "user", content: question }});
                    chatInput.value = "";

                    try {{
                        const response = await fetch("/api/frontline-chat", {{
                            method: "POST",
                            headers: {{ "Content-Type": "application/json" }},
                            body: JSON.stringify({{
                                question,
                                customer_id: selectedCustomerId,
                                role_mode: roleModeSelect ? roleModeSelect.value : "branch_staff",
                                history: frontlineChatHistory.slice(-6),
                            }}),
                        }});
                        const result = await response.json();
                        const message = result.message || "Decipher AI did not return a response.";
                        if (response.ok && result.ok) {{
                            frontlineChatHistory.push({{ role: "assistant", content: message }});
                            renderChatMessage("assistant", message);
                        }} else {{
                            renderChatMessage("system", message);
                        }}
                    }} catch (error) {{
                        renderChatMessage("system", "Decipher AI could not be reached from the preview. Please try again.");
                    }}
                }});
            }}

            modal.addEventListener("click", (event) => {{
                if (event.target === modal) {{
                    closeCustomerModal();
                }}
            }});

            document.addEventListener("keydown", (event) => {{
                if (event.key === "Escape" && modal.classList.contains("open")) {{
                    closeCustomerModal();
                }}
            }});

            if ("{html.escape(selected_customer_id)}") {{
                const initialCustomer = customerData["{html.escape(selected_customer_id)}"];
                if (initialCustomer) {{
                    highlightRow("{html.escape(selected_customer_id)}");
                    updateSidePreview(initialCustomer);
                }}
            }}
        </script>
    </body>
    </html>
    """


class Customer360PreviewHandler(SimpleHTTPRequestHandler):
    def _send_json(self, status, payload):
        response_bytes = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(response_bytes)))
        self.end_headers()
        self.wfile.write(response_bytes)

    def _handle_frontline_chat_request(self):
        parsed = urlparse(self.path)
        if parsed.path != "/api/frontline-chat":
            self.send_error(HTTPStatus.NOT_FOUND, "This API route is not available.")
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            content_length = 0
        if content_length <= 0:
            self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "message": "Please enter a question for Decipher AI."})
            return

        try:
            request_payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
        except json.JSONDecodeError:
            self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "message": "The preview sent an invalid AI request."})
            return

        question = str(request_payload.get("question", "")).strip()
        customer_id = str(request_payload.get("customer_id", "")).strip()
        role_mode = normalize_role_mode(str(request_payload.get("role_mode", "branch_staff")).strip())
        if not question:
            self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "message": "Please enter a question for Decipher AI."})
            return
        if not customer_id:
            self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "message": "Select a customer first, then ask Decipher AI."})
            return

        df = load_priority_customers()
        if df.empty:
            self._send_json(HTTPStatus.SERVICE_UNAVAILABLE, {"ok": False, "message": "Priority customer data is unavailable right now."})
            return

        matched = df[df["customer_id"].astype(str) == customer_id]
        if matched.empty:
            self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "message": "The selected customer could not be found in the priority queue."})
            return

        customer_context = build_customer_ai_context(matched.iloc[0])
        result = ask_gemini_frontline_assistant(
            question,
            customer_context,
            role_mode,
            history=request_payload.get("history"),
        )
        status = result.pop("status", HTTPStatus.OK)
        self._send_json(status, result)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path not in {"/", "/index.html"}:
            self.send_error(HTTPStatus.NOT_FOUND, "Preview route not found.")
            return

        html_bytes = build_dashboard_html(parse_qs(parsed.query)).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(html_bytes)))
        self.end_headers()
        self.wfile.write(html_bytes)

    def do_POST(self):
        self._handle_frontline_chat_request()


def main():
    parser = argparse.ArgumentParser(
        description="Run the standalone Customer 360 preview for urgent customers."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8602)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), Customer360PreviewHandler)
    print(f"Customer 360 preview running at http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop the preview server.")
    server.serve_forever()


if __name__ == "__main__":
    main()
