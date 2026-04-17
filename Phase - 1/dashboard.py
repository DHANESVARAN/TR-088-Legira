import json
import os
import re
from urllib import error as url_error
from urllib import request as url_request

import altair as alt
import pandas as pd
import streamlit as st

from prioritization_engine import MODEL_FEATURES, run_pipeline, save_outputs, score_new_case


st.set_page_config(page_title="Court Case Prioritization Studio", layout="wide")

DEFAULT_LOCAL_MODEL = "qwen2.5:latest"
DEFAULT_LOCAL_LLM_URL = "http://localhost:11434"


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            color: #162331;
            background:
                radial-gradient(circle at top left, rgba(166, 230, 174, 0.58), transparent 30%),
                radial-gradient(circle at top right, rgba(195, 244, 205, 0.5), transparent 24%),
                linear-gradient(180deg, #d9f7dd 0%, #c3efcb 100%);
        }
        [data-testid="stHeader"] {
            background: linear-gradient(90deg, #7ecf8d 0%, #6fc380 100%) !important;
            border-bottom: 1px solid rgba(22, 35, 49, 0.16) !important;
        }
        [data-testid="stToolbar"] * {
            color: #143021 !important;
        }
        [data-testid="stDecoration"] {
            background: #7ecf8d !important;
        }
        .stApp, .stApp p, .stApp label, .stApp span, .stApp li, .stApp div {
            color: #162331;
        }
        .stMarkdown, .stMarkdown p, .stMarkdown div, .stCaption {
            color: #1f3344 !important;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #14212e !important;
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #95d5a0 0%, #7dc98b 100%) !important;
        }
        [data-testid="stSidebar"] * {
            color: #152534 !important;
        }
        [data-testid="stMetric"] {
            background: rgba(255,255,255,0.76);
            border: 1px solid rgba(48, 72, 94, 0.12);
            border-radius: 14px;
            padding: 0.6rem 0.8rem;
        }
        [data-testid="stMetricLabel"],
        [data-testid="stMetricValue"],
        [data-testid="stMetricDelta"] {
            color: #162331 !important;
        }
        [data-baseweb="tab-list"] {
            gap: 0.35rem;
        }
        [data-baseweb="tab"] {
            background: rgba(255,255,255,0.55) !important;
            border: 1px solid rgba(48, 72, 94, 0.12) !important;
            border-radius: 12px 12px 0 0 !important;
            color: #1b2d3d !important;
            font-weight: 600 !important;
        }
        [aria-selected="true"][data-baseweb="tab"] {
            background: #dbe8f4 !important;
            color: #10202d !important;
        }
        .stTextInput input,
        .stTextArea textarea,
        .stNumberInput input,
        div[data-baseweb="select"] > div,
        .stMultiSelect div[data-baseweb="select"] > div {
            color: #162331 !important;
            background: rgba(255,255,255,0.84) !important;
        }
        .stCheckbox label,
        .stRadio label,
        .stSelectbox label,
        .stMultiSelect label,
        .stTextInput label,
        .stTextArea label,
        .stNumberInput label {
            color: #152534 !important;
            font-weight: 600 !important;
        }
        [data-testid="stDataFrame"] {
            background: rgba(255,255,255,0.72);
            border-radius: 14px;
        }
        [data-testid="stDataFrame"] * {
            color: #162331 !important;
        }
        .stAlert {
            color: #162331 !important;
        }
        .stButton > button,
        .stDownloadButton > button,
        [data-testid="stFormSubmitButton"] > button {
            background: #ecfff0 !important;
            color: #0f2a1c !important;
            border: 1px solid #2c6b45 !important;
            font-weight: 700 !important;
        }
        .stButton > button:hover,
        .stDownloadButton > button:hover,
        [data-testid="stFormSubmitButton"] > button:hover {
            background: #d9f7df !important;
            color: #0a2015 !important;
            border-color: #1e5334 !important;
        }
        .stButton > button:focus,
        .stDownloadButton > button:focus,
        [data-testid="stFormSubmitButton"] > button:focus {
            color: #0a2015 !important;
            box-shadow: 0 0 0 2px rgba(30, 83, 52, 0.28) !important;
            outline: none !important;
        }
        .hero-card {
            padding: 1.2rem 1.4rem;
            border-radius: 18px;
            background: rgba(255,255,255,0.72);
            border: 1px solid rgba(48, 72, 94, 0.12);
            box-shadow: 0 16px 38px rgba(37, 53, 70, 0.08);
        }
        .hero-kicker {
            color: #7a4b22;
            font-size: 0.84rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }
        .hero-title {
            color: #1f3344;
            font-size: 2.2rem;
            font-weight: 700;
            line-height: 1.1;
            margin: 0.35rem 0 0.45rem 0;
        }
        .hero-copy {
            color: #42586d;
            font-size: 1rem;
            line-height: 1.5;
        }
        .metric-strip {
            padding: 0.85rem 1rem;
            border-radius: 16px;
            background: rgba(255,255,255,0.74);
            border: 1px solid rgba(48, 72, 94, 0.1);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def load_pipeline_result(data_dir: str, statute_file: str, augment_demo: bool) -> dict:
    return run_pipeline(data_dir=data_dir, statute_file=statute_file, augment_demo=augment_demo)


def summarize_case_with_ollama(
    raw_case_text: str,
    model_name: str,
    ollama_url: str,
    timeout_seconds: int = 60,
) -> tuple[str, list[str], str]:
    text = (raw_case_text or "").strip()
    if not text:
        return "", [], "No case text provided for summarization."

    prompt = (
        "You are helping court backlog triage. Read the case text and return ONLY valid JSON with this schema: "
        '{"sections": ["379", "34"], "summary": "..."}. '
        "Rules: sections must include inferred IPC/BNS section numbers if present or strongly implied; "
        "summary must be 2-4 concise lines including offense, legal status, delay/detention context, and vulnerability/medical mentions. "
        "Do not include markdown or extra text.\n\n"
        f"Case text:\n{text}"
    )

    base = ollama_url.rstrip("/")
    attempts: list[str] = []

    def _extract_sections_from_text(input_text: str) -> list[str]:
        sec_matches = re.findall(r"\b(?:IPC|BNS)\s*[-:/]?\s*(\d{1,4}[A-Za-z]?)\b", input_text, flags=re.IGNORECASE)
        slash_matches = re.findall(r"\b(\d{1,4}[A-Za-z]?)\s*/\s*(\d{1,4}[A-Za-z]?)\b", input_text)
        sections: list[str] = []
        for s in sec_matches:
            sections.append(s.upper())
        for a, b in slash_matches:
            sections.append(a.upper())
            sections.append(b.upper())
        seen = set()
        ordered = []
        for s in sections:
            if s not in seen:
                seen.add(s)
                ordered.append(s)
        return ordered

    def _extract_summary_sections(raw_output: str) -> tuple[str, list[str]]:
        output = (raw_output or "").strip()
        if not output:
            return "", []

        try:
            parsed = json.loads(output)
            summary = str(parsed.get("summary", "")).strip()
            sections = parsed.get("sections", [])
            if isinstance(sections, list):
                section_list = [str(x).strip().upper() for x in sections if str(x).strip()]
            else:
                section_list = _extract_sections_from_text(str(sections))
            if summary:
                return summary, section_list
        except Exception:
            pass

        json_block = re.search(r"\{[\s\S]*\}", output)
        if json_block:
            try:
                parsed = json.loads(json_block.group(0))
                summary = str(parsed.get("summary", "")).strip()
                sections = parsed.get("sections", [])
                if isinstance(sections, list):
                    section_list = [str(x).strip().upper() for x in sections if str(x).strip()]
                else:
                    section_list = _extract_sections_from_text(str(sections))
                if summary:
                    return summary, section_list
            except Exception:
                pass

        line_section = ""
        line_summary = ""
        for line in output.splitlines():
            lower = line.lower().strip()
            if lower.startswith("sections") or lower.startswith("section"):
                line_section = line
            if lower.startswith("summary"):
                line_summary = line

        section_list = _extract_sections_from_text(line_section or output)
        if line_summary:
            summary = line_summary.split(":", 1)[-1].strip()
            if summary:
                return summary, section_list

        return output, section_list

    def _post_json(endpoint: str, payload: dict) -> dict:
        req = url_request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with url_request.urlopen(req, timeout=timeout_seconds) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body)

    # 1) Native Ollama generate endpoint
    try:
        endpoint = base + "/api/generate"
        parsed = _post_json(endpoint, {"model": model_name, "prompt": prompt, "stream": False})
        summary, sections = _extract_summary_sections(str(parsed.get("response", "")).strip())
        if summary:
            return summary, sections, ""
        attempts.append(f"{endpoint}: empty response")
    except Exception as exc:
        attempts.append(f"{endpoint}: {exc}")

    # 2) Native Ollama chat endpoint
    try:
        endpoint = base + "/api/chat"
        parsed = _post_json(
            endpoint,
            {
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
            },
        )
        msg = parsed.get("message", {}) if isinstance(parsed, dict) else {}
        summary, sections = _extract_summary_sections(str(msg.get("content", "")).strip())
        if summary:
            return summary, sections, ""
        attempts.append(f"{endpoint}: empty response")
    except Exception as exc:
        attempts.append(f"{endpoint}: {exc}")

    # 3) OpenAI-compatible endpoint (some local stacks expose this)
    try:
        endpoint = base + "/v1/chat/completions"
        parsed = _post_json(
            endpoint,
            {
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,
            },
        )
        choices = parsed.get("choices", []) if isinstance(parsed, dict) else []
        first = choices[0] if choices else {}
        msg = first.get("message", {}) if isinstance(first, dict) else {}
        summary, sections = _extract_summary_sections(str(msg.get("content", "")).strip())
        if summary:
            return summary, sections, ""
        attempts.append(f"{endpoint}: empty response")
    except Exception as exc:
        attempts.append(f"{endpoint}: {exc}")

    detail = " | ".join(attempts[-3:])
    return "", _extract_sections_from_text(text), (
        "Unable to summarize via local LLM endpoint. Tried /api/generate, /api/chat, and /v1/chat/completions. "
        f"Details: {detail}"
    )


def summarize_case_locally(raw_case_text: str) -> str:
    text = " ".join((raw_case_text or "").split()).strip()
    if not text:
        return "No case details provided."

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    lead = " ".join(sentences[:2]) if sentences else text[:260]

    lower = text.lower()
    offense_terms = ["theft", "robbery", "cheating", "murder", "rape", "trespass", "kidnapping"]
    legal_terms = ["undertrial", "bail", "convicted", "non-bailable", "bailable"]
    timing_terms = ["detention", "pending", "delay", "adjournment", "hearing", "overstay"]
    vulnerability_terms = ["medical", "illness", "disability", "pregnancy", "juvenile", "senior", "chronic"]

    def _pick_terms(candidates: list[str], max_terms: int = 3) -> list[str]:
        found = [term for term in candidates if term in lower]
        return found[:max_terms]

    tags = []
    if _pick_terms(offense_terms):
        tags.append("Offense: " + ", ".join(_pick_terms(offense_terms)))
    if _pick_terms(legal_terms):
        tags.append("Legal status: " + ", ".join(_pick_terms(legal_terms)))
    if _pick_terms(timing_terms):
        tags.append("Delay context: " + ", ".join(_pick_terms(timing_terms)))
    if _pick_terms(vulnerability_terms):
        tags.append("Vulnerability: " + ", ".join(_pick_terms(vulnerability_terms)))

    if tags:
        return lead + "\n" + " | ".join(tags)
    return lead


def extract_sections_from_case_text(raw_case_text: str) -> list[str]:
    sec_matches = re.findall(r"\b(?:IPC|BNS)\s*[-:/]?\s*(\d{1,4}[A-Za-z]?)\b", raw_case_text or "", flags=re.IGNORECASE)
    slash_matches = re.findall(r"\b(\d{1,4}[A-Za-z]?)\s*/\s*(\d{1,4}[A-Za-z]?)\b", raw_case_text or "")
    sections: list[str] = [s.upper() for s in sec_matches]
    for a, b in slash_matches:
        sections.append(a.upper())
        sections.append(b.upper())
    seen = set()
    ordered = []
    for s in sections:
        if s not in seen:
            seen.add(s)
            ordered.append(s)
    return ordered


def render_header(metrics: dict, ranked: pd.DataFrame) -> None:
    critical_count = int((ranked["priority_bucket"] == "Critical").sum())
    flagged_count = int(((ranked["undertrial_flag"] == True) & (ranked["flag_reason"] != "No critical flag")).sum())
    non_finite_count = int((ranked["statutory_non_finite_sentence"] == True).sum())

    left, right = st.columns([1.5, 1])
    with left:
        st.markdown(
            """
            <div class="hero-card">
              <div class="hero-kicker">Judicial Operations Console</div>
              <div class="hero-title">Court Case Prioritization Studio</div>
              <div class="hero-copy">
                Analyze backlog risk, inspect legally grounded urgency signals, review vulnerable undertrial queues,
                and score new matters interactively using the same prioritization engine.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        st.markdown('<div class="metric-strip">', unsafe_allow_html=True)
        st.metric("Expert-label accuracy", f"{metrics['classification_accuracy_vs_expert_labels']:.3f}")
        st.metric("Undertrial recall", f"{metrics['undertrial_detection_recall']:.3f}")
        st.metric("Critical cases", critical_count)
        st.metric("Flagged undertrials", flagged_count)
        st.metric("Non-finite sentence cases", non_finite_count)
        st.markdown("</div>", unsafe_allow_html=True)


def make_bar_chart(df: pd.DataFrame, x: str, y: str, color: str | None = None, title: str = ""):
    chart = alt.Chart(df).mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6).encode(
        x=alt.X(x, sort="-y"),
        y=alt.Y(y),
        tooltip=list(df.columns),
    )
    if color:
        chart = chart.encode(color=alt.Color(color, legend=None))
    return chart.properties(height=320, title=title)


def build_critical_rate_table(df: pd.DataFrame, class_col: str) -> pd.DataFrame:
    if class_col not in df.columns:
        return pd.DataFrame(columns=["class_dimension", "class_name", "total_cases", "critical_cases", "critical_rate_pct"])

    safe = df.copy()
    safe[class_col] = safe[class_col].astype(str).fillna("Unknown")
    safe["is_critical"] = safe["priority_bucket"].astype(str).eq("Critical").astype(int)

    table = (
        safe.groupby(class_col, as_index=False)
        .agg(total_cases=("case_id", "count"), critical_cases=("is_critical", "sum"))
        .sort_values(["critical_cases", "total_cases"], ascending=[False, False])
    )
    table["critical_rate_pct"] = (100 * table["critical_cases"] / table["total_cases"]).round(2)
    table = table.rename(columns={class_col: "class_name"})
    table.insert(0, "class_dimension", class_col)
    return table


def build_distribution_table(df: pd.DataFrame, class_col: str) -> pd.DataFrame:
    if class_col not in df.columns:
        return pd.DataFrame(columns=["class_name", "case_count", "share_pct"])

    safe = df.copy()
    safe[class_col] = safe[class_col].astype(str).fillna("Unknown")
    table = (
        safe.groupby(class_col, as_index=False)
        .agg(case_count=("case_id", "count"))
        .sort_values("case_count", ascending=False)
        .rename(columns={class_col: "class_name"})
    )
    total = max(int(table["case_count"].sum()), 1)
    table["share_pct"] = (100 * table["case_count"] / total).round(2)
    return table


def render_overview_tab(result: dict) -> None:
    ranked = result["ranked"]
    cluster_summary = result["cluster_summary"]

    c1, c2 = st.columns([1.1, 1])
    with c1:
        bucket_df = (
            ranked.groupby("priority_bucket", as_index=False)
            .agg(case_count=("case_id", "count"))
            .sort_values("case_count", ascending=False)
        )
        st.altair_chart(make_bar_chart(bucket_df, "priority_bucket:N", "case_count:Q", "priority_bucket:N", "Priority Bucket Distribution"), use_container_width=True)
    with c2:
        track_df = (
            ranked.groupby("recommended_track", as_index=False)
            .agg(case_count=("case_id", "count"))
            .sort_values("case_count", ascending=False)
        )
        st.altair_chart(make_bar_chart(track_df, "recommended_track:N", "case_count:Q", None, "Recommended Track Volume"), use_container_width=True)

    c3, c4 = st.columns([1, 1.1])
    with c3:
        st.altair_chart(make_bar_chart(cluster_summary, "cluster_label:N", "mean_priority:Q", None, "Cluster Mean Priority"), use_container_width=True)
    with c4:
        vulnerability_df = ranked.assign(
            medical_vulnerable=(ranked["health_flag"] | ranked["disability_flag"])
        )
        vuln_counts = pd.DataFrame(
            {
                "segment": ["Senior/Juvenile", "Female", "Medical/Disability"],
                "cases": [
                    int(vulnerability_df["age_vulnerable"].sum()),
                    int(vulnerability_df["gender_vulnerable"].sum()),
                    int(vulnerability_df["medical_vulnerable"].sum()),
                ],
            }
        )
        st.altair_chart(make_bar_chart(vuln_counts, "segment:N", "cases:Q", None, "Vulnerability Signals"), use_container_width=True)

    st.subheader("Top Critical Queue")
    top_cols = [
        "recommended_hearing_order",
        "case_id",
        "priority_score",
        "priority_bucket",
        "predicted_urgency",
        "recommended_track",
        "ipc_section",
        "offense_type",
        "detention_ratio_calc",
        "flag_reason",
        "medical_notes",
    ]
    st.dataframe(ranked[top_cols].head(25), use_container_width=True, height=460)


def render_queue_tab(result: dict) -> None:
    ranked = result["ranked"]

    st.subheader("Filterable Hearing Queue")
    f1, f2, f3, f4 = st.columns(4)
    priorities = f1.multiselect("Priority bucket", sorted(ranked["priority_bucket"].dropna().unique()), default=sorted(ranked["priority_bucket"].dropna().unique()))
    tracks = f2.multiselect("Track", sorted(ranked["recommended_track"].dropna().unique()), default=sorted(ranked["recommended_track"].dropna().unique()))
    statuses = f3.multiselect("Legal status", sorted(ranked["legal_status"].dropna().unique()), default=sorted(ranked["legal_status"].dropna().unique()))
    sources = f4.multiselect("Data source", sorted(ranked["data_source"].dropna().unique()), default=sorted(ranked["data_source"].dropna().unique()))

    c1, c2, c3 = st.columns(3)
    only_flagged = c1.checkbox("Flagged only", value=False)
    only_medical = c2.checkbox("Medical vulnerability only", value=False)
    only_non_finite = c3.checkbox("Non-finite sentence only", value=False)

    filtered = ranked[
        ranked["priority_bucket"].isin(priorities)
        & ranked["recommended_track"].isin(tracks)
        & ranked["legal_status"].isin(statuses)
        & ranked["data_source"].isin(sources)
    ].copy()

    if only_flagged:
        filtered = filtered[(filtered["undertrial_flag"] == True) & (filtered["flag_reason"] != "No critical flag")]
    if only_medical:
        filtered = filtered[(filtered["health_flag"] == True) | (filtered["disability_flag"] == True)]
    if only_non_finite:
        filtered = filtered[filtered["statutory_non_finite_sentence"] == True]

    st.caption(f"Filtered cases: {len(filtered)}")
    queue_cols = [
        "recommended_hearing_order",
        "case_id",
        "priority_score",
        "priority_bucket",
        "predicted_urgency",
        "predicted_case_type",
        "recommended_track",
        "data_source",
        "synthetic_demo_case",
        "summary",
        "medical_notes",
        "legal_status",
        "offense_type",
        "ipc_section",
        "detention_days",
        "sentence_days_final",
        "statutory_bailability",
        "statutory_non_finite_sentence",
        "detention_ratio_calc",
        "flag_reason",
    ]
    existing_cols = [col for col in queue_cols if col in filtered.columns]
    st.dataframe(filtered[existing_cols], use_container_width=True, height=620)

    csv_data = filtered.to_csv(index=False).encode("utf-8")
    st.download_button("Download filtered queue CSV", data=csv_data, file_name="filtered_hearing_queue.csv", mime="text/csv")


def render_review_tab(result: dict) -> None:
    flagged = result["flagged"]
    demo_cases = result["demo_cases"]

    left, right = st.columns([1.15, 0.85])
    with left:
        st.subheader("Flagged Undertrial Review")
        review_cols = [
            "recommended_hearing_order",
            "case_id",
            "priority_score",
            "priority_bucket",
            "ipc_section",
            "offense_type",
            "detention_days",
            "sentence_days_final",
            "detention_ratio_calc",
            "medical_notes",
            "flag_reason",
        ]
        st.dataframe(flagged[review_cols], use_container_width=True, height=520)

    with right:
        st.subheader("Synthetic Demo Cases")
        if demo_cases.empty:
            st.info("Enable synthetic augmentation in the sidebar to populate this section.")
        else:
            demo_cols = [
                "recommended_hearing_order",
                "case_id",
                "ipc_section",
                "priority_bucket",
                "statutory_non_finite_sentence",
                "medical_notes",
                "flag_reason",
            ]
            st.dataframe(demo_cases[demo_cols], use_container_width=True, height=520)


def render_cluster_tab(result: dict) -> None:
    cluster_summary = result["cluster_summary"]
    ranked = result["ranked"]

    st.subheader("Cluster Analysis")
    c1, c2 = st.columns([0.95, 1.05])
    with c1:
        st.dataframe(cluster_summary, use_container_width=True, height=300)
    with c2:
        cluster_scatter = alt.Chart(ranked).mark_circle(size=72, opacity=0.7).encode(
            x=alt.X("detention_ratio_calc:Q", title="Detention Ratio"),
            y=alt.Y("priority_score:Q", title="Priority Score"),
            color=alt.Color("cluster_label:N", title="Cluster"),
            tooltip=["case_id", "cluster_label", "priority_score", "flag_reason", "medical_notes"],
        ).properties(height=320, title="Case Segmentation by Detention Ratio and Priority")
        st.altair_chart(cluster_scatter, use_container_width=True)


def render_metrics_tab(result: dict) -> None:
    metrics = result["metrics"]
    ranked = result["ranked"]
    report_df = pd.DataFrame(metrics["urgency_classification_report"]).transpose().reset_index().rename(columns={"index": "label"})

    c1, c2 = st.columns([1, 1])
    with c1:
        summary_metrics = pd.DataFrame(
            {
                "metric": [
                    "classification_accuracy_vs_expert_labels",
                    "urgency_nlp_label_agreement",
                    "classification_accuracy_case_type",
                    "undertrial_detection_recall",
                    "dashboard_filter_response_time_ms",
                    "ranked_list_agreement_spearman",
                ],
                "value": [
                    metrics["classification_accuracy_vs_expert_labels"],
                    metrics["urgency_nlp_label_agreement"],
                    metrics["classification_accuracy_case_type"],
                    metrics["undertrial_detection_recall"],
                    metrics["dashboard_filter_response_time_ms"],
                    metrics["ranked_list_agreement_spearman"],
                ],
            }
        )
        st.dataframe(summary_metrics, use_container_width=True, height=280)
    with c2:
        st.dataframe(report_df, use_container_width=True, height=280)

    st.subheader("Critical Rate by Classification Class")
    class_dimensions = [
        "predicted_urgency",
        "predicted_case_type",
        "recommended_track",
        "legal_status",
        "offense_type",
        "statutory_bailability",
        "trial_status",
        "case_complexity",
    ]

    all_tables = [build_critical_rate_table(ranked, col) for col in class_dimensions]
    all_class_rates = pd.concat(all_tables, ignore_index=True)
    st.dataframe(all_class_rates, use_container_width=True, height=320)

    selected_dimension = st.selectbox(
        "Select classification dimension",
        class_dimensions,
        index=0,
    )
    selected_rates = build_critical_rate_table(ranked, selected_dimension)
    st.altair_chart(
        make_bar_chart(
            selected_rates,
            "class_name:N",
            "critical_rate_pct:Q",
            None,
            f"Critical Case Rate (%) by {selected_dimension}",
        ),
        use_container_width=True,
    )
    st.caption("Critical rate = (Critical cases in class / Total cases in class) x 100")

    st.subheader("Raw Metrics JSON")
    st.code(json.dumps(metrics, indent=2), language="json")


def render_about_page(result: dict) -> None:
    ranked = result["ranked"]
    metrics = result["metrics"]

    st.subheader("About The Prioritization Model")
    st.caption("AI-driven legal case backlog prioritization for courts")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total cases scored", len(ranked))
    c2.metric("Critical cases", int((ranked["priority_bucket"] == "Critical").sum()))
    c3.metric("Undertrial recall", f"{metrics['undertrial_detection_recall']:.3f}")
    c4.metric("Ranking agreement", f"{metrics['ranked_list_agreement_spearman']:.3f}")

    st.markdown("### Model Pipeline")
    st.markdown(
        """
        1. Data ingestion and merge across core case details, detention records, temporal status, demographics, and NLP labels.
        2. Legal grounding through IPC/BNS section lookup to derive statutory sentence limits and bailability.
        3. Feature engineering for overstay, detention ratio, pendency/staleness, offense severity, and vulnerability.
        4. Scoring and prioritization to produce hearing order, critical flags, and recommended legal track.
        5. Classification and clustering for urgency prediction, case-type prediction, and queue segmentation.
        """
    )

    st.markdown("### Government Data Sources")
    source_df = pd.DataFrame(
        {
            "source": [
                "National Judicial Data Grid (NJDG)",
                "eCourts India Services",
            ],
            "link": [
                "https://njdg.ecourts.gov.in",
                "https://services.ecourts.gov.in",
            ],
            "collected_for": [
                "Case pendency, durations, category trends, backlog simulation baselines",
                "Case status patterns, section references, legal text patterns, FIR/case narrative structures",
            ],
        }
    )
    st.dataframe(source_df, use_container_width=True, height=200)
    st.markdown(
        """
        1. NJDG: https://njdg.ecourts.gov.in
        2. eCourts India: https://services.ecourts.gov.in
        """
    )

    st.markdown("### How Data Is Transformed")
    transform_df = pd.DataFrame(
        {
            "stage": [
                "Raw ingestion",
                "Schema harmonization",
                "Case-level merge",
                "Statute enrichment",
                "Derived legal indicators",
                "Feature scaling and modeling",
                "Priority scoring and queue output",
            ],
            "details": [
                "Read core cases, detention, temporal, demographics, and NLP datasets.",
                "Normalize key fields and text/categorical types (status, offense, sections, bailability).",
                "Join all sources by case_id to form one analytical table.",
                "Map IPC/BNS sections to punishment behavior and bailability from statute lookup.",
                "Compute overstay flags, detention ratios, vulnerability flags, staleness flags, and track signals.",
                "Train urgency and case-type RandomForest classifiers; run KMeans clustering.",
                "Compute priority score, assign bucket, rank hearing order, and flag critical undertrial cases.",
            ],
        }
    )
    st.dataframe(transform_df, use_container_width=True, height=280)

    st.markdown("### Government Source to Model Dataset Transformation Mapping")
    gov_map_df = pd.DataFrame(
        {
            "government_source_field": [
                "NJDG: case type/category",
                "NJDG: case filing date / pendency",
                "NJDG: hearing progression and delay indicators",
                "eCourts: section text in orders/status",
                "eCourts: legal status and bail-stage hints",
                "eCourts/Narrative: case description text",
                "Prison/custody records: detention duration",
                "Demographic records: age/gender/medical/disability",
            ],
            "project_dataset_target": [
                "dataset1_core_cases.csv -> case_type, offense_type",
                "dataset3_temporal.csv -> days_pending",
                "dataset3_temporal.csv -> last_hearing_gap, no_of_adjournments, trial_status",
                "dataset1_core_cases.csv -> ipc_section (normalized section tokens)",
                "dataset1_core_cases.csv + dataset5_nlp.csv -> legal_status, bailable, bail_eligibility_nlp",
                "dataset1_core_cases.csv + dataset5_nlp.csv -> summary, summary_nlp, keywords, urgency_nlp",
                "dataset2_detention.csv -> detention_days, expected_sentence_days, detention_ratio",
                "dataset4_demographics.csv -> age, gender, health_flag, disability_flag, vulnerability_score",
            ],
            "transformation_applied": [
                "Category harmonization and controlled-value mapping",
                "Date-difference computation to numeric pendency days",
                "Temporal feature derivation and status normalization",
                "Regex/token extraction, cleaning, and multi-section parsing",
                "Text/rule normalization and boolean indicator creation",
                "Summarization/label support via NLP + fallback heuristics",
                "Ratio computation with clipping and safety fallbacks",
                "Binary flags + weighted vulnerability component engineering",
            ],
        }
    )
    st.dataframe(gov_map_df, use_container_width=True, height=320)

    st.markdown("#### Final Analytical Table Construction")
    st.markdown(
        """
        1. All five project datasets are joined on `case_id` to build one case-level analytical table.
        2. Statute lookup enriches each row with sentence behavior and bailability.
        3. Engineered features are computed on top of merged records.
        4. The unified table is then used for scoring, classification, clustering, and queue ranking.
        """
    )

    st.markdown("### Core Features Used")
    feature_table = pd.DataFrame(
        {
            "feature_group": [
                "Overstay logic",
                "Detention pressure",
                "Legal status and offense",
                "Temporal delays",
                "Vulnerability",
                "Statutory enrichment",
                "NLP-assisted inputs",
            ],
            "description": [
                "Flags undertrials who have exceeded statutory or derived sentence duration.",
                "Uses detention ratio and exponential scaling to increase urgency near sentencing limit.",
                "Uses undertrial status, offense type, and bailability to prioritize hearing tracks.",
                "Boosts stale cases based on last hearing gap and overall pendency.",
                "Adds weights for age, gender, health condition, and disability signals.",
                "Maps IPC/BNS sections to punishment type, finite or non-finite sentence, and bailability.",
                "Uses urgency and bail-eligibility hints from NLP fields as additional signals.",
            ],
        }
    )
    st.dataframe(feature_table, use_container_width=True, height=320)

    st.markdown("### Feature Engineering Performed")
    fe_df = pd.DataFrame(
        {
            "engineered_feature": [
                "sentence_days_final",
                "statutory_non_finite_sentence",
                "statutory_bailability",
                "detention_ratio_calc",
                "detention_ratio_exp_norm",
                "overstay_final",
                "stale_case_flag",
                "vulnerability_component",
                "offense_severity_index",
                "offense_priority_component",
                "recommended_track",
            ],
            "how_it_is_derived": [
                "Uses statute sentence days when available, else expected_sentence_days",
                "True for life/death style punishment sections",
                "Derived from statute mapping and used to override base bailability signal",
                "detention_days / sentence_days_final with safe fallbacks",
                "Exponential normalization of detention ratio to emphasize near-limit detention",
                "Combines source overstay input + computed overstay from detention and sentence limits",
                "True when last_hearing_gap is above the upper quartile",
                "Weighted mix of age, gender, health, and disability vulnerability",
                "Maps offense classes into severity bands for scoring",
                "Combines undertrial, bailability, offense simplicity, and severity",
                "Rule-based track assignment: Critical-Review, Bail-Eligible, Fast-Track, Civil, Regular-Criminal",
            ],
        }
    )
    st.dataframe(fe_df, use_container_width=True, height=340)

    st.markdown("### Priority Score Composition")
    score_df = pd.DataFrame(
        {
            "component": [
                "Overstay flag",
                "Detention ratio",
                "Offense and legal priority",
                "Vulnerability weight",
                "Staleness boost",
            ],
            "weight_or_behavior": [
                "Hard dominant component, with critical override for undertrial overstay",
                "Strong non-linear influence near sentence limit",
                "Medium influence toward bail-eligible or simpler tracks",
                "Social justice multiplier",
                "Small additive boost for long-inactive cases",
            ],
        }
    )
    st.dataframe(score_df, use_container_width=True, height=250)

    st.markdown("### Mathematical Formulation")
    st.markdown("Detention ratio and its non-linear normalization:")
    st.latex(r"R = \frac{\text{detention\_days}}{\text{sentence\_days\_final}}")
    st.latex(r"R_c = \min(\max(R, 0), 3)")
    st.latex(r"R_{exp} = \frac{e^{R_c} - 1}{e^3 - 1}")

    st.markdown("Base priority score:")
    st.latex(
        r"S_{base} = 50\cdot O + 25\cdot R_{exp} + 15\cdot P_{offense} + 10\cdot V + 3\cdot S_{stale}"
    )
    st.markdown("where:")
    st.markdown(
        """
        - $O$: overstay flag (0/1)
        - $P_{offense}$: offense/legal priority component in $[0,1]$
        - $V$: vulnerability component in $[0,1]$
        - $S_{stale}$: stale-case indicator (0/1)
        """
    )

    st.markdown("Critical override and clipping:")
    st.latex(r"S = \begin{cases}\max(S_{base},95), & \text{if undertrial and overstay} \\ S_{base}, & \text{otherwise}\end{cases}")
    st.latex(r"S_{final} = \min(100,\max(0,S))")

    st.markdown("Bucket assignment:")
    st.latex(r"\text{Low}: 0\le S<40,\quad \text{Medium}: 40\le S<65,\quad \text{High}: 65\le S<85,\quad \text{Critical}: 85\le S\le100")

    st.markdown("Evaluation metrics used:")
    st.latex(r"\text{Accuracy} = \frac{TP+TN}{TP+TN+FP+FN}")
    st.latex(r"\text{Recall}_{undertrial\_critical} = \frac{TP}{TP+FN}")
    st.latex(r"\rho_{rank} = \text{Spearman}(\text{priority\_score},\ \text{mock\_judicial\_review})")

    st.markdown("### Bias and Risk Analysis")
    bias_df = pd.DataFrame(
        {
            "bias_or_risk": [
                "Historical process bias",
                "Label bias (NLP urgency hints)",
                "Demographic weighting bias",
                "Section-coverage bias",
                "Sampling/synthetic data bias",
                "Measurement bias in detention/sentence fields",
                "Proxy bias from legal-status variables",
                "Automation bias in operations",
            ],
            "how_it_can_appear": [
                "Backlog history may encode unequal treatment patterns by region/court/type.",
                "Text-derived urgency may reflect noisy or inconsistent documentation.",
                "Age/gender/health weights can over- or under-prioritize some groups if miscalibrated.",
                "Missing/incorrect statute mappings can distort overstay and bailability interpretation.",
                "Synthetic augmentation may not represent real-world legal complexity.",
                "Wrong detention days or expected sentence values change score non-linearly.",
                "Undertrial/bail status can act as a proxy for systemic inequities.",
                "Users may follow scores blindly without judicial review context.",
            ],
            "current_mitigation": [
                "Human-in-the-loop review and transparent score decomposition in UI.",
                "NLP used as supporting signal, not sole decision criterion.",
                "Explicit vulnerability components are visible and auditable.",
                "Statute lookup table is explicit and editable; non-finite safeguards included.",
                "Synthetic rows are tagged (`synthetic_demo_case`) and separable in analysis.",
                "Fallbacks, clipping, and consistency checks reduce extreme numeric instability.",
                "Separate status analysis page to inspect distribution and critical rates by status.",
                "System positioned as decision-support, not autonomous case disposition.",
            ],
            "recommended_next_control": [
                "Court-wise calibration and periodic fairness audit by venue and case type.",
                "Label-quality benchmarking against expert-reviewed samples.",
                "Run sensitivity analysis on demographic weights and publish deltas.",
                "Expand section mapping coverage and add legal validation workflow.",
                "Use mixed real/synthetic reporting and isolate synthetic in performance metrics.",
                "Automated data-quality alerts for missing/outlier detention fields.",
                "Group fairness monitoring across legal status and offense classes.",
                "Mandatory review checklist before final hearing prioritization decisions.",
            ],
        }
    )
    st.dataframe(bias_df, use_container_width=True, height=360)

    st.markdown("### Priority Bucket Classification Properties")
    bucket_df = pd.DataFrame(
        {
            "priority_bucket": ["Low", "Medium", "High", "Critical"],
            "score_band": ["0-40", "40-65", "65-85", "85-100"],
            "classification_properties": [
                "No immediate legal urgency. Routine monitoring and scheduled hearing flow.",
                "Moderate urgency. Watch for delay build-up and vulnerability indicators.",
                "Strong urgency due to detention pressure, staleness, or legal/vulnerability factors.",
                "Highest urgency. Includes hard-override overstay undertrial scenarios and immediate judicial review candidates.",
            ],
        }
    )
    st.dataframe(bucket_df, use_container_width=True, height=220)

    bucket_dist = (
        ranked.groupby("priority_bucket", as_index=False)
        .agg(case_count=("case_id", "count"))
        .sort_values("case_count", ascending=False)
    )
    st.altair_chart(
        make_bar_chart(
            bucket_dist,
            "priority_bucket:N",
            "case_count:Q",
            "priority_bucket:N",
            "Current Dataset Distribution by Priority Bucket",
        ),
        use_container_width=True,
    )

    st.markdown("### Outputs Generated")
    st.markdown(
        """
        - Priority score and recommended hearing order.
        - Predicted urgency class and predicted case track.
        - Flagged undertrial cases requiring immediate review.
        - Cluster-based queue segmentation for operations planning.
        - Evaluation metrics for model quality and ranking stability.
        """
    )

    st.markdown("### Legal and Operational Safeguards")
    st.markdown(
        """
        - Non-finite sentence sections are handled separately to avoid false overstay flags.
        - Statutory bailability can override noisy text-derived bailability hints.
        - Medical vulnerability can elevate review priority even when overstay is not triggered.
        - The system is decision-support: final hearing priority remains with judicial authorities.
        """
    )


def render_class_analysis_page(result: dict) -> None:
    ranked = result["ranked"]
    st.subheader("Class Analysis")
    st.caption("Analyze class-wise volume and critical-case rates across model and legal classes")

    class_options = [
        "predicted_urgency",
        "predicted_case_type",
        "recommended_track",
        "offense_type",
        "case_complexity",
        "cluster_label",
    ]
    selected_class = st.selectbox("Classification dimension", class_options, index=0)

    class_rate = build_critical_rate_table(ranked, selected_class)
    distribution = build_distribution_table(ranked, selected_class)
    c1, c2 = st.columns([1, 1])
    with c1:
        st.dataframe(class_rate, use_container_width=True, height=340)
    with c2:
        st.altair_chart(
            make_bar_chart(
                class_rate,
                "class_name:N",
                "critical_rate_pct:Q",
                None,
                f"Critical Rate (%) by {selected_class}",
            ),
            use_container_width=True,
        )

    st.markdown("#### Data Distribution")
    d1, d2 = st.columns([1, 1])
    with d1:
        st.dataframe(distribution, use_container_width=True, height=300)
    with d2:
        dist_chart = alt.Chart(distribution).mark_arc(innerRadius=55).encode(
            theta=alt.Theta("case_count:Q"),
            color=alt.Color("class_name:N", title=selected_class),
            tooltip=["class_name", "case_count", "share_pct"],
        ).properties(height=300, title=f"Distribution by {selected_class}")
        st.altair_chart(dist_chart, use_container_width=True)

    class_mix_input = ranked.copy()
    class_mix_input[selected_class] = class_mix_input[selected_class].astype(str).fillna("Unknown")
    class_mix_input["priority_bucket"] = class_mix_input["priority_bucket"].astype(str).fillna("Unknown")
    class_mix = (
        class_mix_input.groupby([selected_class, "priority_bucket"], as_index=False, observed=False)
        .agg(case_count=("case_id", "count"))
        .rename(columns={selected_class: "class_name"})
    )
    mix_chart = alt.Chart(class_mix).mark_bar().encode(
        x=alt.X("class_name:N", title=selected_class, sort="-y"),
        y=alt.Y("case_count:Q", title="Cases"),
        color=alt.Color("priority_bucket:N", title="Priority bucket"),
        tooltip=["class_name", "priority_bucket", "case_count"],
    ).properties(height=340, title="Priority Bucket Mix inside each Class")
    st.altair_chart(mix_chart, use_container_width=True)


def render_status_analysis_page(result: dict) -> None:
    ranked = result["ranked"]
    st.subheader("Status Analysis")
    st.caption("Track critical rates by legal and procedural status")

    status_dims = ["legal_status", "trial_status", "statutory_bailability", "data_source"]
    tables = [build_critical_rate_table(ranked, dim) for dim in status_dims]
    combined = pd.concat(tables, ignore_index=True)

    st.dataframe(combined, use_container_width=True, height=320)

    selected_status_dim = st.selectbox("Status dimension", status_dims, index=0)
    selected_status_rate = build_critical_rate_table(ranked, selected_status_dim)
    selected_status_distribution = build_distribution_table(ranked, selected_status_dim)

    c1, c2 = st.columns([1, 1])
    with c1:
        st.altair_chart(
            make_bar_chart(
                selected_status_rate,
                "class_name:N",
                "critical_rate_pct:Q",
                None,
                f"Critical Rate (%) by {selected_status_dim}",
            ),
            use_container_width=True,
        )
    with c2:
        stale_status = (
            ranked.groupby(selected_status_dim, as_index=False)
            .agg(
                avg_last_hearing_gap=("last_hearing_gap", "mean"),
                avg_days_pending=("days_pending", "mean"),
            )
            .rename(columns={selected_status_dim: "status_class"})
        )
        stale_chart = alt.Chart(stale_status).mark_circle(size=120, opacity=0.75).encode(
            x=alt.X("avg_last_hearing_gap:Q", title="Avg last hearing gap"),
            y=alt.Y("avg_days_pending:Q", title="Avg days pending"),
            color=alt.Color("status_class:N", title=selected_status_dim),
            tooltip=["status_class", "avg_last_hearing_gap", "avg_days_pending"],
        ).properties(height=330, title="Status Staleness Map")
        st.altair_chart(stale_chart, use_container_width=True)

    st.markdown("#### Status Distribution")
    s1, s2 = st.columns([1, 1])
    with s1:
        st.dataframe(selected_status_distribution, use_container_width=True, height=290)
    with s2:
        status_dist_chart = make_bar_chart(
            selected_status_distribution,
            "class_name:N",
            "share_pct:Q",
            None,
            f"Share (%) by {selected_status_dim}",
        )
        st.altair_chart(status_dist_chart, use_container_width=True)


def render_prediction_tab(result: dict) -> None:
    st.subheader("Single Case Prediction Studio")

    if "predict_raw_text" not in st.session_state:
        st.session_state["predict_raw_text"] = "Undertrial for minor theft, prolonged detention, chronic illness and delayed hearings."
    if "predict_summary_text" not in st.session_state:
        st.session_state["predict_summary_text"] = "Undertrial for minor theft, prolonged detention, chronic illness and delayed hearings."
    if "predict_ipc_section" not in st.session_state:
        st.session_state["predict_ipc_section"] = "379"

    st.markdown("#### Case Summary Assistant")
    raw_text = st.text_area(
        "Raw case text",
        key="predict_raw_text",
        height=140,
        help="Paste detailed case text here, then click summarize.",
    )
    summarize_clicked = st.button("Generate case summary", use_container_width=True)

    if summarize_clicked:
        with st.spinner("Generating case summary..."):
            summary_text, sections, err = summarize_case_with_ollama(
                raw_text,
                DEFAULT_LOCAL_MODEL,
                DEFAULT_LOCAL_LLM_URL,
            )
        if summary_text:
            st.session_state["predict_summary_text"] = summary_text
            if sections:
                st.session_state["predict_ipc_section"] = " ".join(sections)
            st.success("Case summary generated.")
            if sections:
                st.info(f"Detected section(s): {' '.join(sections)}")
        else:
            st.session_state["predict_summary_text"] = summarize_case_locally(raw_text)
            fallback_sections = sections or extract_sections_from_case_text(raw_text)
            if fallback_sections:
                st.session_state["predict_ipc_section"] = " ".join(fallback_sections)
                st.info(f"Detected section(s): {' '.join(fallback_sections)}")
            if err:
                st.warning(err)
            st.info("Using local heuristic summary fallback.")

    with st.form("predict_case_form"):
        c1, c2, c3 = st.columns(3)
        summary = c1.text_area("Case summary", key="predict_summary_text", height=140)
        ipc_section = c2.text_input("IPC/BNS section", key="predict_ipc_section")
        offense_type = c3.text_input("Offense type", value="theft")

        c4, c5, c6 = st.columns(3)
        detention_days = c4.number_input("Detention days", min_value=0, value=1200)
        expected_sentence_days = c5.number_input("Expected sentence days", min_value=0, value=1095)
        days_pending = c6.number_input("Days pending", min_value=0, value=1400)

        c7, c8, c9 = st.columns(3)
        age = c7.number_input("Age", min_value=0, max_value=100, value=62)
        gender = c8.selectbox("Gender", ["M", "F", "Other"], index=1)
        legal_status = c9.selectbox("Legal status", ["undertrial", "bail", "convicted"], index=0)

        c10, c11, c12 = st.columns(3)
        bailable = c10.selectbox("Bailable", ["yes", "no"], index=0)
        case_complexity = c11.selectbox("Case complexity", ["low", "medium", "high"], index=0)
        trial_status = c12.selectbox("Trial status", ["active", "delayed", "pending"], index=1)

        c13, c14, c15 = st.columns(3)
        last_hearing_gap = c13.number_input("Last hearing gap", min_value=0, value=240)
        number_of_trials = c14.number_input("Number of trials", min_value=0, value=4)
        no_of_adjournments = c15.number_input("Adjournments", min_value=0, value=7)

        c16, c17, c18 = st.columns(3)
        health_flag = c16.checkbox("Health condition present", value=True)
        disability_flag = c17.checkbox("Disability present", value=False)
        bail_eligibility_nlp = c18.checkbox("Bail-eligible by NLP", value=True)

        medical_notes = st.text_input("Medical notes", value="Chronic kidney disease and regular medication needs")
        predict = st.form_submit_button("Predict priority")

    if not predict:
        return

    case_payload = {
        "summary": summary,
        "summary_nlp": summary,
        "ipc_section": ipc_section,
        "offense_type": offense_type,
        "bailable": bailable,
        "legal_status": legal_status,
        "case_type": "criminal",
        "detention_days": detention_days,
        "expected_sentence_days": expected_sentence_days,
        "life_sentence_flag": 0,
        "overstay_flag": detention_days > expected_sentence_days if expected_sentence_days > 0 else False,
        "detention_ratio": detention_days / expected_sentence_days if expected_sentence_days > 0 else 0,
        "days_pending": days_pending,
        "last_hearing_gap": last_hearing_gap,
        "number_of_trials": number_of_trials,
        "trial_status": trial_status,
        "no_of_adjournments": no_of_adjournments,
        "age": age,
        "gender": gender,
        "health_flag": int(health_flag),
        "disability_flag": int(disability_flag),
        "vulnerability_score": int(health_flag or disability_flag or age >= 60 or age < 18 or gender == "F"),
        "urgency_nlp": "high" if health_flag or detention_days > expected_sentence_days else "medium",
        "bail_eligibility_nlp": int(bail_eligibility_nlp),
        "case_complexity": case_complexity,
        "keywords": "manual streamlit prediction",
        "medical_notes": medical_notes,
    }

    scored_case = score_new_case(
        case_payload=case_payload,
        reference_df=result["merged"],
        statute_lookup=result["statute_lookup"],
        urgency_model=result["urgency_model"],
        track_model=result["track_model"],
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Priority score", f"{float(scored_case['priority_score']):.2f}")
    c2.metric("Priority bucket", str(scored_case["priority_bucket"]))
    c3.metric("Predicted urgency", str(scored_case["predicted_urgency"]))
    c4.metric("Recommended track", str(scored_case["predicted_case_type"]))

    explanation = pd.DataFrame(
        {
            "signal": [
                "Overstay flag",
                "Detention ratio",
                "Statutory bailability",
                "Non-finite sentence",
                "Medical vulnerability",
                "Flag reason",
            ],
            "value": [
                bool(scored_case["overstay_final"]),
                round(float(scored_case["detention_ratio_calc"]), 3),
                str(scored_case["statutory_bailability"]),
                bool(scored_case["statutory_non_finite_sentence"]),
                bool(scored_case["health_vulnerable"]),
                str(scored_case["flag_reason"]),
            ],
        }
    )
    st.dataframe(explanation, use_container_width=True, height=280)
    st.json(scored_case.to_dict())


def render_urgency_prediction_page(result: dict) -> None:
    st.subheader("Urgency Prediction Form")
    st.caption("Single-record urgency prediction with confidence scores")

    if "urgency_raw_text" not in st.session_state:
        st.session_state["urgency_raw_text"] = "Undertrial case with long detention and repeated hearing delays."
    if "urgency_summary_text" not in st.session_state:
        st.session_state["urgency_summary_text"] = "Undertrial case with long detention and repeated hearing delays."
    if "urgency_ipc_section" not in st.session_state:
        st.session_state["urgency_ipc_section"] = "379"

    st.markdown("#### Case Summary Assistant")
    raw_text = st.text_area(
        "Raw case text for urgency form",
        key="urgency_raw_text",
        height=120,
    )
    summarize_clicked = st.button("Generate summary for urgency form", use_container_width=True)

    if summarize_clicked:
        with st.spinner("Generating case summary..."):
            summary_text, sections, err = summarize_case_with_ollama(
                raw_text,
                DEFAULT_LOCAL_MODEL,
                DEFAULT_LOCAL_LLM_URL,
            )
        if summary_text:
            st.session_state["urgency_summary_text"] = summary_text
            if sections:
                st.session_state["urgency_ipc_section"] = " ".join(sections)
            st.success("Urgency-form summary generated.")
            if sections:
                st.info(f"Detected section(s): {' '.join(sections)}")
        else:
            st.session_state["urgency_summary_text"] = summarize_case_locally(raw_text)
            fallback_sections = sections or extract_sections_from_case_text(raw_text)
            if fallback_sections:
                st.session_state["urgency_ipc_section"] = " ".join(fallback_sections)
                st.info(f"Detected section(s): {' '.join(fallback_sections)}")
            if err:
                st.warning(err)
            st.info("Using local heuristic summary fallback.")

    st.text_area("Generated case summary", key="urgency_summary_text", height=110)

    with st.form("urgency_only_form"):
        c1, c2, c3 = st.columns(3)
        ipc_section = c1.text_input("IPC/BNS section", key="urgency_ipc_section")
        offense_type = c2.text_input("Offense type", value="theft")
        legal_status = c3.selectbox("Legal status", ["undertrial", "bail", "convicted"], index=0)

        c4, c5, c6 = st.columns(3)
        detention_days = c4.number_input("Detention days", min_value=0, value=1000, key="u_detention")
        expected_sentence_days = c5.number_input("Expected sentence days", min_value=0, value=1095, key="u_sentence")
        days_pending = c6.number_input("Days pending", min_value=0, value=1200, key="u_pending")

        c7, c8, c9 = st.columns(3)
        age = c7.number_input("Age", min_value=0, max_value=100, value=55, key="u_age")
        bailable = c8.selectbox("Bailable", ["yes", "no"], index=0, key="u_bailable")
        trial_status = c9.selectbox("Trial status", ["active", "delayed", "pending"], index=1, key="u_trial")

        c10, c11, c12 = st.columns(3)
        case_complexity = c10.selectbox("Case complexity", ["low", "medium", "high"], index=1, key="u_complex")
        health_flag = c11.checkbox("Health condition", value=False, key="u_health")
        disability_flag = c12.checkbox("Disability", value=False, key="u_disability")

        last_hearing_gap = st.number_input("Last hearing gap", min_value=0, value=180, key="u_last_gap")
        predict_urgency = st.form_submit_button("Predict urgency")

    if not predict_urgency:
        return

    case_payload = {
        "summary": st.session_state.get("urgency_summary_text", "Single-record urgency form entry"),
        "summary_nlp": st.session_state.get("urgency_summary_text", "Single-record urgency form entry"),
        "ipc_section": ipc_section,
        "offense_type": offense_type,
        "bailable": bailable,
        "legal_status": legal_status,
        "case_type": "criminal",
        "detention_days": detention_days,
        "expected_sentence_days": expected_sentence_days,
        "life_sentence_flag": 0,
        "overstay_flag": detention_days > expected_sentence_days if expected_sentence_days > 0 else False,
        "detention_ratio": detention_days / expected_sentence_days if expected_sentence_days > 0 else 0,
        "days_pending": days_pending,
        "last_hearing_gap": last_hearing_gap,
        "number_of_trials": 4,
        "trial_status": trial_status,
        "no_of_adjournments": 5,
        "age": age,
        "gender": "M",
        "health_flag": int(health_flag),
        "disability_flag": int(disability_flag),
        "vulnerability_score": int(health_flag or disability_flag or age >= 60 or age < 18),
        "urgency_nlp": "high" if health_flag else "medium",
        "bail_eligibility_nlp": 1 if bailable == "yes" else 0,
        "case_complexity": case_complexity,
        "keywords": "urgency form",
        "medical_notes": "",
    }

    scored_case = score_new_case(
        case_payload=case_payload,
        reference_df=result["merged"],
        statute_lookup=result["statute_lookup"],
        urgency_model=result["urgency_model"],
        track_model=result["track_model"],
    )

    urgency_input = pd.DataFrame([scored_case])[MODEL_FEATURES]
    classes = result["urgency_model"].named_steps["model"].classes_
    probs = result["urgency_model"].predict_proba(urgency_input)[0]
    probability_table = pd.DataFrame({"urgency_class": classes, "probability": probs}).sort_values("probability", ascending=False)
    probability_table["probability_pct"] = (probability_table["probability"] * 100).round(2)

    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted urgency", str(scored_case["predicted_urgency"]))
    c2.metric("Priority score", f"{float(scored_case['priority_score']):.2f}")
    c3.metric("Priority bucket", str(scored_case["priority_bucket"]))

    st.dataframe(probability_table[["urgency_class", "probability_pct"]], use_container_width=True, height=220)
    urgency_chart = alt.Chart(probability_table).mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6).encode(
        x=alt.X("urgency_class:N", title="Urgency class"),
        y=alt.Y("probability_pct:Q", title="Probability (%)"),
        tooltip=["urgency_class", "probability_pct"],
    ).properties(height=280, title="Urgency Prediction Confidence")
    st.altair_chart(urgency_chart, use_container_width=True)


def render_exports(result: dict, output_dir: str) -> None:
    st.sidebar.markdown("---")
    if st.sidebar.button("Save refreshed outputs"):
        save_outputs(result["clustered"], result["metrics"], output_dir)
        st.sidebar.success(f"Saved outputs to {output_dir}")

    st.sidebar.download_button(
        "Download ranked CSV",
        data=result["ranked"].to_csv(index=False).encode("utf-8"),
        file_name="prioritized_cases.csv",
        mime="text/csv",
    )


def main() -> None:
    inject_styles()

    st.sidebar.header("Pipeline Controls")
    data_dir = st.sidebar.text_input("Data directory", value="data")
    statute_file = st.sidebar.text_input("Statute lookup file", value=os.path.join("data", "ipc_bns_max_sentence_lookup.csv"))
    output_dir = st.sidebar.text_input("Output directory", value="outputs")
    augment_demo = st.sidebar.toggle("Include synthetic demo cases", value=True)

    with st.spinner("Running prioritization pipeline..."):
        result = load_pipeline_result(data_dir=data_dir, statute_file=statute_file, augment_demo=augment_demo)

    page = st.sidebar.radio(
        "Navigation",
        [
            "About",
            "Overview",
            "Class Analysis",
            "Status Analysis",
            "Queue Explorer",
            "Undertrial Review",
            "Cluster Analysis",
            "Prediction Studio",
            "Urgency Form",
            "Metrics",
        ],
        index=0,
    )

    render_exports(result, output_dir)
    render_header(result["metrics"], result["ranked"])

    if page == "About":
        render_about_page(result)
    elif page == "Overview":
        render_overview_tab(result)
    elif page == "Class Analysis":
        render_class_analysis_page(result)
    elif page == "Status Analysis":
        render_status_analysis_page(result)
    elif page == "Queue Explorer":
        render_queue_tab(result)
    elif page == "Undertrial Review":
        render_review_tab(result)
    elif page == "Cluster Analysis":
        render_cluster_tab(result)
    elif page == "Prediction Studio":
        render_prediction_tab(result)
    elif page == "Urgency Form":
        render_urgency_prediction_page(result)
    elif page == "Metrics":
        render_metrics_tab(result)


if __name__ == "__main__":
    main()
