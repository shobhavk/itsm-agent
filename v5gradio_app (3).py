"""
Gradio dashboard for the ITSM Quality Analysis Agent.

Runs in-process (mounted into the FastAPI app in main.py) so it calls the
pipeline directly rather than round-tripping through HTTP - simpler, faster,
and avoids needing an API key inside the browser session. The REST API
(/api/v1/...) remains available separately for machine-to-machine/automation
use cases, secured with its own API key as usual.
"""
import io
import os
import re
from datetime import datetime

import gradio as gr
import pandas as pd
import plotly.graph_objects as go

from app.services.pipeline import run_pipeline_from_bytes, run_pipeline_from_text
from app.services.persistence import compute_file_hash, get_cached_result, save_result

CUSTOM_CSS = """
:root {
    --dash-bg: #f5f6fa;
    --dash-border: #e5e9f0;
    --dash-text-muted: #64748b;
    --dash-text: #0f172a;
    --dash-shadow: 0 1px 3px rgba(15, 23, 42, 0.06);
}

.gradio-container {
    max-width: 1560px !important; margin: auto; padding: 16px 28px 32px !important;
    background: var(--dash-bg) !important; font-family: "Inter", "Segoe UI", system-ui, sans-serif;
}
footer {display: none !important;}

#header-banner {
    background: linear-gradient(90deg, #0f2540 0%, #16345c 100%);
    color: white; padding: 22px 28px; border-radius: 14px; margin-bottom: 14px;
}
#header-banner h1 {margin: 0; font-size: 1.35rem; font-weight: 600; letter-spacing: -0.01em;}
#header-banner p {margin: 6px 0 0 0; opacity: 0.85; font-size: 0.88rem;}

.severity-note {font-size: 0.78rem; color: var(--dash-text-muted); margin: 0 0 18px 2px; font-style: italic;}
.severity-note p {margin: 0;}

/* Section headings used above each dash-card block on the dashboard. */
.section-heading {
    font-size: 0.95rem !important; font-weight: 700 !important; color: var(--dash-text) !important;
    margin: 0 0 14px 0 !important; padding-bottom: 10px !important; border-bottom: 1px solid var(--dash-border) !important;
    letter-spacing: -0.01em;
}

#cache-notice {
    background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 10px;
    padding: 8px 14px; font-size: 0.85rem; color: #166534; margin-bottom: 14px;
}

/* Card wrapper used around every major section - gives the dribbble-style
   raised-panel look instead of controls floating on the bare page. */
.dash-card {
    background: #ffffff !important; border: 1px solid var(--dash-border) !important;
    border-radius: 16px !important; padding: 18px 20px !important; box-shadow: var(--dash-shadow);
}

/* KPI strip */
#metrics-row {margin-bottom: 18px; gap: 14px !important;}
.kpi-grid {display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px;}
.kpi-card {
    background: #ffffff; border: 1px solid var(--dash-border); border-radius: 14px;
    padding: 16px 20px; box-shadow: var(--dash-shadow); border-left: 4px solid var(--accent, #3b82f6);
    transition: box-shadow 0.15s ease;
}
.kpi-card:hover {box-shadow: 0 4px 10px rgba(15, 23, 42, 0.09);}
.kpi-label {font-size: 0.72rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: var(--dash-text-muted); margin-bottom: 8px;}
.kpi-value {font-size: 1.65rem; font-weight: 700; color: var(--dash-text); line-height: 1; font-variant-numeric: tabular-nums;}

/* Full-width input bar - a single horizontal card housing the upload,
   paste, analyze, and download controls with consistent alignment. */
#input-row {gap: 20px !important; margin-bottom: 18px; align-items: end !important;}
#input-row label {font-weight: 600; font-size: 0.82rem; color: var(--dash-text);}
#input-row .gr-button.primary, #input-row button.primary {
    border-radius: 10px !important; font-weight: 600 !important; box-shadow: 0 1px 2px rgba(15,23,42,.18);
    height: 42px !important;
}
#action-col {display: flex; flex-direction: column; gap: 10px; justify-content: flex-end;}

/* Compact upload dropzone - the default Gradio File drop area is tall
   and mostly empty space; shrink it down to a slim strip. */
#file-upload {min-height: 0 !important;}
#file-upload .wrap {
    min-height: 54px !important; padding: 8px 12px !important;
}
#file-upload .wrap svg {width: 18px !important; height: 18px !important; margin-bottom: 2px !important;}
#file-upload .wrap > * {font-size: 0.75rem !important;}

#results-section {margin-bottom: 20px;}

/* Results table - fixed-height, single-line rows instead of letting long
   Description/Worklog text blow rows out. The Python side already
   truncates + strips HTML tags (see _preview_html); this just makes sure
   the CSS doesn't fight that by re-wrapping or auto-growing rows. Full
   text is available on hover via the native title tooltip. */
#results-table table th {
    background: #f8fafc !important; font-weight: 600 !important; font-size: 0.72rem !important;
    text-transform: uppercase; letter-spacing: 0.03em; color: var(--dash-text-muted) !important;
    padding: 6px 10px !important;
}
#results-table table td {
    padding: 4px 10px !important; font-size: 0.78rem !important;
    height: 26px !important; max-height: 26px !important; line-height: 1.1 !important;
    vertical-align: middle !important;
    white-space: nowrap !important; overflow: hidden !important; text-overflow: ellipsis !important;
    border-bottom: 1px solid #eef1f5 !important;
}
#results-table table td span[title] {cursor: help;}
#results-table table tbody tr:nth-child(even) td {background: #fbfcfe !important;}
#results-table table tbody tr:hover td {background: #f8fafc !important;}

#pagination-row {
    margin-top: 14px; display: flex; align-items: center; justify-content: center; gap: 16px;
}
#pagination-row button {border-radius: 8px !important; font-weight: 600 !important;}
#page-indicator {text-align: center; font-size: 0.85rem; color: var(--dash-text-muted); padding-top: 8px; font-weight: 500;}

#chart-row {margin-top: 4px; gap: 14px !important;}
#chart-row .plotly {border-radius: 10px;}
#insights-row {margin-top: 14px; gap: 14px !important;}
#insights-row table {font-size: 0.85rem !important;}
#insights-row table th {background: #f8fafc !important; font-weight: 600 !important; color: var(--dash-text-muted) !important;}

/* Animated "agent working" progress bar - shown in one fixed, centered
   spot (a compact card, not a full-width strip) while an analysis is
   running, instead of Gradio's default per-component loading overlays
   scattered across the summary/donut chart section. */
#agent-progress {margin: 0 auto 16px; max-width: 480px;}
.agent-progress {
    display: flex; align-items: center; gap: 14px;
    background: #ffffff; border: 1px solid var(--dash-border); border-radius: 14px;
    padding: 12px 20px; box-shadow: var(--dash-shadow);
}
.agent-progress-icon {
    font-size: 1.3rem; flex-shrink: 0;
    animation: agent-bounce 1s ease-in-out infinite;
}
@keyframes agent-bounce {
    0%, 100% {transform: translateY(0) rotate(0deg);}
    50% {transform: translateY(-4px) rotate(-6deg);}
}
.agent-progress-track {
    position: relative; flex: 1; height: 8px; border-radius: 999px;
    background: #eef1f5; overflow: hidden;
}
.agent-progress-fill {
    position: absolute; top: 0; left: -40%; width: 40%; height: 100%; border-radius: 999px;
    background: linear-gradient(90deg, #16345c, #3b82f6, #16345c);
    animation: agent-slide 1.15s ease-in-out infinite;
}
@keyframes agent-slide {
    0% {left: -40%;}
    100% {left: 100%;}
}
.agent-progress-text {
    font-size: 0.85rem; font-weight: 600; color: var(--dash-text);
    white-space: nowrap; flex-shrink: 0;
}
.agent-progress-text .dots span {
    animation: agent-dot 1.4s infinite; opacity: 0;
}
.agent-progress-text .dots span:nth-child(2) {animation-delay: 0.2s;}
.agent-progress-text .dots span:nth-child(3) {animation-delay: 0.4s;}
@keyframes agent-dot {
    0% {opacity: 0;}
    20% {opacity: 1;}
    100% {opacity: 0;}
}
"""

AGENT_PROGRESS_HTML = """
<div class="agent-progress">
  <span class="agent-progress-icon">🤖</span>
  <div class="agent-progress-track"><div class="agent-progress-fill"></div></div>
  <span class="agent-progress-text">Agent analyzing tickets<span class="dots"><span>.</span><span>.</span><span>.</span></span></span>
</div>
"""

SEVERITY_NOTE = (
    "Free-text fields (description/worklog) are treated as untrusted data end-to-end - "
    "they are never executed as instructions by the underlying models."
)


def _score_badge(score: int) -> str:
    if score >= 75:
        return "🟢 Good"
    if score >= 50:
        return "🟡 Needs improvement"
    return "🔴 Poor"


_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def _strip_html(text: str) -> str:
    """Strips HTML tags and collapses whitespace/newlines so table cells
    render as a single clean line instead of wrapping across many lines."""
    text = text or ""
    text = _TAG_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


def _truncate(text: str, limit: int) -> str:
    text = _strip_html(text)
    return text[:limit] + "…" if len(text) > limit else text


ALL_COLUMNS = [
    "Ticket ID", "Category", "Category Confidence", "Category Method",
    "Short Description", "Description", "Worklog Notes", "Worklog Score",
    "Worklog Rating", "Worklog Flags", "Priority", "Status",
    "Assignment Group", "Host / CI", "Validation Notes",
]
DEFAULT_VISIBLE_COLUMNS = [
    "Ticket ID", "Category", "Description", "Worklog Notes",
    "Worklog Score", "Priority", "Assignment Group",
]


def _results_to_dataframe(analysis) -> pd.DataFrame:
    rows = []
    for r in analysis.results:
        rows.append(
            {
                "Ticket ID": r.ticket_id,
                "Category": r.category,
                "Category Confidence": r.category_confidence,
                "Category Method": r.category_method,
                "Short Description": _truncate(r.short_description, 100),
                "Description": _truncate(r.description, 140),
                "Worklog Notes": _truncate(r.worklog, 140),
                "Worklog Score": r.worklog_score,
                "Worklog Rating": _score_badge(r.worklog_score),
                "Worklog Flags": "; ".join(r.worklog_flags) if r.worklog_flags else "",
                "Priority": r.priority or "",
                "Status": r.status or "",
                "Assignment Group": r.assignment_group or "",
                "Validation Notes": "; ".join(r.validation_flags) if r.validation_flags else "",
            }
        )
    return pd.DataFrame(rows)


def _results_to_full_dataframe(analysis) -> pd.DataFrame:
    """Untruncated version for CSV export - the on-screen table truncates
    long text for readability, but "download full results" should contain
    the actual full text, not the display-truncated version."""
    rows = []
    for r in analysis.results:
        rows.append(
            {
                "Ticket ID": r.ticket_id,
                "Category": r.category,
                "Category Confidence": r.category_confidence,
                "Category Method": r.category_method,
                "Short Description": r.short_description,
                "Description": r.description,
                "Worklog Notes": r.worklog,
                "Worklog Score": r.worklog_score,
                "Worklog Rating": _score_badge(r.worklog_score),
                "Worklog Flags": "; ".join(r.worklog_flags) if r.worklog_flags else "",
                "Priority": r.priority or "",
                "Status": r.status or "",
                "Assignment Group": r.assignment_group or "",
                "Host / CI": r.host or "",
                "Validation Notes": "; ".join(r.validation_flags) if r.validation_flags else "",
            }
        )
    return pd.DataFrame(rows)


def _ranked_markdown(title: str, counts: dict, column_label: str, top_n: int = 5) -> str:
    """Small ranked table for a dashboard insight panel, e.g. top
    categories or top hosts by ticket count."""
    if not counts:
        return f"### {title}\nNo data to rank yet - run an analysis first."
    items = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    lines = [f"### {title}", f"| Rank | {column_label} | Tickets |", "|---|---|---|"]
    for i, (name, count) in enumerate(items, start=1):
        lines.append(f"| {i} | {name} | {count} |")
    return "\n".join(lines)


def _category_chart_df(category_counts: dict) -> pd.DataFrame:
    if not category_counts:
        return pd.DataFrame({"Category": [], "Count": []})
    # Descending here since a donut chart reads clockwise from the top,
    # largest slice first.
    items = sorted(category_counts.items(), key=lambda kv: kv[1], reverse=True)
    return pd.DataFrame(items, columns=["Category", "Count"])


def _category_chart_figure(category_counts: dict) -> go.Figure:
    """Donut chart of ticket volume by category (replaces the old bar chart).
    Takes a plain {category: count} dict so it works for both a fresh
    analysis (analysis.category_counts) and a cached result (loaded from
    the DB, no analysis object available)."""
    chart_df = _category_chart_df(category_counts)
    if chart_df.empty:
        fig = go.Figure()
        fig.update_layout(
            annotations=[dict(text="No data yet", showarrow=False, font=dict(size=14))],
            height=340,
            margin=dict(t=30, b=10, l=10, r=10),
        )
        return fig

    fig = go.Figure(
        data=[
            go.Pie(
                labels=chart_df["Category"],
                values=chart_df["Count"],
                hole=0.55,
                sort=False,
                textinfo="percent",
                hovertemplate="%{label}: %{value} tickets (%{percent})<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        title="Tickets by Category",
        height=340,
        margin=dict(t=40, b=10, l=10, r=10),
        legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.02),
    )
    return fig


DEFAULT_PAGE_SIZE = 10


def _summary_markdown(stats: dict) -> str:
    return (
        f"### Summary\n"
        f"| Metric | Value |\n|---|---|\n"
        f"| Total records seen | {stats['total_records']} |\n"
        f"| Valid records analyzed | {stats['valid_records']} |\n"
        f"| Rejected records | {stats['rejected_records']} |\n"
        f"| Average worklog score | {stats['average_worklog_score']} / 100 |\n"
    )


def _kpi_card(label: str, value, accent: str) -> str:
    return (
        f'<div class="kpi-card" style="--accent:{accent}">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>'
        f"</div>"
    )


_KPI_PLACEHOLDER = (
    '<div class="kpi-grid">'
    + _kpi_card("Total Records Seen", "—", "#3b82f6")
    + _kpi_card("Valid Records Analyzed", "—", "#10b981")
    + _kpi_card("Rejected Records", "—", "#ef4444")
    + _kpi_card("Average Worklog Score", "—", "#f59e0b")
    + "</div>"
)


def _summary_kpi_html(stats: dict) -> str:
    """Renders the same summary_stats dict used by _summary_markdown as a
    row of KPI cards instead of a markdown table - same underlying data,
    a management-dashboard-style presentation."""
    return (
        '<div class="kpi-grid">'
        + _kpi_card("Total Records Seen", stats["total_records"], "#3b82f6")
        + _kpi_card("Valid Records Analyzed", stats["valid_records"], "#10b981")
        + _kpi_card("Rejected Records", stats["rejected_records"], "#ef4444")
        + _kpi_card("Average Worklog Score", f'{stats["average_worklog_score"]} / 100', "#f59e0b")
        + "</div>"
    )


def _analysis_to_summary_stats(analysis) -> dict:
    return {
        "total_records": analysis.total_records,
        "valid_records": analysis.valid_records,
        "rejected_records": analysis.rejected_records,
        "average_worklog_score": analysis.average_worklog_score,
    }


def _truncate_full_df(full_df: pd.DataFrame) -> pd.DataFrame:
    """Builds the on-screen truncated view from a cached full (untruncated)
    dataframe - mirrors what _results_to_dataframe does from a fresh
    analysis, just starting from a dataframe instead of analysis.results."""
    if full_df is None or len(full_df) == 0:
        return pd.DataFrame(columns=ALL_COLUMNS)
    df = full_df.copy()
    if "Short Description" in df.columns:
        df["Short Description"] = df["Short Description"].apply(lambda t: _truncate(t, 100))
    if "Description" in df.columns:
        df["Description"] = df["Description"].apply(lambda t: _truncate(t, 140))
    if "Worklog Notes" in df.columns:
        df["Worklog Notes"] = df["Worklog Notes"].apply(lambda t: _truncate(t, 140))
    return df


async def _analyze(file_obj, pasted_text):
    if file_obj is None and not (pasted_text and pasted_text.strip()):
        raise gr.Error("Upload a file (CSV/XLSX/TXT) or paste incident text first.")

    # Show the animated agent progress bar in its single fixed spot before
    # doing any work; other outputs are left untouched (gr.update()) so
    # nothing under them flickers or shows its own loading state.
    yield gr.update(visible=True), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()

    # Hash the raw input (file bytes, or the pasted text) so an identical
    # upload/paste can be served straight from the DB instead of hitting
    # the LLM pipeline again.
    if file_obj is not None:
        with open(file_obj.name, "rb") as f:
            content = f.read()
        input_label = file_obj.name
    else:
        content = pasted_text.strip().encode("utf-8")
        input_label = "pasted_text"

    file_hash = compute_file_hash(content)
    llm_worklog_scoring_enabled = os.environ.get("ENABLE_LLM_WORKLOG_SCORING", "false").lower() == "true"
    cached = get_cached_result(file_hash, current_llm_worklog_scoring_enabled=llm_worklog_scoring_enabled)

    if cached is not None:
        full_df = cached["full_df"]
        summary_stats = cached["summary_stats"]
        category_counts = cached["category_counts"]
        host_counts = cached["host_counts"]
        cached_on = cached["uploaded_at"][:19].replace("T", " ")
        cache_notice = gr.update(
            value=f"✅ Identical input already analyzed on {cached_on} — showing cached results, no LLM calls made.",
            visible=True,
        )
    else:
        if file_obj is not None:
            analysis = await run_pipeline_from_bytes(file_obj.name, content)
        else:
            analysis = await run_pipeline_from_text(pasted_text)

        full_df = _results_to_full_dataframe(analysis)
        summary_stats = _analysis_to_summary_stats(analysis)
        category_counts = analysis.category_counts
        host_counts = analysis.host_counts

        save_result(
            file_hash=file_hash,
            filename=input_label,
            full_df=full_df,
            summary_stats=summary_stats,
            category_counts=category_counts,
            host_counts=host_counts,
            llm_worklog_scoring_enabled=llm_worklog_scoring_enabled,
        )
        cache_notice = gr.update(value="", visible=False)

    df = _truncate_full_df(full_df)
    summary = _summary_kpi_html(summary_stats)

    # Prepare CSV for download - always the FULL untruncated result set,
    # independent of whatever filter/page/truncation the on-screen table
    # is showing.
    csv_buf = io.StringIO()
    full_df.to_csv(csv_buf, index=False)
    csv_path = f"/tmp/itsm_quality_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    with open(csv_path, "w") as f:
        f.write(csv_buf.getvalue())

    chart = _category_chart_figure(category_counts)
    top_issues_md = _ranked_markdown("🔁 Most Recurring Issues", category_counts, "Category")
    top_hosts_md = _ranked_markdown("🖥️ Top Hosts / Servers by Alerts", host_counts, "Host / CI")

    yield gr.update(visible=False), summary, chart, csv_path, df, cache_notice, top_issues_md, top_hosts_md


def _select_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Restricts a dataframe to the fixed default column set for on-screen
    display, in ALL_COLUMNS order. Filtering/pagination/CSV export always
    operate on the full, un-reduced dataframe - only this final display
    step drops columns."""
    cols = [c for c in ALL_COLUMNS if c in DEFAULT_VISIBLE_COLUMNS]
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=cols or ["Ticket ID"])
    cols = [c for c in cols if c in df.columns]
    if not cols:
        cols = ["Ticket ID"]  # never render a fully empty table
    return df[cols]


def _apply_filters(full_df: pd.DataFrame, category: str, min_score: int) -> pd.DataFrame:
    if full_df is None or len(full_df) == 0:
        return pd.DataFrame() if full_df is None else full_df
    filtered = full_df.copy()
    if category and category != "All":
        filtered = filtered[filtered["Category"] == category]
    filtered = filtered[filtered["Worklog Score"] >= min_score]
    return filtered.reset_index(drop=True)


def _paginate(filtered_df: pd.DataFrame, page: int, page_size: int):
    total = len(filtered_df) if filtered_df is not None else 0
    page_size = page_size or DEFAULT_PAGE_SIZE
    total_pages = max(1, -(-total // page_size))  # ceil division
    page = max(1, min(page, total_pages))
    start = (page - 1) * page_size
    end = start + page_size
    page_df = filtered_df.iloc[start:end] if total else filtered_df
    indicator = f"Page {page} of {total_pages}  ·  {total} ticket{'s' if total != 1 else ''}"
    return page_df, indicator, page


def _refresh_view(full_df, category, min_score, page_size):
    """Re-applies filters, resets to page 1, and returns everything the
    table/pagination controls need. Used after a new analysis runs or
    whenever a filter/page-size control changes."""
    filtered = _apply_filters(full_df, category, min_score)
    page_df, indicator, page = _paginate(filtered, 1, page_size)
    return _select_columns(page_df), indicator, filtered, page


def _go_to_page(filtered_df, page, page_size, delta):
    page_df, indicator, new_page = _paginate(filtered_df, (page or 1) + delta, page_size)
    return _select_columns(page_df), indicator, new_page


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="ITSM Quality Analysis Agent", css=CUSTOM_CSS) as demo:
        gr.HTML(
            """
            <div id="header-banner">
              <h1>🛠️ ITSM Quality Analysis Agent</h1>
              <p>Upload incident data (Excel / CSV / TXT) or paste raw incident text to auto-categorize
              tickets and score worklog quality.</p>
            </div>
            """
        )
        gr.Markdown(f"_{SEVERITY_NOTE}_", elem_classes=["severity-note"])

        agent_progress = gr.HTML(AGENT_PROGRESS_HTML, elem_id="agent-progress", visible=False)
        cache_notice = gr.Markdown(visible=False, elem_id="cache-notice")

        # Single full-width input bar - upload, paste, and the
        # analyze/download actions all sit on one row so the rest of the
        # dashboard (KPIs, table, charts) reads as one continuous page below.
        with gr.Row(elem_id="input-row", elem_classes=["dash-card"], equal_height=False):
            with gr.Column(scale=3, min_width=260):
                file_input = gr.File(
                    label="Upload incident file (.xlsx, .csv, .txt)",
                    file_types=[".xlsx", ".xls", ".csv", ".txt"],
                    elem_id="file-upload",
                )
            with gr.Column(scale=4, min_width=320):
                text_input = gr.Textbox(label="...or paste unstructured incident text", lines=3,
                                          placeholder="INC0012345\nShort description: ...\nWorklog: ...")
            with gr.Column(scale=2, min_width=180, elem_id="action-col"):
                analyze_btn = gr.Button("Analyze", variant="primary")
                download_file = gr.File(label="Download full results as CSV", interactive=False)

        # KPI strip - headline numbers for management at a glance.
        with gr.Row(elem_id="metrics-row"):
            summary_md = gr.HTML(_KPI_PLACEHOLDER)

        # Analyzed Tickets: the primary, full-width table - moved above the
        # charts so it's the first thing reviewers see after the KPIs.
        with gr.Column(elem_id="results-section", elem_classes=["dash-card"]):
            gr.Markdown("### 📋 Analyzed Tickets", elem_classes=["section-heading"])
            results_table = gr.Dataframe(
                label=None,
                show_label=False,
                interactive=False,
                wrap=False,
                max_height=460,
                elem_id="results-table",
            )
            with gr.Row(elem_id="pagination-row"):
                prev_btn = gr.Button("← Previous", size="sm")
                page_indicator = gr.Markdown("Page 1 of 1  ·  0 tickets", elem_id="page-indicator")
                next_btn = gr.Button("Next →", size="sm")

        # Category donut chart below the table.
        with gr.Row(elem_id="chart-row"):
            with gr.Column(scale=1, elem_classes=["dash-card"]):
                gr.Markdown("### 📊 Category Distribution", elem_classes=["section-heading"])
                category_chart = gr.Plot(show_label=False)

        with gr.Row(elem_id="insights-row"):
            with gr.Column(scale=1, elem_classes=["dash-card"]):
                top_issues_md = gr.Markdown("### 🔁 Most Recurring Issues\nRun an analysis to see this.")
            with gr.Column(scale=1, elem_classes=["dash-card"]):
                top_hosts_md = gr.Markdown("### 🖥️ Top Hosts / Servers by Alerts\nRun an analysis to see this.")

        full_results_state = gr.State(pd.DataFrame())
        filtered_results_state = gr.State(pd.DataFrame())
        page_state = gr.State(1)
        # Category/score filters and the rows-per-page control have been
        # removed from the UI; these fixed states keep _apply_filters /
        # _paginate (and their existing behavior) unchanged underneath.
        category_state = gr.State("All")
        min_score_state = gr.State(0)
        page_size_state = gr.State(DEFAULT_PAGE_SIZE)

        analyze_btn.click(
            fn=_analyze,
            inputs=[file_input, text_input],
            outputs=[
                agent_progress, summary_md, category_chart, download_file,
                full_results_state, cache_notice, top_issues_md, top_hosts_md,
            ],
        ).then(
            fn=_refresh_view,
            inputs=[full_results_state, category_state, min_score_state, page_size_state],
            outputs=[results_table, page_indicator, filtered_results_state, page_state],
        )

        prev_btn.click(
            fn=lambda filtered_df, page, page_size: _go_to_page(filtered_df, page, page_size, -1),
            inputs=[filtered_results_state, page_state, page_size_state],
            outputs=[results_table, page_indicator, page_state],
        )
        next_btn.click(
            fn=lambda filtered_df, page, page_size: _go_to_page(filtered_df, page, page_size, 1),
            inputs=[filtered_results_state, page_state, page_size_state],
            outputs=[results_table, page_indicator, page_state],
        )

    return demo
