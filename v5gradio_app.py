"""
Gradio dashboard for the ITSM Quality Analysis Agent.

Runs in-process (mounted into the FastAPI app in main.py) so it calls the
pipeline directly rather than round-tripping through HTTP - simpler, faster,
and avoids needing an API key inside the browser session. The REST API
(/api/v1/...) remains available separately for machine-to-machine/automation
use cases, secured with its own API key as usual.
"""
import io
import re
from datetime import datetime

import gradio as gr
import pandas as pd
import plotly.graph_objects as go

from app.services.pipeline import run_pipeline_from_bytes, run_pipeline_from_text

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

.severity-note {font-size: 0.8rem; color: var(--dash-text-muted); margin: 0 0 18px 2px;}

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
    padding: 14px 18px; box-shadow: var(--dash-shadow); border-left: 4px solid var(--accent, #3b82f6);
}
.kpi-label {font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.04em; color: var(--dash-text-muted); margin-bottom: 6px;}
.kpi-value {font-size: 1.5rem; font-weight: 700; color: var(--dash-text); line-height: 1;}

#input-row {gap: 18px !important; margin-bottom: 18px; align-items: stretch;}
#input-col {max-width: 340px; display: flex; flex-direction: column; gap: 10px;}
#input-col label {font-weight: 600; font-size: 0.85rem;}
#input-col button.primary {
    border-radius: 10px !important; font-weight: 600 !important; box-shadow: 0 1px 2px rgba(15,23,42,.18);
}
#results-col h3 {margin: 0 0 12px; font-size: 1.02rem; font-weight: 600;}

/* Compact upload dropzone - the default Gradio File drop area is tall
   and mostly empty space; shrink it down to a slim strip. */
#file-upload {min-height: 0 !important;}
#file-upload .wrap {
    min-height: 54px !important; padding: 8px 12px !important;
}
#file-upload .wrap svg {width: 18px !important; height: 18px !important; margin-bottom: 2px !important;}
#file-upload .wrap > * {font-size: 0.75rem !important;}

/* Column picker moved into the left rail, next to the upload controls -
   keep it compact so it doesn't stretch the column, with its own small
   scroll once a few columns are picked instead of growing forever. */
#column-select {margin-top: 2px; max-width: 420px;}
#column-select label {font-size: 0.85rem;}
#column-select .wrap-inner {
    max-height: 92px !important; overflow-y: auto !important; gap: 4px !important;
}
#column-select .token {
    font-size: 0.7rem !important; padding: 2px 8px !important; border-radius: 999px !important;
}

#filters-row {margin-bottom: 14px; gap: 18px !important;}

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
    margin-top: 16px; display: flex; align-items: center; justify-content: center; gap: 16px;
}
#pagination-row button {border-radius: 8px !important;}
#page-indicator {text-align: center; font-size: 0.85rem; color: var(--dash-text-muted); padding-top: 8px;}

#chart-row {margin-top: 20px;}

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
    "Assignment Group", "Validation Notes",
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
                "Validation Notes": "; ".join(r.validation_flags) if r.validation_flags else "",
            }
        )
    return pd.DataFrame(rows)


def _category_chart_df(analysis) -> pd.DataFrame:
    if not analysis.category_counts:
        return pd.DataFrame({"Category": [], "Count": []})
    # Descending here since a donut chart reads clockwise from the top,
    # largest slice first.
    items = sorted(analysis.category_counts.items(), key=lambda kv: kv[1], reverse=True)
    return pd.DataFrame(items, columns=["Category", "Count"])


def _category_chart_figure(analysis) -> go.Figure:
    """Donut chart of ticket volume by category (replaces the old bar chart)."""
    chart_df = _category_chart_df(analysis)
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


PAGE_SIZE_CHOICES = [10, 25, 50]
DEFAULT_PAGE_SIZE = 10


def _summary_markdown(analysis) -> str:
    return (
        f"### Summary\n"
        f"| Metric | Value |\n|---|---|\n"
        f"| Total records seen | {analysis.total_records} |\n"
        f"| Valid records analyzed | {analysis.valid_records} |\n"
        f"| Rejected records | {analysis.rejected_records} |\n"
        f"| Average worklog score | {analysis.average_worklog_score} / 100 |\n"
    )


async def _analyze(file_obj, pasted_text):
    if file_obj is None and not (pasted_text and pasted_text.strip()):
        raise gr.Error("Upload a file (CSV/XLSX/TXT) or paste incident text first.")

    # Show the animated agent progress bar in its single fixed spot before
    # doing any work; other outputs are left untouched (gr.update()) so
    # nothing under them flickers or shows its own loading state.
    yield gr.update(visible=True), gr.update(), gr.update(), gr.update(), gr.update()

    if file_obj is not None:
        with open(file_obj.name, "rb") as f:
            content = f.read()
        analysis = await run_pipeline_from_bytes(file_obj.name, content)
    else:
        analysis = await run_pipeline_from_text(pasted_text)

    df = _results_to_dataframe(analysis)
    full_df = _results_to_full_dataframe(analysis)
    summary = _summary_markdown(analysis)

    # Prepare CSV for download - always the FULL untruncated result set,
    # independent of whatever filter/page/truncation the on-screen table
    # is showing.
    csv_buf = io.StringIO()
    full_df.to_csv(csv_buf, index=False)
    csv_path = f"/tmp/itsm_quality_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    with open(csv_path, "w") as f:
        f.write(csv_buf.getvalue())

    chart = _category_chart_figure(analysis)

    yield gr.update(visible=False), summary, chart, csv_path, df


def _select_columns(df: pd.DataFrame, selected_cols) -> pd.DataFrame:
    """Restricts a dataframe to the user-chosen columns for on-screen
    display, keeping them in the fixed ALL_COLUMNS order regardless of the
    order the user (de)selected them in. Filtering/pagination/CSV export
    always operate on the full, un-reduced dataframe - only this final
    display step drops columns."""
    cols = [c for c in ALL_COLUMNS if c in (selected_cols or DEFAULT_VISIBLE_COLUMNS)]
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


def _refresh_view(full_df, category, min_score, page_size, columns):
    """Re-applies filters, resets to page 1, and returns everything the
    table/pagination controls need. Used after a new analysis runs or
    whenever a filter/page-size control changes."""
    filtered = _apply_filters(full_df, category, min_score)
    page_df, indicator, page = _paginate(filtered, 1, page_size)
    return _select_columns(page_df, columns), indicator, filtered, page


def _go_to_page(filtered_df, page, page_size, columns, delta):
    page_df, indicator, new_page = _paginate(filtered_df, (page or 1) + delta, page_size)
    return _select_columns(page_df, columns), indicator, new_page


def _apply_columns(filtered_df, page, page_size, columns):
    """Just re-renders the current page with the newly (de)selected
    columns - doesn't touch filters or reset pagination."""
    page_df, indicator, page = _paginate(filtered_df, page or 1, page_size)
    return _select_columns(page_df, columns), indicator, page


def build_ui() -> gr.Blocks:
    from app.models.schemas import CATEGORIES

    with gr.Blocks(title="ITSM Quality Analysis Agent") as demo:
        gr.HTML(
            """
            <div id="header-banner">
              <h1>🛠️ ITSM Quality Analysis Agent</h1>
              <p>Upload incident data (Excel / CSV / TXT) or paste raw incident text to auto-categorize
              tickets and score worklog quality.</p>
            </div>
            """
        )
        gr.Markdown(f"_{SEVERITY_NOTE}_")

        agent_progress = gr.HTML(AGENT_PROGRESS_HTML, elem_id="agent-progress", visible=False)

        with gr.Row(elem_id="input-row", equal_height=False):
            # Compact input column - just enough for the upload/paste/analyze
            # controls, so the results table gets the bulk of the width.
            with gr.Column(scale=1, min_width=280, elem_id="input-col"):
                file_input = gr.File(
                    label="Upload incident file (.xlsx, .csv, .txt)",
                    file_types=[".xlsx", ".xls", ".csv", ".txt"],
                    elem_id="file-upload",
                )
                text_input = gr.Textbox(label="...or paste unstructured incident text", lines=3,
                                          placeholder="INC0012345\nShort description: ...\nWorklog: ...")
                analyze_btn = gr.Button("Analyze", variant="primary")
                download_file = gr.File(label="Download full results as CSV", interactive=False)

            # Categorized results sit to the right of the input column.
            with gr.Column(scale=3, elem_id="results-col"):
                gr.Markdown("### Categorized Results")
                with gr.Row(elem_id="filters-row"):
                    category_filter = gr.Dropdown(choices=["All"] + [c for c in CATEGORIES], value="All", label="Filter by category")
                    score_filter = gr.Slider(0, 100, value=0, step=5, label="Minimum worklog score")
                    page_size_dd = gr.Dropdown(choices=PAGE_SIZE_CHOICES, value=DEFAULT_PAGE_SIZE, label="Rows per page")

                # Column picker lives right on top of the Analyzed Tickets
                # table itself, and only shows up once there's actually a
                # result set to pick columns from.
                column_select = gr.Dropdown(
                    choices=ALL_COLUMNS, value=DEFAULT_VISIBLE_COLUMNS,
                    multiselect=True, label="Columns to display",
                    elem_id="column-select", visible=False,
                )

                results_table = gr.Dataframe(
                    label="Analyzed Tickets",
                    interactive=False,
                    wrap=False,
                    max_height=400,
                    elem_id="results-table",
                )

                with gr.Row(elem_id="pagination-row"):
                    prev_btn = gr.Button("← Previous", size="sm")
                    page_indicator = gr.Markdown("Page 1 of 1  ·  0 tickets", elem_id="page-indicator")
                    next_btn = gr.Button("Next →", size="sm")

        # Summary stats + category donut chart get their own full-width row
        # below the input/results row.
        with gr.Row(elem_id="chart-row"):
            with gr.Column(scale=1):
                summary_md = gr.Markdown("Run an analysis to see summary stats here.")
            with gr.Column(scale=1):
                category_chart = gr.Plot(label="Category Distribution")

        full_results_state = gr.State(pd.DataFrame())
        filtered_results_state = gr.State(pd.DataFrame())
        page_state = gr.State(1)

        analyze_btn.click(
            fn=_analyze,
            inputs=[file_input, text_input],
            outputs=[agent_progress, summary_md, category_chart, download_file, full_results_state],
        ).then(
            fn=_refresh_view,
            inputs=[full_results_state, category_filter, score_filter, page_size_dd, column_select],
            outputs=[results_table, page_indicator, filtered_results_state, page_state],
        ).then(
            fn=lambda: gr.update(visible=True),
            outputs=[column_select],
        )

        for control in (category_filter, score_filter, page_size_dd):
            control.change(
                fn=_refresh_view,
                inputs=[full_results_state, category_filter, score_filter, page_size_dd, column_select],
                outputs=[results_table, page_indicator, filtered_results_state, page_state],
            )

        column_select.change(
            fn=_apply_columns,
            inputs=[filtered_results_state, page_state, page_size_dd, column_select],
            outputs=[results_table, page_indicator, page_state],
        )

        prev_btn.click(
            fn=lambda filtered_df, page, page_size, columns: _go_to_page(filtered_df, page, page_size, columns, -1),
            inputs=[filtered_results_state, page_state, page_size_dd, column_select],
            outputs=[results_table, page_indicator, page_state],
        )
        next_btn.click(
            fn=lambda filtered_df, page, page_size, columns: _go_to_page(filtered_df, page, page_size, columns, 1),
            inputs=[filtered_results_state, page_state, page_size_dd, column_select],
            outputs=[results_table, page_indicator, page_state],
        )

    return demo
