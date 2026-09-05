"""
Gradio dashboard for the ITSM Quality Analysis Agent.

Runs in-process (mounted into the FastAPI app in main.py) so it calls the
pipeline directly rather than round-tripping through HTTP - simpler, faster,
and avoids needing an API key inside the browser session. The REST API
(/api/v1/...) remains available separately for machine-to-machine/automation
use cases, secured with its own API key as usual.
"""
import io
from datetime import datetime

import gradio as gr
import pandas as pd
import plotly.graph_objects as go

from app.services.pipeline import run_pipeline_from_bytes, run_pipeline_from_text

CUSTOM_CSS = """
.gradio-container {max-width: 1500px !important; margin: auto; padding: 12px 24px !important;}
#header-banner {
    background: linear-gradient(90deg, #0f2540 0%, #16345c 100%);
    color: white; padding: 20px 28px; border-radius: 10px; margin-bottom: 18px;
}
#header-banner h1 {margin: 0; font-size: 1.4rem;}
#header-banner p {margin: 6px 0 0 0; opacity: 0.85; font-size: 0.9rem;}
.metric-card {border: 1px solid #e2e8f0; border-radius: 10px; padding: 14px; text-align: center;}
footer {display: none !important;}

.section-block {margin-top: 28px; margin-bottom: 8px;}
.section-block h3 {margin-bottom: 4px;}

#chart-row {margin-top: 20px; margin-bottom: 20px;}
#filters-row {margin-bottom: 12px; gap: 24px !important;}
#pagination-row {
    margin-top: 14px; display: flex; align-items: center; justify-content: center; gap: 16px;
}
#page-indicator {text-align: center; font-size: 0.9rem; padding-top: 8px;}
#input-row {gap: 24px !important; margin-bottom: 8px; align-items: flex-start;}
#input-col {max-width: 340px;}
#results-col h3 {margin-top: 0;}

/* Gradio 6's Dataframe cells render with white-space:nowrap regardless of
   the wrap=True Python param, which clips long Description/Worklog text
   mid-word instead of wrapping. Keep single-line + ellipsis (multi-line
   wrapping breaks this virtualized grid's row positioning), but keep rows
   compact - smaller padding/font than the default so more rows fit on
   screen at once. */
#results-table .cell-wrap {
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
    padding: 5px 8px !important;
    font-size: 0.85rem !important;
    line-height: 1.3 !important;
}
#results-table td, #results-table th {
    height: auto !important;
    vertical-align: middle !important;
    border-bottom: 1px solid #eef1f5 !important;
    padding: 2px 4px !important;
}
#results-table th .cell-wrap {
    font-size: 0.8rem !important;
    font-weight: 600 !important;
}
#results-table tbody tr:hover td {
    background: #f8fafc !important;
}
#results-table tbody tr:nth-child(even) td {
    background: #fbfcfe;
}
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


def _truncate(text: str, limit: int) -> str:
    text = text or ""
    return text[:limit] + "…" if len(text) > limit else text


def _results_to_dataframe(analysis) -> pd.DataFrame:
    rows = []
    for r in analysis.results:
        rows.append(
            {
                "Ticket ID": r.ticket_id,
                "Category": r.category,
                "Category Confidence": r.category_confidence,
                "Category Method": r.category_method,
                "Description": _truncate(r.description, 60),
                "Worklog Notes": _truncate(r.worklog, 60),
                "Worklog Score": r.worklog_score,
                "Worklog Rating": _score_badge(r.worklog_score),
                "Worklog Flags": "; ".join(r.worklog_flags) if r.worklog_flags else "",
                "Priority": r.priority or "",
                "Status": r.status or "",
                "Assignment Group": r.assignment_group or "",
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

    return summary, chart, csv_path, df


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
    return page_df, indicator, filtered, page


def _go_to_page(filtered_df, page, page_size, delta):
    page_df, indicator, new_page = _paginate(filtered_df, (page or 1) + delta, page_size)
    return page_df, indicator, new_page


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

        with gr.Row(elem_id="input-row", equal_height=False):
            # Compact input column - just enough for the upload/paste/analyze
            # controls, so the results table gets the bulk of the width.
            with gr.Column(scale=1, min_width=280, elem_id="input-col"):
                file_input = gr.File(label="Upload incident file (.xlsx, .csv, .txt)", file_types=[".xlsx", ".xls", ".csv", ".txt"])
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

                results_table = gr.Dataframe(
                    label="Analyzed Tickets",
                    interactive=False,
                    wrap=True,
                    max_height=520,
                    column_widths=[100, 170, 90, 100, 260, 260, 90, 130, 200, 70, 90, 140],
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
            outputs=[summary_md, category_chart, download_file, full_results_state],
        ).then(
            fn=_refresh_view,
            inputs=[full_results_state, category_filter, score_filter, page_size_dd],
            outputs=[results_table, page_indicator, filtered_results_state, page_state],
        )

        for control in (category_filter, score_filter, page_size_dd):
            control.change(
                fn=_refresh_view,
                inputs=[full_results_state, category_filter, score_filter, page_size_dd],
                outputs=[results_table, page_indicator, filtered_results_state, page_state],
            )

        prev_btn.click(
            fn=lambda filtered_df, page, page_size: _go_to_page(filtered_df, page, page_size, -1),
            inputs=[filtered_results_state, page_state, page_size_dd],
            outputs=[results_table, page_indicator, page_state],
        )
        next_btn.click(
            fn=lambda filtered_df, page, page_size: _go_to_page(filtered_df, page, page_size, 1),
            inputs=[filtered_results_state, page_state, page_size_dd],
            outputs=[results_table, page_indicator, page_state],
        )

    return demo
