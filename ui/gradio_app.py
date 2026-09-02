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
#input-row {gap: 32px !important; margin-bottom: 8px;}
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


def _results_to_dataframe(analysis) -> pd.DataFrame:
    rows = []
    for r in analysis.results:
        rows.append(
            {
                "Ticket ID": r.ticket_id,
                "Category": r.category,
                "Category Confidence": r.category_confidence,
                "Category Method": r.category_method,
                "Short Description": r.short_description[:120],
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
    # Ascending here so the horizontal bar chart reads top-to-bottom as
    # highest-to-lowest count (BarPlot draws category-axis bottom-to-top).
    items = sorted(analysis.category_counts.items(), key=lambda kv: kv[1])
    return pd.DataFrame(items, columns=["Category", "Count"])


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
    chart_df = _category_chart_df(analysis)
    summary = _summary_markdown(analysis)

    # Prepare CSV for download - always the FULL result set, independent of
    # whatever filter/page the table happens to be showing.
    csv_buf = io.StringIO()
    df.to_csv(csv_buf, index=False)
    csv_path = f"/tmp/itsm_quality_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    with open(csv_path, "w") as f:
        f.write(csv_buf.getvalue())

    chart = gr.BarPlot(
        chart_df,
        x="Count",
        y="Category",
        title="Tickets by Category",
        x_lim=[0, None],
        x_axis_format="d",
        y_aggregate="sum",
        height=max(320, 34 * max(len(chart_df), 1)),
    )

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

        with gr.Row(elem_id="input-row"):
            with gr.Column(scale=1):
                file_input = gr.File(label="Upload incident file (.xlsx, .csv, .txt)", file_types=[".xlsx", ".xls", ".csv", ".txt"])
                text_input = gr.Textbox(label="...or paste unstructured incident text", lines=6,
                                          placeholder="INC0012345\nShort description: ...\nWorklog: ...")
                analyze_btn = gr.Button("Analyze", variant="primary")
            with gr.Column(scale=1):
                summary_md = gr.Markdown("Run an analysis to see summary stats here.")

        # Chart gets its own full-width row so long category names have
        # room to breathe instead of competing with the input column.
        with gr.Row(elem_id="chart-row"):
            category_chart = gr.BarPlot(label="Category Distribution", height=320)

        gr.Markdown("### Results", elem_classes=["section-block"])
        with gr.Row(elem_id="filters-row"):
            category_filter = gr.Dropdown(choices=["All"] + [c for c in CATEGORIES], value="All", label="Filter by category")
            score_filter = gr.Slider(0, 100, value=0, step=5, label="Minimum worklog score")
            page_size_dd = gr.Dropdown(choices=PAGE_SIZE_CHOICES, value=DEFAULT_PAGE_SIZE, label="Rows per page")

        results_table = gr.Dataframe(
            label="Analyzed Tickets",
            interactive=False,
            wrap=True,
            max_height=520,
            column_widths=[110, 190, 90, 110, 220, 90, 140, 220, 80, 90, 140, 200],
        )

        with gr.Row(elem_id="pagination-row"):
            prev_btn = gr.Button("← Previous", size="sm")
            page_indicator = gr.Markdown("Page 1 of 1  ·  0 tickets", elem_id="page-indicator")
            next_btn = gr.Button("Next →", size="sm")

        download_file = gr.File(label="Download full results as CSV", interactive=False)

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
