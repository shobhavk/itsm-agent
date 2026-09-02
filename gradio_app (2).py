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

#filters-row {margin-bottom: 14px; gap: 18px !important;}

/* Results table - larger, more legible rows instead of the cramped default. */
#results-table table th {
    background: #f8fafc !important; font-weight: 600 !important; font-size: 0.75rem !important;
    text-transform: uppercase; letter-spacing: 0.03em; color: var(--dash-text-muted) !important;
    padding: 12px 14px !important;
}
#results-table table td {
    padding: 14px 14px !important; font-size: 0.85rem !important; vertical-align: top;
    border-bottom: 1px solid #eef1f5 !important;
}
#results-table table tr:hover td {background: #f8fafc !important;}

#pagination-row {
    margin-top: 16px; display: flex; align-items: center; justify-content: center; gap: 16px;
}
#pagination-row button {border-radius: 8px !important;}
#page-indicator {text-align: center; font-size: 0.85rem; color: var(--dash-text-muted); padding-top: 8px;}

#chart-row {margin-top: 20px;}
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
        # NOTE: falls back gracefully if these attributes don't exist on your
        # AnalysisResult model yet - swap in the real field names if different
        # (e.g. r.full_description / r.worklog_text).
        description = getattr(r, "description", None) or r.short_description
        worklog_notes = getattr(r, "worklog_notes", None) or getattr(r, "worklog_text", None) or ""

        rows.append(
            {
                "Ticket ID": r.ticket_id,
                "Category": r.category,
                "Category Confidence": r.category_confidence,
                "Category Method": r.category_method,
                "Short Description": r.short_description[:120],
                "Description": description,
                "Worklog Score": r.worklog_score,
                "Worklog Rating": _score_badge(r.worklog_score),
                "Worklog Notes": worklog_notes,
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


def _kpi_card(label: str, value, accent: str) -> str:
    return (
        f'<div class="kpi-card" style="--accent:{accent};">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>'
        f"</div>"
    )


def _summary_html(analysis=None) -> str:
    if analysis is None:
        cards = "".join(
            [
                _kpi_card("Total records seen", "–", "#94a3b8"),
                _kpi_card("Valid records analyzed", "–", "#94a3b8"),
                _kpi_card("Rejected records", "–", "#94a3b8"),
                _kpi_card("Average worklog score", "–", "#94a3b8"),
            ]
        )
        return f'<div class="kpi-grid">{cards}</div>'

    cards = "".join(
        [
            _kpi_card("Total records seen", analysis.total_records, "#3b82f6"),
            _kpi_card("Valid records analyzed", analysis.valid_records, "#10b981"),
            _kpi_card("Rejected records", analysis.rejected_records, "#ef4444"),
            _kpi_card("Average worklog score", f"{analysis.average_worklog_score} / 100", "#8b5cf6"),
        ]
    )
    return f'<div class="kpi-grid">{cards}</div>'


def _analyze(file_obj, pasted_text):
    if file_obj is None and not (pasted_text and pasted_text.strip()):
        raise gr.Error("Upload a file (CSV/XLSX/TXT) or paste incident text first.")

    if file_obj is not None:
        with open(file_obj.name, "rb") as f:
            content = f.read()
        analysis = run_pipeline_from_bytes(file_obj.name, content)
    else:
        analysis = run_pipeline_from_text(pasted_text)

    df = _results_to_dataframe(analysis)
    summary = _summary_html(analysis)

    # Prepare CSV for download - always the FULL result set, independent of
    # whatever filter/page the table happens to be showing.
    csv_buf = io.StringIO()
    df.to_csv(csv_buf, index=False)
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
        gr.Markdown(f"_{SEVERITY_NOTE}_", elem_classes=["severity-note"])

        # KPI strip up top, dashboard-style, populated once the first
        # analysis completes.
        with gr.Row(elem_id="metrics-row"):
            summary_md = gr.HTML(_summary_html(None))

        with gr.Row(elem_id="input-row", equal_height=False):
            # Compact input column - just enough for the upload/paste/analyze
            # controls, so the results table gets the bulk of the width.
            with gr.Column(scale=1, min_width=280, elem_id="input-col", elem_classes=["dash-card"]):
                file_input = gr.File(label="Upload incident file (.xlsx, .csv, .txt)", file_types=[".xlsx", ".xls", ".csv", ".txt"])
                text_input = gr.Textbox(label="...or paste unstructured incident text", lines=3,
                                          placeholder="INC0012345\nShort description: ...\nWorklog: ...")
                analyze_btn = gr.Button("Analyze", variant="primary")
                download_file = gr.File(label="Download full results as CSV", interactive=False)

            # Categorized results sit to the right of the input column.
            with gr.Column(scale=3, elem_id="results-col", elem_classes=["dash-card"]):
                gr.Markdown("### Categorized results")
                with gr.Row(elem_id="filters-row"):
                    category_filter = gr.Dropdown(choices=["All"] + [c for c in CATEGORIES], value="All", label="Filter by category")
                    score_filter = gr.Slider(0, 100, value=0, step=5, label="Minimum worklog score")
                    page_size_dd = gr.Dropdown(choices=PAGE_SIZE_CHOICES, value=DEFAULT_PAGE_SIZE, label="Rows per page")

                results_table = gr.Dataframe(
                    label="Analyzed Tickets",
                    interactive=False,
                    wrap=True,
                    max_height=520,
                    elem_id="results-table",
                    column_widths=[100, 130, 90, 110, 200, 260, 90, 130, 260, 160, 80, 90, 140, 180],
                )

                with gr.Row(elem_id="pagination-row"):
                    prev_btn = gr.Button("← Previous", size="sm")
                    page_indicator = gr.Markdown("Page 1 of 1  ·  0 tickets", elem_id="page-indicator")
                    next_btn = gr.Button("Next →", size="sm")

        # Category donut chart gets its own card below the working area.
        with gr.Row(elem_id="chart-row"):
            with gr.Column(elem_classes=["dash-card"]):
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
