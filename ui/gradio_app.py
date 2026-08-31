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
.gradio-container {max-width: 1280px !important; margin: auto;}
#header-banner {
    background: linear-gradient(90deg, #0f2540 0%, #16345c 100%);
    color: white; padding: 18px 24px; border-radius: 10px; margin-bottom: 10px;
}
#header-banner h1 {margin: 0; font-size: 1.4rem;}
#header-banner p {margin: 4px 0 0 0; opacity: 0.85; font-size: 0.9rem;}
.metric-card {border: 1px solid #e2e8f0; border-radius: 10px; padding: 14px; text-align: center;}
footer {display: none !important;}
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
    items = sorted(analysis.category_counts.items(), key=lambda kv: kv[1], reverse=True)
    return pd.DataFrame(items, columns=["Category", "Count"])


def _summary_markdown(analysis) -> str:
    return (
        f"### Summary\n"
        f"| Metric | Value |\n|---|---|\n"
        f"| Total records seen | {analysis.total_records} |\n"
        f"| Valid records analyzed | {analysis.valid_records} |\n"
        f"| Rejected records | {analysis.rejected_records} |\n"
        f"| Average worklog score | {analysis.average_worklog_score} / 100 |\n"
    )


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
    chart_df = _category_chart_df(analysis)
    summary = _summary_markdown(analysis)

    # Prepare CSV for download
    csv_buf = io.StringIO()
    df.to_csv(csv_buf, index=False)
    csv_path = f"/tmp/itsm_quality_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    with open(csv_path, "w") as f:
        f.write(csv_buf.getvalue())

    return summary, df, gr.BarPlot(chart_df, x="Category", y="Count", title="Tickets by Category", y_lim=[0, None]), csv_path, df


def _filter_table(full_df, category, min_score):
    if full_df is None or len(full_df) == 0:
        return full_df
    filtered = full_df.copy()
    if category and category != "All":
        filtered = filtered[filtered["Category"] == category]
    filtered = filtered[filtered["Worklog Score"] >= min_score]
    return filtered


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

        with gr.Row():
            with gr.Column(scale=1):
                file_input = gr.File(label="Upload incident file (.xlsx, .csv, .txt)", file_types=[".xlsx", ".xls", ".csv", ".txt"])
                text_input = gr.Textbox(label="...or paste unstructured incident text", lines=8,
                                          placeholder="INC0012345\nShort description: ...\nWorklog: ...")
                analyze_btn = gr.Button("Analyze", variant="primary")
            with gr.Column(scale=1):
                summary_md = gr.Markdown("Run an analysis to see summary stats here.")
                category_chart = gr.BarPlot(label="Category Distribution")

        gr.Markdown("### Results")
        with gr.Row():
            category_filter = gr.Dropdown(choices=["All"] + [c for c in CATEGORIES], value="All", label="Filter by category")
            score_filter = gr.Slider(0, 100, value=0, step=5, label="Minimum worklog score")

        results_table = gr.Dataframe(label="Analyzed Tickets", interactive=False, wrap=True)
        full_results_state = gr.State(pd.DataFrame())
        download_file = gr.File(label="Download full results as CSV", interactive=False)

        analyze_btn.click(
            fn=_analyze,
            inputs=[file_input, text_input],
            outputs=[summary_md, results_table, category_chart, download_file, full_results_state],
        )
        category_filter.change(
            fn=_filter_table, inputs=[full_results_state, category_filter, score_filter], outputs=results_table
        )
        score_filter.change(
            fn=_filter_table, inputs=[full_results_state, category_filter, score_filter], outputs=results_table
        )

    return demo
