"""Gradio web application for SITREP.

Tabs: Query, Ingest, Stats, Train, Lineage, Versioning. ``gradio`` is an optional
dependency (extra ``[web]``); this module imports without it, but :func:`create_app`
raises an informative error if Gradio is unavailable.
"""
from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path
from typing import Optional

_logger = logging.getLogger("sitrep.web")


def _require_gradio():
    """Import and return the ``gradio`` module or raise an informative error."""
    try:
        import gradio as gr  # type: ignore

        return gr
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "gradio is not installed. Install with: uv sync --extra web"
        ) from exc


def create_app(application=None):
    """Build and return the Gradio :class:`Blocks` app wired to *application*."""
    gr = _require_gradio()
    from src.application import Application, build_application

    app: Application = application or build_application()
    last_query = {"id": None}

    with gr.Blocks(title="SITREP", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# 🧠 SITREP\n"
            "Self-Improving Token-Reduced Embeddable Pipeline — local, privacy-first "
            "context engineering."
        )

        # ------------------------------------------------------------- Query
        with gr.Tab("Query"):
            q_in = gr.Textbox(label="Question", placeholder="Ask anything about ingested knowledge…")
            q_btn = gr.Button("Ask", variant="primary")
            q_out = gr.Markdown(label="Answer")
            q_meta = gr.Textbox(label="Confidence / token reduction", interactive=False)
            q_sources = gr.JSON(label="Sources")
            with gr.Row():
                up_btn = gr.Button("👍 Helpful")
                dn_btn = gr.Button("👎 Not helpful")
            fb_out = gr.Markdown()

            def _ask(question: str):
                if not question or not question.strip():
                    return "Please enter a question.", "", []
                dto = app.query_uc.execute(question)
                last_query["id"] = dto.query_id
                md = (
                    f"**Answer:** {dto.answer}\n\n"
                    f"_confidence={dto.confidence:.3f} · token_reduction={dto.token_reduction:.3f} "
                    f"· backend={dto.backend}_"
                )
                if dto.needs_clarification and dto.clarification_question:
                    md += f"\n\n⚠️ **Clarification:** {dto.clarification_question}"
                meta = (
                    f"confidence={dto.confidence:.3f}  "
                    f"token_reduction={dto.token_reduction:.3f}  "
                    f"compression_ratio={dto.compression_ratio:.3f}  "
                    f"tokens={dto.full_tokens}→{dto.compressed_tokens}"
                )
                sources = [
                    {
                        "passage_id": r.passage_id,
                        "score": round(float(r.final_score), 4),
                        "text": r.text[:140],
                    }
                    for r in dto.results
                ]
                return md, meta, sources

            q_btn.click(_ask, inputs=q_in, outputs=[q_out, q_meta, q_sources])

            def _feedback(polarity: str):
                qid = last_query["id"]
                if not qid:
                    return "Ask a question first, then rate it."
                rating = 1.0 if polarity == "positive" else 0.0
                dto = app.feedback_uc.submit(qid, polarity, rating)
                weights = [round(w, 4) for w in (dto.new_weights or [])]
                return f"Recorded **{polarity}**. weights_updated={dto.weights_updated} · fusion={weights}"

            up_btn.click(lambda: _feedback("positive"), outputs=fb_out)
            dn_btn.click(lambda: _feedback("negative"), outputs=fb_out)

        # ------------------------------------------------------------- Ingest
        with gr.Tab("Ingest"):
            gr.Markdown("Paste text or upload a `.txt`/`.md` file to ingest.")
            txt = gr.Textbox(label="Text", lines=8, placeholder="Paste content here…")
            btn = gr.Button("Ingest text", variant="primary")
            out = gr.JSON(label="Result")
            btn.click(lambda t: app.ingest_uc.execute(text=t).to_dict() if t else {}, inputs=txt, outputs=out)
            fu = gr.File(label="Upload .txt / .md", file_types=[".txt", ".md"])
            fu_out = gr.JSON(label="Upload result")

            def _ingest_file(file):
                if file is None:
                    return {}
                try:
                    data = Path(file.name).read_text(encoding="utf-8", errors="ignore")
                except Exception as exc:  # pragma: no cover
                    return {"error": str(exc)}
                return app.ingest_uc.execute(text=data).to_dict()

            fu.upload(_ingest_file, inputs=fu, outputs=fu_out)

        # ------------------------------------------------------------- Stats
        with gr.Tab("Stats"):
            btn = gr.Button("Refresh stats", variant="primary")
            out = gr.JSON(label="System statistics")
            btn.click(lambda: app.stats().to_dict(), outputs=out)

        # ------------------------------------------------------------- Train
        with gr.Tab("Train"):
            gr.Markdown("Train the RL compression agent on accumulated feedback.")
            ts = gr.Slider(0, 50000, value=1000, step=100, label="Total timesteps")
            btn = gr.Button("Train", variant="primary")
            out = gr.JSON(label="Training result")
            btn.click(lambda t: app.train_uc.execute(total_timesteps=int(t)).to_dict(), inputs=ts, outputs=out)

        # ------------------------------------------------------------- Lineage
        with gr.Tab("Lineage"):
            with gr.Row():
                did = gr.Textbox(label="Decision ID")
                trace_btn = gr.Button("Trace")
            out = gr.JSON(label="Decision trace")
            recent_btn = gr.Button("Show recent 20")
            recent_out = gr.JSON(label="Recent decisions")
            trace_btn.click(lambda i: app.lineage_uc.trace(i).to_dict() if i else {}, inputs=did, outputs=out)
            recent_btn.click(lambda: app.lineage_uc.recent(20), outputs=recent_out)

        # ------------------------------------------------------------- Versioning
        with gr.Tab("Versioning"):
            with gr.Row():
                label = gr.Textbox(label="Snapshot label")
                snap_btn = gr.Button("Snapshot", variant="primary")
            snap_out = gr.JSON(label="Snapshot")
            list_btn = gr.Button("List snapshots")
            list_out = gr.JSON(label="Snapshots")
            with gr.Row():
                name = gr.Textbox(label="Snapshot name (restore/delete)")
                rest_btn = gr.Button("Restore")
                del_btn = gr.Button("Delete")
            op_out = gr.JSON(label="Operation result")

            snap_btn.click(lambda l: asdict(app.version_uc.snapshot(l or None)), inputs=label, outputs=snap_out)
            list_btn.click(lambda: [asdict(s) for s in app.version_uc.list_snapshots()], outputs=list_out)
            rest_btn.click(lambda n: {"restored": app.version_uc.restore(n)} if n else {}, inputs=name, outputs=op_out)
            del_btn.click(lambda n: {"deleted": app.version_uc.delete(n)} if n else {}, inputs=name, outputs=op_out)

    return demo


def launch(application=None, **kwargs):
    """Build and launch the Gradio app (blocking)."""
    demo = create_app(application)
    _logger.info("launching SITREP web UI")
    demo.launch(**kwargs)


if __name__ == "__main__":  # pragma: no cover
    launch()
