"""Gradio review tool: play a clip, correct the transcript, flag bad audio, export corrections.

uv run python -m fieldasr.review.app --queue data/review/queue.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gradio as gr

from fieldasr.review.store import ReviewStore


def build(store: ReviewStore, export_path: Path) -> gr.Blocks:
    def show(i: int | None):
        if i is None:
            return None, "", False, "", None, "All items reviewed - use Export."
        r = store.queue.iloc[i]
        cond = r.get("condition", "")
        d, n = store.progress()
        return (
            str(r.path),
            str(r.hyp),
            False,
            "",
            i,
            f"Item {i + 1}/{n} - condition: {cond} - reviewed {d}/{n}",
        )

    def save_next(i, text, bad, note, reviewer):
        if i is not None:
            store.save(str(store.queue.iloc[i].item_id), text, bad, reviewer, note)
        return show(store.next_pending(i if i is not None else -1))

    def skip(i):
        return show(store.next_pending(i if i is not None else -1))

    def export():
        return str(store.export(export_path)) + f"  ->  {export_path}"

    with gr.Blocks(title="FieldASR review") as demo:
        gr.Markdown(
            "## FieldASR transcript review\nListen, fix the draft transcript, flag unusable audio. "
            "Do not type personal details (names, phone numbers, IDs) - exports are auto-redacted."
        )
        idx = gr.State(None)
        status = gr.Markdown()
        audio = gr.Audio(label="Clip", type="filepath", interactive=False, autoplay=False)
        text = gr.Textbox(label="Transcript (edit the model draft)", lines=3, interactive=True)
        with gr.Row():
            bad = gr.Checkbox(label="Bad audio (unusable / not speech / wrong language)")
            reviewer = gr.Textbox(label="Reviewer id (optional)", scale=1)
        note = gr.Textbox(label="Note (optional)")
        with gr.Row():
            b_save = gr.Button("Save & next", variant="primary")
            b_skip = gr.Button("Skip")
            b_exp = gr.Button("Export corrected data")
        out = gr.Textbox(label="Export result", interactive=False)
        outs = [audio, text, bad, note, idx, status]
        b_save.click(save_next, [idx, text, bad, note, reviewer], outs)
        b_skip.click(skip, [idx], outs)
        b_exp.click(export, None, out)
        demo.load(lambda: show(store.next_pending(-1)), None, outs)
    return demo


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", default="data/review/queue.csv")
    ap.add_argument("--log", default="data/review/corrections.jsonl")
    ap.add_argument("--export", default="data/review/corrected.csv")
    ap.add_argument("--port", type=int, default=7860)
    a = ap.parse_args()
    build(ReviewStore(Path(a.queue), Path(a.log)), Path(a.export)).launch(
        server_name="127.0.0.1", server_port=a.port, allowed_paths=[str(Path.cwd())]
    )


if __name__ == "__main__":
    main()
