"""Thin Gradio demo over the saved LoRA adapter. Reuses cesc.modeling inference."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cesc.env import prepare_runtime

prepare_runtime()

import gradio as gr

from cesc.labels import LABELS
from cesc.modeling import load_adapter, load_tokenizer, predict_rows
from cesc.paths import ROOT
from cesc.prose import assert_no_dash

ADAPTER_DEFAULT = ROOT / "adapters" / "lora"

MODEL_INFO = """
## Model

DeBERTa v3 Base + LoRA

Metrics below are from the held out test set.

Test Macro F1: 0.8751

Test Accuracy: 0.8778

Trainable parameters: approximately 888K
""".strip()

TITLE = "Claim Evidence Support Classifier"
SUBTITLE = "Enter a claim and an evidence snippet. The saved LoRA adapter scores the pair."

EXAMPLE_A_CLAIM = "At least 7 years of Angular experience."
EXAMPLE_A_EVIDENCE = "The frontend engineer used Angular for 7 years at Harbor Analytics."

EXAMPLE_B_CLAIM = "At least 4 years of PostgreSQL experience."
EXAMPLE_B_EVIDENCE = "Resume line from Aster Payroll: 4 years with MySQL as the data engineer."

EXAMPLE_C_CLAIM = "The candidate must have production experience with FastAPI."
EXAMPLE_C_EVIDENCE = (
    "Summit Transit used FastAPI for live traffic. "
    "The machine learning engineer handled failures and fixes for 8 years."
)

CUSTOM_CSS = """
body, .gradio-container {
  background: #ffffff !important;
  color: #111111 !important;
  font-family: Inter, Segoe UI, Helvetica, Arial, sans-serif !important;
}
.gradio-container {
  max-width: 880px !important;
  margin: 0 auto !important;
}
footer { display: none !important; }
.prose h2, .prose p, .prose {
  color: #111111 !important;
}
"""


def _format_probs(probs: list[float]) -> str:
    lines = ["## Class probabilities", ""]
    for index, label in enumerate(LABELS):
        lines.append(f"{label}: {probs[index]:.4f}")
    text = "\n".join(lines)
    assert_no_dash(text, "probability panel")
    return text


class DemoState:
    def __init__(self, adapter_dir: Path) -> None:
        adapter_dir = Path(adapter_dir).resolve()
        if not (adapter_dir / "train_meta.json").exists():
            raise SystemExit("No adapter metadata found. Train an adapter before running the demo.")
        print(f"Loading LoRA adapter from: {adapter_dir}", flush=True)
        self.model, self.meta, _label2id = load_adapter(adapter_dir)
        self.max_length = int(self.meta["max_length"])
        self.tokenizer = load_tokenizer(self.meta["model_name"], self.max_length)
        run_name = self.meta.get("run_name")
        print(f"Adapter run_name: {run_name}", flush=True)
        print(f"Trainable parameters recorded: {self.meta.get('trainable_parameters')}", flush=True)
        if run_name != "full":
            print("Warning: expected run_name full for the LinkedIn demo weights.", flush=True)

    def classify(self, claim: str, evidence: str) -> tuple[str, str]:
        claim = (claim or "").strip()
        evidence = (evidence or "").strip()
        if not claim or not evidence:
            return "Enter both a claim and evidence.", ""
        rows = [{"claim": claim, "evidence": evidence, "label": LABELS[0]}]
        pred_ids, probs = predict_rows(
            self.model, self.tokenizer, rows, self.max_length, batch_size=1
        )
        pred = LABELS[pred_ids[0]]
        assert_no_dash(pred, "prediction label")
        return pred, _format_probs(probs[0])


def build_ui(state: DemoState) -> gr.Blocks:
    assert_no_dash(TITLE, "title")
    assert_no_dash(SUBTITLE, "subtitle")
    assert_no_dash(MODEL_INFO, "model info")

    with gr.Blocks(title=TITLE) as demo:
        gr.Markdown(f"# {TITLE}\n\n{SUBTITLE}")
        with gr.Row():
            with gr.Column(scale=3):
                claim = gr.Textbox(
                    label="Claim",
                    lines=3,
                    placeholder="Example: Experience with a named tool or years of work",
                )
                evidence = gr.Textbox(
                    label="Evidence",
                    lines=4,
                    placeholder="Paste one evidence snippet",
                )
                with gr.Row():
                    classify_btn = gr.Button("Classify", variant="primary")
                    clear_btn = gr.Button("Clear")
                with gr.Row():
                    ex_a = gr.Button("Supported example")
                    ex_b = gr.Button("PostgreSQL vs MySQL miss")
                    ex_c = gr.Button("FastAPI supported example")
            with gr.Column(scale=2):
                prediction = gr.Textbox(label="Prediction", interactive=False)
                probabilities = gr.Markdown(value="## Class probabilities\n\nRun Classify to see scores.")
                gr.Markdown(MODEL_INFO)

        classify_btn.click(state.classify, inputs=[claim, evidence], outputs=[prediction, probabilities])
        clear_btn.click(
            lambda: ("", "", "", "## Class probabilities\n\nRun Classify to see scores."),
            inputs=None,
            outputs=[claim, evidence, prediction, probabilities],
        )
        ex_a.click(lambda: (EXAMPLE_A_CLAIM, EXAMPLE_A_EVIDENCE), outputs=[claim, evidence])
        ex_b.click(lambda: (EXAMPLE_B_CLAIM, EXAMPLE_B_EVIDENCE), outputs=[claim, evidence])
        ex_c.click(lambda: (EXAMPLE_C_CLAIM, EXAMPLE_C_EVIDENCE), outputs=[claim, evidence])

    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Local Gradio demo for the saved LoRA adapter.")
    parser.add_argument("--adapter", default=str(ADAPTER_DEFAULT), help="Folder with the saved LoRA adapter.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address.")
    parser.add_argument("--port", type=int, default=7860, help="Local port.")
    args = parser.parse_args()

    adapter_dir = Path(args.adapter)
    if not adapter_dir.is_absolute():
        adapter_dir = ROOT / adapter_dir

    state = DemoState(adapter_dir)
    demo = build_ui(state)

    theme = gr.themes.Soft(
        primary_hue="slate",
        secondary_hue="slate",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Inter"),
    ).set(
        body_background_fill="#ffffff",
        block_background_fill="#ffffff",
        border_color_primary="#e5e7eb",
        button_primary_background_fill="#111827",
        button_primary_text_color="#ffffff",
    )

    print(f"Starting Gradio on http://{args.host}:{args.port}", flush=True)
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        show_error=True,
        theme=theme,
        css=CUSTOM_CSS,
    )


if __name__ == "__main__":
    main()