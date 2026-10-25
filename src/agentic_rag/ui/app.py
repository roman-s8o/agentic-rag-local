from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import gradio as gr

from agentic_rag.config import get_settings
from agentic_rag.services import Services

NODE_STATUS = {
    "analyze": "Analyzing the question...",
    "retrieve": "Searching documents (dense + sparse + image)...",
    "rerank": "Reranking results...",
    "grade": "Grading relevance...",
    "rewrite_query": "Evidence insufficient, rewriting the query...",
    "check": "Verifying the answer is grounded...",
}


def _format_final(event: dict[str, Any]) -> str:
    lines = [event["answer"]]
    if event["citations"]:
        lines.append("\n**Sources**")
        for c in event["citations"]:
            page = f", p.{c['page']}" if c.get("page") else ""
            kind = " (image)" if c["modality"] == "image" else ""
            lines.append(f"- [{c['index']}] `{c['source']}`{page}{kind}")
    if event.get("grounded") is False:
        lines.append("\n_Warning: parts of this answer could not be verified against sources._")
    return "\n".join(lines)


def build_ui(services: Services) -> gr.Blocks:
    def respond(message: dict[str, Any], history: list, request: gr.Request) -> Iterator[str]:
        files = message.get("files") or []
        image = files[0] if files else None
        if isinstance(image, dict):
            image = image.get("path")
        question = (message.get("text") or "").strip() or "Describe this image."
        thread_id = getattr(request, "session_hash", None)
        answer = ""
        for event in services.stream(question, image=image, thread_id=thread_id):
            if event["type"] == "token":
                answer += event["text"]
                yield answer
            elif event["type"] == "node" and event["node"] in NODE_STATUS and not answer:
                yield f"_{NODE_STATUS[event['node']]}_"
            elif event["type"] == "final":
                yield _format_final(event)

    with gr.Blocks(title="Agentic RAG") as demo:
        gr.Markdown(
            "# Agentic RAG\nAsk questions about your indexed documents. "
            "Attach an image to search with it."
        )
        gr.ChatInterface(
            respond,
            multimodal=True,
            textbox=gr.MultimodalTextbox(file_types=["image"], file_count="single"),
            examples=[
                {"text": "What is the peak efficiency of the Helios H5?"},
                {"text": "According to the chart, what is the efficiency at 10% load?"},
                {"text": "Compare the H5 warranty with the expense report deadline."},
            ],
        )
    return demo


def launch(port: int = 7860) -> None:
    build_ui(Services(get_settings())).queue().launch(server_port=port)
