from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table

from agentic_rag.config import CONFIG_DIR, get_settings

if TYPE_CHECKING:
    from agentic_rag.models import HFModelDownloader

app = typer.Typer(no_args_is_help=True, help="Local multimodal agentic RAG.")
models_app = typer.Typer(no_args_is_help=True, help="Download and manage Hugging Face models.")
app.add_typer(models_app, name="models")
console = Console()


@app.callback()
def main(verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    for noisy in ("httpx", "httpcore", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _downloader() -> HFModelDownloader:
    from agentic_rag.models import HFModelDownloader

    settings = get_settings()
    return HFModelDownloader(settings.models_dir, token=os.environ.get("HF_TOKEN"))


@models_app.command("pull")
def models_pull(
    repo_id: Annotated[str, typer.Argument(help="HF repo id or a role from models.yaml")],
    revision: Annotated[str | None, typer.Option(help="Branch, tag or commit")] = None,
    include: Annotated[list[str] | None, typer.Option(help="Glob(s) to download")] = None,
    exclude: Annotated[list[str] | None, typer.Option(help="Glob(s) to skip")] = None,
    local_dir: Annotated[Path | None, typer.Option(help="Plain folder instead of cache")] = None,
    mlx: Annotated[bool, typer.Option(help="Convert to MLX after download")] = False,
    vlm: Annotated[bool, typer.Option(help="Use mlx_vlm for conversion")] = False,
    q_bits: Annotated[int, typer.Option(help="MLX quantization bits (0 = none)")] = 4,
) -> None:
    """Download any model from the Hugging Face Hub."""
    from agentic_rag.models import ModelRegistry

    spec = ModelRegistry.load(CONFIG_DIR / "models.yaml").get(repo_id)
    if spec:
        repo_id = spec.repo_id
        revision = revision or spec.revision
        include = include or spec.include
        exclude = exclude or spec.exclude
    downloader = _downloader()
    with console.status(f"Downloading {repo_id}..."):
        result = downloader.pull(repo_id, revision, include, exclude, local_dir)
    info = result.info
    console.print(f"[green]Downloaded[/] {repo_id} -> {result.path}")
    if info:
        console.print(f"license={info.license or 'unknown'} gated={info.gated} sha={info.sha}")
    if mlx:
        out = downloader.convert_to_mlx(repo_id, "vlm" if vlm else "llm", q_bits or None)
        console.print(f"[green]MLX model[/] -> {out}")


@models_app.command("pull-all")
def models_pull_all() -> None:
    """Download every enabled model from configs/models.yaml."""
    from agentic_rag.models import ModelRegistry

    downloader = _downloader()
    for spec in ModelRegistry.load(CONFIG_DIR / "models.yaml").enabled():
        with console.status(f"Downloading {spec.repo_id}..."):
            result = downloader.pull_spec(spec)
        console.print(f"[green]{spec.role}[/] {spec.repo_id} -> {result.path}")


@models_app.command("list")
def models_list() -> None:
    """Show the registry and locally cached models."""
    from agentic_rag.models import ModelRegistry

    local = {m.repo_id: m for m in _downloader().list_local()}
    table = Table("role", "kind", "repo_id", "enabled", "cached")
    registry = ModelRegistry.load(CONFIG_DIR / "models.yaml")
    for spec in registry.models:
        cached = local.get(spec.repo_id)
        size = f"{cached.size_bytes / 1e9:.2f} GB" if cached else "-"
        table.add_row(spec.role, spec.kind, spec.repo_id, str(spec.enabled), size)
    for repo_id, m in local.items():
        if not registry.get(repo_id):
            table.add_row("-", "-", repo_id, "-", f"{m.size_bytes / 1e9:.2f} GB")
    console.print(table)


@models_app.command("info")
def models_info(repo_id: str, revision: str | None = None) -> None:
    """Show license, gating and pipeline tag of a Hub model."""
    info = _downloader().inspect(repo_id, revision)
    console.print(info)


@app.command()
def ingest(
    path: Annotated[Path, typer.Argument(exists=True)],
    recreate: Annotated[bool, typer.Option(help="Drop and recreate the collection")] = False,
) -> None:
    """Parse, chunk, embed and index files (PDF, DOCX, PPTX, HTML, MD, TXT, images)."""
    from agentic_rag.services import Services

    report = Services(get_settings()).pipeline.ingest(path, recreate=recreate)
    console.print(
        f"[green]Ingested[/] {report.files} files: "
        f"{report.text_chunks} text chunks, {report.image_chunks} image chunks"
    )
    for source, error in report.failed.items():
        console.print(f"[red]Failed[/] {source}: {error}")
    if report.failed:
        raise typer.Exit(1)


@app.command()
def query(
    question: str,
    image: Annotated[Path | None, typer.Option(exists=True, help="Attach an image")] = None,
    thread: Annotated[str | None, typer.Option(help="Conversation thread id")] = None,
    source: Annotated[list[str] | None, typer.Option(help="Restrict to source(s)")] = None,
) -> None:
    """Ask a question; prints the cited answer."""
    from agentic_rag.services import Services

    result = Services(get_settings()).ask(
        question, str(image) if image else None, thread, source or None
    )
    console.print(Markdown(result.answer))
    for c in result.citations:
        page = f" p.{c.page}" if c.page else ""
        console.print(f"  [{c.index}] {c.source}{page} ({c.modality})", style="dim")
    console.print(
        f"route={result.route} grounded={result.grounded} thread={result.thread_id}", style="dim"
    )


@app.command("eval")
def eval_cmd(
    dataset: Annotated[Path | None, typer.Option(help="JSONL golden dataset")] = None,
    limit: Annotated[int | None, typer.Option(help="Evaluate only the first N samples")] = None,
    output: Annotated[Path | None, typer.Option(help="Results directory")] = None,
    gate: Annotated[bool, typer.Option(help="Exit 1 if below thresholds")] = True,
) -> None:
    """Run the agent on the golden dataset and score it with RAGAS."""
    from agentic_rag.evaluation import evaluate_from_settings, render_markdown, write_report

    settings = get_settings()
    report = evaluate_from_settings(settings, dataset, limit)
    write_report(report, settings.eval.thresholds, output or settings.eval.output_dir)
    console.print(Markdown(render_markdown(report, settings.eval.thresholds)))
    if gate and not report.passed:
        raise typer.Exit(1)


@app.command()
def serve(
    host: Annotated[str | None, typer.Option()] = None,
    port: Annotated[int | None, typer.Option()] = None,
) -> None:
    """Run the FastAPI server."""
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "agentic_rag.api.app:create_app",
        factory=True,
        host=host or settings.api.host,
        port=port or settings.api.port,
    )


@app.command()
def ui(port: int = 7860) -> None:
    """Run the Gradio chat UI."""
    from agentic_rag.ui.app import launch

    launch(port=port)


@app.command()
def status() -> None:
    """Check Qdrant, the LLM endpoint and Langfuse."""
    import httpx

    from agentic_rag.observability import langfuse_configured
    from agentic_rag.retrieval import QdrantStore

    settings = get_settings()
    store = QdrantStore.from_settings(settings.qdrant)
    try:
        console.print(f"qdrant: {settings.qdrant.collection} has {store.count()} points")
    except Exception as exc:
        console.print(f"[red]qdrant unreachable[/]: {exc}")
    url = settings.llm.resolved_base_url.rstrip("/") + "/models"
    try:
        httpx.get(url, timeout=5).raise_for_status()
        console.print(f"llm: {settings.llm.backend} {settings.llm.model} at {url}")
    except Exception as exc:
        console.print(f"[red]llm unreachable[/] at {url}: {exc}")
    console.print(
        f"langfuse: {'enabled' if langfuse_configured(settings.observability) else 'off'}"
    )
