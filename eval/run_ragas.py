"""Run the RAGAS evaluation and fail (exit 1) when any metric is below its threshold.

Usage: uv run python eval/run_ragas.py [--dataset PATH] [--limit N]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from agentic_rag.config import get_settings
from agentic_rag.evaluation import evaluate_from_settings, render_markdown, write_report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    settings = get_settings()
    report = evaluate_from_settings(settings, args.dataset, args.limit)
    thresholds = settings.eval.thresholds
    write_report(report, thresholds, args.output or settings.eval.output_dir)
    print(render_markdown(report, thresholds))
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
