from __future__ import annotations

import argparse
import logging

from rich.console import Console
from rich.logging import RichHandler

from stem_worker.separation_service import HTDEMUCS_6S, SeparationService
from stem_worker.worker import run_worker


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            RichHandler(
                console=Console(stderr=True),
                show_time=True,
                show_level=True,
                show_path=False,
                rich_tracebacks=True,
                markup=False,
            )
        ],
        force=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Stem Studio JSONL worker.")
    parser.add_argument("--model", default=HTDEMUCS_6S, help="Demucs model name.")
    parser.add_argument("--device", default="cpu", help="Inference device (default: cpu).")
    args = parser.parse_args()

    configure_logging()
    run_worker(service_factory=lambda: SeparationService(model=args.model, device=args.device))


if __name__ == "__main__":
    main()
