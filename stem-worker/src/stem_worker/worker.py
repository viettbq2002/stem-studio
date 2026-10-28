from __future__ import annotations

import json
import logging
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Protocol, TextIO

from stem_worker.separation_service import (
    InputValidationError,
    SeparationError,
    SeparationResult,
    SeparationService,
)

LOGGER = logging.getLogger(__name__)


class _SeparationService(Protocol):
    def separate(
        self,
        input_path: str | Path,
        output_dir: str | Path,
        progress_callback: Callable[[dict[str, object]], None] | None = None,
    ) -> SeparationResult: ...


def _write_message(output_stream: TextIO, message: dict[str, object]) -> None:
    output_stream.write(json.dumps(message, separators=(",", ":")) + "\n")
    output_stream.flush()


def run_worker(
    input_stream: TextIO = sys.stdin,
    output_stream: TextIO = sys.stdout,
    service_factory: Callable[[], _SeparationService] = SeparationService,
) -> None:
    service: _SeparationService | None = None
    LOGGER.info(
        "Worker started; waiting for JSONL requests. "
        "The Demucs model loads on the first separation request."
    )

    for line_number, line in enumerate(input_stream, start=1):
        if not line.strip():
            continue

        request_id: object = None
        try:
            request = json.loads(line)
            if not isinstance(request, dict):
                raise InputValidationError("Each request must be a JSON object.")
            request_id = request.get("id")
            if not isinstance(request_id, str) or not request_id.strip():
                raise InputValidationError("'id' must be a non-empty string.")
            request_type = request.get("type")

            if request_type == "health":
                LOGGER.info("Health check succeeded for request %s", request_id)
                _write_message(
                    output_stream,
                    {"id": request_id, "type": "ready", "status": "success"},
                )
                continue

            if request_type == "shutdown":
                LOGGER.info("Shutdown requested")
                _write_message(
                    output_stream,
                    {"id": request_id, "type": "stopped", "status": "success"},
                )
                return

            if request_type != "separate":
                raise InputValidationError(
                    "Request type must be 'health', 'shutdown', or 'separate'."
                )

            input_path = request.get("input_path")
            output_dir = request.get("output_dir")
            if not isinstance(input_path, str) or not input_path.strip():
                raise InputValidationError("'input_path' must be a non-empty string.")
            if not isinstance(output_dir, str) or not output_dir.strip():
                raise InputValidationError("'output_dir' must be a non-empty string.")

            if service is None:
                service = service_factory()

            started_at = time.monotonic()
            LOGGER.info(
                "Separation request %s started for input %s",
                request_id,
                input_path,
            )

            def report_progress(
                progress: dict[str, object], *, _request_id: object = request_id
            ) -> None:
                _write_message(
                    output_stream,
                    {
                        "id": _request_id,
                        "type": "progress",
                        "progress": dict(progress),
                    },
                )

            result = service.separate(
                input_path, output_dir, progress_callback=report_progress
            )
            LOGGER.info(
                "Separation request %s completed in %.1f seconds; generated %d stems",
                request_id,
                time.monotonic() - started_at,
                len(result.stems),
            )
            _write_message(
                output_stream,
                {
                    "id": request_id,
                    "type": "completed",
                    "status": "success",
                    "model": result.model,
                    "stems": [
                        {"name": stem.name, "path": str(stem.path)}
                        for stem in result.stems
                    ],
                },
            )
        except json.JSONDecodeError as exc:
            LOGGER.warning("Malformed JSON on worker input line %d: %s", line_number, exc)
            _write_message(
                output_stream,
                {
                    "id": request_id,
                    "type": "error",
                    "status": "error",
                    "error": {"code": "invalid_json", "message": "Malformed JSON."},
                },
            )
        except InputValidationError as exc:
            LOGGER.warning("Invalid worker request %r: %s", request_id, exc)
            _write_message(
                output_stream,
                {
                    "id": request_id,
                    "type": "error",
                    "status": "error",
                    "error": {"code": "invalid_request", "message": str(exc)},
                },
            )
        except SeparationError:
            LOGGER.exception("Separation request %r failed", request_id)
            _write_message(
                output_stream,
                {
                    "id": request_id,
                    "type": "error",
                    "status": "error",
                    "error": {
                        "code": "separation_failed",
                        "message": (
                            "Audio separation failed. Check the input file and "
                            "review worker logs for details."
                        ),
                    },
                },
            )
        except Exception:
            LOGGER.exception(
                "Unexpected error while handling worker request %r", request_id
            )
            _write_message(
                output_stream,
                {
                    "id": request_id,
                    "type": "error",
                    "status": "error",
                    "error": {
                        "code": "internal_error",
                        "message": "An unexpected worker error occurred.",
                    },
                },
            )
    LOGGER.info("Worker input closed; worker stopped")
