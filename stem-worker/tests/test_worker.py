from __future__ import annotations

import io
import json
from collections.abc import Callable
from pathlib import Path

import pytest

from stem_worker.separation_service import SeparationError, SeparationResult, Stem
from stem_worker.worker import run_worker


class FakeService:
    def __init__(self) -> None:
        self.calls: list[tuple[str | Path, str | Path]] = []

    def separate(
        self,
        input_path: str | Path,
        output_dir: str | Path,
        progress_callback: Callable[[dict[str, object]], None] | None = None,
    ) -> SeparationResult:
        self.calls.append((input_path, output_dir))
        if callable(progress_callback):
            progress_callback({"state": "start", "segment_offset": 0})
        return SeparationResult(
            model="test-model",
            stems=(Stem(name="drums", path=Path(output_dir) / "drums.wav"),),
        )


def _messages(output: io.StringIO) -> list[dict[str, object]]:
    return [json.loads(line) for line in output.getvalue().splitlines()]


def test_worker_processes_multiple_requests_and_health() -> None:
    service = FakeService()
    constructed: list[FakeService] = []

    def make_service() -> FakeService:
        constructed.append(service)
        return service

    input_stream = io.StringIO(
        '{"id":"health-1","type":"health"}\n'
        '{"id":"job-1","type":"separate","input_path":"song.mp3",'
        '"output_dir":"out/song"}\n'
        '{"id":"job-2","type":"separate","input_path":"song2.mp3",'
        '"output_dir":"out/song2"}\n'
    )
    output_stream = io.StringIO()

    run_worker(input_stream, output_stream, make_service)

    messages = _messages(output_stream)
    assert messages[0] == {
        "id": "health-1",
        "type": "ready",
        "status": "success",
    }
    assert [message["type"] for message in messages[1:]] == [
        "progress",
        "completed",
        "progress",
        "completed",
    ]
    assert messages[1]["id"] == "job-1"
    assert messages[2]["stems"] == [
        {"name": "drums", "path": str(Path("out/song/drums.wav"))}
    ]
    assert len(service.calls) == 2
    assert constructed == [service]


def test_worker_logs_startup_health_and_request_completion(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level("INFO")
    service = FakeService()
    input_stream = io.StringIO(
        '{"id":"health-log","type":"health"}\n'
        '{"id":"job-log","type":"separate","input_path":"song.mp3",'
        '"output_dir":"out/song"}\n'
    )
    output_stream = io.StringIO()

    run_worker(input_stream, output_stream, lambda: service)

    assert "Worker started; waiting for JSONL requests" in caplog.text
    assert "Health check succeeded for request health-log" in caplog.text
    assert "Separation request job-log started" in caplog.text
    assert "Separation request job-log completed" in caplog.text
    assert all(
        isinstance(json.loads(line), dict)
        for line in output_stream.getvalue().splitlines()
    )


def test_worker_returns_structured_errors_and_continues() -> None:
    service = FakeService()
    input_stream = io.StringIO(
        "{broken json\n"
        '{"type":"health"}\n'
        '{"id":"bad-type","type":"unknown"}\n'
        '{"id":"bad-input","type":"separate","input_path":"","output_dir":"out"}\n'
        '{"id":"good","type":"health"}\n'
    )
    output_stream = io.StringIO()

    run_worker(input_stream, output_stream, lambda: service)

    messages = _messages(output_stream)
    assert [message["error"]["code"] for message in messages[:4]] == [
        "invalid_json",
        "invalid_request",
        "invalid_request",
        "invalid_request",
    ]
    assert messages[4]["type"] == "ready"


def test_shutdown_stops_processing_later_requests() -> None:
    service = FakeService()
    input_stream = io.StringIO(
        '{"id":"stop","type":"shutdown"}\n'
        '{"id":"later","type":"health"}\n'
    )
    output_stream = io.StringIO()

    run_worker(input_stream, output_stream, lambda: service)

    assert _messages(output_stream) == [
        {"id": "stop", "type": "stopped", "status": "success"}
    ]


def test_worker_logs_details_but_returns_safe_separation_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class FailingService:
        def separate(
            self,
            input_path: str | Path,
            output_dir: str | Path,
            progress_callback: Callable[[dict[str, object]], None] | None = None,
        ) -> SeparationResult:
            raise SeparationError("private local detail: model download failed")

    input_stream = io.StringIO(
        '{"id":"failed-job","type":"separate","input_path":"song.mp3",'
        '"output_dir":"out/song"}\n'
        '{"id":"still-alive","type":"health"}\n'
    )
    output_stream = io.StringIO()

    run_worker(input_stream, output_stream, FailingService)

    messages = _messages(output_stream)
    assert messages[0]["error"] == {
        "code": "separation_failed",
        "message": (
            "Audio separation failed. Check the input file and "
            "review worker logs for details."
        ),
    }
    assert "private local detail" not in output_stream.getvalue()
    assert "private local detail" in caplog.text
    assert messages[1]["type"] == "ready"
