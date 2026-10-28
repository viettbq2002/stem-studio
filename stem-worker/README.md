# Stem Worker

Local Python worker for Stem Studio. It uses Demucs to split local audio into
stems and communicates over newline-delimited JSON (JSONL), ready for a future
Tauri integration. Audio is processed locally; the worker does not upload it.

## Requirements

- Python 3.14 or later (the project currently targets Python 3.14)
- [uv](https://docs.astral.sh/uv/)
- FFmpeg available on `PATH` for audio formats that Demucs cannot read with its
  native audio backend

## Setup

From the project directory:

```powershell
uv sync
```

Demucs downloads the selected model the first time a separation request is
processed. The default model is `htdemucs`, and inference defaults to CPU so
the worker does not require CUDA or an NVIDIA GPU.

## Run the worker

Start the persistent worker:

```powershell
uv run stem-worker
```

The worker reads one JSON object per input line and writes one or more JSON
objects per request to standard output. Standard output is reserved for
protocol messages; INFO logs and diagnostics are written to standard error.
Logs use Rich for readable timestamps, severity labels, and formatted
tracebacks in the terminal. At startup the worker logs that it is listening.
It also logs health checks, model loading, and separation start/completion with
elapsed time, so a quiet terminal is not mistaken for an idle or failed
process. The process reuses its loaded Demucs model for later requests.

Choose another model or inference device with command-line options:

```powershell
uv run stem-worker --model htdemucs_6s --device cpu
```

Send a readiness request:

```json
{"id":"health-1","type":"health"}
```

Response:

```json
{"id":"health-1","type":"ready","status":"success"}
```

Request separation:

```json
{"id":"request-001","type":"separate","input_path":"D:\\audio\\song.mp3","output_dir":"D:\\audio\\song-stems"}
```

The worker emits progress messages using Demucs callback information, without
inventing percentage estimates, followed by a completion response containing
the actual stem names and absolute output paths:

```json
{"id":"request-001","type":"progress","progress":{"state":"start","model_idx_in_bag":0}}
{"id":"request-001","type":"completed","status":"success","model":"htdemucs","stems":[{"name":"drums","path":"D:\\audio\\song-stems\\drums.wav"},{"name":"bass","path":"D:\\audio\\song-stems\\bass.wav"},{"name":"other","path":"D:\\audio\\song-stems\\other.wav"},{"name":"vocals","path":"D:\\audio\\song-stems\\vocals.wav"}]}
```

The model determines the stem list; clients should not assume that a specific
instrument stem exists. Invalid requests and processing failures return an
`error` message with a code and description, while the worker remains available
for the next request. Send `{"id":"stop-1","type":"shutdown"}` to stop it
gracefully.

Each generated stem is saved as a WAV file in the requested output directory.
The output path must not already exist; choose a unique directory for each
processing job. All stems are staged in a sibling temporary directory and
published together by renaming that directory only after every WAV has been
saved. A failed save or commit does not leave a partial result directory, and
an existing result is never overwritten.

## Use the service directly

```python
from stem_worker.separation_service import SeparationService

service = SeparationService(model="htdemucs", device="cpu")
result = service.separate("song.mp3", "output/song")
for stem in result.stems:
    print(stem.name, stem.path)
```

## Tests and lint

The tests mock Demucs inference and do not download a model or process real
audio:

```powershell
uv run pytest
uv run ruff check .
```

## Limitations

- CPU inference may take a long time, especially for longer audio.
- Demucs may need FFmpeg for some input formats.
- Standard `htdemucs` provides stems such as drums, bass, other, and vocals.
  Guitar may be part of `other`; `htdemucs_6s` can produce a guitar stem, but
  separation quality is not guaranteed.
- Audio mixing/export is not part of this initial worker milestone.