from __future__ import annotations

import logging
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from torch import Tensor


LOGGER = logging.getLogger(__name__)

DEFAULT_MODEL = "htdemucs"

HTDEMUCS = "htdemucs"

HTDEMUCS_6S = "htdemucs_6s"

HTDEMUCS_MMI = "htdemucs_mmi"

DemucsCallback = Callable[[dict[str, object]], None]


class _DemucsSeparator(Protocol):
    samplerate: int

    def update_parameter(
        self,
        device: str = ...,
        shifts: int = ...,
        overlap: float = ...,
        split: bool = ...,
        segment: int | None = ...,
        jobs: int = ...,
        progress: bool = ...,
        callback: DemucsCallback | None = ...,
        callback_arg: dict[str, object] | None = ...,
    ) -> None: ...

    def separate_audio_file(
        self, file: Path
    ) -> tuple[Tensor, dict[str, Tensor]]: ...


class InputValidationError(ValueError):
    """Raised when an input audio path or output directory is invalid."""


class SeparationError(RuntimeError):
    """Raised when Demucs cannot initialize or process an audio file."""


@dataclass(frozen=True)
class Stem:
    name: str
    path: Path


@dataclass(frozen=True)
class SeparationResult:
    model: str
    stems: tuple[Stem, ...]


ProgressCallback = DemucsCallback


class SeparationService:
    def __init__(self, model: str = HTDEMUCS_6S, device: str = "cpu") -> None:
        if not model.strip():
            raise ValueError("Model name cannot be empty.")
        if not device.strip():
            raise ValueError("Device cannot be empty.")
        self.model_name = model
        self.device = device
        self._separator: _DemucsSeparator | None = None

    def _get_separator(self) -> _DemucsSeparator:
        if self._separator is None:
            try:
                from demucs.api import LoadModelError, Separator

                LOGGER.info(
                    "Loading Demucs model %s on device %s", self.model_name, self.device
                )
                started_at = time.monotonic()
                self._separator = Separator(model=self.model_name, device=self.device)
                LOGGER.info(
                    "Demucs model %s loaded in %.1f seconds",
                    self.model_name,
                    time.monotonic() - started_at,
                )
            except (LoadModelError, OSError, RuntimeError) as exc:
                LOGGER.exception(
                    "Could not initialize Demucs model %s on device %s",
                    self.model_name,
                    self.device,
                )
                raise SeparationError(
                    f"Could not initialize Demucs model '{self.model_name}' "
                    f"on device '{self.device}': {exc}"
                ) from exc
        return self._separator

    def separate(
        self,
        input_path: str | Path,
        output_dir: str | Path,
        progress_callback: ProgressCallback | None = None,
    ) -> SeparationResult:
        audio_path = Path(input_path).expanduser()
        if not audio_path.exists():
            raise InputValidationError(f"Input audio file does not exist: {audio_path}")
        if not audio_path.is_file():
            raise InputValidationError(f"Input audio path is not a file: {audio_path}")

        destination = Path(output_dir).expanduser()
        if destination.exists() or destination.is_symlink():
            raise InputValidationError(
                f"Output directory already exists; choose a unique path: {destination}"
            )
        destination = destination.resolve()
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise InputValidationError(
                f"Could not create output parent directory '{destination.parent}': {exc}"
            ) from exc

        from demucs.api import LoadAudioError, save_audio

        try:
            stage_context = tempfile.TemporaryDirectory(
                prefix=".stem-worker-", dir=destination.parent
            )
        except OSError as exc:
            raise InputValidationError(
                f"Output parent directory is not writable: {destination.parent}: {exc}"
            ) from exc

        try:
            with stage_context as temp:
                staging_dir = Path(temp)
                separator = self._get_separator()
                separator.update_parameter(callback=progress_callback)
                _, separated = separator.separate_audio_file(audio_path)
                if not separated:
                    raise SeparationError("Demucs returned no stems.")

                staged_stems: list[Stem] = []
                seen_names: set[str] = set()
                for name, audio in separated.items():
                    if (
                        not name
                        or Path(name).name != name
                        or name in {".", ".."}
                        or "/" in name
                        or "\\" in name
                    ):
                        raise SeparationError(
                            f"Demucs returned an invalid stem name: {name!r}"
                        )
                    if name.casefold() in seen_names:
                        raise SeparationError(
                            f"Demucs returned conflicting stem names: {name!r}"
                        )
                    seen_names.add(name.casefold())
                    staged_path = staging_dir / f"{name}.wav"
                    save_audio(audio, staged_path, samplerate=separator.samplerate)
                    staged_stems.append(
                        Stem(name=name, path=destination / staged_path.name)
                    )

                if destination.exists() or destination.is_symlink():
                    raise InputValidationError(
                        "Output directory appeared during processing; "
                        "choose a unique path."
                    )
                try:
                    staging_dir.rename(destination)
                except FileExistsError as exc:
                    raise InputValidationError(
                        "Output directory appeared during processing; "
                        "choose a unique path."
                    ) from exc

            return SeparationResult(model=self.model_name, stems=tuple(staged_stems))
        except InputValidationError:
            raise
        except SeparationError:
            raise
        except (LoadAudioError, OSError, RuntimeError) as exc:
            raise SeparationError(f"Audio separation failed: {exc}") from exc
