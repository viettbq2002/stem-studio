from pathlib import Path

import pytest
import torch

from stem_worker.separation_service import (
    InputValidationError,
    SeparationError,
    SeparationService,
)


def test_separate_saves_each_returned_stem(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_file = tmp_path / "song.mp3"
    input_file.touch()
    output_dir = tmp_path / "output"
    separator_instances: list[object] = []

    class FakeSeparator:
        samplerate = 44_100

        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs
            self.callbacks: list[object] = []
            separator_instances.append(self)

        def update_parameter(self, callback: object = None) -> None:
            self.callbacks.append(callback)

        def separate_audio_file(
            self, path: Path
        ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
            assert path == input_file
            audio = torch.zeros((2, 16))
            return audio, {"drums": audio, "vocals": audio}

    def fake_save_audio(
        audio: torch.Tensor, path: Path, samplerate: int
    ) -> None:
        assert samplerate == 44_100
        assert audio.shape == (2, 16)
        path.write_bytes(b"wav")

    monkeypatch.setattr("demucs.api.Separator", FakeSeparator)
    monkeypatch.setattr("demucs.api.save_audio", fake_save_audio)
    service = SeparationService()

    result = service.separate(input_file, output_dir)

    assert [stem.name for stem in result.stems] == ["drums", "vocals"]
    assert isinstance(result.stems, tuple)
    assert all(stem.path.is_absolute() for stem in result.stems)
    assert [stem.path.name for stem in result.stems] == ["drums.wav", "vocals.wav"]
    assert all(stem.path.read_bytes() == b"wav" for stem in result.stems)
    assert len(separator_instances) == 1


def test_separator_is_reused_for_multiple_tracks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    files = [tmp_path / "one.mp3", tmp_path / "two.mp3"]
    for audio_file in files:
        audio_file.touch()
    constructor_calls = 0

    class FakeSeparator:
        samplerate = 44_100

        def __init__(self, **kwargs: object) -> None:
            nonlocal constructor_calls
            constructor_calls += 1

        def update_parameter(self, callback: object = None) -> None:
            pass

        def separate_audio_file(
            self, path: Path
        ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
            audio = torch.zeros((2, 16))
            return audio, {"other": audio}

    def fake_save_audio(audio: torch.Tensor, path: Path, samplerate: int) -> None:
        path.write_bytes(b"wav")

    monkeypatch.setattr("demucs.api.Separator", FakeSeparator)
    monkeypatch.setattr("demucs.api.save_audio", fake_save_audio)
    service = SeparationService()

    service.separate(files[0], tmp_path / "output-one")
    service.separate(files[1], tmp_path / "output-two")

    assert constructor_calls == 1


def test_missing_input_is_rejected(tmp_path: Path) -> None:
    service = SeparationService()

    with pytest.raises(InputValidationError, match="does not exist"):
        service.separate(tmp_path / "missing.mp3", tmp_path / "output")


def test_output_path_must_be_a_directory(tmp_path: Path) -> None:
    input_file = tmp_path / "song.mp3"
    input_file.touch()
    output_file = tmp_path / "output"
    output_file.touch()

    with pytest.raises(InputValidationError, match="already exists"):
        SeparationService().separate(input_file, output_file)


def test_existing_output_directory_is_not_overwritten(tmp_path: Path) -> None:
    input_file = tmp_path / "song.mp3"
    input_file.touch()
    output_dir = tmp_path / "existing"
    output_dir.mkdir()
    old_result = output_dir / "drums.wav"
    old_result.write_bytes(b"old")

    with pytest.raises(InputValidationError, match="unique path"):
        SeparationService().separate(input_file, output_dir)

    assert old_result.read_bytes() == b"old"


def test_output_parent_creation_failure_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_file = tmp_path / "song.mp3"
    input_file.touch()
    output_dir = tmp_path / "unwritable" / "output"
    original_mkdir = Path.mkdir

    def deny_parent(path: Path, *args: object, **kwargs: object) -> None:
        if path == output_dir.parent:
            raise PermissionError("read-only")
        original_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", deny_parent)

    with pytest.raises(InputValidationError, match="Could not create output parent"):
        SeparationService().separate(input_file, output_dir)


def test_demucs_failures_are_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_file = tmp_path / "song.mp3"
    input_file.touch()

    class FakeSeparator:
        def __init__(self, **kwargs: object) -> None:
            raise RuntimeError("model unavailable")

    monkeypatch.setattr("demucs.api.Separator", FakeSeparator)

    with pytest.raises(SeparationError, match="model unavailable") as error:
        SeparationService().separate(input_file, tmp_path / "output")
    assert isinstance(error.value.__cause__, RuntimeError)


def test_inference_failure_is_chained_and_not_published(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_file = tmp_path / "song.mp3"
    input_file.touch()

    class FakeSeparator:
        samplerate = 44_100

        def __init__(self, **kwargs: object) -> None:
            pass

        def update_parameter(self, callback: object = None) -> None:
            pass

        def separate_audio_file(
            self, path: Path
        ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
            raise RuntimeError("inference failed")

    monkeypatch.setattr("demucs.api.Separator", FakeSeparator)

    with pytest.raises(SeparationError, match="inference failed") as error:
        SeparationService().separate(input_file, tmp_path / "output")

    assert isinstance(error.value.__cause__, RuntimeError)
    assert not (tmp_path / "output").exists()


def test_progress_callback_is_registered_and_invoked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_file = tmp_path / "song.mp3"
    input_file.touch()
    received_events: list[dict[str, object]] = []

    class FakeSeparator:
        samplerate = 44_100

        def __init__(self, **kwargs: object) -> None:
            self.callback = None

        def update_parameter(self, callback: object = None) -> None:
            self.callback = callback

        def separate_audio_file(
            self, path: Path
        ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
            assert callable(self.callback)
            self.callback(
                {
                    "model_idx_in_bag": 0,
                    "shift_idx": 0,
                    "segment_offset": 0,
                    "state": "start",
                    "audio_length": 16,
                    "models": 1,
                }
            )
            audio = torch.zeros((2, 16))
            return audio, {"other": audio}

    def fake_save_audio(audio: torch.Tensor, path: Path, samplerate: int) -> None:
        path.write_bytes(b"wav")

    monkeypatch.setattr("demucs.api.Separator", FakeSeparator)
    monkeypatch.setattr("demucs.api.save_audio", fake_save_audio)
    service = SeparationService()

    service.separate(
        input_file,
        tmp_path / "output",
        progress_callback=lambda event: received_events.append(event),
    )

    assert received_events == [
        {
            "model_idx_in_bag": 0,
            "shift_idx": 0,
            "segment_offset": 0,
            "state": "start",
            "audio_length": 16,
            "models": 1,
        }
    ]


def test_empty_stem_results_fail_without_publishing_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_file = tmp_path / "song.mp3"
    input_file.touch()

    class FakeSeparator:
        samplerate = 44_100

        def __init__(self, **kwargs: object) -> None:
            pass

        def update_parameter(self, callback: object = None) -> None:
            pass

        def separate_audio_file(
            self, path: Path
        ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
            return torch.zeros((2, 16)), {}

    monkeypatch.setattr("demucs.api.Separator", FakeSeparator)

    with pytest.raises(SeparationError, match="no stems"):
        SeparationService().separate(input_file, tmp_path / "output")

    assert not (tmp_path / "output").exists()


def test_invalid_stem_name_is_rejected_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_file = tmp_path / "song.mp3"
    input_file.touch()

    class FakeSeparator:
        samplerate = 44_100

        def __init__(self, **kwargs: object) -> None:
            pass

        def update_parameter(self, callback: object = None) -> None:
            pass

        def separate_audio_file(
            self, path: Path
        ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
            audio = torch.zeros((2, 16))
            return audio, {"../escape": audio}

    monkeypatch.setattr("demucs.api.Separator", FakeSeparator)

    with pytest.raises(SeparationError, match="invalid stem name"):
        SeparationService().separate(input_file, tmp_path / "output")

    assert not (tmp_path / "output").exists()


def test_case_insensitive_stem_name_collisions_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_file = tmp_path / "song.mp3"
    input_file.touch()

    class FakeSeparator:
        samplerate = 44_100

        def __init__(self, **kwargs: object) -> None:
            pass

        def update_parameter(self, callback: object = None) -> None:
            pass

        def separate_audio_file(
            self, path: Path
        ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
            audio = torch.zeros((2, 16))
            return audio, {"guitar": audio, "Guitar": audio}

    monkeypatch.setattr("demucs.api.Separator", FakeSeparator)

    with pytest.raises(SeparationError, match="conflicting stem names"):
        SeparationService().separate(input_file, tmp_path / "output")

    assert not (tmp_path / "output").exists()


def test_failed_save_does_not_publish_partial_stems(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_file = tmp_path / "song.mp3"
    input_file.touch()

    class FakeSeparator:
        samplerate = 44_100

        def __init__(self, **kwargs: object) -> None:
            pass

        def update_parameter(self, callback: object = None) -> None:
            pass

        def separate_audio_file(
            self, path: Path
        ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
            audio = torch.zeros((2, 16))
            return audio, {"drums": audio, "vocals": audio}

    def fail_saving(audio: torch.Tensor, path: Path, samplerate: int) -> None:
        path.write_bytes(b"partial")
        if path.name == "vocals.wav":
            raise OSError("disk full")

    monkeypatch.setattr("demucs.api.Separator", FakeSeparator)
    monkeypatch.setattr("demucs.api.save_audio", fail_saving)

    with pytest.raises(SeparationError, match="disk full"):
        SeparationService().separate(input_file, tmp_path / "output")

    assert not (tmp_path / "output").exists()
    assert not list(tmp_path.glob(".stem-worker-*"))


def test_commit_failure_does_not_publish_output_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_file = tmp_path / "song.mp3"
    input_file.touch()

    class FakeSeparator:
        samplerate = 44_100

        def __init__(self, **kwargs: object) -> None:
            pass

        def update_parameter(self, callback: object = None) -> None:
            pass

        def separate_audio_file(
            self, path: Path
        ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
            audio = torch.zeros((2, 16))
            return audio, {"other": audio}

    def fake_save_audio(audio: torch.Tensor, path: Path, samplerate: int) -> None:
        path.write_bytes(b"wav")

    original_rename = Path.rename

    def fail_stage_commit(path: Path, target: Path) -> Path:
        if path.name.startswith(".stem-worker-"):
            raise OSError("commit failed")
        return original_rename(path, target)

    monkeypatch.setattr("demucs.api.Separator", FakeSeparator)
    monkeypatch.setattr("demucs.api.save_audio", fake_save_audio)
    monkeypatch.setattr(Path, "rename", fail_stage_commit)

    with pytest.raises(SeparationError, match="commit failed"):
        SeparationService().separate(input_file, tmp_path / "output")

    assert not (tmp_path / "output").exists()
    assert not list(tmp_path.glob(".stem-worker-*"))
