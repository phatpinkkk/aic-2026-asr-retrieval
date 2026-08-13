from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import importlib
import sys
import wave

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "production"))

from asr_retrieval.audio import (
    build_physical_windows,
    extract_window_wav,
    inspect_video_source,
)
from asr_retrieval.config import AudioWindowConfig, OfflineASRConfig, ParakeetConfig
from asr_retrieval.postprocess import (
    POSTPROCESS_VERSION,
    mark_consecutive_duplicate_windows,
    process_transcription,
)
from asr_retrieval.schemas import PhysicalWindowSpec, TranscriptSegment, TranscriptionResult
from asr_retrieval.transcriber import transcribe_video


def _write_pcm16_wav(path : Path, seconds : int, sample_rate : int = 16_000) -> bytes :
    sample_count = seconds * sample_rate
    frames = bytearray()

    for index in range(sample_count) :
        value = ((index % 1000) - 500) * 20
        frames.extend(int(value).to_bytes(2, byteorder = "little", signed = True))

    with wave.open(str(path), "wb") as wav :
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(bytes(frames))

    return bytes(frames)


def _result(
    window : PhysicalWindowSpec,
    text : str,
    segments : tuple[TranscriptSegment, ...] = (),
) -> TranscriptionResult :
    return TranscriptionResult(
        window = window,
        status = "ok",
        raw_text = text,
        native_segments = segments,
        window_pcm_sha256 = "a" * 64,
        canonical_wav_sha256 = "b" * 64,
        model_name = "test",
        model_revision = "rev",
        runtime_s = 0.1,
        peak_gpu_memory_bytes = 0,
        peak_reserved_memory_bytes = 0,
        warning_reasons = (),
        resolved_arguments = {},
        error = None,
    )


def test_window_policy_exact_samples() -> None :
    config = AudioWindowConfig()
    assert config.window_samples == 960_000
    assert config.stride_samples == 720_000
    assert config.minimum_tail_samples == 240_000


def test_window_tail_semantics() -> None :
    config = AudioWindowConfig()

    short = build_physical_windows("V", 10 * 16_000, config)
    assert [(item.sample_start, item.duration_samples) for item in short] == [
        (0, 160_000),
    ]

    under_tail = build_physical_windows("V", 104 * 16_000, config)
    assert [(item.start_s, item.end_s) for item in under_tail] == [
        (0.0, 60.0),
        (45.0, 104.0),
    ]

    exact_tail = build_physical_windows("V", 105 * 16_000, config)
    assert [(item.start_s, item.end_s) for item in exact_tail] == [
        (0.0, 60.0),
        (45.0, 105.0),
        (90.0, 105.0),
    ]
    assert [item.window_id for item in exact_tail] == [
        "V_0000",
        "V_0001",
        "V_0002",
    ]


def test_window_pcm_hash_matches_exact_source_slice(tmp_path : Path) -> None :
    source = tmp_path / "source.wav"
    pcm = _write_pcm16_wav(source, seconds = 70)
    config = AudioWindowConfig()
    window = build_physical_windows("V", 70 * 16_000, config)[1]
    output = tmp_path / "window.wav"

    observed = extract_window_wav(source, window, output)
    start = window.sample_start * 2
    end = window.sample_end * 2
    expected = sha256(pcm[start:end]).hexdigest()

    assert observed == expected


def test_postprocess_duplicate_segment_and_window() -> None :
    config = AudioWindowConfig()
    windows = build_physical_windows("V", 105 * 16_000, config)
    segments = (
        TranscriptSegment(0, 0.0, 1.0, "Đây là một đoạn văn bản đủ dài để kiểm tra lặp.", None),
        TranscriptSegment(1, 1.0, 2.0, "Đây là một đoạn văn bản đủ dài để kiểm tra lặp.", None),
    )
    first = process_transcription(_result(windows[0], "ignored", segments))
    second = process_transcription(_result(windows[1], first.retrieval_text))

    assert "consecutive_duplicate_segment" in first.warning_reasons
    marked = mark_consecutive_duplicate_windows((first, second))
    assert marked[0].eligible
    assert not marked[1].eligible
    assert marked[1].retrieval_text == ""
    assert "consecutive_duplicate_window" in marked[1].rejection_reasons


def test_dominant_boilerplate_is_ineligible() -> None :
    config = AudioWindowConfig()
    window = build_physical_windows("V", 60 * 16_000, config)[0]
    text = "hãy đăng ký kênh để không bỏ lỡ những video hấp dẫn"
    processed = process_transcription(_result(window, text))

    assert processed.postprocess_version == POSTPROCESS_VERSION
    assert not processed.eligible
    assert processed.retrieval_text == ""
    assert "dominant_known_boilerplate" in processed.rejection_reasons


def test_package_import_does_not_import_nemo() -> None :
    sys.modules.pop("nemo", None)
    module = importlib.import_module("asr_retrieval")
    assert module is not None
    assert "nemo" not in sys.modules


class _FakeTranscriber :
    def __init__(self, config : ParakeetConfig) :
        self.config = config
        self.loaded = True
        self.calls = 0

    def transcribe_window(
        self,
        window_wav : Path,
        window : PhysicalWindowSpec,
        window_pcm_sha256 : str,
        canonical_wav_sha256 : str,
    ) -> TranscriptionResult :
        self.calls += 1
        return TranscriptionResult(
            window = window,
            status = "ok",
            raw_text = f"nội dung cửa sổ {window.window_index}",
            native_segments = (),
            window_pcm_sha256 = window_pcm_sha256,
            canonical_wav_sha256 = canonical_wav_sha256,
            model_name = self.config.model_name,
            model_revision = self.config.revision,
            runtime_s = 0.01,
            peak_gpu_memory_bytes = 0,
            peak_reserved_memory_bytes = 0,
            warning_reasons = (),
            resolved_arguments = {
                "batch_size" : 1,
                "timestamps" : True,
                "decoder" : "greedy_ctc",
                "dtype" : "float32",
            },
            error = None,
        )


def test_transcribe_video_resume_reuses_compatible_windows(tmp_path : Path) -> None :
    source = tmp_path / "L21_V015.wav"
    _write_pcm16_wav(source, seconds = 70)
    workspace = tmp_path / "workspace"
    config = OfflineASRConfig(
        parakeet = ParakeetConfig(device = "cpu"),
    )

    first_runtime = _FakeTranscriber(config.parakeet)
    first = transcribe_video(
        source,
        workspace,
        config,
        first_runtime,
    )

    assert first.complete
    assert len(first.windows) == 2
    assert first_runtime.calls == 2
    assert all(item.eligible for item in first.windows)

    second_runtime = _FakeTranscriber(config.parakeet)
    second = transcribe_video(
        source,
        workspace,
        config,
        second_runtime,
    )

    assert second.complete
    assert second_runtime.calls == 0
    assert [item.raw_text for item in second.windows] == [
        item.raw_text
        for item in first.windows
    ]


def test_source_sha_is_part_of_video_identity(tmp_path : Path) -> None :
    source = tmp_path / "K01_V001.wav"
    _write_pcm16_wav(source, seconds = 1)
    metadata = inspect_video_source(source)

    assert metadata.video_id == "K01_V001"
    assert len(metadata.source_sha256) == 64
