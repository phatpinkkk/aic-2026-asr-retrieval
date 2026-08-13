from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from pathlib import Path
import json
import sys
import wave

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "production"))

from asr_retrieval.artifacts import (
    activate_release,
    build_release,
    load_release,
    resolve_active_release,
    validate_release,
)
from asr_retrieval.bm25 import build_bm25_index, load_bm25_index, save_bm25_index
from asr_retrieval.config import (
    ArtifactConfig,
    BM25Config,
    CorpusBuildConfig,
    E5Config,
    OfflineASRConfig,
    ParakeetConfig,
)
from asr_retrieval.dense import E5Encoder, exact_dense_scores
from asr_retrieval.schemas import PhysicalWindowSpec, TranscriptionResult
from asr_retrieval.transcriber import transcribe_video


def _write_pcm16_wav(path : Path, seconds : int, sample_rate : int = 16_000) -> None :
    frames = bytearray()
    for index in range(seconds * sample_rate) :
        value = ((index % 1000) - 500) * 20
        frames.extend(int(value).to_bytes(2, byteorder = "little", signed = True))

    with wave.open(str(path), "wb") as wav :
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(bytes(frames))


class _FakeTranscriber :
    def __init__(self, config : ParakeetConfig, text_prefix : str = "nội dung tiếng Việt") :
        self.config = config
        self.loaded = True
        self.text_prefix = text_prefix

    def transcribe_window(
        self,
        window_wav : Path,
        window : PhysicalWindowSpec,
        window_pcm_sha256 : str,
        canonical_wav_sha256 : str,
    ) -> TranscriptionResult :
        return TranscriptionResult(
            window = window,
            status = "ok",
            raw_text = f"{self.text_prefix} cửa sổ {window.window_index}",
            native_segments = (),
            window_pcm_sha256 = window_pcm_sha256,
            canonical_wav_sha256 = canonical_wav_sha256,
            model_name = self.config.model_name,
            model_revision = self.config.revision,
            runtime_s = 0.01,
            peak_gpu_memory_bytes = 0,
            peak_reserved_memory_bytes = 0,
            warning_reasons = (),
            resolved_arguments = {},
            error = None,
        )


class _FakeE5Encoder :
    def __init__(self, config : E5Config) :
        self.config = config
        self.loaded = True
        self.calls : list[list[str]] = []

    def encode_documents(self, document_ids, texts) -> np.ndarray :
        ids = [str(value) for value in document_ids]
        self.calls.append(ids)
        rows = []
        for document_id, text in zip(ids, texts) :
            digest = sha256((document_id + "\0" + str(text)).encode("utf-8")).digest()
            raw = np.frombuffer((digest * ((self.config.dimension * 4 // len(digest)) + 1))[ : self.config.dimension * 4], dtype = np.uint32).astype(np.float32)
            raw = (raw % 1009) + 1.0
            raw /= np.linalg.norm(raw)
            rows.append(raw.astype(np.float32))
        return np.stack(rows).astype(np.float32)


def _build_workspace(tmp_path : Path) -> tuple[Path, CorpusBuildConfig] :
    workspace = tmp_path / "workspace"
    offline = OfflineASRConfig(parakeet = ParakeetConfig(device = "cpu"))
    config = CorpusBuildConfig(
        offline = offline,
        e5 = E5Config(device = "cpu"),
    )

    source_a = tmp_path / "L21_V015.wav"
    source_b = tmp_path / "K01_V001.wav"
    _write_pcm16_wav(source_a, seconds = 70)
    _write_pcm16_wav(source_b, seconds = 50)

    transcribe_video(
        source_a,
        workspace,
        offline,
        _FakeTranscriber(offline.parakeet, "hổ quý hiếm ở đồng nai"),
    )
    transcribe_video(
        source_b,
        workspace,
        offline,
        _FakeTranscriber(offline.parakeet, "hãy đăng ký kênh để không bỏ lỡ những video hấp dẫn"),
    )
    return workspace, config


def test_bm25_persistence_and_unique_query_terms(tmp_path : Path) -> None :
    config = BM25Config()
    ids = ["w0", "w1", "w2"]
    texts = [
        "hổ quý hiếm ở đồng nai",
        "đàn hổ Bengal",
        "tàu vũ trụ cực quang",
    ]
    index = build_bm25_index(ids, texts, config)
    single = index.score("hổ")
    repeated = index.score("hổ hổ hổ")

    assert np.array_equal(single, repeated)
    assert single[0] > 0
    assert single[1] > 0
    assert single[2] == 0
    assert np.all(index.score("ho") == 0)

    path = tmp_path / "bm25"
    save_bm25_index(index, path)
    loaded = load_bm25_index(path, 3, config)
    assert np.allclose(loaded.score("hổ đồng nai"), index.score("hổ đồng nai"), atol = 0, rtol = 0)


def test_dense_explicit_normalization_and_exact_scores() -> None :
    raw = np.asarray([[3.0, 4.0], [0.0, 2.0]], dtype = np.float32)
    normalized = E5Encoder.normalize_embeddings(raw, "test")
    assert normalized.dtype == np.float32
    assert np.allclose(np.linalg.norm(normalized, axis = 1), 1.0, atol = 1e-6)

    query = np.asarray([0.6, 0.8], dtype = np.float32)
    scores = exact_dense_scores(query, normalized)
    assert scores.shape == (2,)
    assert np.isclose(scores[0], 1.0, atol = 1e-6)

    with pytest.raises(ValueError) :
        E5Encoder.normalize_embeddings(np.asarray([[0.0, 0.0]], dtype = np.float32), "zero")


def test_build_validate_load_immutable_release(tmp_path : Path) -> None :
    workspace, config = _build_workspace(tmp_path)
    releases = tmp_path / "releases"
    encoder = _FakeE5Encoder(config.e5)

    release = build_release(
        workspace = workspace,
        releases_root = releases,
        release_id = "release-001",
        config = config,
        encoder = encoder,
    )
    manifest = validate_release(release, verify_hashes = True)
    loaded = load_release(release)

    assert manifest.video_count == 2
    assert manifest.physical_window_count == 3
    assert manifest.eligible_window_count == 2
    assert loaded.embeddings.shape == (2, 1024)
    assert loaded.embeddings.dtype == np.float32
    assert np.allclose(np.linalg.norm(loaded.embeddings, axis = 1), 1.0, atol = 1e-5)
    assert loaded.eligible_window_ids == tuple(
        loaded.windows[int(index)].window_id
        for index in loaded.eligible_to_physical
    )
    assert np.array_equal(
        loaded.physical_to_eligible[loaded.eligible_to_physical],
        np.arange(manifest.eligible_window_count, dtype = np.int64),
    )
    assert len(encoder.calls) == 1
    assert len(encoder.calls[0]) == 2
    assert np.count_nonzero(loaded.physical_to_eligible == -1) == 1

    with pytest.raises(FileExistsError) :
        build_release(
            workspace = workspace,
            releases_root = releases,
            release_id = "release-001",
            config = config,
            encoder = encoder,
        )


def test_release_hash_corruption_is_rejected(tmp_path : Path) -> None :
    workspace, config = _build_workspace(tmp_path)
    release = build_release(
        workspace,
        tmp_path / "releases",
        "release-001",
        config,
        _FakeE5Encoder(config.e5),
    )
    with (release / "windows.jsonl").open("a", encoding = "utf-8") as file :
        file.write("\n")

    with pytest.raises(ValueError, match = "hash mismatch") :
        validate_release(release, verify_hashes = True)


def test_previous_release_reuses_all_e5_rows(tmp_path : Path) -> None :
    workspace, config = _build_workspace(tmp_path)
    releases = tmp_path / "releases"
    first_encoder = _FakeE5Encoder(config.e5)
    first = build_release(workspace, releases, "release-001", config, first_encoder)

    second = build_release(
        workspace = workspace,
        releases_root = releases,
        release_id = "release-002",
        config = config,
        encoder = None,
        previous_release = first,
    )

    first_loaded = load_release(first)
    second_loaded = load_release(second)
    assert np.array_equal(first_loaded.embeddings, second_loaded.embeddings)
    assert first_loaded.manifest.eligible_window_axis_sha256 == second_loaded.manifest.eligible_window_axis_sha256


def test_active_release_pointer(tmp_path : Path) -> None :
    workspace, config = _build_workspace(tmp_path)
    releases = tmp_path / "releases"
    release = build_release(
        workspace,
        releases,
        "release-001",
        config,
        _FakeE5Encoder(config.e5),
    )
    activate_release(releases, "release-001")
    assert resolve_active_release(releases) == release


def test_axis_mismatch_is_never_repaired(tmp_path : Path) -> None :
    workspace, config = _build_workspace(tmp_path)
    release = build_release(
        workspace,
        tmp_path / "releases",
        "release-001",
        config,
        _FakeE5Encoder(config.e5),
    )

    payload_path = release / "eligible_window_ids.json"
    payload = json.loads(payload_path.read_text(encoding = "utf-8"))
    payload["window_ids"] = list(reversed(payload["window_ids"]))
    payload_path.write_text(json.dumps(payload, ensure_ascii = False, indent = 2), encoding = "utf-8")

    # Update only the file hash to prove structural validation rejects axis mismatch,
    # rather than relying on the hash failure itself.
    manifest_path = release / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding = "utf-8"))
    manifest["file_hashes"]["eligible_window_ids.json"] = sha256(payload_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest, ensure_ascii = False, indent = 2), encoding = "utf-8")

    with pytest.raises(ValueError, match = "eligible_window_ids") :
        validate_release(release, verify_hashes = True)


def test_previous_release_reencodes_only_changed_document(tmp_path : Path) -> None :
    workspace, config = _build_workspace(tmp_path)
    releases = tmp_path / "releases"
    first = build_release(
        workspace,
        releases,
        "release-001",
        config,
        _FakeE5Encoder(config.e5),
    )

    transcript = workspace / "transcripts" / "L21_V015.json"
    payload = json.loads(transcript.read_text(encoding = "utf-8"))
    payload["windows"][0]["retrieval_text"] += " thay đổi"
    transcript.write_text(json.dumps(payload, ensure_ascii = False, indent = 2), encoding = "utf-8")

    second_encoder = _FakeE5Encoder(config.e5)
    second = build_release(
        workspace,
        releases,
        "release-002",
        config,
        second_encoder,
        previous_release = first,
    )

    assert len(second_encoder.calls) == 1
    assert second_encoder.calls[0] == ["L21_V015_0000"]
    first_loaded = load_release(first)
    second_loaded = load_release(second)
    assert not np.array_equal(first_loaded.embeddings[0], second_loaded.embeddings[0])
    assert np.array_equal(first_loaded.embeddings[1], second_loaded.embeddings[1])


def test_package_import_does_not_import_e5_runtime() -> None :
    import importlib
    sys.modules.pop("sentence_transformers", None)
    sys.modules.pop("torch", None)
    module = importlib.import_module("asr_retrieval")
    assert module is not None
    assert "sentence_transformers" not in sys.modules
    assert "torch" not in sys.modules
