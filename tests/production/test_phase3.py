from __future__ import annotations

from pathlib import Path
import math
import subprocess
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "production"))

from asr_retrieval.artifacts import LoadedRelease
from asr_retrieval.bm25 import build_bm25_index
from asr_retrieval.config import (
    BGEConfig,
    BM25Config,
    CandidateConfig,
    E5Config,
    FusionConfig,
    ProductionConfig,
    RuntimeConfig,
    bm25_identity,
    e5_identity,
)
from asr_retrieval.engine import ASRRetrievalEngine
from asr_retrieval.reranker import BGEReranker, RerankerOutput
from asr_retrieval.schemas import CorpusManifest, ReleaseVideoRecord, ReleaseWindowRecord


class _FakeE5 :
    def __init__(self) :
        self.loaded = True
        self.closed = False
        self.calls = 0

    def encode_query(self, query : str) -> np.ndarray :
        self.calls += 1
        vector = np.zeros(1024, dtype = np.float32)
        vector[0] = 1.0
        return vector

    def close(self) -> None :
        self.closed = True
        self.loaded = False


class _FakeReranker :
    def __init__(self, scores : list[float]) :
        self.loaded = True
        self.closed = False
        self.scores = np.asarray(scores, dtype = np.float32)
        self.calls : list[list[tuple[str, str]]] = []

    def score_pairs(self, pairs) -> RerankerOutput :
        values = [(str(query), str(document)) for query, document in pairs]
        self.calls.append(values)
        if (len(values) != len(self.scores)) :
            raise AssertionError(f"Expected {len(self.scores)} pairs, received {len(values)}")
        return RerankerOutput(
            scores = self.scores.copy(),
            runtime_s = 0.01,
            pair_count = len(values),
            effective_batch_size = 16,
            oom_retries = 0,
        )

    def close(self) -> None :
        self.closed = True
        self.loaded = False


def _unit_vector_with_cosine(score : float) -> np.ndarray :
    vector = np.zeros(1024, dtype = np.float32)
    vector[0] = np.float32(score)
    vector[1] = np.float32(math.sqrt(max(0.0, 1.0 - score * score)))
    vector /= np.linalg.norm(vector)
    return vector


def _fake_release() -> LoadedRelease :
    video_ids = ["A", "B", "C", "D", "E"]
    videos = []
    windows = []
    searchable_texts = []
    searchable_ids = []
    eligible_to_physical = []
    physical_to_eligible = []
    window_to_video = []
    video_window_offsets = [0]
    video_window_physical_ids = []
    cosine_scores = [0.95, 0.20, 0.75, 0.65, 0.55, 0.45, 0.30, 0.25]
    cosine_index = 0

    physical_index = 0
    for video_index, video_id in enumerate(video_ids) :
        first = physical_index
        count = 1 if video_id == "E" else 2
        videos.append(
            ReleaseVideoRecord(
                video_index = video_index,
                video_id = video_id,
                source_path = f"/{video_id}.mp4",
                source_size_bytes = 100,
                source_sha256 = str(video_index) * 64,
                audio_stream_index = 0,
                duration_samples = count * 960_000,
                sample_rate = 16_000,
                first_physical_window = first,
                physical_window_count = count,
            )
        )

        for local_index in range(count) :
            eligible = video_id != "E"
            window_id = f"{video_id}_{local_index:04d}"
            text = f"spoken content {video_id.lower()} {local_index}" if eligible else ""
            document_identity = f"doc-{window_id}" if eligible else None
            windows.append(
                ReleaseWindowRecord(
                    physical_index = physical_index,
                    window_id = window_id,
                    video_id = video_id,
                    video_index = video_index,
                    window_index = local_index,
                    sample_start = local_index * 720_000,
                    sample_end = local_index * 720_000 + 960_000,
                    duration_samples = 960_000,
                    sample_rate = 16_000,
                    status = "ok",
                    raw_text = text,
                    retrieval_text = text,
                    eligible = eligible,
                    warning_reasons = (),
                    rejection_reasons = () if eligible else ("empty_output",),
                    window_pcm_sha256 = "a" * 64,
                    postprocess_version = "1.1",
                    document_identity = document_identity,
                )
            )
            window_to_video.append(video_index)
            video_window_physical_ids.append(physical_index)
            if (eligible) :
                eligible_index = len(searchable_ids)
                eligible_to_physical.append(physical_index)
                physical_to_eligible.append(eligible_index)
                searchable_ids.append(window_id)
                searchable_texts.append(text)
                cosine_index += 1
            else :
                physical_to_eligible.append(-1)
            physical_index += 1

        video_window_offsets.append(len(video_window_physical_ids))

    embeddings = np.stack([
        _unit_vector_with_cosine(score)
        for score in cosine_scores
    ]).astype(np.float32)
    bm25 = build_bm25_index(searchable_ids, searchable_texts, BM25Config())
    manifest = CorpusManifest(
        schema_version = "1.0",
        release_id = "synthetic",
        created_at_utc = "2026-08-13T00:00:00+00:00",
        corpus_identity = "corpus",
        config_identity = "config",
        source_inventory_hash = "source",
        video_count = len(videos),
        physical_window_count = len(windows),
        eligible_window_count = len(searchable_ids),
        video_axis_sha256 = "video",
        physical_window_axis_sha256 = "physical",
        eligible_window_axis_sha256 = "eligible",
        window_policy_identity = "window",
        parakeet_identity = {},
        postprocess_version = "1.1",
        bm25_identity = bm25_identity(BM25Config()),
        e5_identity = e5_identity(E5Config(device = "cpu")),
        embedding_shape = embeddings.shape,
        embedding_dtype = "float32",
        file_hashes = {},
    )
    return LoadedRelease(
        path = Path("/synthetic"),
        manifest = manifest,
        videos = tuple(videos),
        windows = tuple(windows),
        eligible_window_ids = tuple(searchable_ids),
        eligible_to_physical = np.asarray(eligible_to_physical, dtype = np.int64),
        physical_to_eligible = np.asarray(physical_to_eligible, dtype = np.int64),
        window_to_video = np.asarray(window_to_video, dtype = np.int32),
        video_window_offsets = np.asarray(video_window_offsets, dtype = np.int64),
        video_window_physical_ids = np.asarray(video_window_physical_ids, dtype = np.int64),
        embeddings = embeddings,
        bm25 = bm25,
    )


def _engine(reranker_scores : list[float] | None = None) -> tuple[ASRRetrievalEngine, _FakeE5, _FakeReranker | None] :
    fake_e5 = _FakeE5()
    fake_reranker = _FakeReranker(reranker_scores) if reranker_scores is not None else None
    config = ProductionConfig(
        candidates = CandidateConfig(
            video_k = 3,
            windows_per_video = 2,
            max_candidate_pairs = 4,
        ),
        runtime = RuntimeConfig(
            e5_device = "cpu",
            load_reranker = fake_reranker is not None,
            serialize_gpu_requests = True,
            default_top_k = 5,
            default_windows_per_hit = 2,
        ),
        bge = BGEConfig(device = "cpu", dtype = "float32"),
    )
    engine = ASRRetrievalEngine(_fake_release(), config, fake_e5, fake_reranker)
    return engine, fake_e5, fake_reranker


def test_phase3_config_rejects_unsafe_candidate_budget() -> None :
    with pytest.raises(ValueError, match = "video_k cannot exceed") :
        CandidateConfig(video_k = 151, windows_per_video = 1, max_candidate_pairs = 150)

    with pytest.raises(ValueError, match = "must sum to 1") :
        FusionConfig(bm25_weight = 0.5, dense_weight = 0.6)


def test_minmax_constant_source_becomes_zero() -> None :
    values = ASRRetrievalEngine._minmax_normalize(
        np.asarray([4.2, 4.2, 4.2], dtype = np.float32),
        1e-12,
    )
    assert np.array_equal(values, np.zeros(3, dtype = np.float32))


def test_first_stage_search_ranks_videos_and_keeps_no_evidence_video() -> None :
    engine, _, fake_reranker = _engine(None)
    result = engine.search("semantic query", first_stage_only = True, top_k = 5)

    assert fake_reranker is None
    assert result.mode == "first_stage"
    assert [hit.video_id for hit in result.hits] == ["A", "B", "C", "D", "E"]
    assert result.hits[-1].first_stage_score == pytest.approx(0.0)
    assert result.hits[-1].windows == ()
    assert result.timings.candidate_pair_count == 0
    assert result.timings.reranker_batch_size is None


def test_candidate_allocation_is_breadth_first_and_capped() -> None :
    engine, _, fake_reranker = _engine([0.0, 1.0, 4.0, 0.0])
    result = engine.search("semantic query", top_k = 5)

    assert fake_reranker is not None
    assert len(fake_reranker.calls) == 1
    pair_documents = [document for _, document in fake_reranker.calls[0]]
    assert pair_documents == [
        "spoken content a 0",
        "spoken content b 0",
        "spoken content c 0",
        "spoken content a 1",
    ]
    assert result.timings.candidate_pair_count == 4
    assert result.timings.reranker_batch_size == 16


def test_reranked_candidate_prefix_and_non_candidate_order() -> None :
    engine, _, _ = _engine([0.0, 1.0, 4.0, 0.0])
    result = engine.search("semantic query", top_k = 5, windows_per_hit = 2)

    assert result.mode == "reranked"
    assert [hit.video_id for hit in result.hits] == ["C", "A", "B", "D", "E"]
    assert [hit.first_stage_rank for hit in result.hits] == [3, 1, 2, 4, 5]

    c_hit = result.hits[0]
    assert c_hit.reranked
    assert c_hit.final_score is not None
    assert c_hit.windows[0].window_id == "C_0000"
    assert c_hit.windows[0].reranked
    assert c_hit.windows[1].window_id == "C_0001"
    assert not c_hit.windows[1].reranked

    d_hit = result.hits[3]
    assert not d_hit.reranked
    assert d_hit.final_score is None
    assert d_hit.reranker_score is None


def test_first_stage_only_never_calls_loaded_reranker() -> None :
    engine, _, fake_reranker = _engine([0.0, 1.0, 4.0, 0.0])
    result = engine.search("semantic query", first_stage_only = True)
    assert result.mode == "first_stage"
    assert fake_reranker is not None
    assert fake_reranker.calls == []


def test_reranker_oom_fallback_retries_complete_pair_list() -> None :
    class _OOMThenSuccessModel :
        def __init__(self) :
            self.batches = []

        def predict(self, pairs, batch_size, show_progress_bar, convert_to_numpy) :
            self.batches.append((batch_size, list(pairs)))
            if (batch_size == 32) :
                raise RuntimeError("CUDA out of memory")
            return np.arange(len(pairs), dtype = np.float32)

    reranker = BGEReranker(BGEConfig())
    reranker._synchronize_cuda = lambda : None
    reranker._clear_cuda = lambda : None
    model = _OOMThenSuccessModel()
    reranker.model = model
    pairs = [("q", "a"), ("q", "b"), ("q", "c")]

    output = reranker.score_pairs(pairs)

    assert [batch for batch, _ in model.batches] == [32, 16]
    assert model.batches[0][1] == pairs
    assert model.batches[1][1] == pairs
    assert output.effective_batch_size == 16
    assert output.oom_retries == 1
    assert np.array_equal(output.scores, np.asarray([0.0, 1.0, 2.0], dtype = np.float32))


def test_package_import_does_not_import_heavy_model_libraries() -> None :
    code = f"""
import sys
sys.path.insert(0, {str(ROOT / 'production')!r})
import asr_retrieval
assert 'nemo' not in sys.modules
assert 'sentence_transformers' not in sys.modules
print('ok')
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check = True,
        capture_output = True,
        text = True,
    )
    assert completed.stdout.strip() == "ok"
