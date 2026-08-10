from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if (str(SRC_ROOT) not in sys.path) : sys.path.insert(0, str(SRC_ROOT))

from retrieval_backends import DenseBackendSpec, DenseEmbeddingBundle, SentenceTransformerDenseBackend


class FakeEncoder :
    def __init__(self, fail_once : bool = False) :
        self.fail_once  = fail_once
        self.prompts    = {"query" : "Instruct: test\nQuery: "}
        self.last_texts = None
        self.last_kwargs = None

    def encode(self, texts, **kwargs) :
        self.last_texts  = list(texts)
        self.last_kwargs = dict(kwargs)
        if (self.fail_once) :
            self.fail_once = False
            raise RuntimeError("CUDA out of memory")
        rows = []
        for index, _ in enumerate(texts) :
            vector = np.asarray([index + 1.0, 1.0], dtype = np.float32)
            vector /= np.linalg.norm(vector)
            rows.append(vector)
        return np.asarray(rows, dtype = np.float32)

    def get_sentence_embedding_dimension(self) : return 2


def test_dense_backend_spec_requires_pinned_revision() :
    with pytest.raises(ValueError) : DenseBackendSpec(backend_id = "x", model_name = "model", revision = "")


def test_dense_embedding_bundle_validates_shape_and_ids() :
    bundle = DenseEmbeddingBundle(ids = ["a", "b"], embeddings = np.eye(2, dtype = np.float32), metadata = {})
    assert bundle.dimension == 2
    with pytest.raises(ValueError) : DenseEmbeddingBundle(ids = ["a", "a"], embeddings = np.eye(2, dtype = np.float32), metadata = {})


def test_query_prompt_and_prefix_are_forwarded() :
    spec = DenseBackendSpec(backend_id = "qwen", model_name = "fake", revision = "a" * 40, query_prefix = "PREFIX ", query_prompt_name = "query", preferred_batch_size = 4)
    backend = SentenceTransformerDenseBackend(specification = spec, device = "cpu")
    backend.encoder = FakeEncoder()
    bundle = backend.encode_queries(["q1", "q2"], ["hello", "world"])
    assert bundle.ids == ["q1", "q2"]
    assert bundle.embeddings.shape == (2, 2)
    assert backend.encoder.last_texts == ["PREFIX hello", "PREFIX world"]
    assert backend.encoder.last_kwargs["prompt_name"] == "query"
    assert np.allclose(np.linalg.norm(bundle.embeddings, axis = 1), 1.0)


def test_oom_fallback_halves_batch_size() :
    spec = DenseBackendSpec(backend_id = "fake", model_name = "fake", revision = "b" * 40, preferred_batch_size = 8, minimum_batch_size = 2)
    backend = SentenceTransformerDenseBackend(specification = spec, device = "cpu")
    backend.encoder = FakeEncoder(fail_once = True)
    bundle = backend.encode_documents(["d1", "d2"], ["one", "two"])
    assert bundle.metadata["effective_batch_size"] == 4
