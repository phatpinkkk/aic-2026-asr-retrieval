from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if (str(SRC_ROOT) not in sys.path) : sys.path.insert(0, str(SRC_ROOT))

from retrieval_v2 import ChannelDocuments, score_dense_embeddings


def test_dense_cosine_scoring_masks_ineligible_documents_below_cosine_floor() :
    documents = ChannelDocuments(model_id = "model", view = "raw", window_ids = ["w1", "w2", "w3"], texts = ["a", "", "c"], eligibility_mask = np.asarray([True, False, True]), zero_reasons = [None, "empty_text", None])
    query_embeddings    = np.asarray([[1.0, 0.0]], dtype = np.float32)
    document_embeddings = np.asarray([[1.0, 0.0], [-1.0, 0.0]], dtype = np.float32)
    bundle = score_dense_embeddings(query_ids = ["q1"], documents = documents, query_embeddings = query_embeddings, document_embeddings = document_embeddings, method_id = "D0", invalid_document_score = -2.0)
    assert np.allclose(bundle.scores, np.asarray([[1.0, -2.0, -1.0]], dtype = np.float32))
    assert bundle.metadata["score_clip"] is None
    assert bundle.metadata["invalid_document_policy"] == "finite_below_cosine_floor"


def test_dense_scoring_preserves_physical_window_axis() :
    documents = ChannelDocuments(model_id = "model", view = "processed", window_ids = ["w1", "w2", "w3", "w4"], texts = ["a", "b", "c", "d"], eligibility_mask = np.asarray([True, True, False, True]), zero_reasons = [None, None, "processed_rejection", None])
    query_embeddings    = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype = np.float32)
    document_embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]], dtype = np.float32)
    bundle = score_dense_embeddings(query_ids = ["q1", "q2"], documents = documents, query_embeddings = query_embeddings, document_embeddings = document_embeddings, method_id = "D1")
    assert bundle.scores.shape == (2, 4)
    assert np.all(bundle.scores[:, 2] == -2.0)
    assert bundle.window_ids == ["w1", "w2", "w3", "w4"]
