# Relative path: src/retrieval_backends.py
# Purpose: Dense embedding backends for Retrieval v2 Stage 2.

from __future__ import annotations

import gc
import time
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np


DENSE_BACKENDS_VERSION = "1.1.0"
ProgressCallback = Callable[[str], None]


@dataclass(frozen = True)
class DenseBackendSpec :
    backend_id : str
    model_name : str
    revision : str
    query_prefix : str = ""
    document_prefix : str = ""
    query_prompt_name : str | None = None
    document_prompt_name : str | None = None
    normalize_embeddings : bool = True
    normalization_policy : str = "explicit_l2_float32_v1"
    preferred_batch_size : int = 16
    minimum_batch_size : int = 1

    def __post_init__(self) -> None :
        if (not self.backend_id.strip()) : raise ValueError("Dense backend_id must be nonempty")
        if (not self.model_name.strip()) : raise ValueError("Dense model_name must be nonempty")
        if (not self.revision.strip()) : raise ValueError("Dense revision must be pinned")
        if (self.normalize_embeddings and self.normalization_policy != "explicit_l2_float32_v1") : raise ValueError(f"Unsupported dense normalization policy: {self.normalization_policy!r}")
        if (self.preferred_batch_size <= 0) : raise ValueError("preferred_batch_size must be positive")
        if (self.minimum_batch_size <= 0 or self.minimum_batch_size > self.preferred_batch_size) : raise ValueError("minimum_batch_size must be within [1, preferred_batch_size]")

    @classmethod
    def from_config(cls, backend_id : str, config : dict[str, Any]) -> "DenseBackendSpec" :
        return cls(backend_id = backend_id, model_name = str(config["model_name"]), revision = str(config["revision"]), query_prefix = str(config.get("query_prefix", "")), document_prefix = str(config.get("document_prefix", "")), query_prompt_name = config.get("query_prompt_name"), document_prompt_name = config.get("document_prompt_name"), normalize_embeddings = bool(config.get("normalize_embeddings", True)), normalization_policy = str(config.get("normalization_policy", "explicit_l2_float32_v1")), preferred_batch_size = int(config.get("preferred_batch_size", 16)), minimum_batch_size = int(config.get("minimum_batch_size", 1)))

    def identity(self) -> dict[str, Any] :
        return {"backend_id" : self.backend_id, "model_name" : self.model_name, "revision" : self.revision, "query_prefix" : self.query_prefix, "document_prefix" : self.document_prefix, "query_prompt_name" : self.query_prompt_name, "document_prompt_name" : self.document_prompt_name, "normalize_embeddings" : self.normalize_embeddings, "normalization_policy" : self.normalization_policy}


@dataclass
class DenseEmbeddingBundle :
    ids : list[str]
    embeddings : np.ndarray
    metadata : dict[str, Any]

    def __post_init__(self) -> None :
        self.ids        = [str(value) for value in self.ids]
        self.embeddings = np.asarray(self.embeddings, dtype = np.float32)
        if (len(set(self.ids)) != len(self.ids)) : raise ValueError("DenseEmbeddingBundle contains duplicate IDs")
        if (self.embeddings.ndim != 2) : raise ValueError("Dense embeddings must be two-dimensional")
        if (self.embeddings.shape[0] != len(self.ids)) : raise ValueError("Dense embedding row count does not match IDs")
        if (not np.isfinite(self.embeddings).all()) : raise ValueError("Dense embeddings contain NaN or infinite values")

    @property
    def dimension(self) -> int :
        return int(self.embeddings.shape[1]) if self.embeddings.ndim == 2 else 0


class SentenceTransformerDenseBackend :
    def __init__(self, specification : DenseBackendSpec, device : str | None = None, progress_callback : ProgressCallback | None = None) :
        self.specification     = specification
        self.device            = device or self._default_device()
        self.progress_callback = progress_callback
        self.encoder           = None
        self.resolved_revision = None
        self.load_runtime_s    = 0.0
        self.last_batch_size   = None

    @staticmethod
    def _default_device() -> str :
        try :
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception :
            return "cpu"

    def _emit(self, message : str) -> None :
        if (self.progress_callback is not None) : self.progress_callback(message)

    def _resolve_revision(self) -> str :
        if (self.resolved_revision) : return self.resolved_revision
        revision = self.specification.revision.strip()
        if (len(revision) == 40 and all(char in "0123456789abcdefABCDEF" for char in revision)) :
            self.resolved_revision = revision
            return revision
        try :
            from huggingface_hub import HfApi
            info = HfApi().model_info(self.specification.model_name, revision = self.specification.revision)
            self.resolved_revision = str(info.sha or self.specification.revision)
        except Exception :
            self.resolved_revision = self.specification.revision
        return self.resolved_revision

    def _load(self) -> None :
        if (self.encoder is not None) : return
        from sentence_transformers import SentenceTransformer
        self._emit(f"{self.specification.backend_id}: loading {self.specification.model_name}@{self.specification.revision} on {self.device}")
        started      = time.perf_counter()
        self.encoder = SentenceTransformer(self.specification.model_name, revision = self.specification.revision, device = self.device)
        self.load_runtime_s = time.perf_counter() - started
        self._resolve_revision()
        if (self.specification.query_prompt_name is not None and self.specification.query_prompt_name not in getattr(self.encoder, "prompts", {})) : raise ValueError(f"{self.specification.backend_id}: query prompt {self.specification.query_prompt_name!r} is not available in the model")
        if (self.specification.document_prompt_name is not None and self.specification.document_prompt_name not in getattr(self.encoder, "prompts", {})) : raise ValueError(f"{self.specification.backend_id}: document prompt {self.specification.document_prompt_name!r} is not available in the model")
        self._emit(f"{self.specification.backend_id}: model ready in {self.load_runtime_s:.1f}s; resolved revision {self.resolved_revision}")

    @staticmethod
    def _is_oom(error : RuntimeError) -> bool :
        message = str(error).lower()
        return "out of memory" in message or "cuda error: out of memory" in message

    @staticmethod
    def _clear_cuda() -> None :
        try :
            import torch
            if (torch.cuda.is_available()) : torch.cuda.empty_cache()
        except Exception :
            pass

    @staticmethod
    def _explicit_l2_normalize(embeddings : np.ndarray, backend_id : str, label : str) -> np.ndarray :
        values = np.asarray(embeddings, dtype = np.float32)
        if (not np.isfinite(values).all()) : raise ValueError(f"{backend_id}: non-finite {label} embeddings")
        norms = np.linalg.norm(values, axis = 1, keepdims = True)
        if (np.any(~np.isfinite(norms)) or np.any(norms <= 1e-12)) : raise ValueError(f"{backend_id}: zero or non-finite {label} embedding norm")
        values = np.asarray(values / norms, dtype = np.float32)
        normalized_norms = np.linalg.norm(values, axis = 1)
        if (not np.allclose(normalized_norms, 1.0, atol = 1e-5, rtol = 1e-5)) : raise ValueError(f"{backend_id}: explicit L2 normalization failed for {label}")
        return values

    def _prepare_texts(self, texts : Sequence[str], prefix : str) -> list[str] :
        return [prefix + str(text or "") for text in texts]

    def _encode(self, texts : Sequence[str], prefix : str, prompt_name : str | None, label : str, batch_size : int | None = None, show_progress_bar : bool = True, emit_progress : bool = True) -> DenseEmbeddingBundle :
        values = self._prepare_texts(texts, prefix)
        if (not values) : raise ValueError(f"{self.specification.backend_id}: cannot encode an empty {label} collection")
        self._load()
        effective_batch_size = int(batch_size or self.specification.preferred_batch_size)
        if (effective_batch_size < self.specification.minimum_batch_size) : raise ValueError(f"{self.specification.backend_id}: batch size is below minimum_batch_size")
        while True :
            try :
                if (emit_progress) : self._emit(f"{self.specification.backend_id}: encoding {len(values)} {label} with batch={effective_batch_size}")
                started = time.perf_counter()
                kwargs = {"batch_size" : effective_batch_size, "show_progress_bar" : show_progress_bar, "convert_to_numpy" : True, "normalize_embeddings" : False}
                if (prompt_name is not None) : kwargs["prompt_name"] = prompt_name
                embeddings = np.asarray(self.encoder.encode(values, **kwargs), dtype = np.float32)
                if (self.specification.normalize_embeddings) : embeddings = self._explicit_l2_normalize(embeddings, self.specification.backend_id, label)
                elif (not np.isfinite(embeddings).all()) : raise ValueError(f"{self.specification.backend_id}: non-finite {label} embeddings")
                runtime_s = time.perf_counter() - started
                self.last_batch_size = effective_batch_size
                return DenseEmbeddingBundle(ids = [str(index) for index in range(len(values))], embeddings = embeddings, metadata = {"runtime_s" : runtime_s, "item_count" : len(values), "items_per_second" : float(len(values) / runtime_s) if runtime_s > 0 else None, "effective_batch_size" : effective_batch_size, "dimension" : int(embeddings.shape[1]), "normalization_policy" : self.specification.normalization_policy if self.specification.normalize_embeddings else "none"})
            except RuntimeError as error :
                if (not self._is_oom(error) or effective_batch_size <= self.specification.minimum_batch_size) : raise
                next_batch = max(self.specification.minimum_batch_size, effective_batch_size // 2)
                if (emit_progress) : self._emit(f"{self.specification.backend_id}: OOM at batch={effective_batch_size}; retrying with batch={next_batch}")
                effective_batch_size = next_batch
                self._clear_cuda()

    def encode_queries(self, query_ids : Sequence[str], texts : Sequence[str], batch_size : int | None = None, show_progress_bar : bool = True, emit_progress : bool = True) -> DenseEmbeddingBundle :
        if (len(query_ids) != len(texts)) : raise ValueError("Query IDs and texts have different lengths")
        encoded = self._encode(texts, self.specification.query_prefix, self.specification.query_prompt_name, "queries", batch_size = batch_size, show_progress_bar = show_progress_bar, emit_progress = emit_progress)
        return DenseEmbeddingBundle(ids = [str(value) for value in query_ids], embeddings = encoded.embeddings, metadata = encoded.metadata)

    def encode_documents(self, document_ids : Sequence[str], texts : Sequence[str], batch_size : int | None = None, show_progress_bar : bool = True, emit_progress : bool = True) -> DenseEmbeddingBundle :
        if (len(document_ids) != len(texts)) : raise ValueError("Document IDs and texts have different lengths")
        encoded = self._encode(texts, self.specification.document_prefix, self.specification.document_prompt_name, "documents", batch_size = batch_size, show_progress_bar = show_progress_bar, emit_progress = emit_progress)
        return DenseEmbeddingBundle(ids = [str(value) for value in document_ids], embeddings = encoded.embeddings, metadata = encoded.metadata)

    def metadata(self) -> dict[str, Any] :
        dimension = None
        if (self.encoder is not None) :
            try :
                getter    = getattr(self.encoder, "get_embedding_dimension", None) or getattr(self.encoder, "get_sentence_embedding_dimension")
                dimension = int(getter())
            except Exception :
                dimension = None
        return {**self.specification.identity(), "resolved_revision" : self._resolve_revision(), "device" : self.device, "embedding_dimension" : dimension, "load_runtime_s" : self.load_runtime_s, "last_batch_size" : self.last_batch_size}

    def release(self) -> None :
        self.encoder = None
        gc.collect()
        self._clear_cuda()
