# Relative path: src/retrieval_backends.py
# Purpose: Dense embedding backends for Stage 2 and local reranker backends for Stage 7.

from __future__ import annotations

import gc
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np


DENSE_BACKENDS_VERSION      = "1.1.0"
RERANKER_BACKENDS_VERSION   = "1.1.0"
ProgressCallback            = Callable[[str], None]


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

# -----------------------------------------------------------------------------
# Stage 7 pairwise reranker backends
# -----------------------------------------------------------------------------


@dataclass(frozen = True)
class RerankerBackendSpec :
    backend_id          : str
    model_name          : str
    revision            : str
    family              : str = "encoder"
    max_length          : int = 512
    dtype               : str = "float16"
    trust_remote_code   : bool = False
    instruction         : str | None = None
    prompt_name         : str | None = None
    model_kwargs        : dict[str, Any] = field(default_factory = dict)

    def __post_init__(self) -> None :
        if (not self.backend_id.strip()) : raise ValueError("Reranker backend_id must be nonempty")
        if (not self.model_name.strip()) : raise ValueError("Reranker model_name must be nonempty")
        if (len(self.revision.strip()) != 40) : raise ValueError(f"{self.backend_id}: reranker revision must be an exact 40-character commit SHA")
        if (self.family not in {"encoder", "causal_logit"}) : raise ValueError(f"{self.backend_id}: unsupported reranker family {self.family!r}")
        if (self.max_length <= 0) : raise ValueError("Reranker max_length must be positive")
        if (self.dtype not in {"float16", "float32", "bfloat16"}) : raise ValueError(f"Unsupported reranker dtype: {self.dtype!r}")
        if ((self.instruction is None) != (self.prompt_name is None)) :
            raise ValueError(f"{self.backend_id}: instruction and prompt_name must either both be set or both be null")

    @classmethod
    def from_config(
        cls,
        backend_id : str,
        config : dict[str, Any],
        max_length : int,
        dtype : str,
    ) -> "RerankerBackendSpec" :
        return cls(
            backend_id = backend_id,
            model_name = str(config["model_name"]),
            revision = str(config["revision"]),
            family = str(config.get("family", "encoder")),
            max_length = int(max_length),
            dtype = str(dtype),
            trust_remote_code = bool(config.get("trust_remote_code", False)),
            instruction = config.get("instruction"),
            prompt_name = config.get("prompt_name"),
            model_kwargs = dict(config.get("model_kwargs", {})),
        )

    def identity(self) -> dict[str, Any] :
        return {
            "backend_id"        : self.backend_id,
            "model_name"        : self.model_name,
            "revision"          : self.revision,
            "family"            : self.family,
            "max_length"        : self.max_length,
            "dtype"             : self.dtype,
            "trust_remote_code" : self.trust_remote_code,
            "instruction"       : self.instruction,
            "prompt_name"       : self.prompt_name,
            "model_kwargs"      : dict(self.model_kwargs),
        }


@dataclass
class RerankerScoreOutput :
    scores : np.ndarray
    runtime_s : float
    pair_count : int
    effective_batch_size : int

    def __post_init__(self) -> None :
        self.scores = np.asarray(self.scores, dtype = np.float32).reshape(-1)
        if (self.scores.shape != (int(self.pair_count),)) : raise ValueError("Reranker score count mismatch")
        if (not np.isfinite(self.scores).all()) : raise ValueError("Reranker produced NaN or infinite scores")
        if (self.runtime_s < 0) : raise ValueError("Reranker runtime must be nonnegative")
        if (self.effective_batch_size <= 0) : raise ValueError("Reranker batch size must be positive")

    @property
    def pairs_per_second(self) -> float | None :
        return float(self.pair_count / self.runtime_s) if self.runtime_s > 0 else None


class SentenceTransformerCrossEncoderReranker :
    def __init__(
        self,
        specification : RerankerBackendSpec,
        device : str | None = None,
        progress_callback : ProgressCallback | None = None,
    ) :
        self.specification     = specification
        self.device            = device or SentenceTransformerDenseBackend._default_device()
        self.progress_callback = progress_callback
        self.model             = None
        self.resolved_revision = None
        self.load_runtime_s    = 0.0
        self.last_batch_size   = None
        self.oom_count         = 0

    def _emit(self, message : str) -> None :
        if (self.progress_callback is not None) : self.progress_callback(message)

    @staticmethod
    def _is_oom(error : RuntimeError) -> bool :
        return SentenceTransformerDenseBackend._is_oom(error)
    
    @staticmethod
    def _is_fatal_cuda(error : BaseException) -> bool :
        message = str(error).lower()

        markers = (
            "device-side assert",
            "index out of bounds",
            "illegal memory access",
            "unspecified launch failure",
            "cudaerrorassert",
        )

        return any(marker in message for marker in markers)

    @staticmethod
    def _clear_cuda() -> None :
        SentenceTransformerDenseBackend._clear_cuda()

    @staticmethod
    def _synchronize_cuda() -> None :
        try :
            import torch
        except Exception :
            return

        if (torch.cuda.is_available()) :
            torch.cuda.synchronize()

    def _torch_dtype(self) :
        import torch
        if (self.device != "cuda") : return torch.float32
        if (self.specification.dtype == "float16") : return torch.float16
        if (self.specification.dtype == "bfloat16") : return torch.bfloat16
        return torch.float32

    def _load(self) -> None :
        if (self.model is not None) : return
        from sentence_transformers import CrossEncoder

        import torch
        
        model_kwargs = {
            "torch_dtype" : self._torch_dtype(),
            **self.specification.model_kwargs,
        }
        
        kwargs : dict[str, Any] = {
            "revision"          : self.specification.revision,
            "device"            : self.device,
            "max_length"        : self.specification.max_length,
            "trust_remote_code" : self.specification.trust_remote_code,
            "model_kwargs"      : model_kwargs,
            "activation_fn"     : torch.nn.Identity(),
        }
        if (self.specification.instruction is not None) :
            kwargs["prompts"] = {self.specification.prompt_name : self.specification.instruction}
            kwargs["default_prompt_name"] = self.specification.prompt_name

        self._emit(
            f"{self.specification.backend_id}: loading "
            f"{self.specification.model_name}@{self.specification.revision} on {self.device}"
        )
        started = time.perf_counter()
        self.model = CrossEncoder(self.specification.model_name, **kwargs)
        self.load_runtime_s = time.perf_counter() - started
        self.resolved_revision = self._resolve_revision()

        if (self.resolved_revision != self.specification.revision) :
            raise RuntimeError(
                f"{self.specification.backend_id}: resolved revision "
                f"{self.resolved_revision!r} != pinned {self.specification.revision!r}"
            )

        self._emit(
            f"{self.specification.backend_id}: model ready in {self.load_runtime_s:.1f}s; "
            f"resolved revision {self.resolved_revision}"
        )

    def _resolve_revision(self) -> str :
        if (self.resolved_revision is not None) : return self.resolved_revision
        try :
            from huggingface_hub import HfApi
            info = HfApi().model_info(
                self.specification.model_name,
                revision = self.specification.revision,
            )
            self.resolved_revision = str(info.sha or self.specification.revision)
        except Exception :
            self.resolved_revision = self.specification.revision
        return self.resolved_revision

    def tokenizer(self) :
        self._load()
        tokenizer = getattr(self.model, "tokenizer", None)
        if (tokenizer is None) : raise RuntimeError(f"{self.specification.backend_id}: CrossEncoder tokenizer is unavailable")
        return tokenizer


    def smoke_test(self) -> dict[str, Any] :
        pairs = [
            (
                "Which planet is known as the Red Planet?",
                "Mars is known as the Red Planet.",
            )
        ]

        output = self.score_pairs(pairs, batch_size = 1, show_progress_bar = False)

        if (output.scores.shape != (1,)) :
            raise RuntimeError(f"{self.specification.backend_id}: smoke test returned unexpected score shape {output.scores.shape}")

        score = float(output.scores[0])

        if (not np.isfinite(score)) :
            raise RuntimeError(f"{self.specification.backend_id}: smoke test produced a non-finite score")

        return {
            "passed" : True,
            "score"  : score,
        }

    def token_length_audit(
        self,
        pairs : Sequence[tuple[str, str]],
        batch_size : int = 64,
    ) -> dict[str, Any] :
        values = [(str(query), str(document)) for query, document in pairs]
        if (not values) :
            return {
                "pair_count" : 0,
                "total_input_tokens" : 0,
                "mean_input_tokens" : None,
                "p50_input_tokens" : None,
                "p90_input_tokens" : None,
                "p95_input_tokens" : None,
                "max_input_tokens" : None,
                "max_length" : int(self.specification.max_length),
                "truncated_count" : 0,
                "truncated_fraction" : 0.0,
                "audit_policy" : "cross_encoder_preprocess_untruncated_v1",
            }

        self._load()
        lengths = []

        for start in range(0, len(values), max(1, int(batch_size))) :
            batch = values[start : start + max(1, int(batch_size))]
            encoded = None

            # Prefer the CrossEncoder preprocessing path so prompt/chat-template
            # formatting matches inference. Support both current modular and
            # legacy processing_kwargs shapes before falling back to tokenizer.
            for processing_kwargs in [
                {"text" : {"truncation" : False, "padding" : True}},
                {"truncation" : False, "padding" : True},
            ] :
                try :
                    encoded = self.model.preprocess(batch, processing_kwargs = processing_kwargs)
                    break
                except Exception :
                    encoded = None

            if (encoded is not None) :
                attention = encoded.get("attention_mask") if isinstance(encoded, dict) else None
                input_ids = encoded.get("input_ids") if isinstance(encoded, dict) else None

                if (attention is not None) :
                    array = attention.detach().cpu().numpy() if hasattr(attention, "detach") else np.asarray(attention)
                    lengths.extend(np.asarray(array).sum(axis = 1).astype(int).tolist())
                    continue

                if (input_ids is not None) :
                    array = input_ids.detach().cpu().numpy() if hasattr(input_ids, "detach") else np.asarray(input_ids)
                    pad_token_id = getattr(self.tokenizer(), "pad_token_id", None)
                    if (pad_token_id is None) :
                        lengths.extend([int(array.shape[1])] * int(array.shape[0]))
                    else :
                        lengths.extend((array != int(pad_token_id)).sum(axis = 1).astype(int).tolist())
                    continue

            tokenizer = self.tokenizer()
            for query, document in batch :
                query_text = (self.specification.instruction + " " + query) if self.specification.instruction else query
                encoded = tokenizer(
                    query_text,
                    document,
                    add_special_tokens = True,
                    truncation = False,
                )
                lengths.append(len(encoded["input_ids"]))

        values_array = np.asarray(lengths, dtype = np.int64)
        above = values_array > int(self.specification.max_length)
        return {
            "pair_count"          : int(len(values_array)),
            "total_input_tokens"  : int(values_array.sum()) if len(values_array) else 0,
            "mean_input_tokens"   : float(values_array.mean()) if len(values_array) else None,
            "p50_input_tokens"    : float(np.percentile(values_array, 50)) if len(values_array) else None,
            "p90_input_tokens"    : float(np.percentile(values_array, 90)) if len(values_array) else None,
            "p95_input_tokens"    : float(np.percentile(values_array, 95)) if len(values_array) else None,
            "max_input_tokens"    : int(values_array.max()) if len(values_array) else None,
            "max_length"          : int(self.specification.max_length),
            "truncated_count"     : int(above.sum()),
            "truncated_fraction"  : float(above.mean()) if len(values_array) else 0.0,
            "audit_policy"        : "cross_encoder_preprocess_untruncated_v1",
        }

    def score_pairs(
        self,
        pairs : Sequence[tuple[str, str]],
        batch_size : int,
        show_progress_bar : bool = False,
    ) -> RerankerScoreOutput :
        values = [(str(query), str(document)) for query, document in pairs]
        if (not values) : raise ValueError(f"{self.specification.backend_id}: cannot score an empty pair list")
        self._load()
        self._synchronize_cuda()
        started = time.perf_counter()
        try :
            scores = self.model.predict(
                values,
                batch_size = int(batch_size),
                show_progress_bar = show_progress_bar,
                convert_to_numpy = True,
            )
        except RuntimeError as error :
            if (self._is_oom(error)) : self.oom_count += 1
            raise
        self._synchronize_cuda()
        runtime_s = time.perf_counter() - started
        array = np.asarray(scores, dtype = np.float32)
        if (array.ndim == 2 and array.shape[1] == 1) : array = array[:, 0]
        if (array.ndim != 1) : raise ValueError(f"{self.specification.backend_id}: expected one scalar score per pair, received shape {array.shape}")
        self.last_batch_size = int(batch_size)
        return RerankerScoreOutput(
            scores = array,
            runtime_s = runtime_s,
            pair_count = len(values),
            effective_batch_size = int(batch_size),
        )

    def select_batch_size(
        self,
        pairs : Sequence[tuple[str, str]],
        candidates : Sequence[int],
        minimum_batch_size : int,
    ) -> tuple[int, list[dict[str, Any]]] :
        values = [(str(query), str(document)) for query, document in pairs]
        if (not values) : raise ValueError("Batch-size preflight requires at least one pair")
        attempts = []
        self._load()

        for batch_size in [int(value) for value in candidates] :
            if (batch_size < int(minimum_batch_size)) : continue
            sample = values[:min(len(values), batch_size)]
            try :
                output = self.score_pairs(sample, batch_size = batch_size, show_progress_bar = False)
                attempts.append({
                    "batch_size" : batch_size,
                    "passed"     : True,
                    "runtime_s"  : output.runtime_s,
                    "pair_count" : output.pair_count,
                    "error"      : None,
                })
                return batch_size, attempts
            except RuntimeError as error :
                if (not self._is_oom(error)) : raise
                attempts.append({
                    "batch_size" : batch_size,
                    "passed"     : False,
                    "runtime_s"  : None,
                    "pair_count" : len(sample),
                    "error"      : "CUDA_OOM",
                })
                self._clear_cuda()

        raise RuntimeError(
            f"{self.specification.backend_id}: no configured batch size >= "
            f"{minimum_batch_size} completed without OOM"
        )

    def warmup(
        self,
        pairs : Sequence[tuple[str, str]],
        batch_size : int,
        pair_count : int,
    ) -> float :
        values = list(pairs)[:max(1, int(pair_count))]
        output = self.score_pairs(values, batch_size = min(int(batch_size), len(values)))
        return float(output.runtime_s)

    def metadata(self) -> dict[str, Any] :
        return {
            **self.specification.identity(),
            "backend_version"     : RERANKER_BACKENDS_VERSION,
            "resolved_revision"   : self._resolve_revision(),
            "device"              : self.device,
            "load_runtime_s"      : float(self.load_runtime_s),
            "last_batch_size"     : self.last_batch_size,
            "oom_count"           : int(self.oom_count),
            "score_activation"    : "identity_raw_logit",
        }

    def release(self) -> None :
        self.model = None
        gc.collect()
        self._clear_cuda()

