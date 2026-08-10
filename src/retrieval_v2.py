# Relative path: src/retrieval_v2.py
# Purpose: Pure Retrieval v2 score production for lexical methods, Baseline v1 E5 compatibility, and dense cosine retrieval.

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from postprocess import accent_fold, normalize_for_matching


RETRIEVAL_V2_VERSION = "1.1.0"


@dataclass
class ChannelDocuments :
    model_id : str
    view : str
    window_ids : list[str]
    texts : list[str]
    eligibility_mask : np.ndarray
    zero_reasons : list[str | None]

    def __post_init__(self) -> None :
        self.window_ids       = [str(value) for value in self.window_ids]
        self.texts            = [str(value or "") for value in self.texts]
        self.eligibility_mask = np.asarray(self.eligibility_mask, dtype = bool)

        count = len(self.window_ids)

        if (len(set(self.window_ids)) != count) :
            raise ValueError("ChannelDocuments contains duplicate window IDs")

        if (len(self.texts) != count) :
            raise ValueError("ChannelDocuments texts length mismatch")

        if (self.eligibility_mask.shape != (count,)) :
            raise ValueError("ChannelDocuments eligibility_mask length mismatch")

        if (len(self.zero_reasons) != count) :
            raise ValueError("ChannelDocuments zero_reasons length mismatch")

    @property
    def channel_id(self) -> str :
        return f"{self.model_id}__{self.view}"

    @property
    def eligible_indices(self) -> list[int] :
        return np.flatnonzero(self.eligibility_mask).astype(int).tolist()


@dataclass
class ScoreBundle :
    method_id : str
    model_id : str
    view : str
    query_ids : list[str]
    window_ids : list[str]
    scores : np.ndarray
    eligibility_mask : np.ndarray
    component_scores : dict[str, np.ndarray] = field(default_factory = dict)
    metadata : dict[str, Any] = field(default_factory = dict)

    def __post_init__(self) -> None :
        self.query_ids        = [str(value) for value in self.query_ids]
        self.window_ids       = [str(value) for value in self.window_ids]
        self.scores           = np.asarray(self.scores)
        self.eligibility_mask = np.asarray(self.eligibility_mask, dtype = bool)

        expected_shape = (len(self.query_ids), len(self.window_ids))

        if (self.scores.shape != expected_shape) :
            raise ValueError(
                f"ScoreBundle score shape {self.scores.shape} does not match {expected_shape}"
            )

        if (self.eligibility_mask.shape != (len(self.window_ids),)) :
            raise ValueError("ScoreBundle eligibility_mask length mismatch")

        if (not np.isfinite(self.scores).all()) :
            raise ValueError("ScoreBundle contains NaN or infinite scores")

        for name, values in self.component_scores.items() :
            array = np.asarray(values)

            if (array.shape != expected_shape) :
                raise ValueError(
                    f"Component {name!r} shape {array.shape} does not match {expected_shape}"
                )

            if (not np.isfinite(array).all()) :
                raise ValueError(f"Component {name!r} contains NaN or infinite scores")

            self.component_scores[name] = array


def _query_ids(query_ids : Sequence[str], expected_count : int) -> list[str] :
    values = [str(value) for value in query_ids]

    if (len(values) != expected_count) :
        raise ValueError("query_ids length does not match query_texts")

    if (len(set(values)) != len(values)) :
        raise ValueError("query_ids contains duplicates")

    return values


def _fit_dual_tfidf(
    preserving_fit_texts : Sequence[str],
    folded_fit_texts : Sequence[str],
    ngram_range : tuple[int, int],
) -> tuple[TfidfVectorizer, TfidfVectorizer] :
    preserving = TfidfVectorizer(
        analyzer = "char_wb",
        ngram_range = ngram_range,
        lowercase = False,
        min_df = 1,
    )
    folded = TfidfVectorizer(
        analyzer = "char_wb",
        ngram_range = ngram_range,
        lowercase = False,
        min_df = 1,
    )

    preserving_values = [normalize_for_matching(text) for text in preserving_fit_texts]
    folded_values     = [accent_fold(text) for text in folded_fit_texts]

    if (not any(preserving_values)) :
        raise ValueError("TF-IDF preserving fit corpus is empty")

    if (not any(folded_values)) :
        raise ValueError("TF-IDF folded fit corpus is empty")

    preserving.fit(preserving_values)
    folded.fit(folded_values)

    return preserving, folded


def _tfidf_coverage(
    vectorizer : TfidfVectorizer,
    texts : Sequence[str],
    normalizer,
) -> list[float] :
    vocabulary = set(vectorizer.vocabulary_)
    analyzer   = vectorizer.build_analyzer()
    values     = []

    for text in texts :
        features = set(analyzer(normalizer(text)))

        if (not features) :
            values.append(0.0)
            continue

        values.append(len(features & vocabulary) / len(features))

    return values


def _score_dual_tfidf(
    method_id : str,
    query_texts : Sequence[str],
    query_ids : Sequence[str],
    documents : ChannelDocuments,
    preserving : TfidfVectorizer,
    folded : TfidfVectorizer,
) -> ScoreBundle :
    queries = [str(value or "") for value in query_texts]
    ids     = _query_ids(query_ids, len(queries))
    shape   = (len(queries), len(documents.window_ids))

    preserving_scores = np.zeros(shape, dtype = np.float32)
    folded_scores     = np.zeros(shape, dtype = np.float32)
    valid_indices     = documents.eligible_indices

    if (valid_indices) :
        passages = [documents.texts[index] for index in valid_indices]

        query_preserving   = preserving.transform([normalize_for_matching(text) for text in queries])
        passage_preserving = preserving.transform([normalize_for_matching(text) for text in passages])
        query_folded       = folded.transform([accent_fold(text) for text in queries])
        passage_folded     = folded.transform([accent_fold(text) for text in passages])

        preserving_valid = cosine_similarity(query_preserving, passage_preserving)
        folded_valid     = cosine_similarity(query_folded, passage_folded)

        preserving_scores[:, valid_indices] = preserving_valid
        folded_scores[:, valid_indices]     = folded_valid

    final_scores = np.maximum(preserving_scores, folded_scores).astype(np.float32, copy = False)
    preserving_coverage = _tfidf_coverage(preserving, queries, normalize_for_matching)
    folded_coverage     = _tfidf_coverage(folded, queries, accent_fold)

    return ScoreBundle(
        method_id = method_id,
        model_id = documents.model_id,
        view = documents.view,
        query_ids = ids,
        window_ids = documents.window_ids,
        scores = final_scores,
        eligibility_mask = documents.eligibility_mask,
        component_scores = {
            "preserving" : preserving_scores,
            "folded"     : folded_scores,
        },
        metadata = {
            "query_coverage" : [
                max(left, right)
                for left, right in zip(preserving_coverage, folded_coverage)
            ],
            "preserving_query_coverage" : preserving_coverage,
            "folded_query_coverage"     : folded_coverage,
            "analyzer"                  : "char_wb",
            "ngram_range"               : list(preserving.ngram_range),
            "lowercase"                 : False,
            "min_df"                    : 1,
        },
    )


def score_query_fitted_dual_tfidf(
    query_texts : Sequence[str],
    query_ids : Sequence[str],
    documents : ChannelDocuments,
    ngram_range : tuple[int, int] = (3, 5),
    method_id : str = "L0_query_tfidf",
) -> ScoreBundle :
    queries = [str(value or "") for value in query_texts]
    preserving, folded = _fit_dual_tfidf(
        queries,
        queries,
        ngram_range = ngram_range,
    )

    bundle = _score_dual_tfidf(
        method_id,
        queries,
        query_ids,
        documents,
        preserving,
        folded,
    )
    bundle.metadata["fit_source"] = "queries"
    return bundle


def score_corpus_fitted_dual_tfidf(
    query_texts : Sequence[str],
    query_ids : Sequence[str],
    documents : ChannelDocuments,
    ngram_range : tuple[int, int] = (3, 5),
    method_id : str = "L1_corpus_tfidf",
) -> ScoreBundle :
    valid_indices = documents.eligible_indices

    if (not valid_indices) :
        raise ValueError("Corpus-fitted TF-IDF requires at least one eligible document")

    passages = [documents.texts[index] for index in valid_indices]
    preserving, folded = _fit_dual_tfidf(
        passages,
        passages,
        ngram_range = ngram_range,
    )

    bundle = _score_dual_tfidf(
        method_id,
        query_texts,
        query_ids,
        documents,
        preserving,
        folded,
    )
    bundle.metadata["fit_source"] = "eligible_corpus"
    bundle.metadata["indexed_document_count"] = len(valid_indices)
    return bundle


def whitespace_tokens(text : str, fold_accents : bool = False) -> list[str] :
    normalized = accent_fold(text) if fold_accents else normalize_for_matching(text)
    return [token for token in normalized.split() if token]


class BM25Index :
    def __init__(
        self,
        document_ids : Sequence[str],
        tokenized_documents : Sequence[Sequence[str]],
        k1 : float = 1.5,
        b : float = 0.75,
    ) :
        self.document_ids = [str(value) for value in document_ids]
        self.documents    = [list(tokens) for tokens in tokenized_documents]
        self.k1           = float(k1)
        self.b            = float(b)

        if (len(self.document_ids) != len(self.documents)) :
            raise ValueError("BM25 document IDs and token lists have different lengths")

        if (len(set(self.document_ids)) != len(self.document_ids)) :
            raise ValueError("BM25 document IDs contain duplicates")

        if (not self.documents) :
            raise ValueError("BM25 requires at least one indexed document")

        if (self.k1 <= 0) :
            raise ValueError("BM25 k1 must be positive")

        if (not 0 <= self.b <= 1) :
            raise ValueError("BM25 b must be within [0, 1]")

        if (any(len(tokens) == 0 for tokens in self.documents)) :
            raise ValueError("BM25 indexed documents must be nonempty")

        self.document_lengths = np.asarray([len(tokens) for tokens in self.documents], dtype = np.float64)
        self.average_document_length = float(self.document_lengths.mean())

        self.term_frequencies : dict[str, dict[int, int]] = {}
        self.document_frequencies : dict[str, int] = {}

        for document_index, tokens in enumerate(self.documents) :
            counts : dict[str, int] = {}

            for token in tokens :
                counts[token] = counts.get(token, 0) + 1

            for token, frequency in counts.items() :
                self.term_frequencies.setdefault(token, {})[document_index] = frequency
                self.document_frequencies[token] = self.document_frequencies.get(token, 0) + 1

    def idf(self, token : str) -> float :
        df = self.document_frequencies.get(token, 0)

        if (df == 0) :
            return 0.0

        n = len(self.documents)
        return math.log(1.0 + (n - df + 0.5) / (df + 0.5))

    def score_tokens(self, query_tokens : Sequence[str]) -> np.ndarray :
        scores = np.zeros(len(self.documents), dtype = np.float32)
        unique_tokens = list(dict.fromkeys(str(token) for token in query_tokens if str(token)))

        for token in unique_tokens :
            postings = self.term_frequencies.get(token)

            if (not postings) :
                continue

            idf = self.idf(token)

            for document_index, frequency in postings.items() :
                length = self.document_lengths[document_index]
                denominator = frequency + self.k1 * (
                    1.0 - self.b
                    + self.b * length / self.average_document_length
                )
                contribution = idf * (
                    frequency * (self.k1 + 1.0)
                    / denominator
                )
                scores[document_index] += np.float32(contribution)

        if (not np.isfinite(scores).all()) :
            raise ValueError("BM25 produced NaN or infinite scores")

        if ((scores < 0).any()) :
            raise ValueError("BM25 produced negative scores")

        return scores

    def vocabulary(self) -> set[str] :
        return set(self.document_frequencies)


def score_bm25(
    query_texts : Sequence[str],
    query_ids : Sequence[str],
    documents : ChannelDocuments,
    fold_accents : bool,
    k1 : float = 1.5,
    b : float = 0.75,
    method_id : str | None = None,
) -> ScoreBundle :
    queries = [str(value or "") for value in query_texts]
    ids     = _query_ids(query_ids, len(queries))
    valid_indices = documents.eligible_indices

    if (not valid_indices) :
        raise ValueError("BM25 requires at least one eligible document")

    indexed_ids    = [documents.window_ids[index] for index in valid_indices]
    indexed_tokens = [
        whitespace_tokens(documents.texts[index], fold_accents = fold_accents)
        for index in valid_indices
    ]

    if (any(not tokens for tokens in indexed_tokens)) :
        raise ValueError(
            "Eligible BM25 documents must remain nonempty after tokenization"
        )

    index = BM25Index(
        indexed_ids,
        indexed_tokens,
        k1 = k1,
        b = b,
    )

    scores = np.zeros((len(queries), len(documents.window_ids)), dtype = np.float32)
    coverage = []
    vocabulary = index.vocabulary()

    for query_index, query in enumerate(queries) :
        tokens = whitespace_tokens(query, fold_accents = fold_accents)
        unique_tokens = list(dict.fromkeys(tokens))
        valid_scores  = index.score_tokens(unique_tokens)
        scores[query_index, valid_indices] = valid_scores

        if (not unique_tokens) :
            coverage.append(0.0)
        else :
            coverage.append(
                len(set(unique_tokens) & vocabulary) / len(set(unique_tokens))
            )

    resolved_method = method_id or (
        "L3_bm25_folded" if fold_accents else "L2_bm25_preserving"
    )

    return ScoreBundle(
        method_id = resolved_method,
        model_id = documents.model_id,
        view = documents.view,
        query_ids = ids,
        window_ids = documents.window_ids,
        scores = scores,
        eligibility_mask = documents.eligibility_mask,
        metadata = {
            "fold_accents"          : bool(fold_accents),
            "tokenizer"             : "normalized_whitespace",
            "k1"                    : float(k1),
            "b"                     : float(b),
            "idf_policy"            : "log1p_positive_okapi",
            "query_term_frequency"  : "unique_terms",
            "indexed_document_count": len(valid_indices),
            "average_document_length": index.average_document_length,
            "query_coverage"        : coverage,
        },
    )


def positive_evidence_rrf(
    left : ScoreBundle,
    right : ScoreBundle,
    k : int = 60,
    tie_tolerance : float = 1e-12,
    method_id : str = "L4_bm25_rrf",
) -> ScoreBundle :
    if (left.query_ids != right.query_ids) :
        raise ValueError("RRF query IDs differ")

    if (left.window_ids != right.window_ids) :
        raise ValueError("RRF window IDs differ")

    if (left.model_id != right.model_id or left.view != right.view) :
        raise ValueError("RRF channel identities differ")

    if (not np.array_equal(left.eligibility_mask, right.eligibility_mask)) :
        raise ValueError("RRF eligibility masks differ")

    if (k <= 0) :
        raise ValueError("RRF k must be positive")

    from retrieval_v2_evaluation import worst_tied_ranks_array

    fused = np.zeros_like(left.scores, dtype = np.float32)

    for query_index in range(len(left.query_ids)) :
        for source in [left.scores[query_index], right.scores[query_index]] :
            positive = (source > 0) & left.eligibility_mask

            if (not positive.any()) :
                continue

            indices = np.flatnonzero(positive)
            ranks   = worst_tied_ranks_array(
                source[indices],
                tolerance = tie_tolerance,
            )

            for local_index, rank in zip(indices, ranks) :
                fused[query_index, local_index] += np.float32(
                    1.0 / (k + int(rank))
                )

    left_coverage  = list(left.metadata.get("query_coverage", [0.0] * len(left.query_ids)))
    right_coverage = list(right.metadata.get("query_coverage", [0.0] * len(right.query_ids)))

    return ScoreBundle(
        method_id = method_id,
        model_id = left.model_id,
        view = left.view,
        query_ids = left.query_ids,
        window_ids = left.window_ids,
        scores = fused,
        eligibility_mask = left.eligibility_mask,
        component_scores = {
            "left"  : left.scores,
            "right" : right.scores,
        },
        metadata = {
            "rrf_k"          : int(k),
            "positive_only"  : True,
            "query_coverage" : [
                max(left_value, right_value)
                for left_value, right_value in zip(left_coverage, right_coverage)
            ],
            "left_method"    : left.method_id,
            "right_method"   : right.method_id,
        },
    )


class E5CompatibilityScorer :
    def __init__(
        self,
        model_name : str,
        revision : str,
        query_prefix : str = "query: ",
        passage_prefix : str = "passage: ",
        show_progress_bar : bool = True,
    ) :
        from sentence_transformers import SentenceTransformer

        self.model_name        = model_name
        self.revision          = revision
        self.query_prefix      = query_prefix
        self.passage_prefix    = passage_prefix
        self.show_progress_bar = show_progress_bar
        self.encoder = SentenceTransformer(
            model_name,
            revision = revision,
        )
        self._query_cache : dict[tuple[str, ...], np.ndarray] = {}
        self._passage_cache : dict[tuple[str, ...], np.ndarray] = {}

    def _encode_queries(self, texts : Sequence[str]) -> np.ndarray :
        values = tuple(str(text or "") for text in texts)

        if (values not in self._query_cache) :
            embeddings = self.encoder.encode(
                [self.query_prefix + text for text in values],
                normalize_embeddings = True,
                show_progress_bar = self.show_progress_bar,
            )
            self._query_cache[values] = np.asarray(embeddings)

        return self._query_cache[values]

    def _encode_passages(self, texts : Sequence[str]) -> np.ndarray :
        values = tuple(str(text or "") for text in texts)

        if (values not in self._passage_cache) :
            embeddings = self.encoder.encode(
                [self.passage_prefix + text for text in values],
                normalize_embeddings = True,
                show_progress_bar = self.show_progress_bar,
            )
            self._passage_cache[values] = np.asarray(embeddings)

        return self._passage_cache[values]

    def score(
        self,
        query_texts : Sequence[str],
        query_ids : Sequence[str],
        documents : ChannelDocuments,
        method_id : str = "baseline_v1_semantic",
    ) -> ScoreBundle :
        queries = [str(value or "") for value in query_texts]
        ids     = _query_ids(query_ids, len(queries))
        shape   = (len(queries), len(documents.window_ids))
        scores  = np.zeros(shape, dtype = np.float32)
        valid_indices = documents.eligible_indices

        if (valid_indices) :
            passages = [documents.texts[index] for index in valid_indices]
            query_embeddings   = self._encode_queries(queries)
            passage_embeddings = self._encode_passages(passages)
            valid_scores = np.clip(
                query_embeddings @ passage_embeddings.T,
                0.0,
                1.0,
            )
            scores[:, valid_indices] = valid_scores

        return ScoreBundle(
            method_id = method_id,
            model_id = documents.model_id,
            view = documents.view,
            query_ids = ids,
            window_ids = documents.window_ids,
            scores = scores,
            eligibility_mask = documents.eligibility_mask,
            metadata = self.metadata(),
        )

    def metadata(self) -> dict[str, Any] :
        return {
            "model_name"      : self.model_name,
            "revision"        : self.revision,
            "query_prefix"    : self.query_prefix,
            "passage_prefix"  : self.passage_prefix,
            "normalized_embeddings" : True,
            "similarity"      : "dot_product",
            "score_clip"      : [0.0, 1.0],
        }


def weighted_score_bundle(
    lexical : ScoreBundle,
    semantic : ScoreBundle,
    lexical_weight : float = 0.5,
    semantic_weight : float = 0.5,
    method_id : str = "baseline_v1",
) -> ScoreBundle :
    if (lexical.query_ids != semantic.query_ids) :
        raise ValueError("Lexical and semantic query IDs differ")

    if (lexical.window_ids != semantic.window_ids) :
        raise ValueError("Lexical and semantic window IDs differ")

    if (lexical.model_id != semantic.model_id or lexical.view != semantic.view) :
        raise ValueError("Lexical and semantic channels differ")

    if (not np.array_equal(lexical.eligibility_mask, semantic.eligibility_mask)) :
        raise ValueError("Lexical and semantic eligibility masks differ")

    total = float(lexical_weight) + float(semantic_weight)

    if (total <= 0) :
        raise ValueError("Retrieval weights must have positive sum")

    lexical_normalized  = float(lexical_weight) / total
    semantic_normalized = float(semantic_weight) / total

    scores = (
        lexical_normalized * lexical.scores
        + semantic_normalized * semantic.scores
    ).astype(np.float32, copy = False)

    return ScoreBundle(
        method_id = method_id,
        model_id = lexical.model_id,
        view = lexical.view,
        query_ids = lexical.query_ids,
        window_ids = lexical.window_ids,
        scores = scores,
        eligibility_mask = lexical.eligibility_mask,
        component_scores = {
            "lexical"  : lexical.scores,
            "semantic" : semantic.scores,
        },
        metadata = {
            "lexical_weight"  : lexical_normalized,
            "semantic_weight" : semantic_normalized,
            "lexical_method"  : lexical.method_id,
            "semantic_method" : semantic.method_id,
            "query_coverage"  : lexical.metadata.get(
                "query_coverage",
                [0.0] * len(lexical.query_ids),
            ),
        },
    )



def score_dense_embeddings(
    query_ids : Sequence[str],
    documents : ChannelDocuments,
    query_embeddings : np.ndarray,
    document_embeddings : np.ndarray,
    method_id : str,
    invalid_document_score : float = -2.0,
    metadata : dict[str, Any] | None = None,
) -> ScoreBundle :
    queries = _query_ids(query_ids, len(query_embeddings))
    query_matrix = np.asarray(query_embeddings, dtype = np.float32)
    document_matrix = np.asarray(document_embeddings, dtype = np.float32)
    valid_indices = documents.eligible_indices

    if (query_matrix.ndim != 2 or document_matrix.ndim != 2) : raise ValueError("Dense query/document embeddings must be two-dimensional")
    if (query_matrix.shape[1] != document_matrix.shape[1]) : raise ValueError("Dense query/document embedding dimensions differ")
    if (document_matrix.shape[0] != len(valid_indices)) : raise ValueError("Dense document embedding rows must match eligible documents")
    if (not np.isfinite(query_matrix).all() or not np.isfinite(document_matrix).all()) : raise ValueError("Dense embeddings contain NaN or infinite values")
    if (not math.isfinite(float(invalid_document_score)) or invalid_document_score >= -1.0) : raise ValueError("invalid_document_score must be finite and below the cosine floor -1")

    valid_scores = query_matrix @ document_matrix.T
    if (not np.isfinite(valid_scores).all()) : raise ValueError("Dense cosine scoring produced NaN or infinite values")
    if ((valid_scores < -1.0005).any() or (valid_scores > 1.0005).any()) : raise ValueError("Dense cosine scores fall outside the expected normalized range")

    scores = np.full((len(queries), len(documents.window_ids)), np.float32(invalid_document_score), dtype = np.float32)
    if (valid_indices) : scores[:, valid_indices] = valid_scores.astype(np.float32, copy = False)

    return ScoreBundle(method_id = method_id, model_id = documents.model_id, view = documents.view, query_ids = queries, window_ids = documents.window_ids, scores = scores, eligibility_mask = documents.eligibility_mask, metadata = {"similarity" : "cosine", "score_clip" : None, "invalid_document_policy" : "finite_below_cosine_floor", "invalid_document_score" : float(invalid_document_score), **(metadata or {})})
