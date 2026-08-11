# Relative path: src/retrieval_v2_evaluation.py
# Purpose: Independent Retrieval v2 evaluation policy for eligibility, temporal relevance, ranking, metrics, and method comparison.

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd


RETRIEVAL_V2_EVALUATION_VERSION = "1.3.0"
SUCCESS_STATUSES = {"ok", "success"}


def classify_eligibility(
    windows : pd.DataFrame,
    text_field : str,
    rejection_field : str,
    apply_rejections : bool,
) -> tuple[np.ndarray, list[str | None]] :
    mask    = np.zeros(len(windows), dtype = bool)
    reasons = []

    for index, (_, row) in enumerate(windows.iterrows()) :
        status = str(row.get("status", "") or "")

        if (status not in SUCCESS_STATUSES) :
            reasons.append(f"status_{status or 'missing'}")
            continue

        text = str(row.get(text_field, "") or "").strip()

        if (not text) :
            reasons.append("empty_text")
            continue

        if (apply_rejections and bool(row.get(rejection_field, []))) :
            reasons.append("processed_rejection")
            continue

        mask[index] = True
        reasons.append(None)

    return mask, reasons


def silver_relevance_mask(
    query : pd.Series,
    windows : pd.DataFrame,
    silver_radius_s : float,
    minimum_overlap_s : float,
) -> tuple[np.ndarray, np.ndarray] :
    zone_start = float(query["answer_time_s"]) - float(silver_radius_s)
    zone_end   = float(query["answer_time_s"]) + float(silver_radius_s)

    starts = pd.to_numeric(windows["start_s"], errors = "coerce").to_numpy(dtype = float)
    ends   = pd.to_numeric(windows["end_s"], errors = "coerce").to_numpy(dtype = float)

    overlap = np.maximum(0.0, np.minimum(ends, zone_end) - np.maximum(starts, zone_start))
    correct = windows["video_id"].astype(str).to_numpy() == str(query["video_id"])
    relevant = correct & (overlap >= float(minimum_overlap_s))

    return relevant, overlap


def worst_tied_ranks_array(scores : np.ndarray, tolerance : float = 1e-12) -> np.ndarray :
    values = np.asarray(scores, dtype = float)

    if (values.ndim != 1) :
        raise ValueError("scores must be one-dimensional")

    if (not np.isfinite(values).all()) :
        raise ValueError("scores contain NaN or infinite values")

    order = sorted(range(len(values)), key = lambda index : (-values[index], index))
    ranks = np.zeros(len(values), dtype = np.int64)
    start = 0

    while (start < len(order)) :
        end        = start + 1
        base_score = values[order[start]]

        while (end < len(order) and abs(values[order[end]] - base_score) <= tolerance) :
            end += 1

        worst_rank = end

        for position in range(start, end) :
            ranks[order[position]] = worst_rank

        start = end

    return ranks


def _stable_top_index(
    scores : np.ndarray,
    windows : pd.DataFrame,
) -> int | None :
    if (len(scores) == 0) :
        return None

    order = pd.DataFrame({
        "_index"     : np.arange(len(scores)),
        "score"      : np.asarray(scores, dtype = float),
        "start_s"    : pd.to_numeric(windows["start_s"], errors = "coerce").to_numpy(),
        "window_id"  : windows["window_id"].astype(str).to_numpy(),
    }).sort_values(
        ["score", "start_s", "window_id"],
        ascending = [False, True, True],
        kind = "mergesort",
    )

    return int(order.iloc[0]["_index"])


def evaluate_score_matrix(
    scores : np.ndarray,
    queries : pd.DataFrame,
    windows : pd.DataFrame,
    model_id : str,
    view : str,
    method_id : str,
    query_set : str,
    corpus_id : str,
    silver_radius_s : float,
    minimum_overlap_s : float,
    tie_tolerance : float = 1e-12,
    eligibility_mask : np.ndarray | None = None,
    zero_reasons : Sequence[str | None] | None = None,
    query_coverage : Sequence[float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame] :
    values = np.asarray(scores, dtype = float)

    if (values.shape != (len(queries), len(windows))) :
        raise ValueError(
            f"Score shape {values.shape} does not match "
            f"{len(queries)} queries x {len(windows)} windows"
        )

    if (not np.isfinite(values).all()) :
        raise ValueError("Score matrix contains NaN or infinite values")

    eligibility = (
        np.asarray(eligibility_mask, dtype = bool)
        if eligibility_mask is not None
        else np.ones(len(windows), dtype = bool)
    )

    if (eligibility.shape != (len(windows),)) :
        raise ValueError("eligibility_mask length does not match windows")

    reasons = list(zero_reasons or [None] * len(windows))

    if (len(reasons) != len(windows)) :
        raise ValueError("zero_reasons length does not match windows")

    coverages = list(query_coverage or [math.nan] * len(queries))

    if (len(coverages) != len(queries)) :
        raise ValueError("query_coverage length does not match queries")

    window_ids = windows["window_id"].astype(str).to_numpy()
    video_ids  = windows["video_id"].astype(str).to_numpy()
    starts     = pd.to_numeric(windows["start_s"], errors = "coerce").to_numpy(dtype = float)
    ends       = pd.to_numeric(windows["end_s"], errors = "coerce").to_numpy(dtype = float)

    query_rows_out = []
    evidence_rows  = []

    for query_index, (_, query) in enumerate(queries.reset_index(drop = True).iterrows()) :
        row_scores = values[query_index]
        relevant, overlap = silver_relevance_mask(
            query,
            windows,
            silver_radius_s = silver_radius_s,
            minimum_overlap_s = minimum_overlap_s,
        )

        correct_video = str(query["video_id"])
        story_indices = np.flatnonzero(video_ids == correct_video)
        story_scores  = row_scores[story_indices]
        story_ranks   = worst_tied_ranks_array(story_scores, tolerance = tie_tolerance)
        story_relevant = relevant[story_indices]
        relevant_story_indices = np.flatnonzero(story_relevant)

        first_rank = (
            int(story_ranks[relevant_story_indices].min())
            if len(relevant_story_indices)
            else None
        )

        relevant_scores   = story_scores[story_relevant]
        irrelevant_scores = story_scores[~story_relevant]
        best_relevant_score   = float(relevant_scores.max()) if len(relevant_scores) else None
        best_irrelevant_score = float(irrelevant_scores.max()) if len(irrelevant_scores) else None
        story_margin = (
            best_relevant_score - best_irrelevant_score
            if best_relevant_score is not None and best_irrelevant_score is not None
            else None
        )

        story_window_frame = windows.iloc[story_indices].reset_index(drop = True)
        top_story_local = _stable_top_index(story_scores, story_window_frame)
        top_story_global = (
            int(story_indices[top_story_local])
            if top_story_local is not None
            else None
        )

        video_frame = pd.DataFrame({
            "video_id" : video_ids,
            "score"    : row_scores,
        }).groupby("video_id", as_index = False)["score"].max()

        video_frame = video_frame.sort_values(
            ["video_id"],
            ascending = [True],
            kind = "mergesort",
        ).reset_index(drop = True)

        video_ranks = worst_tied_ranks_array(
            video_frame["score"].to_numpy(dtype = float),
            tolerance = tie_tolerance,
        )
        video_frame["metric_rank"] = video_ranks
        video_frame = video_frame.sort_values(
            ["score", "video_id"],
            ascending = [False, True],
            kind = "mergesort",
        ).reset_index(drop = True)

        video_match = video_frame[video_frame["video_id"] == correct_video]
        video_rank  = int(video_match["metric_rank"].iloc[0]) if not video_match.empty else None
        correct_video_score = float(video_match["score"].iloc[0]) if not video_match.empty else None
        wrong_videos = video_frame[video_frame["video_id"] != correct_video]
        best_wrong_video_score = float(wrong_videos["score"].max()) if not wrong_videos.empty else None
        video_margin = (
            correct_video_score - best_wrong_video_score
            if correct_video_score is not None and best_wrong_video_score is not None
            else None
        )

        top_video = video_frame.iloc[0] if not video_frame.empty else None
        positive_window_mask = (row_scores > 0) & eligibility
        positive_video_count = len(set(video_ids[positive_window_mask].tolist()))
        correct_positive_count = int(((video_ids == correct_video) & positive_window_mask).sum())

        query_rows_out.append({
            "query_set"                  : query_set,
            "corpus_id"                  : corpus_id,
            "method_id"                  : method_id,
            "model_id"                   : model_id,
            "view"                       : view,
            "query_id"                   : query["query_id"],
            "query_text"                 : query["query_text"],
            "query_category"             : query.get("query_category", "other"),
            "task_type"                  : query.get("task_type", "KIS"),
            "difficulty"                 : query.get("difficulty", "unknown"),
            "evaluation_split"           : query.get("evaluation_split", "unspecified"),
            "answer_text"                : query.get("answer_text"),
            "correct_video"              : correct_video,
            "frame_id"                   : query.get("frame_id"),
            "answer_time_s"              : query["answer_time_s"],
            "first_relevant_rank"        : first_rank,
            "story_recall_at_1"          : int(first_rank is not None and first_rank <= 1),
            "story_recall_at_3"          : int(first_rank is not None and first_rank <= 3),
            "story_recall_at_5"          : int(first_rank is not None and first_rank <= 5),
            "story_recall_at_10"         : int(first_rank is not None and first_rank <= 10),
            "story_rr"                   : 1.0 / first_rank if first_rank else 0.0,
            "best_relevant_score"        : best_relevant_score,
            "best_irrelevant_score"      : best_irrelevant_score,
            "story_score_margin"         : story_margin,
            "video_rank"                 : video_rank,
            "video_recall_at_1"          : int(video_rank is not None and video_rank <= 1),
            "video_recall_at_3"          : int(video_rank is not None and video_rank <= 3),
            "video_recall_at_5"          : int(video_rank is not None and video_rank <= 5),
            "video_recall_at_10"         : int(video_rank is not None and video_rank <= 10),
            "video_recall_at_20"         : int(video_rank is not None and video_rank <= 20),
            "video_rr"                   : 1.0 / video_rank if video_rank else 0.0,
            "correct_video_score"        : correct_video_score,
            "best_wrong_video_score"     : best_wrong_video_score,
            "video_score_margin"         : video_margin,
            "top_story_window_id"        : window_ids[top_story_global] if top_story_global is not None else None,
            "top_story_start_s"          : starts[top_story_global] if top_story_global is not None else None,
            "top_story_end_s"            : ends[top_story_global] if top_story_global is not None else None,
            "top_story_score"            : float(row_scores[top_story_global]) if top_story_global is not None else None,
            "top_video_id"               : str(top_video["video_id"]) if top_video is not None else None,
            "top_video_score"            : float(top_video["score"]) if top_video is not None else None,
            "physical_window_count"      : len(windows),
            "eligible_window_count"      : int(eligibility.sum()),
            "positive_window_count"      : int(positive_window_mask.sum()),
            "positive_window_fraction"   : float(positive_window_mask.mean()),
            "positive_video_count"       : int(positive_video_count),
            "correct_video_positive_windows" : correct_positive_count,
            "query_coverage"             : None if not math.isfinite(float(coverages[query_index])) else float(coverages[query_index]),
            "zero_evidence"              : bool(not positive_window_mask.any()),
        })

        top_window_order = pd.DataFrame({
            "_index"    : np.arange(len(windows)),
            "score"     : row_scores,
            "video_id"  : video_ids,
            "start_s"   : starts,
            "window_id" : window_ids,
        }).sort_values(
            ["score", "video_id", "start_s", "window_id"],
            ascending = [False, True, True, True],
            kind = "mergesort",
        ).head(20)

        for rank, (_, item) in enumerate(top_window_order.iterrows(), start = 1) :
            index = int(item["_index"])
            evidence_rows.append({
                "query_set"  : query_set,
                "method_id"  : method_id,
                "model_id"   : model_id,
                "view"       : view,
                "query_id"   : query["query_id"],
                "entity_type": "window",
                "rank"       : rank,
                "entity_id"  : window_ids[index],
                "video_id"   : video_ids[index],
                "start_s"    : starts[index],
                "end_s"      : ends[index],
                "score"      : float(row_scores[index]),
                "correct"    : bool(video_ids[index] == correct_video),
                "relevant"   : bool(relevant[index]),
                "eligible"   : bool(eligibility[index]),
                "zero_reason": reasons[index],
                "silver_overlap_s": float(overlap[index]),
            })

        for rank, (_, item) in enumerate(video_frame.head(10).iterrows(), start = 1) :
            evidence_rows.append({
                "query_set"  : query_set,
                "method_id"  : method_id,
                "model_id"   : model_id,
                "view"       : view,
                "query_id"   : query["query_id"],
                "entity_type": "video",
                "rank"       : rank,
                "entity_id"  : str(item["video_id"]),
                "video_id"   : str(item["video_id"]),
                "start_s"    : None,
                "end_s"      : None,
                "score"      : float(item["score"]),
                "correct"    : bool(str(item["video_id"]) == correct_video),
                "relevant"   : bool(str(item["video_id"]) == correct_video),
                "eligible"   : True,
                "zero_reason": None,
                "silver_overlap_s": None,
            })

    return pd.DataFrame(query_rows_out), pd.DataFrame(evidence_rows)


def summarize_retrieval_metrics(query_results : pd.DataFrame) -> pd.DataFrame :
    if (query_results.empty) :
        return pd.DataFrame()

    group_columns = ["query_set", "corpus_id", "method_id", "model_id", "view"]
    rows = []

    for keys, group in query_results.groupby(group_columns, dropna = False, sort = True) :
        row = dict(zip(group_columns, keys))
        first_ranks = pd.to_numeric(group["first_relevant_rank"], errors = "coerce")
        video_ranks = pd.to_numeric(group["video_rank"], errors = "coerce")

        row.update({
            "query_count"               : len(group),
            "story_recall_at_1"         : float(group["story_recall_at_1"].mean()),
            "story_recall_at_3"         : float(group["story_recall_at_3"].mean()),
            "story_recall_at_5"         : float(group["story_recall_at_5"].mean()),
            "story_recall_at_10"        : float(group["story_recall_at_10"].mean()),
            "story_mrr"                 : float(group["story_rr"].mean()),
            "story_first_rank_mean"     : float(first_ranks.mean()) if first_ranks.notna().any() else None,
            "story_first_rank_median"   : float(first_ranks.median()) if first_ranks.notna().any() else None,
            "video_recall_at_1"         : float(group["video_recall_at_1"].mean()),
            "video_recall_at_3"         : float(group["video_recall_at_3"].mean()),
            "video_recall_at_5"         : float(group["video_recall_at_5"].mean()),
            "video_recall_at_10"        : float(group["video_recall_at_10"].mean()),
            "video_recall_at_20"        : float(group["video_recall_at_20"].mean()),
            "video_mrr"                 : float(group["video_rr"].mean()),
            "video_rank_mean"           : float(video_ranks.mean()) if video_ranks.notna().any() else None,
            "video_rank_median"         : float(video_ranks.median()) if video_ranks.notna().any() else None,
            "story_score_margin_mean"   : float(pd.to_numeric(group["story_score_margin"], errors = "coerce").mean()),
            "video_score_margin_mean"   : float(pd.to_numeric(group["video_score_margin"], errors = "coerce").mean()),
            "zero_evidence_query_count" : int(group["zero_evidence"].sum()),
            "query_coverage_mean"       : float(pd.to_numeric(group["query_coverage"], errors = "coerce").mean()),
        })
        rows.append(row)

    return pd.DataFrame(rows)


def rank_relation(candidate : Any, reference : Any) -> str :
    left  = float(candidate) if pd.notna(candidate) else math.inf
    right = float(reference) if pd.notna(reference) else math.inf

    if (left < right) :
        return "better"
    if (left > right) :
        return "worse"
    return "tie"


def compare_methods(
    query_results : pd.DataFrame,
    reference_method : str = "L0_query_tfidf",
) -> pd.DataFrame :
    if (query_results.empty) :
        return pd.DataFrame()

    reference = query_results[query_results["method_id"] == reference_method].copy()
    rows = []

    for method_id in sorted(set(query_results["method_id"]) - {reference_method}) :
        candidate = query_results[query_results["method_id"] == method_id].copy()
        merged = candidate.merge(
            reference[
                [
                    "query_set", "model_id", "view", "query_id",
                    "first_relevant_rank", "video_rank", "story_rr", "video_rr",
                    "top_story_window_id", "top_video_id",
                ]
            ],
            on = ["query_set", "model_id", "view", "query_id"],
            how = "inner",
            suffixes = ("", "_reference"),
        )

        for _, item in merged.iterrows() :
            rows.append({
                "query_set"      : item["query_set"],
                "model_id"       : item["model_id"],
                "view"           : item["view"],
                "query_id"       : item["query_id"],
                "reference_method": reference_method,
                "candidate_method": method_id,
                "story_relation" : rank_relation(item["first_relevant_rank"], item["first_relevant_rank_reference"]),
                "video_relation" : rank_relation(item["video_rank"], item["video_rank_reference"]),
                "story_rr_delta" : float(item["story_rr"] - item["story_rr_reference"]),
                "video_rr_delta" : float(item["video_rr"] - item["video_rr_reference"]),
                "top_story_changed": bool(item["top_story_window_id"] != item["top_story_window_id_reference"]),
                "top_video_changed": bool(item["top_video_id"] != item["top_video_id_reference"]),
            })

    return pd.DataFrame(rows)


def _bootstrap_interval(
    values : np.ndarray,
    confidence : float,
    samples : int,
    seed : int,
) -> tuple[float, float] :
    if (len(values) == 0) :
        return math.nan, math.nan

    generator = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype = float)

    for index in range(samples) :
        draw = generator.choice(values, size = len(values), replace = True)
        estimates[index] = float(draw.mean())

    alpha = (1.0 - confidence) / 2.0
    return (
        float(np.quantile(estimates, alpha)),
        float(np.quantile(estimates, 1.0 - alpha)),
    )


def summarize_stage1_methods(
    query_results : pd.DataFrame,
    reference_method : str = "L0_query_tfidf",
    bootstrap_samples : int = 2000,
    confidence : float = 0.90,
    seed : int = 20260810,
) -> pd.DataFrame :
    if (query_results.empty) :
        return pd.DataFrame()

    per_query = (
        query_results.groupby(["method_id", "query_id"], as_index = False)
        .agg(
            composite_video_rr = ("video_rr", "mean"),
            composite_story_rr = ("story_rr", "mean"),
        )
    )

    reference = per_query[per_query["method_id"] == reference_method][
        ["query_id", "composite_video_rr", "composite_story_rr"]
    ].rename(columns={
        "composite_video_rr" : "reference_video_rr",
        "composite_story_rr" : "reference_story_rr",
    })

    rows = []

    for method_id, group in per_query.groupby("method_id", sort = True) :
        merged = group.merge(reference, on = "query_id", how = "inner")
        video_delta = (merged["composite_video_rr"] - merged["reference_video_rr"]).to_numpy(dtype = float)
        story_delta = (merged["composite_story_rr"] - merged["reference_story_rr"]).to_numpy(dtype = float)
        low, high = _bootstrap_interval(video_delta, confidence, bootstrap_samples, seed)

        relations = [
            "better" if value > 0 else "worse" if value < 0 else "tie"
            for value in video_delta
        ]
        counts = Counter(relations)

        rows.append({
            "method_id"                        : method_id,
            "query_count"                      : len(merged),
            "composite_video_rr_mean"          : float(merged["composite_video_rr"].mean()),
            "composite_story_rr_mean"          : float(merged["composite_story_rr"].mean()),
            "video_rr_delta_mean_vs_L0"        : float(video_delta.mean()),
            "video_rr_delta_median_vs_L0"      : float(np.median(video_delta)),
            "story_rr_delta_mean_vs_L0"        : float(story_delta.mean()),
            "video_better_query_count_vs_L0"   : int(counts.get("better", 0)),
            "video_tie_query_count_vs_L0"      : int(counts.get("tie", 0)),
            "video_worse_query_count_vs_L0"    : int(counts.get("worse", 0)),
            "video_rr_delta_bootstrap_90_low"  : low,
            "video_rr_delta_bootstrap_90_high" : high,
            "provisional_default"              : None,
        })

    return pd.DataFrame(rows)


def compare_expected_metrics(
    metrics : pd.DataFrame,
    expected : dict[str, Any],
) -> dict[str, Any] :
    checks = []
    errors = []

    for query_set, model_spec in expected.items() :
        for model_id, view_spec in model_spec.items() :
            for view, expectation in view_spec.items() :
                subset = metrics[
                    (metrics["query_set"] == query_set)
                    & (metrics["model_id"] == model_id)
                    & (metrics["view"] == view)
                    & (metrics["method_id"] == expectation.get("method_id", "baseline_v1"))
                ]

                if (subset.empty) :
                    errors.append(f"Missing metrics for {query_set}/{model_id}/{view}")
                    continue

                row = subset.iloc[0]
                tolerance = float(expectation.get("tolerance", 1e-9))

                for metric, target in expectation.get("metrics", {}).items() :
                    actual = float(row[metric])
                    difference = abs(actual - float(target))
                    passed = difference <= tolerance

                    checks.append({
                        "query_set" : query_set,
                        "model_id"  : model_id,
                        "view"      : view,
                        "metric"    : metric,
                        "expected"  : float(target),
                        "actual"    : actual,
                        "difference": difference,
                        "tolerance" : tolerance,
                        "passed"    : passed,
                    })

                    if (not passed) :
                        errors.append(
                            f"{query_set}/{model_id}/{view}/{metric}: "
                            f"{actual:.12f} != {float(target):.12f}"
                        )

    return {
        "passed" : not errors,
        "checks" : checks,
        "errors" : errors,
    }

# -----------------------------------------------------------------------------
# Frozen Baseline v1 fixture regression
# -----------------------------------------------------------------------------


def _fixture_is_null(value : Any) -> bool :
    if (value is None) :
        return True

    try :
        return bool(pd.isna(value))
    except Exception :
        return False


def _fixture_exact_equal(actual : Any, expected : Any) -> bool :
    if (_fixture_is_null(expected)) :
        return _fixture_is_null(actual)

    if (_fixture_is_null(actual)) :
        return False

    if (isinstance(expected, (int, np.integer)) and not isinstance(expected, bool)) :
        try :
            return int(actual) == int(expected)
        except (TypeError, ValueError, OverflowError) :
            return False

    return str(actual) == str(expected)


def _fixture_numeric_difference(actual : Any, expected : Any) -> float :
    if (_fixture_is_null(expected)) :
        return 0.0 if _fixture_is_null(actual) else math.inf

    if (_fixture_is_null(actual)) :
        return math.inf

    try :
        return abs(float(actual) - float(expected))
    except (TypeError, ValueError, OverflowError) :
        return math.inf


def compare_regression_fixture(
    query_results : pd.DataFrame,
    metrics : pd.DataFrame,
    fixture : dict[str, Any],
) -> dict[str, Any] :
    if (fixture.get("schema_version") != "1.0") :
        raise ValueError(
            f"Unsupported regression fixture schema: {fixture.get('schema_version')!r}"
        )

    policy = fixture.get("comparison_policy", {})
    expected = fixture.get("expected", {})
    expected_queries = expected.get("query_results", [])
    expected_metrics = expected.get("aggregate_metrics", [])

    query_key_fields = list(policy.get(
        "query_key_fields",
        ["query_set", "model_id", "view", "query_id", "method_id"],
    ))
    query_exact_fields = list(policy.get("query_exact_fields", []))
    query_score_fields = list(policy.get("query_score_fields", []))
    aggregate_key_fields = list(policy.get(
        "aggregate_key_fields",
        ["query_set", "model_id", "view", "method_id"],
    ))
    aggregate_exact_fields = list(policy.get("aggregate_exact_fields", []))
    aggregate_rank_fields = list(policy.get("aggregate_rank_metric_fields", []))
    aggregate_score_fields = list(policy.get("aggregate_score_metric_fields", []))
    rank_tolerance = float(policy.get("rank_metric_tolerance", 1e-12))
    score_tolerance = float(policy.get("score_tolerance", 1e-5))

    query_frame = query_results.copy()
    metric_frame = metrics.copy()
    errors = []
    query_checks = []
    metric_checks = []
    maximum_score_difference = 0.0
    maximum_rank_metric_difference = 0.0
    maximum_score_metric_difference = 0.0

    for expectation in expected_queries :
        mask = np.ones(len(query_frame), dtype = bool)

        for field in query_key_fields :
            if (field not in query_frame.columns) :
                raise KeyError(f"Query results are missing fixture key field: {field}")
            mask &= query_frame[field].astype(str).to_numpy() == str(expectation.get(field))

        matches = query_frame.loc[mask]
        key = {field : expectation.get(field) for field in query_key_fields}

        if (len(matches) != 1) :
            errors.append(f"Per-query fixture row {key}: expected 1 row, found {len(matches)}")
            query_checks.append({"key" : key, "passed" : False})
            continue

        row = matches.iloc[0]
        field_checks = []
        passed = True

        for field in query_exact_fields :
            actual_value = row.get(field)
            expected_value = expectation.get(field)
            field_passed = _fixture_exact_equal(actual_value, expected_value)
            field_checks.append({
                "field"    : field,
                "expected" : expected_value,
                "actual"   : None if _fixture_is_null(actual_value) else actual_value,
                "passed"   : field_passed,
            })

            if (not field_passed) :
                passed = False
                errors.append(
                    f"Per-query exact mismatch {key}/{field}: "
                    f"{actual_value!r} != {expected_value!r}"
                )

        for field in query_score_fields :
            actual_value = row.get(field)
            expected_value = expectation.get(field)
            difference = _fixture_numeric_difference(actual_value, expected_value)

            if (math.isfinite(difference)) :
                maximum_score_difference = max(maximum_score_difference, difference)

            field_passed = difference <= score_tolerance
            field_checks.append({
                "field"      : field,
                "expected"   : expected_value,
                "actual"     : None if _fixture_is_null(actual_value) else float(actual_value),
                "difference" : difference,
                "tolerance"  : score_tolerance,
                "passed"     : field_passed,
            })

            if (not field_passed) :
                passed = False
                errors.append(
                    f"Per-query score mismatch {key}/{field}: "
                    f"difference={difference:.9g} > {score_tolerance:.9g}"
                )

        query_checks.append({
            "key"          : key,
            "passed"       : passed,
            "field_checks" : field_checks,
        })

    for expectation in expected_metrics :
        mask = np.ones(len(metric_frame), dtype = bool)

        for field in aggregate_key_fields :
            if (field not in metric_frame.columns) :
                raise KeyError(f"Metrics are missing fixture key field: {field}")
            mask &= metric_frame[field].astype(str).to_numpy() == str(expectation.get(field))

        matches = metric_frame.loc[mask]
        key = {field : expectation.get(field) for field in aggregate_key_fields}

        if (len(matches) != 1) :
            errors.append(f"Aggregate fixture row {key}: expected 1 row, found {len(matches)}")
            metric_checks.append({"key" : key, "passed" : False})
            continue

        row = matches.iloc[0]
        field_checks = []
        passed = True

        for field in aggregate_exact_fields :
            actual_value = row.get(field)
            expected_value = expectation.get(field)
            field_passed = _fixture_exact_equal(actual_value, expected_value)
            field_checks.append({
                "field"    : field,
                "expected" : expected_value,
                "actual"   : None if _fixture_is_null(actual_value) else actual_value,
                "passed"   : field_passed,
            })

            if (not field_passed) :
                passed = False
                errors.append(
                    f"Aggregate exact mismatch {key}/{field}: "
                    f"{actual_value!r} != {expected_value!r}"
                )

        for field in aggregate_rank_fields :
            difference = _fixture_numeric_difference(row.get(field), expectation.get(field))

            if (math.isfinite(difference)) :
                maximum_rank_metric_difference = max(maximum_rank_metric_difference, difference)

            field_passed = difference <= rank_tolerance
            field_checks.append({
                "field"      : field,
                "expected"   : expectation.get(field),
                "actual"     : None if _fixture_is_null(row.get(field)) else float(row.get(field)),
                "difference" : difference,
                "tolerance"  : rank_tolerance,
                "passed"     : field_passed,
            })

            if (not field_passed) :
                passed = False
                errors.append(
                    f"Aggregate rank-metric mismatch {key}/{field}: "
                    f"difference={difference:.9g} > {rank_tolerance:.9g}"
                )

        for field in aggregate_score_fields :
            difference = _fixture_numeric_difference(row.get(field), expectation.get(field))

            if (math.isfinite(difference)) :
                maximum_score_metric_difference = max(maximum_score_metric_difference, difference)

            field_passed = difference <= score_tolerance
            field_checks.append({
                "field"      : field,
                "expected"   : expectation.get(field),
                "actual"     : None if _fixture_is_null(row.get(field)) else float(row.get(field)),
                "difference" : difference,
                "tolerance"  : score_tolerance,
                "passed"     : field_passed,
            })

            if (not field_passed) :
                passed = False
                errors.append(
                    f"Aggregate score-metric mismatch {key}/{field}: "
                    f"difference={difference:.9g} > {score_tolerance:.9g}"
                )

        metric_checks.append({
            "key"          : key,
            "passed"       : passed,
            "field_checks" : field_checks,
        })

    return {
        "passed"                          : not errors,
        "fixture_id"                      : fixture.get("fixture_id"),
        "expected_query_row_count"        : len(expected_queries),
        "passed_query_row_count"          : sum(item.get("passed", False) for item in query_checks),
        "expected_aggregate_row_count"    : len(expected_metrics),
        "passed_aggregate_row_count"      : sum(item.get("passed", False) for item in metric_checks),
        "maximum_query_score_difference"  : maximum_score_difference,
        "maximum_rank_metric_difference"  : maximum_rank_metric_difference,
        "maximum_score_metric_difference" : maximum_score_metric_difference,
        "score_tolerance"                 : score_tolerance,
        "rank_metric_tolerance"           : rank_tolerance,
        "query_checks"                    : query_checks,
        "aggregate_checks"                : metric_checks,
        "errors"                          : errors,
    }


# -----------------------------------------------------------------------------
# Stage 3 video and temporal evidence aggregation
# -----------------------------------------------------------------------------


@dataclass(frozen = True)
class VideoAggregationSpec :
    method_id : str
    kind : str
    k : int | None = None
    span : int | None = None

    def __post_init__(self) -> None :
        if (self.kind not in {"max", "topk_mean", "contiguous_mean"}) :
            raise ValueError(f"Unsupported video aggregation kind: {self.kind!r}")

        if (self.kind == "max" and (self.k is not None or self.span is not None)) :
            raise ValueError("max aggregation must not define k/span")

        if (self.kind == "topk_mean") :
            if (self.k is None or int(self.k) < 1) :
                raise ValueError("topk_mean requires k >= 1")
            if (self.span is not None) :
                raise ValueError("topk_mean must not define span")

        if (self.kind == "contiguous_mean") :
            if (self.span is None or int(self.span) < 1) :
                raise ValueError("contiguous_mean requires span >= 1")
            if (self.k is not None) :
                raise ValueError("contiguous_mean must not define k")


def build_video_window_groups(
    windows : pd.DataFrame,
) -> tuple[list[str], dict[str, np.ndarray]] :
    required = {"window_id", "video_id", "start_s", "end_s"}
    missing = sorted(required - set(windows.columns))

    if (missing) :
        raise KeyError(f"Windows are missing Stage 3 fields: {missing}")

    if (windows["window_id"].astype(str).duplicated().any()) :
        raise ValueError("Stage 3 physical window axis contains duplicate window IDs")

    starts = pd.to_numeric(windows["start_s"], errors = "coerce").to_numpy(dtype = float)
    ends   = pd.to_numeric(windows["end_s"], errors = "coerce").to_numpy(dtype = float)

    if (not np.isfinite(starts).all() or not np.isfinite(ends).all()) :
        raise ValueError("Stage 3 physical window axis contains invalid timestamps")

    if ((ends < starts).any()) :
        raise ValueError("Stage 3 physical window axis contains end_s < start_s")

    groups = {}
    working = windows.reset_index(drop = True).copy()
    working["_physical_index"] = np.arange(len(working), dtype = np.int64)
    working["_window_id"] = working["window_id"].astype(str)
    working["_video_id"]  = working["video_id"].astype(str)
    working["_start_s"]   = pd.to_numeric(working["start_s"], errors = "coerce")

    for video_id, group in working.groupby("_video_id", sort = True) :
        ordered = group.sort_values(
            ["_start_s", "_window_id"],
            ascending = [True, True],
            kind = "mergesort",
        )
        indices = ordered["_physical_index"].to_numpy(dtype = np.int64)

        if (len(indices) == 0) :
            raise ValueError(f"Video {video_id!r} has no physical windows")

        groups[str(video_id)] = indices

    return sorted(groups), groups


def _topk_support_indices(
    scores : np.ndarray,
    windows : pd.DataFrame,
    indices : np.ndarray,
    k : int,
) -> list[int] :
    count = min(int(k), len(indices))

    if (count <= 0) :
        return []

    frame = pd.DataFrame({
        "_index"    : indices,
        "score"     : np.asarray(scores, dtype = float)[indices],
        "start_s"   : pd.to_numeric(windows.iloc[indices]["start_s"], errors = "coerce").to_numpy(dtype = float),
        "window_id" : windows.iloc[indices]["window_id"].astype(str).to_numpy(),
    }).sort_values(
        ["score", "start_s", "window_id"],
        ascending = [False, True, True],
        kind = "mergesort",
    )

    return frame.head(count)["_index"].astype(int).tolist()


def _contiguous_support_indices(
    scores : np.ndarray,
    windows : pd.DataFrame,
    indices : np.ndarray,
    span : int,
) -> list[int] :
    use_span = min(int(span), len(indices))

    if (use_span <= 0) :
        return []

    values = np.asarray(scores, dtype = float)
    candidates = []

    for start in range(0, len(indices) - use_span + 1) :
        support = indices[start : start + use_span]
        score = float(values[support].mean(dtype = np.float64))
        first = int(support[0])
        candidates.append((
            -score,
            float(windows.iloc[first]["start_s"]),
            str(windows.iloc[first]["window_id"]),
            start,
            support,
        ))

    candidates.sort(key = lambda item : item[:4])
    return candidates[0][4].astype(int).tolist()


def aggregate_video_scores_for_query(
    window_scores : np.ndarray,
    windows : pd.DataFrame,
    specification : VideoAggregationSpec,
    video_ids : Sequence[str] | None = None,
    groups : dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, list[list[int]]] :
    values = np.asarray(window_scores, dtype = np.float64)

    if (values.ndim != 1 or values.shape[0] != len(windows)) :
        raise ValueError("Stage 3 query scores must match the physical window axis")

    if (not np.isfinite(values).all()) :
        raise ValueError("Stage 3 query scores contain NaN or infinite values")

    if (groups is None or video_ids is None) :
        resolved_video_ids, resolved_groups = build_video_window_groups(windows)
        video_ids = resolved_video_ids
        groups = resolved_groups
    else :
        video_ids = [str(value) for value in video_ids]

    video_scores = np.empty(len(video_ids), dtype = np.float64)
    support_rows = []

    for video_index, video_id in enumerate(video_ids) :
        indices = np.asarray(groups[str(video_id)], dtype = np.int64)

        if (specification.kind == "max") :
            support = _topk_support_indices(values, windows, indices, 1)
        elif (specification.kind == "topk_mean") :
            support = _topk_support_indices(values, windows, indices, int(specification.k))
        else :
            support = _contiguous_support_indices(values, windows, indices, int(specification.span))

        if (not support) :
            raise ValueError(f"Video {video_id!r} has no support windows")

        video_scores[video_index] = float(values[support].mean(dtype = np.float64))
        support_rows.append(support)

    return video_scores, support_rows


def aggregate_video_score_matrix(
    window_scores : np.ndarray,
    windows : pd.DataFrame,
    specification : VideoAggregationSpec,
) -> tuple[np.ndarray, list[str], list[list[list[int]]]] :
    values = np.asarray(window_scores)

    if (values.ndim != 2 or values.shape[1] != len(windows)) :
        raise ValueError("Stage 3 score matrix must be queries x physical windows")

    if (not np.isfinite(values).all()) :
        raise ValueError("Stage 3 score matrix contains NaN or infinite values")

    video_ids, groups = build_video_window_groups(windows)
    video_scores = np.empty((values.shape[0], len(video_ids)), dtype = np.float64)
    supports = []

    for query_index in range(values.shape[0]) :
        row_scores, row_supports = aggregate_video_scores_for_query(
            values[query_index],
            windows,
            specification,
            video_ids = video_ids,
            groups = groups,
        )
        video_scores[query_index] = row_scores
        supports.append(row_supports)

    return video_scores, video_ids, supports


def peak_support_statistics(
    window_scores : np.ndarray,
    windows : pd.DataFrame,
    indices : Sequence[int],
) -> dict[str, Any] :
    physical = np.asarray(indices, dtype = np.int64)

    if (len(physical) == 0) :
        raise ValueError("Peak statistics require at least one physical window")

    values = np.asarray(window_scores, dtype = np.float64)
    ordered = _topk_support_indices(values, windows, physical, min(3, len(physical)))
    padded = ordered + [None] * (3 - len(ordered))
    peak = int(padded[0])

    local_position = int(np.flatnonzero(physical == peak)[0])
    left_index  = int(physical[local_position - 1]) if local_position > 0 else None
    right_index = int(physical[local_position + 1]) if local_position + 1 < len(physical) else None

    neighbor_scores = [
        float(values[index])
        for index in [left_index, right_index]
        if index is not None
    ]
    strongest_neighbor = max(neighbor_scores) if neighbor_scores else None
    isolation_gap = (
        float(values[peak]) - strongest_neighbor
        if strongest_neighbor is not None
        else None
    )

    return {
        "window_count"             : int(len(physical)),
        "peak_score"               : float(values[peak]),
        "second_score"             : float(values[padded[1]]) if padded[1] is not None else None,
        "third_score"              : float(values[padded[2]]) if padded[2] is not None else None,
        "peak_window_id"           : str(windows.iloc[peak]["window_id"]),
        "peak_start_s"             : float(windows.iloc[peak]["start_s"]),
        "peak_end_s"               : float(windows.iloc[peak]["end_s"]),
        "left_neighbor_score"      : float(values[left_index]) if left_index is not None else None,
        "right_neighbor_score"     : float(values[right_index]) if right_index is not None else None,
        "strongest_neighbor_score" : strongest_neighbor,
        "isolation_gap"            : isolation_gap,
    }


def evaluate_stage3_video_scores(
    video_scores : np.ndarray,
    video_ids : Sequence[str],
    supports : list[list[list[int]]],
    window_scores : np.ndarray,
    queries : pd.DataFrame,
    windows : pd.DataFrame,
    source_query_results : pd.DataFrame,
    source_id : str,
    source_method_id : str,
    model_id : str,
    view : str,
    aggregation_id : str,
    query_set : str,
    corpus_id : str,
    tie_tolerance : float = 1e-12,
) -> tuple[pd.DataFrame, pd.DataFrame] :
    scores = np.asarray(video_scores, dtype = np.float64)
    source_windows = np.asarray(window_scores)
    video_ids = [str(value) for value in video_ids]

    if (scores.shape != (len(queries), len(video_ids))) :
        raise ValueError("Stage 3 video score matrix shape mismatch")

    if (source_windows.shape != (len(queries), len(windows))) :
        raise ValueError("Stage 3 source window score matrix shape mismatch")

    if (not np.isfinite(scores).all() or not np.isfinite(source_windows).all()) :
        raise ValueError("Stage 3 scores contain NaN or infinite values")

    if (len(supports) != len(queries)) :
        raise ValueError("Stage 3 support metadata query count mismatch")

    video_id_to_index = {video_id : index for index, video_id in enumerate(video_ids)}

    if (len(video_id_to_index) != len(video_ids)) :
        raise ValueError("Stage 3 video axis contains duplicates")

    physical_video_ids = windows["video_id"].astype(str).to_numpy()
    _, groups = build_video_window_groups(windows)
    source_lookup = source_query_results.copy()

    required_source = {
        "query_id", "first_relevant_rank", "story_recall_at_1", "story_recall_at_3",
        "story_recall_at_5", "story_recall_at_10", "story_rr", "best_relevant_score",
        "best_irrelevant_score", "story_score_margin", "top_story_window_id",
        "top_story_start_s", "top_story_end_s", "top_story_score", "eligible_window_count",
        "positive_window_count", "positive_window_fraction", "positive_video_count",
        "correct_video_positive_windows", "query_coverage", "zero_evidence",
    }
    missing = sorted(required_source - set(source_lookup.columns))

    if (missing) :
        raise KeyError(f"Source query results are missing Stage 3 story fields: {missing}")

    source_lookup["query_id"] = source_lookup["query_id"].astype(str)

    if (source_lookup["query_id"].duplicated().any()) :
        raise ValueError("Stage 3 source query results contain duplicate query IDs")

    source_lookup = source_lookup.set_index("query_id", drop = False)
    query_rows = []
    diagnostics = []

    for query_index, (_, query) in enumerate(queries.reset_index(drop = True).iterrows()) :
        query_id = str(query["query_id"])

        if (query_id not in source_lookup.index) :
            raise ValueError(f"Stage 3 source query result missing query {query_id!r}")

        source = source_lookup.loc[query_id]
        row_scores = scores[query_index]
        metric_ranks = worst_tied_ranks_array(row_scores, tolerance = tie_tolerance)
        display_order = sorted(
            range(len(video_ids)),
            key = lambda index : (-row_scores[index], video_ids[index]),
        )

        correct_video = str(query["video_id"])

        if (correct_video not in video_id_to_index) :
            raise ValueError(f"Correct video {correct_video!r} is absent from the Stage 3 video axis")

        correct_index = video_id_to_index[correct_video]
        video_rank = int(metric_ranks[correct_index])
        wrong_indices = [index for index in display_order if index != correct_index]
        best_wrong_index = wrong_indices[0] if wrong_indices else None
        correct_score = float(row_scores[correct_index])
        best_wrong_score = float(row_scores[best_wrong_index]) if best_wrong_index is not None else None
        margin = correct_score - best_wrong_score if best_wrong_score is not None else None
        top_index = display_order[0]

        query_rows.append({
            "query_set"                  : query_set,
            "corpus_id"                  : corpus_id,
            "source_id"                  : source_id,
            "source_method_id"           : source_method_id,
            "method_id"                  : aggregation_id,
            "aggregation_id"             : aggregation_id,
            "model_id"                   : model_id,
            "view"                       : view,
            "query_id"                   : query_id,
            "query_text"                 : query["query_text"],
            "query_category"             : query.get("query_category", "other"),
            "task_type"                  : query.get("task_type", "KIS"),
            "difficulty"                 : query.get("difficulty", "unknown"),
            "evaluation_split"           : query.get("evaluation_split", "unspecified"),
            "answer_text"                : query.get("answer_text"),
            "correct_video"              : correct_video,
            "frame_id"                   : query.get("frame_id"),
            "answer_time_s"              : query["answer_time_s"],
            "first_relevant_rank"        : source["first_relevant_rank"],
            "story_recall_at_1"          : int(source["story_recall_at_1"]),
            "story_recall_at_3"          : int(source["story_recall_at_3"]),
            "story_recall_at_5"          : int(source["story_recall_at_5"]),
            "story_recall_at_10"         : int(source["story_recall_at_10"]),
            "story_rr"                   : float(source["story_rr"]),
            "best_relevant_score"        : source["best_relevant_score"],
            "best_irrelevant_score"      : source["best_irrelevant_score"],
            "story_score_margin"         : source["story_score_margin"],
            "video_rank"                 : video_rank,
            "video_recall_at_1"          : int(video_rank <= 1),
            "video_recall_at_3"          : int(video_rank <= 3),
            "video_recall_at_5"          : int(video_rank <= 5),
            "video_recall_at_10"         : int(video_rank <= 10),
            "video_recall_at_20"         : int(video_rank <= 20),
            "video_rr"                   : 1.0 / video_rank,
            "correct_video_score"        : correct_score,
            "best_wrong_video_id"        : video_ids[best_wrong_index] if best_wrong_index is not None else None,
            "best_wrong_video_score"     : best_wrong_score,
            "video_score_margin"         : margin,
            "top_story_window_id"        : source["top_story_window_id"],
            "top_story_start_s"          : source["top_story_start_s"],
            "top_story_end_s"            : source["top_story_end_s"],
            "top_story_score"            : source["top_story_score"],
            "top_video_id"               : video_ids[top_index],
            "top_video_score"            : float(row_scores[top_index]),
            "physical_window_count"      : len(windows),
            "eligible_window_count"      : int(source["eligible_window_count"]),
            "positive_window_count"      : int(source["positive_window_count"]),
            "positive_window_fraction"   : float(source["positive_window_fraction"]),
            "positive_video_count"       : int(source["positive_video_count"]),
            "correct_video_positive_windows" : int(source["correct_video_positive_windows"]),
            "query_coverage"             : source["query_coverage"],
            "zero_evidence"              : bool(source["zero_evidence"]),
        })

        correct_physical = groups[correct_video]
        correct_stats = peak_support_statistics(
            source_windows[query_index],
            windows,
            correct_physical,
        )
        wrong_video = video_ids[best_wrong_index] if best_wrong_index is not None else None
        wrong_stats = (
            peak_support_statistics(source_windows[query_index], windows, groups[wrong_video])
            if wrong_video is not None
            else {}
        )

        diagnostics.append({
            "query_set"                        : query_set,
            "corpus_id"                        : corpus_id,
            "source_id"                        : source_id,
            "source_method_id"                 : source_method_id,
            "aggregation_id"                   : aggregation_id,
            "model_id"                         : model_id,
            "view"                             : view,
            "query_id"                         : query_id,
            "correct_video_id"                 : correct_video,
            "correct_video_rank"               : video_rank,
            "correct_video_window_count"       : correct_stats["window_count"],
            "correct_video_peak_score"         : correct_stats["peak_score"],
            "correct_video_second_score"       : correct_stats["second_score"],
            "correct_video_third_score"        : correct_stats["third_score"],
            "correct_video_peak_window_id"     : correct_stats["peak_window_id"],
            "correct_video_peak_start_s"       : correct_stats["peak_start_s"],
            "correct_video_peak_end_s"         : correct_stats["peak_end_s"],
            "correct_video_left_neighbor_score": correct_stats["left_neighbor_score"],
            "correct_video_right_neighbor_score": correct_stats["right_neighbor_score"],
            "correct_video_isolation_gap"      : correct_stats["isolation_gap"],
            "correct_video_support_window_ids" : "|".join(
                str(windows.iloc[index]["window_id"])
                for index in supports[query_index][correct_index]
            ),
            "best_wrong_video_id"              : wrong_video,
            "best_wrong_video_window_count"    : wrong_stats.get("window_count"),
            "best_wrong_video_peak_score"      : wrong_stats.get("peak_score"),
            "best_wrong_video_second_score"    : wrong_stats.get("second_score"),
            "best_wrong_video_third_score"     : wrong_stats.get("third_score"),
            "best_wrong_video_peak_window_id"  : wrong_stats.get("peak_window_id"),
            "best_wrong_video_peak_start_s"    : wrong_stats.get("peak_start_s"),
            "best_wrong_video_peak_end_s"      : wrong_stats.get("peak_end_s"),
            "best_wrong_video_left_neighbor_score" : wrong_stats.get("left_neighbor_score"),
            "best_wrong_video_right_neighbor_score": wrong_stats.get("right_neighbor_score"),
            "best_wrong_video_isolation_gap"   : wrong_stats.get("isolation_gap"),
            "best_wrong_video_support_window_ids" : (
                "|".join(
                    str(windows.iloc[index]["window_id"])
                    for index in supports[query_index][best_wrong_index]
                )
                if best_wrong_index is not None
                else None
            ),
            "correct_video_aggregated_score"   : correct_score,
            "best_wrong_video_aggregated_score": best_wrong_score,
            "margin"                           : margin,
        })

    return pd.DataFrame(query_rows), pd.DataFrame(diagnostics)


def summarize_stage3_metrics(query_results : pd.DataFrame) -> pd.DataFrame :
    if (query_results.empty) :
        return pd.DataFrame()

    group_columns = [
        "query_set", "corpus_id", "source_id", "source_method_id",
        "aggregation_id", "model_id", "view",
    ]
    rows = []

    for keys, group in query_results.groupby(group_columns, dropna = False, sort = True) :
        row = dict(zip(group_columns, keys))
        first_ranks = pd.to_numeric(group["first_relevant_rank"], errors = "coerce")
        video_ranks = pd.to_numeric(group["video_rank"], errors = "coerce")
        row.update({
            "query_count"               : len(group),
            "story_recall_at_1"         : float(group["story_recall_at_1"].mean()),
            "story_recall_at_3"         : float(group["story_recall_at_3"].mean()),
            "story_recall_at_5"         : float(group["story_recall_at_5"].mean()),
            "story_recall_at_10"        : float(group["story_recall_at_10"].mean()),
            "story_mrr"                 : float(group["story_rr"].mean()),
            "story_first_rank_mean"     : float(first_ranks.mean()) if first_ranks.notna().any() else None,
            "story_first_rank_median"   : float(first_ranks.median()) if first_ranks.notna().any() else None,
            "video_recall_at_1"         : float(group["video_recall_at_1"].mean()),
            "video_recall_at_3"         : float(group["video_recall_at_3"].mean()),
            "video_recall_at_5"         : float(group["video_recall_at_5"].mean()),
            "video_recall_at_10"        : float(group["video_recall_at_10"].mean()),
            "video_recall_at_20"        : float(group["video_recall_at_20"].mean()),
            "video_mrr"                 : float(group["video_rr"].mean()),
            "video_rank_mean"           : float(video_ranks.mean()) if video_ranks.notna().any() else None,
            "video_rank_median"         : float(video_ranks.median()) if video_ranks.notna().any() else None,
            "story_score_margin_mean"   : float(pd.to_numeric(group["story_score_margin"], errors = "coerce").mean()),
            "video_score_margin_mean"   : float(pd.to_numeric(group["video_score_margin"], errors = "coerce").mean()),
            "zero_evidence_query_count" : int(group["zero_evidence"].sum()),
            "query_coverage_mean"       : float(pd.to_numeric(group["query_coverage"], errors = "coerce").mean()),
        })
        rows.append(row)

    return pd.DataFrame(rows)


def compare_stage3_aggregations(
    query_results : pd.DataFrame,
    reference_aggregation : str = "P0_max",
) -> pd.DataFrame :
    if (query_results.empty) :
        return pd.DataFrame()

    key_columns = ["source_id", "model_id", "view", "query_id"]
    reference = query_results[
        query_results["aggregation_id"] == reference_aggregation
    ][key_columns + ["video_rank", "video_rr", "top_video_id"]].rename(columns={
        "video_rank"   : "reference_video_rank",
        "video_rr"     : "reference_video_rr",
        "top_video_id" : "reference_top_video_id",
    })

    rows = []

    for aggregation_id in sorted(set(query_results["aggregation_id"]) - {reference_aggregation}) :
        candidate = query_results[
            query_results["aggregation_id"] == aggregation_id
        ].merge(reference, on = key_columns, how = "inner")

        for _, item in candidate.iterrows() :
            rows.append({
                "source_id"                 : item["source_id"],
                "source_method_id"          : item["source_method_id"],
                "aggregation_id"            : aggregation_id,
                "reference_aggregation_id"  : reference_aggregation,
                "model_id"                  : item["model_id"],
                "view"                      : item["view"],
                "query_id"                  : item["query_id"],
                "reference_video_rank"      : int(item["reference_video_rank"]),
                "candidate_video_rank"      : int(item["video_rank"]),
                "reference_video_rr"        : float(item["reference_video_rr"]),
                "candidate_video_rr"        : float(item["video_rr"]),
                "video_rank_delta"          : int(item["video_rank"] - item["reference_video_rank"]),
                "video_rr_delta"            : float(item["video_rr"] - item["reference_video_rr"]),
                "outcome"                   : rank_relation(item["video_rank"], item["reference_video_rank"]),
                "top_video_changed"         : bool(item["top_video_id"] != item["reference_top_video_id"]),
            })

    return pd.DataFrame(rows)


def summarize_stage3_aggregations(
    query_results : pd.DataFrame,
    source_labels : dict[str, str],
    reference_aggregation : str = "P0_max",
    bootstrap_samples : int = 10000,
    confidence : float = 0.90,
    seed : int = 20260811,
) -> pd.DataFrame :
    if (query_results.empty) :
        return pd.DataFrame()

    per_source_query = (
        query_results.groupby(
            ["source_id", "aggregation_id", "query_id"],
            as_index = False,
        )
        .agg(video_rr = ("video_rr", "mean"))
    )
    aggregations = sorted(query_results["aggregation_id"].unique().tolist())
    rows = []

    for aggregation_id in aggregations :
        row = {"aggregation_id" : aggregation_id}

        for source_id, label in source_labels.items() :
            candidate = per_source_query[
                (per_source_query["source_id"] == source_id)
                & (per_source_query["aggregation_id"] == aggregation_id)
            ][["query_id", "video_rr"]]
            reference = per_source_query[
                (per_source_query["source_id"] == source_id)
                & (per_source_query["aggregation_id"] == reference_aggregation)
            ][["query_id", "video_rr"]].rename(columns={"video_rr" : "reference_rr"})
            merged = candidate.merge(reference, on = "query_id", how = "inner")
            delta = (merged["video_rr"] - merged["reference_rr"]).to_numpy(dtype = float)
            low, high = _bootstrap_interval(delta, confidence, bootstrap_samples, seed)

            row.update({
                f"{label}_video_rr_mean"              : float(merged["video_rr"].mean()) if len(merged) else None,
                f"{label}_delta_mean_vs_P0"           : float(delta.mean()) if len(delta) else None,
                f"{label}_delta_median_vs_P0"         : float(np.median(delta)) if len(delta) else None,
                f"{label}_better"                     : int((delta > 0).sum()),
                f"{label}_tie"                        : int((delta == 0).sum()),
                f"{label}_worse"                      : int((delta < 0).sum()),
                f"{label}_bootstrap_90_low"           : low,
                f"{label}_bootstrap_90_high"          : high,
            })

        cross = (
            per_source_query[
                per_source_query["aggregation_id"] == aggregation_id
            ]
            .groupby("query_id", as_index = False)
            .agg(video_rr = ("video_rr", "mean"))
        )
        cross_reference = (
            per_source_query[
                per_source_query["aggregation_id"] == reference_aggregation
            ]
            .groupby("query_id", as_index = False)
            .agg(reference_rr = ("video_rr", "mean"))
        )
        cross_merged = cross.merge(cross_reference, on = "query_id", how = "inner")
        cross_delta = (cross_merged["video_rr"] - cross_merged["reference_rr"]).to_numpy(dtype = float)
        low, high = _bootstrap_interval(cross_delta, confidence, bootstrap_samples, seed)
        row.update({
            "cross_source_video_rr_mean"      : float(cross_merged["video_rr"].mean()) if len(cross_merged) else None,
            "cross_source_delta_mean_vs_P0"   : float(cross_delta.mean()) if len(cross_delta) else None,
            "cross_source_delta_median_vs_P0" : float(np.median(cross_delta)) if len(cross_delta) else None,
            "cross_source_better"             : int((cross_delta > 0).sum()),
            "cross_source_tie"                : int((cross_delta == 0).sum()),
            "cross_source_worse"              : int((cross_delta < 0).sum()),
            "cross_source_bootstrap_90_low"   : low,
            "cross_source_bootstrap_90_high"  : high,
        })
        rows.append(row)

    return pd.DataFrame(rows)


def validate_stage3_p0_reproduction(
    stage3_query_results : pd.DataFrame,
    source_query_results : dict[str, pd.DataFrame],
    tolerance : float = 1e-12,
) -> dict[str, Any] :
    errors = []
    checks = []
    exact_fields = ["video_rank", "top_video_id"]
    score_fields = [
        "correct_video_score",
        "best_wrong_video_score",
        "video_score_margin",
    ]

    p0 = stage3_query_results[stage3_query_results["aggregation_id"] == "P0_max"]

    for source_id, reference in source_query_results.items() :
        candidate = p0[p0["source_id"] == source_id]
        keys = ["model_id", "view", "query_id"]
        merged = candidate.merge(
            reference[keys + exact_fields + score_fields],
            on = keys,
            how = "outer",
            suffixes = ("", "_reference"),
            indicator = True,
        )

        for _, item in merged.iterrows() :
            key = {field : item.get(field) for field in keys}
            passed = item["_merge"] == "both"
            field_checks = []

            if (not passed) :
                errors.append(f"{source_id}: missing P0/source row {key}")
                checks.append({"source_id" : source_id, "key" : key, "passed" : False})
                continue

            for field in exact_fields :
                same = _fixture_exact_equal(item[field], item[f"{field}_reference"])
                field_checks.append({"field" : field, "passed" : same})

                if (not same) :
                    passed = False
                    errors.append(f"{source_id}/{key}/{field}: P0 reproduction mismatch")

            for field in score_fields :
                difference = _fixture_numeric_difference(
                    item[field],
                    item[f"{field}_reference"],
                )
                same = difference <= float(tolerance)
                field_checks.append({
                    "field" : field,
                    "difference" : difference,
                    "tolerance" : float(tolerance),
                    "passed" : same,
                })

                if (not same) :
                    passed = False
                    errors.append(
                        f"{source_id}/{key}/{field}: "
                        f"difference={difference:.9g} > {float(tolerance):.9g}"
                    )

            checks.append({
                "source_id" : source_id,
                "key" : key,
                "passed" : passed,
                "fields" : field_checks,
            })

    return {
        "passed" : not errors,
        "check_count" : len(checks),
        "failed_count" : sum(not item["passed"] for item in checks),
        "errors" : errors,
        "checks" : checks,
    }


def validate_stage3_story_invariance(
    query_results : pd.DataFrame,
) -> dict[str, Any] :
    errors = []
    checked = 0
    story_fields = [
        "first_relevant_rank",
        "story_recall_at_1",
        "story_recall_at_3",
        "story_recall_at_5",
        "story_recall_at_10",
        "story_rr",
        "top_story_window_id",
    ]

    group_fields = ["source_id", "model_id", "view", "query_id"]

    for keys, group in query_results.groupby(group_fields, sort = True, dropna = False) :
        reference = group.sort_values("aggregation_id", kind = "mergesort").iloc[0]

        for _, item in group.iterrows() :
            checked += 1

            for field in story_fields :
                if (not _fixture_exact_equal(item[field], reference[field])) :
                    errors.append(
                        f"Story invariance mismatch {dict(zip(group_fields, keys))}/{field}"
                    )

    return {
        "passed" : not errors,
        "check_count" : checked,
        "errors" : errors,
    }
# -----------------------------------------------------------------------------
# Stage 4 sparse + dense hybrid retrieval
# -----------------------------------------------------------------------------


def _stage4_reference_frame(
    query_results : pd.DataFrame,
    method_id : str,
) -> pd.DataFrame :
    columns = [
        "model_id",
        "view",
        "query_id",
        "first_relevant_rank",
        "story_rr",
        "video_rank",
        "video_rr",
        "top_story_window_id",
        "top_video_id",
        "correct_video_score",
        "video_score_margin",
    ]
    frame = query_results[query_results["method_id"] == method_id][columns].copy()

    if (frame.empty) :
        raise ValueError(f"Stage 4 reference method is missing: {method_id}")

    keys = ["model_id", "view", "query_id"]

    if (frame.duplicated(keys).any()) :
        raise ValueError(f"Stage 4 reference method contains duplicate rows: {method_id}")

    return frame


def compare_stage4_methods(
    query_results : pd.DataFrame,
    sparse_reference : str = "H0_sparse_only",
    dense_reference : str = "H1_dense_only",
    minimum_rank_drop : int = 10,
    result_rank_above : int = 10,
    flag_reference_rank_1_candidate_above : int = 20,
) -> pd.DataFrame :
    if (query_results.empty) :
        return pd.DataFrame()

    keys = ["model_id", "view", "query_id"]
    references = {
        sparse_reference : _stage4_reference_frame(query_results, sparse_reference),
        dense_reference  : _stage4_reference_frame(query_results, dense_reference),
    }
    rows = []

    for reference_method, reference in references.items() :
        renamed = reference.rename(columns={
            column : f"reference_{column}"
            for column in reference.columns
            if column not in keys
        })

        for method_id in sorted(query_results["method_id"].astype(str).unique()) :
            if (method_id == reference_method) :
                continue

            candidate = query_results[
                query_results["method_id"].astype(str) == method_id
            ].copy()
            merged = candidate.merge(renamed, on = keys, how = "inner")

            for _, item in merged.iterrows() :
                video_rank_delta = int(
                    item["video_rank"] - item["reference_video_rank"]
                )
                story_rank = (
                    int(item["first_relevant_rank"])
                    if pd.notna(item["first_relevant_rank"])
                    else None
                )
                reference_story_rank = (
                    int(item["reference_first_relevant_rank"])
                    if pd.notna(item["reference_first_relevant_rank"])
                    else None
                )

                rows.append({
                    "reference_method" : reference_method,
                    "candidate_method" : method_id,
                    "model_id" : item["model_id"],
                    "view" : item["view"],
                    "query_id" : item["query_id"],
                    "reference_story_rank" : reference_story_rank,
                    "candidate_story_rank" : story_rank,
                    "story_relation" : rank_relation(
                        item["first_relevant_rank"],
                        item["reference_first_relevant_rank"],
                    ),
                    "story_rr_delta" : float(
                        item["story_rr"] - item["reference_story_rr"]
                    ),
                    "reference_video_rank" : int(item["reference_video_rank"]),
                    "candidate_video_rank" : int(item["video_rank"]),
                    "video_rank_delta" : video_rank_delta,
                    "video_relation" : rank_relation(
                        item["video_rank"],
                        item["reference_video_rank"],
                    ),
                    "video_rr_delta" : float(
                        item["video_rr"] - item["reference_video_rr"]
                    ),
                    "reference_top_story_window_id" : item["reference_top_story_window_id"],
                    "candidate_top_story_window_id" : item["top_story_window_id"],
                    "top_story_changed" : bool(
                        item["top_story_window_id"]
                        != item["reference_top_story_window_id"]
                    ),
                    "reference_top_video_id" : item["reference_top_video_id"],
                    "candidate_top_video_id" : item["top_video_id"],
                    "top_video_changed" : bool(
                        item["top_video_id"] != item["reference_top_video_id"]
                    ),
                    "large_regression" : bool(
                        video_rank_delta >= int(minimum_rank_drop)
                        and int(item["video_rank"]) > int(result_rank_above)
                    ),
                    "top1_catastrophe" : bool(
                        int(item["reference_video_rank"]) == 1
                        and int(item["video_rank"])
                        > int(flag_reference_rank_1_candidate_above)
                    ),
                })

    return pd.DataFrame(rows)


def summarize_stage4_methods(
    query_results : pd.DataFrame,
    sparse_reference : str = "H0_sparse_only",
    dense_reference : str = "H1_dense_only",
    bootstrap_samples : int = 10000,
    confidence : float = 0.90,
    seed : int = 20260811,
) -> pd.DataFrame :
    if (query_results.empty) :
        return pd.DataFrame()

    per_query = (
        query_results.groupby(["method_id", "query_id"], as_index = False)
        .agg(
            video_rr = ("video_rr", "mean"),
            story_rr = ("story_rr", "mean"),
            video_recall_at_1 = ("video_recall_at_1", "mean"),
            video_recall_at_5 = ("video_recall_at_5", "mean"),
            video_recall_at_10 = ("video_recall_at_10", "mean"),
            story_recall_at_1 = ("story_recall_at_1", "mean"),
        )
    )

    references = {}

    for method_id, prefix in [
        (sparse_reference, "sparse"),
        (dense_reference, "dense"),
    ] :
        frame = per_query[per_query["method_id"] == method_id][
            ["query_id", "video_rr", "story_rr"]
        ].rename(columns={
            "video_rr" : f"{prefix}_reference_video_rr",
            "story_rr" : f"{prefix}_reference_story_rr",
        })

        if (frame.empty) :
            raise ValueError(f"Stage 4 summary reference is missing: {method_id}")

        references[prefix] = frame

    rows = []

    for method_id, group in per_query.groupby("method_id", sort = True) :
        row = {
            "method_id" : method_id,
            "query_count" : len(group),
            "composite_video_mrr" : float(group["video_rr"].mean()),
            "composite_video_recall_at_1" : float(group["video_recall_at_1"].mean()),
            "composite_video_recall_at_5" : float(group["video_recall_at_5"].mean()),
            "composite_video_recall_at_10" : float(group["video_recall_at_10"].mean()),
            "composite_story_mrr" : float(group["story_rr"].mean()),
            "composite_story_recall_at_1" : float(group["story_recall_at_1"].mean()),
        }

        for prefix, reference in references.items() :
            merged = group.merge(reference, on = "query_id", how = "inner")
            video_delta = (
                merged["video_rr"] - merged[f"{prefix}_reference_video_rr"]
            ).to_numpy(dtype = float)
            story_delta = (
                merged["story_rr"] - merged[f"{prefix}_reference_story_rr"]
            ).to_numpy(dtype = float)
            low, high = _bootstrap_interval(
                video_delta,
                confidence,
                bootstrap_samples,
                seed,
            )

            row.update({
                f"video_rr_delta_mean_vs_{prefix}" : float(video_delta.mean()),
                f"video_rr_delta_median_vs_{prefix}" : float(np.median(video_delta)),
                f"story_rr_delta_mean_vs_{prefix}" : float(story_delta.mean()),
                f"video_better_vs_{prefix}" : int((video_delta > 0).sum()),
                f"video_tie_vs_{prefix}" : int((video_delta == 0).sum()),
                f"video_worse_vs_{prefix}" : int((video_delta < 0).sum()),
                f"video_rr_bootstrap_90_low_vs_{prefix}" : low,
                f"video_rr_bootstrap_90_high_vs_{prefix}" : high,
            })

        rows.append(row)

    return pd.DataFrame(rows)


def build_stage4_fusion_diagnostics(
    query_results : pd.DataFrame,
    normalization_stats : pd.DataFrame | None = None,
    sparse_reference : str = "H0_sparse_only",
    dense_reference : str = "H1_dense_only",
    minimum_rank_drop : int = 10,
    result_rank_above : int = 10,
    flag_reference_rank_1_candidate_above : int = 20,
) -> pd.DataFrame :
    if (query_results.empty) :
        return pd.DataFrame()

    keys = ["model_id", "view", "query_id"]
    fields = [
        "first_relevant_rank",
        "story_rr",
        "video_rank",
        "video_rr",
        "top_video_id",
        "correct_video_score",
        "video_score_margin",
    ]
    sparse = query_results[
        query_results["method_id"] == sparse_reference
    ][keys + fields].rename(columns={
        field : f"sparse_{field}"
        for field in fields
    })
    dense = query_results[
        query_results["method_id"] == dense_reference
    ][keys + fields].rename(columns={
        field : f"dense_{field}"
        for field in fields
    })
    base = sparse.merge(dense, on = keys, how = "inner")
    rows = []

    hybrid_methods = sorted(
        set(query_results["method_id"].astype(str))
        - {sparse_reference, dense_reference}
    )

    for method_id in hybrid_methods :
        hybrid = query_results[
            query_results["method_id"] == method_id
        ][keys + fields].rename(columns={
            field : f"hybrid_{field}"
            for field in fields
        })
        merged = base.merge(hybrid, on = keys, how = "inner")

        for _, item in merged.iterrows() :
            sparse_rank = int(item["sparse_video_rank"])
            dense_rank = int(item["dense_video_rank"])
            hybrid_rank = int(item["hybrid_video_rank"])
            delta_sparse = hybrid_rank - sparse_rank
            delta_dense = hybrid_rank - dense_rank

            rows.append({
                "method_id" : method_id,
                "model_id" : item["model_id"],
                "view" : item["view"],
                "query_id" : item["query_id"],
                "sparse_story_rank" : item["sparse_first_relevant_rank"],
                "dense_story_rank" : item["dense_first_relevant_rank"],
                "hybrid_story_rank" : item["hybrid_first_relevant_rank"],
                "sparse_video_rank" : sparse_rank,
                "dense_video_rank" : dense_rank,
                "hybrid_video_rank" : hybrid_rank,
                "sparse_video_rr" : float(item["sparse_video_rr"]),
                "dense_video_rr" : float(item["dense_video_rr"]),
                "hybrid_video_rr" : float(item["hybrid_video_rr"]),
                "video_rank_delta_vs_sparse" : delta_sparse,
                "video_rank_delta_vs_dense" : delta_dense,
                "video_rr_delta_vs_sparse" : float(
                    item["hybrid_video_rr"] - item["sparse_video_rr"]
                ),
                "video_rr_delta_vs_dense" : float(
                    item["hybrid_video_rr"] - item["dense_video_rr"]
                ),
                "sparse_top_video_id" : item["sparse_top_video_id"],
                "dense_top_video_id" : item["dense_top_video_id"],
                "hybrid_top_video_id" : item["hybrid_top_video_id"],
                "sparse_correct_video_score" : item["sparse_correct_video_score"],
                "dense_correct_video_score" : item["dense_correct_video_score"],
                "hybrid_correct_video_score" : item["hybrid_correct_video_score"],
                "sparse_video_margin" : item["sparse_video_score_margin"],
                "dense_video_margin" : item["dense_video_score_margin"],
                "hybrid_video_margin" : item["hybrid_video_score_margin"],
                "sparse_dense_top1_agree" : bool(
                    item["sparse_top_video_id"] == item["dense_top_video_id"]
                ),
                "hybrid_beats_both" : bool(
                    hybrid_rank < sparse_rank and hybrid_rank < dense_rank
                ),
                "hybrid_worse_than_both" : bool(
                    hybrid_rank > sparse_rank and hybrid_rank > dense_rank
                ),
                "large_regression_vs_sparse" : bool(
                    delta_sparse >= int(minimum_rank_drop)
                    and hybrid_rank > int(result_rank_above)
                ),
                "large_regression_vs_dense" : bool(
                    delta_dense >= int(minimum_rank_drop)
                    and hybrid_rank > int(result_rank_above)
                ),
                "top1_catastrophe_vs_sparse" : bool(
                    sparse_rank == 1
                    and hybrid_rank > int(flag_reference_rank_1_candidate_above)
                ),
                "top1_catastrophe_vs_dense" : bool(
                    dense_rank == 1
                    and hybrid_rank > int(flag_reference_rank_1_candidate_above)
                ),
            })

    result = pd.DataFrame(rows)

    if (
        normalization_stats is not None
        and not normalization_stats.empty
        and not result.empty
    ) :
        result = result.merge(
            normalization_stats,
            on = ["method_id", "model_id", "view", "query_id"],
            how = "left",
        )

    return result


def validate_stage4_control_reproduction(
    stage4_query_results : pd.DataFrame,
    source_query_results : dict[str, pd.DataFrame],
    control_map : dict[str, str],
    tolerance : float = 1e-12,
) -> dict[str, Any] :
    errors = []
    checks = []
    keys = ["model_id", "view", "query_id"]
    exact_fields = [
        "first_relevant_rank",
        "video_rank",
        "top_story_window_id",
        "top_video_id",
        "story_recall_at_1",
        "story_recall_at_3",
        "story_recall_at_5",
        "story_recall_at_10",
        "video_recall_at_1",
        "video_recall_at_3",
        "video_recall_at_5",
        "video_recall_at_10",
        "video_recall_at_20",
    ]
    score_fields = [
        "story_rr",
        "video_rr",
        "best_relevant_score",
        "best_irrelevant_score",
        "story_score_margin",
        "correct_video_score",
        "best_wrong_video_score",
        "video_score_margin",
        "top_story_score",
        "top_video_score",
    ]

    for control_method, source_id in control_map.items() :
        if (source_id not in source_query_results) :
            raise KeyError(f"Stage 4 source results missing {source_id!r}")

        candidate = stage4_query_results[
            stage4_query_results["method_id"] == control_method
        ]
        reference = source_query_results[source_id]
        merged = candidate.merge(
            reference[keys + exact_fields + score_fields],
            on = keys,
            how = "outer",
            suffixes = ("", "_reference"),
            indicator = True,
        )

        for _, item in merged.iterrows() :
            key = {field : item.get(field) for field in keys}
            passed = item["_merge"] == "both"
            fields = []

            if (not passed) :
                errors.append(f"{control_method}: missing control/source row {key}")
                checks.append({
                    "method_id" : control_method,
                    "key" : key,
                    "passed" : False,
                })
                continue

            for field in exact_fields :
                same = _fixture_exact_equal(
                    item[field],
                    item[f"{field}_reference"],
                )
                fields.append({"field" : field, "passed" : same})

                if (not same) :
                    passed = False
                    errors.append(
                        f"{control_method}/{key}/{field}: control reproduction mismatch"
                    )

            for field in score_fields :
                difference = _fixture_numeric_difference(
                    item[field],
                    item[f"{field}_reference"],
                )
                same = difference <= float(tolerance)
                fields.append({
                    "field" : field,
                    "difference" : difference,
                    "tolerance" : float(tolerance),
                    "passed" : same,
                })

                if (not same) :
                    passed = False
                    errors.append(
                        f"{control_method}/{key}/{field}: "
                        f"difference={difference:.9g} > {float(tolerance):.9g}"
                    )

            checks.append({
                "method_id" : control_method,
                "key" : key,
                "passed" : passed,
                "fields" : fields,
            })

    return {
        "passed" : not errors,
        "check_count" : len(checks),
        "failed_count" : sum(not item["passed"] for item in checks),
        "errors" : errors,
        "checks" : checks,
    }


# -----------------------------------------------------------------------------
# Stage 5 ASR and transcript-view selection
# -----------------------------------------------------------------------------


def compare_stage5_views(
    query_results : pd.DataFrame,
) -> pd.DataFrame :
    if (query_results.empty) :
        return pd.DataFrame()

    rows = []
    keys = ["model_id", "query_id"]
    fields = [
        "first_relevant_rank",
        "story_rr",
        "video_rank",
        "video_rr",
        "top_story_window_id",
        "top_video_id",
    ]

    for model_id, group in query_results.groupby("model_id", sort = True) :
        raw = group[group["view"] == "raw"][keys + fields].rename(columns={
            field : f"raw_{field}"
            for field in fields
        })
        processed = group[group["view"] == "processed"][keys + fields].rename(columns={
            field : f"processed_{field}"
            for field in fields
        })
        merged = processed.merge(raw, on = keys, how = "inner")

        for _, item in merged.iterrows() :
            rows.append({
                "model_id" : model_id,
                "query_id" : item["query_id"],
                "raw_story_rank" : item["raw_first_relevant_rank"],
                "processed_story_rank" : item["processed_first_relevant_rank"],
                "story_relation_processed_vs_raw" : rank_relation(
                    item["processed_first_relevant_rank"],
                    item["raw_first_relevant_rank"],
                ),
                "story_rr_delta_processed_vs_raw" : float(
                    item["processed_story_rr"] - item["raw_story_rr"]
                ),
                "raw_video_rank" : int(item["raw_video_rank"]),
                "processed_video_rank" : int(item["processed_video_rank"]),
                "video_relation_processed_vs_raw" : rank_relation(
                    item["processed_video_rank"],
                    item["raw_video_rank"],
                ),
                "video_rank_delta_processed_minus_raw" : int(
                    item["processed_video_rank"] - item["raw_video_rank"]
                ),
                "video_rr_delta_processed_vs_raw" : float(
                    item["processed_video_rr"] - item["raw_video_rr"]
                ),
                "top_story_changed" : bool(
                    item["processed_top_story_window_id"]
                    != item["raw_top_story_window_id"]
                ),
                "top_video_changed" : bool(
                    item["processed_top_video_id"]
                    != item["raw_top_video_id"]
                ),
            })

    return pd.DataFrame(rows)


def summarize_stage5_views(
    view_comparison : pd.DataFrame,
) -> pd.DataFrame :
    if (view_comparison.empty) :
        return pd.DataFrame()

    rows = []

    for model_id, group in view_comparison.groupby("model_id", sort = True) :
        video_delta = group["video_rr_delta_processed_vs_raw"].to_numpy(dtype = float)
        story_delta = group["story_rr_delta_processed_vs_raw"].to_numpy(dtype = float)

        rows.append({
            "model_id" : model_id,
            "query_count" : len(group),
            "video_rr_delta_mean_processed_vs_raw" : float(video_delta.mean()),
            "video_rr_delta_median_processed_vs_raw" : float(np.median(video_delta)),
            "video_better_processed" : int((video_delta > 0).sum()),
            "video_tie" : int((video_delta == 0).sum()),
            "video_worse_processed" : int((video_delta < 0).sum()),
            "story_rr_delta_mean_processed_vs_raw" : float(story_delta.mean()),
            "story_better_processed" : int((story_delta > 0).sum()),
            "story_tie" : int((story_delta == 0).sum()),
            "story_worse_processed" : int((story_delta < 0).sum()),
        })

    return pd.DataFrame(rows)


def compare_stage5_asr_pairs(
    query_results : pd.DataFrame,
    quality_thresholds : dict[str, float],
    catastrophic_failure : dict[str, int],
) -> tuple[pd.DataFrame, pd.DataFrame] :
    if (query_results.empty) :
        return pd.DataFrame(), pd.DataFrame()

    whisper_model = "whisper_large_v3"
    parakeet_model = "parakeet_ctc_0_6b_vietnamese"
    views = ["raw", "processed"]
    query_rows = []
    summary_rows = []

    metrics = summarize_retrieval_metrics(query_results)

    for whisper_view in views :
        whisper = query_results[
            (query_results["model_id"] == whisper_model)
            & (query_results["view"] == whisper_view)
        ].copy()

        if (whisper.empty) :
            raise ValueError(f"Stage 5 Whisper view is missing: {whisper_view}")

        for parakeet_view in views :
            parakeet = query_results[
                (query_results["model_id"] == parakeet_model)
                & (query_results["view"] == parakeet_view)
            ].copy()

            if (parakeet.empty) :
                raise ValueError(f"Stage 5 Parakeet view is missing: {parakeet_view}")

            pair_id = f"whisper_{whisper_view}__vs__parakeet_{parakeet_view}"
            fields = [
                "query_id",
                "query_text",
                "first_relevant_rank",
                "story_rr",
                "video_rank",
                "video_rr",
                "video_score_margin",
                "top_video_id",
            ]
            left = whisper[fields].rename(columns={
                field : f"whisper_{field}"
                for field in fields
                if field not in {"query_id", "query_text"}
            })
            right = parakeet[fields].rename(columns={
                field : f"parakeet_{field}"
                for field in fields
                if field not in {"query_id", "query_text"}
            })
            merged = left.merge(
                right,
                on = ["query_id", "query_text"],
                how = "inner",
            )

            catastrophic_count = 0
            hard_count = 0

            for _, item in merged.iterrows() :
                whisper_rank = int(item["whisper_video_rank"])
                parakeet_rank = int(item["parakeet_video_rank"])
                rank_delta = parakeet_rank - whisper_rank
                catastrophic = bool(
                    rank_delta >= int(catastrophic_failure["minimum_rank_drop"])
                    and parakeet_rank > int(catastrophic_failure["result_rank_above"])
                )
                hard = bool(
                    whisper_rank == 1
                    and parakeet_rank
                    > int(catastrophic_failure["flag_reference_rank_1_candidate_above"])
                )
                catastrophic_count += int(catastrophic)
                hard_count += int(hard)

                query_rows.append({
                    "pair_id" : pair_id,
                    "whisper_view" : whisper_view,
                    "parakeet_view" : parakeet_view,
                    "query_id" : item["query_id"],
                    "query_text" : item["query_text"],
                    "whisper_story_rank" : item["whisper_first_relevant_rank"],
                    "parakeet_story_rank" : item["parakeet_first_relevant_rank"],
                    "story_rank_delta_parakeet_minus_whisper" : (
                        float(item["parakeet_first_relevant_rank"])
                        - float(item["whisper_first_relevant_rank"])
                        if pd.notna(item["parakeet_first_relevant_rank"])
                        and pd.notna(item["whisper_first_relevant_rank"])
                        else None
                    ),
                    "whisper_story_rr" : float(item["whisper_story_rr"]),
                    "parakeet_story_rr" : float(item["parakeet_story_rr"]),
                    "story_rr_delta_parakeet_minus_whisper" : float(
                        item["parakeet_story_rr"] - item["whisper_story_rr"]
                    ),
                    "whisper_video_rank" : whisper_rank,
                    "parakeet_video_rank" : parakeet_rank,
                    "video_rank_delta_parakeet_minus_whisper" : rank_delta,
                    "whisper_video_rr" : float(item["whisper_video_rr"]),
                    "parakeet_video_rr" : float(item["parakeet_video_rr"]),
                    "video_rr_delta_parakeet_minus_whisper" : float(
                        item["parakeet_video_rr"] - item["whisper_video_rr"]
                    ),
                    "whisper_video_margin" : item["whisper_video_score_margin"],
                    "parakeet_video_margin" : item["parakeet_video_score_margin"],
                    "whisper_top_video_id" : item["whisper_top_video_id"],
                    "parakeet_top_video_id" : item["parakeet_top_video_id"],
                    "catastrophic_regression" : catastrophic,
                    "hard_top1_regression" : hard,
                })

            whisper_metric = metrics[
                (metrics["model_id"] == whisper_model)
                & (metrics["view"] == whisper_view)
            ].iloc[0]
            parakeet_metric = metrics[
                (metrics["model_id"] == parakeet_model)
                & (metrics["view"] == parakeet_view)
            ].iloc[0]

            checks = {}

            for metric_name, threshold_key in [
                ("video_recall_at_1", "video_recall_at_1_max_deficit"),
                ("video_mrr", "video_mrr_max_deficit"),
                ("video_recall_at_5", "video_recall_at_5_max_deficit"),
                ("video_recall_at_10", "video_recall_at_10_max_deficit"),
                ("story_recall_at_1", "story_recall_at_1_max_deficit"),
                ("story_mrr", "story_mrr_max_deficit"),
            ] :
                deficit = max(
                    0.0,
                    float(whisper_metric[metric_name])
                    - float(parakeet_metric[metric_name]),
                )
                threshold = float(quality_thresholds[threshold_key])
                checks[metric_name] = {
                    "whisper" : float(whisper_metric[metric_name]),
                    "parakeet" : float(parakeet_metric[metric_name]),
                    "deficit" : deficit,
                    "threshold" : threshold,
                    "passed" : deficit <= threshold,
                }

            summary_rows.append({
                "pair_id" : pair_id,
                "whisper_view" : whisper_view,
                "parakeet_view" : parakeet_view,
                "whisper_video_recall_at_1" : checks["video_recall_at_1"]["whisper"],
                "parakeet_video_recall_at_1" : checks["video_recall_at_1"]["parakeet"],
                "video_recall_at_1_deficit" : checks["video_recall_at_1"]["deficit"],
                "video_recall_at_1_passed" : checks["video_recall_at_1"]["passed"],
                "whisper_video_mrr" : checks["video_mrr"]["whisper"],
                "parakeet_video_mrr" : checks["video_mrr"]["parakeet"],
                "video_mrr_deficit" : checks["video_mrr"]["deficit"],
                "video_mrr_passed" : checks["video_mrr"]["passed"],
                "whisper_video_recall_at_5" : checks["video_recall_at_5"]["whisper"],
                "parakeet_video_recall_at_5" : checks["video_recall_at_5"]["parakeet"],
                "video_recall_at_5_deficit" : checks["video_recall_at_5"]["deficit"],
                "video_recall_at_5_passed" : checks["video_recall_at_5"]["passed"],
                "whisper_video_recall_at_10" : checks["video_recall_at_10"]["whisper"],
                "parakeet_video_recall_at_10" : checks["video_recall_at_10"]["parakeet"],
                "video_recall_at_10_deficit" : checks["video_recall_at_10"]["deficit"],
                "video_recall_at_10_passed" : checks["video_recall_at_10"]["passed"],
                "whisper_story_recall_at_1" : checks["story_recall_at_1"]["whisper"],
                "parakeet_story_recall_at_1" : checks["story_recall_at_1"]["parakeet"],
                "story_recall_at_1_deficit" : checks["story_recall_at_1"]["deficit"],
                "story_recall_at_1_passed" : checks["story_recall_at_1"]["passed"],
                "whisper_story_mrr" : checks["story_mrr"]["whisper"],
                "parakeet_story_mrr" : checks["story_mrr"]["parakeet"],
                "story_mrr_deficit" : checks["story_mrr"]["deficit"],
                "story_mrr_passed" : checks["story_mrr"]["passed"],
                "quality_thresholds_passed" : all(
                    item["passed"] for item in checks.values()
                ),
                "catastrophic_regression_count" : catastrophic_count,
                "hard_top1_regression_count" : hard_count,
                "catastrophic_gate_passed" : catastrophic_count <= int(
                    catastrophic_failure["maximum_additional_failures"]
                ),
                "manual_inspection_required" : hard_count > 0,
            })

    return pd.DataFrame(query_rows), pd.DataFrame(summary_rows)


# -----------------------------------------------------------------------------
# Stage 6 hierarchical candidate construction
# -----------------------------------------------------------------------------


def _stage67_video_scores(
    row_scores : np.ndarray,
    windows : pd.DataFrame,
) -> pd.DataFrame :
    frame = pd.DataFrame({
        "video_id" : windows["video_id"].astype(str).to_numpy(),
        "score"    : np.asarray(row_scores, dtype = np.float64),
    }).groupby("video_id", as_index = False)["score"].max()

    return frame.sort_values(
        ["score", "video_id"],
        ascending = [False, True],
        kind = "mergesort",
    ).reset_index(drop = True)


def _stage67_window_order(
    row_scores : np.ndarray,
    windows : pd.DataFrame,
    indices : Sequence[int],
) -> list[int] :
    physical = np.asarray(indices, dtype = np.int64)

    if (len(physical) == 0) :
        return []

    frame = pd.DataFrame({
        "_index"    : physical,
        "score"     : np.asarray(row_scores, dtype = np.float64)[physical],
        "start_s"   : pd.to_numeric(windows.iloc[physical]["start_s"], errors = "coerce").to_numpy(dtype = float),
        "window_id" : windows.iloc[physical]["window_id"].astype(str).to_numpy(),
    }).sort_values(
        ["score", "start_s", "window_id"],
        ascending = [False, True, True],
        kind = "mergesort",
    )
    return frame["_index"].astype(int).tolist()


def construct_stage6_candidate_policy(
    first_stage_scores : np.ndarray,
    queries : pd.DataFrame,
    windows : pd.DataFrame,
    eligibility_mask : np.ndarray,
    video_k : int,
    windows_per_video : int,
    silver_radius_s : float,
    minimum_overlap_s : float,
    tie_tolerance : float,
    policy_id : str,
    text_field : str = "canonical_retrieval_text",
) -> tuple[pd.DataFrame, pd.DataFrame] :
    values      = np.asarray(first_stage_scores, dtype = np.float64)
    eligibility = np.asarray(eligibility_mask, dtype = bool)

    if (values.shape != (len(queries), len(windows))) :
        raise ValueError("Stage 6 first-stage scores do not match query/window axes")

    if (eligibility.shape != (len(windows),)) :
        raise ValueError("Stage 6 eligibility mask does not match the window axis")

    if (not np.isfinite(values).all()) :
        raise ValueError("Stage 6 first-stage scores contain NaN or infinite values")

    if (video_k <= 0 or windows_per_video <= 0) :
        raise ValueError("Stage 6 K and M must be positive")

    required = {"window_id", "video_id", "start_s", "end_s"}
    missing = sorted(required - set(windows.columns))

    if (missing) :
        raise KeyError(f"Stage 6 windows are missing fields: {missing}")

    if (windows["window_id"].astype(str).duplicated().any()) :
        raise ValueError("Stage 6 physical window axis contains duplicate IDs")

    physical_video_ids = windows["video_id"].astype(str).to_numpy()
    physical_window_ids = windows["window_id"].astype(str).to_numpy()
    starts = pd.to_numeric(windows["start_s"], errors = "coerce").to_numpy(dtype = float)
    ends   = pd.to_numeric(windows["end_s"], errors = "coerce").to_numpy(dtype = float)
    texts  = (
        windows[text_field].fillna("").astype(str).to_numpy()
        if text_field in windows.columns
        else np.asarray([""] * len(windows), dtype = object)
    )
    all_video_ids = sorted(set(physical_video_ids.tolist()))

    if (video_k > len(all_video_ids)) :
        raise ValueError(f"Stage 6 K={video_k} exceeds the {len(all_video_ids)}-video corpus")

    pool_rows = []
    query_rows = []

    for query_index, (_, query) in enumerate(queries.reset_index(drop = True).iterrows()) :
        query_id     = str(query["query_id"])
        correct_video = str(query["video_id"])
        row_scores   = values[query_index]
        relevant, overlap = silver_relevance_mask(
            query,
            windows,
            silver_radius_s = silver_radius_s,
            minimum_overlap_s = minimum_overlap_s,
        )

        video_frame = _stage67_video_scores(row_scores, windows)
        metric_frame = video_frame.sort_values("video_id", kind = "mergesort").reset_index(drop = True)
        metric_frame["metric_rank"] = worst_tied_ranks_array(
            metric_frame["score"].to_numpy(dtype = float),
            tolerance = tie_tolerance,
        )
        metric_rank_by_video = dict(zip(metric_frame["video_id"], metric_frame["metric_rank"].astype(int)))
        selected_videos = video_frame.head(video_k).copy()
        selected_video_ids = selected_videos["video_id"].astype(str).tolist()
        correct_video_retained = correct_video in set(selected_video_ids)

        correct_indices = np.flatnonzero((physical_video_ids == correct_video) & eligibility)
        correct_order   = _stage67_window_order(row_scores, windows, correct_indices)
        correct_top_m   = correct_order[ : min(windows_per_video, len(correct_order))]
        relevant_window_in_top_m = bool(any(relevant[index] for index in correct_top_m))

        pair_count = 0

        for candidate_video_rank, (_, video_row) in enumerate(selected_videos.iterrows(), start = 1) :
            video_id = str(video_row["video_id"])
            video_indices = np.flatnonzero((physical_video_ids == video_id) & eligibility)
            ordered_windows = _stage67_window_order(row_scores, windows, video_indices)
            selected_windows = ordered_windows[ : min(windows_per_video, len(ordered_windows))]

            for candidate_window_rank, index in enumerate(selected_windows, start = 1) :
                pair_count += 1
                pool_rows.append({
                    "policy_id"                : policy_id,
                    "video_k"                  : int(video_k),
                    "windows_per_video_m"      : int(windows_per_video),
                    "query_id"                 : query_id,
                    "query_text"               : str(query["query_text"]),
                    "historical_split"         : str(query.get("evaluation_split", "unspecified")),
                    "task_type"                : str(query.get("task_type", "KIS")),
                    "query_category"           : str(query.get("query_category", "other")),
                    "difficulty"               : str(query.get("difficulty", "unknown")),
                    "correct_video"            : correct_video,
                    "video_id"                 : video_id,
                    "candidate_video_rank"     : int(candidate_video_rank),
                    "first_stage_video_metric_rank" : int(metric_rank_by_video[video_id]),
                    "first_stage_video_score"  : float(video_row["score"]),
                    "window_id"                : str(physical_window_ids[index]),
                    "candidate_window_rank"    : int(candidate_window_rank),
                    "window_start_s"           : float(starts[index]),
                    "window_end_s"             : float(ends[index]),
                    "window_text"              : str(texts[index]),
                    "first_stage_window_score" : float(row_scores[index]),
                    "eligible"                 : bool(eligibility[index]),
                    "is_correct_video"         : bool(video_id == correct_video),
                    "is_relevant_window"       : bool(relevant[index]),
                    "silver_overlap_s"         : float(overlap[index]),
                })

        failure_type = None
        if (not correct_video_retained) :
            failure_type = "CANDIDATE_VIDEO_MISS"
        elif (not relevant_window_in_top_m) :
            failure_type = "CANDIDATE_WINDOW_MISS"

        query_rows.append({
            "policy_id"                   : policy_id,
            "video_k"                     : int(video_k),
            "windows_per_video_m"         : int(windows_per_video),
            "query_id"                    : query_id,
            "query_text"                  : str(query["query_text"]),
            "historical_split"            : str(query.get("evaluation_split", "unspecified")),
            "task_type"                   : str(query.get("task_type", "KIS")),
            "query_category"              : str(query.get("query_category", "other")),
            "difficulty"                  : str(query.get("difficulty", "unknown")),
            "correct_video"               : correct_video,
            "correct_video_rank"          : int(metric_rank_by_video[correct_video]),
            "correct_video_retained"      : bool(correct_video_retained),
            "relevant_window_retained"    : bool(relevant_window_in_top_m),
            "joint_success"               : bool(correct_video_retained and relevant_window_in_top_m),
            "candidate_pair_count"        : int(pair_count),
            "candidate_video_count"       : int(len(selected_video_ids)),
            "failure_type"                : failure_type,
        })

    pool = pd.DataFrame(pool_rows)
    results = pd.DataFrame(query_rows)

    if (not pool.empty and pool.duplicated(["policy_id", "query_id", "window_id"]).any()) :
        raise ValueError("Stage 6 candidate pool contains duplicate query/window identities")

    return pool, results


def summarize_stage6_candidate_policies(
    candidate_query_results : pd.DataFrame,
) -> pd.DataFrame :
    if (candidate_query_results.empty) :
        return pd.DataFrame()

    rows = []

    for policy_id, group in candidate_query_results.groupby("policy_id", sort = True) :
        video_success = group["correct_video_retained"].astype(bool)
        temporal_success = group["relevant_window_retained"].astype(bool)
        joint_success = group["joint_success"].astype(bool)
        retained = group[video_success]
        conditional_temporal = (
            float(retained["relevant_window_retained"].astype(bool).mean())
            if len(retained)
            else 0.0
        )

        rows.append({
            "policy_id"                       : str(policy_id),
            "video_k"                         : int(group["video_k"].iloc[0]),
            "windows_per_video_m"             : int(group["windows_per_video_m"].iloc[0]),
            "query_count"                     : int(len(group)),
            "candidate_pairs_mean"            : float(group["candidate_pair_count"].mean()),
            "candidate_pairs_max"             : int(group["candidate_pair_count"].max()),
            "video_candidate_recall"          : float(video_success.mean()),
            "temporal_candidate_recall"       : float(temporal_success.mean()),
            "conditional_temporal_recall"     : conditional_temporal,
            "joint_candidate_recall"          : float(joint_success.mean()),
            "video_miss_count"                : int((~video_success).sum()),
            "temporal_miss_count"             : int((~temporal_success).sum()),
            "joint_miss_count"                : int((~joint_success).sum()),
            "video_oracle_recall"             : float(video_success.mean()),
            "temporal_oracle_recall"          : float(temporal_success.mean()),
            "joint_oracle_recall"             : float(joint_success.mean()),
        })

    return pd.DataFrame(rows)


def recommend_stage6_policy(
    policy_summary : pd.DataFrame,
    target_joint_recall : float = 1.0,
    tolerance : float = 1e-12,
) -> tuple[pd.DataFrame, str | None] :
    if (policy_summary.empty) :
        return policy_summary.copy(), None

    frame = policy_summary.copy()
    target = float(target_joint_recall)
    qualifying = frame[
        frame["joint_candidate_recall"] >= target - float(tolerance)
    ].copy()

    if (qualifying.empty) :
        best = float(frame["joint_candidate_recall"].max())
        qualifying = frame[
            frame["joint_candidate_recall"] >= best - float(tolerance)
        ].copy()

    qualifying = qualifying.sort_values(
        ["candidate_pairs_max", "video_k", "windows_per_video_m", "policy_id"],
        ascending = [True, True, True, True],
        kind = "mergesort",
    )
    recommended = str(qualifying.iloc[0]["policy_id"])
    frame["recommended"] = frame["policy_id"].astype(str) == recommended
    return frame, recommended


def stage6_candidate_failures(
    candidate_query_results : pd.DataFrame,
) -> pd.DataFrame :
    if (candidate_query_results.empty) :
        return pd.DataFrame()

    return candidate_query_results[
        candidate_query_results["failure_type"].notna()
    ].copy().sort_values(
        ["policy_id", "failure_type", "correct_video_rank", "query_id"],
        kind = "mergesort",
    ).reset_index(drop = True)


# -----------------------------------------------------------------------------
# Stage 7 candidate reranking
# -----------------------------------------------------------------------------


def evaluate_stage7_candidate_scores(
    candidate_pool : pd.DataFrame,
    candidate_scores : Sequence[float] | np.ndarray,
    first_stage_scores : np.ndarray,
    queries : pd.DataFrame,
    windows : pd.DataFrame,
    eligibility_mask : np.ndarray,
    method_id : str,
    model_id : str,
    view : str,
    query_set : str,
    corpus_id : str,
    silver_radius_s : float,
    minimum_overlap_s : float,
    tie_tolerance : float,
    catastrophic_failure : dict[str, Any] | None = None,
) -> pd.DataFrame :
    pool = candidate_pool.reset_index(drop = True).copy()
    reranked = np.asarray(candidate_scores, dtype = np.float64).reshape(-1)
    first_stage = np.asarray(first_stage_scores, dtype = np.float64)
    eligibility = np.asarray(eligibility_mask, dtype = bool)

    if (len(pool) != len(reranked)) :
        raise ValueError("Stage 7 candidate score count does not match candidate pool")

    if (not np.isfinite(reranked).all()) :
        raise ValueError("Stage 7 candidate scores contain NaN or infinite values")

    if (first_stage.shape != (len(queries), len(windows))) :
        raise ValueError("Stage 7 first-stage score matrix does not match query/window axes")

    if (eligibility.shape != (len(windows),)) :
        raise ValueError("Stage 7 eligibility mask does not match window axis")

    required_pool = {
        "query_id", "video_id", "window_id", "candidate_video_rank",
        "candidate_window_rank", "first_stage_window_score", "is_relevant_window",
    }
    missing_pool = sorted(required_pool - set(pool.columns))

    if (missing_pool) :
        raise KeyError(f"Stage 7 candidate pool is missing fields: {missing_pool}")

    pool["_reranker_score"] = reranked
    window_ids = windows["window_id"].astype(str).to_numpy()
    video_ids  = windows["video_id"].astype(str).to_numpy()
    starts     = pd.to_numeric(windows["start_s"], errors = "coerce").to_numpy(dtype = float)
    ends       = pd.to_numeric(windows["end_s"], errors = "coerce").to_numpy(dtype = float)
    window_lookup = {window_id : index for index, window_id in enumerate(window_ids)}

    if (len(window_lookup) != len(window_ids)) :
        raise ValueError("Stage 7 physical window axis contains duplicate IDs")

    query_lookup = {
        str(row["query_id"]) : (index, row)
        for index, (_, row) in enumerate(queries.reset_index(drop = True).iterrows())
    }
    rows = []
    catastrophic = catastrophic_failure or {}
    minimum_drop = int(catastrophic.get("minimum_rank_drop", 10))
    result_above = int(catastrophic.get("result_rank_above", 10))
    top1_above   = int(catastrophic.get("flag_reference_rank_1_candidate_above", 5))

    for query_id, (query_index, query) in query_lookup.items() :
        subset = pool[pool["query_id"].astype(str) == query_id].copy()

        if (subset.empty) :
            raise ValueError(f"Stage 7 candidate pool contains no rows for {query_id!r}")

        row_first = first_stage[query_index]
        relevant, _ = silver_relevance_mask(
            query,
            windows,
            silver_radius_s = silver_radius_s,
            minimum_overlap_s = minimum_overlap_s,
        )
        correct_video = str(query["video_id"])

        first_video_frame = _stage67_video_scores(row_first, windows)
        first_video_metric = first_video_frame.sort_values("video_id", kind = "mergesort").reset_index(drop = True)
        first_video_metric["metric_rank"] = worst_tied_ranks_array(
            first_video_metric["score"].to_numpy(dtype = float),
            tolerance = tie_tolerance,
        )
        first_metric_by_video = dict(zip(first_video_metric["video_id"], first_video_metric["metric_rank"].astype(int)))
        first_display_video_ids = first_video_frame["video_id"].astype(str).tolist()
        first_score_by_video = dict(zip(first_video_frame["video_id"], first_video_frame["score"].astype(float)))
        first_video_rank = int(first_metric_by_video[correct_video])

        candidate_video_order = (
            subset[["video_id", "candidate_video_rank"]]
            .drop_duplicates("video_id")
            .sort_values("candidate_video_rank", kind = "mergesort")
        )
        original_candidate_video_ids = candidate_video_order["video_id"].astype(str).tolist()
        candidate_set = set(original_candidate_video_ids)
        expected_prefix = first_display_video_ids[ : len(original_candidate_video_ids)]

        if (original_candidate_video_ids != expected_prefix) :
            raise ValueError(f"Stage 7 candidate video prefix mismatch for {query_id!r}")

        candidate_video_scores = (
            subset.groupby("video_id", as_index = False)["_reranker_score"]
            .max()
            .sort_values(["_reranker_score", "video_id"], ascending = [False, True], kind = "mergesort")
            .reset_index(drop = True)
        )
        candidate_metric = candidate_video_scores.sort_values("video_id", kind = "mergesort").reset_index(drop = True)
        candidate_metric["metric_rank"] = worst_tied_ranks_array(
            candidate_metric["_reranker_score"].to_numpy(dtype = float),
            tolerance = tie_tolerance,
        )
        candidate_metric_by_video = dict(zip(candidate_metric["video_id"], candidate_metric["metric_rank"].astype(int)))
        final_candidate_video_ids = candidate_video_scores["video_id"].astype(str).tolist()
        non_candidate_video_ids = [video_id for video_id in first_display_video_ids if video_id not in candidate_set]
        final_display_video_ids = final_candidate_video_ids + non_candidate_video_ids

        if (len(final_display_video_ids) != len(first_display_video_ids) or len(set(final_display_video_ids)) != len(final_display_video_ids)) :
            raise ValueError(f"Stage 7 reconstructed video ranking is invalid for {query_id!r}")

        if (correct_video in candidate_set) :
            video_rank = int(candidate_metric_by_video[correct_video])
            correct_video_score = float(candidate_video_scores.loc[candidate_video_scores["video_id"] == correct_video, "_reranker_score"].iloc[0])
            wrong = candidate_video_scores[candidate_video_scores["video_id"] != correct_video]
            best_wrong_score = float(wrong["_reranker_score"].max()) if not wrong.empty else None
            video_margin = correct_video_score - best_wrong_score if best_wrong_score is not None else None
        else :
            video_rank = int(final_display_video_ids.index(correct_video) + 1)
            correct_video_score = float(first_score_by_video[correct_video])
            best_wrong_score = None
            video_margin = None

        correct_physical = np.flatnonzero(video_ids == correct_video)
        first_story_scores = row_first[correct_physical]
        first_story_ranks = worst_tied_ranks_array(first_story_scores, tolerance = tie_tolerance)
        first_story_relevant = relevant[correct_physical]
        first_relevant_local = np.flatnonzero(first_story_relevant)
        first_stage_story_rank = int(first_story_ranks[first_relevant_local].min()) if len(first_relevant_local) else None

        correct_selected = subset[subset["video_id"].astype(str) == correct_video].copy()
        selected_count = len(correct_selected)
        selected_ids = correct_selected["window_id"].astype(str).tolist()
        selected_index_set = {window_lookup[window_id] for window_id in selected_ids}

        if (correct_video in candidate_set and selected_count) :
            selected_order = correct_selected.sort_values(
                ["_reranker_score", "window_start_s", "window_id"],
                ascending = [False, True, True],
                kind = "mergesort",
            ).reset_index(drop = True)
            selected_metric = correct_selected.sort_values("window_id", kind = "mergesort").reset_index(drop = True)
            selected_metric["metric_rank"] = worst_tied_ranks_array(
                selected_metric["_reranker_score"].to_numpy(dtype = float),
                tolerance = tie_tolerance,
            )
            selected_rank_by_window = dict(zip(selected_metric["window_id"].astype(str), selected_metric["metric_rank"].astype(int)))
            selected_relevant_ranks = [
                selected_rank_by_window[str(row["window_id"])]
                for _, row in correct_selected.iterrows()
                if bool(row["is_relevant_window"])
            ]

            remaining = [index for index in correct_physical if index not in selected_index_set]
            remaining_order = _stage67_window_order(row_first, windows, remaining)

            if (selected_relevant_ranks) :
                story_rank = int(min(selected_relevant_ranks))
            else :
                remaining_scores = np.asarray([row_first[index] for index in remaining], dtype = float)
                remaining_ranks = worst_tied_ranks_array(remaining_scores, tolerance = tie_tolerance) if len(remaining) else np.asarray([], dtype = np.int64)
                remaining_rank_by_index = {index : int(rank) for index, rank in zip(remaining, remaining_ranks)}
                relevant_remaining = [remaining_rank_by_index[index] for index in remaining if relevant[index]]
                story_rank = int(selected_count + min(relevant_remaining)) if relevant_remaining else None

            top_story_window_id = str(selected_order.iloc[0]["window_id"])
            top_story_index = window_lookup[top_story_window_id]
            top_story_score = float(selected_order.iloc[0]["_reranker_score"])
        else :
            story_rank = first_stage_story_rank
            first_story_order = _stage67_window_order(row_first, windows, correct_physical)
            top_story_index = first_story_order[0] if first_story_order else None
            top_story_window_id = str(window_ids[top_story_index]) if top_story_index is not None else None
            top_story_score = float(row_first[top_story_index]) if top_story_index is not None else None

        correct_video_retained = correct_video in candidate_set
        relevant_window_retained = bool(correct_selected["is_relevant_window"].astype(bool).any()) if correct_video_retained else False
        joint_success = bool(correct_video_retained and relevant_window_retained)
        rank_delta = int(video_rank - first_video_rank)
        large_regression = bool(rank_delta >= minimum_drop and video_rank > result_above)
        hard_top1_regression = bool(first_video_rank == 1 and video_rank > top1_above)

        if (not correct_video_retained) :
            failure_type = "CANDIDATE_VIDEO_MISS"
        elif (not relevant_window_retained) :
            failure_type = "CANDIDATE_WINDOW_MISS"
        elif (video_rank > 1) :
            failure_type = "RERANKER_FAILURE"
        else :
            failure_type = None

        if ((large_regression or hard_top1_regression) and failure_type is None) :
            failure_type = "RERANKER_REGRESSION"

        top_video_id = str(final_display_video_ids[0])
        top_video_score = (
            float(candidate_video_scores.iloc[0]["_reranker_score"])
            if len(candidate_video_scores)
            else None
        )
        positive_window_mask = (row_first > 0) & eligibility

        rows.append({
            "query_set"                  : query_set,
            "corpus_id"                  : corpus_id,
            "method_id"                  : method_id,
            "model_id"                   : model_id,
            "view"                       : view,
            "query_id"                   : query_id,
            "query_text"                 : str(query["query_text"]),
            "query_category"             : query.get("query_category", "other"),
            "task_type"                  : query.get("task_type", "KIS"),
            "difficulty"                 : query.get("difficulty", "unknown"),
            "evaluation_split"           : query.get("evaluation_split", "unspecified"),
            "answer_text"                : query.get("answer_text"),
            "correct_video"              : correct_video,
            "frame_id"                   : query.get("frame_id"),
            "answer_time_s"              : query["answer_time_s"],
            "first_relevant_rank"        : story_rank,
            "story_recall_at_1"          : int(story_rank is not None and story_rank <= 1),
            "story_recall_at_3"          : int(story_rank is not None and story_rank <= 3),
            "story_recall_at_5"          : int(story_rank is not None and story_rank <= 5),
            "story_recall_at_10"         : int(story_rank is not None and story_rank <= 10),
            "story_rr"                   : 1.0 / story_rank if story_rank else 0.0,
            "best_relevant_score"        : None,
            "best_irrelevant_score"      : None,
            "story_score_margin"         : None,
            "video_rank"                 : int(video_rank),
            "video_recall_at_1"          : int(video_rank <= 1),
            "video_recall_at_3"          : int(video_rank <= 3),
            "video_recall_at_5"          : int(video_rank <= 5),
            "video_recall_at_10"         : int(video_rank <= 10),
            "video_recall_at_20"         : int(video_rank <= 20),
            "video_rr"                   : 1.0 / video_rank,
            "correct_video_score"        : correct_video_score,
            "best_wrong_video_score"     : best_wrong_score,
            "video_score_margin"         : video_margin,
            "top_story_window_id"        : top_story_window_id,
            "top_story_start_s"          : float(starts[top_story_index]) if top_story_index is not None else None,
            "top_story_end_s"            : float(ends[top_story_index]) if top_story_index is not None else None,
            "top_story_score"            : top_story_score,
            "top_video_id"               : top_video_id,
            "top_video_score"            : top_video_score,
            "physical_window_count"      : len(windows),
            "eligible_window_count"      : int(eligibility.sum()),
            "positive_window_count"      : int(positive_window_mask.sum()),
            "positive_window_fraction"   : float(positive_window_mask.mean()),
            "positive_video_count"       : int(len(set(video_ids[positive_window_mask].tolist()))),
            "correct_video_positive_windows" : int(((video_ids == correct_video) & positive_window_mask).sum()),
            "query_coverage"             : None,
            "zero_evidence"              : bool(not positive_window_mask.any()),
            "first_stage_video_rank"     : int(first_video_rank),
            "first_stage_story_rank"     : first_stage_story_rank,
            "video_rank_delta"           : int(rank_delta),
            "correct_video_retained"     : bool(correct_video_retained),
            "relevant_window_retained"   : bool(relevant_window_retained),
            "joint_candidate_success"    : bool(joint_success),
            "candidate_video_count"      : int(len(candidate_set)),
            "candidate_pair_count"       : int(len(subset)),
            "large_regression"           : bool(large_regression),
            "hard_top1_regression"       : bool(hard_top1_regression),
            "failure_type"               : failure_type,
        })

    return pd.DataFrame(rows)


def compare_stage7_methods(
    query_results : pd.DataFrame,
    reference_method : str = "S0_first_stage",
) -> pd.DataFrame :
    if (query_results.empty) :
        return pd.DataFrame()

    reference = query_results[
        query_results["method_id"].astype(str) == str(reference_method)
    ].copy()

    if (reference["query_id"].astype(str).duplicated().any()) :
        raise ValueError("Stage 7 reference method contains duplicate query IDs")

    reference = reference[
        ["query_id", "video_rank", "first_relevant_rank", "video_rr", "story_rr", "top_video_id", "top_story_window_id"]
    ].rename(columns={
        "video_rank"          : "reference_video_rank",
        "first_relevant_rank" : "reference_story_rank",
        "video_rr"            : "reference_video_rr",
        "story_rr"            : "reference_story_rr",
        "top_video_id"        : "reference_top_video_id",
        "top_story_window_id" : "reference_top_story_window_id",
    })

    rows = []

    for method_id, group in query_results.groupby("method_id", sort = True) :
        if (str(method_id) == str(reference_method)) :
            continue

        merged = group.merge(reference, on = "query_id", how = "inner")

        for _, item in merged.iterrows() :
            rows.append({
                "method_id"              : str(method_id),
                "model_id"               : str(item["model_id"]),
                "view"                   : str(item["view"]),
                "query_id"               : str(item["query_id"]),
                "reference_method"       : str(reference_method),
                "video_relation"         : rank_relation(item["video_rank"], item["reference_video_rank"]),
                "story_relation"         : rank_relation(item["first_relevant_rank"], item["reference_story_rank"]),
                "video_rank"             : int(item["video_rank"]),
                "reference_video_rank"   : int(item["reference_video_rank"]),
                "video_rank_delta"       : int(item["video_rank"] - item["reference_video_rank"]),
                "story_rank"             : item["first_relevant_rank"],
                "reference_story_rank"   : item["reference_story_rank"],
                "video_rr_delta"         : float(item["video_rr"] - item["reference_video_rr"]),
                "story_rr_delta"         : float(item["story_rr"] - item["reference_story_rr"]),
                "rank1_recovery"         : bool(item["reference_video_rank"] > 1 and item["video_rank"] == 1),
                "rank1_loss"             : bool(item["reference_video_rank"] == 1 and item["video_rank"] > 1),
                "top_video_changed"      : bool(item["top_video_id"] != item["reference_top_video_id"]),
                "top_story_changed"      : bool(item["top_story_window_id"] != item["reference_top_story_window_id"]),
                "correct_video_retained" : bool(item.get("correct_video_retained", False)),
                "relevant_window_retained": bool(item.get("relevant_window_retained", False)),
                "failure_type"           : item.get("failure_type"),
                "large_regression"       : bool(item.get("large_regression", False)),
                "hard_top1_regression"   : bool(item.get("hard_top1_regression", False)),
            })

    return pd.DataFrame(rows)


def summarize_stage7_methods(
    query_results : pd.DataFrame,
    reference_method : str,
    bootstrap_samples : int,
    confidence : float,
    seed : int,
) -> pd.DataFrame :
    if (query_results.empty) :
        return pd.DataFrame()

    reference = query_results[
        query_results["method_id"].astype(str) == str(reference_method)
    ][["query_id", "video_rr", "story_rr", "video_rank"]].rename(columns={
        "video_rr"   : "reference_video_rr",
        "story_rr"   : "reference_story_rr",
        "video_rank" : "reference_video_rank",
    })

    rows = []

    for method_id, group in query_results.groupby("method_id", sort = True) :
        merged = group.merge(reference, on = "query_id", how = "inner")
        video_delta = (merged["video_rr"] - merged["reference_video_rr"]).to_numpy(dtype = float)
        story_delta = (merged["story_rr"] - merged["reference_story_rr"]).to_numpy(dtype = float)
        low, high = _bootstrap_interval(video_delta, confidence, bootstrap_samples, seed)
        relations = ["better" if value > 0 else "worse" if value < 0 else "tie" for value in video_delta]
        counts = Counter(relations)

        rows.append({
            "method_id"                       : str(method_id),
            "model_id"                        : str(group["model_id"].iloc[0]),
            "view"                            : str(group["view"].iloc[0]),
            "query_count"                     : int(len(group)),
            "video_rr_mean"                   : float(group["video_rr"].mean()),
            "story_rr_mean"                   : float(group["story_rr"].mean()),
            "video_rr_delta_mean_vs_control"  : float(video_delta.mean()),
            "story_rr_delta_mean_vs_control"  : float(story_delta.mean()),
            "video_better_count"              : int(counts.get("better", 0)),
            "video_tie_count"                 : int(counts.get("tie", 0)),
            "video_worse_count"               : int(counts.get("worse", 0)),
            "rank1_recovery_count"            : int(((merged["reference_video_rank"] > 1) & (merged["video_rank"] == 1)).sum()),
            "rank1_loss_count"                : int(((merged["reference_video_rank"] == 1) & (merged["video_rank"] > 1)).sum()),
            "video_rr_delta_bootstrap_90_low" : low,
            "video_rr_delta_bootstrap_90_high": high,
            "candidate_video_miss_count"      : int((group.get("failure_type", pd.Series(dtype = object)) == "CANDIDATE_VIDEO_MISS").sum()),
            "candidate_window_miss_count"     : int((group.get("failure_type", pd.Series(dtype = object)) == "CANDIDATE_WINDOW_MISS").sum()),
            "reranker_failure_count"          : int((group.get("failure_type", pd.Series(dtype = object)) == "RERANKER_FAILURE").sum()),
            "large_regression_count"          : int(group.get("large_regression", pd.Series(dtype = bool)).fillna(False).astype(bool).sum()),
            "hard_top1_regression_count"      : int(group.get("hard_top1_regression", pd.Series(dtype = bool)).fillna(False).astype(bool).sum()),
        })

    return pd.DataFrame(rows)
