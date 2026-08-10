# Relative path: src/retrieval_v2_evaluation.py
# Purpose: Independent Retrieval v2 evaluation policy for eligibility, temporal relevance, ranking, metrics, and method comparison.

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Sequence

import numpy as np
import pandas as pd


RETRIEVAL_V2_EVALUATION_VERSION = "1.0.0"
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
