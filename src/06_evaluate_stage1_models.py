# Relative path: src/06_evaluate_stage1_models.py
# Purpose: Orchestrate the frozen Stage 1 ASR evaluation, validation gates, retrieval, diagnostics, and reproducible report generation.

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd
import sklearn


SCRIPT_PATH = Path(__file__).resolve()
SRC_ROOT    = SCRIPT_PATH.parent

if (str(SRC_ROOT) not in sys.path) :
    sys.path.insert(0, str(SRC_ROOT))

import evaluation
import postprocess

from evaluation import (
    SemanticScorer,
    aggregate_query_metrics,
    build_reference_agreement,
    build_window_diagnostics,
    compare_model_window_identity,
    compare_text_views,
    find_duplicate_pairs,
    load_json,
    load_model_output_windows,
    mine_repeated_phrases,
    postprocess_consistency_summary,
    query_rows,
    score_windows,
    strong_duplicate_window_ids,
    summarize_operational,
    summarize_reference_agreement,
    summarize_retrieval_metrics,
    summarize_run_payloads,
    validate_benchmark_manifest,
    validate_benchmark_union,
    validate_model_windows_against_benchmark,
)


PROJECT_ROOT = Path(
    os.environ.get(
        "ASR_PROJECT_ROOT",
        "/content/drive/MyDrive/aic26/asr_model_comparison",
    )
).resolve()

CONFIG_PATH = Path(
    os.environ.get(
        "STAGE1_EVALUATION_CONFIG",
        str(PROJECT_ROOT / "configs" / "stage1_evaluation.json"),
    )
).resolve()


ProgressCallback = Callable[[str], None]


def log_stage(index : int, total : int, message : str) -> float :
    print(f"[{index}/{total}] {message}", flush=True)
    return time.perf_counter()


def log_done(start_time : float) -> None :
    print(f"      done in {time.perf_counter() - start_time:.1f}s", flush=True)


def log_detail(message : str) -> None :
    print(f"      {message}", flush=True)


def parse_args() -> argparse.Namespace :
    parser = argparse.ArgumentParser(
        description="Evaluate the frozen Stage 1 ASR benchmark.",
    )

    parser.add_argument(
        "--mode",
        choices=["regression", "development", "holdout", "final"],
        default=os.environ.get("STAGE1_EVALUATION_MODE", "regression"),
        help="Evaluation mode. Holdout/final require evaluation_frozen=true.",
    )

    return parser.parse_args()


def utc_now() -> str :
    return datetime.now(timezone.utc).isoformat()


def resolve_project_path(path_value : str | Path) -> Path :
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def sha256_file(path : Path, chunk_size : int = 8 * 1024 * 1024) -> str :
    digest = hashlib.sha256()

    with Path(path).open("rb") as file :
        while True :
            chunk = file.read(chunk_size)

            if (not chunk) :
                break

            digest.update(chunk)

    return digest.hexdigest()


def canonical_json_hash(value : Any) -> str :
    serialized = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def aggregate_file_hash(entries : list[dict[str, Any]]) -> str :
    normalized = [
        {
            "path"   : item["path"],
            "sha256" : item["sha256"],
        }
        for item in sorted(entries, key=lambda item : item["path"])
    ]
    return canonical_json_hash(normalized)


def json_clean(value : Any) -> Any :
    if (isinstance(value, dict)) :
        return {str(key) : json_clean(item) for key, item in value.items()}

    if (isinstance(value, (list, tuple, set))) :
        return [json_clean(item) for item in value]

    if (isinstance(value, np.generic)) :
        return json_clean(value.item())

    if (isinstance(value, float) and not math.isfinite(value)) :
        return None

    if (pd.isna(value) if not isinstance(value, (str, bytes, dict, list, tuple, set)) else False) :
        return None

    return value


def write_json(path : Path, value : Any) -> None :
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            json_clean(value),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )


def write_csv(path : Path, frame : pd.DataFrame) -> None :
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def load_config() -> dict[str, Any] :
    if (not CONFIG_PATH.exists()) :
        raise FileNotFoundError(f"Evaluation config not found: {CONFIG_PATH}")

    config = load_json(CONFIG_PATH)

    required = [
        "evaluation_id",
        "reference_model_id",
        "candidate_model_ids",
        "models",
        "benchmarks",
        "query_sets",
        "modes",
        "semantic_model",
        "retrieval",
        "diagnostics",
        "regression_gate",
        "reports",
    ]

    missing = [
        key
        for key in required
        if key not in config
    ]

    if (missing) :
        raise ValueError(f"Missing evaluation config keys: {missing}")

    return config


def output_root(config : dict[str, Any], mode : str) -> Path :
    root = resolve_project_path(config["reports"]["root"])

    if (mode == "final") :
        return root

    return root / mode


def validate_expected_benchmark_identity(
    benchmark_key : str,
    benchmark : dict[str, Any],
    specification : dict[str, Any],
) -> dict[str, Any] :
    errors = []

    checks = {
        "stage_id"               : (benchmark.get("stage_id"), specification.get("expected_stage_id")),
        "video_count"            : (benchmark.get("video_count"), specification.get("expected_video_count")),
        "query_count"            : (benchmark.get("query_count"), specification.get("expected_query_count")),
        "window_count"           : (benchmark.get("window_count"), specification.get("expected_window_count")),
        "benchmark_content_hash" : (
            benchmark.get("benchmark_content_hash"),
            specification.get("expected_benchmark_content_hash"),
        ),
    }

    for field, (actual, expected) in checks.items() :
        if (expected is not None and actual != expected) :
            errors.append(
                f"{benchmark_key}: {field} mismatch ({actual!r} != {expected!r})"
            )

    return {
        "benchmark_key" : benchmark_key,
        "passed"        : not errors,
        "errors"        : errors,
    }


def load_benchmarks(config : dict[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, Path]] :
    benchmarks = {}
    paths      = {}

    for benchmark_key, specification in config["benchmarks"].items() :
        path = resolve_project_path(specification["path"])

        if (not path.exists()) :
            raise FileNotFoundError(f"Benchmark not found: {path}")

        benchmark = load_json(path)
        benchmarks[benchmark_key] = benchmark
        paths[benchmark_key]      = path

    return benchmarks, paths


def required_benchmark_keys(config : dict[str, Any], mode : str) -> set[str] :
    query_sets = config["modes"][mode]["query_sets"]
    required   = set()

    for query_set in query_sets :
        query_spec = config["query_sets"][query_set]

        if (query_spec["corpus"] == "all50") :
            required.update(config["benchmarks"].keys())

        for source in query_spec["sources"] :
            required.add(source["benchmark"])

    return required


def load_stage_windows(
    config : dict[str, Any],
    benchmarks : dict[str, dict[str, Any]],
    benchmark_keys : Iterable[str],
    progress_callback : ProgressCallback | None = None,
) -> tuple[dict[tuple[str, str], pd.DataFrame], list[dict[str, Any]], list[dict[str, Any]]] :
    stage_frames       = {}
    validation_results = []
    identity_results   = []
    model_ids = [
        config["reference_model_id"],
        *config["candidate_model_ids"],
    ]

    for benchmark_key in sorted(benchmark_keys) :
        benchmark = benchmarks[benchmark_key]
        expected_windows = benchmark.get("windows", [])

        for model_id in model_ids :
            model_spec = config["models"][model_id]
            output_key = f"{benchmark_key}_output"

            if (output_key not in model_spec) :
                raise KeyError(f"{model_id} has no {output_key} in the evaluation config")

            output_dir = resolve_project_path(model_spec[output_key])

            frame = load_model_output_windows(
                model_id=model_id,
                output_dir=output_dir,
                expected_windows=expected_windows,
                benchmark=benchmark,
                progress_callback=progress_callback,
            )

            stage_frames[(model_id, benchmark_key)] = frame

            validation = validate_model_windows_against_benchmark(
                windows=frame,
                benchmark=benchmark,
                model_id=model_id,
                require_complete=True,
            )
            validation["benchmark_key"] = benchmark_key
            validation["output_dir"]    = str(output_dir)
            validation_results.append(validation)

        if (len(model_ids) >= 2) :
            combined = pd.concat(
                [
                    stage_frames[(model_id, benchmark_key)]
                    for model_id in model_ids
                ],
                ignore_index=True,
            )

            for candidate_model_id in config["candidate_model_ids"] :
                identity = compare_model_window_identity(
                    windows=combined,
                    model_a=config["reference_model_id"],
                    model_b=candidate_model_id,
                )
                identity["benchmark_key"] = benchmark_key
                identity_results.append(identity)

    return stage_frames, validation_results, identity_results


def combined_windows(
    stage_frames : dict[tuple[str, str], pd.DataFrame],
    model_ids : list[str],
    benchmark_keys : Iterable[str],
) -> pd.DataFrame :
    frames = []

    for benchmark_key in benchmark_keys :
        for model_id in model_ids :
            frame = stage_frames.get((model_id, benchmark_key))

            if (frame is not None and not frame.empty) :
                frames.append(frame)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def corpus_windows(
    stage_frames : dict[tuple[str, str], pd.DataFrame],
    model_ids : list[str],
    corpus_name : str,
) -> pd.DataFrame :
    if (corpus_name == "core10") :
        return combined_windows(
            stage_frames,
            model_ids,
            ["core10"],
        )

    if (corpus_name == "all50") :
        return combined_windows(
            stage_frames,
            model_ids,
            ["core10", "extension40"],
        )

    raise ValueError(f"Unsupported corpus: {corpus_name}")


def build_query_set(
    query_set_name : str,
    config : dict[str, Any],
    benchmarks : dict[str, dict[str, Any]],
) -> pd.DataFrame :
    specification = config["query_sets"][query_set_name]
    frames = []

    for source in specification["sources"] :
        benchmark_key = source["benchmark"]
        split         = source.get("split", "all")
        frame         = query_rows(benchmarks[benchmark_key])

        if (split != "all") :
            frame = frame[
                frame["evaluation_split"] == split
            ].copy()

        frames.append(frame)

    frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    if (not frame.empty and frame["query_id"].duplicated().any()) :
        duplicates = sorted(
            frame.loc[
                frame["query_id"].duplicated(keep=False),
                "query_id",
            ].unique().tolist()
        )
        raise ValueError(f"{query_set_name} has duplicate query IDs: {duplicates}")

    frame["query_set"] = query_set_name
    return frame


def query_set_corpus_hash(
    corpus_name : str,
    benchmarks : dict[str, dict[str, Any]],
) -> str :
    if (corpus_name == "core10") :
        benchmark_keys = ["core10"]
    elif (corpus_name == "all50") :
        benchmark_keys = ["core10", "extension40"]
    else :
        raise ValueError(f"Unsupported corpus: {corpus_name}")

    identity = {
        "corpus" : corpus_name,
        "benchmarks" : [
            {
                "benchmark_key"          : key,
                "stage_id"               : benchmarks[key].get("stage_id"),
                "benchmark_content_hash" : benchmarks[key].get("benchmark_content_hash"),
                "window_policy_hash"     : benchmarks[key].get("window_policy_hash"),
            }
            for key in benchmark_keys
        ],
    }

    return canonical_json_hash(identity)


def score_query_set(
    query_set_name : str,
    config : dict[str, Any],
    benchmarks : dict[str, dict[str, Any]],
    stage_frames : dict[tuple[str, str], pd.DataFrame],
    semantic_scorer : SemanticScorer,
    progress_callback : ProgressCallback | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame] :
    model_ids = [
        config["reference_model_id"],
        *config["candidate_model_ids"],
    ]

    query_frame = build_query_set(
        query_set_name,
        config,
        benchmarks,
    )

    corpus_name = config["query_sets"][query_set_name]["corpus"]
    windows     = corpus_windows(
        stage_frames,
        model_ids,
        corpus_name,
    )

    retrieval = config["retrieval"]
    score_frames = []
    result_frames = []

    for model_id in model_ids :
        for view_name, view_spec in retrieval["text_views"].items() :
            scores = score_windows(
                windows=windows,
                queries=query_frame,
                model_id=model_id,
                text_field=view_spec["text_field"],
                view_name=view_name,
                semantic_scorer=semantic_scorer,
                silver_radius_s=float(retrieval["silver_radius_s"]),
                minimum_overlap_s=float(retrieval["minimum_overlap_s"]),
                lexical_weight=float(retrieval["lexical_weight"]),
                semantic_weight=float(retrieval["semantic_weight"]),
                lexical_ngram_range=(
                    int(retrieval["lexical_ngram_min"]),
                    int(retrieval["lexical_ngram_max"]),
                ),
                rejection_field=view_spec["rejection_field"],
                progress_callback=progress_callback,
            )

            scores["query_set"]   = query_set_name
            scores["corpus_name"] = corpus_name
            score_frames.append(scores)

            results = aggregate_query_metrics(scores)
            results["query_set"]   = query_set_name
            results["corpus_name"] = corpus_name
            result_frames.append(results)

    return (
        pd.concat(score_frames, ignore_index=True) if score_frames else pd.DataFrame(),
        pd.concat(result_frames, ignore_index=True) if result_frames else pd.DataFrame(),
    )


def regression_gate(
    query_results : pd.DataFrame,
    config : dict[str, Any],
) -> dict[str, Any] :
    specification = config["regression_gate"]
    query_set_name = specification["query_set"]
    model_id       = specification["model_id"]
    tolerance      = float(specification["tolerance"])

    subset = query_results[
        (query_results["query_set"] == query_set_name)
        & (query_results["model_id"] == model_id)
    ].copy()

    summary = summarize_retrieval_metrics(subset)
    errors  = []
    checks  = []

    for view_name, expected_metrics in specification["expected"].items() :
        row = summary[
            summary["view"] == view_name
        ]

        if (row.empty) :
            errors.append(f"Missing regression view: {view_name}")
            continue

        row = row.iloc[0]

        for metric, expected in expected_metrics.items() :
            actual = float(row[metric])
            difference = abs(actual - float(expected))
            passed = difference <= tolerance

            checks.append({
                "view"       : view_name,
                "metric"     : metric,
                "expected"   : float(expected),
                "actual"     : actual,
                "difference" : difference,
                "passed"     : passed,
            })

            if (not passed) :
                errors.append(
                    f"{view_name}/{metric}: {actual:.12f} != {float(expected):.12f}"
                )

    return {
        "passed"    : not errors,
        "model_id"  : model_id,
        "query_set" : query_set_name,
        "tolerance" : tolerance,
        "checks"    : checks,
        "errors"    : errors,
    }


def retrieval_metric_tables(query_results : pd.DataFrame) -> pd.DataFrame :
    if (query_results.empty) :
        return pd.DataFrame()

    frames = []

    for query_set_name, subset in query_results.groupby("query_set", sort=True) :
        overall = summarize_retrieval_metrics(subset)
        overall["query_set"]       = query_set_name
        overall["breakdown_type"]  = "overall"
        overall["breakdown_value"] = "all"
        frames.append(overall)

        for column in [
            "task_type",
            "difficulty",
            "query_category",
            "evaluation_split",
        ] :
            breakdown = summarize_retrieval_metrics(
                subset,
                breakdown_columns=[column],
            )
            breakdown["query_set"]       = query_set_name
            breakdown["breakdown_type"]  = column
            breakdown["breakdown_value"] = breakdown[column].astype(str)
            breakdown = breakdown.drop(columns=[column])
            frames.append(breakdown)

    columns = [
        "query_set",
        "model_id",
        "view",
        "breakdown_type",
        "breakdown_value",
    ]

    merged = pd.concat(frames, ignore_index=True)
    remaining = [
        column
        for column in merged.columns
        if column not in columns
    ]

    return merged[columns + remaining]


def text_view_effects(query_results : pd.DataFrame) -> pd.DataFrame :
    frames = []

    for query_set_name, subset in query_results.groupby("query_set", sort=True) :
        effects = compare_text_views(
            subset,
            raw_view="raw",
            processed_view="processed",
        )
        effects["query_set"] = query_set_name
        frames.append(effects)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def rank_relation(candidate_rank : Any, reference_rank : Any) -> str :
    candidate = float(candidate_rank) if pd.notna(candidate_rank) else math.inf
    reference = float(reference_rank) if pd.notna(reference_rank) else math.inf

    if (candidate < reference) :
        return "better"
    if (candidate > reference) :
        return "worse"
    return "tie"


def build_query_comparison(
    query_results : pd.DataFrame,
    config : dict[str, Any],
) -> pd.DataFrame :
    if (query_results.empty) :
        return pd.DataFrame()

    reference_model_id = config["reference_model_id"]
    candidate_model_ids = config["candidate_model_ids"]

    metadata_columns = [
        "query_set",
        "query_id",
        "query_text",
        "query_category",
        "task_type",
        "difficulty",
        "evaluation_split",
        "correct_video",
        "frame_id",
        "answer_time_s",
        "answer_text",
        "corpus_name",
    ]

    base = query_results[
        metadata_columns
    ].drop_duplicates(
        subset=["query_set", "query_id"],
    )

    metric_columns = [
        "first_relevant_rank",
        "video_rank",
        "top_story_window_id",
        "top_story_score",
        "best_relevant_score",
        "best_irrelevant_score",
        "story_score_margin",
        "correct_video_score",
        "best_wrong_video_score",
        "video_score_margin",
        "top_video_id",
        "top_video_score",
    ]

    wide = base.copy()

    for model_id in [
        reference_model_id,
        *candidate_model_ids,
    ] :
        for view_name in ["raw", "processed"] :
            subset = query_results[
                (query_results["model_id"] == model_id)
                & (query_results["view"] == view_name)
            ][
                ["query_set", "query_id"] + metric_columns
            ].copy()

            subset = subset.rename(
                columns={
                    column : f"{model_id}__{view_name}__{column}"
                    for column in metric_columns
                }
            )

            wide = wide.merge(
                subset,
                on=["query_set", "query_id"],
                how="left",
            )

    for candidate_model_id in candidate_model_ids :
        for view_name in ["raw", "processed"] :
            candidate_story = f"{candidate_model_id}__{view_name}__first_relevant_rank"
            reference_story = f"{reference_model_id}__{view_name}__first_relevant_rank"
            candidate_video = f"{candidate_model_id}__{view_name}__video_rank"
            reference_video = f"{reference_model_id}__{view_name}__video_rank"

            wide[f"{candidate_model_id}__{view_name}__story_vs_reference"] = wide.apply(
                lambda row : rank_relation(
                    row[candidate_story],
                    row[reference_story],
                ),
                axis=1,
            )

            wide[f"{candidate_model_id}__{view_name}__video_vs_reference"] = wide.apply(
                lambda row : rank_relation(
                    row[candidate_video],
                    row[reference_video],
                ),
                axis=1,
            )

    return wide


def analysis_video_ids(
    config : dict[str, Any],
    benchmarks : dict[str, dict[str, Any]],
    mode : str,
) -> set[str] :
    scope = config["modes"][mode]["analysis_scope"]

    core_videos = set(
        benchmarks["core10"].get("selected_video_ids", [])
    )

    extension = benchmarks.get("extension40", {})
    splits    = extension.get("evaluation_splits", {})

    if (scope == "core10") :
        return core_videos

    if (scope == "non_holdout30") :
        return core_videos | set(
            splits.get("development20", [])
        )

    if (scope == "holdout20") :
        return set(
            splits.get("holdout20", [])
        )

    if (scope == "all50") :
        return core_videos | set(
            extension.get("selected_video_ids", [])
        )

    raise ValueError(f"Unsupported analysis scope: {scope}")


def build_diagnostics(
    analysis_windows : pd.DataFrame,
    config : dict[str, Any],
    semantic_scorer : SemanticScorer,
    progress_callback : ProgressCallback | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame] :
    diagnostics_config = config["diagnostics"]
    model_ids = [
        config["reference_model_id"],
        *config["candidate_model_ids"],
    ]

    if (progress_callback is not None) :
        progress_callback("building window-level repetition and reliability diagnostics")

    window_diagnostics = build_window_diagnostics(
        windows=analysis_windows,
        ngram_sizes=tuple(
            int(value)
            for value in diagnostics_config["repetition_ngram_sizes"]
        ),
        repetition_flag_threshold=diagnostics_config.get(
            "repetition_flag_threshold"
        ),
    )

    duplicate_frames = []
    phrase_frames    = []

    for model_id in model_ids :
        duplicates = find_duplicate_pairs(
            windows=analysis_windows,
            model_id=model_id,
            text_field=diagnostics_config["duplicate_text_field"],
            near_similarity_threshold=float(
                diagnostics_config["near_duplicate_similarity_threshold"]
            ),
            semantic_scorer=semantic_scorer,
            include_adjacent=bool(
                diagnostics_config["include_adjacent_duplicate_pairs"]
            ),
            maximum_pairs=int(
                diagnostics_config["maximum_duplicate_pairs_per_model"]
            ),
            progress_callback=progress_callback,
        )
        duplicate_frames.append(duplicates)

        phrases = mine_repeated_phrases(
            windows=analysis_windows,
            model_id=model_id,
            text_field=diagnostics_config["duplicate_text_field"],
            min_n=int(diagnostics_config["phrase_min_tokens"]),
            max_n=int(diagnostics_config["phrase_max_tokens"]),
            minimum_video_count=int(
                diagnostics_config["minimum_phrase_video_count"]
            ),
            maximum_results=int(
                diagnostics_config["maximum_repeated_phrases_per_model"]
            ),
            progress_callback=progress_callback,
        )
        phrase_frames.append(phrases)

    duplicate_pairs = pd.concat(
        [
            frame
            for frame in duplicate_frames
            if not frame.empty
        ],
        ignore_index=True,
    ) if any(not frame.empty for frame in duplicate_frames) else pd.DataFrame()

    repeated_phrases = pd.concat(
        [
            frame
            for frame in phrase_frames
            if not frame.empty
        ],
        ignore_index=True,
    ) if any(not frame.empty for frame in phrase_frames) else pd.DataFrame()

    reference_duplicates = (
        duplicate_pairs[
            duplicate_pairs["model_id"] == config["reference_model_id"]
        ]
        if not duplicate_pairs.empty
        else pd.DataFrame()
    )

    untrusted_reference_ids = strong_duplicate_window_ids(
        reference_duplicates
    )

    agreement_frames = []

    for candidate_model_id in config["candidate_model_ids"] :
        agreement = build_reference_agreement(
            windows=analysis_windows,
            reference_model_id=config["reference_model_id"],
            candidate_model_id=candidate_model_id,
            semantic_scorer=semantic_scorer,
            reference_untrusted_window_ids=untrusted_reference_ids,
            progress_callback=progress_callback,
        )
        agreement_frames.append(agreement)

    reference_agreement = pd.concat(
        [
            frame
            for frame in agreement_frames
            if not frame.empty
        ],
        ignore_index=True,
    ) if any(not frame.empty for frame in agreement_frames) else pd.DataFrame()

    agreement_summary = (
        summarize_reference_agreement(reference_agreement)
        if not reference_agreement.empty
        else pd.DataFrame()
    )

    return (
        window_diagnostics,
        duplicate_pairs,
        repeated_phrases,
        reference_agreement,
        agreement_summary,
    )


def load_run_summaries(
    config : dict[str, Any],
    benchmark_keys : Iterable[str],
) -> tuple[pd.DataFrame, list[dict[str, Any]]] :
    payloads = []
    records  = []

    for model_id, model_spec in config["models"].items() :
        for benchmark_key in benchmark_keys :
            field = f"{benchmark_key}_run_summary"
            path_value = model_spec.get(field)

            if (not path_value) :
                continue

            path = resolve_project_path(path_value)

            if (not path.exists()) :
                records.append({
                    "model_id"      : model_id,
                    "benchmark_key" : benchmark_key,
                    "path"          : str(path),
                    "exists"        : False,
                })
                continue

            payload = load_json(path)
            payloads.append(payload)
            records.append({
                "model_id"      : model_id,
                "benchmark_key" : benchmark_key,
                "path"          : str(path),
                "exists"        : True,
            })

    return summarize_run_payloads(payloads), records


def summarize_text_view_effects(effects : pd.DataFrame) -> list[dict[str, Any]] :
    if (effects.empty) :
        return []

    rows = []

    for (query_set, model_id), group in effects.groupby(
        ["query_set", "model_id"],
        sort=True,
    ) :
        rows.append({
            "query_set" : query_set,
            "model_id"  : model_id,
            "story"     : dict(Counter(group["story_effect"].tolist())),
            "video"     : dict(Counter(group["video_effect"].tolist())),
        })

    return rows


def summarize_candidate_relations(
    query_comparison : pd.DataFrame,
    config : dict[str, Any],
) -> list[dict[str, Any]] :
    if (query_comparison.empty) :
        return []

    rows = []

    for query_set, group in query_comparison.groupby("query_set", sort=True) :
        for candidate_model_id in config["candidate_model_ids"] :
            for view_name in ["raw", "processed"] :
                story_column = (
                    f"{candidate_model_id}__{view_name}__story_vs_reference"
                )
                video_column = (
                    f"{candidate_model_id}__{view_name}__video_vs_reference"
                )

                rows.append({
                    "query_set"          : query_set,
                    "candidate_model_id" : candidate_model_id,
                    "view"               : view_name,
                    "story"              : dict(
                        Counter(group[story_column].tolist())
                    ),
                    "video"              : dict(
                        Counter(group[video_column].tolist())
                    ),
                })

    return rows


def model_summary_table(
    operational : pd.DataFrame,
    postprocess_summary : pd.DataFrame,
    window_diagnostics : pd.DataFrame,
    duplicate_pairs : pd.DataFrame,
    repeated_phrases : pd.DataFrame,
    reference_agreement : pd.DataFrame,
    config : dict[str, Any],
) -> pd.DataFrame :
    model_ids = [
        config["reference_model_id"],
        *config["candidate_model_ids"],
    ]

    rows = []

    for model_id in model_ids :
        operational_group = operational[
            operational["model_id"] == model_id
        ] if not operational.empty else pd.DataFrame()

        postprocess_group = postprocess_summary[
            postprocess_summary["model_id"] == model_id
        ] if not postprocess_summary.empty else pd.DataFrame()

        diagnostics_group = window_diagnostics[
            window_diagnostics["model_id"] == model_id
        ] if not window_diagnostics.empty else pd.DataFrame()

        duplicate_group = duplicate_pairs[
            duplicate_pairs["model_id"] == model_id
        ] if not duplicate_pairs.empty else pd.DataFrame()

        phrase_group = repeated_phrases[
            repeated_phrases["model_id"] == model_id
        ] if not repeated_phrases.empty else pd.DataFrame()

        runtime = pd.to_numeric(
            operational_group.get(
                "total_inference_runtime_s",
                pd.Series(dtype=float),
            ),
            errors="coerce",
        )

        audio = pd.to_numeric(
            operational_group.get(
                "total_attempted_audio_s",
                pd.Series(dtype=float),
            ),
            errors="coerce",
        )

        total_runtime = float(runtime.fillna(0.0).sum())
        total_audio   = float(audio.fillna(0.0).sum())

        exact_unrelated = 0

        if (not duplicate_group.empty) :
            exact_unrelated = int(
                (
                    duplicate_group["exact_normalized"]
                    & duplicate_group["relationship"].isin(
                        ["non_adjacent_same_video", "cross_video"]
                    )
                ).sum()
            )

        row = {
            "model_id"                     : model_id,
            "role"                         : config["models"][model_id]["role"],
            "analysis_window_count"        : len(diagnostics_group),
            "successful_window_count"      : int(
                operational_group["successful_window_count"].sum()
            ) if not operational_group.empty else 0,
            "failed_window_count"          : int(
                operational_group["failed_window_count"].sum()
            ) if not operational_group.empty else 0,
            "empty_window_count"           : int(
                operational_group["empty_window_count"].sum()
            ) if not operational_group.empty else 0,
            "total_inference_runtime_s"    : total_runtime,
            "total_attempted_audio_s"      : total_audio,
            "aggregate_rtf"                : (
                total_runtime / total_audio
                if total_audio > 0
                else None
            ),
            "peak_gpu_memory_bytes"        : float(
                operational_group["peak_gpu_memory_bytes"].max()
            ) if not operational_group.empty else None,
            "peak_reserved_memory_bytes"   : float(
                operational_group["peak_reserved_memory_bytes"].max()
            ) if not operational_group.empty else None,
            "postprocess_match_rate"       : float(
                postprocess_group["postprocess_match_count"].sum()
                / max(
                    int(postprocess_group["present_window_count"].sum()),
                    1,
                )
            ) if not postprocess_group.empty else None,
            "invalid_timestamp_count"      : int(
                diagnostics_group["invalid_timestamps"].sum()
            ) if not diagnostics_group.empty else 0,
            "processed_rejection_count"    : int(
                diagnostics_group["processed_rejected"].sum()
            ) if not diagnostics_group.empty else 0,
            "high_repetition_flag_count"   : int(
                diagnostics_group["reliability_flags"].map(
                    lambda flags : "high_internal_repetition" in (flags or [])
                ).sum()
            ) if not diagnostics_group.empty else 0,
            "exact_unrelated_duplicate_pairs": exact_unrelated,
            "repeated_phrase_count"        : len(phrase_group),
        }

        if (model_id != config["reference_model_id"] and not reference_agreement.empty) :
            agreement_group = reference_agreement[
                reference_agreement["candidate_model_id"] == model_id
            ]

            trusted = agreement_group[
                agreement_group["reference_trusted"]
            ]

            row["trusted_whisper_reference_wer_mean"] = float(
                pd.to_numeric(
                    trusted["whisper_reference_wer"],
                    errors="coerce",
                ).mean()
            ) if not trusted.empty else None

            row["trusted_whisper_reference_cer_mean"] = float(
                pd.to_numeric(
                    trusted["whisper_reference_cer"],
                    errors="coerce",
                ).mean()
            ) if not trusted.empty else None

            row["trusted_token_f1_mean"] = float(
                pd.to_numeric(
                    trusted["token_f1"],
                    errors="coerce",
                ).mean()
            ) if not trusted.empty else None

            row["trusted_semantic_similarity_mean"] = float(
                pd.to_numeric(
                    trusted["semantic_similarity"],
                    errors="coerce",
                ).mean()
            ) if (
                not trusted.empty
                and trusted["semantic_similarity"].notna().any()
            ) else None

        rows.append(row)

    return pd.DataFrame(rows)


def preferred_inspection_query_set(mode : str) -> str :
    mapping = {
        "regression"  : "regression_core10",
        "development" : "development20",
        "holdout"     : "holdout20",
        "final"       : "all50",
    }
    return mapping[mode]


def select_inspection_cases(
    mode : str,
    query_results : pd.DataFrame,
    query_comparison : pd.DataFrame,
    window_diagnostics : pd.DataFrame,
    duplicate_pairs : pd.DataFrame,
    repeated_phrases : pd.DataFrame,
    reference_agreement : pd.DataFrame,
    allowed_video_ids : set[str],
    config : dict[str, Any],
) -> list[dict[str, Any]] :
    maximum_cases = int(
        config["diagnostics"]["maximum_inspection_cases"]
    )
    query_set_name = preferred_inspection_query_set(mode)
    reference_model_id = config["reference_model_id"]
    candidate_model_id = config["candidate_model_ids"][0]

    cases = []
    used  = set()

    def add_case(
        case_type : str,
        query_id : str | None = None,
        window_id : str | None = None,
        model_id : str | None = None,
        details : dict[str, Any] | None = None,
    ) -> None :
        key = (
            case_type,
            query_id,
            window_id,
            model_id,
        )

        if (key in used or len(cases) >= maximum_cases) :
            return

        used.add(key)
        cases.append({
            "case_type" : case_type,
            "query_id"  : query_id,
            "window_id" : window_id,
            "model_id"  : model_id,
            "details"   : details or {},
        })

    agreement = reference_agreement[
        reference_agreement["video_id"].isin(allowed_video_ids)
    ].copy() if not reference_agreement.empty else pd.DataFrame()

    if (not agreement.empty) :
        row = agreement.sort_values(
            "symmetric_token_distance",
            ascending=False,
        ).iloc[0]

        add_case(
            "largest_transcript_disagreement",
            window_id=row["window_id"],
            model_id=candidate_model_id,
            details={
                "symmetric_token_distance" : row["symmetric_token_distance"],
                "whisper_reference_wer"    : row["whisper_reference_wer"],
            },
        )

        agreement["length_deviation"] = (
            pd.to_numeric(
                agreement["candidate_reference_length_ratio"],
                errors="coerce",
            )
            - 1.0
        ).abs()

        row = agreement.sort_values(
            "length_deviation",
            ascending=False,
        ).iloc[0]

        add_case(
            "largest_length_disagreement",
            window_id=row["window_id"],
            model_id=candidate_model_id,
            details={
                "candidate_reference_length_ratio" : (
                    row["candidate_reference_length_ratio"]
                ),
            },
        )

        suspicious = agreement[
            ~agreement["reference_trusted"]
        ]

        if (not suspicious.empty) :
            row = suspicious.iloc[0]

            add_case(
                "whisper_reference_suspicious",
                window_id=row["window_id"],
                model_id=reference_model_id,
                details={
                    "reasons" : row["reference_untrusted_reasons"],
                },
            )

    comparison = query_comparison[
        query_comparison["query_set"] == query_set_name
    ].copy() if not query_comparison.empty else pd.DataFrame()

    if (not comparison.empty) :
        candidate_story = (
            f"{candidate_model_id}__processed__first_relevant_rank"
        )
        reference_story = (
            f"{reference_model_id}__processed__first_relevant_rank"
        )
        candidate_video = (
            f"{candidate_model_id}__processed__video_rank"
        )
        reference_video = (
            f"{reference_model_id}__processed__video_rank"
        )
        candidate_window = (
            f"{candidate_model_id}__processed__top_story_window_id"
        )
        reference_window = (
            f"{reference_model_id}__processed__top_story_window_id"
        )

        wins = comparison[
            comparison[
                f"{candidate_model_id}__processed__story_vs_reference"
            ] == "better"
        ].copy()

        if (not wins.empty) :
            wins["gain"] = (
                pd.to_numeric(
                    wins[reference_story],
                    errors="coerce",
                )
                - pd.to_numeric(
                    wins[candidate_story],
                    errors="coerce",
                )
            )

            row = wins.sort_values(
                "gain",
                ascending=False,
            ).iloc[0]

            add_case(
                "parakeet_retrieval_win",
                query_id=row["query_id"],
                window_id=row[candidate_window],
                model_id=candidate_model_id,
                details={
                    "reference_story_rank" : row[reference_story],
                    "candidate_story_rank" : row[candidate_story],
                },
            )

        losses = comparison[
            comparison[
                f"{candidate_model_id}__processed__story_vs_reference"
            ] == "worse"
        ].copy()

        if (not losses.empty) :
            losses["loss"] = (
                pd.to_numeric(
                    losses[candidate_story],
                    errors="coerce",
                )
                - pd.to_numeric(
                    losses[reference_story],
                    errors="coerce",
                )
            )

            row = losses.sort_values(
                "loss",
                ascending=False,
            ).iloc[0]

            add_case(
                "whisper_retrieval_win",
                query_id=row["query_id"],
                window_id=row[reference_window],
                model_id=reference_model_id,
                details={
                    "reference_story_rank" : row[reference_story],
                    "candidate_story_rank" : row[candidate_story],
                },
            )

        both_fail = comparison[
            (
                pd.to_numeric(
                    comparison[reference_video],
                    errors="coerce",
                ) > 1
            )
            & (
                pd.to_numeric(
                    comparison[candidate_video],
                    errors="coerce",
                ) > 1
            )
        ]

        if (not both_fail.empty) :
            row = both_fail.iloc[0]

            add_case(
                "both_retrieval_fail",
                query_id=row["query_id"],
                window_id=row[candidate_window],
                details={
                    "reference_video_rank" : row[reference_video],
                    "candidate_video_rank" : row[candidate_video],
                },
            )

    effects = text_view_effects(
        query_results[
            query_results["query_set"] == query_set_name
        ]
    )

    changed = effects[
        (effects["story_effect"] != "unchanged")
        | (effects["video_effect"] != "unchanged")
    ] if not effects.empty else pd.DataFrame()

    if (not changed.empty) :
        row = changed.iloc[0]

        add_case(
            "raw_vs_processed_rank_change",
            query_id=row["query_id"],
            model_id=row["model_id"],
            details={
                "story_effect" : row["story_effect"],
                "video_effect" : row["video_effect"],
            },
        )

    diagnostics = window_diagnostics[
        window_diagnostics["video_id"].isin(allowed_video_ids)
    ].copy() if not window_diagnostics.empty else pd.DataFrame()

    for model_id, case_type in [
        (reference_model_id, "highest_whisper_repetition"),
        (candidate_model_id, "highest_parakeet_repetition"),
    ] :
        subset = diagnostics[
            diagnostics["model_id"] == model_id
        ]

        if (not subset.empty) :
            row = subset.sort_values(
                "intra_window_repetition_score",
                ascending=False,
            ).iloc[0]

            add_case(
                case_type,
                window_id=row["window_id"],
                model_id=model_id,
                details={
                    "intra_window_repetition_score" : (
                        row["intra_window_repetition_score"]
                    ),
                },
            )

    if (not duplicate_pairs.empty) :
        eligible_pairs = duplicate_pairs[
            duplicate_pairs["left_video_id"].isin(allowed_video_ids)
            & duplicate_pairs["right_video_id"].isin(allowed_video_ids)
            & duplicate_pairs["exact_normalized"]
            & (duplicate_pairs["relationship"] == "cross_video")
        ]

        if (not eligible_pairs.empty) :
            row = eligible_pairs.iloc[0]

            add_case(
                "exact_cross_video_duplicate",
                window_id=row["left_window_id"],
                model_id=row["model_id"],
                details={
                    "other_window_id" : row["right_window_id"],
                    "other_video_id"  : row["right_video_id"],
                },
            )

    if (not repeated_phrases.empty) :
        phrase_rows = repeated_phrases.sort_values(
            "suspicious_score",
            ascending=False,
        )

        if (not phrase_rows.empty) :
            row = phrase_rows.iloc[0]

            add_case(
                "suspicious_repeated_phrase",
                model_id=row["model_id"],
                details={
                    "phrase"               : row["phrase"],
                    "distinct_video_count" : row["distinct_video_count"],
                    "example_window_ids"   : row["example_window_ids"],
                },
            )

    if (not diagnostics.empty) :
        runtime = pd.to_numeric(
            diagnostics["runtime_s"],
            errors="coerce",
        )

        if (runtime.notna().any()) :
            row = diagnostics.loc[
                runtime.idxmax()
            ]

            add_case(
                "runtime_outlier",
                window_id=row["window_id"],
                model_id=row["model_id"],
                details={
                    "runtime_s"        : row["runtime_s"],
                    "real_time_factor" : row["real_time_factor"],
                },
            )

    return cases


def source_hash_record(path : Path) -> dict[str, Any] :
    return {
        "path"   : str(path),
        "exists" : path.exists(),
        "sha256" : sha256_file(path) if path.exists() else None,
    }


def model_output_hashes(
    config : dict[str, Any],
    benchmarks : dict[str, dict[str, Any]],
    benchmark_keys : Iterable[str],
    progress_callback : ProgressCallback | None = None,
) -> dict[str, Any] :
    results = {}

    for model_id, model_spec in config["models"].items() :
        results[model_id] = {}

        for benchmark_key in sorted(benchmark_keys) :
            output_dir = resolve_project_path(model_spec[f"{benchmark_key}_output"])
            video_ids  = benchmarks[benchmark_key].get("selected_video_ids", [])
            entries    = []

            if (progress_callback is not None) :
                progress_callback(f"{model_id}/{benchmark_key}: hashing {len(video_ids)} output files")

            for index, video_id in enumerate(video_ids, start = 1) :
                path = output_dir / f"{video_id}.json"

                if (not path.exists()) :
                    entries.append({
                        "path"   : str(path.relative_to(PROJECT_ROOT)),
                        "exists" : False,
                        "sha256" : None,
                    })
                    continue

                entries.append({
                    "path"   : str(path.relative_to(PROJECT_ROOT)),
                    "exists" : True,
                    "sha256" : sha256_file(path),
                })

                if (progress_callback is not None and (index == len(video_ids) or index % 10 == 0)) :
                    progress_callback(f"{model_id}/{benchmark_key}: hashed {index}/{len(video_ids)} files")

            existing = [item for item in entries if item["exists"]]

            results[model_id][benchmark_key] = {
                "output_dir"     : str(output_dir),
                "files"          : entries,
                "aggregate_hash" : aggregate_file_hash(existing) if existing else None,
            }

    return results


def build_evaluation_manifest(
    config : dict[str, Any],
    mode : str,
    benchmark_paths : dict[str, Path],
    benchmarks : dict[str, dict[str, Any]],
    benchmark_keys : Iterable[str],
) -> dict[str, Any] :
    import sentence_transformers

    sources = {
        "config" : source_hash_record(CONFIG_PATH),
        "evaluation" : source_hash_record(
            Path(evaluation.__file__).resolve()
        ),
        "orchestrator" : source_hash_record(SCRIPT_PATH),
        "postprocess" : source_hash_record(
            Path(postprocess.__file__).resolve()
        ),
        "benchmarks" : {
            key : source_hash_record(path)
            for key, path in benchmark_paths.items()
            if key in benchmark_keys
        },
    }

    return {
        "schema_version"     : "1.0",
        "evaluation_id"      : config["evaluation_id"],
        "evaluation_version" : evaluation.EVALUATION_VERSION,
        "mode"               : mode,
        "generated_at_utc"   : utc_now(),
        "evaluation_frozen"  : bool(config["evaluation_frozen"]),
        "config_hash"        : sha256_file(CONFIG_PATH),
        "postprocess_version": postprocess.POSTPROCESS_VERSION,
        "semantic_model"     : config["semantic_model"],
        "retrieval"          : config["retrieval"],
        "diagnostics"        : config["diagnostics"],
        "logical_corpora" : {
            "core10" : query_set_corpus_hash(
                "core10",
                benchmarks,
            ),
            "all50" : (
                query_set_corpus_hash(
                    "all50",
                    benchmarks,
                )
                if set(["core10", "extension40"]).issubset(benchmark_keys)
                else None
            ),
        },
        "sources"       : sources,
        "model_outputs" : model_output_hashes(
            config,
            benchmarks,
            benchmark_keys,
            progress_callback=log_detail,
        ),
        "environment" : {
            "python"                : sys.version,
            "platform"              : platform.platform(),
            "numpy"                 : np.__version__,
            "pandas"                : pd.__version__,
            "scikit_learn"          : sklearn.__version__,
            "sentence_transformers" : sentence_transformers.__version__,
        },
    }


def summarize_validation(
    benchmark_validations : list[dict[str, Any]],
    expected_identities : list[dict[str, Any]],
    union_validation : dict[str, Any] | None,
    model_validations : list[dict[str, Any]],
    identity_validations : list[dict[str, Any]],
) -> dict[str, Any] :
    components = {
        "benchmarks"               : benchmark_validations,
        "expected_benchmark_identity": expected_identities,
        "benchmark_union"          : union_validation,
        "model_outputs"            : model_validations,
        "cross_model_window_identity": identity_validations,
    }

    failures = []

    for group_name, group in components.items() :
        if (group is None) :
            continue

        items = group if isinstance(group, list) else [group]

        for item in items :
            if (not item.get("passed", False)) :
                failures.append({
                    "group"  : group_name,
                    "detail" : item,
                })

    return {
        "passed"     : not failures,
        "components" : components,
        "failures"   : failures,
    }


def bundle_top_suspicious_windows(
    window_diagnostics : pd.DataFrame,
    maximum : int,
) -> list[dict[str, Any]] :
    if (window_diagnostics.empty) :
        return []

    frame = window_diagnostics.copy()
    frame["flag_count"] = frame["reliability_flags"].map(
        lambda values : len(values or [])
    )

    frame = frame.sort_values(
        [
            "flag_count",
            "intra_window_repetition_score",
            "runtime_s",
        ],
        ascending=[False, False, False],
    )

    columns = [
        "model_id",
        "video_id",
        "window_id",
        "start_s",
        "end_s",
        "intra_window_repetition_score",
        "invalid_timestamps",
        "runtime_s",
        "real_time_factor",
        "reliability_flags",
    ]

    return frame[columns].head(maximum).to_dict(
        orient="records"
    )


def build_bundle(
    config : dict[str, Any],
    mode : str,
    validation : dict[str, Any],
    regression : dict[str, Any],
    retrieval_metrics : pd.DataFrame,
    query_comparison : pd.DataFrame,
    effects : pd.DataFrame,
    model_summary : pd.DataFrame,
    operational : pd.DataFrame,
    run_summaries : pd.DataFrame,
    postprocess_summary : pd.DataFrame,
    agreement_summary : pd.DataFrame,
    repeated_phrases : pd.DataFrame,
    window_diagnostics : pd.DataFrame,
    inspection_cases : list[dict[str, Any]],
    benchmarks : dict[str, dict[str, Any]],
) -> dict[str, Any] :
    top_phrase_count = int(
        config["reports"]["top_repeated_phrases_in_bundle"]
    )

    top_window_count = int(
        config["reports"]["top_suspicious_windows_in_bundle"]
    )

    overall_retrieval = retrieval_metrics[
        retrieval_metrics["breakdown_type"] == "overall"
    ].copy()

    phrase_records = (
        repeated_phrases.sort_values(
            ["model_id", "suspicious_score"],
            ascending=[True, False],
        ).groupby(
            "model_id",
            sort=True,
        ).head(
            top_phrase_count
        ).to_dict(
            orient="records"
        )
        if not repeated_phrases.empty
        else []
    )

    return {
        "schema_version"     : "1.0",
        "evaluation_id"      : config["evaluation_id"],
        "evaluation_version" : evaluation.EVALUATION_VERSION,
        "mode"               : mode,
        "generated_at_utc"   : utc_now(),
        "evaluation_frozen"  : bool(config["evaluation_frozen"]),
        "reference_model_id" : config["reference_model_id"],
        "candidate_model_ids": config["candidate_model_ids"],
        "query_sets"         : config["modes"][mode]["query_sets"],
        "benchmark_identity" : {
            key : {
                "stage_id"               : item.get("stage_id"),
                "video_count"            : item.get("video_count"),
                "query_count"            : item.get("query_count"),
                "window_count"           : item.get("window_count"),
                "benchmark_content_hash" : item.get("benchmark_content_hash"),
                "window_policy_hash"     : item.get("window_policy_hash"),
            }
            for key, item in benchmarks.items()
        },
        "validation"      : validation,
        "regression_gate" : regression,
        "retrieval"       : overall_retrieval.to_dict(
            orient="records"
        ),
        "model_summary" : model_summary.to_dict(
            orient="records"
        ),
        "operational" : operational.to_dict(
            orient="records"
        ),
        "run_summaries" : run_summaries.to_dict(
            orient="records"
        ),
        "postprocess_consistency" : postprocess_summary.to_dict(
            orient="records"
        ),
        "reference_agreement" : agreement_summary.to_dict(
            orient="records"
        ),
        "raw_vs_processed" : summarize_text_view_effects(
            effects
        ),
        "candidate_vs_reference" : summarize_candidate_relations(
            query_comparison,
            config,
        ),
        "top_repeated_phrases" : phrase_records,
        "top_suspicious_windows" : bundle_top_suspicious_windows(
            window_diagnostics,
            top_window_count,
        ),
        "inspection_cases" : inspection_cases,
    }


def main() -> None :
    args   = parse_args()
    mode   = args.mode
    config = load_config()

    if (mode not in config["modes"]) :
        raise ValueError(f"Unsupported evaluation mode: {mode}")

    mode_spec = config["modes"][mode]

    if (
        mode_spec.get("requires_frozen_evaluation", False)
        and not bool(config.get("evaluation_frozen", False))
    ) :
        raise RuntimeError(
            f"Mode '{mode}' is locked because evaluation_frozen=false. "
            "Freeze the evaluator/config after development review before opening holdout."
        )

    reports_root = output_root(
        config,
        mode,
    )
    reports_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 88)
    print("STAGE 1 ASR EVALUATION")
    print("=" * 88)
    print(f"Evaluation: {config['evaluation_id']}")
    print(f"Mode:       {mode}")
    print(f"Frozen:     {config['evaluation_frozen']}")
    print(f"Project:    {PROJECT_ROOT}")
    print(f"Config:     {CONFIG_PATH}")
    print(f"Reports:    {reports_root}")
    print()

    stage_start = log_stage(1, 8, "Loading benchmark manifests...")
    benchmarks, benchmark_paths = load_benchmarks(config)
    benchmark_keys = required_benchmark_keys(config, mode)
    log_detail(f"benchmark keys: {', '.join(sorted(benchmark_keys))}")
    log_done(stage_start)

    stage_start = log_stage(2, 8, "Validating benchmarks and model outputs...")
    benchmark_validations = []
    expected_identities    = []

    for benchmark_key in sorted(benchmark_keys) :
        benchmark = benchmarks[benchmark_key]

        benchmark_validations.append(
            validate_benchmark_manifest(
                benchmark
            )
        )

        expected_identities.append(
            validate_expected_benchmark_identity(
                benchmark_key,
                benchmark,
                config["benchmarks"][benchmark_key],
            )
        )

    union_validation = None

    if (
        "core10" in benchmark_keys
        and "extension40" in benchmark_keys
    ) :
        union_validation = validate_benchmark_union(
            [
                benchmarks["core10"],
                benchmarks["extension40"],
            ]
        )

    stage_frames, model_validations, identity_validations = (
        load_stage_windows(
            config,
            benchmarks,
            benchmark_keys,
            progress_callback=log_detail,
        )
    )

    validation = summarize_validation(
        benchmark_validations,
        expected_identities,
        union_validation,
        model_validations,
        identity_validations,
    )

    write_json(
        reports_root / "validation_summary.json",
        validation,
    )

    if (not validation["passed"]) :
        raise RuntimeError(
            "Evaluation integrity validation failed. "
            f"See {reports_root / 'validation_summary.json'}"
        )

    log_detail("integrity validation: PASS")
    log_done(stage_start)

    model_ids = [
        config["reference_model_id"],
        *config["candidate_model_ids"],
    ]

    all_loaded_windows = combined_windows(
        stage_frames,
        model_ids,
        benchmark_keys,
    )

    postprocess_summary = postprocess_consistency_summary(
        all_loaded_windows
    )

    semantic = config["semantic_model"]
    stage_start = log_stage(3, 8, "Loading multilingual E5 semantic scorer...")
    log_detail(f"model: {semantic['model_name']}")
    log_detail(f"revision: {semantic['revision']}")
    semantic_scorer = SemanticScorer(model_name=semantic["model_name"], revision=semantic["revision"], query_prefix=semantic["query_prefix"], passage_prefix=semantic["passage_prefix"], show_progress_bar=True)
    log_done(stage_start)

    # Gate the expanded evaluator against the frozen core10 baseline first.
    stage_start = log_stage(4, 8, "Running frozen core10 retrieval regression...")

    regression_scores, regression_results = score_query_set(
        config["regression_gate"]["query_set"],
        config,
        benchmarks,
        stage_frames,
        semantic_scorer,
        progress_callback=log_detail,
    )

    regression = regression_gate(
        regression_results,
        config,
    )

    write_json(
        reports_root / "regression_gate.json",
        regression,
    )

    if (not regression["passed"]) :
        raise RuntimeError(
            "Core10 regression gate failed. "
            f"See {reports_root / 'regression_gate.json'}"
        )

    log_detail("regression gate: PASS")
    log_done(stage_start)

    score_frames  = [regression_scores]
    result_frames = [regression_results]

    stage_start = log_stage(5, 8, "Scoring requested query sets and aggregating retrieval metrics...")
    remaining_query_sets = [query_set for query_set in mode_spec["query_sets"] if query_set != config["regression_gate"]["query_set"]]

    if (not remaining_query_sets) :
        log_detail("no additional query sets in regression mode")

    for query_set_name in remaining_query_sets :
        log_detail(f"query set: {query_set_name}")

        scores, results = score_query_set(
            query_set_name,
            config,
            benchmarks,
            stage_frames,
            semantic_scorer,
            progress_callback=log_detail,
        )

        score_frames.append(scores)
        result_frames.append(results)

    window_scores = pd.concat(
        score_frames,
        ignore_index=True,
    )

    query_results = pd.concat(
        result_frames,
        ignore_index=True,
    )

    retrieval_metrics = retrieval_metric_tables(
        query_results
    )

    effects = text_view_effects(
        query_results
    )

    query_comparison = build_query_comparison(query_results, config)
    log_done(stage_start)

    allowed_video_ids = analysis_video_ids(
        config,
        benchmarks,
        mode,
    )

    analysis_windows = all_loaded_windows[
        all_loaded_windows["video_id"].isin(
            allowed_video_ids
        )
    ].copy()

    stage_start = log_stage(6, 8, f"Running reliability, duplicate, phrase, and transcript-agreement diagnostics on {len(allowed_video_ids)} videos...")
    log_detail(f"model-window rows: {len(analysis_windows):,}")

    (
        window_diagnostics,
        duplicate_pairs,
        repeated_phrases,
        reference_agreement,
        agreement_summary,
    ) = build_diagnostics(
        analysis_windows,
        config,
        semantic_scorer,
        progress_callback=log_detail,
    )
    log_done(stage_start)

    stage_start = log_stage(7, 8, "Building operational summaries and inspection cases...")
    operational = summarize_operational(
        analysis_windows
    )

    run_summaries, run_summary_records = load_run_summaries(
        config,
        benchmark_keys,
    )

    model_summary = model_summary_table(
        operational,
        postprocess_summary,
        window_diagnostics,
        duplicate_pairs,
        repeated_phrases,
        reference_agreement,
        config,
    )

    inspection_cases = select_inspection_cases(
        mode, query_results, query_comparison, window_diagnostics, duplicate_pairs,
        repeated_phrases, reference_agreement, allowed_video_ids, config,
    )
    log_done(stage_start)

    stage_start = log_stage(8, 8, "Writing evaluation reports and reproducibility manifest...")

    write_csv(
        reports_root / "model_summary.csv",
        model_summary,
    )
    write_csv(
        reports_root / "retrieval_metrics.csv",
        retrieval_metrics,
    )
    write_csv(
        reports_root / "query_results.csv",
        query_results,
    )
    write_csv(
        reports_root / "query_comparison.csv",
        query_comparison,
    )
    write_csv(
        reports_root / "raw_vs_processed.csv",
        effects,
    )
    write_csv(
        reports_root / "window_diagnostics.csv",
        window_diagnostics,
    )
    write_csv(
        reports_root / "reference_agreement.csv",
        reference_agreement,
    )
    write_csv(
        reports_root / "duplicate_pairs.csv",
        duplicate_pairs,
    )
    write_csv(
        reports_root / "repeated_phrases.csv",
        repeated_phrases,
    )
    write_csv(
        reports_root / "operational_summary.csv",
        operational,
    )
    write_csv(
        reports_root / "run_summaries.csv",
        run_summaries,
    )
    write_csv(
        reports_root / "postprocess_consistency.csv",
        postprocess_summary,
    )

    write_json(
        reports_root / "inspection_cases.json",
        inspection_cases,
    )

    if (bool(config["reports"].get("save_window_scores", False))) :
        write_csv(
            reports_root / "window_scores.csv",
            window_scores,
        )

    bundle = build_bundle(
        config=config,
        mode=mode,
        validation=validation,
        regression=regression,
        retrieval_metrics=retrieval_metrics,
        query_comparison=query_comparison,
        effects=effects,
        model_summary=model_summary,
        operational=operational,
        run_summaries=run_summaries,
        postprocess_summary=postprocess_summary,
        agreement_summary=agreement_summary,
        repeated_phrases=repeated_phrases,
        window_diagnostics=window_diagnostics,
        inspection_cases=inspection_cases,
        benchmarks={
            key : benchmarks[key]
            for key in sorted(benchmark_keys)
        },
    )

    write_json(
        reports_root / "stage1_evaluation_bundle.json",
        bundle,
    )

    manifest = build_evaluation_manifest(
        config=config,
        mode=mode,
        benchmark_paths=benchmark_paths,
        benchmarks=benchmarks,
        benchmark_keys=benchmark_keys,
    )

    manifest["run_summary_files"] = run_summary_records

    write_json(reports_root / "evaluation_manifest.json", manifest)
    log_done(stage_start)

    print()
    print("=" * 88)
    print("EVALUATION COMPLETE")
    print("=" * 88)
    print(f"Regression gate: PASS")
    print(f"Query sets:       {', '.join(mode_spec['query_sets'])}")
    print(f"Reports:          {reports_root}")
    print(f"Bundle:           {reports_root / 'stage1_evaluation_bundle.json'}")
    print(f"Manifest:         {reports_root / 'evaluation_manifest.json'}")


if (__name__ == "__main__") :
    main()
