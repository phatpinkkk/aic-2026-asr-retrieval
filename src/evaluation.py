# Relative path: src/evaluation.py
# Purpose: Pure Stage 1 ASR evaluation utilities for validation, retrieval, diagnostics, agreement, and operational summaries.

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from postprocess import (
    accent_fold,
    apply_unit_aliases,
    mark_consecutive_duplicate_windows,
    normalize_for_matching,
    normalize_unicode,
    postprocess_window,
    suppress_consecutive_duplicate_segments,
)


EVALUATION_VERSION = "2.1.0"
SUCCESS_STATUSES   = {"ok", "success"}

POSTPROCESS_WARNING_REASONS = {
    "consecutive_duplicate_segment",
    "known_boilerplate",
    "partial_known_boilerplate",
    "empty_output",
    "consecutive_duplicate_window",
}

POSTPROCESS_REJECTION_REASONS = {
    "dominant_known_boilerplate",
    "known_boilerplate_only_after_removal",
    "consecutive_duplicate_window",
}


ProgressCallback = Callable[[str], None]


def _emit_progress(progress_callback : ProgressCallback | None, message : str) -> None :
    if (progress_callback is not None) :
        progress_callback(message)


# -----------------------------------------------------------------------------
# Generic loading and query normalization
# -----------------------------------------------------------------------------


def load_json(path : Path) -> dict[str, Any] :
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_selected_cases(path : Path) -> dict[str, Any] :
    return load_json(path)


def query_rows(source : dict[str, Any]) -> pd.DataFrame :
    """Normalize either a benchmark manifest or the legacy selected-cases schema."""
    rows = []

    if (isinstance(source.get("queries"), list)) :
        stage_id = source.get("stage_id")
        benchmark_hash = source.get("benchmark_content_hash")

        for query in source.get("queries", []) :
            rows.append({
                "stage_id"               : stage_id,
                "benchmark_content_hash" : benchmark_hash,
                "query_id"               : query["query_id"],
                "query_text"             : query["query_text"],
                "video_id"               : query["video_id"],
                "frame_id"               : query.get("frame_id"),
                "answer_time_s"          : query["answer_time_s"],
                "query_category"         : query.get("query_category", "other"),
                "task_type"              : query.get("task_type", "KIS"),
                "difficulty"             : query.get("difficulty", "unknown"),
                "evaluation_split"       : query.get("evaluation_split", "unspecified"),
                "answer_text"            : query.get("answer_text"),
            })

        return pd.DataFrame(rows)

    for video in source.get("videos", []) :
        for query in video.get("queries", []) :
            rows.append({
                "stage_id"               : source.get("stage_id"),
                "benchmark_content_hash" : source.get("benchmark_content_hash"),
                "query_id"               : query["query_id"],
                "query_text"             : query["query_text"],
                "video_id"               : video["video_id"],
                "frame_id"               : query.get("frame_id"),
                "answer_time_s"          : query["answer_time_s"],
                "query_category"         : query.get("query_category", "other"),
                "task_type"              : query.get("task_type", "KIS"),
                "difficulty"             : query.get("difficulty", "unknown"),
                "evaluation_split"       : query.get("evaluation_split", "unspecified"),
                "answer_text"            : query.get("answer_text"),
            })

    return pd.DataFrame(rows)


def combine_query_rows(manifests : Sequence[dict[str, Any]]) -> pd.DataFrame :
    frames = [query_rows(manifest) for manifest in manifests]
    frames = [frame for frame in frames if not frame.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# -----------------------------------------------------------------------------
# Canonical post-processing and standardized model windows
# -----------------------------------------------------------------------------


def _base_postprocess_reasons(record : dict[str, Any]) -> tuple[list[str], list[str]] :
    warnings = [
        reason
        for reason in (record.get("warning_reasons", []) or [])
        if reason not in POSTPROCESS_WARNING_REASONS
    ]
    rejections = [
        reason
        for reason in (record.get("rejection_reasons", []) or [])
        if reason not in POSTPROCESS_REJECTION_REASONS
    ]
    return warnings, rejections


def _canonicalize_video_records(records : list[dict[str, Any]]) -> dict[str, dict[str, Any]] :
    prepared = []

    for record in sorted(records, key=lambda item : (item.get("start_s", 0.0), item.get("window_id", ""))) :
        warnings, rejections = _base_postprocess_reasons(record)
        working = dict(record)
        working["warning_reasons"]   = warnings
        working["rejection_reasons"] = rejections
        working.update(postprocess_window(working))
        prepared.append(working)

    marked = mark_consecutive_duplicate_windows(prepared)
    return {item.get("window_id") : item for item in marked if item.get("window_id")}


def _postprocess_transformations(record : dict[str, Any], canonical : dict[str, Any]) -> list[str] :
    raw_text        = str(record.get("raw_text", "") or "")
    native_segments = record.get("native_segments", []) or []
    deduplicated, segment_warnings = suppress_consecutive_duplicate_segments(native_segments)
    source_text = deduplicated or normalize_unicode(raw_text)
    normalized  = normalize_for_matching(source_text)
    aliased     = apply_unit_aliases(source_text)

    warnings   = set(canonical.get("warning_reasons", []) or [])
    rejections = set(canonical.get("rejection_reasons", []) or [])
    changes    = []

    if ("consecutive_duplicate_segment" in segment_warnings or "consecutive_duplicate_segment" in warnings) :
        changes.append("segment_duplicate_removed")
    if (aliased != normalized) :
        changes.append("unit_alias_applied")
    if ("partial_known_boilerplate" in warnings) :
        changes.append("partial_boilerplate_removed")
    if ("dominant_known_boilerplate" in rejections) :
        changes.append("dominant_boilerplate_rejected")
    if ("consecutive_duplicate_window" in rejections) :
        changes.append("duplicate_window_rejected")
    if (not normalize_unicode(raw_text)) :
        changes.append("empty_output")
    if (not changes) :
        changes.append("normalization_only")

    return changes


def _audio_file_map(benchmark : dict[str, Any] | None) -> dict[str, dict[str, Any]] :
    if (not benchmark) :
        return {}
    return {item["video_id"] : item for item in benchmark.get("audio_files", []) if item.get("video_id")}


def _standardized_window_row(
    model_id : str,
    payload : dict[str, Any],
    expected : dict[str, Any],
    record : dict[str, Any] | None,
    canonical : dict[str, Any] | None,
    result_path : Path,
    benchmark : dict[str, Any] | None,
) -> dict[str, Any] :
    record     = record or {}
    canonical  = canonical or {}
    has_record = bool(record)

    expected_start = float(expected["start_s"])
    expected_end   = float(expected["end_s"])

    start = (
        float(record.get("start_s", expected_start))
        if has_record
        else expected_start
    )

    end = (
        float(record.get("end_s", expected_end))
        if has_record
        else expected_end
    )

    audio_map = _audio_file_map(benchmark)
    audio     = audio_map.get(expected["video_id"], {})

    stored_warnings      = list(record.get("warning_reasons", []) or [])
    stored_rejections    = list(record.get("rejection_reasons", []) or [])
    canonical_warnings   = list(canonical.get("warning_reasons", []) or [])
    canonical_rejections = list(canonical.get("rejection_reasons", []) or [])

    observed_benchmark_hash = payload.get("benchmark_content_hash")
    expected_benchmark_hash = (benchmark or {}).get("benchmark_content_hash")
    observed_stage_id        = payload.get("stage_id")
    expected_stage_id        = (benchmark or {}).get("stage_id")

    return {
        "stage_id"                     : observed_stage_id or expected_stage_id,
        "observed_stage_id"            : observed_stage_id,
        "expected_stage_id"            : expected_stage_id,
        "result_schema_version"        : payload.get("schema_version"),
        "benchmark_content_hash"       : observed_benchmark_hash,
        "expected_benchmark_content_hash": expected_benchmark_hash,
        "benchmark_hash_source"        : (
            "payload"
            if observed_benchmark_hash is not None
            else "legacy_missing"
        ),
        "model_id"                     : model_id,
        "video_id"                     : (
            record.get("video_id", expected["video_id"])
            if has_record
            else expected["video_id"]
        ),
        "window_id"                    : expected["window_id"],
        "window_index"                 : (
            record.get("window_index", expected.get("window_index"))
            if has_record
            else expected.get("window_index")
        ),
        "start_s"                      : start,
        "end_s"                        : end,
        "sample_start"                 : (
            record.get("sample_start", expected.get("sample_start"))
            if has_record
            else expected.get("sample_start")
        ),
        "sample_end"                   : (
            record.get("sample_end", expected.get("sample_end"))
            if has_record
            else expected.get("sample_end")
        ),
        "duration_samples"             : (
            record.get("duration_samples", expected.get("duration_samples"))
            if has_record
            else expected.get("duration_samples")
        ),
        "duration_s"                   : end - start,
        "status"                       : record.get("status", "missing"),
        "runtime_s"                    : record.get("runtime_s"),
        "peak_gpu_memory_bytes"        : record.get("peak_gpu_memory_bytes"),
        "peak_reserved_memory_bytes"   : record.get("peak_reserved_memory_bytes"),
        "raw_text"                     : str(record.get("raw_text", "") or ""),
        "stored_retrieval_text"        : str(record.get("retrieval_text", "") or ""),
        "stored_normalized_text"       : str(record.get("normalized_text", "") or ""),
        "stored_accent_folded_text"    : str(record.get("accent_folded_text", "") or ""),
        "stored_warning_reasons"       : stored_warnings,
        "stored_rejection_reasons"     : stored_rejections,
        "canonical_retrieval_text"     : str(canonical.get("retrieval_text", "") or ""),
        "canonical_normalized_text"    : str(canonical.get("normalized_text", "") or ""),
        "canonical_accent_folded_text" : str(canonical.get("accent_folded_text", "") or ""),
        "canonical_warning_reasons"    : canonical_warnings,
        "canonical_rejection_reasons"  : canonical_rejections,
        "postprocess_transformations"  : _postprocess_transformations(record, canonical) if record else [],
        "postprocess_version"          : canonical.get("postprocess_version") or record.get("postprocess_version"),
        "postprocess_matches_stored"   : bool(record) and (
            str(record.get("retrieval_text", "") or "") == str(canonical.get("retrieval_text", "") or "")
            and sorted(stored_warnings) == sorted(canonical_warnings)
            and sorted(stored_rejections) == sorted(canonical_rejections)
        ),
        "retrieval_text"               : str(canonical.get("retrieval_text", "") or ""),
        "normalized_text"              : str(canonical.get("normalized_text", "") or ""),
        "accent_folded_text"           : str(canonical.get("accent_folded_text", "") or ""),
        "warning_reasons"              : canonical_warnings,
        "rejection_reasons"            : canonical_rejections,
        "boilerplate_coverage"         : canonical.get("boilerplate_coverage", record.get("boilerplate_coverage")),
        "boilerplate_matches"          : canonical.get("boilerplate_matches", record.get("boilerplate_matches", [])),
        "native_segments"              : record.get("native_segments", []) or [],
        "native_segment_count"         : len(record.get("native_segments", []) or []),
        "error"                        : record.get("error"),
        "wav_hash"                     : record.get("wav_hash"),
        "expected_wav_hash"            : audio.get("wav_sha256"),
        "window_pcm_hash"              : record.get("window_pcm_hash"),
        "window_policy_hash"           : record.get("window_policy_hash"),
        "expected_window_policy_hash"  : (benchmark or {}).get("window_policy_hash"),
        "model_configuration_hash"     : record.get("model_configuration_hash") or payload.get("model_configuration_hash"),
        "adapter_hash"                 : record.get("adapter_hash") or payload.get("adapter_hash"),
        "adapter_version"              : record.get("adapter_version") or payload.get("adapter_version"),
        "model_revision"               : record.get("model_revision") or payload.get("model_revision"),
        "result_path"                  : str(result_path),
    }


def load_model_output_windows(
    model_id : str,
    output_dir : Path,
    expected_windows : list[dict[str, Any]],
    benchmark : dict[str, Any] | None = None,
    progress_callback : ProgressCallback | None = None,
) -> pd.DataFrame :
    """Load one model directory and create one standardized row per expected window."""
    output_dir = Path(output_dir)
    payloads   = {}

    expected_video_ids = sorted({item["video_id"] for item in expected_windows})

    if (output_dir.exists()) :
        for video_id in expected_video_ids :
            result_path = output_dir / f"{video_id}.json"
            if (not result_path.exists()) :
                continue
            payloads[video_id] = (load_json(result_path), result_path)

    rows = []

    total_videos = len(expected_video_ids)

    for video_number, video_id in enumerate(expected_video_ids, start=1) :
        _emit_progress(progress_callback, f"{model_id}: loading {video_id} ({video_number}/{total_videos})")
        payload, result_path = payloads.get(video_id, ({}, output_dir / f"{video_id}.json"))
        records = payload.get("windows", []) or []
        record_map    = {item.get("window_id") : item for item in records if item.get("window_id")}
        canonical_map = _canonicalize_video_records(records) if records else {}

        for expected in [item for item in expected_windows if item["video_id"] == video_id] :
            window_id = expected["window_id"]
            rows.append(_standardized_window_row(
                model_id=model_id,
                payload=payload,
                expected=expected,
                record=record_map.get(window_id),
                canonical=canonical_map.get(window_id),
                result_path=result_path,
                benchmark=benchmark,
            ))

    return pd.DataFrame(rows)


def load_model_windows(outputs_root : Path, expected_windows : list[dict[str, Any]]) -> pd.DataFrame :
    """Legacy wrapper: load each direct model subdirectory below one output root."""
    outputs_root = Path(outputs_root)
    if (not outputs_root.exists()) :
        return pd.DataFrame()

    frames = []
    for model_dir in sorted(path for path in outputs_root.iterdir() if path.is_dir()) :
        frame = load_model_output_windows(model_dir.name, model_dir, expected_windows, benchmark=None)
        if (not frame.empty) :
            frames.append(frame)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def postprocess_consistency_summary(windows : pd.DataFrame) -> pd.DataFrame :
    if (windows.empty) :
        return pd.DataFrame()

    rows = []
    for (model_id, stage_id), group in windows.groupby(["model_id", "stage_id"], dropna=False, sort=True) :
        present = group[group["status"] != "missing"]
        rows.append({
            "model_id"                     : model_id,
            "stage_id"                     : stage_id,
            "present_window_count"         : len(present),
            "postprocess_match_count"      : int(present["postprocess_matches_stored"].fillna(False).sum()),
            "postprocess_mismatch_count"   : int((~present["postprocess_matches_stored"].fillna(False)).sum()),
            "postprocess_match_rate"       : float(present["postprocess_matches_stored"].fillna(False).mean()) if len(present) else None,
            "stored_postprocess_versions"  : sorted({str(value) for value in present["postprocess_version"].dropna().tolist()}),
        })

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Benchmark and output validation
# -----------------------------------------------------------------------------


def validate_benchmark_manifest(benchmark : dict[str, Any], tolerance : float = 1e-6) -> dict[str, Any] :
    errors   = []
    warnings = []
    windows  = benchmark.get("windows", []) or []
    queries  = benchmark.get("queries", []) or []
    videos   = benchmark.get("selected_video_ids", []) or []
    audio    = benchmark.get("audio_files", []) or []

    if (benchmark.get("video_count") != len(videos)) :
        errors.append(f"video_count={benchmark.get('video_count')} but observed {len(videos)} selected_video_ids")
    if (benchmark.get("query_count") != len(queries)) :
        errors.append(f"query_count={benchmark.get('query_count')} but observed {len(queries)} queries")
    if (benchmark.get("window_count") != len(windows)) :
        errors.append(f"window_count={benchmark.get('window_count')} but observed {len(windows)} windows")

    video_ids  = [item for item in videos]
    query_ids  = [item.get("query_id") for item in queries]
    window_ids = [item.get("window_id") for item in windows]

    if (len(video_ids) != len(set(video_ids))) :
        errors.append("duplicate selected_video_ids")
    if (len(query_ids) != len(set(query_ids))) :
        errors.append("duplicate query_ids")
    if (len(window_ids) != len(set(window_ids))) :
        errors.append("duplicate window_ids")

    audio_video_ids = [item.get("video_id") for item in audio]
    if (audio and len(audio_video_ids) != len(set(audio_video_ids))) :
        errors.append("duplicate audio_file video_ids")

    sample_rates = sorted({item.get("wav_sample_rate") for item in audio if item.get("wav_sample_rate") is not None})
    if (len(sample_rates) > 1) :
        errors.append(f"multiple audio sample rates found: {sample_rates}")

    sample_rate = sample_rates[0] if len(sample_rates) == 1 else None

    for item in windows :
        start        = float(item["start_s"])
        end          = float(item["end_s"])
        sample_start = int(item["sample_start"])
        sample_end   = int(item["sample_end"])
        duration     = int(item["duration_samples"])

        if (end <= start) :
            errors.append(f"{item.get('window_id')}: end_s <= start_s")
        if (sample_end <= sample_start) :
            errors.append(f"{item.get('window_id')}: sample_end <= sample_start")
        if (sample_end - sample_start != duration) :
            errors.append(f"{item.get('window_id')}: duration_samples mismatch")
        if (sample_rate and abs((end - start) - duration / float(sample_rate)) > tolerance) :
            errors.append(f"{item.get('window_id')}: time/sample duration mismatch")

    query_video_counts = Counter(item.get("video_id") for item in queries)
    if (queries and any(count != 1 for count in query_video_counts.values())) :
        warnings.append("benchmark does not contain exactly one query per queried video")

    splits = benchmark.get("evaluation_splits", {}) or {}
    if (splits) :
        split_sets = {name : set(values) for name, values in splits.items()}
        names = sorted(split_sets)
        for index, left in enumerate(names) :
            for right in names[index + 1 : ] :
                overlap = sorted(split_sets[left].intersection(split_sets[right]))
                if (overlap) :
                    errors.append(f"evaluation split overlap between {left} and {right}: {overlap}")

    return {
        "passed"                : not errors,
        "stage_id"              : benchmark.get("stage_id"),
        "benchmark_content_hash": benchmark.get("benchmark_content_hash"),
        "video_count"           : len(videos),
        "query_count"           : len(queries),
        "window_count"          : len(windows),
        "sample_rates"          : sample_rates,
        "errors"                : errors,
        "warnings"              : warnings,
    }


def validate_model_windows_against_benchmark(
    windows : pd.DataFrame,
    benchmark : dict[str, Any],
    model_id : str,
    require_complete : bool = True,
    tolerance : float = 1e-6,
) -> dict[str, Any] :
    errors   = []
    warnings = []

    model_windows = (
        windows[windows["model_id"] == model_id].copy()
        if not windows.empty
        else pd.DataFrame()
    )

    expected_windows = benchmark.get("windows", []) or []
    expected_map      = {item["window_id"] : item for item in expected_windows}
    actual_map        = {row["window_id"] : row for _, row in model_windows.iterrows()}

    missing_ids = sorted(set(expected_map) - set(actual_map))
    extra_ids   = sorted(set(actual_map) - set(expected_map))

    if (missing_ids) :
        errors.append(
            f"{len(missing_ids)} expected windows are absent from the standardized table"
        )

    if (extra_ids) :
        errors.append(
            f"{len(extra_ids)} unexpected windows are present"
        )

    missing_records               = []
    legacy_missing_benchmark_hash = []
    legacy_missing_stage_id       = []

    expected_benchmark_hash = benchmark.get("benchmark_content_hash")
    expected_policy_hash    = benchmark.get("window_policy_hash")
    expected_stage_id       = benchmark.get("stage_id")

    for window_id, expected in expected_map.items() :
        if (window_id not in actual_map) :
            continue

        row = actual_map[window_id]

        if (row["status"] == "missing") :
            missing_records.append(window_id)
            continue

        direct_checks = {
            "video_id"         : (str(row["video_id"]), str(expected["video_id"])),
            "window_index"     : (row["window_index"], expected.get("window_index")),
            "sample_start"     : (row["sample_start"], expected.get("sample_start")),
            "sample_end"       : (row["sample_end"], expected.get("sample_end")),
            "duration_samples" : (row["duration_samples"], expected.get("duration_samples")),
        }

        for field, (actual, target) in direct_checks.items() :
            if (pd.isna(actual) and target is None) :
                continue

            if (actual != target) :
                errors.append(
                    f"{window_id}: {field} mismatch ({actual!r} != {target!r})"
                )

        if (abs(float(row["start_s"]) - float(expected["start_s"])) > tolerance) :
            errors.append(
                f"{window_id}: start_s mismatch"
            )

        if (abs(float(row["end_s"]) - float(expected["end_s"])) > tolerance) :
            errors.append(
                f"{window_id}: end_s mismatch"
            )

        observed_stage_id = row.get("observed_stage_id")

        if (expected_stage_id) :
            if (observed_stage_id is None) :
                legacy_missing_stage_id.append(window_id)

            elif (str(observed_stage_id) != str(expected_stage_id)) :
                errors.append(
                    f"{window_id}: stage_id mismatch "
                    f"({observed_stage_id!r} != {expected_stage_id!r})"
                )

        observed_benchmark_hash = row.get("benchmark_content_hash")

        if (expected_benchmark_hash) :
            if (observed_benchmark_hash is None) :
                schema_version = str(
                    row.get("result_schema_version")
                    or ""
                )

                if (schema_version in {"1.0", "1.1"}) :
                    legacy_missing_benchmark_hash.append(window_id)

                else :
                    errors.append(
                        f"{window_id}: benchmark_content_hash missing "
                        f"from non-legacy output schema {schema_version!r}"
                    )

            elif (observed_benchmark_hash != expected_benchmark_hash) :
                errors.append(
                    f"{window_id}: benchmark_content_hash mismatch"
                )

        if (
            expected_policy_hash
            and row.get("window_policy_hash") != expected_policy_hash
        ) :
            errors.append(
                f"{window_id}: window_policy_hash mismatch"
            )

        expected_wav_hash = row.get("expected_wav_hash")

        if (
            expected_wav_hash
            and row.get("wav_hash") != expected_wav_hash
        ) :
            errors.append(
                f"{window_id}: wav_hash mismatch"
            )

        if (row.get("model_id") != model_id) :
            errors.append(
                f"{window_id}: model_id mismatch"
            )

    if (missing_records and require_complete) :
        errors.append(
            f"{len(missing_records)} expected window records are missing"
        )

    elif (missing_records) :
        warnings.append(
            f"{len(missing_records)} expected window records are currently missing"
        )

    if (legacy_missing_stage_id) :
        warnings.append(
            f"{len(legacy_missing_stage_id)} windows come from legacy output "
            "whose video payload does not store stage_id; the benchmark stage_id "
            "is used only as standardized context."
        )

    if (legacy_missing_benchmark_hash) :
        warnings.append(
            f"{len(legacy_missing_benchmark_hash)} windows come from legacy "
            "schema 1.0/1.1 output whose video payload does not store "
            "benchmark_content_hash. The hash is unavailable as direct output "
            "provenance; timing/sample metadata, WAV hash, window-policy hash, "
            "and cross-model PCM identity remain validated."
        )

    present = (
        model_windows[
            model_windows["status"] != "missing"
        ]
        if not model_windows.empty
        else pd.DataFrame()
    )

    provenance = {}

    for field in [
        "result_schema_version",
        "model_revision",
        "model_configuration_hash",
        "adapter_version",
        "adapter_hash",
    ] :
        provenance[field] = (
            sorted(
                {
                    str(value)
                    for value in present[field].dropna().tolist()
                }
            )
            if field in present
            else []
        )

        if (len(provenance[field]) > 1) :
            warnings.append(
                f"multiple {field} values found within model/stage output"
            )

    return {
        "passed"                         : not errors,
        "model_id"                       : model_id,
        "stage_id"                       : benchmark.get("stage_id"),
        "expected_window_count"          : len(expected_windows),
        "present_window_count"           : int(
            (model_windows["status"] != "missing").sum()
        ) if not model_windows.empty else 0,
        "missing_record_count"           : len(missing_records),
        "missing_window_ids"             : missing_records,
        "unexpected_window_ids"          : extra_ids,
        "legacy_missing_stage_id_count"  : len(legacy_missing_stage_id),
        "legacy_missing_benchmark_hash_count": len(
            legacy_missing_benchmark_hash
        ),
        "provenance"                     : provenance,
        "errors"                         : errors,
        "warnings"                       : warnings,
    }


def compare_model_window_identity(
    windows : pd.DataFrame,
    model_a : str,
    model_b : str,
) -> dict[str, Any] :
    left  = windows[windows["model_id"] == model_a].copy()
    right = windows[windows["model_id"] == model_b].copy()

    columns = ["stage_id", "window_id", "video_id", "start_s", "end_s", "sample_start", "sample_end", "window_pcm_hash"]
    left  = left[columns].rename(columns={column : f"{column}_a" for column in columns if column not in {"stage_id", "window_id"}})
    right = right[columns].rename(columns={column : f"{column}_b" for column in columns if column not in {"stage_id", "window_id"}})
    merged = left.merge(right, on=["stage_id", "window_id"], how="outer", indicator=True)

    errors = []
    only_a = merged[merged["_merge"] == "left_only"]["window_id"].tolist()
    only_b = merged[merged["_merge"] == "right_only"]["window_id"].tolist()

    if (only_a) :
        errors.append(f"{len(only_a)} windows exist only for {model_a}")
    if (only_b) :
        errors.append(f"{len(only_b)} windows exist only for {model_b}")

    both = merged[merged["_merge"] == "both"]
    mismatch_counts = {}

    for field in ["video_id", "start_s", "end_s", "sample_start", "sample_end", "window_pcm_hash"] :
        left_field  = f"{field}_a"
        right_field = f"{field}_b"

        if (field in {"start_s", "end_s"}) :
            mismatch = (pd.to_numeric(both[left_field], errors="coerce") - pd.to_numeric(both[right_field], errors="coerce")).abs() > 1e-6
        else :
            comparable = both[left_field].notna() & both[right_field].notna()
            mismatch = comparable & (both[left_field] != both[right_field])

        mismatch_counts[field] = int(mismatch.sum())
        if (mismatch.any()) :
            errors.append(f"{field} differs for {int(mismatch.sum())} shared windows")

    return {
        "passed"          : not errors,
        "model_a"         : model_a,
        "model_b"         : model_b,
        "shared_windows"  : len(both),
        "only_model_a"    : only_a,
        "only_model_b"    : only_b,
        "mismatch_counts" : mismatch_counts,
        "errors"          : errors,
    }


def validate_benchmark_union(manifests : Sequence[dict[str, Any]]) -> dict[str, Any] :
    errors   = []
    warnings = []
    video_sets  = [set(manifest.get("selected_video_ids", [])) for manifest in manifests]
    query_sets  = [set(manifest.get("selected_query_ids", [])) for manifest in manifests]
    window_sets = [set(item.get("window_id") for item in manifest.get("windows", [])) for manifest in manifests]

    for index in range(len(manifests)) :
        for other in range(index + 1, len(manifests)) :
            video_overlap  = sorted(video_sets[index].intersection(video_sets[other]))
            query_overlap  = sorted(query_sets[index].intersection(query_sets[other]))
            window_overlap = sorted(window_sets[index].intersection(window_sets[other]))

            if (video_overlap) :
                errors.append(f"benchmark {index} and {other} share video IDs: {video_overlap}")
            if (query_overlap) :
                errors.append(f"benchmark {index} and {other} share query IDs: {query_overlap}")
            if (window_overlap) :
                errors.append(f"benchmark {index} and {other} share window IDs")

    policies = [manifest.get("window_policy") for manifest in manifests]
    non_null_policies = [policy for policy in policies if policy is not None]
    if (non_null_policies and any(policy != non_null_policies[0] for policy in non_null_policies[1 : ])) :
        errors.append("benchmark window_policy values differ")

    sample_rate_sets = []
    for manifest in manifests :
        rates = {item.get("wav_sample_rate") for item in manifest.get("audio_files", []) if item.get("wav_sample_rate") is not None}
        sample_rate_sets.append(rates)

    combined_rates = set().union(*sample_rate_sets) if sample_rate_sets else set()
    if (len(combined_rates) > 1) :
        errors.append(f"benchmark audio sample rates differ: {sorted(combined_rates)}")

    return {
        "passed"             : not errors,
        "video_count"        : sum(len(values) for values in video_sets),
        "query_count"        : sum(len(values) for values in query_sets),
        "window_count"       : sum(len(values) for values in window_sets),
        "sample_rates"       : sorted(combined_rates),
        "window_policy"      : non_null_policies[0] if non_null_policies else None,
        "errors"             : errors,
        "warnings"           : warnings,
    }


# -----------------------------------------------------------------------------
# Frozen lexical + semantic retrieval scoring
# -----------------------------------------------------------------------------


def _fit_lexical_vectorizers(queries : list[str], ngram_range : tuple[int, int] = (3, 5)) -> tuple[TfidfVectorizer, TfidfVectorizer] :
    preserving = TfidfVectorizer(analyzer="char_wb", ngram_range=ngram_range, lowercase=False, min_df=1)
    folded     = TfidfVectorizer(analyzer="char_wb", ngram_range=ngram_range, lowercase=False, min_df=1)

    preserving.fit([normalize_for_matching(text) for text in queries])
    folded.fit([accent_fold(text) for text in queries])
    return preserving, folded


class SemanticScorer :
    def __init__(
        self,
        model_name : str,
        revision : str | None = None,
        query_prefix : str = "query: ",
        passage_prefix : str = "passage: ",
        show_progress_bar : bool = False,
    ) :
        from sentence_transformers import SentenceTransformer

        self.model_name     = model_name
        self.revision       = revision
        self.query_prefix   = query_prefix
        self.passage_prefix   = passage_prefix
        self.show_progress_bar = show_progress_bar
        self.encoder           = SentenceTransformer(model_name, revision=revision)
        self._query_cache   = {}
        self._passage_cache = {}

    def query_embeddings(self, query_texts : list[str]) -> np.ndarray :
        cache_key = tuple(query_texts)
        if (cache_key not in self._query_cache) :
            embeddings = self.encoder.encode(
                [self.query_prefix + text for text in query_texts],
                normalize_embeddings=True,
                show_progress_bar=self.show_progress_bar,
            )
            self._query_cache[cache_key] = np.asarray(embeddings)

        return self._query_cache[cache_key]

    def passage_embeddings(self, passage_texts : list[str], use_cache : bool = True) -> np.ndarray :
        cache_key = tuple(passage_texts)
        if (use_cache and cache_key in self._passage_cache) :
            return self._passage_cache[cache_key]

        embeddings = self.encoder.encode(
            [self.passage_prefix + text for text in passage_texts],
            normalize_embeddings=True,
            show_progress_bar=self.show_progress_bar,
        )
        values = np.asarray(embeddings)

        if (use_cache) :
            self._passage_cache[cache_key] = values

        return values

    def score(self, query_texts : list[str], passage_texts : list[str]) -> np.ndarray :
        if (not passage_texts) :
            return np.zeros((len(query_texts), 0), dtype=np.float32)

        query_embeddings   = self.query_embeddings(query_texts)
        passage_embeddings = self.passage_embeddings(passage_texts)
        scores = query_embeddings @ passage_embeddings.T
        return np.clip(scores, 0.0, 1.0)

    def passage_similarity(self, left_texts : list[str], right_texts : list[str]) -> np.ndarray :
        if (len(left_texts) != len(right_texts)) :
            raise ValueError("left_texts and right_texts must have the same length")
        if (not left_texts) :
            return np.zeros(0, dtype=np.float32)

        left  = self.passage_embeddings(left_texts, use_cache=False)
        right = self.passage_embeddings(right_texts, use_cache=False)
        return np.clip(np.sum(left * right, axis=1), -1.0, 1.0)

    def metadata(self) -> dict[str, Any] :
        return {
            "model_name"     : self.model_name,
            "revision"       : self.revision,
            "query_prefix"   : self.query_prefix,
            "passage_prefix"   : self.passage_prefix,
            "show_progress_bar" : self.show_progress_bar,
        }


def retrieval_eligibility(
    window : pd.Series,
    text_field : str,
    view_name : str,
    rejection_field : str = "rejection_reasons",
) -> tuple[bool, str | None] :
    if (str(window.get("status", "")) not in SUCCESS_STATUSES) :
        return False, f"status_{window.get('status', 'missing')}"

    text = str(window.get(text_field, "") or "").strip()
    if (not text) :
        return False, "empty_text"

    if (view_name.startswith("processed") and bool(window.get(rejection_field, []))) :
        return False, "processed_rejection"

    return True, None


def score_windows(
    windows : pd.DataFrame,
    queries : pd.DataFrame,
    model_id : str,
    text_field : str,
    view_name : str,
    semantic_scorer : SemanticScorer,
    silver_radius_s : float = 60.0,
    minimum_overlap_s : float = 30.0,
    lexical_weight : float = 0.5,
    semantic_weight : float = 0.5,
    lexical_ngram_range : tuple[int, int] = (3, 5),
    rejection_field : str = "rejection_reasons",
    progress_callback : ProgressCallback | None = None,
) -> pd.DataFrame :
    _emit_progress(progress_callback, f"{model_id}/{view_name}: preparing retrieval inputs")
    model_windows = windows[windows["model_id"] == model_id].copy().reset_index(drop=True)
    if (model_windows.empty or queries.empty) :
        return pd.DataFrame()

    total_weight = float(lexical_weight) + float(semantic_weight)
    if (total_weight <= 0) :
        raise ValueError("lexical_weight + semantic_weight must be positive")

    lexical_weight  = float(lexical_weight) / total_weight
    semantic_weight = float(semantic_weight) / total_weight

    model_windows[text_field] = model_windows[text_field].fillna("").astype(str)
    query_texts   = queries["query_text"].astype(str).tolist()
    passage_texts = model_windows[text_field].tolist()
    preserving, folded = _fit_lexical_vectorizers(query_texts, ngram_range=lexical_ngram_range)

    shape           = (len(query_texts), len(passage_texts))
    lexical_scores  = np.zeros(shape, dtype=np.float32)
    semantic_scores = np.zeros(shape, dtype=np.float32)
    eligibility = [
        retrieval_eligibility(row, text_field, view_name, rejection_field=rejection_field)
        for _, row in model_windows.iterrows()
    ]
    valid_indices = [index for index, (is_valid, _) in enumerate(eligibility) if is_valid]

    if (valid_indices) :
        _emit_progress(progress_callback, f"{model_id}/{view_name}: scoring {len(query_texts)} queries against {len(valid_indices)} valid windows")
        valid_passages     = [passage_texts[index] for index in valid_indices]
        query_preserving   = preserving.transform([normalize_for_matching(text) for text in query_texts])
        passage_preserving = preserving.transform([normalize_for_matching(text) for text in valid_passages])
        query_folded       = folded.transform([accent_fold(text) for text in query_texts])
        passage_folded     = folded.transform([accent_fold(text) for text in valid_passages])
        lexical_preserving = cosine_similarity(query_preserving, passage_preserving)
        lexical_folded     = cosine_similarity(query_folded, passage_folded)
        valid_lexical      = np.maximum(lexical_preserving, lexical_folded)
        valid_semantic     = semantic_scorer.score(query_texts, valid_passages)
        lexical_scores[:, valid_indices]  = valid_lexical
        semantic_scores[:, valid_indices] = valid_semantic

    _emit_progress(progress_callback, f"{model_id}/{view_name}: assembling retrieval rows")
    final_scores = lexical_weight * lexical_scores + semantic_weight * semantic_scores
    rows = []

    for query_index, query in queries.reset_index(drop=True).iterrows() :
        zone_start = float(query["answer_time_s"]) - silver_radius_s
        zone_end   = float(query["answer_time_s"]) + silver_radius_s

        for window_index, window in model_windows.iterrows() :
            overlap = max(0.0, min(float(window["end_s"]), zone_end) - max(float(window["start_s"]), zone_start))
            relevant = bool(window["video_id"] == query["video_id"] and overlap >= minimum_overlap_s)
            is_scored, zero_reason = eligibility[window_index]

            rows.append({
                "model_id"          : model_id,
                "view"              : view_name,
                "query_id"          : query["query_id"],
                "query_text"        : query["query_text"],
                "correct_video"     : query["video_id"],
                "frame_id"          : query.get("frame_id"),
                "answer_time_s"     : query["answer_time_s"],
                "query_category"    : query.get("query_category", "other"),
                "task_type"         : query.get("task_type", "KIS"),
                "difficulty"        : query.get("difficulty", "unknown"),
                "evaluation_split"  : query.get("evaluation_split", "unspecified"),
                "answer_text"       : query.get("answer_text"),
                "video_id"          : window["video_id"],
                "window_id"         : window["window_id"],
                "start_s"           : window["start_s"],
                "end_s"             : window["end_s"],
                "status"            : window["status"],
                "error"             : window.get("error"),
                "warning_reasons"   : window.get("warning_reasons", []),
                "rejection_reasons" : window.get(rejection_field, []),
                "is_scored"         : is_scored,
                "score_zero_reason" : zero_reason,
                "lexical_score"     : float(lexical_scores[query_index, window_index]),
                "semantic_score"    : float(semantic_scores[query_index, window_index]),
                "final_score"       : float(final_scores[query_index, window_index]),
                "silver_overlap_s"  : overlap,
                "is_relevant"       : relevant,
                "text_field"        : text_field,
            })

    _emit_progress(progress_callback, f"{model_id}/{view_name}: retrieval scoring complete")
    return pd.DataFrame(rows)


def worst_tied_ranks(scores : pd.Series, tolerance : float = 1e-12) -> pd.Series :
    ordered = sorted(((index, float(score)) for index, score in scores.items()), key=lambda item : (-item[1], str(item[0])))
    ranks = {}
    start = 0

    while (start < len(ordered)) :
        end        = start + 1
        base_score = ordered[start][1]

        while (end < len(ordered) and abs(ordered[end][1] - base_score) <= tolerance) :
            end += 1

        worst_rank = end
        for index, _ in ordered[start : end] :
            ranks[index] = worst_rank
        start = end

    return pd.Series(ranks, dtype="int64")


def aggregate_query_metrics(scores : pd.DataFrame) -> pd.DataFrame :
    if (scores.empty) :
        return pd.DataFrame()

    rows = []

    for (model_id, view, query_id), group in scores.groupby(["model_id", "view", "query_id"], sort=True) :
        correct_video = group["correct_video"].iloc[0]
        story_group   = group[group["video_id"] == correct_video].copy()
        story_group["metric_rank"] = worst_tied_ranks(story_group["final_score"])
        story_group = story_group.sort_values(
            ["final_score", "start_s", "window_id"],
            ascending=[False, True, True],
            kind="mergesort",
        )

        relevant_rows  = story_group[story_group["is_relevant"]]
        irrelevant_rows = story_group[~story_group["is_relevant"]]
        relevant_ranks = relevant_rows["metric_rank"].tolist()
        first_rank     = min(relevant_ranks) if relevant_ranks else None
        best_relevant_score   = float(relevant_rows["final_score"].max()) if not relevant_rows.empty else None
        best_irrelevant_score = float(irrelevant_rows["final_score"].max()) if not irrelevant_rows.empty else None
        story_margin = (
            best_relevant_score - best_irrelevant_score
            if best_relevant_score is not None and best_irrelevant_score is not None
            else None
        )

        video_scores = group.groupby("video_id", as_index=False)["final_score"].max()
        video_scores["metric_rank"] = worst_tied_ranks(video_scores.set_index("video_id")["final_score"]).reindex(video_scores["video_id"]).to_numpy()
        video_scores = video_scores.sort_values(["final_score", "video_id"], ascending=[False, True], kind="mergesort")
        video_match = video_scores[video_scores["video_id"] == correct_video]
        video_rank  = int(video_match["metric_rank"].iloc[0]) if not video_match.empty else None
        correct_video_score = float(video_match["final_score"].iloc[0]) if not video_match.empty else None
        wrong_videos = video_scores[video_scores["video_id"] != correct_video]
        best_wrong_video_score = float(wrong_videos["final_score"].max()) if not wrong_videos.empty else None
        video_margin = (
            correct_video_score - best_wrong_video_score
            if correct_video_score is not None and best_wrong_video_score is not None
            else None
        )

        top_story = story_group.iloc[0] if not story_group.empty else None
        top_video = video_scores.iloc[0] if not video_scores.empty else None

        rows.append({
            "model_id"              : model_id,
            "view"                  : view,
            "query_id"              : query_id,
            "query_text"            : group["query_text"].iloc[0],
            "query_category"        : group["query_category"].iloc[0],
            "task_type"             : group["task_type"].iloc[0],
            "difficulty"            : group["difficulty"].iloc[0],
            "evaluation_split"      : group["evaluation_split"].iloc[0],
            "answer_text"           : group["answer_text"].iloc[0],
            "correct_video"         : correct_video,
            "frame_id"              : group["frame_id"].iloc[0],
            "answer_time_s"         : group["answer_time_s"].iloc[0],
            "first_relevant_rank"   : first_rank,
            "story_recall_at_1"     : int(first_rank is not None and first_rank <= 1),
            "story_recall_at_3"     : int(first_rank is not None and first_rank <= 3),
            "story_recall_at_5"     : int(first_rank is not None and first_rank <= 5),
            "story_recall_at_10"    : int(first_rank is not None and first_rank <= 10),
            "story_rr"              : 1.0 / first_rank if first_rank else 0.0,
            "best_relevant_score"   : best_relevant_score,
            "best_irrelevant_score" : best_irrelevant_score,
            "story_score_margin"    : story_margin,
            "video_rank"            : video_rank,
            "video_recall_at_1"     : int(video_rank is not None and video_rank <= 1),
            "video_recall_at_3"     : int(video_rank is not None and video_rank <= 3),
            "video_recall_at_5"     : int(video_rank is not None and video_rank <= 5),
            "video_recall_at_10"    : int(video_rank is not None and video_rank <= 10),
            "video_recall_at_20"    : int(video_rank is not None and video_rank <= 20),
            "video_rr"              : 1.0 / video_rank if video_rank else 0.0,
            "correct_video_score"   : correct_video_score,
            "best_wrong_video_score": best_wrong_video_score,
            "video_score_margin"    : video_margin,
            "top_story_window_id"   : top_story["window_id"] if top_story is not None else None,
            "top_story_start_s"     : top_story["start_s"] if top_story is not None else None,
            "top_story_end_s"       : top_story["end_s"] if top_story is not None else None,
            "top_story_score"       : float(top_story["final_score"]) if top_story is not None else None,
            "top_video_id"          : top_video["video_id"] if top_video is not None else None,
            "top_video_score"       : float(top_video["final_score"]) if top_video is not None else None,
        })

    return pd.DataFrame(rows)


def summarize_retrieval_metrics(
    query_results : pd.DataFrame,
    breakdown_columns : Sequence[str] | None = None,
) -> pd.DataFrame :
    if (query_results.empty) :
        return pd.DataFrame()

    breakdown_columns = list(breakdown_columns or [])
    group_columns = ["model_id", "view"] + breakdown_columns
    rows = []

    for keys, group in query_results.groupby(group_columns, dropna=False, sort=True) :
        if (not isinstance(keys, tuple)) :
            keys = (keys,)
        row = dict(zip(group_columns, keys))

        first_ranks = pd.to_numeric(group["first_relevant_rank"], errors="coerce")
        video_ranks = pd.to_numeric(group["video_rank"], errors="coerce")
        row.update({
            "query_count"                : len(group),
            "story_recall_at_1"          : float(group["story_recall_at_1"].mean()),
            "story_recall_at_3"          : float(group["story_recall_at_3"].mean()),
            "story_recall_at_5"          : float(group["story_recall_at_5"].mean()),
            "story_recall_at_10"         : float(group["story_recall_at_10"].mean()),
            "story_mrr"                  : float(group["story_rr"].mean()),
            "story_first_rank_mean"      : float(first_ranks.mean()) if first_ranks.notna().any() else None,
            "story_first_rank_median"    : float(first_ranks.median()) if first_ranks.notna().any() else None,
            "video_recall_at_1"          : float(group["video_recall_at_1"].mean()),
            "video_recall_at_3"          : float(group["video_recall_at_3"].mean()),
            "video_recall_at_5"          : float(group["video_recall_at_5"].mean()),
            "video_recall_at_10"         : float(group["video_recall_at_10"].mean()),
            "video_recall_at_20"         : float(group["video_recall_at_20"].mean()),
            "video_mrr"                  : float(group["video_rr"].mean()),
            "video_rank_mean"            : float(video_ranks.mean()) if video_ranks.notna().any() else None,
            "video_rank_median"          : float(video_ranks.median()) if video_ranks.notna().any() else None,
            "story_score_margin_mean"    : float(pd.to_numeric(group["story_score_margin"], errors="coerce").mean()),
            "video_score_margin_mean"    : float(pd.to_numeric(group["video_score_margin"], errors="coerce").mean()),
        })
        rows.append(row)

    return pd.DataFrame(rows)


def _rank_effect(raw_rank : Any, processed_rank : Any) -> str :
    raw_value       = float(raw_rank) if pd.notna(raw_rank) else math.inf
    processed_value = float(processed_rank) if pd.notna(processed_rank) else math.inf
    if (processed_value < raw_value) :
        return "improved"
    if (processed_value > raw_value) :
        return "worsened"
    return "unchanged"


def compare_text_views(
    query_results : pd.DataFrame,
    raw_view : str = "raw",
    processed_view : str = "processed",
) -> pd.DataFrame :
    if (query_results.empty) :
        return pd.DataFrame()

    raw = query_results[query_results["view"] == raw_view].copy()
    processed = query_results[query_results["view"] == processed_view].copy()
    keys = ["model_id", "query_id"]
    columns = ["first_relevant_rank", "video_rank", "top_story_window_id", "top_video_id"]

    raw = raw[keys + columns].rename(columns={column : f"raw_{column}" for column in columns})
    processed = processed[keys + columns].rename(columns={column : f"processed_{column}" for column in columns})
    merged = raw.merge(processed, on=keys, how="outer")
    merged["story_effect"] = merged.apply(lambda row : _rank_effect(row["raw_first_relevant_rank"], row["processed_first_relevant_rank"]), axis=1)
    merged["video_effect"] = merged.apply(lambda row : _rank_effect(row["raw_video_rank"], row["processed_video_rank"]), axis=1)
    return merged


# -----------------------------------------------------------------------------
# Transcript reliability and repetition diagnostics
# -----------------------------------------------------------------------------


def _has_invalid_timestamps(segments : list[dict[str, Any]], duration_s : float) -> bool :
    for segment in segments or [] :
        start = segment.get("start_s")
        end   = segment.get("end_s")
        if (start is None or end is None) :
            continue

        try :
            start = float(start)
            end   = float(end)
        except (TypeError, ValueError) :
            return True

        if (not math.isfinite(start) or not math.isfinite(end)) :
            return True
        if (start < 0 or end < start or end > duration_s + 1.0) :
            return True

    return False


def _tokens(text : str) -> list[str] :
    return normalize_for_matching(text).split()


def _repeated_token_runs(tokens : list[str]) -> tuple[int, int] :
    if (not tokens) :
        return 0, 0

    maximum = 1
    repeated_runs = 0
    current = 1

    for index in range(1, len(tokens)) :
        if (tokens[index] == tokens[index - 1]) :
            current += 1
            maximum = max(maximum, current)
        else :
            if (current >= 2) :
                repeated_runs += 1
            current = 1

    if (current >= 2) :
        repeated_runs += 1

    return maximum, repeated_runs


def _ngram_repetition(tokens : list[str], n : int) -> dict[str, Any] :
    if (n <= 0 or len(tokens) < n) :
        return {
            "total"             : 0,
            "unique"            : 0,
            "repeated"          : 0,
            "repetition_ratio"  : 0.0,
            "most_repeated"     : None,
            "maximum_occurrence": 0,
        }

    ngrams = [tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1)]
    counts = Counter(ngrams)
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    most_repeated, maximum = max(counts.items(), key=lambda item : (item[1], item[0]))

    return {
        "total"             : len(ngrams),
        "unique"            : len(counts),
        "repeated"          : repeated,
        "repetition_ratio"  : repeated / len(ngrams) if ngrams else 0.0,
        "most_repeated"     : " ".join(most_repeated) if maximum > 1 else None,
        "maximum_occurrence": int(maximum),
    }


def build_window_diagnostics(
    windows : pd.DataFrame,
    ngram_sizes : Sequence[int] = (3, 4, 5),
    repetition_flag_threshold : float | None = None,
) -> pd.DataFrame :
    if (windows.empty) :
        return pd.DataFrame()

    rows = []

    for _, window in windows.iterrows() :
        raw_text       = str(window.get("raw_text", "") or "")
        processed_text = str(window.get("canonical_retrieval_text", window.get("retrieval_text", "")) or "")
        raw_tokens       = _tokens(raw_text)
        processed_tokens = _tokens(processed_text)
        max_run, repeated_runs = _repeated_token_runs(raw_tokens)
        ngram_stats = {n : _ngram_repetition(raw_tokens, n) for n in ngram_sizes}
        repetition_score = max((stats["repetition_ratio"] for stats in ngram_stats.values()), default=0.0)
        invalid_timestamps = _has_invalid_timestamps(window.get("native_segments", []) or [], float(window.get("duration_s", 0.0) or 0.0))

        runtime = pd.to_numeric(pd.Series([window.get("runtime_s")]), errors="coerce").iloc[0]
        duration = float(window.get("duration_s", 0.0) or 0.0)
        rtf = float(runtime) / duration if pd.notna(runtime) and duration > 0 else None

        flags = []
        status = str(window.get("status", ""))
        if (status not in SUCCESS_STATUSES) :
            flags.append(f"status_{status or 'missing'}")
        if (not raw_text.strip()) :
            flags.append("empty_raw_output")
        if (not processed_text.strip()) :
            flags.append("empty_processed_output")
        if (bool(window.get("canonical_rejection_reasons", window.get("rejection_reasons", [])))) :
            flags.append("processed_rejected")
        if (invalid_timestamps) :
            flags.append("invalid_timestamps")
        if (repetition_flag_threshold is not None and repetition_score >= repetition_flag_threshold) :
            flags.append("high_internal_repetition")

        row = {
            "stage_id"                    : window.get("stage_id"),
            "model_id"                    : window.get("model_id"),
            "video_id"                    : window.get("video_id"),
            "window_id"                   : window.get("window_id"),
            "window_index"                : window.get("window_index"),
            "start_s"                     : window.get("start_s"),
            "end_s"                       : window.get("end_s"),
            "duration_s"                  : duration,
            "status"                      : status,
            "raw_character_count"         : len(raw_text),
            "raw_token_count"             : len(raw_tokens),
            "processed_character_count"   : len(processed_text),
            "processed_token_count"       : len(processed_tokens),
            "processed_to_raw_length_ratio": len(processed_text) / len(raw_text) if raw_text else None,
            "native_segment_count"        : int(window.get("native_segment_count", len(window.get("native_segments", []) or [])) or 0),
            "empty_raw_output"            : not bool(raw_text.strip()),
            "empty_processed_output"      : not bool(processed_text.strip()),
            "processed_rejected"          : bool(window.get("canonical_rejection_reasons", window.get("rejection_reasons", []))),
            "warning_count"               : len(window.get("canonical_warning_reasons", window.get("warning_reasons", [])) or []),
            "rejection_count"             : len(window.get("canonical_rejection_reasons", window.get("rejection_reasons", [])) or []),
            "invalid_timestamps"          : invalid_timestamps,
            "runtime_s"                   : float(runtime) if pd.notna(runtime) else None,
            "real_time_factor"            : rtf,
            "maximum_repeated_token_run"  : max_run,
            "repeated_token_run_count"    : repeated_runs,
            "intra_window_repetition_score": repetition_score,
            "postprocess_transformations" : window.get("postprocess_transformations", []),
            "reliability_flags"           : flags,
        }

        for n, stats in ngram_stats.items() :
            prefix = f"ngram_{n}"
            row[f"{prefix}_total"]              = stats["total"]
            row[f"{prefix}_unique"]             = stats["unique"]
            row[f"{prefix}_repeated"]           = stats["repeated"]
            row[f"{prefix}_repetition_ratio"]   = stats["repetition_ratio"]
            row[f"{prefix}_most_repeated"]      = stats["most_repeated"]
            row[f"{prefix}_maximum_occurrence"] = stats["maximum_occurrence"]

        rows.append(row)

    return pd.DataFrame(rows)


def _bag_token_f1(left_tokens : list[str], right_tokens : list[str]) -> tuple[float, float, float] :
    if (not left_tokens and not right_tokens) :
        return 1.0, 1.0, 1.0
    if (not left_tokens or not right_tokens) :
        return 0.0, 0.0, 0.0

    left_counts  = Counter(left_tokens)
    right_counts = Counter(right_tokens)
    overlap = sum((left_counts & right_counts).values())
    precision = overlap / len(right_tokens)
    recall    = overlap / len(left_tokens)
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def find_duplicate_pairs(
    windows : pd.DataFrame,
    model_id : str,
    text_field : str = "raw_text",
    near_similarity_threshold : float = 0.90,
    semantic_scorer : SemanticScorer | None = None,
    include_adjacent : bool = True,
    maximum_pairs : int | None = None,
    progress_callback : ProgressCallback | None = None,
) -> pd.DataFrame :
    model_windows = windows[windows["model_id"] == model_id].copy().reset_index(drop=True)
    if (len(model_windows) < 2) :
        return pd.DataFrame()

    _emit_progress(progress_callback, f"{model_id}: duplicate scan over {len(model_windows)} windows")
    texts      = model_windows[text_field].fillna("").astype(str).tolist()
    normalized = [normalize_for_matching(text) for text in texts]

    if (not any(normalized)) :
        return pd.DataFrame()

    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), lowercase=False, min_df=1)
    matrix = vectorizer.fit_transform(normalized)
    similarity = cosine_similarity(matrix)
    candidate_pairs = []

    for left_index in range(len(model_windows)) :
        left = model_windows.iloc[left_index]
        for right_index in range(left_index + 1, len(model_windows)) :
            right = model_windows.iloc[right_index]
            same_video = left["video_id"] == right["video_id"]
            index_distance = abs(int(left.get("window_index", 0) or 0) - int(right.get("window_index", 0) or 0)) if same_video else None
            relationship = (
                "adjacent_same_video"
                if same_video and index_distance == 1
                else "non_adjacent_same_video"
                if same_video
                else "cross_video"
            )

            exact = bool(normalized[left_index] and normalized[left_index] == normalized[right_index])
            char_similarity = float(similarity[left_index, right_index])

            if (not exact and char_similarity < near_similarity_threshold and not (include_adjacent and relationship == "adjacent_same_video")) :
                continue

            _, _, token_f1 = _bag_token_f1(_tokens(texts[left_index]), _tokens(texts[right_index]))
            candidate_pairs.append({
                "model_id"             : model_id,
                "relationship"         : relationship,
                "left_video_id"        : left["video_id"],
                "left_window_id"       : left["window_id"],
                "left_window_index"    : left.get("window_index"),
                "right_video_id"       : right["video_id"],
                "right_window_id"      : right["window_id"],
                "right_window_index"   : right.get("window_index"),
                "exact_normalized"     : exact,
                "character_similarity" : char_similarity,
                "token_f1"             : token_f1,
                "semantic_similarity"  : None,
            })

    if (semantic_scorer is not None and candidate_pairs) :
        _emit_progress(progress_callback, f"{model_id}: semantic scoring for {len(candidate_pairs)} duplicate candidates")
        embeddings = semantic_scorer.passage_embeddings(texts, use_cache=False)
        index_by_window = {window_id : index for index, window_id in enumerate(model_windows["window_id"].tolist())}

        for pair in candidate_pairs :
            left_index  = index_by_window[pair["left_window_id"]]
            right_index = index_by_window[pair["right_window_id"]]
            pair["semantic_similarity"] = float(np.clip(np.dot(embeddings[left_index], embeddings[right_index]), -1.0, 1.0))

    relationship_priority = {"cross_video" : 0, "non_adjacent_same_video" : 1, "adjacent_same_video" : 2}
    candidate_pairs.sort(key=lambda item : (
        not item["exact_normalized"],
        relationship_priority[item["relationship"]],
        -item["character_similarity"],
        item["left_window_id"],
        item["right_window_id"],
    ))

    if (maximum_pairs is not None) :
        exact_pairs = [item for item in candidate_pairs if item["exact_normalized"]]
        remaining   = [item for item in candidate_pairs if not item["exact_normalized"]]
        candidate_pairs = exact_pairs + remaining[ : max(0, maximum_pairs - len(exact_pairs))]

    _emit_progress(progress_callback, f"{model_id}: duplicate scan complete ({len(candidate_pairs)} reported pairs)")
    return pd.DataFrame(candidate_pairs)


def strong_duplicate_window_ids(duplicate_pairs : pd.DataFrame) -> set[str] :
    if (duplicate_pairs.empty) :
        return set()

    strong = duplicate_pairs[
        duplicate_pairs["exact_normalized"]
        & duplicate_pairs["relationship"].isin(["non_adjacent_same_video", "cross_video"])
    ]
    return set(strong["left_window_id"].tolist()) | set(strong["right_window_id"].tolist())


def mine_repeated_phrases(
    windows : pd.DataFrame,
    model_id : str,
    text_field : str = "raw_text",
    min_n : int = 4,
    max_n : int = 8,
    minimum_video_count : int = 2,
    maximum_results : int | None = 500,
    progress_callback : ProgressCallback | None = None,
) -> pd.DataFrame :
    model_windows = windows[windows["model_id"] == model_id].copy()
    if (model_windows.empty) :
        return pd.DataFrame()

    stats : dict[tuple[str, ...], dict[str, Any]] = {}
    _emit_progress(progress_callback, f"{model_id}: mining {min_n}-{max_n} token phrases across {len(model_windows)} windows")

    for _, window in model_windows.iterrows() :
        tokens = _tokens(str(window.get(text_field, "") or ""))
        if (not tokens) :
            continue

        local_counts = Counter()
        for n in range(min_n, max_n + 1) :
            for index in range(len(tokens) - n + 1) :
                local_counts[tuple(tokens[index : index + n])] += 1

        for phrase, count in local_counts.items() :
            item = stats.setdefault(phrase, {
                "occurrences"          : 0,
                "windows"              : set(),
                "videos"               : set(),
                "maximum_window_coverage": 0.0,
            })
            item["occurrences"] += count
            item["windows"].add(window["window_id"])
            item["videos"].add(window["video_id"])
            coverage = min(1.0, count * len(phrase) / len(tokens))
            item["maximum_window_coverage"] = max(item["maximum_window_coverage"], coverage)

    rows = []
    for phrase, item in stats.items() :
        if (len(item["videos"]) < minimum_video_count) :
            continue

        videos  = sorted(item["videos"])
        windows_ids = sorted(item["windows"])
        suspicious_score = len(videos) * item["maximum_window_coverage"]
        rows.append({
            "model_id"                 : model_id,
            "phrase"                   : " ".join(phrase),
            "token_length"             : len(phrase),
            "occurrence_count"         : int(item["occurrences"]),
            "distinct_window_count"    : len(windows_ids),
            "distinct_video_count"     : len(videos),
            "maximum_window_coverage" : float(item["maximum_window_coverage"]),
            "suspicious_score"         : float(suspicious_score),
            "example_window_ids"       : windows_ids[ : 5],
            "example_video_ids"        : videos[ : 5],
        })

    rows.sort(key=lambda item : (-item["suspicious_score"], -item["distinct_video_count"], -item["occurrence_count"], item["phrase"]))
    if (maximum_results is not None) :
        rows = rows[ : maximum_results]
    _emit_progress(progress_callback, f"{model_id}: phrase mining complete ({len(rows)} reported phrases)")
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Whisper-reference transcript agreement
# -----------------------------------------------------------------------------


def _agreement_normalize(text : str) -> str :
    text = normalize_unicode(text).lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE).replace("_", " ")
    return re.sub(r"\s+", " ", text).strip()


def _levenshtein_distance(left : Sequence[Any], right : Sequence[Any]) -> int :
    if (len(left) < len(right)) :
        left, right = right, left
    if (not right) :
        return len(left)

    previous = list(range(len(right) + 1))
    for left_index, left_value in enumerate(left, start=1) :
        current = [left_index]
        for right_index, right_value in enumerate(right, start=1) :
            insertion    = current[right_index - 1] + 1
            deletion     = previous[right_index] + 1
            substitution = previous[right_index - 1] + (left_value != right_value)
            current.append(min(insertion, deletion, substitution))
        previous = current

    return previous[-1]


def build_reference_agreement(
    windows : pd.DataFrame,
    reference_model_id : str,
    candidate_model_id : str,
    semantic_scorer : SemanticScorer | None = None,
    reference_untrusted_window_ids : set[str] | None = None,
    progress_callback : ProgressCallback | None = None,
) -> pd.DataFrame :
    reference_untrusted_window_ids = reference_untrusted_window_ids or set()
    reference = windows[windows["model_id"] == reference_model_id].copy()
    candidate = windows[windows["model_id"] == candidate_model_id].copy()

    keep = [
        "stage_id", "video_id", "window_id", "window_index", "start_s", "end_s",
        "status", "raw_text", "canonical_rejection_reasons", "canonical_warning_reasons",
    ]
    reference = reference[keep].rename(columns={column : f"reference_{column}" for column in keep if column not in {"stage_id", "window_id"}})
    candidate = candidate[keep].rename(columns={column : f"candidate_{column}" for column in keep if column not in {"stage_id", "window_id"}})
    merged = reference.merge(candidate, on=["stage_id", "window_id"], how="inner")
    _emit_progress(progress_callback, f"{candidate_model_id} vs {reference_model_id}: agreement over {len(merged)} shared windows")

    rows = []
    semantic_left  = []
    semantic_right = []

    for _, row in merged.iterrows() :
        reference_text = _agreement_normalize(str(row["reference_raw_text"] or ""))
        candidate_text = _agreement_normalize(str(row["candidate_raw_text"] or ""))
        reference_tokens = reference_text.split()
        candidate_tokens = candidate_text.split()
        reference_chars = list(reference_text.replace(" ", ""))
        candidate_chars = list(candidate_text.replace(" ", ""))

        token_distance = _levenshtein_distance(reference_tokens, candidate_tokens) if reference_tokens else None
        char_distance  = _levenshtein_distance(reference_chars, candidate_chars) if reference_chars else None
        precision, recall, token_f1 = _bag_token_f1(reference_tokens, candidate_tokens)
        symmetric_token_distance = (
            _levenshtein_distance(reference_tokens, candidate_tokens) / max(len(reference_tokens), len(candidate_tokens), 1)
        )
        character_similarity = SequenceMatcher(None, reference_text, candidate_text).ratio()
        length_ratio = len(candidate_tokens) / len(reference_tokens) if reference_tokens else None

        trusted_reasons = []
        if (str(row["reference_status"]) not in SUCCESS_STATUSES) :
            trusted_reasons.append(f"reference_status_{row['reference_status']}")
        if (not reference_text) :
            trusted_reasons.append("reference_empty")
        reference_rejections = set(row["reference_canonical_rejection_reasons"] or [])
        if ("dominant_known_boilerplate" in reference_rejections) :
            trusted_reasons.append("reference_dominant_known_boilerplate")
        if (row["window_id"] in reference_untrusted_window_ids) :
            trusted_reasons.append("reference_exact_unrelated_duplicate")

        rows.append({
            "stage_id"                       : row["stage_id"],
            "video_id"                       : row["reference_video_id"],
            "window_id"                      : row["window_id"],
            "window_index"                   : row["reference_window_index"],
            "start_s"                        : row["reference_start_s"],
            "end_s"                          : row["reference_end_s"],
            "reference_model_id"             : reference_model_id,
            "candidate_model_id"             : candidate_model_id,
            "reference_status"               : row["reference_status"],
            "candidate_status"               : row["candidate_status"],
            "reference_token_count"          : len(reference_tokens),
            "candidate_token_count"          : len(candidate_tokens),
            "whisper_reference_wer"          : token_distance / len(reference_tokens) if token_distance is not None else None,
            "whisper_reference_cer"          : char_distance / len(reference_chars) if char_distance is not None else None,
            "token_precision"                : precision,
            "token_recall"                   : recall,
            "token_f1"                       : token_f1,
            "symmetric_token_distance"       : symmetric_token_distance,
            "character_similarity"           : character_similarity,
            "candidate_reference_length_ratio": length_ratio,
            "semantic_similarity"            : None,
            "reference_trusted"              : not trusted_reasons,
            "reference_untrusted_reasons"    : trusted_reasons,
        })
        semantic_left.append(reference_text)
        semantic_right.append(candidate_text)

    if (semantic_scorer is not None and rows) :
        _emit_progress(progress_callback, f"{candidate_model_id} vs {reference_model_id}: computing transcript semantic similarity")
        semantic_scores = semantic_scorer.passage_similarity(semantic_left, semantic_right)
        for item, score in zip(rows, semantic_scores) :
            item["semantic_similarity"] = float(score)

    _emit_progress(progress_callback, f"{candidate_model_id} vs {reference_model_id}: agreement complete")
    return pd.DataFrame(rows)


def summarize_reference_agreement(agreement : pd.DataFrame) -> pd.DataFrame :
    if (agreement.empty) :
        return pd.DataFrame()

    rows = []
    subsets = {
        "all_reference_windows"     : agreement,
        "trusted_reference_windows" : agreement[agreement["reference_trusted"]],
    }

    for subset_name, subset in subsets.items() :
        rows.append({
            "subset"                    : subset_name,
            "window_count"              : len(subset),
            "whisper_reference_wer_mean": float(pd.to_numeric(subset["whisper_reference_wer"], errors="coerce").mean()),
            "whisper_reference_wer_median": float(pd.to_numeric(subset["whisper_reference_wer"], errors="coerce").median()),
            "whisper_reference_cer_mean": float(pd.to_numeric(subset["whisper_reference_cer"], errors="coerce").mean()),
            "whisper_reference_cer_median": float(pd.to_numeric(subset["whisper_reference_cer"], errors="coerce").median()),
            "token_f1_mean"             : float(pd.to_numeric(subset["token_f1"], errors="coerce").mean()),
            "character_similarity_mean" : float(pd.to_numeric(subset["character_similarity"], errors="coerce").mean()),
            "semantic_similarity_mean"  : float(pd.to_numeric(subset["semantic_similarity"], errors="coerce").mean()) if subset["semantic_similarity"].notna().any() else None,
            "length_ratio_mean"          : float(pd.to_numeric(subset["candidate_reference_length_ratio"], errors="coerce").mean()),
        })

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Operational summaries
# -----------------------------------------------------------------------------


def summarize_operational(
    windows : pd.DataFrame,
    group_columns : Sequence[str] = ("model_id", "stage_id"),
) -> pd.DataFrame :
    if (windows.empty) :
        return pd.DataFrame()

    group_columns = list(group_columns)
    rows = []

    for keys, group in windows.groupby(group_columns, dropna=False, sort=True) :
        if (not isinstance(keys, tuple)) :
            keys = (keys,)
        row = dict(zip(group_columns, keys))
        runtime = pd.to_numeric(group["runtime_s"], errors="coerce")
        duration = pd.to_numeric(group["duration_s"], errors="coerce")
        attempted = runtime.notna()
        rtf_values = runtime[attempted] / duration[attempted].replace(0, np.nan)
        successful = group["status"].isin(SUCCESS_STATUSES)
        empty = successful & ~group["raw_text"].fillna("").astype(str).str.strip().astype(bool)
        failed = ~group["status"].isin(SUCCESS_STATUSES) & (group["status"] != "missing")
        present = group["status"] != "missing"
        total_runtime = float(runtime.fillna(0.0).sum())
        total_audio   = float(duration[attempted].fillna(0.0).sum())

        row.update({
            "expected_window_count"        : len(group),
            "present_window_count"         : int(present.sum()),
            "successful_window_count"      : int(successful.sum()),
            "failed_window_count"          : int(failed.sum()),
            "empty_window_count"           : int(empty.sum()),
            "total_inference_runtime_s"    : total_runtime,
            "total_attempted_audio_s"      : total_audio,
            "aggregate_rtf"                : total_runtime / total_audio if total_audio > 0 else None,
            "mean_window_rtf"              : float(rtf_values.mean()) if rtf_values.notna().any() else None,
            "median_window_rtf"            : float(rtf_values.median()) if rtf_values.notna().any() else None,
            "p90_window_rtf"               : float(rtf_values.quantile(0.90)) if rtf_values.notna().any() else None,
            "p95_window_rtf"               : float(rtf_values.quantile(0.95)) if rtf_values.notna().any() else None,
            "windows_per_hour"             : float(attempted.sum()) / (total_runtime / 3600.0) if total_runtime > 0 else None,
            "peak_gpu_memory_bytes"        : float(pd.to_numeric(group["peak_gpu_memory_bytes"], errors="coerce").fillna(0.0).max()),
            "peak_reserved_memory_bytes"   : float(pd.to_numeric(group["peak_reserved_memory_bytes"], errors="coerce").fillna(0.0).max()),
        })
        rows.append(row)

    return pd.DataFrame(rows)


def summarize_run_payloads(payloads : Iterable[dict[str, Any]]) -> pd.DataFrame :
    rows = []

    for payload in payloads :
        rows.append({
            "stage_id"                    : payload.get("stage_id"),
            "model_id"                    : payload.get("model_id"),
            "device"                      : payload.get("device"),
            "model_revision"              : payload.get("model_revision") or payload.get("revision"),
            "model_configuration_hash"    : payload.get("model_configuration_hash"),
            "adapter_version"             : payload.get("adapter_version"),
            "adapter_hash"                : payload.get("adapter_hash"),
            "model_load_runtime_s"        : payload.get("model_load_runtime_s") or payload.get("model_load_time_s") or payload.get("load_time_s"),
            "total_inference_runtime_s"   : payload.get("total_inference_runtime_s"),
            "aggregate_rtf"               : payload.get("aggregate_rtf"),
            "expected_video_count"        : payload.get("expected_video_count"),
            "expected_window_count"       : payload.get("expected_window_count"),
            "saved_window_count"          : payload.get("saved_window_count"),
            "successful_window_count"     : payload.get("successful_window_count"),
            "failed_window_count"         : payload.get("failed_window_count"),
            "empty_window_count"          : payload.get("empty_window_count"),
            "peak_gpu_memory_bytes"       : payload.get("peak_gpu_memory_bytes"),
            "peak_reserved_memory_bytes"  : payload.get("peak_reserved_memory_bytes"),
            "complete"                    : payload.get("complete"),
        })

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Backward-compatible summary, inspection, and report helpers
# -----------------------------------------------------------------------------


def summarize_models(query_results : pd.DataFrame, windows : pd.DataFrame) -> pd.DataFrame :
    if (query_results.empty) :
        return pd.DataFrame()

    retrieval = summarize_retrieval_metrics(query_results)
    operational = summarize_operational(windows, group_columns=("model_id",))
    summary = retrieval.merge(operational, on="model_id", how="left")

    rows = []
    for _, row in summary.iterrows() :
        model_windows = windows[windows["model_id"] == row["model_id"]]
        warning_lists = model_windows["warning_reasons"].tolist() if not model_windows.empty else []
        known_reasons = {"known_boilerplate", "partial_known_boilerplate"}
        output = row.to_dict()
        output.update({
            "known_boilerplate_count" : sum(bool(known_reasons.intersection(reasons or [])) for reasons in warning_lists),
            "repeated_window_count"   : sum("consecutive_duplicate_window" in (reasons or []) for reasons in warning_lists),
            "invalid_timestamp_count" : sum(_has_invalid_timestamps(item["native_segments"], item["duration_s"]) for _, item in model_windows.iterrows()),
        })
        rows.append(output)

    return pd.DataFrame(rows)


def select_inspection_cases(
    model_id : str,
    query_results : pd.DataFrame,
    scores : pd.DataFrame,
    windows : pd.DataFrame,
    maximum_cases : int = 5,
) -> list[dict[str, Any]] :
    model_queries = query_results[(query_results["model_id"] == model_id) & query_results["view"].astype(str).str.startswith("processed")].copy()
    model_windows = windows[windows["model_id"] == model_id].copy()
    if (model_queries.empty) :
        return []

    cases = []
    used  = set()

    def add_case(label : str, query_id : str | None, window_id : str | None = None) -> None :
        key = (query_id, window_id)
        if (key in used or len(cases) >= maximum_cases) :
            return
        used.add(key)
        cases.append({"case_type" : label, "query_id" : query_id, "window_id" : window_id})

    successes = model_queries[model_queries["first_relevant_rank"] == 1].sort_values("top_story_score", ascending=False)
    if (not successes.empty) :
        row = successes.iloc[0]
        add_case("strong_success", row["query_id"], row["top_story_window_id"])

    poor = model_queries.assign(rank_sort=model_queries["first_relevant_rank"].fillna(math.inf)).sort_values("rank_sort", ascending=False)
    if (not poor.empty) :
        row = poor.iloc[0]
        add_case("poor_retrieval", row["query_id"], row["top_story_window_id"])

    reliability_mask = (
        model_windows["warning_reasons"].map(lambda reasons : bool(reasons))
        | ~model_windows["status"].isin(SUCCESS_STATUSES)
        | ~model_windows["raw_text"].fillna("").astype(str).str.strip().astype(bool)
    )
    warning_rows = model_windows[reliability_mask].copy()
    if (not warning_rows.empty) :
        warning_rows["severity"] = (
            warning_rows["warning_reasons"].map(len)
            + (~warning_rows["status"].isin(SUCCESS_STATUSES)).astype(int)
            + (~warning_rows["raw_text"].fillna("").astype(str).str.strip().astype(bool)).astype(int)
        )
        row = warning_rows.sort_values(["severity", "duration_s"], ascending=[False, False]).iloc[0]
        add_case("reliability_warning", None, row["window_id"])

    processed_queries = query_results[query_results["view"].astype(str).str.startswith("processed")].copy()
    processed_queries["disagreement_rank"] = processed_queries["first_relevant_rank"].fillna(1_000_000)
    cross_model = processed_queries.pivot_table(index="query_id", columns="model_id", values="disagreement_rank", aggfunc="first")
    if (model_id in cross_model.columns and cross_model.shape[1] > 1) :
        spread = cross_model.apply(lambda row : row.dropna().max() - row.dropna().min() if row.dropna().size >= 2 else -1, axis=1)
        if (spread.max() >= 0) :
            query_id = spread.idxmax()
            result   = model_queries[model_queries["query_id"] == query_id]
            if (not result.empty) :
                add_case("cross_model_disagreement", query_id, result.iloc[0]["top_story_window_id"])

    for _, row in poor.iterrows() :
        add_case("additional_review", row["query_id"], row["top_story_window_id"])

    return cases[ : maximum_cases]


def _json_clean(value : Any) -> Any :
    if (isinstance(value, (np.integer,)) ) :
        return int(value)
    if (isinstance(value, (np.floating,)) ) :
        value = float(value)
    if (isinstance(value, float) and not math.isfinite(value)) :
        return None
    if (isinstance(value, dict)) :
        return {str(key) : _json_clean(item) for key, item in value.items()}
    if (isinstance(value, (list, tuple, set))) :
        return [_json_clean(item) for item in value]
    return value


def save_reports(
    reports_root : Path,
    window_scores : pd.DataFrame,
    query_results : pd.DataFrame,
    model_summary : pd.DataFrame,
    metadata : dict[str, Any],
) -> None :
    """Legacy compact report writer. The Stage 1 orchestrator writes the richer v2 report set."""
    reports_root = Path(reports_root)
    reports_root.mkdir(parents=True, exist_ok=True)

    window_scores.to_csv(reports_root / "window_scores.csv", index=False)
    query_results.to_csv(reports_root / "query_results.csv", index=False)
    model_summary.to_csv(reports_root / "model_summary.csv", index=False)

    comparison = {
        "evaluation_version" : EVALUATION_VERSION,
        "metadata"           : metadata,
        "models"             : model_summary.to_dict(orient="records"),
        "queries"            : query_results.to_dict(orient="records"),
    }
    clean = _json_clean(comparison)
    (reports_root / "comparison.json").write_text(
        json.dumps(clean, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
