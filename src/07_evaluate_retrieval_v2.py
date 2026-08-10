# Relative path: src/07_evaluate_retrieval_v2.py
# Purpose: Orchestrate Retrieval v2 Stage 0 baseline reproduction and Stage 1 lexical experiments.

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import sklearn


SCRIPT_PATH = Path(__file__).resolve()
SRC_ROOT    = SCRIPT_PATH.parent
CODE_ROOT   = SCRIPT_PATH.parents[1]

if (str(SRC_ROOT) not in sys.path) :
    sys.path.insert(0, str(SRC_ROOT))

import evaluation as stage1_evaluation
import postprocess
import retrieval_v2
import retrieval_v2_evaluation

from retrieval_v2 import (
    ChannelDocuments,
    E5CompatibilityScorer,
    ScoreBundle,
    positive_evidence_rrf,
    score_bm25,
    score_corpus_fitted_dual_tfidf,
    score_query_fitted_dual_tfidf,
    weighted_score_bundle,
)
from retrieval_v2_evaluation import (
    classify_eligibility,
    compare_expected_metrics,
    compare_methods,
    compare_regression_fixture,
    evaluate_score_matrix,
    summarize_retrieval_metrics,
    summarize_stage1_methods,
)


CONFIG_PATH = Path(
    os.environ.get(
        "AIC_RETRIEVAL_CONFIG",
        str(CODE_ROOT / "configs" / "retrieval_v2.json"),
    )
).resolve()


def parse_args() -> argparse.Namespace :
    parser = argparse.ArgumentParser(
        description = "Run AIC 2026 Retrieval v2 Stage 0 or Stage 1.",
    )
    parser.add_argument(
        "--stage",
        choices = ["stage00_baseline", "stage01_lexical"],
        default = os.environ.get("AIC_RETRIEVAL_STAGE", "stage00_baseline"),
    )
    return parser.parse_args()


def utc_now() -> str :
    return datetime.now(timezone.utc).isoformat()


def log_stage(index : int, total : int, message : str) -> float :
    print(f"[{index}/{total}] {message}", flush = True)
    return time.perf_counter()


def log_done(start : float) -> None :
    print(f"      done in {time.perf_counter() - start:.1f}s", flush = True)


def log_detail(message : str) -> None :
    print(f"      {message}", flush = True)


def load_json(path : Path) -> dict[str, Any] :
    return json.loads(Path(path).read_text(encoding = "utf-8"))


def json_clean(value : Any) -> Any :
    if (isinstance(value, dict)) :
        return {str(key) : json_clean(item) for key, item in value.items()}

    if (isinstance(value, (list, tuple, set))) :
        return [json_clean(item) for item in value]

    if (isinstance(value, np.generic)) :
        return json_clean(value.item())

    if (isinstance(value, float) and not math.isfinite(value)) :
        return None

    if (value is pd.NA) :
        return None

    return value


def write_json(path : Path, value : Any) -> None :
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_text(
        json.dumps(
            json_clean(value),
            ensure_ascii = False,
            indent = 2,
            allow_nan = False,
        ),
        encoding = "utf-8",
    )


def write_csv(path : Path, frame : pd.DataFrame) -> None :
    path.parent.mkdir(parents = True, exist_ok = True)
    frame.to_csv(path, index = False)


def sha256_file(path : Path, chunk_size : int = 8 * 1024 * 1024) -> str :
    digest = hashlib.sha256()

    with Path(path).open("rb") as handle :
        while True :
            chunk = handle.read(chunk_size)

            if (not chunk) :
                break

            digest.update(chunk)

    return digest.hexdigest()


def canonical_json_hash(value : Any) -> str :
    serialized = json.dumps(
        json_clean(value),
        ensure_ascii = False,
        sort_keys = True,
        separators = (",", ":"),
        allow_nan = False,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def source_record(path : Path) -> dict[str, Any] :
    return {
        "path"   : str(path),
        "exists" : path.exists(),
        "sha256" : sha256_file(path) if path.exists() else None,
    }


def load_regression_fixture(config : dict[str, Any]) -> tuple[dict[str, Any], Path, dict[str, Any]] :
    specification = config["regression"]
    path = resolve_code_path(specification["fixture_path"])
    required = bool(specification.get("fixture_required", False))

    if (not path.exists()) :
        if (required) :
            raise FileNotFoundError(f"Required regression fixture is missing: {path}")

        return {}, path, {
            "passed"   : True,
            "required" : False,
            "exists"   : False,
            "path"     : str(path),
            "sha256"   : None,
            "errors"   : [],
        }

    actual_hash = sha256_file(path)
    expected_hash = specification.get("expected_fixture_sha256")
    errors = []

    if (expected_hash and actual_hash != expected_hash) :
        errors.append(
            f"Regression fixture SHA-256 mismatch: {actual_hash} != {expected_hash}"
        )

    fixture = load_json(path)

    if (fixture.get("schema_version") != "1.0") :
        errors.append(
            f"Unsupported regression fixture schema: {fixture.get('schema_version')!r}"
        )

    return fixture, path, {
        "passed"          : not errors,
        "required"        : required,
        "exists"          : True,
        "path"            : str(path),
        "sha256"          : actual_hash,
        "expected_sha256" : expected_hash,
        "fixture_id"      : fixture.get("fixture_id"),
        "errors"          : errors,
    }


def validate_fixture_provenance(
    fixture : dict[str, Any],
    config : dict[str, Any],
    benchmark_paths : dict[str, Path],
    benchmarks : dict[str, dict[str, Any]],
    output_hashes : dict[str, Any],
) -> dict[str, Any] :
    if (not fixture) :
        return {
            "passed" : not bool(config["regression"].get("fixture_required", False)),
            "checks" : [],
            "errors" : ["Required regression fixture is unavailable"],
        }

    provenance = fixture.get("provenance", {})
    checks = []
    errors = []

    def check(name : str, actual : Any, expected : Any) -> None :
        passed = actual == expected
        checks.append({
            "name"     : name,
            "actual"   : actual,
            "expected" : expected,
            "passed"   : passed,
        })

        if (not passed) :
            errors.append(f"{name} mismatch")

    source_spec = config["source_stage1"]
    check(
        "stage1_config_sha256",
        sha256_file(resolve_code_path(source_spec["config_path"])),
        provenance.get("config_sha256"),
    )
    check(
        "stage1_evaluation_sha256",
        sha256_file(resolve_code_path(source_spec["evaluation_path"])),
        provenance.get("evaluation_sha256"),
    )
    check(
        "stage1_orchestrator_sha256",
        sha256_file(resolve_code_path(source_spec["orchestrator_path"])),
        provenance.get("orchestrator_sha256"),
    )
    check(
        "postprocess_sha256",
        sha256_file(resolve_code_path(source_spec["postprocess_path"])),
        provenance.get("postprocess_sha256"),
    )
    check(
        "postprocess_version",
        postprocess.POSTPROCESS_VERSION,
        provenance.get("postprocess_version"),
    )

    for benchmark_key in ["core10", "extension40"] :
        expected_benchmark = provenance.get("benchmarks", {}).get(benchmark_key, {})
        benchmark = benchmarks[benchmark_key]
        check(
            f"{benchmark_key}_file_sha256",
            sha256_file(benchmark_paths[benchmark_key]),
            expected_benchmark.get("file_sha256"),
        )
        check(
            f"{benchmark_key}_content_hash",
            benchmark.get("benchmark_content_hash"),
            expected_benchmark.get("benchmark_content_hash"),
        )
        check(
            f"{benchmark_key}_window_policy_hash",
            benchmark.get("window_policy_hash"),
            expected_benchmark.get("window_policy_hash"),
        )

    expected_outputs = provenance.get("model_output_aggregate_hashes", {})

    for model_id, model_spec in expected_outputs.items() :
        for benchmark_key, expected_hash in model_spec.items() :
            actual_hash = (
                output_hashes
                .get(model_id, {})
                .get(benchmark_key, {})
                .get("aggregate_hash")
            )
            check(
                f"model_output_{model_id}_{benchmark_key}",
                actual_hash,
                expected_hash,
            )

    semantic_expected = provenance.get("semantic_model", {})
    semantic_actual = config["stage00_baseline"]["semantic"]

    for field in ["model_name", "revision", "query_prefix", "passage_prefix"] :
        check(
            f"semantic_{field}",
            semantic_actual.get(field),
            semantic_expected.get(field),
        )

    check(
        "historical_validation_passed",
        bool(provenance.get("historical_validation_passed")),
        True,
    )
    check(
        "historical_regression_gate_passed",
        bool(provenance.get("historical_regression_gate_passed")),
        True,
    )

    return {
        "passed" : not errors,
        "checks" : checks,
        "errors" : errors,
    }


def resolve_code_path(value : str | Path) -> Path :
    path = Path(value)
    return path.resolve() if path.is_absolute() else (CODE_ROOT / path).resolve()


def artifact_root(config : dict[str, Any]) -> Path :
    environment_name = config["paths"]["artifact_root_env"]
    value = os.environ.get(environment_name, "").strip()

    if (not value) :
        raise RuntimeError(
            f"{environment_name} is not set. "
            "Point it to the canonical asr_model_comparison artifact workspace."
        )

    root = Path(value).expanduser().resolve()

    if (not root.exists()) :
        raise FileNotFoundError(f"Artifact root does not exist: {root}")

    return root


def resolve_artifact_path(root : Path, value : str | Path) -> Path :
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def load_retrieval_config() -> dict[str, Any] :
    if (not CONFIG_PATH.exists()) :
        raise FileNotFoundError(f"Retrieval v2 config not found: {CONFIG_PATH}")

    config = load_json(CONFIG_PATH)
    required = [
        "schema_version",
        "retrieval_id",
        "source_stage1",
        "benchmarks",
        "query_sets",
        "corpora",
        "channels",
        "stage00_baseline",
        "stage01_lexical",
        "evaluation_policy",
        "regression",
        "selection",
        "paths",
    ]
    missing = [key for key in required if key not in config]

    if (missing) :
        raise ValueError(f"Missing Retrieval v2 config keys: {missing}")

    return config


def validate_source_contract(config : dict[str, Any]) -> dict[str, Any] :
    specification = config["source_stage1"]
    paths = {
        "config"       : resolve_code_path(specification["config_path"]),
        "evaluation"   : resolve_code_path(specification["evaluation_path"]),
        "postprocess"  : resolve_code_path(specification["postprocess_path"]),
        "orchestrator" : resolve_code_path(specification["orchestrator_path"]),
    }
    errors = []
    checks = []

    for key, path in paths.items() :
        actual   = sha256_file(path) if path.exists() else None
        expected = specification["expected_sha256"].get(key)
        passed   = actual == expected

        checks.append({
            "source"   : key,
            "path"     : str(path),
            "expected" : expected,
            "actual"   : actual,
            "passed"   : passed,
        })

        if (specification.get("enforce_hashes", True) and not passed) :
            errors.append(f"{key} SHA-256 mismatch")

    if (stage1_evaluation.EVALUATION_VERSION != specification["expected_evaluation_version"]) :
        errors.append(
            f"Stage 1 evaluation version mismatch: "
            f"{stage1_evaluation.EVALUATION_VERSION!r}"
        )

    if (postprocess.POSTPROCESS_VERSION != specification["expected_postprocess_version"]) :
        errors.append(
            f"Postprocess version mismatch: {postprocess.POSTPROCESS_VERSION!r}"
        )

    return {
        "passed" : not errors,
        "checks" : checks,
        "errors" : errors,
        "paths"  : {key : str(path) for key, path in paths.items()},
    }


def load_benchmarks(
    config : dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, Path], list[dict[str, Any]]] :
    benchmarks = {}
    paths      = {}
    validations = []

    for key, specification in config["benchmarks"].items() :
        path = resolve_code_path(specification["path"])

        if (not path.exists()) :
            raise FileNotFoundError(f"Benchmark not found: {path}")

        benchmark = load_json(path)
        benchmarks[key] = benchmark
        paths[key]      = path

        base_validation = stage1_evaluation.validate_benchmark_manifest(benchmark)
        errors = list(base_validation.get("errors", []))

        identity_checks = {
            "file_sha256"          : (sha256_file(path), specification["expected_file_sha256"]),
            "stage_id"             : (benchmark.get("stage_id"), specification["expected_stage_id"]),
            "video_count"          : (benchmark.get("video_count"), specification["expected_video_count"]),
            "query_count"          : (benchmark.get("query_count"), specification["expected_query_count"]),
            "window_count"         : (benchmark.get("window_count"), specification["expected_window_count"]),
            "benchmark_content_hash": (
                benchmark.get("benchmark_content_hash"),
                specification["expected_benchmark_content_hash"],
            ),
            "window_policy_hash"   : (
                benchmark.get("window_policy_hash"),
                specification["expected_window_policy_hash"],
            ),
        }

        for field, (actual, expected) in identity_checks.items() :
            if (actual != expected) :
                errors.append(f"{key}: {field} mismatch ({actual!r} != {expected!r})")

        validations.append({
            "benchmark_key" : key,
            "passed"        : not errors,
            "errors"        : errors,
            "warnings"      : base_validation.get("warnings", []),
        })

    union = stage1_evaluation.validate_benchmark_union(
        [benchmarks["core10"], benchmarks["extension40"]]
    )
    validations.append({
        "benchmark_key" : "all50_union",
        **union,
    })

    return benchmarks, paths, validations


def _query_rows(benchmark : dict[str, Any]) -> pd.DataFrame :
    return stage1_evaluation.query_rows(benchmark)


def build_query_set(
    name : str,
    config : dict[str, Any],
    benchmarks : dict[str, dict[str, Any]],
) -> pd.DataFrame :
    specification = config["query_sets"][name]
    frames = []

    for source in specification["sources"] :
        frame = _query_rows(benchmarks[source["benchmark"]])
        split = source.get("split", "all")

        if (split != "all") :
            frame = frame[frame["evaluation_split"] == split].copy()

        frames.append(frame)

    frame = pd.concat(frames, ignore_index = True) if frames else pd.DataFrame()

    if (frame.empty) :
        raise ValueError(f"Query set {name!r} is empty")

    if (frame["query_id"].duplicated().any()) :
        raise ValueError(f"Query set {name!r} contains duplicate query IDs")

    frame["query_set"] = name
    return frame


def query_set_hash(
    name : str,
    frame : pd.DataFrame,
    corpus_id : str,
    benchmark_hashes : list[str],
) -> str :
    records = []

    for _, item in frame.sort_values("query_id", kind = "mergesort").iterrows() :
        records.append({
            "query_id"          : item["query_id"],
            "query_text"        : item["query_text"],
            "correct_video"     : item["video_id"],
            "answer_time_s"     : float(item["answer_time_s"]),
            "frame_id"          : item.get("frame_id"),
            "query_category"    : item.get("query_category", "other"),
            "task_type"         : item.get("task_type", "KIS"),
            "difficulty"        : item.get("difficulty", "unknown"),
            "evaluation_split"  : item.get("evaluation_split", "unspecified"),
            "answer_text"       : item.get("answer_text"),
        })

    return canonical_json_hash({
        "query_set"                : name,
        "corpus_id"                : corpus_id,
        "benchmark_content_hashes" : benchmark_hashes,
        "queries"                  : records,
    })


def corpus_hash(
    corpus_id : str,
    config : dict[str, Any],
    benchmarks : dict[str, dict[str, Any]],
) -> str :
    records = []

    for key in config["corpora"][corpus_id]["benchmarks"] :
        benchmark = benchmarks[key]
        records.append({
            "stage_id"               : benchmark["stage_id"],
            "benchmark_content_hash" : benchmark["benchmark_content_hash"],
            "window_policy_hash"     : benchmark["window_policy_hash"],
            "selected_video_ids"     : sorted(benchmark["selected_video_ids"]),
            "window_ids"             : sorted(item["window_id"] for item in benchmark["windows"]),
        })

    return canonical_json_hash({
        "corpus"     : corpus_id,
        "benchmarks" : records,
    })


def validate_query_and_corpus_contract(
    config : dict[str, Any],
    benchmarks : dict[str, dict[str, Any]],
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]] :
    query_frames = {}
    checks = []
    errors = []

    holdout_ids = set(
        _query_rows(benchmarks["extension40"])
        .loc[lambda frame : frame["evaluation_split"] == "holdout20", "query_id"]
        .astype(str)
        .tolist()
    )

    for name, specification in config["query_sets"].items() :
        frame = build_query_set(name, config, benchmarks)
        query_frames[name] = frame

        corpus_id = specification["corpus"]
        benchmark_keys = config["corpora"][corpus_id]["benchmarks"]
        benchmark_hashes = [
            benchmarks[key]["benchmark_content_hash"]
            for key in benchmark_keys
        ]
        actual_hash = query_set_hash(name, frame, corpus_id, benchmark_hashes)
        expected_hash = specification["expected_query_hash"]
        count_ok = len(frame) == int(specification["expected_query_count"])
        hash_ok  = actual_hash == expected_hash
        holdout_overlap = sorted(set(frame["query_id"].astype(str)) & holdout_ids)

        checks.append({
            "query_set"       : name,
            "query_count"     : len(frame),
            "expected_count"  : specification["expected_query_count"],
            "query_hash"      : actual_hash,
            "expected_hash"   : expected_hash,
            "holdout_overlap" : holdout_overlap,
            "passed"          : count_ok and hash_ok and not holdout_overlap,
        })

        if (not count_ok) :
            errors.append(f"{name}: query count mismatch")
        if (not hash_ok) :
            errors.append(f"{name}: query hash mismatch")
        if (holdout_overlap) :
            errors.append(f"{name}: holdout query leakage: {holdout_overlap}")

    corpus_checks = []

    for corpus_id, specification in config["corpora"].items() :
        actual_hash   = corpus_hash(corpus_id, config, benchmarks)
        expected_hash = specification["expected_corpus_hash"]
        benchmark_keys = specification["benchmarks"]
        window_count = sum(benchmarks[key]["window_count"] for key in benchmark_keys)
        video_count  = sum(benchmarks[key]["video_count"] for key in benchmark_keys)
        passed = (
            actual_hash == expected_hash
            and window_count == int(specification["expected_window_count"])
            and video_count == int(specification["expected_video_count"])
        )

        corpus_checks.append({
            "corpus_id"      : corpus_id,
            "window_count"   : window_count,
            "video_count"    : video_count,
            "corpus_hash"    : actual_hash,
            "expected_hash"  : expected_hash,
            "passed"         : passed,
        })

        if (not passed) :
            errors.append(f"{corpus_id}: corpus identity mismatch")

    return query_frames, {
        "passed"        : not errors,
        "query_sets"    : checks,
        "corpora"       : corpus_checks,
        "holdout_ids"   : sorted(holdout_ids),
        "errors"        : errors,
    }


def load_stage1_config(config : dict[str, Any]) -> dict[str, Any] :
    path = resolve_code_path(config["source_stage1"]["config_path"])
    return load_json(path)


def load_model_windows(
    retrieval_config : dict[str, Any],
    stage1_config : dict[str, Any],
    benchmarks : dict[str, dict[str, Any]],
    artifacts : Path,
) -> tuple[dict[tuple[str, str], pd.DataFrame], list[dict[str, Any]], list[dict[str, Any]]] :
    stage_frames = {}
    validations  = []
    identities   = []
    model_ids = [
        stage1_config["reference_model_id"],
        *stage1_config["candidate_model_ids"],
    ]

    for benchmark_key in ["core10", "extension40"] :
        benchmark = benchmarks[benchmark_key]

        for model_id in model_ids :
            output_path = resolve_artifact_path(
                artifacts,
                stage1_config["models"][model_id][f"{benchmark_key}_output"],
            )
            log_detail(
                f"{model_id}/{benchmark_key}: "
                f"loading {benchmark['window_count']} expected windows"
            )

            frame = stage1_evaluation.load_model_output_windows(
                model_id = model_id,
                output_dir = output_path,
                expected_windows = benchmark["windows"],
                benchmark = benchmark,
                progress_callback = None,
            )
            stage_frames[(model_id, benchmark_key)] = frame

            validation = stage1_evaluation.validate_model_windows_against_benchmark(
                windows = frame,
                benchmark = benchmark,
                model_id = model_id,
                require_complete = True,
            )
            validation["benchmark_key"] = benchmark_key
            validation["output_dir"]    = str(output_path)
            validations.append(validation)

        combined = pd.concat(
            [stage_frames[(model_id, benchmark_key)] for model_id in model_ids],
            ignore_index = True,
        )

        for candidate_id in stage1_config["candidate_model_ids"] :
            identity = stage1_evaluation.compare_model_window_identity(
                windows = combined,
                model_a = stage1_config["reference_model_id"],
                model_b = candidate_id,
            )
            identity["benchmark_key"] = benchmark_key
            identities.append(identity)

    return stage_frames, validations, identities


def model_corpus_windows(
    stage_frames : dict[tuple[str, str], pd.DataFrame],
    model_id : str,
    corpus_id : str,
) -> pd.DataFrame :
    benchmark_keys = ["core10"] if corpus_id == "core10" else ["core10", "extension40"]
    frames = [stage_frames[(model_id, key)] for key in benchmark_keys]
    frame = pd.concat(frames, ignore_index = True)

    if (frame["window_id"].duplicated().any()) :
        raise ValueError(f"{model_id}/{corpus_id}: duplicate window IDs")

    return frame.reset_index(drop = True)


def channel_documents(
    windows : pd.DataFrame,
    channel_spec : dict[str, Any],
) -> ChannelDocuments :
    mask, reasons = classify_eligibility(
        windows,
        text_field = channel_spec["text_field"],
        rejection_field = channel_spec["rejection_field"],
        apply_rejections = bool(channel_spec["apply_rejections"]),
    )

    return ChannelDocuments(
        model_id = channel_spec["model_id"],
        view = channel_spec["view"],
        window_ids = windows["window_id"].astype(str).tolist(),
        texts = windows[channel_spec["text_field"]].fillna("").astype(str).tolist(),
        eligibility_mask = mask,
        zero_reasons = reasons,
    )


def channel_identity_hash(
    documents : ChannelDocuments,
) -> str :
    return canonical_json_hash([
        {
            "window_id"   : window_id,
            "text"        : text,
            "eligible"    : bool(eligible),
            "zero_reason" : reason,
        }
        for window_id, text, eligible, reason in zip(
            documents.window_ids,
            documents.texts,
            documents.eligibility_mask.tolist(),
            documents.zero_reasons,
        )
    ])


def _aggregate_file_hash(entries : list[dict[str, Any]]) -> str | None :
    existing = [
        {"path" : item["path"], "sha256" : item["sha256"]}
        for item in entries
        if item["exists"]
    ]

    if (not existing) :
        return None

    return canonical_json_hash(sorted(existing, key = lambda item : item["path"]))


def model_output_hashes(
    stage1_config : dict[str, Any],
    benchmarks : dict[str, dict[str, Any]],
    artifacts : Path,
) -> dict[str, Any] :
    results = {}
    model_ids = [
        stage1_config["reference_model_id"],
        *stage1_config["candidate_model_ids"],
    ]

    for model_id in model_ids :
        results[model_id] = {}

        for benchmark_key in ["core10", "extension40"] :
            output_dir = resolve_artifact_path(
                artifacts,
                stage1_config["models"][model_id][f"{benchmark_key}_output"],
            )
            entries = []

            for video_id in sorted(benchmarks[benchmark_key]["selected_video_ids"]) :
                path = output_dir / f"{video_id}.json"
                relative = str(path.relative_to(artifacts)) if path.is_relative_to(artifacts) else str(path)
                entries.append({
                    "path"   : relative,
                    "exists" : path.exists(),
                    "sha256" : sha256_file(path) if path.exists() else None,
                })

            results[model_id][benchmark_key] = {
                "output_dir"     : str(output_dir),
                "file_count"     : sum(item["exists"] for item in entries),
                "expected_count" : len(entries),
                "aggregate_hash" : _aggregate_file_hash(entries),
                "missing"        : [item["path"] for item in entries if not item["exists"]],
            }

    return results


def validate_model_outputs(
    model_validations : list[dict[str, Any]],
    identity_validations : list[dict[str, Any]],
) -> dict[str, Any] :
    failures = []

    for group, items in [
        ("model_outputs", model_validations),
        ("cross_model_identity", identity_validations),
    ] :
        for item in items :
            if (not item.get("passed", False)) :
                failures.append({"group" : group, "detail" : item})

    return {
        "passed" : not failures,
        "model_outputs" : model_validations,
        "cross_model_identity" : identity_validations,
        "failures" : failures,
    }


def validate_channels(
    stage_frames : dict[tuple[str, str], pd.DataFrame],
    config : dict[str, Any],
) -> dict[str, Any] :
    rows = []
    errors = []

    for corpus_id in ["core10", "all50"] :
        for channel_id, specification in config["channels"].items() :
            windows = model_corpus_windows(
                stage_frames,
                specification["model_id"],
                corpus_id,
            )
            documents = channel_documents(windows, specification)

            row = {
                "corpus_id"             : corpus_id,
                "channel_id"            : channel_id,
                "model_id"              : documents.model_id,
                "view"                  : documents.view,
                "physical_window_count" : len(documents.window_ids),
                "eligible_window_count" : int(documents.eligibility_mask.sum()),
                "channel_hash"           : channel_identity_hash(documents),
            }
            rows.append(row)

            expected_count = int(config["corpora"][corpus_id]["expected_window_count"])

            if (len(documents.window_ids) != expected_count) :
                errors.append(
                    f"{channel_id}/{corpus_id}: "
                    f"{len(documents.window_ids)} windows != {expected_count}"
                )

    return {
        "passed" : not errors,
        "channels" : rows,
        "errors" : errors,
    }


def evaluate_bundle(
    bundle : ScoreBundle,
    query_frame : pd.DataFrame,
    windows : pd.DataFrame,
    documents : ChannelDocuments,
    query_set : str,
    corpus_id : str,
    config : dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame] :
    policy = config["evaluation_policy"]

    return evaluate_score_matrix(
        scores = bundle.scores,
        queries = query_frame,
        windows = windows,
        model_id = bundle.model_id,
        view = bundle.view,
        method_id = bundle.method_id,
        query_set = query_set,
        corpus_id = corpus_id,
        silver_radius_s = float(policy["silver_radius_s"]),
        minimum_overlap_s = float(policy["minimum_overlap_s"]),
        tie_tolerance = float(policy["tie_tolerance"]),
        eligibility_mask = documents.eligibility_mask,
        zero_reasons = documents.zero_reasons,
        query_coverage = bundle.metadata.get("query_coverage"),
    )


def run_stage00(
    config : dict[str, Any],
    query_frames : dict[str, pd.DataFrame],
    stage_frames : dict[tuple[str, str], pd.DataFrame],
    fixture : dict[str, Any],
) -> tuple[list[ScoreBundle], pd.DataFrame, pd.DataFrame, dict[str, Any]] :
    semantic_spec = config["stage00_baseline"]["semantic"]
    semantic = E5CompatibilityScorer(
        model_name = semantic_spec["model_name"],
        revision = semantic_spec["revision"],
        query_prefix = semantic_spec["query_prefix"],
        passage_prefix = semantic_spec["passage_prefix"],
        show_progress_bar = True,
    )

    bundles = []
    result_frames = []
    evidence_frames = []

    for query_set in config["stage00_baseline"]["query_sets"] :
        query_frame = query_frames[query_set]
        corpus_id   = config["query_sets"][query_set]["corpus"]
        query_texts = query_frame["query_text"].astype(str).tolist()
        query_ids   = query_frame["query_id"].astype(str).tolist()

        log_detail(
            f"{query_set}: {len(query_frame)} queries against "
            f"{config['corpora'][corpus_id]['expected_window_count']} windows"
        )

        for channel_id, channel_spec in config["channels"].items() :
            windows = model_corpus_windows(
                stage_frames,
                channel_spec["model_id"],
                corpus_id,
            )
            documents = channel_documents(windows, channel_spec)
            log_detail(
                f"{query_set}/{channel_id}: "
                f"{documents.eligibility_mask.sum()}/{len(documents.window_ids)} eligible"
            )

            lexical = score_query_fitted_dual_tfidf(
                query_texts,
                query_ids,
                documents,
                ngram_range = (
                    int(config["stage00_baseline"]["lexical"]["ngram_min"]),
                    int(config["stage00_baseline"]["lexical"]["ngram_max"]),
                ),
                method_id = "baseline_v1_lexical",
            )
            semantic_bundle = semantic.score(
                query_texts,
                query_ids,
                documents,
                method_id = "baseline_v1_semantic",
            )
            baseline = weighted_score_bundle(
                lexical,
                semantic_bundle,
                lexical_weight = float(config["stage00_baseline"]["lexical_weight"]),
                semantic_weight = float(config["stage00_baseline"]["semantic_weight"]),
                method_id = "baseline_v1",
            )
            baseline.metadata["query_set"] = query_set
            baseline.metadata["corpus_id"] = corpus_id
            bundles.append(baseline)

            results, evidence = evaluate_bundle(
                baseline,
                query_frame,
                windows,
                documents,
                query_set,
                corpus_id,
                config,
            )
            result_frames.append(results)
            evidence_frames.append(evidence)

    query_results = pd.concat(result_frames, ignore_index = True)
    top_evidence  = pd.concat(evidence_frames, ignore_index = True)
    metrics = summarize_retrieval_metrics(query_results)
    aggregate_regression = compare_expected_metrics(
        metrics,
        config["stage00_baseline"]["expected_metrics"],
    )
    fixture_regression = compare_regression_fixture(
        query_results,
        metrics,
        fixture,
    )
    regression = {
        "passed"            : bool(
            aggregate_regression["passed"]
            and fixture_regression["passed"]
        ),
        "aggregate_metrics" : aggregate_regression,
        "strict_fixture"    : fixture_regression,
        "errors"            : [
            *aggregate_regression.get("errors", []),
            *fixture_regression.get("errors", []),
        ],
    }

    return bundles, query_results, top_evidence, regression


def run_stage01(
    config : dict[str, Any],
    query_frames : dict[str, pd.DataFrame],
    stage_frames : dict[tuple[str, str], pd.DataFrame],
) -> tuple[list[ScoreBundle], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame] :
    stage_config = config["stage01_lexical"]
    query_set    = stage_config["query_set"]
    query_frame  = query_frames[query_set]
    corpus_id    = config["query_sets"][query_set]["corpus"]
    query_texts  = query_frame["query_text"].astype(str).tolist()
    query_ids    = query_frame["query_id"].astype(str).tolist()

    bundles = []
    result_frames = []
    evidence_frames = []

    for channel_id, channel_spec in config["channels"].items() :
        windows = model_corpus_windows(
            stage_frames,
            channel_spec["model_id"],
            corpus_id,
        )
        documents = channel_documents(windows, channel_spec)
        log_detail(
            f"{channel_id}: {documents.eligibility_mask.sum()}/"
            f"{len(documents.window_ids)} eligible windows"
        )

        methods : dict[str, ScoreBundle] = {}

        l0 = stage_config["methods"]["L0_query_tfidf"]
        methods["L0_query_tfidf"] = score_query_fitted_dual_tfidf(
            query_texts,
            query_ids,
            documents,
            ngram_range = (int(l0["ngram_min"]), int(l0["ngram_max"])),
            method_id = "L0_query_tfidf",
        )

        l1 = stage_config["methods"]["L1_corpus_tfidf"]
        methods["L1_corpus_tfidf"] = score_corpus_fitted_dual_tfidf(
            query_texts,
            query_ids,
            documents,
            ngram_range = (int(l1["ngram_min"]), int(l1["ngram_max"])),
            method_id = "L1_corpus_tfidf",
        )

        l2 = stage_config["methods"]["L2_bm25_preserving"]
        methods["L2_bm25_preserving"] = score_bm25(
            query_texts,
            query_ids,
            documents,
            fold_accents = False,
            k1 = float(l2["k1"]),
            b = float(l2["b"]),
            method_id = "L2_bm25_preserving",
        )

        l3 = stage_config["methods"]["L3_bm25_folded"]
        methods["L3_bm25_folded"] = score_bm25(
            query_texts,
            query_ids,
            documents,
            fold_accents = True,
            k1 = float(l3["k1"]),
            b = float(l3["b"]),
            method_id = "L3_bm25_folded",
        )

        l4 = stage_config["methods"]["L4_bm25_rrf"]
        methods["L4_bm25_rrf"] = positive_evidence_rrf(
            methods[l4["left_method"]],
            methods[l4["right_method"]],
            k = int(l4["k"]),
            tie_tolerance = float(config["evaluation_policy"]["tie_tolerance"]),
            method_id = "L4_bm25_rrf",
        )

        for method_id in [
            "L0_query_tfidf",
            "L1_corpus_tfidf",
            "L2_bm25_preserving",
            "L3_bm25_folded",
            "L4_bm25_rrf",
        ] :
            bundle = methods[method_id]
            bundle.metadata["query_set"] = query_set
            bundle.metadata["corpus_id"] = corpus_id
            bundles.append(bundle)
            results, evidence = evaluate_bundle(
                bundle,
                query_frame,
                windows,
                documents,
                query_set,
                corpus_id,
                config,
            )
            result_frames.append(results)
            evidence_frames.append(evidence)

    query_results = pd.concat(result_frames, ignore_index = True)
    top_evidence  = pd.concat(evidence_frames, ignore_index = True)
    metrics       = summarize_retrieval_metrics(query_results)
    comparisons   = compare_methods(
        query_results,
        reference_method = config["selection"]["reference_method"],
    )
    method_summary = summarize_stage1_methods(
        query_results,
        reference_method = config["selection"]["reference_method"],
        bootstrap_samples = int(config["selection"]["bootstrap_samples"]),
        confidence = float(config["selection"]["bootstrap_confidence"]),
        seed = int(config["selection"]["bootstrap_seed"]),
    )

    return bundles, query_results, top_evidence, comparisons, method_summary


def _safe_key(value : str) -> str :
    return "".join(char if char.isalnum() else "_" for char in value)


def save_score_cache(
    bundles : list[ScoreBundle],
    query_results : pd.DataFrame,
    cache_root : Path,
    stage : str,
) -> dict[str, Any] :
    target = cache_root / stage
    target.mkdir(parents = True, exist_ok = True)

    arrays = {}
    axes   = {
        "schema_version" : "1.0",
        "stage"          : stage,
        "bundles"        : [],
    }

    for index, bundle in enumerate(bundles) :
        query_set = str(bundle.metadata.get("query_set", "unknown"))
        base = _safe_key(
            f"{index:03d}__{query_set}__{bundle.model_id}__"
            f"{bundle.view}__{bundle.method_id}"
        )
        arrays[f"{base}__scores"]      = bundle.scores
        arrays[f"{base}__eligibility"] = bundle.eligibility_mask.astype(np.uint8)

        component_keys = {}

        for component_name, component_values in bundle.component_scores.items() :
            key = f"{base}__component__{_safe_key(component_name)}"
            arrays[key] = np.asarray(component_values)
            component_keys[component_name] = key

        axes["bundles"].append({
            "array_key"       : f"{base}__scores",
            "eligibility_key" : f"{base}__eligibility",
            "component_keys"  : component_keys,
            "query_set"       : query_set,
            "method_id"       : bundle.method_id,
            "model_id"        : bundle.model_id,
            "view"            : bundle.view,
            "query_ids"       : bundle.query_ids,
            "window_ids"      : bundle.window_ids,
            "score_dtype"     : str(bundle.scores.dtype),
            "metadata"        : bundle.metadata,
        })

    npz_path  = target / "window_scores.npz"
    axes_path = target / "score_axes.json"
    np.savez_compressed(npz_path, **arrays)
    write_json(axes_path, axes)

    return {
        "window_scores" : {
            "path"   : str(npz_path),
            "sha256" : sha256_file(npz_path),
        },
        "score_axes" : {
            "path"   : str(axes_path),
            "sha256" : sha256_file(axes_path),
        },
    }


def git_state() -> dict[str, Any] :
    try :
        commit = subprocess.check_output(
            ["git", "-C", str(CODE_ROOT), "rev-parse", "HEAD"],
            text = True,
            stderr = subprocess.DEVNULL,
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "-C", str(CODE_ROOT), "status", "--porcelain"],
                text = True,
                stderr = subprocess.DEVNULL,
            ).strip()
        )
        return {"commit" : commit, "dirty" : dirty}
    except Exception :
        return {"commit" : None, "dirty" : None}


def package_version(name : str) -> str | None :
    try :
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError :
        return None


def build_manifest(
    stage : str,
    config : dict[str, Any],
    benchmarks : dict[str, dict[str, Any]],
    benchmark_paths : dict[str, Path],
    artifacts : Path,
    stage1_config : dict[str, Any],
    query_validation : dict[str, Any],
    channel_validation : dict[str, Any],
    output_hashes : dict[str, Any],
    cache_records : dict[str, Any],
) -> dict[str, Any] :
    sources = {
        "retrieval_config"       : source_record(CONFIG_PATH),
        "retrieval_v2"           : source_record(SRC_ROOT / "retrieval_v2.py"),
        "retrieval_v2_evaluation": source_record(SRC_ROOT / "retrieval_v2_evaluation.py"),
        "runner"                 : source_record(SCRIPT_PATH),
        "stage1_config"          : source_record(resolve_code_path(config["source_stage1"]["config_path"])),
        "stage1_evaluation"      : source_record(Path(stage1_evaluation.__file__).resolve()),
        "postprocess"            : source_record(Path(postprocess.__file__).resolve()),
        "regression_fixture"     : source_record(
            resolve_code_path(config["regression"]["fixture_path"])
        ),
        "benchmarks"             : {
            key : source_record(path)
            for key, path in benchmark_paths.items()
        },
    }

    return {
        "schema_version"  : "1.0",
        "retrieval_id"    : config["retrieval_id"],
        "stage"           : stage,
        "generated_at_utc": utc_now(),
        "code_root"       : str(CODE_ROOT),
        "artifact_root"   : str(artifacts),
        "git"             : git_state(),
        "sources"         : sources,
        "benchmarks"      : {
            key : {
                "stage_id"               : benchmark["stage_id"],
                "benchmark_content_hash" : benchmark["benchmark_content_hash"],
                "window_policy_hash"     : benchmark["window_policy_hash"],
                "video_count"            : benchmark["video_count"],
                "query_count"            : benchmark["query_count"],
                "window_count"           : benchmark["window_count"],
            }
            for key, benchmark in benchmarks.items()
        },
        "query_identity"   : query_validation,
        "channel_identity" : channel_validation,
        "model_outputs"    : output_hashes,
        "stage00_baseline" : config["stage00_baseline"],
        "stage01_lexical"  : config["stage01_lexical"],
        "evaluation_policy": config["evaluation_policy"],
        "regression"       : config["regression"],
        "cache"            : cache_records,
        "environment" : {
            "python"                : sys.version,
            "platform"              : platform.platform(),
            "numpy"                 : np.__version__,
            "pandas"                : pd.__version__,
            "scikit_learn"          : sklearn.__version__,
            "sentence_transformers" : package_version("sentence-transformers"),
            "torch"                 : package_version("torch"),
            "transformers"          : package_version("transformers"),
        },
    }


def stage00_prerequisite(
    config : dict[str, Any],
    artifacts : Path,
) -> dict[str, Any] :
    stage_root = resolve_artifact_path(
        artifacts,
        config["paths"]["reports_root"],
    ) / "stage00_baseline"
    summary_path = stage_root / "stage_summary.json"

    if (not summary_path.exists()) :
        raise RuntimeError(
            "Stage 1 lexical retrieval requires a completed Stage 0 run. "
            f"Missing: {summary_path}"
        )

    summary = load_json(summary_path)
    current_config_hash = sha256_file(CONFIG_PATH)

    if (not summary.get("passed", False)) :
        raise RuntimeError("Stage 0 did not pass")

    if (summary.get("retrieval_config_sha256") != current_config_hash) :
        raise RuntimeError(
            "Retrieval config changed after Stage 0. Re-run Stage 0 first."
        )

    return summary


def compact_stage00_regression(regression : dict[str, Any] | None) -> dict[str, Any] | None:
    if (regression is None) :
        return None

    aggregate = regression.get("aggregate_metrics", {})
    strict    = regression.get("strict_fixture", {})
    aggregate_checks = aggregate.get("checks", [])

    return {
        "passed" : bool(regression.get("passed", False)),
        "aggregate_metrics" : {
            "passed"       : bool(aggregate.get("passed", False)),
            "check_count"  : len(aggregate_checks),
            "failed_count" : sum(not item.get("passed", False) for item in aggregate_checks),
            "errors"       : aggregate.get("errors", [])[:20],
        },
        "strict_fixture" : {
            "passed"                          : bool(strict.get("passed", False)),
            "fixture_id"                      : strict.get("fixture_id"),
            "expected_query_row_count"        : strict.get("expected_query_row_count"),
            "passed_query_row_count"          : strict.get("passed_query_row_count"),
            "expected_aggregate_row_count"    : strict.get("expected_aggregate_row_count"),
            "passed_aggregate_row_count"      : strict.get("passed_aggregate_row_count"),
            "maximum_query_score_difference"  : strict.get("maximum_query_score_difference"),
            "maximum_rank_metric_difference"  : strict.get("maximum_rank_metric_difference"),
            "maximum_score_metric_difference" : strict.get("maximum_score_metric_difference"),
            "score_tolerance"                 : strict.get("score_tolerance"),
            "rank_metric_tolerance"           : strict.get("rank_metric_tolerance"),
            "error_count"                     : len(strict.get("errors", [])),
            "errors"                          : strict.get("errors", [])[:20],
        },
    }


def main() -> None :
    args   = parse_args()
    config = load_retrieval_config()
    artifacts = artifact_root(config)
    reports_root = resolve_artifact_path(artifacts, config["paths"]["reports_root"]) / args.stage
    cache_root   = resolve_artifact_path(artifacts, config["paths"]["cache_root"])
    reports_root.mkdir(parents = True, exist_ok = True)

    print("=" * 88)
    print("AIC 2026 RETRIEVAL V2")
    print("=" * 88)
    print(f"Stage:         {args.stage}")
    print(f"Code root:     {CODE_ROOT}")
    print(f"Artifact root: {artifacts}")
    print(f"Reports:       {reports_root}")
    print()

    total_stages = 7

    start = log_stage(1, total_stages, "Loading configuration and benchmark manifests...")
    source_validation = validate_source_contract(config)
    stage1_config     = load_stage1_config(config)
    fixture, fixture_path, fixture_file_validation = load_regression_fixture(config)
    benchmarks, benchmark_paths, benchmark_validations = load_benchmarks(config)

    if (not source_validation["passed"]) :
        write_json(reports_root / "validation_summary.json", {
            "passed" : False,
            "source_contract" : source_validation,
            "benchmarks" : benchmark_validations,
        })
        raise RuntimeError("Frozen Stage 1 source contract failed")

    if (not fixture_file_validation["passed"]) :
        write_json(reports_root / "validation_summary.json", {
            "passed"             : False,
            "source_contract"    : source_validation,
            "regression_fixture" : fixture_file_validation,
            "benchmarks"         : benchmark_validations,
        })
        raise RuntimeError("Regression fixture validation failed")

    if (any(not item.get("passed", False) for item in benchmark_validations)) :
        write_json(reports_root / "validation_summary.json", {
            "passed"             : False,
            "source_contract"    : source_validation,
            "regression_fixture" : fixture_file_validation,
            "benchmarks"         : benchmark_validations,
        })
        raise RuntimeError("Benchmark validation failed")

    log_done(start)

    start = log_stage(2, total_stages, "Validating query, corpus, and ASR artifact identity...")
    query_frames, query_validation = validate_query_and_corpus_contract(
        config,
        benchmarks,
    )

    if (not query_validation["passed"]) :
        raise RuntimeError(
            f"Query/corpus validation failed: {query_validation['errors']}"
        )

    stage_frames, model_validations, identity_validations = load_model_windows(
        config,
        stage1_config,
        benchmarks,
        artifacts,
    )
    model_validation = validate_model_outputs(
        model_validations,
        identity_validations,
    )

    if (not model_validation["passed"]) :
        raise RuntimeError("ASR model-output validation failed")

    output_hashes = model_output_hashes(
        stage1_config,
        benchmarks,
        artifacts,
    )
    missing_outputs = [
        f"{model_id}/{benchmark_key}"
        for model_id, model_data in output_hashes.items()
        for benchmark_key, item in model_data.items()
        if item["file_count"] != item["expected_count"]
    ]

    if (missing_outputs) :
        raise RuntimeError(f"Missing ASR output files: {missing_outputs}")

    fixture_provenance = validate_fixture_provenance(
        fixture,
        config,
        benchmark_paths,
        benchmarks,
        output_hashes,
    )

    if (not fixture_provenance["passed"]) :
        raise RuntimeError(
            f"Regression fixture provenance mismatch: {fixture_provenance['errors']}"
        )

    log_done(start)

    start = log_stage(3, total_stages, "Preparing independent transcript channels...")
    channel_validation = validate_channels(stage_frames, config)

    if (not channel_validation["passed"]) :
        raise RuntimeError(
            f"Channel validation failed: {channel_validation['errors']}"
        )

    write_json(reports_root / "validation_summary.json", {
        "passed"             : True,
        "source_contract"    : source_validation,
        "regression_fixture" : {
            "file"       : fixture_file_validation,
            "provenance" : fixture_provenance,
        },
        "benchmarks"         : benchmark_validations,
        "query_and_corpus"   : query_validation,
        "model_outputs"      : model_validation,
        "channels"           : channel_validation,
    })
    log_done(start)

    start = log_stage(4, total_stages, "Running retrieval...")
    regression = None
    comparisons = pd.DataFrame()
    method_summary = pd.DataFrame()

    if (args.stage == "stage00_baseline") :
        bundles, query_results, top_evidence, regression = run_stage00(
            config,
            query_frames,
            stage_frames,
            fixture,
        )
        metrics = summarize_retrieval_metrics(query_results)

    else :
        stage00_prerequisite(config, artifacts)
        bundles, query_results, top_evidence, comparisons, method_summary = run_stage01(
            config,
            query_frames,
            stage_frames,
        )
        metrics = summarize_retrieval_metrics(query_results)

    log_done(start)

    start = log_stage(5, total_stages, "Evaluating rankings and regression/paired comparisons...")
    holdout_ids = set(query_validation["holdout_ids"])
    result_query_ids = set(query_results["query_id"].astype(str))
    holdout_overlap = sorted(result_query_ids & holdout_ids)

    if (holdout_overlap) :
        raise RuntimeError(
            f"Holdout query leakage detected in results: {holdout_overlap}"
        )

    if (args.stage == "stage00_baseline" and not regression["passed"]) :
        log_detail("Stage 0 regression: FAIL")
    elif (args.stage == "stage00_baseline") :
        log_detail("Stage 0 aggregate + strict per-query regression: PASS")
    else :
        log_detail("Stage 1 paired comparison tables built")
    log_done(start)

    start = log_stage(6, total_stages, "Writing reports and window-score cache...")
    cache_records = save_score_cache(
        bundles,
        query_results,
        cache_root,
        args.stage,
    )

    write_csv(reports_root / "retrieval_metrics.csv", metrics)
    write_csv(reports_root / "query_results.csv", query_results)

    if (args.stage == "stage00_baseline") :
        write_json(reports_root / "regression.json", regression)
    else :
        write_csv(reports_root / "query_comparison.csv", comparisons)
        diagnostics_columns = [
            "query_set", "method_id", "model_id", "view", "query_id",
            "eligible_window_count", "positive_window_count",
            "positive_window_fraction", "positive_video_count",
            "correct_video_positive_windows", "query_coverage",
            "zero_evidence", "story_score_margin", "video_score_margin",
        ]
        write_csv(
            reports_root / "lexical_diagnostics.csv",
            query_results[diagnostics_columns],
        )
        write_csv(reports_root / "method_summary.csv", method_summary)
        write_csv(reports_root / "top_evidence.csv", top_evidence)

    log_done(start)

    start = log_stage(7, total_stages, "Finalizing reproducibility manifest and stage status...")
    manifest = build_manifest(
        args.stage,
        config,
        benchmarks,
        benchmark_paths,
        artifacts,
        stage1_config,
        query_validation,
        channel_validation,
        output_hashes,
        cache_records,
    )
    write_json(reports_root / "retrieval_manifest.json", manifest)

    strict_regression = (
        regression.get("strict_fixture")
        if regression is not None
        else None
    )
    fixture_status = {
        "path"       : str(fixture_path),
        "required"   : bool(config["regression"]["fixture_required"]),
        "exists"     : fixture_path.exists(),
        "checked"    : strict_regression is not None,
        "file"       : fixture_file_validation,
        "provenance" : fixture_provenance,
        "regression" : (
            compact_stage00_regression(regression).get("strict_fixture")
            if regression is not None
            else None
        ),
        "passed"     : bool(
            fixture_file_validation["passed"]
            and fixture_provenance["passed"]
            and (strict_regression["passed"] if strict_regression is not None else True)
        ),
        "note"       : (
            "Stage 0 uses the frozen historical per-query Baseline v1 fixture. "
            "Ranks and top IDs must match exactly; score fields use the fixture tolerance."
        ),
    }

    passed = bool(
        source_validation["passed"]
        and query_validation["passed"]
        and model_validation["passed"]
        and channel_validation["passed"]
        and fixture_status["passed"]
        and not holdout_overlap
        and (regression["passed"] if regression is not None else True)
    )

    summary = {
        "schema_version"          : "1.0",
        "retrieval_id"            : config["retrieval_id"],
        "stage"                   : args.stage,
        "generated_at_utc"        : utc_now(),
        "passed"                  : passed,
        "retrieval_config_sha256" : sha256_file(CONFIG_PATH),
        "holdout_query_overlap"   : holdout_overlap,
        "regression"              : compact_stage00_regression(regression),
        "strict_fixture"          : fixture_status,
        "reports_root"            : str(reports_root),
        "cache"                   : cache_records,
        "provisional_default"     : config["selection"]["provisional_default"],
    }

    if (args.stage == "stage01_lexical") :
        summary["method_summary"] = method_summary.to_dict(orient = "records")
        summary["decision_required"] = True
        summary["decision_note"] = (
            "Review paired query evidence before selecting a provisional lexical default. "
            "No method is selected automatically."
        )

    write_json(reports_root / "stage_summary.json", summary)
    log_done(start)

    print()
    print("=" * 88)
    print("RETRIEVAL V2 COMPLETE")
    print("=" * 88)
    print(f"Stage:   {args.stage}")
    print(f"Status:  {'PASS' if passed else 'FAIL'}")
    print(f"Reports: {reports_root}")
    print(f"Summary: {reports_root / 'stage_summary.json'}")

    if (not passed) :
        raise SystemExit(2)


if (__name__ == "__main__") :
    main()
