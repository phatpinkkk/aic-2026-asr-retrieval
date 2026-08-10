# Relative path: src/07_evaluate_retrieval_v2.py
# Purpose: Orchestrate Retrieval v2 Stage 0 baseline, Stage 1 lexical, and Stage 2 dense retrieval experiments.

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
import retrieval_backends
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
    score_dense_embeddings,
    weighted_score_bundle,
)
from retrieval_backends import DenseBackendSpec, DenseEmbeddingBundle, SentenceTransformerDenseBackend
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
        description = "Run AIC 2026 Retrieval v2 Stage 0, Stage 1, or Stage 2.",
    )
    parser.add_argument(
        "--stage",
        choices = ["stage00_baseline", "stage01_lexical", "stage02_dense"],
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
        "stage02_dense",
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




def _embedding_cache_paths(cache_root : Path, stage_config : dict[str, Any], backend_id : str, group : str, item_id : str) -> tuple[Path, Path] :
    base = cache_root / stage_config["cache"]["embeddings_subdir"] / _safe_key(backend_id) / group / _safe_key(item_id)
    return base.with_suffix(".npz"), base.with_suffix(".json")


def _embedding_identity_hash(identity : dict[str, Any]) -> str :
    return canonical_json_hash(identity)


def _load_embedding_cache(npz_path : Path, json_path : Path, expected_ids : list[str], identity : dict[str, Any]) -> DenseEmbeddingBundle | None :
    if (not npz_path.exists() or not json_path.exists()) : return None
    metadata = load_json(json_path)
    if (metadata.get("identity_hash") != _embedding_identity_hash(identity)) : return None
    if (metadata.get("ids") != expected_ids) : return None
    if (metadata.get("npz_sha256") != sha256_file(npz_path)) : return None
    with np.load(npz_path, allow_pickle = False) as payload : embeddings = np.asarray(payload["embeddings"], dtype = np.float32)
    return DenseEmbeddingBundle(ids = expected_ids, embeddings = embeddings, metadata = metadata.get("embedding_metadata", {}))


def _write_embedding_cache(npz_path : Path, json_path : Path, bundle : DenseEmbeddingBundle, identity : dict[str, Any]) -> dict[str, Any] :
    npz_path.parent.mkdir(parents = True, exist_ok = True)
    np.savez_compressed(npz_path, embeddings = bundle.embeddings)
    record = {"schema_version" : "1.0", "identity" : identity, "identity_hash" : _embedding_identity_hash(identity), "ids" : bundle.ids, "embedding_metadata" : bundle.metadata, "dimension" : bundle.dimension, "dtype" : str(bundle.embeddings.dtype), "npz_sha256" : sha256_file(npz_path)}
    write_json(json_path, record)
    return {"npz_path" : str(npz_path), "json_path" : str(json_path), "npz_sha256" : record["npz_sha256"], "json_sha256" : sha256_file(json_path), "identity_hash" : record["identity_hash"]}


def _query_embedding_identity(stage_config : dict[str, Any], specification : DenseBackendSpec, query_set : str, query_ids : list[str], query_texts : list[str]) -> dict[str, Any] :
    cache = stage_config["cache"]
    return {"kind" : "queries", "dense_backends_version" : retrieval_backends.DENSE_BACKENDS_VERSION, "embedding_cache_version" : cache["embedding_cache_version"], "normalization_policy" : cache["normalization_policy"], "backend" : specification.identity(), "query_set" : query_set, "ordered_query_ids" : query_ids, "query_text_hash" : canonical_json_hash(query_texts), "input_policy_hash" : canonical_json_hash({"prefix" : specification.query_prefix, "prompt_name" : specification.query_prompt_name, "normalize_embeddings" : specification.normalize_embeddings, "normalization_policy" : specification.normalization_policy})}


def _document_embedding_identity(stage_config : dict[str, Any], specification : DenseBackendSpec, channel_id : str, documents : ChannelDocuments) -> dict[str, Any] :
    cache          = stage_config["cache"]
    eligible_ids   = [documents.window_ids[index] for index in documents.eligible_indices]
    eligible_texts = [documents.texts[index] for index in documents.eligible_indices]
    return {"kind" : "documents", "dense_backends_version" : retrieval_backends.DENSE_BACKENDS_VERSION, "embedding_cache_version" : cache["embedding_cache_version"], "normalization_policy" : cache["normalization_policy"], "backend" : specification.identity(), "channel_id" : channel_id, "ordered_physical_window_ids_hash" : canonical_json_hash(documents.window_ids), "ordered_eligible_window_ids" : eligible_ids, "eligibility_mask_hash" : canonical_json_hash(documents.eligibility_mask.astype(int).tolist()), "selected_text_hash" : canonical_json_hash(eligible_texts), "input_policy_hash" : canonical_json_hash({"prefix" : specification.document_prefix, "prompt_name" : specification.document_prompt_name, "normalize_embeddings" : specification.normalize_embeddings, "normalization_policy" : specification.normalization_policy})}


def _gpu_memory_snapshot() -> dict[str, Any] :
    try :
        import torch
        if (not torch.cuda.is_available()) : return {"cuda_available" : False, "peak_allocated_bytes" : None, "peak_reserved_bytes" : None}
        return {"cuda_available" : True, "device_name" : torch.cuda.get_device_name(0), "peak_allocated_bytes" : int(torch.cuda.max_memory_allocated()), "peak_reserved_bytes" : int(torch.cuda.max_memory_reserved())}
    except Exception :
        return {"cuda_available" : False, "peak_allocated_bytes" : None, "peak_reserved_bytes" : None}


def _reset_gpu_peak_memory() -> None :
    try :
        import torch
        if (torch.cuda.is_available()) : torch.cuda.reset_peak_memory_stats()
    except Exception :
        pass


def validate_stage02_environment(config : dict[str, Any]) -> dict[str, Any] :
    stage_config = config["stage02_dense"]
    errors = []
    checks = []
    try :
        from packaging.version import Version
    except Exception :
        Version = None
    installed_transformers          = package_version("transformers")
    installed_sentence_transformers = package_version("sentence-transformers")
    if (installed_sentence_transformers is None) : errors.append("sentence-transformers is not installed")
    for backend_id, specification in stage_config["backends"].items() :
        minimum_transformers = specification.get("minimum_transformers_version")
        minimum_sentence     = specification.get("minimum_sentence_transformers_version")
        transformers_ok      = minimum_transformers is None or (Version is not None and installed_transformers is not None and Version(installed_transformers) >= Version(str(minimum_transformers)))
        sentence_ok          = minimum_sentence is None or (Version is not None and installed_sentence_transformers is not None and Version(installed_sentence_transformers) >= Version(str(minimum_sentence)))
        checks.append({"backend_id" : backend_id, "minimum_transformers_version" : minimum_transformers, "installed_transformers_version" : installed_transformers, "minimum_sentence_transformers_version" : minimum_sentence, "installed_sentence_transformers_version" : installed_sentence_transformers, "passed" : transformers_ok and sentence_ok})
        if (not transformers_ok) : errors.append(f"{backend_id} requires transformers>={minimum_transformers}; found {installed_transformers}")
        if (not sentence_ok) : errors.append(f"{backend_id} requires sentence-transformers>={minimum_sentence}; found {installed_sentence_transformers}")
    return {"passed" : not errors, "sentence_transformers" : installed_sentence_transformers, "transformers" : installed_transformers, "gpu" : _gpu_memory_snapshot(), "checks" : checks, "errors" : errors}


def _dense_query_diagnostics(bundle : ScoreBundle, query_frame : pd.DataFrame, windows : pd.DataFrame, documents : ChannelDocuments, channel_id : str, top_k_values : list[int]) -> pd.DataFrame :
    rows = []
    video_ids = windows["video_id"].astype(str).to_numpy()
    valid_indices = np.flatnonzero(documents.eligibility_mask)
    if (not len(valid_indices)) : raise ValueError(f"{channel_id}: dense diagnostics require at least one eligible window")
    for query_index, (_, query) in enumerate(query_frame.reset_index(drop = True).iterrows()) :
        valid_scores = bundle.scores[query_index, valid_indices]
        order = valid_indices[np.argsort(-valid_scores, kind = "stable")]
        row = {"query_set" : bundle.metadata["query_set"], "method_id" : bundle.method_id, "model_id" : bundle.model_id, "view" : bundle.view, "channel_id" : channel_id, "query_id" : str(query["query_id"]), "eligible_window_count" : len(valid_indices), "score_mean" : float(valid_scores.mean()), "score_std" : float(valid_scores.std()), "top_score" : float(valid_scores.max()), "bottom_score" : float(valid_scores.min())}
        for top_k in top_k_values :
            selected = order[:min(top_k, len(order))]
            selected_scores = bundle.scores[query_index, selected]
            row[f"top{top_k}_score_mean"] = float(selected_scores.mean()) if len(selected_scores) else None
            row[f"correct_video_windows_in_top{top_k}"] = int((video_ids[selected] == str(query["video_id"])).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def _dense_method_summary(query_results : pd.DataFrame, reference_method : str, samples : int, confidence : float, seed : int) -> pd.DataFrame :
    per_query = query_results.groupby(["method_id", "query_id"], as_index = False).agg(composite_video_rr = ("video_rr", "mean"), composite_story_rr = ("story_rr", "mean"))
    reference = per_query[per_query["method_id"] == reference_method][["query_id", "composite_video_rr", "composite_story_rr"]].rename(columns = {"composite_video_rr" : "reference_video_rr", "composite_story_rr" : "reference_story_rr"})
    generator = np.random.default_rng(seed)
    rows = []
    for method_id, group in per_query.groupby("method_id", sort = True) :
        merged = group.merge(reference, on = "query_id", how = "inner")
        video_delta = (merged["composite_video_rr"] - merged["reference_video_rr"]).to_numpy(dtype = float)
        story_delta = (merged["composite_story_rr"] - merged["reference_story_rr"]).to_numpy(dtype = float)
        estimates = np.asarray([generator.choice(video_delta, size = len(video_delta), replace = True).mean() for _ in range(samples)], dtype = float) if len(video_delta) else np.asarray([], dtype = float)
        alpha = (1.0 - confidence) / 2.0
        low, high = (float(np.quantile(estimates, alpha)), float(np.quantile(estimates, 1.0 - alpha))) if len(estimates) else (math.nan, math.nan)
        rows.append({"method_id" : method_id, "query_count" : len(merged), "composite_video_rr_mean" : float(merged["composite_video_rr"].mean()), "composite_story_rr_mean" : float(merged["composite_story_rr"].mean()), "video_rr_delta_mean_vs_D0" : float(video_delta.mean()), "video_rr_delta_median_vs_D0" : float(np.median(video_delta)), "story_rr_delta_mean_vs_D0" : float(story_delta.mean()), "video_better_query_count_vs_D0" : int((video_delta > 0).sum()), "video_tie_query_count_vs_D0" : int((video_delta == 0).sum()), "video_worse_query_count_vs_D0" : int((video_delta < 0).sum()), "video_rr_delta_bootstrap_90_low" : low, "video_rr_delta_bootstrap_90_high" : high, "provisional_default" : None})
    return pd.DataFrame(rows)


def _asr_gap_summary(metrics : pd.DataFrame) -> pd.DataFrame :
    rows = []
    for (method_id, view), group in metrics.groupby(["method_id", "view"], sort = True) :
        whisper = group[group["model_id"] == "whisper_large_v3"]
        parakeet = group[group["model_id"] == "parakeet_ctc_0_6b_vietnamese"]
        if (whisper.empty or parakeet.empty) : continue
        left, right = whisper.iloc[0], parakeet.iloc[0]
        rows.append({"method_id" : method_id, "view" : view, "whisper_video_recall_at_1" : float(left["video_recall_at_1"]), "parakeet_video_recall_at_1" : float(right["video_recall_at_1"]), "video_recall_at_1_gap" : float(left["video_recall_at_1"] - right["video_recall_at_1"]), "whisper_video_mrr" : float(left["video_mrr"]), "parakeet_video_mrr" : float(right["video_mrr"]), "video_mrr_gap" : float(left["video_mrr"] - right["video_mrr"]), "whisper_story_mrr" : float(left["story_mrr"]), "parakeet_story_mrr" : float(right["story_mrr"]), "story_mrr_gap" : float(left["story_mrr"] - right["story_mrr"])})
    return pd.DataFrame(rows)


def _asr_gap_queries(query_results : pd.DataFrame) -> pd.DataFrame :
    rows = []
    for (method_id, view, query_id), group in query_results.groupby(["method_id", "view", "query_id"], sort = True) :
        whisper = group[group["model_id"] == "whisper_large_v3"]
        parakeet = group[group["model_id"] == "parakeet_ctc_0_6b_vietnamese"]
        if (whisper.empty or parakeet.empty) : continue
        left, right = whisper.iloc[0], parakeet.iloc[0]
        rows.append({"method_id" : method_id, "view" : view, "query_id" : query_id, "whisper_video_rank" : int(left["video_rank"]), "parakeet_video_rank" : int(right["video_rank"]), "video_rank_gap_parakeet_minus_whisper" : int(right["video_rank"] - left["video_rank"]), "whisper_story_rank" : int(left["first_relevant_rank"]) if pd.notna(left["first_relevant_rank"]) else None, "parakeet_story_rank" : int(right["first_relevant_rank"]) if pd.notna(right["first_relevant_rank"]) else None, "whisper_video_rr" : float(left["video_rr"]), "parakeet_video_rr" : float(right["video_rr"]), "video_rr_gap_whisper_minus_parakeet" : float(left["video_rr"] - right["video_rr"])})
    return pd.DataFrame(rows)


def _system_reference_summary(metrics : pd.DataFrame, fixture : dict[str, Any]) -> pd.DataFrame :
    historical = pd.DataFrame(fixture["expected"]["aggregate_metrics"])
    historical = historical[(historical["query_set"] == "development20") & (historical["method_id"] == "baseline_v1")]
    rows = []
    for _, item in metrics.iterrows() :
        reference = historical[(historical["model_id"] == item["model_id"]) & (historical["view"] == item["view"])]
        if (reference.empty) : continue
        baseline = reference.iloc[0]
        rows.append({"method_id" : item["method_id"], "model_id" : item["model_id"], "view" : item["view"], "video_recall_at_1" : float(item["video_recall_at_1"]), "baseline_v1_video_recall_at_1" : float(baseline["video_recall_at_1"]), "video_recall_at_1_delta_vs_baseline_v1" : float(item["video_recall_at_1"] - baseline["video_recall_at_1"]), "video_mrr" : float(item["video_mrr"]), "baseline_v1_video_mrr" : float(baseline["video_mrr"]), "video_mrr_delta_vs_baseline_v1" : float(item["video_mrr"] - baseline["video_mrr"]), "story_mrr" : float(item["story_mrr"]), "baseline_v1_story_mrr" : float(baseline["story_mrr"]), "story_mrr_delta_vs_baseline_v1" : float(item["story_mrr"] - baseline["story_mrr"])})
    return pd.DataFrame(rows)


def _dense_search_once(query_embedding : np.ndarray, runtime_data : dict[str, Any], invalid_document_score : float) -> tuple[float, float] :
    started = time.perf_counter()
    valid_scores = np.asarray(query_embedding @ runtime_data["document_embeddings"].T, dtype = np.float32)
    similarity_ms = (time.perf_counter() - started) * 1000.0

    started = time.perf_counter()
    scores = np.full(runtime_data["physical_window_count"], np.float32(invalid_document_score), dtype = np.float32)
    scores[runtime_data["eligible_indices"]] = valid_scores
    window_order = np.argsort(-scores, kind = "stable")
    video_scores = np.full(runtime_data["video_count"], np.float32(invalid_document_score), dtype = np.float32)
    np.maximum.at(video_scores, runtime_data["video_indices"], scores)
    video_order = np.argsort(-video_scores, kind = "stable")
    _ = int(window_order[0]), int(video_order[0])
    rank_aggregate_ms = (time.perf_counter() - started) * 1000.0
    return similarity_ms, rank_aggregate_ms


def _measure_dense_latency(backend_id : str, backend : SentenceTransformerDenseBackend, query_ids : list[str], query_texts : list[str], channel_runtime_data : dict[str, dict[str, Any]], runtime_config : dict[str, Any], invalid_document_score : float) -> pd.DataFrame :
    if (not runtime_config.get("measure_online_latency", True)) : return pd.DataFrame()
    warmup_count = min(int(runtime_config.get("warmup_query_count", 1)), len(query_ids))
    batch_size   = int(runtime_config.get("single_query_batch_size", 1))
    log_detail(f"{backend_id}: measuring warm single-query latency over {len(query_ids)} queries")

    for index in range(warmup_count) :
        warmup = backend.encode_queries([query_ids[index]], [query_texts[index]], batch_size = batch_size, show_progress_bar = False, emit_progress = False)
        for runtime_data in channel_runtime_data.values() : _dense_search_once(warmup.embeddings[0], runtime_data, invalid_document_score)

    rows = []
    for query_id, query_text in zip(query_ids, query_texts) :
        started = time.perf_counter()
        encoded = backend.encode_queries([query_id], [query_text], batch_size = batch_size, show_progress_bar = False, emit_progress = False)
        query_encode_ms = (time.perf_counter() - started) * 1000.0
        query_embedding = encoded.embeddings[0]
        for channel_id, runtime_data in channel_runtime_data.items() :
            similarity_ms, rank_aggregate_ms = _dense_search_once(query_embedding, runtime_data, invalid_document_score)
            retrieval_ms = similarity_ms + rank_aggregate_ms
            rows.append({"backend_id" : backend_id, "channel_id" : channel_id, "query_id" : query_id, "query_encode_ms" : query_encode_ms, "similarity_ms" : similarity_ms, "rank_aggregate_ms" : rank_aggregate_ms, "retrieval_ms" : retrieval_ms, "end_to_end_ms" : query_encode_ms + retrieval_ms})

    return pd.DataFrame(rows)


def _latency_summary(query_latency : pd.DataFrame, percentiles : list[int]) -> pd.DataFrame :
    if (query_latency.empty) : return pd.DataFrame()
    metrics = ["query_encode_ms", "similarity_ms", "rank_aggregate_ms", "retrieval_ms", "end_to_end_ms"]
    rows = []
    for (backend_id, channel_id), group in query_latency.groupby(["backend_id", "channel_id"], sort = True) :
        row = {"backend_id" : backend_id, "channel_id" : channel_id, "query_count" : len(group)}
        for metric in metrics :
            values = group[metric].to_numpy(dtype = float)
            row[f"{metric.removesuffix('_ms')}_mean_ms"] = float(values.mean())
            for percentile in percentiles : row[f"{metric.removesuffix('_ms')}_p{int(percentile)}_ms"] = float(np.percentile(values, percentile))
        row["warm_qps"] = float(1000.0 / row["end_to_end_mean_ms"]) if row["end_to_end_mean_ms"] > 0 else None
        rows.append(row)
    return pd.DataFrame(rows)


def run_stage02(config : dict[str, Any], query_frames : dict[str, pd.DataFrame], stage_frames : dict[tuple[str, str], pd.DataFrame], cache_root : Path, fixture : dict[str, Any]) -> tuple[list[ScoreBundle], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]] :
    stage_config = config["stage02_dense"]
    query_set    = stage_config["query_set"]
    corpus_id    = stage_config["corpus"]
    query_frame  = query_frames[query_set]
    query_ids    = query_frame["query_id"].astype(str).tolist()
    query_texts  = query_frame["query_text"].astype(str).tolist()
    top_k_values = [int(value) for value in stage_config["diagnostics"]["top_k_windows"]]
    runtime_config = stage_config.get("runtime", {})
    bundles, result_frames, evidence_frames, diagnostic_frames, latency_frames = [], [], [], [], []
    backend_records, embedding_cache_records = [], []

    for backend_id, backend_config in stage_config["backends"].items() :
        specification = DenseBackendSpec.from_config(backend_id, backend_config)
        backend       = SentenceTransformerDenseBackend(specification, progress_callback = log_detail)
        _reset_gpu_peak_memory()
        query_identity = _query_embedding_identity(stage_config, specification, query_set, query_ids, query_texts)
        query_npz, query_json = _embedding_cache_paths(cache_root, stage_config, backend_id, "queries", query_set)
        query_bundle = _load_embedding_cache(query_npz, query_json, query_ids, query_identity) if stage_config["cache"]["reuse_embeddings"] else None
        query_cache_hit = query_bundle is not None
        if (query_cache_hit) :
            log_detail(f"{backend_id}: query cache HIT")
            embedding_cache_records.append({"backend_id" : backend_id, "kind" : "queries", "item_id" : query_set, "cache_hit" : True, "npz_path" : str(query_npz), "json_path" : str(query_json), "npz_sha256" : sha256_file(query_npz), "json_sha256" : sha256_file(query_json), "identity_hash" : _embedding_identity_hash(query_identity)})
        else :
            log_detail(f"{backend_id}: query cache MISS")
            query_bundle = backend.encode_queries(query_ids, query_texts)
            cache_record = _write_embedding_cache(query_npz, query_json, query_bundle, query_identity)
            embedding_cache_records.append({"backend_id" : backend_id, "kind" : "queries", "item_id" : query_set, "cache_hit" : False, **cache_record})

        document_cache_hits, document_cache_misses, document_runtime_s, document_count_encoded, total_eligible_documents = 0, 0, 0.0, 0, 0
        dimensions = {query_bundle.dimension}
        channel_runtime_data = {}

        for channel_id, channel_spec in config["channels"].items() :
            windows   = model_corpus_windows(stage_frames, channel_spec["model_id"], corpus_id)
            documents = channel_documents(windows, channel_spec)
            eligible_ids   = [documents.window_ids[index] for index in documents.eligible_indices]
            eligible_texts = [documents.texts[index] for index in documents.eligible_indices]
            total_eligible_documents += len(eligible_ids)
            document_identity = _document_embedding_identity(stage_config, specification, channel_id, documents)
            document_npz, document_json = _embedding_cache_paths(cache_root, stage_config, backend_id, "documents", channel_id)
            document_bundle = _load_embedding_cache(document_npz, document_json, eligible_ids, document_identity) if stage_config["cache"]["reuse_embeddings"] else None
            document_cache_hit = document_bundle is not None
            if (document_cache_hit) :
                document_cache_hits += 1
                log_detail(f"{backend_id}/{channel_id}: document cache HIT ({len(eligible_ids)} eligible)")
                embedding_cache_records.append({"backend_id" : backend_id, "kind" : "documents", "item_id" : channel_id, "cache_hit" : True, "npz_path" : str(document_npz), "json_path" : str(document_json), "npz_sha256" : sha256_file(document_npz), "json_sha256" : sha256_file(document_json), "identity_hash" : _embedding_identity_hash(document_identity)})
            else :
                document_cache_misses += 1
                log_detail(f"{backend_id}/{channel_id}: document cache MISS ({len(eligible_ids)} eligible)")
                document_bundle = backend.encode_documents(eligible_ids, eligible_texts)
                document_runtime_s += float(document_bundle.metadata.get("runtime_s", 0.0))
                document_count_encoded += len(eligible_ids)
                cache_record = _write_embedding_cache(document_npz, document_json, document_bundle, document_identity)
                embedding_cache_records.append({"backend_id" : backend_id, "kind" : "documents", "item_id" : channel_id, "cache_hit" : False, **cache_record})

            dimensions.add(document_bundle.dimension)
            if (len(dimensions) != 1) : raise ValueError(f"{backend_id}: inconsistent embedding dimensions across query/document caches")
            bundle = score_dense_embeddings(query_ids = query_ids, documents = documents, query_embeddings = query_bundle.embeddings, document_embeddings = document_bundle.embeddings, method_id = backend_id, invalid_document_score = float(stage_config["score"]["invalid_document_score"]), metadata = {"query_set" : query_set, "corpus_id" : corpus_id, "channel_id" : channel_id, "backend" : specification.identity(), "embedding_dimension" : query_bundle.dimension, "query_cache_hit" : query_cache_hit, "document_cache_hit" : document_cache_hit})
            bundles.append(bundle)
            results, evidence = evaluate_bundle(bundle, query_frame, windows, documents, query_set, corpus_id, config)
            result_frames.append(results)
            evidence_frames.append(evidence)
            diagnostic_frames.append(_dense_query_diagnostics(bundle, query_frame, windows, documents, channel_id, top_k_values))

            unique_videos = sorted(windows["video_id"].astype(str).unique().tolist())
            video_lookup  = {video_id : index for index, video_id in enumerate(unique_videos)}
            channel_runtime_data[channel_id] = {"document_embeddings" : document_bundle.embeddings, "eligible_indices" : np.asarray(documents.eligible_indices, dtype = np.int64), "physical_window_count" : len(documents.window_ids), "video_indices" : np.asarray([video_lookup[video_id] for video_id in windows["video_id"].astype(str).tolist()], dtype = np.int64), "video_count" : len(unique_videos)}

        latency_frame = _measure_dense_latency(backend_id, backend, query_ids, query_texts, channel_runtime_data, runtime_config, float(stage_config["score"]["invalid_document_score"]))
        if (not latency_frame.empty) : latency_frames.append(latency_frame)

        backend_metadata = backend.metadata()
        backend_records.append({"backend_id" : backend_id, "model_name" : specification.model_name, "configured_revision" : specification.revision, "resolved_revision" : backend_metadata.get("resolved_revision"), "device" : backend_metadata.get("device"), "embedding_dimension" : next(iter(dimensions)), "normalization_policy" : specification.normalization_policy, "query_cache_hit" : query_cache_hit, "document_cache_hit_count" : document_cache_hits, "document_cache_miss_count" : document_cache_misses, "batched_query_encoding_runtime_s_this_run" : float(query_bundle.metadata.get("runtime_s", 0.0)) if not query_cache_hit else 0.0, "batched_query_items_per_second" : float(query_bundle.metadata.get("items_per_second")) if not query_cache_hit and query_bundle.metadata.get("items_per_second") is not None else None, "document_encoding_runtime_s_this_run" : document_runtime_s, "documents_encoded_this_run" : document_count_encoded, "documents_per_second_this_run" : float(document_count_encoded / document_runtime_s) if document_runtime_s > 0 else None, "total_eligible_documents" : total_eligible_documents, "model_load_runtime_s" : float(backend_metadata.get("load_runtime_s", 0.0)), "last_effective_batch_size" : backend_metadata.get("last_batch_size"), "online_latency_measured" : not latency_frame.empty, **_gpu_memory_snapshot()})
        backend.release()

    query_results = pd.concat(result_frames, ignore_index = True)
    top_evidence  = pd.concat(evidence_frames, ignore_index = True)
    diagnostics   = pd.concat(diagnostic_frames, ignore_index = True)
    query_latency = pd.concat(latency_frames, ignore_index = True) if latency_frames else pd.DataFrame()
    latency_summary = _latency_summary(query_latency, [int(value) for value in runtime_config.get("latency_percentiles", [50, 90])])
    metrics       = summarize_retrieval_metrics(query_results)
    comparisons   = compare_methods(query_results, reference_method = stage_config["dense_reference"])
    selection     = stage_config["selection"]
    method_summary = _dense_method_summary(query_results, stage_config["dense_reference"], int(selection["bootstrap_samples"]), float(selection["bootstrap_confidence"]), int(selection["bootstrap_seed"]))
    asr_gap       = _asr_gap_summary(metrics)
    system_reference = _system_reference_summary(metrics, fixture)
    diagnostics = diagnostics.merge(query_results[["query_set", "method_id", "model_id", "view", "query_id", "first_relevant_rank", "video_rank", "correct_video_score", "best_wrong_video_score", "video_score_margin", "story_score_margin"]], on = ["query_set", "method_id", "model_id", "view", "query_id"], how = "left")
    return bundles, query_results, top_evidence, comparisons, diagnostics, method_summary, asr_gap, system_reference, query_latency, latency_summary, backend_records, embedding_cache_records


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
        "retrieval_backends"      : source_record(SRC_ROOT / "retrieval_backends.py"),
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
        "stage02_dense"    : config["stage02_dense"],
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


def _stage_prerequisite(config : dict[str, Any], artifacts : Path, stage : str, contract_keys : list[str]) -> dict[str, Any] :
    stage_root   = resolve_artifact_path(artifacts, config["paths"]["reports_root"]) / stage
    summary_path = stage_root / "stage_summary.json"
    manifest_path = stage_root / "retrieval_manifest.json"
    if (not summary_path.exists()) : raise RuntimeError(f"{stage} prerequisite is missing: {summary_path}")
    if (not manifest_path.exists()) : raise RuntimeError(f"{stage} manifest is missing: {manifest_path}")
    summary  = load_json(summary_path)
    manifest = load_json(manifest_path)
    if (not summary.get("passed", False)) : raise RuntimeError(f"{stage} did not pass")
    current_contract  = {key : config[key] for key in contract_keys}
    previous_contract = {key : manifest.get(key) for key in contract_keys}
    if (canonical_json_hash(current_contract) != canonical_json_hash(previous_contract)) : raise RuntimeError(f"{stage} contract changed. Re-run {stage} before continuing.")
    return summary


def stage00_prerequisite(config : dict[str, Any], artifacts : Path) -> dict[str, Any] :
    return _stage_prerequisite(config, artifacts, "stage00_baseline", ["stage00_baseline", "evaluation_policy", "regression"])


def stage01_prerequisite(config : dict[str, Any], artifacts : Path) -> dict[str, Any] :
    summary = _stage_prerequisite(config, artifacts, "stage01_lexical", ["stage01_lexical", "evaluation_policy"])
    decision = config["selection"].get("stage01_decision", {})
    if (decision.get("lexical_default") != config["stage02_dense"]["lexical_reference"]) : raise RuntimeError("Stage 1 lexical decision does not match the Stage 2 lexical reference")
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
    dense_diagnostics = pd.DataFrame()
    asr_gap_summary = pd.DataFrame()
    asr_gap_queries = pd.DataFrame()
    system_reference = pd.DataFrame()
    backend_summary = pd.DataFrame()
    query_latency = pd.DataFrame()
    latency_summary = pd.DataFrame()
    embedding_cache_records = []
    stage02_environment = None

    if (args.stage == "stage00_baseline") :
        bundles, query_results, top_evidence, regression = run_stage00(config, query_frames, stage_frames, fixture)
    elif (args.stage == "stage01_lexical") :
        stage00_prerequisite(config, artifacts)
        bundles, query_results, top_evidence, comparisons, method_summary = run_stage01(config, query_frames, stage_frames)
    else :
        stage00_prerequisite(config, artifacts)
        stage01_prerequisite(config, artifacts)
        stage02_environment = validate_stage02_environment(config)
        if (not stage02_environment["passed"]) : raise RuntimeError(f"Stage 2 environment validation failed: {stage02_environment['errors']}")
        bundles, query_results, top_evidence, comparisons, dense_diagnostics, method_summary, asr_gap_summary, system_reference, query_latency, latency_summary, backend_records, embedding_cache_records = run_stage02(config, query_frames, stage_frames, cache_root, fixture)
        backend_summary = pd.DataFrame(backend_records)
        asr_gap_queries = _asr_gap_queries(query_results)

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

    if (args.stage == "stage00_baseline" and not regression["passed"]) : log_detail("Stage 0 regression: FAIL")
    elif (args.stage == "stage00_baseline") : log_detail("Stage 0 aggregate + strict per-query regression: PASS")
    elif (args.stage == "stage01_lexical") : log_detail("Stage 1 paired comparison tables built")
    else : log_detail("Stage 2 dense comparisons, ASR gaps, latency, and diagnostics built")
    log_done(start)

    start = log_stage(6, total_stages, "Writing reports and window-score cache...")
    cache_records = save_score_cache(bundles, query_results, cache_root, args.stage)
    if (embedding_cache_records) : cache_records["embeddings"] = embedding_cache_records

    write_csv(reports_root / "retrieval_metrics.csv", metrics)
    write_csv(reports_root / "query_results.csv", query_results)

    if (args.stage == "stage00_baseline") :
        write_json(reports_root / "regression.json", regression)
    elif (args.stage == "stage01_lexical") :
        write_csv(reports_root / "query_comparison.csv", comparisons)
        diagnostics_columns = ["query_set", "method_id", "model_id", "view", "query_id", "eligible_window_count", "positive_window_count", "positive_window_fraction", "positive_video_count", "correct_video_positive_windows", "query_coverage", "zero_evidence", "story_score_margin", "video_score_margin"]
        write_csv(reports_root / "lexical_diagnostics.csv", query_results[diagnostics_columns])
        write_csv(reports_root / "method_summary.csv", method_summary)
        write_csv(reports_root / "top_evidence.csv", top_evidence)
    else :
        write_csv(reports_root / "query_comparison.csv", comparisons)
        write_csv(reports_root / "dense_diagnostics.csv", dense_diagnostics)
        write_csv(reports_root / "backend_summary.csv", backend_summary)
        write_csv(reports_root / "method_summary.csv", method_summary)
        write_csv(reports_root / "asr_gap_summary.csv", asr_gap_summary)
        write_csv(reports_root / "asr_gap_queries.csv", asr_gap_queries)
        write_csv(reports_root / "system_reference_comparison.csv", system_reference)
        write_csv(reports_root / "query_latency.csv", query_latency)
        write_csv(reports_root / "latency_summary.csv", latency_summary)
        write_csv(reports_root / "top_evidence.csv", top_evidence)

    log_done(start)

    if (stage02_environment is not None) :
        validation_payload = load_json(reports_root / "validation_summary.json")
        validation_payload["stage02_environment"] = stage02_environment
        write_json(reports_root / "validation_summary.json", validation_payload)

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

    passed = bool(source_validation["passed"] and query_validation["passed"] and model_validation["passed"] and channel_validation["passed"] and fixture_status["passed"] and not holdout_overlap and (regression["passed"] if regression is not None else True) and (stage02_environment["passed"] if stage02_environment is not None else True))

    summary = {"schema_version" : "1.0", "retrieval_id" : config["retrieval_id"], "stage" : args.stage, "generated_at_utc" : utc_now(), "passed" : passed, "retrieval_config_sha256" : sha256_file(CONFIG_PATH), "holdout_query_overlap" : holdout_overlap, "regression" : compact_stage00_regression(regression), "strict_fixture" : fixture_status, "reports_root" : str(reports_root), "cache" : cache_records, "provisional_default" : config["selection"]["provisional_default"]}

    if (args.stage == "stage01_lexical") :
        summary["method_summary"] = method_summary.to_dict(orient = "records")
        summary["decision_required"] = False
        summary["decision_note"] = "Stage 1 lexical decision is frozen in config: L2_bm25_preserving; no alternative retained."
    elif (args.stage == "stage02_dense") :
        summary["method_summary"] = method_summary.to_dict(orient = "records")
        summary["backend_summary"] = backend_summary.to_dict(orient = "records")
        summary["latency_summary"] = latency_summary.to_dict(orient = "records")
        summary["stage02_environment"] = stage02_environment
        summary["provisional_default"] = config["stage02_dense"]["selection"]["provisional_default"]
        summary["retained_alternative"] = config["stage02_dense"]["selection"]["retained_alternative"]
        summary["decision_required"] = True
        reference = config["stage02_dense"]["dense_reference"]
        candidates = [backend_id for backend_id in config["stage02_dense"]["backends"] if backend_id != reference]
        summary["decision_note"] = f"Review {', '.join(candidates)} against {reference} using paired query evidence, ASR gaps, and warm online latency. No dense backend is selected automatically."

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
