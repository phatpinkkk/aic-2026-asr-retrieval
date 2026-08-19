from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Sequence
import csv
import json
import math
import os
import re
import sys
from time import perf_counter
import logging

import numpy as np
from tqdm.auto import tqdm


PRODUCTION_ROOT = Path(__file__).resolve().parents[1]
if (str(PRODUCTION_ROOT) not in sys.path) :
    sys.path.insert(0, str(PRODUCTION_ROOT))

from asr_retrieval.audio import build_physical_windows
from asr_retrieval.config import BM25Config, CorpusBuildConfig, E5Config, OfflineASRConfig, ParakeetConfig, parakeet_identity, window_policy_identity
from asr_retrieval.postprocess import POSTPROCESS_VERSION
from asr_retrieval.transcriber import TRANSCRIPT_ARTIFACT_SCHEMA_VERSION, ParakeetTranscriber, transcribe_video

VIDEO_ID_RE = re.compile(r"^[KL]\d{2}_V\d{3}$")
INVENTORY_SCHEMA_VERSION = "1.0"
PHASE1_ACCEPTANCE_SCHEMA_VERSION = "1.0"
TIMING_FIELDS = [
    "total_ms", "bm25_ms", "e5_encode_ms", "dense_search_ms", "first_stage_fusion_ms",
    "video_aggregation_ms", "candidate_selection_ms", "bge_ms", "candidate_fusion_ms", "result_build_ms",
]


# Shared helpers


def corpus_paths(artifact_root : Path) -> dict[str, Path] :
    """Return the small set of directories used by the full-corpus workflow."""
    artifact_root  = Path(artifact_root)
    phase1         = artifact_root / "phase1"
    phase2         = artifact_root / "phase2"
    control        = artifact_root / "control"
    reports        = artifact_root / "reports"
    model_cache    = artifact_root / "model_cache"

    return {
        "artifact_root" : artifact_root,
        "phase1"        : phase1,
        "transcripts"   : phase1 / "transcripts",
        "releases"      : phase2 / "releases",
        "control"       : control,
        "reports"       : reports,
        "model_cache"   : model_cache,
        "parakeet_cache" : model_cache / "parakeet",
        "e5_cache"      : model_cache / "e5",
        "inventory"     : control / "source_inventory.jsonl",
        "inventory_summary" : control / "source_inventory_summary.json",
        "inventory_verification" : control / "source_inventory_verification.json",
        "phase1_acceptance" : control / "phase1_acceptance.json",
    }


def ensure_directories(artifact_root : Path) -> dict[str, Path] :
    paths = corpus_paths(artifact_root)
    for key in ["phase1", "transcripts", "releases", "control", "reports", "model_cache", "parakeet_cache", "e5_cache"] :
        paths[key].mkdir(parents = True, exist_ok = True)
    return paths


def read_json(path : Path) -> Any :
    with Path(path).open("r", encoding = "utf-8") as file :
        return json.load(file)


def write_json(path : Path, value : Any) -> None :
    path = Path(path)
    path.parent.mkdir(parents = True, exist_ok = True)
    with path.open("w", encoding = "utf-8") as file :
        json.dump(value, file, ensure_ascii = False, indent = 2, allow_nan = False, default = str)


def read_jsonl(path : Path) -> list[dict[str, Any]] :
    rows = []
    with Path(path).open("r", encoding = "utf-8") as file :
        for line_number, line in enumerate(file, start = 1) :
            text = line.strip()
            if (not text) :
                continue
            value = json.loads(text)
            if (not isinstance(value, dict)) :
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            rows.append(value)
    return rows


def write_jsonl(path : Path, rows : Sequence[dict[str, Any]]) -> None :
    path = Path(path)
    path.parent.mkdir(parents = True, exist_ok = True)
    with path.open("w", encoding = "utf-8") as file :
        for row in rows :
            file.write(json.dumps(row, ensure_ascii = False, allow_nan = False, default = str) + "\n")


def write_csv(path : Path, rows : Sequence[dict[str, Any]]) -> None :
    path = Path(path)
    path.parent.mkdir(parents = True, exist_ok = True)

    fieldnames = []
    for row in rows :
        for key in row :
            if (key not in fieldnames) :
                fieldnames.append(key)

    with path.open("w", encoding = "utf-8", newline = "") as file :
        writer = csv.DictWriter(file, fieldnames = fieldnames)
        writer.writeheader()
        for row in rows :
            output = {}
            for key in fieldnames :
                value = row.get(key)
                if (isinstance(value, (list, tuple, dict))) :
                    value = json.dumps(value, ensure_ascii = False, allow_nan = False)
                output[key] = value
            writer.writerow(output)


def utc_now() -> str :
    return datetime.now(timezone.utc).isoformat()


def canonical_hash(value : Any) -> str :
    text = json.dumps(value, ensure_ascii = False, sort_keys = True, separators = (",", ":"), allow_nan = False)
    return sha256(text.encode("utf-8")).hexdigest()


def directory_size_bytes(path : Path) -> int :
    path = Path(path)
    if (not path.exists()) :
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def make_offline_config(artifact_root : Path) -> OfflineASRConfig :
    paths = corpus_paths(artifact_root)
    return OfflineASRConfig(parakeet = ParakeetConfig(model_cache_dir = paths["parakeet_cache"]))


def make_corpus_build_config(artifact_root : Path) -> CorpusBuildConfig :
    paths   = corpus_paths(artifact_root)
    offline = make_offline_config(artifact_root)
    return CorpusBuildConfig(offline = offline, bm25 = BM25Config(), e5 = E5Config(model_cache_dir = paths["e5_cache"]))

# Source inventory

def load_fps_map(path : Path) -> dict[str, float] :
    payload = read_json(path)
    if (not isinstance(payload, dict) or not payload) :
        raise ValueError("fps_map.json must be a nonempty JSON object")

    fps_map = {}
    for raw_video_id, raw_fps in payload.items() :
        video_id = str(raw_video_id).strip()
        fps      = float(raw_fps)

        if (not VIDEO_ID_RE.fullmatch(video_id)) :
            raise ValueError(f"Malformed video ID in fps_map.json: {video_id!r}")
        if (not math.isfinite(fps) or fps <= 0.0) :
            raise ValueError(f"Invalid FPS for {video_id}: {raw_fps!r}")

        fps_map[video_id] = fps

    return fps_map


def inventory_identity(rows : Sequence[dict[str, Any]]) -> str :
    stable_rows = [
        [row["video_id"], row["relative_source_path"], int(row["size_bytes"]), int(row["mtime_ns"]), float(row["fps"])]
        for row in sorted(rows, key = lambda item : item["video_id"])
    ]
    return canonical_hash(stable_rows)


def inspect_source_inventory(video_root : Path, fps_map_path : Path) -> tuple[list[dict[str, Any]], dict[str, Any]] :
    video_root   = Path(video_root)
    fps_map_path = Path(fps_map_path)

    if (not video_root.is_dir()) :
        raise FileNotFoundError(f"VIDEO_ROOT is not a directory: {video_root}")
    if (not fps_map_path.is_file()) :
        raise FileNotFoundError(f"FPS_MAP_PATH is not a file: {fps_map_path}")

    fps_map      = load_fps_map(fps_map_path)
    expected_ids = set(fps_map)
    source_paths = sorted(
        [path for path in video_root.rglob("*") if path.is_file() and path.suffix.lower() == ".mp4"],
        key = lambda path : str(path.relative_to(video_root)).lower(),
    )

    id_to_paths : dict[str, list[Path]] = defaultdict(list)
    rows = []

    for source_path in source_paths :
        video_id = source_path.stem
        stat     = source_path.stat()
        id_to_paths[video_id].append(source_path)
        rows.append({
            "video_id"             : video_id,
            "source_path"          : str(source_path),
            "relative_source_path" : str(source_path.relative_to(video_root)),
            "size_bytes"           : int(stat.st_size),
            "mtime_ns"             : int(stat.st_mtime_ns),
            "expected_from_fps_map" : video_id in expected_ids,
            "fps"                  : float(fps_map[video_id]) if video_id in fps_map else None,
        })

    actual_ids    = set(id_to_paths)
    duplicate_ids = sorted(video_id for video_id, paths in id_to_paths.items() if len(paths) > 1)
    malformed_ids = sorted(video_id for video_id in actual_ids if not VIDEO_ID_RE.fullmatch(video_id))
    missing_ids   = sorted(expected_ids - actual_ids)
    extra_ids     = sorted(actual_ids - expected_ids)
    zero_byte_ids = sorted(row["video_id"] for row in rows if row["size_bytes"] == 0)

    passed = not duplicate_ids and not malformed_ids and not missing_ids and not extra_ids and not zero_byte_ids
    frozen_rows = sorted(rows, key = lambda row : row["video_id"]) if passed else rows

    summary = {
        "schema_version"         : INVENTORY_SCHEMA_VERSION,
        "created_at_utc"         : utc_now(),
        "video_root"             : str(video_root),
        "fps_map_path"           : str(fps_map_path),
        "expected_count"         : len(expected_ids),
        "actual_file_count"      : len(source_paths),
        "actual_unique_id_count" : len(actual_ids),
        "matched_count"          : len(expected_ids & actual_ids),
        "missing_ids"            : missing_ids,
        "extra_ids"              : extra_ids,
        "duplicate_ids"          : duplicate_ids,
        "malformed_ids"          : malformed_ids,
        "zero_byte_ids"          : zero_byte_ids,
        "total_size_bytes"       : sum(int(row["size_bytes"]) for row in rows),
        "inventory_identity"     : inventory_identity(frozen_rows) if passed else None,
        "passed"                 : passed,
    }
    return frozen_rows, summary


def load_inventory(artifact_root : Path) -> tuple[list[dict[str, Any]], dict[str, Any]] :
    paths = corpus_paths(artifact_root)
    if (not paths["inventory"].is_file() or not paths["inventory_summary"].is_file()) :
        raise FileNotFoundError("Frozen inventory does not exist. Run prepare_inventory() first.")

    rows    = read_jsonl(paths["inventory"])
    summary = read_json(paths["inventory_summary"])

    if (not summary.get("passed", False)) :
        raise RuntimeError("Frozen inventory summary did not pass")
    if (inventory_identity(rows) != summary.get("inventory_identity")) :
        raise RuntimeError("source_inventory.jsonl no longer matches its frozen identity")

    return rows, summary


def verify_inventory(artifact_root : Path) -> dict[str, Any] :
    paths                  = corpus_paths(artifact_root)
    frozen_rows, frozen    = load_inventory(artifact_root)
    current_rows, current  = inspect_source_inventory(Path(frozen["video_root"]), Path(frozen["fps_map_path"]))

    passed = bool(current["passed"]) and current.get("inventory_identity") == frozen.get("inventory_identity")
    verification = {
        "schema_version"             : INVENTORY_SCHEMA_VERSION,
        "verified_at_utc"            : utc_now(),
        "frozen_inventory_identity"  : frozen["inventory_identity"],
        "current_inventory_identity" : current.get("inventory_identity"),
        "frozen_count"               : len(frozen_rows),
        "current_count"              : len(current_rows),
        "current_gate"               : current,
        "passed"                     : passed,
    }
    write_json(paths["inventory_verification"], verification)

    if (not passed) :
        raise RuntimeError("Current MP4/FPS inventory no longer matches the frozen inventory.")

    return verification


def prepare_inventory(video_root : Path, fps_map_path : Path, artifact_root : Path, refresh : bool = False) -> tuple[list[dict[str, Any]], dict[str, Any]] :
    """Create the frozen source inventory once, then verify it on later runs."""
    paths = ensure_directories(artifact_root)

    if (paths["inventory"].is_file() and paths["inventory_summary"].is_file() and not refresh) :
        verify_inventory(artifact_root)
        return load_inventory(artifact_root)

    rows, summary = inspect_source_inventory(video_root, fps_map_path)
    if (not summary["passed"]) :
        write_json(paths["control"] / "source_inventory_candidate_summary.json", summary)
        raise RuntimeError("Source inventory gate failed. Review source_inventory_candidate_summary.json.")

    write_jsonl(paths["inventory"], rows)
    write_json(paths["inventory_summary"], summary)
    return rows, summary


# Phase 1 transcription


def transcript_status(transcript_path : Path, inventory_row : dict[str, Any], offline_config : OfflineASRConfig) -> dict[str, Any] :
    """Read and validate one saved transcript without loading Parakeet or hashing the source MP4."""
    video_id = str(inventory_row["video_id"])
    base = {
        "video_id"             : video_id,
        "artifact_path"        : str(transcript_path),
        "artifact_exists"      : transcript_path.is_file(),
        "complete"             : False,
        "window_count"         : 0,
        "ok_nonempty_windows"  : 0,
        "ok_empty_windows"     : 0,
        "failed_windows"       : 0,
        "eligible_windows"     : 0,
        "classification"       : "missing",
        "invalid_reason"       : None,
    }

    if (not transcript_path.is_file()) :
        return base

    try :
        payload = read_json(transcript_path)
        if (not isinstance(payload, dict)) :
            raise ValueError("artifact root must be a JSON object")
        if (payload.get("schema_version") != TRANSCRIPT_ARTIFACT_SCHEMA_VERSION) :
            raise ValueError("transcript schema does not match production")
        if (payload.get("window_policy_identity") != window_policy_identity(offline_config.audio)) :
            raise ValueError("window policy does not match production")
        if (payload.get("parakeet_identity") != parakeet_identity(offline_config.parakeet)) :
            raise ValueError("Parakeet identity does not match production")
        if (payload.get("postprocess_version") != POSTPROCESS_VERSION) :
            raise ValueError("postprocess version does not match production")

        video = payload.get("video")
        if (not isinstance(video, dict) or str(video.get("video_id", "")) != video_id) :
            raise ValueError("artifact video_id does not match the frozen inventory")
        if (int(video.get("source_size_bytes", -1)) != int(inventory_row["size_bytes"])) :
            raise ValueError("artifact source size does not match the frozen inventory")
        if (not str(video.get("source_sha256", ""))) :
            raise ValueError("artifact source SHA-256 is missing")

        canonical_audio = payload.get("canonical_audio")
        if (not isinstance(canonical_audio, dict)) :
            raise ValueError("canonical audio metadata is missing")
        if (str(canonical_audio.get("video_id", "")) != video_id) :
            raise ValueError("canonical audio video_id does not match the artifact")
        if (int(canonical_audio.get("sample_rate", -1)) != offline_config.audio.sample_rate) :
            raise ValueError("canonical audio sample rate does not match production")
        if (int(canonical_audio.get("channels", -1)) != offline_config.audio.channels) :
            raise ValueError("canonical audio channel count does not match production")
        if (int(canonical_audio.get("sample_width_bytes", -1)) != offline_config.audio.sample_width_bytes) :
            raise ValueError("canonical audio sample width does not match production")
        if (str(canonical_audio.get("source_sha256", "")) != str(video.get("source_sha256", ""))) :
            raise ValueError("canonical audio source identity does not match the video")

        sample_count          = int(canonical_audio.get("sample_count", 0))
        canonical_wav_sha256  = str(canonical_audio.get("wav_sha256", ""))
        if (sample_count <= 0) :
            raise ValueError("canonical audio sample count must be positive")
        if (not canonical_wav_sha256) :
            raise ValueError("canonical audio WAV hash is missing")

        windows = payload.get("windows")
        if (not isinstance(windows, list) or not windows) :
            raise ValueError("artifact has no transcript windows")

        window_ids = [str(window.get("window_id", "")) for window in windows if isinstance(window, dict)]
        if (len(window_ids) != len(windows) or any(not value for value in window_ids) or len(window_ids) != len(set(window_ids))) :
            raise ValueError("artifact contains malformed or duplicate window IDs")

        indices = [int(window.get("window_index", -1)) for window in windows]
        if (indices != list(range(len(windows)))) :
            raise ValueError("artifact windows are not in deterministic index order")

        expected_windows = build_physical_windows(video_id, sample_count, offline_config.audio)
        if (len(windows) > len(expected_windows)) :
            raise ValueError("artifact contains more windows than the production window policy")

        for observed, expected in zip(windows, expected_windows) :
            expected_fields = {
                "window_id"        : expected.window_id,
                "video_id"         : expected.video_id,
                "window_index"     : expected.window_index,
                "sample_start"     : expected.sample_start,
                "sample_end"       : expected.sample_end,
                "duration_samples" : expected.duration_samples,
                "sample_rate"      : expected.sample_rate,
            }
            if (any(observed.get(key) != value for key, value in expected_fields.items())) :
                raise ValueError(f"window {expected.window_id} does not match the production window policy")

        complete = bool(payload.get("complete", False))
        if (complete and len(windows) != len(expected_windows)) :
            raise ValueError("artifact is marked complete but does not contain every expected window")
        if (not complete and len(windows) >= len(expected_windows)) :
            raise ValueError("artifact completeness flag does not match its window count")

        failed = 0
        empty = 0
        nonempty = 0
        eligible = 0

        for window in windows :
            if (str(window.get("video_id", "")) != video_id) :
                raise ValueError("window video_id does not match artifact video_id")
            if (str(window.get("model_name", "")) != offline_config.parakeet.model_name) :
                raise ValueError("window model_name does not match production")
            if (str(window.get("model_revision", "")) != offline_config.parakeet.revision) :
                raise ValueError("window model_revision does not match production")
            if (str(window.get("postprocess_version", "")) != POSTPROCESS_VERSION) :
                raise ValueError("window postprocess version does not match production")
            if (not str(window.get("window_pcm_sha256", ""))) :
                raise ValueError("window PCM hash is missing")
            if (str(window.get("canonical_wav_sha256", "")) != canonical_wav_sha256) :
                raise ValueError("window canonical WAV hash does not match the artifact")

            status         = str(window.get("status", ""))
            raw_text       = str(window.get("raw_text", ""))
            retrieval_text = str(window.get("retrieval_text", ""))
            rejections     = window.get("rejection_reasons", [])

            if (status not in {"ok", "failed"}) :
                raise ValueError(f"unexpected window status: {status!r}")

            expected_eligible = status == "ok" and bool(retrieval_text.strip()) and not rejections
            if (bool(window.get("eligible", False)) != expected_eligible) :
                raise ValueError("window eligibility does not match the production rule")

            failed   += int(status == "failed")
            empty    += int(status == "ok" and not raw_text.strip())
            nonempty += int(status == "ok" and bool(raw_text.strip()))
            eligible += int(expected_eligible)

        if (not complete) :
            classification = "partial"
        elif (eligible == 0) :
            classification = "complete_no_searchable_windows"
        elif (failed > 0) :
            classification = "complete_with_failed_windows"
        elif (empty > 0) :
            classification = "complete_with_empty_windows"
        else :
            classification = "complete_clean"

        return {
            **base,
            "complete"            : complete,
            "window_count"        : len(windows),
            "ok_nonempty_windows" : nonempty,
            "ok_empty_windows"    : empty,
            "failed_windows"      : failed,
            "eligible_windows"    : eligible,
            "classification"      : classification,
        }
    except Exception as error :
        return {
            **base,
            "artifact_exists" : True,
            "classification"  : "invalid_artifact",
            "invalid_reason"  : f"{type(error).__name__}: {error}",
        }


def scan_phase1(artifact_root : Path) -> tuple[list[dict[str, Any]], dict[str, Any]] :
    """Fast reconnect status. This reads transcript JSON only and never loads ASR models."""
    paths                       = corpus_paths(artifact_root)
    inventory_rows, inventory  = load_inventory(artifact_root)
    offline_config              = make_offline_config(artifact_root)
    expected_ids                = {row["video_id"] for row in inventory_rows}

    rows = [
        transcript_status(paths["transcripts"] / f"{row['video_id']}.json", row, offline_config)
        for row in inventory_rows
    ]

    actual_ids = {path.stem for path in paths["transcripts"].glob("*.json")} if paths["transcripts"].exists() else set()
    extra_ids  = sorted(actual_ids - expected_ids)
    counts     = Counter(row["classification"] for row in rows)

    summary = {
        "inventory_identity"            : inventory["inventory_identity"],
        "total_expected_videos"         : len(rows),
        "complete_videos"               : sum(int(row["complete"]) for row in rows),
        "partial_videos"                : counts.get("partial", 0),
        "missing_videos"                : counts.get("missing", 0),
        "invalid_artifacts"             : counts.get("invalid_artifact", 0),
        "failed_windows"                : sum(int(row["failed_windows"]) for row in rows),
        "empty_ok_windows"              : sum(int(row["ok_empty_windows"]) for row in rows),
        "eligible_windows"              : sum(int(row["eligible_windows"]) for row in rows),
        "physical_windows"              : sum(int(row["window_count"]) for row in rows),
        "extra_transcript_artifact_ids" : extra_ids,
        "classification_counts"         : dict(sorted(counts.items())),
    }
    return rows, summary


def select_pilot_ids(inventory_rows : Sequence[dict[str, Any]], count : int = 8) -> list[str] :
    """Pick deterministic videos spread across corpus prefixes."""
    if (count <= 0) :
        raise ValueError("pilot count must be positive")

    by_prefix : dict[str, list[str]] = defaultdict(list)
    for row in inventory_rows :
        video_id = str(row["video_id"])
        by_prefix[video_id.split("_")[0]].append(video_id)

    prefixes = sorted(by_prefix)
    count    = min(count, len(prefixes))

    if (count == 1) :
        prefix_indices = [len(prefixes) // 2]
    else :
        prefix_indices = [round(index * (len(prefixes) - 1) / (count - 1)) for index in range(count)]

    selected = []
    for prefix_index in prefix_indices :
        video_ids = sorted(by_prefix[prefixes[prefix_index]])
        selected.append(video_ids[len(video_ids) // 2])
    return selected


def transcribe_corpus(
    artifact_root : Path,
    video_ids : Sequence[str] | None = None,
    retry_failed : bool = False,
    retry_empty : bool = False,
) -> dict[str, Any] :
    """Run or resume Phase 1 directly in the current Python process."""
    verify_inventory(artifact_root)
    paths                       = ensure_directories(artifact_root)
    inventory_rows, _          = load_inventory(artifact_root)
    status_rows, status_summary = scan_phase1(artifact_root)

    inventory_by_id = {row["video_id"] : row for row in inventory_rows}
    status_by_id    = {row["video_id"] : row for row in status_rows}
    selected_ids    = list(video_ids) if video_ids is not None else [row["video_id"] for row in inventory_rows]

    unknown_ids = sorted(set(selected_ids) - set(inventory_by_id))
    if (unknown_ids) :
        raise ValueError(f"Video IDs are outside the frozen inventory: {unknown_ids[ : 20]}")

    invalid_ids = [video_id for video_id in selected_ids if status_by_id[video_id]["classification"] == "invalid_artifact"]
    if (invalid_ids) :
        raise RuntimeError(f"Invalid transcript artifacts must be reviewed before transcription: {invalid_ids[ : 20]}")

    if (retry_failed) :
        work_ids = [video_id for video_id in selected_ids if status_by_id[video_id]["failed_windows"] > 0]
    elif (retry_empty) :
        work_ids = [video_id for video_id in selected_ids if status_by_id[video_id]["ok_empty_windows"] > 0]
    else :
        work_ids = [video_id for video_id in selected_ids if not status_by_id[video_id]["complete"]]

    skipped_count = len(selected_ids) - len(work_ids)
    print(f"Corpus complete : {status_summary['complete_videos']:,} / {status_summary['total_expected_videos']:,}")
    print(f"Selected        : {len(selected_ids):,}")
    print(f"Skipped         : {skipped_count:,}")
    print(f"To process      : {len(work_ids):,}")

    report = {
        "requested_video_count" : len(selected_ids),
        "skipped_video_count"   : skipped_count,
        "work_video_count"      : len(work_ids),
        "completed_video_ids"   : [],
        "failures"              : [],
    }

    if (not work_ids) :
        print("Nothing to transcribe. Parakeet was not loaded.")
        return report

    offline_config = make_offline_config(artifact_root)
    transcriber    = ParakeetTranscriber(offline_config.parakeet)
    
    nemo_logger = logging.getLogger("nemo_logger")
    noisy_nemo_messages = (
        "The following configuration keys are ignored by Lhotse dataloader",
        "You are using a non-tarred dataset and requested tokenization during data sampling",
    )
    
    nemo_logger.addFilter(
        lambda record : not any(message in record.getMessage() for message in noisy_nemo_messages)
    )
    
    logging.getLogger("nv_one_logger").setLevel(logging.ERROR)
    
    print("Loading Parakeet...")
    transcriber.load()
    print(f"Loaded: {offline_config.parakeet.model_name} @ {offline_config.parakeet.revision}")
    
    full_run = video_ids is None and not retry_failed and not retry_empty
    
    if (full_run) :
        start_index = status_summary["complete_videos"] + 1
        total_logs  = status_summary["total_expected_videos"]
    else :
        start_index = 1
        total_logs  = len(work_ids)
    
    try :
        for index, video_id in enumerate(work_ids, start = start_index) :
            source_path = Path(inventory_by_id[video_id]["source_path"])
    
            print(f"[{index:>5,}/{total_logs:,}] {video_id} | running...", end = "", flush = True)
    
            started = perf_counter()
    
            try :
                artifact = transcribe_video(
                    source_video                 = source_path,
                    workspace                    = paths["phase1"],
                    config                       = offline_config,
                    transcriber                  = transcriber,
                    retry_failed                 = retry_failed,
                    retry_empty                  = retry_empty,
                    force_rebuild                = False,
                    remove_canonical_audio_after = True,
                )
    
                elapsed_s        = perf_counter() - started
                eligible_windows = sum(int(window.eligible) for window in artifact.windows)
    
                print(
                    f"\r[{index:>5,}/{total_logs:,}] {video_id} | DONE | "
                    f"windows={len(artifact.windows):,} | eligible={eligible_windows:,} | {elapsed_s:,.1f}s"
                )
    
                report["completed_video_ids"].append(video_id)
    
            except Exception as error :
                print(
                    f"\r[{index:>5,}/{total_logs:,}] {video_id} | FAILED | "
                    f"{type(error).__name__}: {error}"
                )
    
                report["failures"].append({
                    "video_id"      : video_id,
                    "error_type"    : type(error).__name__,
                    "error_message" : str(error),
                })
    finally :
        transcriber.close()
    
    completed_this_run = len(report["completed_video_ids"])
    completed_total    = status_summary["complete_videos"] + completed_this_run
    
    print()
    print(f"Completed this run : {completed_this_run:,}")
    print(f"Corpus complete    : {completed_total:,} / {len(inventory_rows):,}")
    print(f"Source failures    : {len(report['failures']):,}")
    return report


def audit_phase1(artifact_root : Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]] :
    """Write a compact Phase 1 audit and the details of failed ASR windows."""
    verify_inventory(artifact_root)
    paths                = ensure_directories(artifact_root)
    status_rows, summary = scan_phase1(artifact_root)
    failed_window_rows   = []

    for status in status_rows :
        if (not status["artifact_exists"] or status["classification"] == "invalid_artifact") :
            continue

        payload = read_json(Path(status["artifact_path"]))
        for window in payload.get("windows", []) :
            if (window.get("status") != "failed") :
                continue

            error       = window.get("error") if isinstance(window.get("error"), dict) else {}
            sample_rate = int(window.get("sample_rate", 1) or 1)
            failed_window_rows.append({
                "video_id"      : status["video_id"],
                "window_id"     : str(window.get("window_id", "")),
                "window_index"  : int(window.get("window_index", -1)),
                "start_s"       : int(window.get("sample_start", 0)) / sample_rate,
                "end_s"         : int(window.get("sample_end", 0)) / sample_rate,
                "error_type"    : str(error.get("error_type", "")),
                "error_message" : str(error.get("message", "")),
            })

    summary = {
        **summary,
        "created_at_utc" : utc_now(),
        "passed" : (
            summary["complete_videos"] == summary["total_expected_videos"]
            and summary["missing_videos"] == 0
            and summary["partial_videos"] == 0
            and summary["invalid_artifacts"] == 0
            and not summary["extra_transcript_artifact_ids"]
            and summary["failed_windows"] == 0
            and summary["eligible_windows"] > 0
        ),
    }

    write_csv(paths["reports"] / "phase1_video_audit.csv", status_rows)
    write_csv(paths["reports"] / "phase1_window_failures.csv", failed_window_rows)
    write_json(paths["reports"] / "phase1_summary.json", summary)
    return status_rows, failed_window_rows, summary


def phase1_acceptance(artifact_root : Path, allow_residual_failures : bool = False) -> dict[str, Any] :
    """Enforce the small set of Phase 1 conditions required before release building."""
    paths = corpus_paths(artifact_root)
    verification                = verify_inventory(artifact_root)
    _, failed_windows, summary  = audit_phase1(artifact_root)
    blockers                    = []

    if (summary["missing_videos"] > 0) :
        blockers.append(f"{summary['missing_videos']} expected videos are missing")
    if (summary["partial_videos"] > 0) :
        blockers.append(f"{summary['partial_videos']} transcript artifacts are partial")
    if (summary["invalid_artifacts"] > 0) :
        blockers.append(f"{summary['invalid_artifacts']} transcript artifacts are invalid")
    if (summary["extra_transcript_artifact_ids"]) :
        blockers.append(f"extra transcript artifacts exist: {summary['extra_transcript_artifact_ids'][ : 20]}")
    if (summary["complete_videos"] != summary["total_expected_videos"]) :
        blockers.append("not every expected video is complete")
    if (summary["eligible_windows"] <= 0) :
        blockers.append("the corpus has no retrieval-eligible transcript windows")
    if (summary["failed_windows"] > 0 and not allow_residual_failures) :
        blockers.append(f"{summary['failed_windows']} failed ASR windows remain")

    acceptance = {
        "schema_version"                        : PHASE1_ACCEPTANCE_SCHEMA_VERSION,
        "accepted_at_utc"                       : utc_now(),
        "inventory_identity"                    : summary["inventory_identity"],
        "inventory_verified_at_utc"             : verification["verified_at_utc"],
        "total_expected_videos"                 : summary["total_expected_videos"],
        "complete_videos"                       : summary["complete_videos"],
        "physical_window_count"                 : summary["physical_windows"],
        "eligible_window_count"                 : summary["eligible_windows"],
        "residual_failed_window_count"          : summary["failed_windows"],
        "residual_failed_window_ids"            : [row["window_id"] for row in failed_windows],
        "residual_failures_explicitly_accepted" : summary["failed_windows"] > 0 and allow_residual_failures,
        "blockers"                              : blockers,
        "passed"                                : not blockers,
    }
    write_json(paths["phase1_acceptance"], acceptance)

    if (blockers) :
        raise RuntimeError("Phase 1 acceptance failed:\n- " + "\n- ".join(blockers))

    return acceptance


# Phase 2 release


def build_corpus_release(
    artifact_root : Path,
    release_id : str,
    allow_residual_failures : bool = False,
    previous_release : Path | None = None,
) -> dict[str, Any] :
    """Build BM25 + E5 from accepted Phase 1 artifacts and validate the immutable release."""
    from asr_retrieval.artifacts import build_release, resolve_active_release, validate_release
    from asr_retrieval.dense import E5Encoder

    paths      = ensure_directories(artifact_root)
    acceptance = phase1_acceptance(artifact_root, allow_residual_failures = allow_residual_failures)
    config     = make_corpus_build_config(artifact_root)

    previous = Path(previous_release) if previous_release is not None else None
    if (previous is None and (paths["releases"] / "CURRENT").is_file()) :
        previous = resolve_active_release(paths["releases"])
    if (previous is not None) :
        validate_release(previous, verify_hashes = True)

    encoder = E5Encoder(config.e5)
    print("Loading E5 document encoder...")
    encoder.load()

    try :
        release_path = build_release(
            workspace        = paths["phase1"],
            releases_root    = paths["releases"],
            release_id       = release_id,
            config           = config,
            encoder          = encoder,
            previous_release = previous,
        )
    finally :
        encoder.close()

    manifest = validate_release(release_path, verify_hashes = True)
    summary = {
        "created_at_utc"        : utc_now(),
        "release_id"            : manifest.release_id,
        "release_path"          : str(release_path),
        "previous_release"      : str(previous) if previous is not None else None,
        "corpus_identity"       : manifest.corpus_identity,
        "video_count"           : manifest.video_count,
        "physical_window_count" : manifest.physical_window_count,
        "eligible_window_count" : manifest.eligible_window_count,
        "embedding_shape"       : list(manifest.embedding_shape),
        "embedding_dtype"       : manifest.embedding_dtype,
        "transcripts_size_bytes" : directory_size_bytes(paths["transcripts"]),
        "bm25_size_bytes"       : directory_size_bytes(release_path / "bm25"),
        "e5_size_bytes"         : int((release_path / "e5_embeddings.npy").stat().st_size),
        "release_size_bytes"    : directory_size_bytes(release_path),
        "phase1_acceptance"     : acceptance,
        "passed"                : True,
    }
    write_json(paths["reports"] / "phase2_build_summary.json", summary)

    print(f"Release     : {manifest.release_id}")
    print(f"Videos      : {manifest.video_count:,}")
    print(f"Windows     : {manifest.physical_window_count:,} physical / {manifest.eligible_window_count:,} searchable")
    print(f"Release path: {release_path}")
    return summary


def validate_corpus_release(artifact_root : Path, release_id : str) -> dict[str, Any] :
    """Run the production hash validator and load the release mappings once."""
    from asr_retrieval.artifacts import load_release, validate_release
    from asr_retrieval.config import ArtifactConfig

    paths        = corpus_paths(artifact_root)
    release_path = paths["releases"] / release_id
    manifest     = validate_release(release_path, verify_hashes = True)
    loaded       = load_release(release_path, ArtifactConfig(verify_file_hashes = False, load_embeddings_into_ram = False))

    summary = {
        "validated_at_utc"      : utc_now(),
        "release_id"            : manifest.release_id,
        "release_path"          : str(release_path),
        "corpus_identity"       : manifest.corpus_identity,
        "video_count"           : manifest.video_count,
        "physical_window_count" : manifest.physical_window_count,
        "eligible_window_count" : manifest.eligible_window_count,
        "embedding_shape"       : list(manifest.embedding_shape),
        "embedding_dtype"       : manifest.embedding_dtype,
        "bm25_document_count"   : int(loaded.bm25.document_count),
        "verify_hashes"         : True,
        "passed"                : True,
    }
    write_json(paths["reports"] / "release_validation.json", summary)
    return summary


# Benchmark and activation


def finite_optional(value : float | None) -> bool :
    return value is None or math.isfinite(float(value))


def search_result_errors(result, candidate_cap : int) -> list[str] :
    errors = []
    ranks  = [int(hit.rank) for hit in result.hits]

    if (ranks != list(range(1, len(ranks) + 1))) :
        errors.append("video ranks are not contiguous from 1")
    if (int(result.timings.candidate_pair_count) > candidate_cap) :
        errors.append("candidate pair count exceeds the production cap")

    timing_values = [getattr(result.timings, field) for field in TIMING_FIELDS]
    if (not all(math.isfinite(float(value)) and float(value) >= 0.0 for value in timing_values)) :
        errors.append("one or more timing fields are non-finite or negative")

    for hit in result.hits :
        if (not finite_optional(hit.first_stage_score) or not finite_optional(hit.reranker_score) or not finite_optional(hit.final_score)) :
            errors.append(f"non-finite score in video hit {hit.video_id}")

        for window in hit.windows :
            if (window.video_id != hit.video_id) :
                errors.append(f"supporting window {window.window_id} belongs to the wrong video")
            if (not finite_optional(window.first_stage_score) or not finite_optional(window.reranker_score) or not finite_optional(window.final_score)) :
                errors.append(f"non-finite score in supporting window {window.window_id}")

    return errors


def numeric_summary(values : Sequence[float]) -> dict[str, float] :
    array = np.asarray(values, dtype = np.float64)
    if (array.size == 0) :
        return {}
    return {
        "mean" : float(np.mean(array)),
        "p50"  : float(np.percentile(array, 50)),
        "p95"  : float(np.percentile(array, 95)),
        "max"  : float(np.max(array)),
    }


def benchmark_release(
    artifact_root : Path,
    release_id : str,
    queries : Sequence[str],
    repetitions : int = 3,
    top_k : int = 50,
    ground_truth : Sequence[dict[str, Any]] | None = None,
    device : str = "cuda",
) -> dict[str, Any] :
    """Run visible first-stage and reranked smoke/latency checks on the final release."""
    from asr_retrieval.artifacts import validate_release
    from asr_retrieval.config import BGEConfig, ProductionConfig, RuntimeConfig
    from asr_retrieval.engine import ASRRetrievalEngine

    if (not queries) :
        raise ValueError("At least one benchmark query is required")
    if (repetitions <= 0) :
        raise ValueError("repetitions must be positive")

    paths        = ensure_directories(artifact_root)
    release_path = paths["releases"] / release_id
    manifest     = validate_release(release_path, verify_hashes = True)
    os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(paths["e5_cache"]))

    runtime = RuntimeConfig(
        e5_device               = device,
        load_reranker           = True,
        serialize_gpu_requests  = True,
        default_top_k           = top_k,
        default_windows_per_hit = 3,
    )
    bge        = BGEConfig(device = "cpu", dtype = "float32") if device == "cpu" else BGEConfig()
    production = ProductionConfig(bge = bge, runtime = runtime)
    engine     = ASRRetrievalEngine.from_artifacts(release_path, config = production, warmup = True)

    result_rows = []
    all_errors  = []
    modes       = [("first_stage", True), ("reranked", False)]
    total_runs  = len(queries) * len(modes) * repetitions
    progress    = tqdm(total = total_runs, desc = "Benchmark", unit = "search", dynamic_ncols = True)
    
    try :
        for mode, first_stage_only in modes :
            for query_index, query in enumerate(queries, start = 1) :
                baseline_order = None
                baseline_pairs = None

                for repetition in range(1, repetitions + 1) :
                    result = engine.search(
                        query,
                        top_k           = min(top_k, manifest.video_count),
                        first_stage_only = first_stage_only,
                    )

                    errors = search_result_errors(result, production.candidates.max_candidate_pairs)
                    order  = [hit.video_id for hit in result.hits]
                    pairs  = int(result.timings.candidate_pair_count)

                    if (baseline_order is None) :
                        baseline_order = order
                        baseline_pairs = pairs
                    else :
                        if (order != baseline_order) :
                            errors.append("repeated search returned a different video order")
                        if (pairs != baseline_pairs) :
                            errors.append("repeated search returned a different candidate pair count")

                    all_errors.extend({"query" : query, "mode" : mode, "error" : error} for error in errors)
                    result_rows.append({
                        "query"                : query,
                        "mode"                 : mode,
                        "repetition"           : repetition,
                        "top_video_id"         : result.hits[0].video_id if result.hits else None,
                        **asdict(result.timings),
                        "validation_errors"    : errors,
                    })
                    progress.update(1)

        aggregate = {}
        for mode, _ in modes :
            mode_rows = [row for row in result_rows if row["mode"] == mode]
            aggregate[mode] = {
                field : numeric_summary([float(row[field]) for row in mode_rows])
                for field in TIMING_FIELDS + ["candidate_pair_count"]
            }

        summary = {
            "created_at_utc"              : utc_now(),
            "release_id"                  : release_id,
            "corpus_identity"             : manifest.corpus_identity,
            "query_count"                 : len(queries),
            "repetitions"                 : repetitions,
            "top_k"                       : min(top_k, manifest.video_count),
            "device"                      : device,
            "reranker_loaded"             : True,
            "candidate_pair_cap"          : production.candidates.max_candidate_pairs,
            "max_candidate_pairs_observed" : max(int(row["candidate_pair_count"]) for row in result_rows),
            "aggregate"                   : aggregate,
            "validation_errors"           : all_errors,
            "passed"                      : not all_errors,
        }

        write_csv(paths["reports"] / "benchmark_query_results.csv", result_rows)
        write_json(paths["reports"] / "latency_summary.json", summary)

        if (ground_truth is not None) :
            if (not ground_truth) :
                raise ValueError("ground_truth must contain at least one labeled query")

            recall_rows = []
            for row in tqdm(ground_truth, desc = "Recall@K", unit = "query", dynamic_ncols = True) :
                query           = str(row["query"])
                target_video_id = str(row["target_video_id"])
                result          = engine.search(query, top_k = min(100, manifest.video_count), first_stage_only = True)
                rank            = next((hit.rank for hit in result.hits if hit.video_id == target_video_id), None)
                recall_rows.append({"query" : query, "target_video_id" : target_video_id, "rank" : rank})

            denominator = len(recall_rows)
            recall_summary = {
                "release_id"    : release_id,
                "query_count"   : denominator,
                "recall_at_30"  : sum(int(row["rank"] is not None and row["rank"] <= 30) for row in recall_rows) / denominator,
                "recall_at_50"  : sum(int(row["rank"] is not None and row["rank"] <= 50) for row in recall_rows) / denominator,
                "recall_at_75"  : sum(int(row["rank"] is not None and row["rank"] <= 75) for row in recall_rows) / denominator,
                "recall_at_100" : sum(int(row["rank"] is not None and row["rank"] <= 100) for row in recall_rows) / denominator,
                "queries"       : recall_rows,
            }
            write_json(paths["reports"] / "candidate_recall_summary.json", recall_summary)
            summary["candidate_recall"] = recall_summary

        if (not summary["passed"]) :
            raise RuntimeError("Benchmark structural checks failed. Review benchmark_query_results.csv.")

        return summary
    finally :
        progress.close()
        engine.close()


def activate_corpus_release(artifact_root : Path, release_id : str) -> dict[str, Any] :
    """Activate only a release whose matching validation and reranked benchmark passed."""
    from asr_retrieval.artifacts import activate_release, resolve_active_release, validate_release

    paths        = corpus_paths(artifact_root)
    release_path = paths["releases"] / release_id
    manifest     = validate_release(release_path, verify_hashes = True)

    validation_path = paths["reports"] / "release_validation.json"
    benchmark_path  = paths["reports"] / "latency_summary.json"

    if (not validation_path.is_file()) :
        raise RuntimeError("release_validation.json is missing")
    if (not benchmark_path.is_file()) :
        raise RuntimeError("latency_summary.json is missing")

    validation = read_json(validation_path)
    benchmark  = read_json(benchmark_path)

    if (validation.get("release_id") != release_id or not validation.get("passed", False)) :
        raise RuntimeError("The latest hash validation does not pass for this release")
    if (benchmark.get("release_id") != release_id or not benchmark.get("passed", False)) :
        raise RuntimeError("The latest benchmark does not pass for this release")
    if (not benchmark.get("reranker_loaded", False)) :
        raise RuntimeError("Activation requires a benchmark with the production reranker loaded")

    acceptance = read_json(paths["phase1_acceptance"])
    inventory  = read_json(paths["inventory_summary"])
    if (not acceptance.get("passed", False) or acceptance.get("inventory_identity") != inventory.get("inventory_identity")) :
        raise RuntimeError("Phase 1 acceptance is missing, failed, or belongs to another inventory")

    activate_release(paths["releases"], release_id)
    active = resolve_active_release(paths["releases"])
    if (active.resolve() != release_path.resolve()) :
        raise RuntimeError("CURRENT does not resolve to the requested release after activation")

    summary = {
        "created_at_utc"   : utc_now(),
        "active_release_id" : release_id,
        "active_release_path" : str(active),
        "corpus_identity"   : manifest.corpus_identity,
        "video_count"       : manifest.video_count,
        "physical_windows"  : manifest.physical_window_count,
        "eligible_windows"  : manifest.eligible_window_count,
        "passed"            : True,
    }
    write_json(paths["reports"] / "final_build_summary.json", summary)
    return summary
