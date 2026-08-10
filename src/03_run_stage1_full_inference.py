# Relative path: src/03_run_stage1_full_inference.py
# Purpose: Resume-safe ASR inference for one selected model on any frozen benchmark manifest.

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any
import gc
import hashlib
import importlib
import json
import math
import os
import shutil
import subprocess
import sys
import traceback


PROJECT_ROOT = Path(
    os.environ.get(
        "ASR_PROJECT_ROOT",
        "/content/drive/MyDrive/aic26/asr_model_comparison",
    )
)


def project_path_from_env(
    name : str,
    default_relative_path : str,
) -> Path :
    configured = Path(
        os.environ.get(
            name,
            default_relative_path,
        )
    )

    return (
        configured
        if configured.is_absolute()
        else PROJECT_ROOT / configured
    )


CONFIG_PATH = project_path_from_env(
    "ASR_CONFIG_PATH",
    "configs/stage1_models.json",
)
BENCHMARK_PATH = project_path_from_env(
    "BENCHMARK_PATH",
    "data/benchmark_manifest.json",
)
ADAPTER_PATH = PROJECT_ROOT / "src" / "model_adapters.py"
POSTPROCESS_PATH = PROJECT_ROOT / "src" / "postprocess.py"

ACTIVE_MODEL_ID = os.environ.get("ACTIVE_MODEL_ID", "").strip()
OUTPUT_NAMESPACE = os.environ.get("OUTPUT_NAMESPACE", "").strip()

INSTALL_DEPENDENCIES = os.environ.get("INSTALL_DEPENDENCIES", "1") == "1"
RETRY_FAILED         = os.environ.get("RETRY_FAILED", "0") == "1"
RETRY_EMPTY          = os.environ.get("RETRY_EMPTY", "0") == "1"
ALLOW_UNSELECTED     = os.environ.get("ALLOW_UNSELECTED_MODEL", "0") == "1"

REFERENCE_MODEL_ID = "whisper_large_v3"
SCHEMA_VERSION     = "1.2"

TEMP_RUN_LABEL = (
    OUTPUT_NAMESPACE
    or BENCHMARK_PATH.stem
    or "benchmark"
)
TEMP_AUDIO_ROOT = Path(
    os.environ.get(
        "ASR_TEMP_AUDIO_ROOT",
        f"/content/asr_full_inference/{TEMP_RUN_LABEL}/{ACTIVE_MODEL_ID or 'unselected'}",
    )
)


def utc_now() -> str :
    return datetime.now(timezone.utc).isoformat()


def load_json(path : Path) -> Any :
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_json_bytes(value : Any) -> bytes :
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def hash_json(value : Any) -> str :
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def hash_file(path : Path, chunk_size : int = 8 * 1024 * 1024) -> str :
    digest = hashlib.sha256()

    with path.open("rb") as file :
        while True :
            chunk = file.read(chunk_size)

            if (not chunk) :
                break

            digest.update(chunk)

    return digest.hexdigest()


def write_json_atomic(path : Path, value : Any) -> None :
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")

    temporary_path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    os.replace(temporary_path, path)


def finite_float(value : Any) -> float | None :
    try :
        number = float(value)
    except (TypeError, ValueError) :
        return None

    return number if math.isfinite(number) else None


def prepare_window_audio(
    source_wav : Path,
    destination_wav : Path,
    window : dict[str, Any],
    expected_sample_rate : int,
) -> str :
    import numpy as np
    import soundfile as sf

    destination_wav.parent.mkdir(parents=True, exist_ok=True)

    with sf.SoundFile(str(source_wav), "r") as source :
        if (source.samplerate != expected_sample_rate) :
            raise RuntimeError(
                f"Unexpected sample rate for {source_wav}: "
                f"{source.samplerate}, expected {expected_sample_rate}."
            )

        source.seek(int(window["sample_start"]))
        audio = source.read(
            frames=int(window["duration_samples"]),
            dtype="int16",
            always_2d=True,
        )

    if (audio.shape[0] != int(window["duration_samples"])) :
        raise RuntimeError(
            f"{window['window_id']} produced {audio.shape[0]} samples, "
            f"expected {window['duration_samples']}."
        )

    if (audio.shape[1] == 1) :
        mono_audio = audio[:, 0]
    else :
        mono_audio = np.rint(
            audio.astype(np.float32).mean(axis=1)
        ).clip(-32768, 32767).astype(np.int16)

    pcm_hash = hashlib.sha256(
        mono_audio.astype("<i2", copy=False).tobytes()
    ).hexdigest()

    sf.write(
        str(destination_wav),
        mono_audio,
        expected_sample_rate,
        subtype="PCM_16",
        format="WAV",
    )

    return pcm_hash


def timestamp_warnings(
    segments : list[dict[str, Any]],
    duration_s : float,
) -> list[str] :
    for segment in segments :
        start = finite_float(segment.get("start_s"))
        end   = finite_float(segment.get("end_s"))

        if (start is None and end is None) :
            continue

        if (
            start is None
            or end is None
            or start < -0.25
            or end < start
            or end > duration_s + 1.0
        ) :
            return ["invalid_timestamps"]

    return []


def rebuild_postprocessing(
    windows : list[dict[str, Any]],
    postprocess_module : Any,
) -> list[dict[str, Any]] :
    rebuilt = []

    for window in windows :
        current = dict(window)
        current.update(postprocess_module.postprocess_window(current))
        rebuilt.append(current)

    return postprocess_module.mark_consecutive_duplicate_windows(rebuilt)


def validate_existing_payload(
    payload : dict[str, Any],
    stage_id : str,
    model_id : str,
    video_id : str,
    benchmark_content_hash : str,
    model_configuration_hash : str,
    adapter_hash : str,
) -> None :
    expected = {
        "stage_id"                 : stage_id,
        "model_id"                 : model_id,
        "video_id"                 : video_id,
        "benchmark_content_hash"   : benchmark_content_hash,
        "model_configuration_hash" : model_configuration_hash,
        "adapter_hash"             : adapter_hash,
    }

    mismatches = {
        key : {
            "expected" : expected_value,
            "found"    : payload.get(key),
        }
        for key, expected_value in expected.items()
        if payload.get(key) != expected_value
    }

    if (mismatches) :
        raise RuntimeError(
            "Existing output is incompatible with this run:\n"
            + json.dumps(mismatches, ensure_ascii=False, indent=2)
        )


def reusable_window(
    existing : dict[str, Any],
    expected : dict[str, Any],
    pcm_hash : str,
    wav_hash : str,
    window_policy_hash : str,
    model_configuration_hash : str,
    adapter_hash : str,
    resolved_revision : str | None,
) -> bool :
    required = {
        "window_id"                : expected["window_id"],
        "video_id"                 : expected["video_id"],
        "window_index"             : expected["window_index"],
        "sample_start"             : expected["sample_start"],
        "sample_end"               : expected["sample_end"],
        "duration_samples"         : expected["duration_samples"],
        "window_pcm_hash"          : pcm_hash,
        "wav_hash"                 : wav_hash,
        "window_policy_hash"       : window_policy_hash,
        "model_configuration_hash" : model_configuration_hash,
        "adapter_hash"             : adapter_hash,
        "model_revision"           : resolved_revision,
    }

    if (
        any(
            existing.get(key) != expected_value
            for key, expected_value in required.items()
        )
    ) :
        return False

    status   = str(existing.get("status", "")).lower()
    raw_text = str(existing.get("raw_text", "") or "").strip()

    if (status == "ok" and raw_text) :
        return True

    if (status == "ok" and not raw_text) :
        return not RETRY_EMPTY

    if (status == "failed") :
        return not RETRY_FAILED

    return False


def resolve_output_paths(
    stage_id : str,
    model_id : str,
) -> tuple[Path, Path] :
    if (OUTPUT_NAMESPACE) :
        output_root = (
            PROJECT_ROOT
            / "outputs"
            / OUTPUT_NAMESPACE
            / model_id
        )
        report_path = (
            PROJECT_ROOT
            / "reports"
            / OUTPUT_NAMESPACE
            / "full_inference"
            / f"{model_id}.json"
        )

        return output_root, report_path

    if (stage_id == "stage1_10_video") :
        return (
            PROJECT_ROOT / "outputs" / model_id,
            PROJECT_ROOT
            / "reports"
            / "stage1"
            / "full_inference"
            / f"{model_id}.json",
        )

    return (
        PROJECT_ROOT / "outputs" / stage_id / model_id,
        PROJECT_ROOT
        / "reports"
        / stage_id
        / "full_inference"
        / f"{model_id}.json",
    )


def build_summary(
    stage_id : str,
    benchmark_path : Path,
    output_namespace : str,
    model_id : str,
    model_specification : dict[str, Any],
    resolved_revision : str | None,
    model_load_runtime_s : float,
    output_root : Path,
    expected_windows_by_video : dict[str, list[dict[str, Any]]],
    sample_rate : int,
    benchmark_content_hash : str,
    model_configuration_hash : str,
    adapter_hash : str,
    adapter_version : str,
    run_started_at : str,
    device_name : str,
) -> dict[str, Any] :
    all_windows = []
    per_video   = []

    for video_id, expected_windows in expected_windows_by_video.items() :
        output_path = output_root / f"{video_id}.json"
        payload = load_json(output_path) if output_path.exists() else {"windows" : []}
        windows = payload.get("windows", [])
        all_windows.extend(windows)

        statuses = Counter(
            str(window.get("status", "missing"))
            for window in windows
        )
        empty_count = sum(
            window.get("status") == "ok"
            and not str(window.get("raw_text", "") or "").strip()
            for window in windows
        )

        per_video.append({
            "video_id"              : video_id,
            "expected_window_count" : len(expected_windows),
            "saved_window_count"    : len(windows),
            "ok_window_count"       : statuses.get("ok", 0),
            "failed_window_count"   : statuses.get("failed", 0),
            "empty_window_count"    : empty_count,
            "complete"              : len(windows) == len(expected_windows),
            "output_path"           : str(output_path.relative_to(PROJECT_ROOT)),
        })

    successful = [
        window
        for window in all_windows
        if window.get("status") == "ok"
    ]
    failed = [
        window
        for window in all_windows
        if window.get("status") == "failed"
    ]
    empty = [
        window
        for window in successful
        if not str(window.get("raw_text", "") or "").strip()
    ]

    total_audio_s = sum(
        float(window.get("duration_samples", 0)) / float(sample_rate)
        for window in successful
    )
    total_runtime_s = sum(
        float(window.get("runtime_s", 0.0) or 0.0)
        for window in successful
    )
    runtime_values = [
        float(window["runtime_s"])
        for window in successful
        if window.get("runtime_s") is not None
    ]

    warning_counts = Counter(
        reason
        for window in all_windows
        for reason in window.get("warning_reasons", [])
    )
    rejection_counts = Counter(
        reason
        for window in all_windows
        for reason in window.get("rejection_reasons", [])
    )

    expected_window_count = sum(
        len(windows)
        for windows in expected_windows_by_video.values()
    )

    return {
        "schema_version"                 : SCHEMA_VERSION,
        "stage_id"                       : stage_id,
        "benchmark_path"                 : str(benchmark_path),
        "output_namespace"               : output_namespace or None,
        "model_id"                       : model_id,
        "adapter"                        : model_specification["adapter"],
        "model_name"                     : model_specification["model_name"],
        "configured_revision"            : model_specification.get("revision"),
        "resolved_revision"              : resolved_revision,
        "benchmark_content_hash"         : benchmark_content_hash,
        "model_configuration_hash"       : model_configuration_hash,
        "adapter_hash"                   : adapter_hash,
        "adapter_version"                : adapter_version,
        "device_name"                    : device_name,
        "run_started_at_utc"             : run_started_at,
        "updated_at_utc"                 : utc_now(),
        "model_load_runtime_s"            : model_load_runtime_s,
        "expected_video_count"            : len(expected_windows_by_video),
        "expected_window_count"           : expected_window_count,
        "saved_window_count"              : len(all_windows),
        "successful_window_count"         : len(successful),
        "failed_window_count"             : len(failed),
        "empty_window_count"              : len(empty),
        "total_audio_s"                   : total_audio_s,
        "total_inference_runtime_s"       : total_runtime_s,
        "aggregate_rtf"                   : (
            total_runtime_s / total_audio_s
            if total_audio_s > 0
            else None
        ),
        "mean_window_runtime_s"            : (
            sum(runtime_values) / len(runtime_values)
            if runtime_values
            else None
        ),
        "peak_gpu_memory_bytes"            : max(
            (
                int(window.get("peak_gpu_memory_bytes", 0) or 0)
                for window in all_windows
            ),
            default=0,
        ),
        "peak_reserved_memory_bytes"       : max(
            (
                int(window.get("peak_reserved_memory_bytes", 0) or 0)
                for window in all_windows
            ),
            default=0,
        ),
        "warning_reason_counts"            : dict(warning_counts),
        "rejection_reason_counts"          : dict(rejection_counts),
        "retry_failed"                     : RETRY_FAILED,
        "retry_empty"                      : RETRY_EMPTY,
        "complete"                         : (
            len(all_windows) == expected_window_count
            and len(failed) == 0
        ),
        "per_video"                        : per_video,
        "resolved_model_configuration"     : model_specification,
    }


def main() -> None :
    if (not ACTIVE_MODEL_ID) :
        raise RuntimeError(
            "Set ACTIVE_MODEL_ID before running this script."
        )

    for path in [
        CONFIG_PATH,
        BENCHMARK_PATH,
        ADAPTER_PATH,
        POSTPROCESS_PATH,
    ] :
        if (not path.exists()) :
            raise FileNotFoundError(path)

    configuration = load_json(CONFIG_PATH)
    benchmark     = load_json(BENCHMARK_PATH)

    stage_id = str(
        benchmark.get(
            "stage_id",
            BENCHMARK_PATH.stem,
        )
    )

    model_map = {
        model["model_id"] : model
        for model in configuration["models"]
    }

    if (ACTIVE_MODEL_ID not in model_map) :
        raise KeyError(
            f"Unknown model: {ACTIVE_MODEL_ID}. "
            f"Available: {sorted(model_map)}"
        )

    model_specification = model_map[ACTIVE_MODEL_ID]

    if (not bool(model_specification.get("enabled", True))) :
        raise ValueError(
            f"{ACTIVE_MODEL_ID} is disabled in {CONFIG_PATH}."
        )

    if (
        not bool(
            model_specification.get(
                "selected_for_stage1_full_run",
                False,
            )
        )
        and not ALLOW_UNSELECTED
    ) :
        raise ValueError(
            f"{ACTIVE_MODEL_ID} is not selected for a full run. "
            "Set ALLOW_UNSELECTED_MODEL=1 only for an intentional diagnostic run."
        )

    if (
        stage_id == "stage1_10_video"
        and ACTIVE_MODEL_ID == REFERENCE_MODEL_ID
        and OUTPUT_NAMESPACE in {"", "stage1_10_video"}
    ) :
        raise ValueError(
            "Whisper large-v3 is frozen for the original 10-video benchmark. "
            "Use the extension40 benchmark for new reference inference."
        )

    benchmark_content_hash = benchmark["benchmark_content_hash"]
    window_policy          = benchmark["window_policy"]
    window_policy_hash     = benchmark["window_policy_hash"]
    sample_rate            = int(window_policy["sample_rate"])

    model_configuration_hash = hash_json(model_specification)
    adapter_hash             = hash_file(ADAPTER_PATH)

    expected_windows_by_video = {}

    for window in benchmark["windows"] :
        expected_windows_by_video.setdefault(
            window["video_id"],
            [],
        ).append(window)

    for video_id in expected_windows_by_video :
        expected_windows_by_video[video_id] = sorted(
            expected_windows_by_video[video_id],
            key=lambda item : item["window_index"],
        )

    observed_video_count = len(expected_windows_by_video)
    observed_window_count = sum(
        len(windows)
        for windows in expected_windows_by_video.values()
    )

    expected_video_count = int(
        benchmark.get(
            "video_count",
            observed_video_count,
        )
    )
    expected_window_count = int(
        benchmark.get(
            "window_count",
            observed_window_count,
        )
    )
    expected_query_count = int(
        benchmark.get(
            "query_count",
            len(benchmark.get("queries", [])),
        )
    )

    validation_errors = []

    if (observed_video_count != expected_video_count) :
        validation_errors.append(
            f"video_count: observed {observed_video_count}, "
            f"manifest {expected_video_count}"
        )

    if (observed_window_count != expected_window_count) :
        validation_errors.append(
            f"window_count: observed {observed_window_count}, "
            f"manifest {expected_window_count}"
        )

    if (len(benchmark.get("queries", [])) != expected_query_count) :
        validation_errors.append(
            f"query_count: observed {len(benchmark.get('queries', []))}, "
            f"manifest {expected_query_count}"
        )

    selected_video_ids = set(
        benchmark.get(
            "selected_video_ids",
            expected_windows_by_video,
        )
    )

    if (selected_video_ids != set(expected_windows_by_video)) :
        validation_errors.append(
            "selected_video_ids do not match the video IDs represented by windows."
        )

    if (validation_errors) :
        raise RuntimeError(
            "Frozen benchmark validation failed:\n  - "
            + "\n  - ".join(validation_errors)
        )

    audio_map = {
        item["video_id"] : {
            **item,
            "absolute_path" : (
                Path(item["wav_path"])
                if Path(item["wav_path"]).is_absolute()
                else PROJECT_ROOT / item["wav_path"]
            ),
        }
        for item in benchmark["audio_files"]
    }

    if (set(audio_map) != set(expected_windows_by_video)) :
        raise RuntimeError(
            "audio_files do not match the videos represented by the benchmark windows."
        )

    for video_id, audio in audio_map.items() :
        wav_path = audio["absolute_path"]

        if (not wav_path.exists()) :
            raise FileNotFoundError(wav_path)

        actual_hash = hash_file(wav_path)

        if (actual_hash != audio["wav_sha256"]) :
            raise RuntimeError(
                f"WAV hash mismatch for {video_id}: "
                f"{actual_hash} != {audio['wav_sha256']}"
            )

    os.environ.setdefault(
        "HF_HOME",
        str(PROJECT_ROOT / "model_cache" / "huggingface"),
    )
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    sys.path.insert(0, str(PROJECT_ROOT))

    from src import model_adapters
    from src import postprocess

    importlib.reload(model_adapters)
    importlib.reload(postprocess)

    if (
        INSTALL_DEPENDENCIES
        and not bool(
            model_specification.get(
                "install_dependencies_in_full_run",
                True,
            )
        )
    ) :
        raise RuntimeError(
            f"{ACTIVE_MODEL_ID} is configured to preserve its validated environment. "
            "Set INSTALL_DEPENDENCIES=0 and rerun in the prepared runtime."
        )

    if (INSTALL_DEPENDENCIES) :
        packages = model_adapters.install_instructions_for(
            model_specification
        )

        if (packages) :
            print("Installing dependencies:")

            for package in packages :
                print(f"  - {package}")

            subprocess.check_call([
                sys.executable,
                "-m",
                "pip",
                "install",
                *packages,
            ])

            importlib.invalidate_caches()
            importlib.reload(model_adapters)
            importlib.reload(postprocess)

    import torch

    device_name = (
        torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else "cpu"
    )

    output_root, report_path = resolve_output_paths(
        stage_id,
        ACTIVE_MODEL_ID,
    )

    output_root.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    TEMP_AUDIO_ROOT.mkdir(parents=True, exist_ok=True)

    adapter = model_adapters.create_adapter(
        ACTIVE_MODEL_ID,
        model_specification,
    )

    if (torch.cuda.is_available()) :
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    run_started_at = utc_now()
    load_started   = perf_counter()

    try :
        adapter.load()

        if (torch.cuda.is_available()) :
            torch.cuda.synchronize()

    except Exception as error :
        failure = {
            "schema_version"  : SCHEMA_VERSION,
            "stage_id"        : stage_id,
            "benchmark_path"  : str(BENCHMARK_PATH),
            "output_namespace": OUTPUT_NAMESPACE or None,
            "model_id"        : ACTIVE_MODEL_ID,
            "status"          : "model_load_failed",
            "error_type"      : type(error).__name__,
            "error_message"   : str(error),
            "traceback"       : traceback.format_exc(),
            "updated_at_utc"  : utc_now(),
        }
        write_json_atomic(report_path, failure)
        raise

    model_load_runtime_s = perf_counter() - load_started
    resolved_revision    = adapter.resolved_revision()

    print("\n" + "=" * 88)
    print("ASR FULL INFERENCE")
    print("=" * 88)
    print(f"Stage:             {stage_id}")
    print(f"Benchmark:         {BENCHMARK_PATH}")
    print(f"Output namespace:  {OUTPUT_NAMESPACE or '(legacy/default)'}")
    print(f"Model:             {ACTIVE_MODEL_ID}")
    print(f"Revision:          {resolved_revision}")
    print(f"Device:            {device_name}")
    print(f"Model load time:   {model_load_runtime_s:.2f}s")
    print(f"Expected videos:   {expected_video_count}")
    print(f"Expected windows:  {expected_window_count}")
    print(f"Retry failed:      {RETRY_FAILED}")
    print(f"Retry empty:       {RETRY_EMPTY}")
    print(f"Outputs:           {output_root}")

    progress_index = 0

    for video_id, expected_windows in expected_windows_by_video.items() :
        output_path = output_root / f"{video_id}.json"
        existing_payload = None
        existing_window_map = {}

        if (output_path.exists()) :
            existing_payload = load_json(output_path)
            validate_existing_payload(
                payload=existing_payload,
                stage_id=stage_id,
                model_id=ACTIVE_MODEL_ID,
                video_id=video_id,
                benchmark_content_hash=benchmark_content_hash,
                model_configuration_hash=model_configuration_hash,
                adapter_hash=adapter_hash,
            )
            existing_window_map = {
                window["window_id"] : window
                for window in existing_payload.get("windows", [])
            }

        video_payload = {
            "schema_version"            : SCHEMA_VERSION,
            "stage_id"                  : stage_id,
            "benchmark_path"            : str(BENCHMARK_PATH),
            "output_namespace"          : OUTPUT_NAMESPACE or None,
            "model_id"                  : ACTIVE_MODEL_ID,
            "video_id"                  : video_id,
            "benchmark_content_hash"    : benchmark_content_hash,
            "model_configuration_hash"  : model_configuration_hash,
            "adapter_version"           : adapter.adapter_version,
            "adapter_hash"              : adapter_hash,
            "model_revision"            : resolved_revision,
            "created_at_utc"            : (
                existing_payload.get("created_at_utc")
                if existing_payload
                else utc_now()
            ),
            "updated_at_utc"            : utc_now(),
            "windows"                   : list(existing_window_map.values()),
        }

        print("\n" + "-" * 88)
        print(
            f"{video_id}: {len(expected_windows)} expected, "
            f"{len(existing_window_map)} already saved"
        )

        for expected_window in expected_windows :
            progress_index += 1
            window_id  = expected_window["window_id"]
            source_wav = audio_map[video_id]["absolute_path"]
            wav_hash   = audio_map[video_id]["wav_sha256"]
            temp_wav   = TEMP_AUDIO_ROOT / f"{window_id}.wav"

            pcm_hash = prepare_window_audio(
                source_wav=source_wav,
                destination_wav=temp_wav,
                window=expected_window,
                expected_sample_rate=sample_rate,
            )

            existing_window = existing_window_map.get(window_id)

            if (
                existing_window is not None
                and reusable_window(
                    existing=existing_window,
                    expected=expected_window,
                    pcm_hash=pcm_hash,
                    wav_hash=wav_hash,
                    window_policy_hash=window_policy_hash,
                    model_configuration_hash=model_configuration_hash,
                    adapter_hash=adapter_hash,
                    resolved_revision=resolved_revision,
                )
            ) :
                print(
                    f"[{progress_index:04d}/{expected_window_count}] "
                    f"{window_id}: reused ({existing_window['status']})"
                )
                temp_wav.unlink(missing_ok=True)
                continue

            if (torch.cuda.is_available()) :
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()

            started = perf_counter()

            try :
                native_output = adapter.transcribe(temp_wav)

                if (torch.cuda.is_available()) :
                    torch.cuda.synchronize()

                runtime_s = perf_counter() - started
                raw_text = str(native_output.get("text", "")).strip()
                native_segments = native_output.get("native_segments", []) or []
                duration_s = expected_window["duration_samples"] / sample_rate

                result = {
                    **expected_window,
                    "wav_hash"                   : wav_hash,
                    "window_length_seconds"      : window_policy["window_seconds"],
                    "window_stride_seconds"      : window_policy["stride_seconds"],
                    "window_policy_hash"         : window_policy_hash,
                    "window_pcm_hash"            : pcm_hash,
                    "adapter_version"            : adapter.adapter_version,
                    "adapter_hash"               : adapter_hash,
                    "model_configuration_hash"   : model_configuration_hash,
                    "model_id"                   : ACTIVE_MODEL_ID,
                    "model_revision"             : resolved_revision,
                    "status"                     : "ok",
                    "runtime_s"                  : runtime_s,
                    "peak_gpu_memory_bytes"      : (
                        int(torch.cuda.max_memory_allocated())
                        if torch.cuda.is_available()
                        else 0
                    ),
                    "peak_reserved_memory_bytes" : (
                        int(torch.cuda.max_memory_reserved())
                        if torch.cuda.is_available()
                        else 0
                    ),
                    "raw_text"                   : raw_text,
                    "native_result"              : native_output.get("native_result"),
                    "native_segments"            : native_segments,
                    "resolved_arguments"         : native_output.get(
                        "resolved_arguments",
                        {},
                    ),
                    "warning_reasons"            : timestamp_warnings(
                        native_segments,
                        duration_s,
                    ),
                    "rejection_reasons"          : [],
                    "error"                      : None,
                }
                result.update(postprocess.postprocess_window(result))

                print(
                    f"[{progress_index:04d}/{expected_window_count}] "
                    f"{window_id}: "
                    f"{'ok' if raw_text else 'ok-empty'}, "
                    f"{runtime_s:.2f}s, RTF {runtime_s / duration_s:.3f}"
                )

            except Exception as error :
                runtime_s = perf_counter() - started

                result = {
                    **expected_window,
                    "wav_hash"                   : wav_hash,
                    "window_length_seconds"      : window_policy["window_seconds"],
                    "window_stride_seconds"      : window_policy["stride_seconds"],
                    "window_policy_hash"         : window_policy_hash,
                    "window_pcm_hash"            : pcm_hash,
                    "adapter_version"            : adapter.adapter_version,
                    "adapter_hash"               : adapter_hash,
                    "model_configuration_hash"   : model_configuration_hash,
                    "model_id"                   : ACTIVE_MODEL_ID,
                    "model_revision"             : resolved_revision,
                    "status"                     : "failed",
                    "runtime_s"                  : runtime_s,
                    "peak_gpu_memory_bytes"      : (
                        int(torch.cuda.max_memory_allocated())
                        if torch.cuda.is_available()
                        else 0
                    ),
                    "peak_reserved_memory_bytes" : (
                        int(torch.cuda.max_memory_reserved())
                        if torch.cuda.is_available()
                        else 0
                    ),
                    "raw_text"                   : "",
                    "native_result"              : None,
                    "native_segments"            : [],
                    "resolved_arguments"         : {},
                    "warning_reasons"            : [],
                    "rejection_reasons"          : [],
                    "error"                      : {
                        "error_type"    : type(error).__name__,
                        "error_message" : str(error),
                        "traceback"     : traceback.format_exc(),
                    },
                }
                result.update(postprocess.postprocess_window(result))

                print(
                    f"[{progress_index:04d}/{expected_window_count}] "
                    f"{window_id}: FAILED – "
                    f"{type(error).__name__}: {error}"
                )

            existing_window_map[window_id] = result
            ordered_windows = [
                existing_window_map[window["window_id"]]
                for window in expected_windows
                if window["window_id"] in existing_window_map
            ]

            video_payload["updated_at_utc"] = utc_now()
            video_payload["windows"] = rebuild_postprocessing(
                ordered_windows,
                postprocess,
            )
            write_json_atomic(output_path, video_payload)

            summary = build_summary(
                stage_id=stage_id,
                benchmark_path=BENCHMARK_PATH,
                output_namespace=OUTPUT_NAMESPACE,
                model_id=ACTIVE_MODEL_ID,
                model_specification=model_specification,
                resolved_revision=resolved_revision,
                model_load_runtime_s=model_load_runtime_s,
                output_root=output_root,
                expected_windows_by_video=expected_windows_by_video,
                sample_rate=sample_rate,
                benchmark_content_hash=benchmark_content_hash,
                model_configuration_hash=model_configuration_hash,
                adapter_hash=adapter_hash,
                adapter_version=adapter.adapter_version,
                run_started_at=run_started_at,
                device_name=device_name,
            )
            write_json_atomic(report_path, summary)
            temp_wav.unlink(missing_ok=True)

    final_summary = build_summary(
        stage_id=stage_id,
        benchmark_path=BENCHMARK_PATH,
        output_namespace=OUTPUT_NAMESPACE,
        model_id=ACTIVE_MODEL_ID,
        model_specification=model_specification,
        resolved_revision=resolved_revision,
        model_load_runtime_s=model_load_runtime_s,
        output_root=output_root,
        expected_windows_by_video=expected_windows_by_video,
        sample_rate=sample_rate,
        benchmark_content_hash=benchmark_content_hash,
        model_configuration_hash=model_configuration_hash,
        adapter_hash=adapter_hash,
        adapter_version=adapter.adapter_version,
        run_started_at=run_started_at,
        device_name=device_name,
    )
    write_json_atomic(report_path, final_summary)

    adapter.close()
    del adapter
    gc.collect()

    if (torch.cuda.is_available()) :
        torch.cuda.empty_cache()

    shutil.rmtree(TEMP_AUDIO_ROOT, ignore_errors=True)

    print("\n" + "=" * 88)
    print("ASR FULL INFERENCE COMPLETE")
    print("=" * 88)
    print(f"Stage:                {stage_id}")
    print(f"Model:                {ACTIVE_MODEL_ID}")
    print(
        f"Successful windows:   "
        f"{final_summary['successful_window_count']}/"
        f"{final_summary['expected_window_count']}"
    )
    print(f"Failed windows:       {final_summary['failed_window_count']}")
    print(f"Empty windows:        {final_summary['empty_window_count']}")
    print(f"Aggregate RTF:        {final_summary['aggregate_rtf']}")
    print(f"Peak GPU memory:      {final_summary['peak_gpu_memory_bytes']}")
    print(f"Outputs:              {output_root}")
    print(f"Summary:              {report_path}")


if (__name__ == "__main__") :
    main()
