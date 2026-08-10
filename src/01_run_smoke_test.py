# Stage 1 smoke-test runner.
# Change ACTIVE_MODEL_ID and run this cell once per clean Colab runtime.

from google.colab import drive
drive.mount("/content/drive")

from pathlib import Path
from time import perf_counter
from typing import Any
import gc
import importlib
import json
import os
import subprocess
import sys
import traceback

import soundfile as sf


PROJECT_ROOT   = Path("/content/drive/MyDrive/aic26/asr_model_comparison")
CONFIG_PATH    = PROJECT_ROOT / "configs" / "stage1_models.json"
BENCHMARK_PATH = PROJECT_ROOT / "data" / "benchmark_manifest.json"

ACTIVE_MODEL_ID = os.environ.get("ACTIVE_MODEL_ID", "whisper_small")

TEMP_AUDIO_ROOT = Path("/content/asr_stage1_smoke_audio")


def load_json(path : Path) -> Any :
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_atomic(path : Path, value : Any) -> None :
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(temporary_path, path)


config = load_json(CONFIG_PATH)
benchmark = load_json(BENCHMARK_PATH)

model_map = {
    model["model_id"] : model
    for model in config["models"]
}

if (ACTIVE_MODEL_ID not in model_map) :
    raise KeyError(
        f"Unknown ACTIVE_MODEL_ID: {ACTIVE_MODEL_ID}. "
        f"Available: {sorted(model_map)}"
    )

if (ACTIVE_MODEL_ID == "whisper_large_v3") :
    raise ValueError(
        "whisper_large_v3 is already frozen as the reference. "
        "Select one of the five candidate models."
    )

model_specification = model_map[ACTIVE_MODEL_ID]
smoke_window_ids = config["smoke_test"]["window_ids"]
output_root = PROJECT_ROOT / config["smoke_test"]["output_directory"]
output_path = output_root / f"{ACTIVE_MODEL_ID}.json"

sys.path.insert(0, str(PROJECT_ROOT))

from src import model_adapters
importlib.reload(model_adapters)

install_packages = model_adapters.install_instructions_for(
    model_specification
)

if (install_packages) :
    print("Installing dependencies:")
    print("\n".join(f"  - {package}" for package in install_packages))

    subprocess.check_call([
        sys.executable,
        "-m",
        "pip",
        "install",
        *install_packages,
    ])

importlib.invalidate_caches()
importlib.reload(model_adapters)

window_map = {
    window["window_id"] : window
    for window in benchmark["windows"]
}

audio_map = {
    item["video_id"] : PROJECT_ROOT / item["wav_path"]
    for item in benchmark["audio_files"]
}

missing_windows = [
    window_id
    for window_id in smoke_window_ids
    if window_id not in window_map
]

if (missing_windows) :
    raise RuntimeError(f"Smoke-test windows are missing: {missing_windows}")

TEMP_AUDIO_ROOT.mkdir(parents=True, exist_ok=True)

temporary_window_paths = {}

for window_id in smoke_window_ids :
    window = window_map[window_id]
    source_wav = audio_map[window["video_id"]]
    destination_wav = TEMP_AUDIO_ROOT / f"{window_id}.wav"

    with sf.SoundFile(str(source_wav), "r") as source :
        if (source.samplerate != benchmark["window_policy"]["sample_rate"]) :
            raise RuntimeError(
                f"Unexpected sample rate for {source_wav}: {source.samplerate}"
            )

        source.seek(window["sample_start"])
        audio = source.read(
            frames=window["duration_samples"],
            dtype="float32",
            always_2d=False,
        )

    sf.write(
        str(destination_wav),
        audio,
        benchmark["window_policy"]["sample_rate"],
        subtype="PCM_16",
    )
    temporary_window_paths[window_id] = destination_wav

import torch

device_name = (
    torch.cuda.get_device_name(0)
    if torch.cuda.is_available()
    else "cpu"
)

if (torch.cuda.is_available()) :
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

adapter = model_adapters.create_adapter(
    ACTIVE_MODEL_ID,
    model_specification,
)

load_started = perf_counter()
load_error = None

try :
    adapter.load()

    if (torch.cuda.is_available()) :
        torch.cuda.synchronize()

except Exception as error :
    load_error = {
        "error_type"    : type(error).__name__,
        "error_message" : str(error),
        "traceback"     : traceback.format_exc(),
    }

load_runtime_s = perf_counter() - load_started

results = []

if (load_error is None) :
    for window_id in smoke_window_ids :
        window = window_map[window_id]
        window_audio = temporary_window_paths[window_id]

        if (torch.cuda.is_available()) :
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        started = perf_counter()

        try :
            output = adapter.transcribe(window_audio)

            if (torch.cuda.is_available()) :
                torch.cuda.synchronize()

            runtime_s = perf_counter() - started
            duration_s = (
                window["duration_samples"]
                / benchmark["window_policy"]["sample_rate"]
            )

            results.append({
                "window_id"                  : window_id,
                "video_id"                   : window["video_id"],
                "start_s"                    : window["start_s"],
                "end_s"                      : window["end_s"],
                "duration_s"                 : duration_s,
                "status"                     : "ok",
                "runtime_s"                  : runtime_s,
                "real_time_factor"           : runtime_s / duration_s,
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
                "raw_text"                   : str(output.get("text", "")).strip(),
                "native_segments"            : output.get("native_segments", []),
                "native_result"              : output.get("native_result"),
                "resolved_arguments"         : output.get("resolved_arguments", {}),
                "error_type"                 : None,
                "error_message"              : None,
            })

        except Exception as error :
            runtime_s = perf_counter() - started
            duration_s = (
                window["duration_samples"]
                / benchmark["window_policy"]["sample_rate"]
            )

            results.append({
                "window_id"                  : window_id,
                "video_id"                   : window["video_id"],
                "start_s"                    : window["start_s"],
                "end_s"                      : window["end_s"],
                "duration_s"                 : duration_s,
                "status"                     : "failed",
                "runtime_s"                  : runtime_s,
                "real_time_factor"           : runtime_s / duration_s,
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
                "native_segments"            : [],
                "native_result"              : None,
                "resolved_arguments"         : {},
                "error_type"                 : type(error).__name__,
                "error_message"              : str(error),
                "traceback"                  : traceback.format_exc(),
            })

resolved_revision = (
    adapter.resolved_revision()
    if load_error is None
    else model_specification.get("revision")
)

successful_results = [
    result
    for result in results
    if result["status"] == "ok"
]

total_audio_s = sum(
    result["duration_s"]
    for result in successful_results
)
total_runtime_s = sum(
    result["runtime_s"]
    for result in successful_results
)

smoke_result = {
    "stage_id"                  : config["stage_id"],
    "model_id"                  : ACTIVE_MODEL_ID,
    "adapter"                   : model_specification["adapter"],
    "model_name"                : model_specification["model_name"],
    "configured_revision"       : model_specification.get("revision"),
    "resolved_revision"         : resolved_revision,
    "role"                      : model_specification["role"],
    "device_name"               : device_name,
    "adapter_version"           : adapter.adapter_version,
    "model_load_status"         : "ok" if load_error is None else "failed",
    "model_load_runtime_s"      : load_runtime_s,
    "model_load_error"          : load_error,
    "requested_window_ids"      : smoke_window_ids,
    "successful_window_count"   : len(successful_results),
    "failed_window_count"       : len(results) - len(successful_results),
    "total_audio_s"             : total_audio_s,
    "total_inference_runtime_s" : total_runtime_s,
    "aggregate_rtf"             : (
        total_runtime_s / total_audio_s
        if total_audio_s > 0
        else None
    ),
    "results"                   : results,
}

write_json_atomic(output_path, smoke_result)

adapter.close()
del adapter
gc.collect()

if (torch.cuda.is_available()) :
    torch.cuda.empty_cache()

print("\n" + "=" * 80)
print("STAGE 1 MODEL SMOKE TEST")
print("=" * 80)
print(f"Model:                  {ACTIVE_MODEL_ID}")
print(f"Load status:            {smoke_result['model_load_status']}")
print(f"Resolved revision:      {resolved_revision}")
print(f"Successful windows:     {len(successful_results)}/{len(smoke_window_ids)}")
print(f"Aggregate RTF:          {smoke_result['aggregate_rtf']}")
print(f"Saved result:           {output_path}")

for result in results :
    print("\n" + "-" * 80)
    print(
        f"{result['window_id']} | {result['status']} | "
        f"{result['runtime_s']:.2f}s | RTF {result['real_time_factor']:.3f}"
    )

    if (result["status"] == "ok") :
        print(result["raw_text"][:1_000])
    else :
        print(f"{result['error_type']}: {result['error_message']}")
