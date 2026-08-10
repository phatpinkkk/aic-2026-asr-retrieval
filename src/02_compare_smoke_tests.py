# Compare all completed smoke-test outputs with frozen Whisper large-v3.

from google.colab import drive
drive.mount("/content/drive")

from itertools import combinations
from pathlib import Path
from typing import Any
import json
import re
import subprocess
import sys
import unicodedata

import pandas as pd


PROJECT_ROOT   = Path("/content/drive/MyDrive/aic26/asr_model_comparison")
CONFIG_PATH    = PROJECT_ROOT / "configs" / "stage1_models.json"
REFERENCE_ROOT = PROJECT_ROOT / "data" / "references" / "whisper_large_v3"

subprocess.check_call([
    sys.executable,
    "-m",
    "pip",
    "install",
    "--quiet",
    "jiwer",
])

from jiwer import cer, wer


def load_json(path : Path) -> Any :
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_text(text : str) -> str :
    text = unicodedata.normalize("NFKC", str(text)).lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def has_known_boilerplate(text : str) -> bool :
    normalized = normalize_text(text)
    phrases = [
        "hãy subscribe cho kênh ghiền mì gõ",
        "để không bỏ lỡ những video hấp dẫn",
    ]
    return any(
        normalize_text(phrase) in normalized
        for phrase in phrases
    )


config = load_json(CONFIG_PATH)
smoke_root = PROJECT_ROOT / config["smoke_test"]["output_directory"]
window_ids = config["smoke_test"]["window_ids"]

reference_windows = {}

for path in sorted(REFERENCE_ROOT.glob("*.json")) :
    if (path.name == "reference_manifest.json") :
        continue

    payload = load_json(path)

    for window in payload["windows"] :
        if (window["window_id"] in window_ids) :
            reference_windows[window["window_id"]] = window

missing_reference = sorted(set(window_ids) - set(reference_windows))

if (missing_reference) :
    raise RuntimeError(
        f"Missing frozen Whisper reference windows: {missing_reference}"
    )

candidate_models = [
    model
    for model in config["models"]
    if model["model_id"] != "whisper_large_v3"
]

available_results = {}
missing_result_files = []

for model in candidate_models :
    model_id = model["model_id"]
    result_path = smoke_root / f"{model_id}.json"

    if (not result_path.exists()) :
        missing_result_files.append(model_id)
        continue

    available_results[model_id] = load_json(result_path)

rows = []

for window_id in window_ids :
    reference = reference_windows[window_id]
    reference_text = str(reference.get("raw_text", "")).strip()
    reference_normalized = normalize_text(reference_text)
    duration_s = reference["duration_samples"] / 16_000

    rows.append({
        "model_id"                  : "whisper_large_v3",
        "window_id"                 : window_id,
        "status"                    : "reference",
        "runtime_s"                 : reference.get("runtime_s"),
        "real_time_factor"          : (
            reference.get("runtime_s") / duration_s
            if reference.get("runtime_s") is not None
            else None
        ),
        "peak_gpu_memory_bytes"      : reference.get("peak_gpu_memory_bytes"),
        "peak_reserved_memory_bytes" : reference.get("peak_reserved_memory_bytes"),
        "whisper_reference_wer"      : 0.0,
        "whisper_reference_cer"      : 0.0,
        "length_ratio"               : 1.0,
        "known_boilerplate"          : has_known_boilerplate(reference_text),
        "raw_text"                   : reference_text,
        "error_message"              : None,
    })

    for model_id, payload in available_results.items() :
        result_map = {
            result["window_id"] : result
            for result in payload.get("results", [])
        }
        result = result_map.get(window_id)

        if (result is None) :
            rows.append({
                "model_id"                  : model_id,
                "window_id"                 : window_id,
                "status"                    : "missing",
                "runtime_s"                 : None,
                "real_time_factor"          : None,
                "peak_gpu_memory_bytes"      : None,
                "peak_reserved_memory_bytes" : None,
                "whisper_reference_wer"      : None,
                "whisper_reference_cer"      : None,
                "length_ratio"               : None,
                "known_boilerplate"          : False,
                "raw_text"                   : "",
                "error_message"              : "Window missing from smoke output.",
            })
            continue

        candidate_text = str(result.get("raw_text", "")).strip()
        candidate_normalized = normalize_text(candidate_text)

        if (
            result.get("status") == "ok"
            and reference_normalized
            and candidate_normalized
        ) :
            reference_wer = wer(reference_normalized, candidate_normalized)
            reference_cer = cer(reference_normalized, candidate_normalized)
            length_ratio = (
                len(candidate_normalized.split())
                / len(reference_normalized.split())
            )
        else :
            reference_wer = None
            reference_cer = None
            length_ratio = None

        rows.append({
            "model_id"                  : model_id,
            "window_id"                 : window_id,
            "status"                    : result.get("status"),
            "runtime_s"                 : result.get("runtime_s"),
            "real_time_factor"          : result.get("real_time_factor"),
            "peak_gpu_memory_bytes"      : result.get("peak_gpu_memory_bytes"),
            "peak_reserved_memory_bytes" : result.get("peak_reserved_memory_bytes"),
            "whisper_reference_wer"      : reference_wer,
            "whisper_reference_cer"      : reference_cer,
            "length_ratio"               : length_ratio,
            "known_boilerplate"          : has_known_boilerplate(candidate_text),
            "raw_text"                   : candidate_text,
            "error_message"              : result.get("error_message"),
        })

comparison = pd.DataFrame(rows)

summary = (
    comparison[comparison["model_id"] != "whisper_large_v3"]
    .groupby("model_id", as_index=False)
    .agg(
        successful_windows=("status", lambda values : int(
            sum(value == "ok" for value in values)
        )),
        mean_rtf=("real_time_factor", "mean"),
        mean_whisper_reference_wer=("whisper_reference_wer", "mean"),
        mean_whisper_reference_cer=("whisper_reference_cer", "mean"),
        mean_length_ratio=("length_ratio", "mean"),
        maximum_peak_gpu_memory_bytes=("peak_gpu_memory_bytes", "max"),
        boilerplate_windows=("known_boilerplate", "sum"),
    )
)

pairwise_rows = []

model_texts = {
    model_id : {
        result["window_id"] : normalize_text(result.get("raw_text", ""))
        for result in payload.get("results", [])
        if result.get("status") == "ok"
    }
    for model_id, payload in available_results.items()
}
model_texts["whisper_large_v3"] = {
    window_id : normalize_text(reference_windows[window_id].get("raw_text", ""))
    for window_id in window_ids
}

for first_model, second_model in combinations(sorted(model_texts), 2) :
    window_scores = []

    for window_id in window_ids :
        first_text  = model_texts[first_model].get(window_id, "")
        second_text = model_texts[second_model].get(window_id, "")

        if (first_text and second_text) :
            window_scores.append(wer(first_text, second_text))

    pairwise_rows.append({
        "first_model"       : first_model,
        "second_model"      : second_model,
        "compared_windows"  : len(window_scores),
        "mean_pairwise_wer" : (
            sum(window_scores) / len(window_scores)
            if window_scores
            else None
        ),
    })

pairwise = pd.DataFrame(pairwise_rows)

comparison_path = smoke_root / "smoke_test_comparison.csv"
summary_path    = smoke_root / "smoke_test_model_summary.csv"
pairwise_path   = smoke_root / "smoke_test_pairwise_wer.csv"

comparison.to_csv(comparison_path, index=False)
summary.to_csv(summary_path, index=False)
pairwise.to_csv(pairwise_path, index=False)

print("\n" + "=" * 100)
print("SMOKE-TEST MODEL SUMMARY")
print("=" * 100)
display(summary)

print("\nPairwise transcript disagreement:")
display(pairwise)

for window_id in window_ids :
    print("\n" + "=" * 100)
    print(window_id)
    print("=" * 100)

    window_rows = comparison[
        comparison["window_id"] == window_id
    ][[
        "model_id",
        "status",
        "runtime_s",
        "real_time_factor",
        "whisper_reference_wer",
        "whisper_reference_cer",
        "length_ratio",
        "known_boilerplate",
        "raw_text",
        "error_message",
    ]]

    display(window_rows)

if (missing_result_files) :
    print("\nNot run yet: " + ", ".join(missing_result_files))

print("\nSaved:")
print(comparison_path)
print(summary_path)
print(pairwise_path)
