# Relative path: src/04_smoke_test_new_candidates.py
# Purpose: Compatibility smoke test for three additional Stage 1 ASR candidates.

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any
import csv
import gc
import hashlib
import importlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import traceback
import unicodedata


PROJECT_ROOT = Path(
    os.environ.get(
        "ASR_PROJECT_ROOT",
        "/content/drive/MyDrive/aic26/asr_model_comparison",
    )
)

BENCHMARK_PATH = PROJECT_ROOT / "data" / "benchmark_manifest.json"
REFERENCE_ROOT = PROJECT_ROOT / "data" / "references" / "whisper_large_v3"
POSTPROCESS_PATH = PROJECT_ROOT / "src" / "postprocess.py"

ACTIVE_MODEL_ID = os.environ.get("ACTIVE_MODEL_ID", "").strip()
INSTALL_DEPENDENCIES = os.environ.get("INSTALL_DEPENDENCIES", "1").strip() == "1"

SMOKE_WINDOW_IDS = [
    "L21_V015_0000",
    "L21_V015_0019",
    "K01_V009_0008",
]

OUTPUT_ROOT = PROJECT_ROOT / "reports" / "stage1" / "smoke_test_round2"
TEMP_AUDIO_ROOT = Path(f"/content/asr_smoke_round2/{ACTIVE_MODEL_ID or 'unselected'}")

MODEL_SPECS = {
    "whisper_large_v3_turbo" : {
        "model_name"       : "openai/whisper-large-v3-turbo",
        "backend"          : "transformers_whisper",
        "dependency_group" : "transformers_4_48",
        "dtype"            : "float16",
        "inference_mode"   : "documented_chunked_long_form",
        "install"          : [
            "transformers==4.48.0",
            "accelerate",
            "safetensors",
            "sentencepiece",
            "soundfile",
            "scipy",
            "jiwer",
            "huggingface_hub",
        ],
    },
    "phoasr_whisper_small" : {
        "model_name"       : "Qualcomm-AI-Research/PhoASR-whisper-small",
        "backend"          : "transformers_whisper",
        "dependency_group" : "transformers_4_48",
        "dtype"            : "float32",
        "inference_mode"   : "model_card_pipeline",
        "install"          : [
            "transformers==4.48.0",
            "accelerate",
            "safetensors",
            "sentencepiece",
            "soundfile",
            "scipy",
            "jiwer",
            "huggingface_hub",
        ],
    },
    "parakeet_ctc_0_6b_vietnamese" : {
        "model_name"       : "nvidia/parakeet-ctc-0.6b-Vietnamese",
        "model_filename"   : "parakeet-ctc-0.6b-vi.nemo",
        "backend"          : "nemo_parakeet_ctc",
        "dependency_group" : "nemo_asr",
        "dtype"            : "float32",
        "inference_mode"   : "nemo_greedy_ctc_with_timestamps",
        "install"          : [
            "Cython",
            "packaging",
            "nemo_toolkit[asr]==2.7.3",
            "huggingface_hub",
            "soundfile",
            "scipy",
            "jiwer",
        ],
    },
}


def utc_now() -> str :
    return datetime.now(timezone.utc).isoformat()


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


def json_safe(value : Any) -> Any :
    if (value is None or isinstance(value, (str, int, bool))) :
        return value

    if (isinstance(value, float)) :
        return value if math.isfinite(value) else None

    if (isinstance(value, Path)) :
        return str(value)

    if (isinstance(value, dict)) :
        return {str(key) : json_safe(item) for key, item in value.items()}

    if (isinstance(value, (list, tuple, set))) :
        return [json_safe(item) for item in value]

    for method_name in ["detach", "cpu"] :
        if (hasattr(value, method_name)) :
            try :
                value = getattr(value, method_name)()
            except Exception :
                pass

    if (hasattr(value, "tolist")) :
        try :
            return json_safe(value.tolist())
        except Exception :
            pass

    if (hasattr(value, "item")) :
        try :
            return json_safe(value.item())
        except Exception :
            pass

    if (hasattr(value, "__dict__")) :
        try :
            return json_safe(vars(value))
        except Exception :
            pass

    return str(value)


def hash_file(path : Path, chunk_size : int = 8 * 1024 * 1024) -> str :
    digest = hashlib.sha256()

    with path.open("rb") as file :
        while True :
            chunk = file.read(chunk_size)
            if (not chunk) :
                break
            digest.update(chunk)

    return digest.hexdigest()


def normalize_for_metric(text : str) -> str :
    text = unicodedata.normalize("NFKC", str(text or "")).lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def install_dependencies(specification : dict[str, Any]) -> None :
    if (not INSTALL_DEPENDENCIES) :
        return

    if (specification["backend"] == "nemo_parakeet_ctc") :
        subprocess.check_call(["apt-get", "update", "-qq"])
        subprocess.check_call([
            "apt-get", "install", "-y", "-qq", "libsndfile1", "ffmpeg",
        ])

    print("Installing dependencies:")
    for package in specification["install"] :
        print(f"  - {package}")

    subprocess.check_call([
        sys.executable,
        "-m",
        "pip",
        "install",
        *specification["install"],
    ])
    importlib.invalidate_caches()


def resolve_huggingface_revision(model_name : str, token : str | None) -> str :
    from huggingface_hub import HfApi

    information = HfApi().model_info(
        repo_id=model_name,
        revision="main",
        token=token,
    )

    if (not information.sha) :
        raise RuntimeError(f"Could not resolve a commit SHA for {model_name}.")

    return str(information.sha)


def prepare_window_audio(
    source_wav : Path,
    destination_wav : Path,
    window : dict[str, Any],
    expected_sample_rate : int,
) -> dict[str, Any] :
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

    return {
        "path"          : destination_wav,
        "pcm_hash"      : pcm_hash,
        "sample_rate"   : expected_sample_rate,
        "duration_s"    : len(mono_audio) / expected_sample_rate,
        "channel_count" : 1,
    }


def canonical_hf_segments(chunks : list[dict[str, Any]]) -> list[dict[str, Any]] :
    segments = []

    for index, chunk in enumerate(chunks) :
        timestamp = chunk.get("timestamp", (None, None))
        if (isinstance(timestamp, (list, tuple)) and len(timestamp) == 2) :
            start, end = timestamp
        else :
            start, end = None, None

        segments.append({
            "segment_id" : index,
            "start_s"    : float(start) if start is not None else None,
            "end_s"      : float(end) if end is not None else None,
            "text"       : str(chunk.get("text", "")).strip(),
            "confidence" : chunk.get("confidence"),
        })

    return segments


def canonical_nemo_segments(
    timestamps : dict[str, Any] | None,
) -> list[dict[str, Any]] :
    if (not isinstance(timestamps, dict)) :
        return []

    source_segments = timestamps.get("segment", [])
    if (not isinstance(source_segments, list)) :
        return []

    segments = []

    for index, segment in enumerate(source_segments) :
        if (not isinstance(segment, dict)) :
            continue

        text = segment.get(
            "segment",
            segment.get("text", segment.get("word", "")),
        )

        segments.append({
            "segment_id" : index,
            "start_s"    : (
                float(segment["start"])
                if segment.get("start") is not None
                else None
            ),
            "end_s"      : (
                float(segment["end"])
                if segment.get("end") is not None
                else None
            ),
            "text"       : str(text).strip(),
            "confidence" : segment.get("confidence"),
        })

    return segments


class TransformersWhisperRuntime :
    def __init__(
        self,
        specification : dict[str, Any],
        revision : str,
    ) :
        self.specification = specification
        self.revision      = revision
        self.model         = None
        self.processor     = None
        self.pipeline      = None

    def load(self) -> None :
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

        device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
        pipeline_device = 0 if torch.cuda.is_available() else -1
        torch_dtype = (
            torch.float16
            if (
                torch.cuda.is_available()
                and self.specification["dtype"] == "float16"
            )
            else torch.float32
        )

        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.specification["model_name"],
            revision=self.revision,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        self.model.to(device_name)

        self.processor = AutoProcessor.from_pretrained(
            self.specification["model_name"],
            revision=self.revision,
        )

        self.pipeline = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            chunk_length_s=30,
            device=pipeline_device,
            torch_dtype=torch_dtype,
            return_timestamps="word",
            generate_kwargs={
                "language" : "vi",
                "task"     : "transcribe",
            },
        )

    def transcribe(self, audio_path : Path) -> dict[str, Any] :
        if (self.pipeline is None) :
            raise RuntimeError("Transformers pipeline is not loaded.")

        native_result = self.pipeline(str(audio_path))
        chunks = native_result.get("chunks", []) if isinstance(native_result, dict) else []

        return {
            "text"               : str(native_result.get("text", "")).strip(),
            "native_result"      : json_safe(native_result),
            "native_segments"    : canonical_hf_segments(chunks),
            "resolved_arguments" : {
                "chunk_length_s"    : 30,
                "return_timestamps" : "word",
                "language"          : "vi",
                "task"              : "transcribe",
                "dtype"             : self.specification["dtype"],
            },
        }

    def close(self) -> None :
        self.pipeline  = None
        self.processor = None
        self.model     = None


class NeMoParakeetRuntime :
    def __init__(
        self,
        specification : dict[str, Any],
        revision : str,
        token : str | None,
    ) :
        self.specification = specification
        self.revision      = revision
        self.token         = token
        self.model         = None
        self.local_model_path = None

    def load(self) -> None :
        import torch
        from huggingface_hub import hf_hub_download
        import nemo.collections.asr as nemo_asr

        version_parts = tuple(
            int(part)
            for part in torch.__version__.split("+")[0].split(".")[:2]
        )
        if (version_parts < (2, 7)) :
            raise RuntimeError(
                "NeMo 2.7.3 requires PyTorch 2.7 or newer. "
                f"Current version: {torch.__version__}."
            )

        self.local_model_path = Path(
            hf_hub_download(
                repo_id=self.specification["model_name"],
                filename=self.specification["model_filename"],
                revision=self.revision,
                token=self.token,
            )
        )

        self.model = nemo_asr.models.ASRModel.restore_from(
            restore_path=str(self.local_model_path),
            map_location=torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            ),
        )
        self.model.eval()

        if (torch.cuda.is_available()) :
            self.model = self.model.cuda()

    def transcribe(self, audio_path : Path) -> dict[str, Any] :
        if (self.model is None) :
            raise RuntimeError("NeMo Parakeet model is not loaded.")

        outputs = self.model.transcribe(
            audio=[str(audio_path)],
            batch_size=1,
            timestamps=True,
        )

        hypothesis = outputs[0]
        text = hypothesis if isinstance(hypothesis, str) else getattr(hypothesis, "text", "")
        timestamps = (
            getattr(hypothesis, "timestamp", None)
            if not isinstance(hypothesis, str)
            else None
        )

        return {
            "text"               : str(text).strip(),
            "native_result"      : {
                "text"      : str(text).strip(),
                "timestamp" : json_safe(timestamps),
            },
            "native_segments"    : canonical_nemo_segments(timestamps),
            "resolved_arguments" : {
                "batch_size" : 1,
                "timestamps" : True,
                "decoder"    : "greedy_ctc",
                "dtype"      : self.specification["dtype"],
            },
        }

    def close(self) -> None :
        self.model = None


def load_reference_windows() -> dict[str, dict[str, Any]] :
    references = {}

    for path in sorted(REFERENCE_ROOT.glob("*.json")) :
        if (path.name == "reference_manifest.json") :
            continue

        payload = load_json(path)
        for window in payload.get("windows", []) :
            if (window.get("window_id") in SMOKE_WINDOW_IDS) :
                references[window["window_id"]] = window

    missing = sorted(set(SMOKE_WINDOW_IDS) - set(references))
    if (missing) :
        raise RuntimeError(f"Frozen Whisper reference windows are missing: {missing}")

    return references


def add_reference_metrics(
    result : dict[str, Any],
    reference : dict[str, Any],
) -> None :
    from jiwer import cer, wer

    reference_text = normalize_for_metric(reference.get("raw_text", ""))
    candidate_text = normalize_for_metric(result.get("raw_text", ""))
    metric_allowed = (
        result.get("status") == "ok"
        and bool(reference_text)
        and bool(candidate_text)
    )

    result["trusted_reference"] = result["window_id"] != "L21_V015_0000"
    result["reference_raw_text"] = reference.get("raw_text", "")
    result["whisper_reference_wer"] = (
        wer(reference_text, candidate_text) if metric_allowed else None
    )
    result["whisper_reference_cer"] = (
        cer(reference_text, candidate_text) if metric_allowed else None
    )
    result["reference_length_ratio"] = (
        len(candidate_text.split()) / len(reference_text.split())
        if metric_allowed
        else None
    )


def update_summary_csv() -> None :
    rows = []

    for model_id in MODEL_SPECS :
        path = OUTPUT_ROOT / f"{model_id}.json"
        if (not path.exists()) :
            continue

        payload = load_json(path)
        results = payload.get("results", [])
        trusted = [
            item
            for item in results
            if (
                item.get("trusted_reference")
                and item.get("whisper_reference_wer") is not None
            )
        ]
        all_metric = [
            item
            for item in results
            if item.get("whisper_reference_wer") is not None
        ]

        rows.append({
            "model_id"                    : model_id,
            "model_load_status"            : payload.get("model_load_status"),
            "successful_windows"           : payload.get("successful_window_count"),
            "empty_windows"                : payload.get("empty_window_count"),
            "failed_windows"               : payload.get("failed_window_count"),
            "aggregate_rtf"                : payload.get("aggregate_rtf"),
            "mean_reference_wer_all"       : (
                sum(item["whisper_reference_wer"] for item in all_metric)
                / len(all_metric)
                if all_metric
                else None
            ),
            "mean_reference_cer_all"       : (
                sum(item["whisper_reference_cer"] for item in all_metric)
                / len(all_metric)
                if all_metric
                else None
            ),
            "mean_reference_wer_trusted"   : (
                sum(item["whisper_reference_wer"] for item in trusted)
                / len(trusted)
                if trusted
                else None
            ),
            "mean_reference_cer_trusted"   : (
                sum(item["whisper_reference_cer"] for item in trusted)
                / len(trusted)
                if trusted
                else None
            ),
            "peak_gpu_memory_bytes"        : max(
                (int(item.get("peak_gpu_memory_bytes", 0) or 0) for item in results),
                default=0,
            ),
            "resolved_revision"            : payload.get("resolved_revision"),
        })

    summary_path = OUTPUT_ROOT / "round2_model_summary.csv"
    fieldnames = [
        "model_id",
        "model_load_status",
        "successful_windows",
        "empty_windows",
        "failed_windows",
        "aggregate_rtf",
        "mean_reference_wer_all",
        "mean_reference_cer_all",
        "mean_reference_wer_trusted",
        "mean_reference_cer_trusted",
        "peak_gpu_memory_bytes",
        "resolved_revision",
    ]

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8", newline="") as file :
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Updated summary: {summary_path}")


def main() -> None :
    if (ACTIVE_MODEL_ID not in MODEL_SPECS) :
        raise RuntimeError(
            "Set ACTIVE_MODEL_ID to one of:\n"
            + "\n".join(f"  - {model_id}" for model_id in MODEL_SPECS)
        )

    for required_path in [BENCHMARK_PATH, REFERENCE_ROOT, POSTPROCESS_PATH] :
        if (not required_path.exists()) :
            raise FileNotFoundError(required_path)

    specification = MODEL_SPECS[ACTIVE_MODEL_ID]

    print("\n" + "=" * 92)
    print("STAGE 1 ROUND-2 ASR SMOKE TEST")
    print("=" * 92)
    print(f"Model ID:          {ACTIVE_MODEL_ID}")
    print(f"Repository:        {specification['model_name']}")
    print(f"Backend:           {specification['backend']}")
    print(f"Inference mode:    {specification['inference_mode']}")
    print("Run policy:        one model per clean runtime")

    install_dependencies(specification)

    os.environ.setdefault(
        "HF_HOME",
        str(PROJECT_ROOT / "model_cache" / "huggingface"),
    )
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    if (specification["backend"] == "nemo_parakeet_ctc") :
        os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    revision = resolve_huggingface_revision(specification["model_name"], token)
    print(f"Resolved revision: {revision}")

    benchmark = load_json(BENCHMARK_PATH)
    references = load_reference_windows()
    sample_rate = int(benchmark["window_policy"]["sample_rate"])

    window_map = {window["window_id"] : window for window in benchmark["windows"]}
    audio_map = {
        item["video_id"] : {
            **item,
            "absolute_path" : PROJECT_ROOT / item["wav_path"],
        }
        for item in benchmark["audio_files"]
    }

    missing_windows = sorted(set(SMOKE_WINDOW_IDS) - set(window_map))
    if (missing_windows) :
        raise RuntimeError(f"Smoke windows are missing: {missing_windows}")

    for video_id, audio in audio_map.items() :
        path = audio["absolute_path"]
        if (not path.exists()) :
            raise FileNotFoundError(path)
        if (hash_file(path) != audio["wav_sha256"]) :
            raise RuntimeError(f"WAV hash mismatch for {video_id}.")

    TEMP_AUDIO_ROOT.mkdir(parents=True, exist_ok=True)
    prepared_audio = {}

    for window_id in SMOKE_WINDOW_IDS :
        window = window_map[window_id]
        source_wav = audio_map[window["video_id"]]["absolute_path"]
        prepared_audio[window_id] = prepare_window_audio(
            source_wav=source_wav,
            destination_wav=TEMP_AUDIO_ROOT / f"{window_id}.wav",
            window=window,
            expected_sample_rate=sample_rate,
        )

    sys.path.insert(0, str(PROJECT_ROOT))
    from src import postprocess
    importlib.reload(postprocess)

    import torch

    if (torch.cuda.is_available()) :
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    if (specification["backend"] == "transformers_whisper") :
        runtime = TransformersWhisperRuntime(specification, revision)
    else :
        runtime = NeMoParakeetRuntime(specification, revision, token)

    load_started = perf_counter()
    load_error = None

    try :
        runtime.load()
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
        for window_id in SMOKE_WINDOW_IDS :
            window = window_map[window_id]
            audio_information = prepared_audio[window_id]

            if (torch.cuda.is_available()) :
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()

            started = perf_counter()

            try :
                native_output = runtime.transcribe(audio_information["path"])
                if (torch.cuda.is_available()) :
                    torch.cuda.synchronize()

                runtime_s = perf_counter() - started
                raw_text = str(native_output.get("text", "")).strip()
                status = "ok" if raw_text else "empty_output"

                result = {
                    "window_id"                  : window_id,
                    "video_id"                   : window["video_id"],
                    "start_s"                    : window["start_s"],
                    "end_s"                      : window["end_s"],
                    "duration_s"                 : audio_information["duration_s"],
                    "sample_rate"                : audio_information["sample_rate"],
                    "channel_count"              : audio_information["channel_count"],
                    "window_pcm_hash"            : audio_information["pcm_hash"],
                    "status"                     : status,
                    "runtime_s"                  : runtime_s,
                    "real_time_factor"           : runtime_s / audio_information["duration_s"],
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
                    "native_segments"            : native_output.get("native_segments", []),
                    "resolved_arguments"         : native_output.get("resolved_arguments", {}),
                    "warning_reasons"            : [],
                    "rejection_reasons"          : [],
                    "error"                      : None,
                }
                result.update(postprocess.postprocess_window(result))

            except Exception as error :
                runtime_s = perf_counter() - started
                result = {
                    "window_id"                  : window_id,
                    "video_id"                   : window["video_id"],
                    "start_s"                    : window["start_s"],
                    "end_s"                      : window["end_s"],
                    "duration_s"                 : audio_information["duration_s"],
                    "sample_rate"                : audio_information["sample_rate"],
                    "channel_count"              : audio_information["channel_count"],
                    "window_pcm_hash"            : audio_information["pcm_hash"],
                    "status"                     : "failed",
                    "runtime_s"                  : runtime_s,
                    "real_time_factor"           : runtime_s / audio_information["duration_s"],
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

            add_reference_metrics(result, references[window_id])
            results.append(result)

            print("\n" + "-" * 92)
            print(
                f"{window_id} | {result['status']} | "
                f"{result['runtime_s']:.2f}s | "
                f"RTF {result['real_time_factor']:.3f}"
            )
            print(
                f"Reference WER: {result['whisper_reference_wer']} | "
                f"Trusted: {result['trusted_reference']}"
            )

            if (result["status"] == "failed") :
                print(
                    f"{result['error']['error_type']}: "
                    f"{result['error']['error_message']}"
                )
            else :
                print(result["raw_text"][:1_500])

    successful_results = [item for item in results if item["status"] == "ok"]
    total_audio_s = sum(item["duration_s"] for item in successful_results)
    total_runtime_s = sum(item["runtime_s"] for item in successful_results)

    output = {
        "schema_version"          : "1.0",
        "stage_id"                : "stage1_round2_smoke",
        "model_id"                : ACTIVE_MODEL_ID,
        "model_name"              : specification["model_name"],
        "resolved_revision"       : revision,
        "backend"                 : specification["backend"],
        "dependency_group"        : specification["dependency_group"],
        "inference_mode"          : specification["inference_mode"],
        "configured_dtype"        : specification["dtype"],
        "created_at_utc"          : utc_now(),
        "device_name"             : (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else "cpu"
        ),
        "python_version"          : sys.version,
        "torch_version"           : torch.__version__,
        "model_load_status"       : "ok" if load_error is None else "failed",
        "model_load_runtime_s"    : load_runtime_s,
        "model_load_error"        : load_error,
        "requested_window_ids"    : SMOKE_WINDOW_IDS,
        "successful_window_count" : len(successful_results),
        "empty_window_count"      : sum(item["status"] == "empty_output" for item in results),
        "failed_window_count"     : sum(item["status"] == "failed" for item in results),
        "aggregate_rtf"           : (
            total_runtime_s / total_audio_s
            if total_audio_s > 0
            else None
        ),
        "results"                 : results,
    }

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_ROOT / f"{ACTIVE_MODEL_ID}.json"
    write_json_atomic(output_path, output)

    runtime.close()
    del runtime
    gc.collect()

    if (torch.cuda.is_available()) :
        torch.cuda.empty_cache()

    shutil.rmtree(TEMP_AUDIO_ROOT, ignore_errors=True)
    update_summary_csv()

    print("\n" + "=" * 92)
    print("ROUND-2 SMOKE TEST COMPLETE")
    print("=" * 92)
    print(f"Model:                {ACTIVE_MODEL_ID}")
    print(f"Load status:          {output['model_load_status']}")
    print(f"Successful windows:   {output['successful_window_count']}/3")
    print(f"Empty windows:        {output['empty_window_count']}")
    print(f"Failed windows:       {output['failed_window_count']}")
    print(f"Aggregate RTF:        {output['aggregate_rtf']}")
    print(f"Saved result:         {output_path}")


if (__name__ == "__main__") :
    main()
