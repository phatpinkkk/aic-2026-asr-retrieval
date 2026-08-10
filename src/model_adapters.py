# Relative path: src/model_adapters.py
# Purpose: Lazy-loaded ASR adapters that expose one common inference interface.

from __future__ import annotations

import gc
import inspect
import math
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


ADAPTER_VERSION = "1.3"


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

    if (hasattr(value, "detach")) :
        try :
            value = value.detach()
        except Exception :
            pass

    if (hasattr(value, "cpu")) :
        try :
            value = value.cpu()
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


def _canonical_segments(native_segments : list[dict[str, Any]]) -> list[dict[str, Any]] :
    canonical = []

    for index, segment in enumerate(native_segments) :
        start = segment.get("start", segment.get("start_s"))
        end   = segment.get("end", segment.get("end_s"))
        text  = str(segment.get("text", "")).strip()

        canonical.append({
            "segment_id" : segment.get("id", segment.get("segment_id", index)),
            "start_s"    : float(start) if start is not None else None,
            "end_s"      : float(end) if end is not None else None,
            "text"       : text,
            "confidence" : segment.get("confidence"),
        })

    return canonical


def _canonical_nemo_timestamp_segments(
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


def _adapter_name(specification : dict[str, Any]) -> str :
    value = specification.get("adapter", specification.get("adapter_type"))

    if (value is None) :
        raise ValueError("Model specification must define 'adapter' or 'adapter_type'.")

    aliases = {
        "huggingface_whisper" : "hf_pipeline",
        "huggingface_ctc"     : "mms_ctc",
        "seamless"            : "seamless_m4t",
    }

    return aliases.get(str(value), str(value))


def _model_name(specification : dict[str, Any]) -> str :
    value = specification.get("model_name", specification.get("checkpoint"))

    if (value is None) :
        raise ValueError("Model specification must define 'model_name' or 'checkpoint'.")

    return str(value)


def _torch_dtype(specification : dict[str, Any], device : str) :
    import torch

    dtype_name = specification.get(
        "torch_dtype",
        specification.get("dtype", "float16" if device == "cuda" else "float32"),
    )

    if (device != "cuda" and dtype_name == "float16") :
        dtype_name = "float32"

    if (not hasattr(torch, dtype_name)) :
        raise ValueError(f"Unsupported torch dtype: {dtype_name}")

    return getattr(torch, dtype_name)


def _load_audio_mono(window_audio : Path, target_sample_rate : int = 16_000) :
    import numpy as np
    import soundfile as sf

    audio, sample_rate = sf.read(
        str(window_audio),
        dtype="float32",
        always_2d=True,
    )
    audio = audio.mean(axis=1)

    if (sample_rate != target_sample_rate) :
        from scipy.signal import resample_poly
        from math import gcd

        divisor = gcd(sample_rate, target_sample_rate)
        audio = resample_poly(
            audio,
            target_sample_rate // divisor,
            sample_rate // divisor,
        ).astype(np.float32)

    return audio, target_sample_rate


def _extract_text_and_segments(native_result : Any) -> tuple[str, list[dict[str, Any]]] :
    if (native_result is None) :
        return "", []

    if (isinstance(native_result, str)) :
        return native_result.strip(), []

    if (isinstance(native_result, (list, tuple)) and len(native_result) > 0) :
        if (len(native_result) == 1) :
            return _extract_text_and_segments(native_result[0])

        texts = []
        segments = []

        for item in native_result :
            text, item_segments = _extract_text_and_segments(item)

            if (text) :
                texts.append(text)

            segments.extend(item_segments)

        return " ".join(texts).strip(), segments

    if (isinstance(native_result, dict)) :
        text = ""

        for key in ["text", "transcription", "transcript", "prediction", "pred_text"] :
            if (native_result.get(key) is not None) :
                text = str(native_result[key]).strip()
                break

        native_segments = []

        for key in ["segments", "chunks", "timestamps"] :
            if (isinstance(native_result.get(key), list)) :
                native_segments = native_result[key]
                break

        segments = []

        for index, segment in enumerate(native_segments) :
            if (not isinstance(segment, dict)) :
                continue

            timestamp = segment.get("timestamp")
            start = segment.get("start", segment.get("start_s"))
            end   = segment.get("end", segment.get("end_s"))

            if (
                isinstance(timestamp, (list, tuple))
                and len(timestamp) == 2
            ) :
                start, end = timestamp

            segments.append({
                "id"         : segment.get("id", segment.get("segment_id", index)),
                "start"      : start,
                "end"        : end,
                "text"       : segment.get("text", ""),
                "confidence" : segment.get("confidence"),
            })

        return text, segments

    text = str(getattr(native_result, "text", "") or "").strip()
    segments = getattr(native_result, "segments", [])

    return text, segments if isinstance(segments, list) else []


def _commit_hash_from(value : Any) -> str | None :
    candidates = [
        value,
        getattr(value, "model", None),
        getattr(value, "processor", None),
        getattr(value, "tokenizer", None),
        getattr(value, "feature_extractor", None),
    ]

    for candidate in candidates :
        if (candidate is None) :
            continue

        config = getattr(candidate, "config", None)

        for source in [candidate, config] :
            if (source is None) :
                continue

            commit_hash = getattr(source, "_commit_hash", None)

            if (commit_hash) :
                return str(commit_hash)

    return None


class BaseASRAdapter(ABC) :
    install_instructions : list[str] = []

    def __init__(self, model_id : str, specification : dict[str, Any]) :
        self.model_id      = model_id
        self.specification = specification
        self.model         = None
        self.processor     = None

    @property
    def adapter_version(self) -> str :
        return ADAPTER_VERSION

    def resolved_configuration(self) -> dict[str, Any] :
        return json_safe(self.specification)

    def resolved_revision(self) -> str | None :
        return (
            _commit_hash_from(self.model)
            or _commit_hash_from(self.processor)
            or self.specification.get("revision")
        )

    @abstractmethod
    def load(self) -> None :
        raise NotImplementedError

    @abstractmethod
    def transcribe(self, window_audio : Path) -> dict[str, Any] :
        raise NotImplementedError

    def close(self) -> None :
        self.model     = None
        self.processor = None
        gc.collect()

        try :
            import torch

            if (torch.cuda.is_available()) :
                torch.cuda.empty_cache()
        except Exception :
            pass


class WhisperAdapter(BaseASRAdapter) :
    install_instructions = ["openai-whisper==20250625", "soundfile", "scipy"]

    def load(self) -> None :
        import whisper

        model_name = _model_name(self.specification)
        device     = self.specification.get("device", "cuda")
        cache_dir  = self.specification.get("cache_dir")

        self.model = whisper.load_model(
            model_name,
            device=device,
            download_root=cache_dir,
        )

    def resolved_revision(self) -> str | None :
        return str(
            self.specification.get(
                "revision",
                "openai-whisper-20250625",
            )
        )

    def transcribe(self, window_audio : Path) -> dict[str, Any] :
        if (self.model is None) :
            raise RuntimeError("Whisper model is not loaded.")

        import torch

        arguments = {
            "language"                   : self.specification.get("language", "vi"),
            "task"                       : "transcribe",
            "condition_on_previous_text" : self.specification.get("condition_on_previous_text", False),
            "word_timestamps"            : self.specification.get("word_timestamps", False),
            "temperature"                : self.specification.get("temperature", 0.0),
            "verbose"                    : False,
            "fp16"                       : bool(
                torch.cuda.is_available()
                and self.specification.get("device", "cuda") == "cuda"
            ),
        }

        native_result   = self.model.transcribe(str(window_audio), **arguments)
        native_segments = native_result.get("segments", [])

        return {
            "text"               : str(native_result.get("text", "")).strip(),
            "native_result"      : json_safe(native_result),
            "native_segments"    : _canonical_segments(native_segments),
            "resolved_arguments" : json_safe(arguments),
        }


class HFPipelineASRAdapter(BaseASRAdapter) :
    install_instructions = [
        "transformers>=4.48,<5",
        "accelerate",
        "safetensors",
        "sentencepiece",
        "soundfile",
        "scipy",
    ]

    def load(self) -> None :
        import torch
        from transformers import pipeline

        model_name  = _model_name(self.specification)
        revision    = self.specification.get("revision")
        device_name = self.specification.get("device", "cuda")
        device      = 0 if torch.cuda.is_available() and device_name == "cuda" else -1
        torch_dtype = _torch_dtype(
            self.specification,
            "cuda" if device >= 0 else "cpu",
        )

        pipeline_arguments = {
            "task"              : "automatic-speech-recognition",
            "model"             : model_name,
            "revision"          : revision,
            "device"            : device,
            "torch_dtype"       : torch_dtype,
            "trust_remote_code" : self.specification.get("trust_remote_code", False),
        }

        for key in ["chunk_length_s", "stride_length_s"] :
            if (self.specification.get(key) is not None) :
                pipeline_arguments[key] = self.specification[key]

        self.model = pipeline(**pipeline_arguments)

    def resolved_revision(self) -> str | None :
        return (
            _commit_hash_from(getattr(self.model, "model", None))
            or super().resolved_revision()
        )

    def transcribe(self, window_audio : Path) -> dict[str, Any] :
        if (self.model is None) :
            raise RuntimeError("Transformers ASR pipeline is not loaded.")

        arguments = {
            "return_timestamps" : self.specification.get("return_timestamps", True),
        }

        generate_kwargs = self.specification.get("generate_kwargs")

        if (generate_kwargs) :
            arguments["generate_kwargs"] = generate_kwargs

        native_result = self.model(str(window_audio), **arguments)
        text, segments = _extract_text_and_segments(native_result)

        return {
            "text"               : text,
            "native_result"      : json_safe(native_result),
            "native_segments"    : _canonical_segments(segments),
            "resolved_arguments" : json_safe(arguments),
        }


class ChunkFormerAdapter(BaseASRAdapter) :
    install_instructions = [
        "chunkformer",
        "huggingface_hub",
        "soundfile",
        "scipy",
    ]

    def load(self) -> None :
        import torch
        from chunkformer import ChunkFormerModel

        model_name = _model_name(self.specification)

        self.model = ChunkFormerModel.from_pretrained(model_name)

        if (hasattr(self.model, "to")) :
            device = (
                "cuda"
                if (
                    torch.cuda.is_available()
                    and self.specification.get("device", "cuda") == "cuda"
                )
                else "cpu"
            )
            self.model = self.model.to(device)

        if (hasattr(self.model, "eval")) :
            self.model.eval()

    def transcribe(self, window_audio : Path) -> dict[str, Any] :
        if (self.model is None) :
            raise RuntimeError("ChunkFormer model is not loaded.")

        arguments = {
            "audio_path"           : str(window_audio),
            "chunk_size"           : self.specification.get(
                "chunk_size",
                64,
            ),
            "left_context_size"    : self.specification.get(
                "left_context_size",
                128,
            ),
            "right_context_size"   : self.specification.get(
                "right_context_size",
                128,
            ),
            "total_batch_duration" : self.specification.get(
                "total_batch_duration",
                1_800,
            ),
            "return_timestamps"    : self.specification.get(
                "return_timestamps",
                True,
            ),
        }

        native_result = self.model.endless_decode(**arguments)
        text, segments = _extract_text_and_segments(native_result)

        # Some versions return timestamp records directly as a list.
        if (
            not text
            and isinstance(native_result, list)
        ) :
            text_parts = []

            for item in native_result :
                if (isinstance(item, dict)) :
                    item_text = str(
                        item.get(
                            "text",
                            item.get("transcription", ""),
                        )
                    ).strip()

                    if (item_text) :
                        text_parts.append(item_text)

            text = " ".join(text_parts).strip()

        return {
            "text"               : text,
            "native_result"      : json_safe(native_result),
            "native_segments"    : _canonical_segments(segments),
            "resolved_arguments" : json_safe(arguments),
        }


class MMSCTCAdapter(BaseASRAdapter) :
    install_instructions = [
        "transformers>=4.48,<5",
        "accelerate",
        "safetensors",
        "sentencepiece",
        "soundfile",
        "scipy",
    ]

    def load(self) -> None :
        import torch
        from transformers import AutoProcessor, Wav2Vec2ForCTC

        model_name  = _model_name(self.specification)
        revision    = self.specification.get("revision")
        target_lang = self.specification.get("target_lang", "vie")
        device      = (
            "cuda"
            if torch.cuda.is_available()
            and self.specification.get("device", "cuda") == "cuda"
            else "cpu"
        )

        self.processor = AutoProcessor.from_pretrained(
            model_name,
            revision=revision,
            target_lang=target_lang,
        )
        self.model = Wav2Vec2ForCTC.from_pretrained(
            model_name,
            revision=revision,
        )

        tokenizer = getattr(self.processor, "tokenizer", None)

        if (tokenizer is not None and hasattr(tokenizer, "set_target_lang")) :
            tokenizer.set_target_lang(target_lang)

        if (hasattr(self.model, "load_adapter")) :
            self.model.load_adapter(target_lang)

        self.model = self.model.to(device)
        self.model.eval()

    def transcribe(self, window_audio : Path) -> dict[str, Any] :
        if (self.model is None or self.processor is None) :
            raise RuntimeError("MMS model is not loaded.")

        import torch

        audio, sample_rate = _load_audio_mono(window_audio)
        device = next(self.model.parameters()).device

        inputs = self.processor(
            audio,
            sampling_rate=sample_rate,
            return_tensors="pt",
        )
        input_values = inputs.input_values.to(device)

        with torch.inference_mode() :
            logits = self.model(input_values).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        text = self.processor.batch_decode(predicted_ids)[0].strip()

        native_result = {
            "text"                  : text,
            "logit_frame_count"     : int(logits.shape[1]),
            "predicted_token_count" : int(predicted_ids.shape[1]),
        }

        return {
            "text"               : text,
            "native_result"      : native_result,
            "native_segments"    : [],
            "resolved_arguments" : {
                "target_lang" : self.specification.get("target_lang", "vie"),
                "sample_rate" : sample_rate,
            },
        }


class SeamlessM4TAdapter(BaseASRAdapter) :
    install_instructions = [
        "transformers>=4.48,<5",
        "accelerate",
        "safetensors",
        "sentencepiece",
        "protobuf",
        "soundfile",
        "scipy",
    ]

    def load(self) -> None :
        import torch
        from transformers import (
            AutoProcessor,
            SeamlessM4TForSpeechToText,
        )

        model_name = _model_name(self.specification)
        revision   = self.specification.get("revision")
        device     = (
            "cuda"
            if (
                torch.cuda.is_available()
                and self.specification.get("device", "cuda") == "cuda"
            )
            else "cpu"
        )
        model_dtype = _torch_dtype(
            self.specification,
            device,
        )

        self.processor = AutoProcessor.from_pretrained(
            model_name,
            revision=revision,
        )
        self.model = SeamlessM4TForSpeechToText.from_pretrained(
            model_name,
            revision=revision,
            dtype=model_dtype,
        )
        self.model = self.model.to(device)
        self.model.eval()

    def transcribe(self, window_audio : Path) -> dict[str, Any] :
        if (self.model is None or self.processor is None) :
            raise RuntimeError("SeamlessM4T model is not loaded.")

        import torch

        audio, sample_rate = _load_audio_mono(window_audio)
        first_parameter    = next(self.model.parameters())
        device             = first_parameter.device
        model_dtype        = first_parameter.dtype

        inputs = self.processor(
            audio=audio,
            sampling_rate=sample_rate,
            return_tensors="pt",
        )

        prepared_inputs = {}

        for key, value in inputs.items() :
            if (not hasattr(value, "to")) :
                prepared_inputs[key] = value
                continue

            value = value.to(device)

            if (torch.is_floating_point(value)) :
                value = value.to(dtype=model_dtype)

            prepared_inputs[key] = value

        arguments = {
            "tgt_lang" : self.specification.get(
                "target_lang",
                "vie",
            ),
            "num_beams" : self.specification.get(
                "num_beams",
                1,
            ),
        }

        with torch.inference_mode() :
            output_tokens = self.model.generate(
                **prepared_inputs,
                **arguments,
            )

        text = self.processor.batch_decode(
            output_tokens,
            skip_special_tokens=True,
        )[0].strip()

        native_result = {
            "text"               : text,
            "output_token_count" : int(output_tokens.shape[-1]),
        }

        return {
            "text"               : text,
            "native_result"      : native_result,
            "native_segments"    : [],
            "resolved_arguments" : {
                **arguments,
                "sample_rate" : sample_rate,
                "model_dtype" : str(model_dtype),
            },
        }


class NeMoASRAdapter(BaseASRAdapter) :
    install_instructions = ["Cython", "packaging", "nemo_toolkit[asr]"]

    def load(self) -> None :
        import nemo.collections.asr as nemo_asr

        model_name = _model_name(self.specification)
        self.model = nemo_asr.models.ASRModel.from_pretrained(
            model_name=model_name
        )
        self.model.eval()

        if (self.specification.get("device", "cuda") == "cuda") :
            self.model = self.model.cuda()

    def transcribe(self, window_audio : Path) -> dict[str, Any] :
        if (self.model is None) :
            raise RuntimeError("NeMo ASR model is not loaded.")

        arguments = {
            "batch_size"        : 1,
            "return_hypotheses" : True,
            "verbose"           : False,
        }
        native_result = self.model.transcribe(
            [str(window_audio)],
            **arguments,
        )
        hypothesis = (
            native_result[0]
            if isinstance(native_result, (list, tuple))
            else native_result
        )
        text = (
            hypothesis
            if isinstance(hypothesis, str)
            else getattr(hypothesis, "text", "")
        )
        timestamp = getattr(hypothesis, "timestamp", None)
        segments = []

        if (isinstance(timestamp, dict)) :
            for key in ["segment", "word", "char"] :
                if (isinstance(timestamp.get(key), list)) :
                    segments = timestamp[key]
                    break

        return {
            "text"               : str(text).strip(),
            "native_result"      : json_safe(native_result),
            "native_segments"    : _canonical_segments(segments),
            "resolved_arguments" : json_safe(arguments),
        }


class ParakeetCTCAdapter(BaseASRAdapter) :
    install_instructions = [
        "Cython",
        "packaging",
        "nemo_toolkit[asr]==2.7.3",
        "huggingface_hub",
        "soundfile",
        "scipy",
    ]

    def __init__(
        self,
        model_id : str,
        specification : dict[str, Any],
    ) :
        super().__init__(model_id, specification)
        self.local_model_path = None

    def load(self) -> None :
        import torch
        from huggingface_hub import hf_hub_download
        import nemo.collections.asr as nemo_asr

        version_parts = tuple(
            int(part)
            for part in torch.__version__.split("+")[0].split(".")[ : 2]
        )

        if (version_parts < (2, 7)) :
            raise RuntimeError(
                "NeMo 2.7.3 requires PyTorch 2.7 or newer. "
                f"Current version: {torch.__version__}."
            )

        model_name     = _model_name(self.specification)
        model_filename = self.specification.get(
            "model_filename",
            "parakeet-ctc-0.6b-vi.nemo",
        )
        revision = self.specification.get("revision")
        token = (
            self.specification.get("token")
            or os.environ.get("HF_TOKEN")
            or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        )

        self.local_model_path = Path(
            hf_hub_download(
                repo_id=model_name,
                filename=model_filename,
                revision=revision,
                token=token,
            )
        )

        device = torch.device(
            "cuda"
            if (
                torch.cuda.is_available()
                and self.specification.get("device", "cuda") == "cuda"
            )
            else "cpu"
        )

        self.model = nemo_asr.models.ASRModel.restore_from(
            restore_path=str(self.local_model_path),
            map_location=device,
        )
        self.model.eval()
        self.model = self.model.to(device)

    def resolved_revision(self) -> str | None :
        return (
            str(self.specification["revision"])
            if self.specification.get("revision")
            else super().resolved_revision()
        )

    def transcribe(self, window_audio : Path) -> dict[str, Any] :
        if (self.model is None) :
            raise RuntimeError("NeMo Parakeet model is not loaded.")

        arguments = {
            "audio"      : [str(window_audio)],
            "batch_size" : 1,
            "timestamps" : bool(
                self.specification.get("timestamps", True)
            ),
        }

        native_result = self.model.transcribe(
            **arguments,
        )
        hypothesis = (
            native_result[0]
            if isinstance(native_result, (list, tuple))
            else native_result
        )

        if (isinstance(hypothesis, str)) :
            text       = hypothesis
            timestamps = None
        else :
            text       = getattr(hypothesis, "text", "")
            timestamps = getattr(hypothesis, "timestamp", None)

        return {
            "text"          : str(text).strip(),
            "native_result" : {
                "text"      : str(text).strip(),
                "timestamp" : json_safe(timestamps),
            },
            "native_segments" : _canonical_nemo_timestamp_segments(
                timestamps
            ),
            "resolved_arguments" : {
                "batch_size" : 1,
                "timestamps" : arguments["timestamps"],
                "decoder"    : self.specification.get(
                    "decoder",
                    "greedy_ctc",
                ),
                "dtype"      : self.specification.get(
                    "dtype",
                    "float32",
                ),
            },
        }

    def close(self) -> None :
        self.local_model_path = None
        super().close()


def create_adapter(
    model_id : str,
    specification : dict[str, Any],
) -> BaseASRAdapter :
    adapter_name = _adapter_name(specification)
    adapter_map = {
        "whisper"      : WhisperAdapter,
        "hf_pipeline"  : HFPipelineASRAdapter,
        "chunkformer"  : ChunkFormerAdapter,
        "mms_ctc"      : MMSCTCAdapter,
        "seamless_m4t" : SeamlessM4TAdapter,
        "nemo"              : NeMoASRAdapter,
        "nemo_parakeet_ctc" : ParakeetCTCAdapter,
    }

    if (adapter_name not in adapter_map) :
        raise ValueError(f"Unsupported adapter: {adapter_name}")

    return adapter_map[adapter_name](model_id, specification)


def install_instructions_for(
    specification : dict[str, Any],
) -> list[str] :
    adapter_name = _adapter_name(specification)
    adapter_map = {
        "whisper"      : WhisperAdapter,
        "hf_pipeline"  : HFPipelineASRAdapter,
        "chunkformer"  : ChunkFormerAdapter,
        "mms_ctc"      : MMSCTCAdapter,
        "seamless_m4t" : SeamlessM4TAdapter,
        "nemo"              : NeMoASRAdapter,
        "nemo_parakeet_ctc" : ParakeetCTCAdapter,
    }

    if (adapter_name not in adapter_map) :
        raise ValueError(f"Unsupported adapter: {adapter_name}")

    default_install = adapter_map[adapter_name].install_instructions
    return list(specification.get("install", default_install))
