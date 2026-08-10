# Relative path: src/05_prepare_stage1_extension40.py
# Purpose: Freeze the 40-video extension benchmark, extract canonical WAVs, and build hashed manifests.

from __future__ import annotations

from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import wave
import zipfile
import xml.etree.ElementTree as ET


PROJECT_ROOT = Path(
    os.environ.get(
        "ASR_PROJECT_ROOT",
        "/content/drive/MyDrive/aic26/asr_model_comparison",
    )
)

SOURCE_DOCUMENT = Path(
    os.environ.get(
        "QUERY_SOURCE_DOCUMENT",
        "/content/drive/MyDrive/aic26/Documents/Đánh giá Query AIC2025.docx",
    )
)

INVENTORY_PATH = Path(
    os.environ.get(
        "VIDEO_INVENTORY_PATH",
        str(PROJECT_ROOT / "reports" / "full_video_inventory.csv"),
    )
)

CORE_BENCHMARK_PATH = PROJECT_ROOT / "data" / "benchmark_manifest.json"
OUTPUT_ROOT = PROJECT_ROOT / "data" / "stage1_extension40"
AUDIO_ROOT  = OUTPUT_ROOT / "audio"

SELECTED_CASES_PATH = OUTPUT_ROOT / "selected_cases.json"
WINDOWS_PATH        = OUTPUT_ROOT / "windows.json"
BENCHMARK_PATH      = OUTPUT_ROOT / "benchmark_manifest.json"
SELECTION_REPORT    = OUTPUT_ROOT / "selection_report.json"

FORCE_REBUILD_AUDIO = os.environ.get("FORCE_REBUILD_AUDIO", "0") == "1"

WINDOW_POLICY = {
    "sample_rate"          : 16_000,
    "window_seconds"       : 60.0,
    "stride_seconds"       : 45.0,
    "minimum_tail_seconds" : 15.0,
}

EXPECTED_VIDEO_COUNT  = 40
EXPECTED_QUERY_COUNT  = 40
EXPECTED_WINDOW_COUNT = 775

SELECTION = [
    # development20
    {"evaluation_split" : "development20", "query_id" : "R1-2",  "video_id" : "L21_V029", "task_type" : "KIS", "difficulty" : "easy"},
    {"evaluation_split" : "development20", "query_id" : "R1-9",  "video_id" : "L27_V013", "task_type" : "KIS", "difficulty" : "medium"},
    {"evaluation_split" : "development20", "query_id" : "R1-13", "video_id" : "L30_V095", "task_type" : "KIS", "difficulty" : "medium"},
    {"evaluation_split" : "development20", "query_id" : "R1-15", "video_id" : "L30_V072", "task_type" : "QA",  "difficulty" : "easy"},
    {"evaluation_split" : "development20", "query_id" : "R1-17", "video_id" : "L30_V092", "task_type" : "KIS", "difficulty" : "medium"},
    {"evaluation_split" : "development20", "query_id" : "R1-22", "video_id" : "L26_V178", "task_type" : "QA",  "difficulty" : "easy"},
    {"evaluation_split" : "development20", "query_id" : "R1-23", "video_id" : "L22_V022", "task_type" : "KIS", "difficulty" : "hard"},
    {"evaluation_split" : "development20", "query_id" : "R2-11", "video_id" : "L30_V014", "task_type" : "KIS", "difficulty" : "medium"},
    {"evaluation_split" : "development20", "query_id" : "R3-2",  "video_id" : "L29_V020", "task_type" : "KIS", "difficulty" : "medium"},
    {"evaluation_split" : "development20", "query_id" : "R3-15", "video_id" : "L26_V222", "task_type" : "KIS", "difficulty" : "medium"},
    {"evaluation_split" : "development20", "query_id" : "R2-1",  "video_id" : "K03_V019", "task_type" : "KIS", "difficulty" : "easy"},
    {"evaluation_split" : "development20", "query_id" : "R2-3",  "video_id" : "K17_V003", "task_type" : "QA",  "difficulty" : "medium"},
    {"evaluation_split" : "development20", "query_id" : "R2-5",  "video_id" : "K02_V005", "task_type" : "KIS", "difficulty" : "medium"},
    {"evaluation_split" : "development20", "query_id" : "R2-7",  "video_id" : "K05_V025", "task_type" : "KIS", "difficulty" : "easy"},
    {"evaluation_split" : "development20", "query_id" : "R2-20", "video_id" : "K06_V010", "task_type" : "KIS", "difficulty" : "medium"},
    {"evaluation_split" : "development20", "query_id" : "R2-24", "video_id" : "K05_V013", "task_type" : "KIS", "difficulty" : "medium"},
    {"evaluation_split" : "development20", "query_id" : "R2-8",  "video_id" : "K03_V023", "task_type" : "KIS", "difficulty" : "medium"},
    {"evaluation_split" : "development20", "query_id" : "R3-10", "video_id" : "K20_V009", "task_type" : "KIS", "difficulty" : "medium"},
    {"evaluation_split" : "development20", "query_id" : "R3-20", "video_id" : "K06_V022", "task_type" : "QA",  "difficulty" : "medium"},
    {"evaluation_split" : "development20", "query_id" : "R3-28", "video_id" : "K12_V007", "task_type" : "KIS", "difficulty" : "hard"},

    # holdout20
    {"evaluation_split" : "holdout20", "query_id" : "R1-7",  "video_id" : "L29_V023", "task_type" : "KIS", "difficulty" : "medium"},
    {"evaluation_split" : "holdout20", "query_id" : "R1-10", "video_id" : "L30_V017", "task_type" : "KIS", "difficulty" : "medium"},
    {"evaluation_split" : "holdout20", "query_id" : "R1-12", "video_id" : "L26_V200", "task_type" : "KIS", "difficulty" : "easy"},
    {"evaluation_split" : "holdout20", "query_id" : "R1-14", "video_id" : "L21_V027", "task_type" : "KIS", "difficulty" : "medium"},
    {"evaluation_split" : "holdout20", "query_id" : "R1-19", "video_id" : "L27_V010", "task_type" : "QA",  "difficulty" : "easy"},
    {"evaluation_split" : "holdout20", "query_id" : "R1-20", "video_id" : "L26_V004", "task_type" : "KIS", "difficulty" : "easy"},
    {"evaluation_split" : "holdout20", "query_id" : "R1-24", "video_id" : "L23_V007", "task_type" : "KIS", "difficulty" : "very_hard"},
    {"evaluation_split" : "holdout20", "query_id" : "R2-18", "video_id" : "L30_V040", "task_type" : "KIS", "difficulty" : "easy"},
    {"evaluation_split" : "holdout20", "query_id" : "R2-21", "video_id" : "L25_V058", "task_type" : "QA",  "difficulty" : "hard"},
    {"evaluation_split" : "holdout20", "query_id" : "R3-32", "video_id" : "L26_V444", "task_type" : "QA",  "difficulty" : "hard"},
    {"evaluation_split" : "holdout20", "query_id" : "R2-2",  "video_id" : "K14_V027", "task_type" : "KIS", "difficulty" : "medium"},
    {"evaluation_split" : "holdout20", "query_id" : "R2-15", "video_id" : "K01_V018", "task_type" : "KIS", "difficulty" : "medium"},
    {"evaluation_split" : "holdout20", "query_id" : "R3-7",  "video_id" : "K08_V019", "task_type" : "KIS", "difficulty" : "medium"},
    {"evaluation_split" : "holdout20", "query_id" : "R3-11", "video_id" : "K19_V022", "task_type" : "KIS", "difficulty" : "medium"},
    {"evaluation_split" : "holdout20", "query_id" : "R3-16", "video_id" : "K04_V021", "task_type" : "KIS", "difficulty" : "easy"},
    {"evaluation_split" : "holdout20", "query_id" : "R3-18", "video_id" : "K04_V013", "task_type" : "KIS", "difficulty" : "medium"},
    {"evaluation_split" : "holdout20", "query_id" : "R3-23", "video_id" : "K06_V014", "task_type" : "KIS", "difficulty" : "hard"},
    {"evaluation_split" : "holdout20", "query_id" : "R3-26", "video_id" : "K13_V003", "task_type" : "KIS", "difficulty" : "medium"},
    {"evaluation_split" : "holdout20", "query_id" : "R3-33", "video_id" : "K02_V007", "task_type" : "KIS", "difficulty" : "medium"},
    {"evaluation_split" : "holdout20", "query_id" : "R3-35", "video_id" : "K02_V012", "task_type" : "QA",  "difficulty" : "medium"},
]


def utc_now() -> str :
    return datetime.now(timezone.utc).isoformat()


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


def parse_docx_query_table(path : Path) -> dict[str, dict[str, str]] :
    namespace = {
        "w" : "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    }

    with zipfile.ZipFile(path, "r") as archive :
        xml = archive.read("word/document.xml")

    root = ET.fromstring(xml)
    query_rows = {}

    for row in root.findall(".//w:tr", namespace) :
        cells = []

        for cell in row.findall("./w:tc", namespace) :
            paragraphs = []

            for paragraph in cell.findall(".//w:p", namespace) :
                texts = [
                    node.text or ""
                    for node in paragraph.findall(".//w:t", namespace)
                ]
                paragraph_text = "".join(texts).strip()

                if (paragraph_text) :
                    paragraphs.append(paragraph_text)

            cells.append("\n".join(paragraphs).strip())

        if (not cells) :
            continue

        query_id = cells[0].strip()

        if (not re.fullmatch(r"R\d+-\d+", query_id)) :
            continue

        padded = cells + [""] * max(0, 6 - len(cells))

        query_rows[query_id] = {
            "query_id"      : query_id,
            "task_type"     : padded[1].strip(),
            "query_text"    : padded[2].strip(),
            "difficulty"    : padded[3].strip(),
            "answer"        : padded[4].strip(),
            "evaluation"    : padded[5].strip(),
        }

    return query_rows


def load_inventory(path : Path) -> dict[str, dict[str, str]] :
    rows = {}

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file :
        reader = csv.DictReader(file)

        for row in reader :
            video_name = str(row.get("video_name", "") or "").strip()
            video_id   = Path(video_name).stem

            if (video_id) :
                rows[video_id] = row

    return rows


def parse_answer(
    query_id : str,
    task_type : str,
    expected_video_id : str,
    answer : str,
) -> tuple[int, str | None] :
    parts = [
        part.strip()
        for part in answer.split(",", 2)
    ]

    if (len(parts) < 2) :
        raise RuntimeError(
            f"{query_id} does not contain a usable video/frame answer: {answer!r}"
        )

    answer_video_id = parts[0]
    frame_match = re.search(r"-?\d+", parts[1])

    if (answer_video_id != expected_video_id) :
        raise RuntimeError(
            f"{query_id} video mismatch: source={answer_video_id}, "
            f"selection={expected_video_id}."
        )

    if (frame_match is None) :
        raise RuntimeError(
            f"{query_id} does not contain a numeric frame ID: {answer!r}"
        )

    frame_id = int(frame_match.group(0))
    answer_text = None

    if (task_type == "QA") :
        if (len(parts) < 3 or not parts[2].strip()) :
            raise RuntimeError(
                f"{query_id} is QA but has no answer text: {answer!r}"
            )

        answer_text = parts[2].strip().strip('"').strip()

    return frame_id, answer_text


def parse_fraction(value : str | None) -> float | None :
    value = str(value or "").strip()

    if (not value or value == "0/0") :
        return None

    try :
        return float(Fraction(value))
    except (ValueError, ZeroDivisionError) :
        return None


def probe_video(path : Path) -> dict[str, Any] :
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_streams",
        "-show_format",
        "-of",
        "json",
        str(path),
    ]

    process = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )

    if (process.returncode != 0) :
        raise RuntimeError(
            f"FFprobe failed for {path}:\n{process.stderr.strip()}"
        )

    return json.loads(process.stdout)


def choose_streams(
    probe : dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]] :
    streams = probe.get("streams", [])

    video_streams = [
        stream
        for stream in streams
        if stream.get("codec_type") == "video"
    ]
    audio_streams = [
        stream
        for stream in streams
        if stream.get("codec_type") == "audio"
    ]

    if (not video_streams) :
        raise RuntimeError("No video stream found.")

    if (not audio_streams) :
        raise RuntimeError("No audio stream found.")

    video_stream = video_streams[0]
    audio_stream = sorted(
        audio_streams,
        key=lambda stream : int(
            stream.get("disposition", {}).get("default", 0)
        ),
        reverse=True,
    )[0]

    return video_stream, audio_stream


def resolve_fps(
    video_stream : dict[str, Any],
) -> tuple[float, str, list[str]] :
    warnings = []

    for source in [
        "avg_frame_rate",
        "r_frame_rate",
    ] :
        value = parse_fraction(
            video_stream.get(source)
        )

        if (value is not None and value > 0) :
            if (source != "avg_frame_rate") :
                warnings.append(
                    "avg_frame_rate_unavailable"
                )

            return value, source, warnings

    raise RuntimeError(
        "Could not resolve a positive video FPS."
    )


def canonical_audio_stream_metadata(
    stream : dict[str, Any],
) -> dict[str, Any] :
    keys = [
        "index",
        "codec_name",
        "codec_long_name",
        "profile",
        "codec_type",
        "codec_tag_string",
        "codec_tag",
        "sample_fmt",
        "sample_rate",
        "channels",
        "channel_layout",
        "bits_per_sample",
        "r_frame_rate",
        "avg_frame_rate",
        "time_base",
        "start_pts",
        "start_time",
        "duration_ts",
        "duration",
        "bit_rate",
        "nb_frames",
        "disposition",
        "tags",
    ]

    return {
        key : stream.get(key)
        for key in keys
        if key in stream
    }


def wav_info(path : Path) -> dict[str, int] :
    with wave.open(str(path), "rb") as wav :
        return {
            "channels"     : int(wav.getnchannels()),
            "sample_width" : int(wav.getsampwidth()),
            "sample_rate"  : int(wav.getframerate()),
            "sample_count" : int(wav.getnframes()),
        }


def extract_canonical_wav(
    video_id : str,
    source_video : Path,
    audio_stream_index : int,
    destination_wav : Path,
    source_size : int,
    source_mtime_ns : int,
) -> tuple[dict[str, Any], dict[str, Any]] :
    destination_meta = destination_wav.with_suffix(
        destination_wav.suffix + ".meta.json"
    )

    extraction = {
        "audio_stream_index" : audio_stream_index,
        "sample_rate"        : WINDOW_POLICY["sample_rate"],
        "channels"           : 1,
        "sample_width_bytes" : 2,
        "codec"              : "pcm_s16le",
    }

    if (
        not FORCE_REBUILD_AUDIO
        and destination_wav.exists()
        and destination_meta.exists()
    ) :
        existing_meta = json.loads(
            destination_meta.read_text(
                encoding="utf-8"
            )
        )

        expected = {
            "source_video"      : str(source_video),
            "source_size_bytes" : source_size,
            "source_mtime_ns"   : source_mtime_ns,
            "extraction"        : extraction,
        }

        if all(
            existing_meta.get(key) == value
            for key, value in expected.items()
        ) :
            information = wav_info(destination_wav)
            actual_hash = hash_file(destination_wav)

            if (
                information["channels"] == 1
                and information["sample_width"] == 2
                and information["sample_rate"] == WINDOW_POLICY["sample_rate"]
                and actual_hash == existing_meta.get("wav_sha256")
            ) :
                return information, existing_meta

    destination_wav.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    local_root = Path(
        tempfile.mkdtemp(
            prefix=f"asr_extension40_{video_id}_",
            dir="/content",
        )
    )
    local_wav = local_root / f"{video_id}.wav"

    command = [
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-y",
        "-i",
        str(source_video),
        "-map",
        f"0:{audio_stream_index}",
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(WINDOW_POLICY["sample_rate"]),
        "-c:a",
        "pcm_s16le",
        str(local_wav),
    ]

    try :
        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=900,
            check=False,
        )

        if (process.returncode != 0) :
            raise RuntimeError(
                f"FFmpeg failed for {video_id}:\n{process.stderr.strip()}"
            )

        information = wav_info(local_wav)

        if (
            information["channels"] != 1
            or information["sample_width"] != 2
            or information["sample_rate"] != WINDOW_POLICY["sample_rate"]
        ) :
            raise RuntimeError(
                f"Unexpected canonical WAV format for {video_id}: {information}"
            )

        local_hash = hash_file(local_wav)
        shutil.copy2(local_wav, destination_wav)

        final_hash = hash_file(destination_wav)

        if (final_hash != local_hash) :
            raise RuntimeError(
                f"WAV hash mismatch after copying {video_id} to Drive."
            )

        meta = {
            "video_id"          : video_id,
            "source_video"      : str(source_video),
            "source_size_bytes" : source_size,
            "source_mtime_ns"   : source_mtime_ns,
            "extraction"        : extraction,
            "wav_path"          : str(destination_wav),
            "wav_sha256"        : final_hash,
            "wav_size_bytes"    : destination_wav.stat().st_size,
            "wav_sample_rate"   : information["sample_rate"],
            "wav_sample_count"  : information["sample_count"],
            "created_at_utc"    : utc_now(),
        }
        write_json_atomic(
            destination_meta,
            meta,
        )

        return information, meta

    finally :
        shutil.rmtree(
            local_root,
            ignore_errors=True,
        )


def build_windows(
    video_id : str,
    sample_count : int,
) -> list[dict[str, Any]] :
    sample_rate = int(
        WINDOW_POLICY["sample_rate"]
    )
    window_samples = int(
        round(
            WINDOW_POLICY["window_seconds"]
            * sample_rate
        )
    )
    stride_samples = int(
        round(
            WINDOW_POLICY["stride_seconds"]
            * sample_rate
        )
    )
    minimum_tail_samples = int(
        round(
            WINDOW_POLICY["minimum_tail_seconds"]
            * sample_rate
        )
    )

    windows = []
    sample_start = 0
    window_index = 0

    while (sample_start < sample_count) :
        remaining = sample_count - sample_start

        if (
            window_index > 0
            and remaining < minimum_tail_samples
        ) :
            break

        duration_samples = min(
            window_samples,
            remaining,
        )
        sample_end = (
            sample_start
            + duration_samples
        )

        windows.append({
            "window_id"        : f"{video_id}_{window_index:04d}",
            "video_id"         : video_id,
            "window_index"     : window_index,
            "start_s"          : sample_start / sample_rate,
            "end_s"            : sample_end / sample_rate,
            "sample_start"     : sample_start,
            "sample_end"       : sample_end,
            "duration_samples" : duration_samples,
        })

        sample_start += stride_samples
        window_index += 1

    return windows


def classify_query(text : str) -> str :
    normalized = str(text or "").lower()

    if re.search(
        r"\b\d+(?:[.,]\d+)?\s*(?:kg|km|g|%|ml|giờ|phút|con|người)?\b",
        normalized,
    ) :
        return "number_or_measurement"

    if any(
        token in normalized
        for token in [
            "địa điểm",
            "tỉnh",
            "xã ",
            "thành phố",
            "nghệ sĩ",
            "tổng thống",
            "người phụ nữ",
            "người đàn ông",
            "câu lạc bộ",
        ]
    ) :
        return "person_organization_location"

    return "other"


def main() -> None :
    for path in [
        SOURCE_DOCUMENT,
        INVENTORY_PATH,
        CORE_BENCHMARK_PATH,
    ] :
        if (not path.exists()) :
            raise FileNotFoundError(path)

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )
    AUDIO_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    if (len(SELECTION) != EXPECTED_VIDEO_COUNT) :
        raise RuntimeError(
            f"Selection contains {len(SELECTION)} rows, "
            f"expected {EXPECTED_VIDEO_COUNT}."
        )

    query_ids = [
        row["query_id"]
        for row in SELECTION
    ]
    video_ids = [
        row["video_id"]
        for row in SELECTION
    ]

    if (len(set(query_ids)) != EXPECTED_QUERY_COUNT) :
        raise RuntimeError(
            "Extension selection does not contain 40 unique query IDs."
        )

    if (len(set(video_ids)) != EXPECTED_VIDEO_COUNT) :
        raise RuntimeError(
            "Extension selection does not contain 40 unique video IDs."
        )

    core_benchmark = json.loads(
        CORE_BENCHMARK_PATH.read_text(
            encoding="utf-8"
        )
    )
    core_video_ids = set(
        core_benchmark["selected_video_ids"]
    )
    overlap = sorted(
        core_video_ids
        & set(video_ids)
    )

    if (overlap) :
        raise RuntimeError(
            f"Extension overlaps core10: {overlap}"
        )

    query_catalog = parse_docx_query_table(
        SOURCE_DOCUMENT
    )
    inventory = load_inventory(
        INVENTORY_PATH
    )

    missing_queries = [
        query_id
        for query_id in query_ids
        if query_id not in query_catalog
    ]
    missing_videos = [
        video_id
        for video_id in video_ids
        if video_id not in inventory
    ]

    if (missing_queries) :
        raise RuntimeError(
            f"Selected queries missing from source document: {missing_queries}"
        )

    if (missing_videos) :
        raise RuntimeError(
            f"Selected videos missing from inventory: {missing_videos}"
        )

    bad_inventory = []

    for video_id in video_ids :
        status = str(
            inventory[video_id].get(
                "probe_status",
                "",
            )
            or ""
        ).strip().lower()

        if (status != "ok") :
            bad_inventory.append(
                {
                    "video_id" : video_id,
                    "status"   : status,
                    "error"    : inventory[video_id].get(
                        "probe_error"
                    ),
                }
            )

    if (bad_inventory) :
        raise RuntimeError(
            "Selected extension videos must all have probe_status=ok:\n"
            + json.dumps(
                bad_inventory,
                ensure_ascii=False,
                indent=2,
            )
        )

    selection_hash = hash_json(
        SELECTION
    )
    source_document_hash = hash_file(
        SOURCE_DOCUMENT
    )
    window_policy_hash = hash_json(
        WINDOW_POLICY
    )

    selected_videos = []
    all_windows     = []
    benchmark_queries = []
    audio_files = []

    print("\n" + "=" * 88)
    print("PREPARING STAGE 1 EXTENSION40")
    print("=" * 88)
    print(f"Source document: {SOURCE_DOCUMENT}")
    print(f"Inventory:       {INVENTORY_PATH}")
    print(f"Output root:     {OUTPUT_ROOT}")
    print(f"Videos:          {len(SELECTION)}")
    print(f"Force audio:     {FORCE_REBUILD_AUDIO}")

    for position, selection in enumerate(
        SELECTION,
        start=1,
    ) :
        query_id = selection["query_id"]
        video_id = selection["video_id"]
        source_query = query_catalog[query_id]
        inventory_row = inventory[video_id]

        source_task_type = source_query["task_type"].strip().upper()

        if (source_task_type != selection["task_type"]) :
            raise RuntimeError(
                f"{query_id} task mismatch: "
                f"source={source_task_type}, selection={selection['task_type']}."
            )

        source_video = Path(
            inventory_row["video_path"]
        )

        if (not source_video.exists()) :
            raise FileNotFoundError(source_video)

        frame_id, answer_text = parse_answer(
            query_id=query_id,
            task_type=selection["task_type"],
            expected_video_id=video_id,
            answer=source_query["answer"],
        )

        probe = probe_video(
            source_video
        )
        video_stream, audio_stream = choose_streams(
            probe
        )
        fps, fps_source, fps_warnings = resolve_fps(
            video_stream
        )

        source_stat = source_video.stat()
        destination_wav = (
            AUDIO_ROOT
            / f"{video_id}.wav"
        )

        wav_information, wav_meta = extract_canonical_wav(
            video_id=video_id,
            source_video=source_video,
            audio_stream_index=int(audio_stream["index"]),
            destination_wav=destination_wav,
            source_size=int(source_stat.st_size),
            source_mtime_ns=int(source_stat.st_mtime_ns),
        )

        windows = build_windows(
            video_id=video_id,
            sample_count=int(
                wav_information["sample_count"]
            ),
        )

        expected_inventory_windows = int(
            float(
                inventory_row["window_count"]
            )
        )

        if (len(windows) != expected_inventory_windows) :
            raise RuntimeError(
                f"{video_id} generated {len(windows)} windows, "
                f"inventory expected {expected_inventory_windows}."
            )

        wav_meta_path = destination_wav.with_suffix(
            destination_wav.suffix + ".meta.json"
        )
        wav_sha256 = hash_file(
            destination_wav
        )
        wav_meta_sha256 = hash_file(
            wav_meta_path
        )

        query_record = {
            "query_id"         : query_id,
            "query_text"       : source_query["query_text"],
            "original_answer"  : source_query["answer"],
            "frame_id"         : frame_id,
            "answer_time_s"    : frame_id / fps,
            "query_category"   : classify_query(
                source_query["query_text"]
            ),
            "video_id"         : video_id,
            "task_type"        : selection["task_type"],
            "difficulty"       : selection["difficulty"],
            "evaluation_split" : selection["evaluation_split"],
            "answer_text"      : answer_text,
        }

        video_record = {
            "video_id"              : video_id,
            "mp4_path"              : str(source_video),
            "mp4_size"              : int(source_stat.st_size),
            "mp4_mtime_ns"          : int(source_stat.st_mtime_ns),
            "audio_stream_index"    : int(audio_stream["index"]),
            "audio_stream_metadata" : canonical_audio_stream_metadata(
                audio_stream
            ),
            "duration_s"            : float(
                probe.get(
                    "format",
                    {},
                ).get(
                    "duration",
                    inventory_row["duration_s"],
                )
            ),
            "fps"                   : fps,
            "fps_source"            : fps_source,
            "fps_warning_reasons"   : fps_warnings,
            "fps_confirmed"         : False,
            "wav_path"              : str(
                destination_wav
            ),
            "wav_meta_path"         : str(
                wav_meta_path
            ),
            "previously_inspected"  : False,
            "evaluation_split"      : selection["evaluation_split"],
            "queries"               : [query_record],
            "wav_hash"              : wav_sha256,
            "wav_sample_rate"       : int(
                wav_information["sample_rate"]
            ),
            "wav_sample_count"      : int(
                wav_information["sample_count"]
            ),
            "windows"               : windows,
        }

        relative_wav = destination_wav.relative_to(
            PROJECT_ROOT
        )
        relative_meta = wav_meta_path.relative_to(
            PROJECT_ROOT
        )

        audio_files.append({
            "video_id"         : video_id,
            "wav_path"         : str(relative_wav),
            "wav_sha256"       : wav_sha256,
            "wav_size_bytes"   : destination_wav.stat().st_size,
            "wav_sample_rate"  : int(
                wav_information["sample_rate"]
            ),
            "wav_sample_count" : int(
                wav_information["sample_count"]
            ),
            "wav_meta_path"    : str(relative_meta),
            "wav_meta_sha256"  : wav_meta_sha256,
        })

        selected_videos.append(
            video_record
        )
        benchmark_queries.append(
            query_record
        )
        all_windows.extend(
            windows
        )

        print(
            f"[{position:02d}/{EXPECTED_VIDEO_COUNT}] "
            f"{query_id} → {video_id}: "
            f"{len(windows)} windows, "
            f"{wav_information['sample_count'] / WINDOW_POLICY['sample_rate']:.1f}s WAV"
        )

    if (len(all_windows) != EXPECTED_WINDOW_COUNT) :
        raise RuntimeError(
            f"Generated {len(all_windows)} windows, "
            f"expected {EXPECTED_WINDOW_COUNT}."
        )

    created_at = utc_now()

    selected_cases = {
        "version"              : "1.2",
        "source_document"      : str(SOURCE_DOCUMENT),
        "source_document_hash" : source_document_hash,
        "supervision_level"    : "silver",
        "created_at"           : created_at,
        "benchmark_subset"     : "extension40",
        "replacement_note"     : (
            "R3-8 / K16_V004 was replaced by "
            "R2-8 / K03_V023 after the former failed the inventory probe."
        ),
        "videos"               : selected_videos,
        "selection_hash"       : selection_hash,
        "window_policy"        : WINDOW_POLICY,
        "window_policy_hash"   : window_policy_hash,
    }
    write_json_atomic(
        SELECTED_CASES_PATH,
        selected_cases,
    )

    windows_payload = {
        "window_policy"      : WINDOW_POLICY,
        "window_policy_hash" : window_policy_hash,
        "windows"            : all_windows,
    }
    write_json_atomic(
        WINDOWS_PATH,
        windows_payload,
    )

    selected_cases_sha256 = hash_file(
        SELECTED_CASES_PATH
    )
    windows_file_sha256 = hash_file(
        WINDOWS_PATH
    )
    window_manifest_hash = hash_json(
        all_windows
    )

    benchmark = {
        "stage_id"                  : "stage1_extension40",
        "benchmark_role"            : "expanded_multi_model_asr_comparison",
        "supervision_level"         : "silver",
        "source_document"           : str(SOURCE_DOCUMENT),
        "source_document_hash"      : source_document_hash,
        "selection_hash"            : selection_hash,
        "selected_cases_sha256"     : selected_cases_sha256,
        "windows_file_sha256"       : windows_file_sha256,
        "video_count"               : len(selected_videos),
        "query_count"               : len(benchmark_queries),
        "window_count"              : len(all_windows),
        "selected_video_ids"        : sorted(video_ids),
        "selected_query_ids"        : sorted(query_ids),
        "evaluation_splits"         : {
            "development20" : sorted(
                row["video_id"]
                for row in SELECTION
                if row["evaluation_split"] == "development20"
            ),
            "holdout20" : sorted(
                row["video_id"]
                for row in SELECTION
                if row["evaluation_split"] == "holdout20"
            ),
        },
        "queries"                   : benchmark_queries,
        "window_policy"             : WINDOW_POLICY,
        "window_policy_hash"        : window_policy_hash,
        "window_manifest_hash"      : window_manifest_hash,
        "audio_files"               : audio_files,
        "windows"                   : all_windows,
    }

    benchmark_content_hash = hash_json(
        benchmark
    )
    benchmark["benchmark_content_hash"] = (
        benchmark_content_hash
    )
    benchmark["frozen_at_utc"] = utc_now()

    write_json_atomic(
        BENCHMARK_PATH,
        benchmark,
    )

    split_counts = {}

    for row in SELECTION :
        key = row["evaluation_split"]
        split_counts[key] = (
            split_counts.get(key, 0)
            + 1
        )

    source_duration_s = sum(
        float(video["duration_s"])
        for video in selected_videos
    )
    windowed_audio_s = sum(
        window["duration_samples"]
        / WINDOW_POLICY["sample_rate"]
        for window in all_windows
    )

    selection_report = {
        "stage_id"                : "stage1_extension40",
        "status"                  : "PASS",
        "video_count"             : len(selected_videos),
        "query_count"             : len(benchmark_queries),
        "window_count"            : len(all_windows),
        "split_counts"            : split_counts,
        "k_video_count"           : sum(
            video_id.startswith("K")
            for video_id in video_ids
        ),
        "l_video_count"           : sum(
            video_id.startswith("L")
            for video_id in video_ids
        ),
        "source_duration_s"       : source_duration_s,
        "source_duration_h"       : source_duration_s / 3600.0,
        "windowed_audio_s"        : windowed_audio_s,
        "windowed_audio_h"        : windowed_audio_s / 3600.0,
        "selection_hash"          : selection_hash,
        "window_policy_hash"      : window_policy_hash,
        "window_manifest_hash"    : window_manifest_hash,
        "benchmark_content_hash"  : benchmark_content_hash,
        "selected_cases_sha256"   : selected_cases_sha256,
        "windows_file_sha256"     : windows_file_sha256,
        "benchmark_manifest_path" : str(BENCHMARK_PATH),
        "replacement"             : {
            "removed" : {
                "query_id" : "R3-8",
                "video_id" : "K16_V004",
            },
            "added" : {
                "query_id" : "R2-8",
                "video_id" : "K03_V023",
            },
        },
        "created_at_utc" : utc_now(),
    }
    write_json_atomic(
        SELECTION_REPORT,
        selection_report,
    )

    print("\n" + "=" * 88)
    print("STAGE 1 EXTENSION40 FROZEN")
    print("=" * 88)
    print(f"Videos:             {len(selected_videos)}")
    print(f"Queries:            {len(benchmark_queries)}")
    print(f"Windows:            {len(all_windows)}")
    print(f"Source duration:    {source_duration_s / 3600.0:.3f} h")
    print(f"Windowed audio:     {windowed_audio_s / 3600.0:.3f} h")
    print(f"Selection hash:     {selection_hash}")
    print(f"Benchmark hash:     {benchmark_content_hash}")
    print(f"Selected cases:     {SELECTED_CASES_PATH}")
    print(f"Windows:            {WINDOWS_PATH}")
    print(f"Benchmark manifest: {BENCHMARK_PATH}")
    print(f"Selection report:   {SELECTION_REPORT}")


if (__name__ == "__main__") :
    main()
