# Relative path: src/postprocess.py
# Purpose: Conservative, query-independent transcript normalization and warning rules.

from __future__ import annotations

import re
import unicodedata
from typing import Any


POSTPROCESS_VERSION = "1.1"
BOILERPLATE_DOMINANCE_THRESHOLD = 0.80

KNOWN_BOILERPLATE = [
    "hãy subscribe cho kênh ghiền mì gõ để không bỏ lỡ những video hấp dẫn",
    "các bạn hãy đăng ký kênh để ủng hộ kênh của chúng mình nhé",
    "đừng quên đăng ký kênh để không bỏ lỡ những video hấp dẫn",
    "hãy đăng ký kênh để không bỏ lỡ những video hấp dẫn",
]

KNOWN_BOILERPLATE_SIGNATURES = [
    "hãy subscribe cho kênh ghiền mì gõ",
    "các bạn hãy đăng ký kênh để ủng hộ kênh của chúng mình",
    "đừng quên đăng ký kênh để không bỏ lỡ những video hấp dẫn",
    "hãy đăng ký kênh để không bỏ lỡ những video hấp dẫn",
]

UNIT_PATTERNS = [
    (r"\bki\s*l[oô]\s*gam\b", "kg"),
    (r"\bkilogram\b", "kg"),
    (r"\bkilograms\b", "kg"),
    (r"\bki\s*l[oô]\b", "kg"),
    (r"(?<=\d)\s*k[ýy]\b", " kg"),
]


def normalize_whitespace(text : str) -> str :
    return re.sub(r"\s+", " ", str(text or "")).strip()


def normalize_unicode(text : str) -> str :
    return unicodedata.normalize("NFC", normalize_whitespace(text))


def normalize_for_matching(text : str) -> str :
    text = normalize_unicode(text).lower()
    text = re.sub(r"[^0-9a-zA-ZÀ-ỹđĐ\s]", " ", text)
    return normalize_whitespace(text)


def accent_fold(text : str) -> str :
    text = normalize_for_matching(text).replace("đ", "d").replace("Đ", "D")
    decomposed = unicodedata.normalize("NFD", text)
    return normalize_whitespace("".join(char for char in decomposed if unicodedata.category(char) != "Mn"))


def apply_unit_aliases(text : str) -> str :
    aliased = normalize_for_matching(text)

    for pattern, replacement in UNIT_PATTERNS :
        aliased = re.sub(pattern, replacement, aliased, flags=re.IGNORECASE)

    return normalize_whitespace(aliased)


def _segment_text(segment : dict[str, Any]) -> str :
    return normalize_unicode(str(segment.get("text", "")))


def suppress_consecutive_duplicate_segments(segments : list[dict[str, Any]]) -> tuple[str, list[str]] :
    kept       = []
    warnings   = []
    last_match = None

    for segment in segments :
        text       = _segment_text(segment)
        match_text = accent_fold(text)

        if (match_text and match_text == last_match) :
            warnings.append("consecutive_duplicate_segment")
            continue

        if (text) :
            kept.append(text)
            last_match = match_text

    return normalize_whitespace(" ".join(kept)), sorted(set(warnings))


def _merge_spans(spans : list[tuple[int, int]]) -> list[tuple[int, int]] :
    merged = []

    for start, end in sorted(spans) :
        if (not merged or start > merged[-1][1]) :
            merged.append([start, end])
        else :
            merged[-1][1] = max(merged[-1][1], end)

    return [(start, end) for start, end in merged]


def find_boilerplate_matches(text : str) -> dict[str, Any] :
    normalized = accent_fold(text)
    matches    = []
    phrases    = list(dict.fromkeys(KNOWN_BOILERPLATE + KNOWN_BOILERPLATE_SIGNATURES))

    for phrase in phrases :
        candidate = accent_fold(phrase)
        if (not candidate) :
            continue

        for match in re.finditer(re.escape(candidate), normalized) :
            matches.append({"phrase" : phrase, "start" : match.start(), "end" : match.end()})

    merged_spans       = _merge_spans([(item["start"], item["end"]) for item in matches])
    denominator        = sum(not char.isspace() for char in normalized)
    matched_characters = sum(sum(not char.isspace() for char in normalized[start : end]) for start, end in merged_spans)
    coverage           = matched_characters / denominator if denominator else 0.0

    return {
        "normalized_text"    : normalized,
        "matches"            : matches,
        "merged_spans"       : merged_spans,
        "matched_characters" : matched_characters,
        "total_characters"   : denominator,
        "coverage"           : coverage,
    }


def remove_matching_spans(text : str, spans : list[tuple[int, int]]) -> str :
    characters = list(text)

    for start, end in spans :
        for index in range(max(0, start), min(len(characters), end)) :
            characters[index] = " "

    return normalize_whitespace("".join(characters))


def postprocess_window(window : dict[str, Any]) -> dict[str, Any] :
    raw_text        = str(window.get("raw_text", ""))
    native_segments = window.get("native_segments", []) or []
    warning_reasons = list(window.get("warning_reasons", []) or [])
    rejection_reasons = list(window.get("rejection_reasons", []) or [])

    deduplicated_text, segment_warnings = suppress_consecutive_duplicate_segments(native_segments)
    source_text = deduplicated_text or normalize_unicode(raw_text)

    normalized_text    = normalize_for_matching(source_text)
    accent_folded_text = accent_fold(source_text)
    retrieval_text     = apply_unit_aliases(source_text)

    warning_reasons.extend(segment_warnings)

    boilerplate = find_boilerplate_matches(source_text)
    coverage    = float(boilerplate["coverage"])

    if (boilerplate["matches"] and coverage >= BOILERPLATE_DOMINANCE_THRESHOLD) :
        warning_reasons.append("known_boilerplate")
        rejection_reasons.append("dominant_known_boilerplate")
        retrieval_text = ""
    elif (boilerplate["matches"]) :
        warning_reasons.append("partial_known_boilerplate")
        remaining_text = remove_matching_spans(normalized_text, boilerplate["merged_spans"])
        retrieval_text = apply_unit_aliases(remaining_text)

        if (not retrieval_text) :
            rejection_reasons.append("known_boilerplate_only_after_removal")

    if (not normalize_whitespace(raw_text)) :
        warning_reasons.append("empty_output")

    return {
        "normalized_text"    : normalized_text,
        "accent_folded_text" : accent_folded_text,
        "retrieval_text"     : retrieval_text,
        "warning_reasons"    : sorted(set(warning_reasons)),
        "rejection_reasons"  : sorted(set(rejection_reasons)),
        "boilerplate_coverage": coverage,
        "boilerplate_matches" : boilerplate["matches"],
        "postprocess_version": POSTPROCESS_VERSION,
    }


def mark_consecutive_duplicate_windows(windows : list[dict[str, Any]]) -> list[dict[str, Any]] :
    ordered    = sorted(windows, key=lambda item : (item.get("start_s", 0.0), item.get("window_id", "")))
    last_text  = None

    for window in ordered :
        current = accent_fold(window.get("retrieval_text", ""))

        if (current and len(current) >= 30 and current == last_text) :
            warnings   = list(window.get("warning_reasons", []) or [])
            rejections = list(window.get("rejection_reasons", []) or [])
            warnings.append("consecutive_duplicate_window")
            rejections.append("consecutive_duplicate_window")

            window["warning_reasons"]   = sorted(set(warnings))
            window["rejection_reasons"] = sorted(set(rejections))
            window["retrieval_text"]    = ""

        if (current) :
            last_text = current

    return ordered
