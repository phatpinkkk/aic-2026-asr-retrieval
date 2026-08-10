# Evaluation Protocol

## Purpose

This document defines the fixed evaluation protocol used for ASR-assisted text
retrieval experiments.

Changes to this protocol must be deliberate and documented in
`docs/decisions.md`.

## Benchmark Partitions

### Core10

- 10 videos
- 10 queries
- 219 ASR windows

Core10 is used primarily for regression and implementation sanity checks.

### Extension40

- 40 additional videos
- 40 queries
- 775 ASR windows

The query set is divided into development20 and holdout20.

### All50 Retrieval Corpus

Core10 and Extension40 together form the full retrieval corpus:

- 50 videos
- 50 queries
- 994 ASR windows

## Development

Architecture and hyperparameter decisions are made using `development20`.

Each development query is retrieved against the full All50 corpus.

Core10 may continue to be used as a regression set.

## Holdout

`holdout20` must not be used for iterative development or model selection.

It is evaluated only after the retrieval configuration has been frozen.

## Retrieval Metrics

Video-level metrics:

- Recall@1
- Recall@3
- Recall@5
- Recall@10
- Recall@20
- MRR

Story-level metrics:

- Recall@1
- Recall@3
- Recall@5
- Recall@10
- MRR

Per-query ranks and better/tie/worse comparisons should also be retained.

## ASR Reference Policy

Manually verified ground-truth transcripts are not currently available.

Whisper Large-v3 outputs are therefore frozen as reference transcripts for
candidate-ASR agreement diagnostics.

WER and CER against this reference measure agreement with Whisper, not absolute
transcription accuracy.

Retrieval metrics, reliability diagnostics, and manual inspection must be
considered alongside transcript-agreement measurements.
