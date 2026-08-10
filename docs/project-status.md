# Project Status

Last updated: 2026-08-10

## Current Phase

Stage 1 ASR evaluation is complete.

The project has moved from ASR model selection to Retrieval v2 development.

## ASR Models Retained

- Whisper Large-v3
- NVIDIA Parakeet CTC 0.6B Vietnamese

No additional ASR model search is currently planned.

## Main Finding

ASR provides useful temporal evidence once the correct video is known, while
global video identification remains substantially harder.

The current bottleneck is therefore retrieval rather than transcription.

## Current Retrieval Baseline

Baseline v1 uses:

- query-fitted character TF-IDF
- `intfloat/multilingual-e5-small`
- fixed 50/50 lexical-semantic score averaging
- maximum-window video aggregation

This implementation is frozen as the historical baseline.

## Current Work

Retrieval v2.

Immediate milestone:

1. reproduce Baseline v1 exactly in the new retrieval framework
2. implement corpus-fitted TF-IDF
3. implement BM25 with accent-preserving and accent-folded representations
4. compare stronger multilingual dense retrievers
5. introduce rank-level fusion
6. improve video-level evidence aggregation

## Holdout Policy

The holdout20 query set must remain untouched until the Retrieval v2
architecture and hyperparameters are frozen.
