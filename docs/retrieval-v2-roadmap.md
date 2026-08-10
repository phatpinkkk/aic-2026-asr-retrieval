# Retrieval v2 Roadmap

## Goal

Improve global video retrieval while preserving strong within-video temporal
localization.

The Stage 1 retrieval implementation remains frozen as Baseline v1.

## Stage 0 – Baseline Reproduction

Question:
Can Retrieval v2 reproduce Baseline v1 exactly?

Work:

- reproduce existing TF-IDF + E5 scoring
- reproduce raw and processed transcript views
- reproduce video and story ranks
- create a Retrieval v2 regression gate

Gate:
Current development metrics and per-query ranks must match.

## Stage 1 – Lexical Retrieval

Compare:

- old query-fitted character TF-IDF
- corpus-fitted character TF-IDF
- BM25 with Vietnamese accents
- accent-folded BM25
- rank fusion of both BM25 representations

Select one lexical configuration.

## Stage 2 – Dense Retrieval

Compare:

- multilingual E5-small
- Qwen3-Embedding-0.6B
- BGE-M3 dense retrieval

Cache corpus embeddings and pin exact model revisions.

Select one primary dense retriever.

## Stage 2B – BGE-M3 Retrieval Modes

If BGE-M3 remains competitive, evaluate:

- dense
- learned sparse
- multi-vector
- controlled rank fusion

## Stage 3 – Hybrid Retrieval

Combine the selected lexical and dense systems.

Evaluate:

- Reciprocal Rank Fusion
- a small predefined normalized-score fusion grid

Avoid extensive tuning on development20.

## Stage 4 – Video Evidence Aggregation

Replace fragile maximum-window scoring.

Compare:

- maximum
- top-2 mean
- top-3 mean
- temporally diverse top-2
- temporally diverse top-3

## Stage 5 – ASR Channel Fusion

Treat these as separate retrieval channels:

- Whisper raw
- Whisper processed
- Parakeet raw
- Parakeet processed

Evaluate within-model and cross-model rank fusion.

## Stage 6 – Hierarchical Retrieval

Use:

query
→ candidate video retrieval
→ top-K videos
→ within-video temporal retrieval

Track candidate-video Recall@5, Recall@10, and Recall@20.

## Stage 7 – Second-Stage Reranking

Begin with Qwen3-Reranker-0.6B.

Rerank only first-stage candidates rather than the full corpus.

## Stage 8 – Query Variants

Evaluate controlled query representations:

- original query
- deterministic content-focused query
- entity / location / number-focused query
- later ASR-oriented query reformulation

The original query always remains available as a retrieval channel.

## Stage 9 – Final Text Retrieval Ablation

Produce one additive ablation table showing the contribution of each selected
component.

## Stage 10 – Freeze and Holdout

Freeze:

- source hashes
- config
- model revisions
- query instructions
- fusion policy
- pooling policy
- selected ASR channels
- candidate K
- reranker
- query processing

Then evaluate holdout20 once.
