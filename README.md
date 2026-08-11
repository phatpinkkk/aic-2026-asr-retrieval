# AIC 2026 ASR & Retrieval Research

This repository contains the research and implementation of the ASR-based text retrieval subsystem used in the AIC 2026 video-search pipeline. The subsystem makes spoken video content searchable: it transcribes short audio windows, retrieves relevant transcript passages with lexical and semantic search, ranks videos from those passage scores, and can rerank a small candidate set with a stronger cross-encoder.

The production video-search application is maintained separately in `aic-2026-pipeline`. This repository is focused on controlled retrieval experiments, reproducible implementation, and the evidence used to choose the current design.

## 1. Current System

The current pipeline has two parts. Corpus-side processing is done before search, while query-dependent scoring happens online.

```text
OFFLINE

video audio
    ↓
60 s audio windows, 45 s stride
    ↓
Parakeet CTC 0.6B Vietnamese
    ↓
conservative transcript post-processing
    ↓
processed retrieval text
    ↓
BM25 index + E5 document embeddings


ONLINE

natural-language query
    ↓
BM25 + multilingual E5-large
    ↓
per-query score normalization
    ↓
25% BM25 + 75% E5
    ↓
best matching window → first-stage video ranking
    ↓
candidate selection
    ↓
cross-encoder reranking
    ↓
50% first-stage + 50% reranker score
    ↓
ranked videos + supporting transcript moments
```

A full video is not transcribed first and divided into text afterward. The benchmark defines fixed temporal windows, and the inference runner reads the exact audio samples for each window before sending that window to ASR. The current window policy is 60 seconds with a 45-second stride, giving 15 seconds of overlap.

The decisions through Stage 6 are frozen. Stage 7 evaluation is complete and passed, but the final Stage 7 selection is still marked `pending_review` in the current configuration. BGE reranker v2 M3 with the predefined 50/50 first-stage/reranker fusion is the current operational recommendation.

| Component | Current choice |
|---|---|
| ASR | NVIDIA Parakeet CTC 0.6B Vietnamese |
| Audio window | 60 s |
| Window stride | 45 s |
| Transcript representation | Processed |
| Lexical retrieval | Accent-preserving BM25 |
| Dense retrieval | `intfloat/multilingual-e5-large-instruct` |
| First-stage fusion | 25% normalized BM25 + 75% normalized E5 |
| Video score | Best matching transcript window |
| Stage 6 benchmark candidate policy | Top 30 videos × up to 5 windows/video |
| Stage 7 score policy | 50% normalized first-stage + 50% normalized reranker |
| Stage 7 operational recommendation | BGE reranker v2 M3, pending final freeze |

The main Full40 results are:

| System | Video R@1 | Video MRR | Story MRR | Reranking p90 |
|---|---:|---:|---:|---:|
| First stage only | 0.725 | 0.8110 | 0.9042 | — |
| BGE + first-stage fusion | 0.750 | 0.8205 | **0.9125** | 2.54 s |
| Qwen3-0.6B + first-stage fusion | **0.825** | **0.8642** | 0.8625 | 7.31 s |

Qwen produced the strongest video-ranking result, but its p90 reranking time exceeded the 5-second limit that was fixed before Stage 7. BGE gave a much smaller video improvement, but it improved both Video and Story MRR while remaining within the latency limit. The no-reranker first stage is still a competitive option when latency matters.

Stage 6 should also be interpreted carefully. `K30_M5` was selected on the 50-video All50 benchmark, where it retained every correct video and a relevant window for 39 of 40 queries. It is **not** assumed to be optimal for the much larger competition corpus. Candidate-video recall must be measured again before using the same `K=30` at full scale.

## 2. Repository and Documentation

The repository keeps source code and compact metadata in Git while large generated artifacts remain outside the repository.

```text
configs/      experiment and retrieval configuration
data/         compact benchmark and reference metadata
notebooks/    stage-specific experiment notebooks
src/          ASR, retrieval, reranking, and evaluation code
docs/         maintained technical documentation
reports/      generated reports; local/external and ignored by Git
```

There are only two maintained technical documents in addition to this README:

- [`docs/asr-retrieval-system.md`](docs/asr-retrieval-system.md) explains **how the current system works**. Read it for audio preparation, ASR, transcript post-processing, BM25, E5, candidate construction, reranking, runtime, scaling, and implementation boundaries.
- [`docs/retrieval-v2.md`](docs/retrieval-v2.md) explains **why the system has this design**. It documents the benchmark, evaluation chronology, Stages 1–7, results, decisions, failure patterns, and remaining uncertainty.

Older rolling progress documentation should not be treated as current. Its stable findings are incorporated into `docs/retrieval-v2.md`.

## 3. Development and Evaluation Status

Retrieval v2 was developed in controlled stages. Each stage changed one major part of the system and then froze the chosen result before the next stage depended on it.

| Stage | Question | Outcome |
|---|---|---|
| 1 | Which lexical retriever should replace the historical query-fitted TF-IDF? | Accent-preserving BM25 |
| 2 | Which multilingual dense retriever gives the best semantic retrieval? | E5-large-instruct |
| 3 | How should transcript-window scores become video scores? | Maximum window score |
| 4 | How should BM25 and E5 be combined? | 25% BM25 + 75% E5 |
| 5 | Which ASR and transcript representation should feed retrieval? | Parakeet processed |
| 6 | How much evidence should be passed to a reranker? | `K30_M5` on All50 |
| 7 | Does a cross-encoder improve the candidate ranking at acceptable cost? | Evaluation complete; BGE is the operational recommendation, final freeze pending |

The data chronology matters when interpreting the numbers. Stages 1–5 used `development20` against the complete 50-video All50 corpus. `holdout20` remained closed through those decisions and was evaluated once after the Stage 1–5 subsystem had been frozen. After that evaluation, it was no longer an untouched holdout.

Stages 6–7 therefore used all 40 Extension40 queries (`extension40_full`) against All50 for development and comparison. The old `development20` and `holdout20` labels were retained only as diagnostic slices. They should not be presented as independent validation splits for the Stage 6–7 decisions.

For full methodology and results, see [`docs/retrieval-v2.md`](docs/retrieval-v2.md).

## 4. Data, Artifacts, and Running the Code

Large files are intentionally kept outside Git. This includes source video/audio, ASR outputs, model weights, embedding caches, retrieval score caches, reranker caches, and generated reports.

The retrieval code separates two roots:

```text
CODE_ROOT
    source code, configuration, notebooks, and documentation

ARTIFACT_ROOT
    ASR outputs, embeddings, score caches, candidate pools, and reports
```

The artifact workspace can be set with:

```text
AIC_RETRIEVAL_ARTIFACT_ROOT
```

This separation is important because the research workspace is much larger than the Git repository.

Corpus-side work should also be reused rather than repeated for every query. ASR transcripts and E5 document embeddings are generated once and cached. A production-style retrieval service should similarly build or load the BM25 index and dense document matrix at startup, keep the models resident, and perform only query-dependent work after a search request arrives.

The canonical stage runner is:

```text
src/07_evaluate_retrieval_v2.py
```

Stage-specific notebooks under `notebooks/` are execution and review interfaces; the retrieval algorithms themselves belong in the source modules rather than notebook cells.
