# ASR Retrieval System

This document describes the current ASR-based text retrieval system from audio preparation to final ranking. It is an implementation reference: the experiments that led to these choices are documented separately in [`retrieval-v2.md`](retrieval-v2.md).

The system is designed around a simple idea. Cheap retrieval methods search the whole transcript corpus, while an expensive cross-encoder is only allowed to inspect a small candidate set.

## 1. System Overview

Many competition queries contain information that is spoken rather than visible: names, places, measurements, news topics, descriptions from a presenter, or other details that may never appear clearly in a frame. The ASR retrieval subsystem converts that speech into local searchable passages and uses those passages to rank both videos and approximate moments.

The pipeline separates **offline corpus preparation** from **online query processing**.

```text
OFFLINE CORPUS PREPARATION

source video/audio
        ↓
fixed 60 s audio windows
45 s stride / 15 s overlap
        ↓
Parakeet CTC 0.6B Vietnamese
        ↓
raw ASR result
        ↓
conservative post-processing
        ↓
processed retrieval text
        ↓
┌─────────────────────┬─────────────────────────┐
│ BM25 corpus index   │ E5 document embeddings  │
└─────────────────────┴─────────────────────────┘


ONLINE QUERY PROCESSING

natural-language query
        ↓
┌─────────────────────┬─────────────────────────┐
│ BM25 lexical score  │ E5 semantic score       │
└─────────────────────┴─────────────────────────┘
        ↓
normalize each score source per query
        ↓
25% BM25 + 75% E5
        ↓
hybrid score for each transcript window
        ↓
best window score for each video
        ↓
first-stage video ranking
        ↓
candidate videos + candidate windows
        ↓
cross-encoder reranking
        ↓
50% first-stage + 50% reranker score
        ↓
final ranked videos + supporting moments
```

The first stage is the stable core of the system. It uses Parakeet processed transcripts, accent-preserving BM25, multilingual E5-large-instruct, normalized 25/75 fusion, and maximum-window video scoring.

Stage 6 adds candidate selection for expensive reranking. Stage 7 has been fully evaluated; BGE reranker v2 M3 with 50/50 first-stage/reranker fusion is the current operational recommendation, although the final Stage 7 decision remains `pending_review` in the configuration.

## 2. Preparing the Searchable Transcript Corpus

### 2.1 Audio windows are created before ASR

The ASR runner does not transcribe an entire video first and then cut the transcript into text windows. The benchmark manifest already defines physical audio windows. For each window, the runner uses the stored sample indices to read the exact section of the source WAV.

The current policy is:

| Setting | Value |
|---|---:|
| Window length | 60 s |
| Stride | 45 s |
| Overlap | 15 s |

Using local windows solves two problems. First, a long video may contain only one short passage related to the query, so indexing the whole transcript as one document would mix relevant and unrelated speech. Second, the 15-second overlap reduces boundary errors when useful speech starts near the end of one window and continues into the next.

The overlap also has an important consequence: adjacent windows are correlated and can contain repeated speech. Two neighboring high scores therefore represent nearby support, not two independent observations.

The inference runner makes window extraction deterministic. It verifies the expected sample rate, seeks to the exact `sample_start`, reads the requested number of `int16` samples, and converts stereo audio to mono by averaging channels when necessary. It writes a temporary mono PCM-16 WAV for inference and stores a SHA-256 hash of the resulting PCM samples.

This is more than bookkeeping. It guarantees that different ASR models are compared on the same audio and allows a completed window to be reused safely when the audio, model, code, and benchmark identities have not changed.

### 2.2 Parakeet transcription

The selected ASR is NVIDIA Parakeet CTC 0.6B Vietnamese. The implementation loads the NeMo model from its pinned Hugging Face revision and transcribes each prepared window independently.

Each saved window contains enough information to reproduce or diagnose the inference:

- window and video identity;
- sample and timing information;
- source-WAV and window-PCM hashes;
- model and adapter identity;
- raw transcript text;
- native timestamp segments when available;
- inference runtime and peak GPU memory;
- warnings, rejections, and any inference error.

The runner is resume-safe. A previous window can only be reused when its window identity, audio hashes, windowing policy, model configuration, adapter code, and model revision match the current run. Failed or empty windows can also be retried explicitly.

### 2.3 Transcript post-processing

The processed transcript is **not** a corrected or rewritten transcript. Post-processing is deliberately conservative and query-independent. Its purpose is to make the ASR output safer for retrieval without inventing information that the model did not produce.

Version 1.1 currently performs the following operations:

- normalize whitespace and Unicode to NFC;
- create a lowercased matching representation with punctuation removed while preserving Vietnamese letters;
- create an auxiliary accent-folded representation;
- remove immediately repeated native ASR segments;
- normalize a small set of kilogram expressions to `kg`;
- detect a short list of known subscription-style boilerplate phrases;
- reject a window when known boilerplate dominates at least 80% of its non-space characters;
- remove known boilerplate when it appears only as part of a longer transcript;
- warn on empty ASR output; and
- reject consecutive windows whose processed retrieval text is effectively identical.

The last rule helps prevent repeated ASR artifacts from creating duplicate retrieval evidence across neighboring windows.

The main stored representations have different purposes:

| Representation | Purpose |
|---|---|
| Raw ASR text | Preserve the original model output for inspection and comparison |
| Normalized text | Lowercase/punctuation-normalized matching form with Vietnamese accents preserved |
| Accent-folded text | Auxiliary form used for diagnostics and experiments |
| Processed retrieval text | Conservative retrieval representation after duplicate/boilerplate handling and limited unit aliases |

A **warning** records a suspicious condition but does not necessarily exclude the window. A **rejection** means the processed transcript should not contribute evidence to the processed retrieval channel.

For the selected processed channel, a window is eligible only when ASR completed successfully, the processed text is nonempty, and the window has no active rejection reason. Ineligible windows remain on the physical window axis so IDs and timestamps stay aligned, but their retrieval contribution is disabled.

## 3. First-Stage Retrieval

The first-stage retriever scores every eligible transcript window using two complementary methods. BM25 rewards exact lexical evidence, while E5 captures semantic similarity when the query and transcript use different wording.

### 3.1 BM25 for exact lexical evidence

The selected lexical method is accent-preserving BM25 (`L2_bm25_preserving`).

BM25 is useful for words that should match literally, such as person names, locations, numbers, organizations, and uncommon topic terms. Vietnamese accents are preserved because the Stage 1 experiment showed that accent folding removed useful distinctions and substantially reduced retrieval quality.

The current implementation uses:

| Setting | Value |
|---|---|
| Text normalization | Lowercase matching form with Vietnamese accents preserved |
| Tokenization | Normalized whitespace tokens |
| `k1` | 1.5 |
| `b` | 0.75 |
| Query term frequency | Unique query terms |
| Indexed documents | Eligible, nonempty processed windows only |

The BM25 index is conceptually corpus-side state. In a deployment implementation it should be built or loaded once and kept in memory rather than reconstructed for every user query.

### 3.2 E5 for semantic retrieval

The selected dense model is:

```text
intfloat/multilingual-e5-large-instruct
```

Pinned revision:

```text
274baa43b0e13e37fafa6428dbc7938e62e5c439
```

The query is encoded with the instruction:

```text
Instruct: Given a detailed description of a target video moment, retrieve transcript passages that are relevant to the described event.
Query: <query>
```

Transcript windows are encoded from their processed text without an additional document prefix.

The backend converts embeddings to `float32`, checks that they are finite and nonzero, and explicitly L2-normalizes them. Query-document similarity is then the dot product of normalized vectors, equivalent to cosine similarity.

The expensive document side is reusable:

```text
offline:
encode all transcript windows once

online:
encode only the new query
→ compare against cached document vectors
```

This is why a much larger video corpus does not imply rerunning E5 over every transcript for every search.

### 3.3 Combining BM25 and E5

BM25 scores and E5 cosine scores have different numeric scales, so the system does not average their raw values directly.

For each query, each source is normalized independently over **eligible windows only**:

```text
normalized = (score - min) / (max - min)
```

If all eligible scores from one source are effectively constant, that normalized source is set to zero for the query. Ineligible windows receive zero after normalization.

The final first-stage window score is:

```text
hybrid_window_score
=
0.25 × normalized_BM25
+
0.75 × normalized_E5
```

Dense retrieval is therefore the main signal. BM25 acts as a smaller lexical correction when exact wording contains information that semantic retrieval may underweight.

No Whisper+Parakeet fusion and no raw+processed transcript fusion are used in the selected pipeline.

### 3.4 Turning window scores into video scores

A query may describe only a small part of a long video. Requiring several windows to score highly can therefore penalize a correct video whose relevant speech is brief.

The selected video score is simply the strongest hybrid window:

```text
video_score(video)
=
max(hybrid_window_score for windows in video)
```

This is the Stage 3 `P0_max` policy.

The video ranking is produced from these maximum scores. The strongest window also provides an initial temporal clue for that video.

Evaluation keeps ranking deterministic. Scores within `1e-12` are treated as tied for metrics and assigned the worst rank in the tie group; stable secondary identifiers are used when a display order is needed.

## 4. Candidate Selection and Reranking

### 4.1 Why candidate selection is necessary

BM25 and E5 are cheap enough to search the whole transcript corpus. A cross-encoder is different: it reads the query and candidate passage together, so its cost grows roughly with the number of query-window pairs it must process.

Stage 6 therefore introduces a candidate-selection layer before cross-encoding.

For the All50 research benchmark, the selected policy is `K30_M5`:

```text
first-stage scores for all 994 windows
        ↓
rank all 50 videos by their best window
        ↓
keep top 30 videos
        ↓
within each retained video,
keep up to 5 highest-scoring windows
        ↓
mean 149.275 pairs/query
maximum 150 pairs/query
```

This stage does not try to improve MRR. Its job is to reduce expensive reranker work while preserving recoverable evidence.

On the 40-query Full40 evaluation:

| Candidate measure | `K30_M5` |
|---|---:|
| Correct-video recall | 1.000 (40/40) |
| Relevant-window recall | 0.975 (39/40) |
| Joint recall | 0.975 (39/40) |
| Mean pairs/query | 149.275 |
| Maximum pairs/query | 150 |

The only remaining Stage 6 temporal miss is `R2-8`: its correct video is retained, but none of the five selected windows from that video satisfies the benchmark temporal relevance rule.

`K30_M5` should be understood as a **research-benchmark setting**, not a universal production constant. On All50, keeping 30 videos means retaining 60% of the video corpus. On a corpus of roughly 1,490 videos, the same K would retain only about 2%. Full-scale deployment must therefore remeasure first-stage Video Recall@K and candidate recall before deciding how many videos to expose to the reranker.

A sensible production design may keep a fixed reranker pair budget while spreading those pairs across more videos, but that has not been evaluated in the current Stage 1–7 experiments and should not be presented as a measured result.

### 4.2 Cross-encoder reranking

E5 embeds the query and transcript separately. A cross-encoder instead reads the query and candidate transcript together. This allows richer token-level interaction, but it is much more expensive.

Stage 7 evaluated six zero-shot local multilingual rerankers on the exact same frozen `K30_M5` candidate pool and Tesla T4 environment:

- mMARCO MiniLM;
- GTE multilingual reranker;
- BGE reranker v2 M3;
- Qwen3-Reranker-0.6B;
- Mixedbread mxbai-rerank-base-v2; and
- Jina reranker v2 multilingual.

The maximum input length was 512 tokens. All six models passed the compatibility/runtime preflight and used batch size 32 without OOM retries in the final valid run.

Stage 7 tested two reranking uses:

```text
S1:
reranker score only

S2:
50% normalized first-stage candidate score
+
50% normalized reranker score
```

There was no weight sweep.

The important result is that **every S1 reranker-only system produced lower Video MRR than the first-stage control**. The reranker is therefore useful as a refinement signal, not as a replacement for BM25+E5 retrieval.

For S2, the main Full40 results were:

| Reranker | Video R@1 | Video MRR | Story MRR | p90 | Meets 5 s limit |
|---|---:|---:|---:|---:|---|
| No reranker | 0.725 | 0.8110 | 0.9042 | — | Yes |
| MiniLM | 0.625 | 0.7154 | 0.8729 | 0.34 s | Yes |
| GTE | 0.700 | 0.7972 | 0.8988 | 1.04 s | Yes |
| **BGE v2 M3** | **0.750** | **0.8205** | **0.9125** | **2.54 s** | **Yes** |
| Qwen3-0.6B | **0.825** | **0.8642** | 0.8625 | 7.31 s | No |
| Mixedbread | 0.725 | 0.7955 | 0.8896 | 4.57 s | Yes |
| Jina v2 | 0.675 | 0.7543 | 0.9113 | 1.02 s | Yes |

Qwen produced the strongest video result, but it exceeded the 5-second p90 limit that was fixed before the experiment and reduced Story MRR. BGE produced a much smaller improvement, but it improved both Video and Story MRR and remained within the latency limit.

For this reason, **BGE v2 M3 with S2 is the current operational recommendation**. The Stage 7 run itself passed, but the current configuration still marks `selection.stage07_decision` as `pending_review`; the documentation therefore does not call BGE formally frozen yet.

## 5. Runtime and Scaling

Runtime measurements in this project have different scopes and should not be added together as though they were one directly measured end-to-end latency.

For the selected E5-large model on `parakeet_processed`, the measured Stage 2 query-encoding latency was about 24.4 ms p50 and 29.6 ms p90. Dense similarity search over the 994-window All50 corpus was much smaller than the encoding cost.

Stage 7 reranking is much more expensive. With `K30_M5`, each query contains about 149 candidate pairs. On the same T4:

| Reranker | Mean/query | p90/query | Pairs/s |
|---|---:|---:|---:|
| MiniLM | 0.278 s | 0.336 s | 536.7 |
| GTE | 0.913 s | 1.042 s | 163.5 |
| BGE v2 M3 | 2.127 s | 2.539 s | 70.2 |
| Qwen3-0.6B | 6.688 s | 7.308 s | 22.3 |
| Mixedbread | 4.128 s | 4.566 s | 36.2 |
| Jina v2 | 0.840 s | 1.017 s | 177.6 |

These are reranking-stage measurements, not complete application latency.

The online and offline responsibilities should remain separate:

```text
OFFLINE
- extract / prepare audio
- run ASR
- post-process transcripts
- construct/load BM25 corpus state
- encode E5 documents
- store compact window/video metadata

ONLINE
- tokenize the query for BM25
- encode the query with E5
- score transcript windows
- combine BM25 and E5
- aggregate to videos
- select candidate pairs
- optionally run the cross-encoder
- reconstruct the final ranking
```

For a much larger corpus, the first-stage index grows with the number of transcript windows. Reranking does **not** have to grow in the same way if the number of reranker pairs is explicitly capped.

The larger risk is candidate recall. A K value that works against 50 videos can become too narrow against roughly 1,490 distractor videos even if first-stage search itself remains fast. Full-corpus validation should therefore focus first on Video Recall@K, candidate recall, reranker pair budget, and warm query latency.

Models and indexes should stay resident during interactive use. Loading E5 or BGE from disk for each query would turn model startup into the dominant latency and does not reflect the intended retrieval design.

## 6. Implementation and Reproducibility

The source modules have intentionally separate responsibilities:

| File | Responsibility |
|---|---|
| `configs/retrieval_v2.json` | Frozen method definitions, model revisions, evaluation policy, stage configuration, and manual decisions |
| `src/03_run_stage1_full_inference.py` | Resume-safe window-level ASR inference and output validation |
| `src/model_adapters.py` | Common ASR interface and model-specific loading/transcription |
| `src/postprocess.py` | Conservative query-independent transcript normalization, warnings, and rejections |
| `src/retrieval_v2.py` | BM25, dense score bundles, normalization, first-stage fusion, and candidate-score blending |
| `src/retrieval_backends.py` | Dense and reranker model loading/inference |
| `src/retrieval_v2_evaluation.py` | Eligibility, temporal relevance, ranking, metrics, Stage 6 candidates, and Stage 7 ranking policy |
| `src/07_evaluate_retrieval_v2.py` | Contract validation, stage orchestration, cache/report management, and reproducibility checks |

Large artifacts are kept outside Git. `CODE_ROOT` points to source/configuration/documentation, while `ARTIFACT_ROOT` points to ASR outputs, embeddings, score caches, candidate pools, and reports. The artifact root can be configured with:

```text
AIC_RETRIEVAL_ARTIFACT_ROOT
```

Caches are treated as typed artifacts rather than anonymous arrays. A cached score or embedding is reused only when its model/config identity and query/window axes match the expected contract. The runner also stores hashes and manifests so a later stage can detect when an earlier-stage configuration changed and must be rerun.

The main principle is simple: **never silently realign or reinterpret cached scores**. If query IDs, window IDs, method identity, or source provenance differ, the cache should be rejected rather than “fixed” by sorting or intersecting axes.

For the research history, selection evidence, and per-stage results behind this implementation, see [`retrieval-v2.md`](retrieval-v2.md).
