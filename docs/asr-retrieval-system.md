# ASR Retrieval System

## 1. System Overview

The ASR retrieval subsystem converts spoken content from videos into searchable transcript windows and retrieves videos and temporal moments for a natural-language query. The current Stage 1–5 baseline uses Parakeet for Vietnamese ASR, combines BM25 lexical matching with E5-large semantic retrieval, and ranks each video by its strongest transcript window.

```text
Video audio
    ↓
Parakeet CTC 0.6B Vietnamese
    ↓
processed transcript
    ↓
60-second overlapping windows
    ↓
BM25 lexical retrieval
       +
E5-large semantic retrieval
    ↓
per-query score normalization
    ↓
25% BM25 + 75% E5 fusion
    ↓
hybrid window scores
    ↓
max score per video
    ↓
ranked videos + supporting windows
```

This document describes the selected ASR-text retrieval subsystem after Stage 5. It focuses on what the system does and how it is implemented, not on the experiments that led to these choices.

---

## 2. Inputs and Outputs

### 2.1 Inputs

The subsystem operates on three main inputs:

| Input | Description |
|---|---|
| Video/audio corpus | Source videos whose spoken content is transcribed offline |
| Transcript-window corpus | Frozen ASR windows used by the lexical and dense retrievers |
| Natural-language query | User or benchmark description of the target video moment |

The current retrieval corpus contains transcript windows generated from the video collection. Retrieval operates on these windows rather than on one transcript representation per full video.

### 2.2 Outputs

For each query, the subsystem produces:

| Output | Meaning |
|---|---|
| Sparse window score | BM25 relevance score for each transcript window |
| Dense window score | E5 semantic similarity for each transcript window |
| Hybrid window score | Weighted combination of normalized sparse and dense evidence |
| Video score | Maximum hybrid window score among the video's windows |
| Ranked videos | Videos ordered by video score |
| Supporting window | Highest-scoring transcript window that provides the strongest ASR evidence for a video |

The highest-scoring window also provides temporal evidence for where the relevant spoken content is likely to occur.

---

## 3. Offline Indexing Pipeline

### 3.1 ASR

The selected ASR model is:

```text
nvidia/parakeet-ctc-0.6b-Vietnamese
```

The selected transcript view is:

```text
processed
```

The retrieval system therefore uses the canonical processed transcript rather than the raw ASR text.

### 3.2 Windowing

Videos are represented as overlapping transcript windows:

| Setting | Value |
|---|---:|
| Window length | 60 s |
| Stride | 45 s |
| Overlap | 15 s |

Long videos are split into overlapping windows so retrieval can identify local spoken evidence instead of representing an entire video with one transcript.

The physical window axis is preserved consistently across ASR channels, caches, retrieval scores, and evaluation. Window IDs must remain unique and aligned with the benchmark manifest.

### 3.3 Retrieval eligibility

A processed transcript window contributes retrieval evidence only when:

- the ASR record completed successfully;
- the selected processed text is nonempty; and
- the canonical processed view has no rejection reason.

Ineligible windows remain on the physical window axis so IDs and temporal structure stay aligned, but they do not contribute valid retrieval evidence.

For lexical retrieval, ineligible windows receive zero evidence. Dense retrieval may use a finite invalid-score sentinel internally, but normalization and hybrid fusion exclude ineligible windows from score-range estimation and set their final fused contribution to zero.

### 3.4 Sparse index

The sparse retriever is:

```text
L2_bm25_preserving
```

Its main settings are:

| Setting | Value |
|---|---|
| Text normalization | `normalize_for_matching` |
| Accent handling | Preserve Vietnamese accents |
| Tokenization | Normalized whitespace tokens |
| BM25 `k1` | 1.5 |
| BM25 `b` | 0.75 |
| IDF | Positive Okapi-style `log(1 + ...)` |
| Query term frequency | Unique query terms |

Only eligible, nonempty transcript windows contribute to BM25 index statistics.

### 3.5 Dense index

The dense retriever is:

```text
D1_e5_large_instruct
```

Model:

```text
intfloat/multilingual-e5-large-instruct
```

Pinned revision:

```text
274baa43b0e13e37fafa6428dbc7938e62e5c439
```

The query input uses the frozen instruction:

```text
Instruct: Given a detailed description of a target video moment, retrieve transcript passages that are relevant to the described event.
Query: <query>
```

Document passages use the processed transcript text without an additional instruction prefix.

Embeddings are converted to `float32`, checked for finite values and nonzero norms, explicitly L2-normalized, and compared through normalized dot-product cosine similarity. Document embeddings can be cached offline and reused across query runs.

---

## 4. Online Query Retrieval

### 4.1 Window scoring

For one natural-language query, the sparse and dense retrievers score the same physical transcript-window axis:

```text
query
  ├─ BM25 → sparse score for every window
  └─ E5   → dense score for every window
```

The two score sources are kept aligned by query ID, window ID, ASR model, transcript view, and eligibility mask.

### 4.2 Per-query score normalization

BM25 and dense cosine scores use different numeric scales. Before fusion, each source is normalized independently for each query using only eligible windows.

For an eligible source score:

```text
normalized = (score - min) / (max - min)
```

where `min` and `max` are calculated from that source's eligible window scores for the current query.

If the eligible score range is effectively zero:

```text
normalized eligible scores = 0
```

Ineligible windows are assigned:

```text
0
```

after normalization.

This policy is referred to as:

```text
per-query eligible-only min-max normalization
```

### 4.3 Hybrid score

The selected fusion method is:

```text
H3_norm_25_75
```

For every transcript window:

```text
hybrid score =
0.25 × normalized BM25
+
0.75 × normalized E5
```

Dense retrieval is the main signal. BM25 acts as a smaller lexical correction for exact names, numbers, locations, rare words, and other distinctive terms.

No ASR fusion or raw+processed transcript fusion is used.

### 4.4 Video aggregation

The selected video aggregation policy is:

```text
P0_max
```

For each video:

```text
video score = maximum hybrid score among its windows
```

The video therefore receives the score of its strongest transcript match. This preserves partial relevance, which is important when only a short spoken segment of a long video matches the query.

### 4.5 Ranking and supporting windows

Videos are ranked by video score in descending order.

The window that produces the maximum score is the strongest supporting ASR window for that video and can be used as the initial temporal candidate.

Evaluation uses the frozen tie policy:

```text
tie tolerance = 1e-12
tie policy = worst rank within tolerance
```

Displayed ordering should remain deterministic by using stable secondary identifiers when equal scores need an explicit order.

---

## 5. Implementation Map

### 5.1 Core files

| File | Responsibility |
|---|---|
| `configs/retrieval_v2.json` | Frozen model IDs, revisions, stage decisions, score policies, fusion weights, paths |
| `src/retrieval_v2.py` | Score bundles, BM25, RRF, dense scoring, score normalization, hybrid fusion |
| `src/retrieval_backends.py` | Dense model loading, input formatting, embedding, normalization, runtime metadata |
| `src/retrieval_v2_evaluation.py` | Story relevance, video aggregation, ranking, ties, metrics, paired comparisons |
| `src/07_evaluate_retrieval_v2.py` | Config validation, artifact loading, cache validation, stage orchestration, reports |

Retrieval algorithms and evaluation policy should remain separate. Model loading belongs in the backend module, score production belongs in `retrieval_v2.py`, and ranking/evaluation policy belongs in `retrieval_v2_evaluation.py`.

### 5.2 Code and artifact roots

The project separates source code from large generated artifacts.

```text
CODE_ROOT
```

points to the Git repository containing source, configuration, notebooks, and compact documentation.

```text
ARTIFACT_ROOT
```

points to the external workspace containing large ASR outputs, embeddings, score caches, reports, and other generated artifacts.

The artifact root can be configured through:

```text
AIC_RETRIEVAL_ARTIFACT_ROOT
```

Code should resolve repository files from the code root and generated data from the artifact root rather than assuming both are the same directory.

### 5.3 Score and embedding caches

Important cached artifacts include:

```text
window_scores.npz
score_axes.json
```

`window_scores.npz` stores numeric score matrices. `score_axes.json` stores the identities required to interpret each matrix, including query set, method, ASR model, transcript view, query IDs, and window IDs.

Dense document embeddings are cached separately so the corpus does not need to be re-encoded for every experiment or query run.

Cache consumers must validate identities and axes before reuse. A score matrix should never be realigned silently by sorting, intersecting IDs, or dropping unmatched windows.

---

## 6. Frozen Configuration and Current Scope

### 6.1 Current Stage 1–5 baseline

| Component | Current setting |
|---|---|
| ASR | Parakeet CTC 0.6B Vietnamese |
| Transcript view | Processed |
| Window length | 60 s |
| Window stride | 45 s |
| Window overlap | 15 s |
| Sparse retriever | Accent-preserving BM25 |
| Dense retriever | E5-large-instruct |
| Normalization | Per-query eligible-only min-max |
| Sparse weight | 0.25 |
| Dense weight | 0.75 |
| Video scoring | Max window |
| Query form | Original natural-language query |

The corresponding method identities are:

```text
ASR channel:       parakeet_processed
Sparse retriever:  L2_bm25_preserving
Dense retriever:   D1_e5_large_instruct
Hybrid retriever:  H3_norm_25_75
Video aggregation: P0_max
```

### 6.2 Current scope

This subsystem currently covers:

- Vietnamese ASR transcript generation;
- processed transcript selection;
- transcript-window retrieval;
- sparse and dense text scoring;
- score normalization and hybrid fusion;
- global video ranking; and
- supporting transcript-window retrieval.

It does **not yet include**:

```text
hierarchical video → temporal candidate retrieval
second-stage reranking
query reformulation
alternative transcript representation
visual retrieval
OCR retrieval
multimodal fusion
final holdout evaluation
```

This document describes the frozen ASR-text retrieval baseline after Stage 5. Later retrieval stages may wrap or extend this subsystem, but should treat these components as the current starting point unless a later reviewed decision explicitly replaces them.
