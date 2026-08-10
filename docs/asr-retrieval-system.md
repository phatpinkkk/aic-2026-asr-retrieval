# ASR Retrieval System

## 1. Overview

The ASR retrieval subsystem makes spoken video content searchable.

Each video is transcribed, divided into overlapping transcript windows, and indexed in two complementary ways:

- **BM25** finds windows that share important words with the query.
- **Multilingual E5-large** finds windows that are semantically similar to the query even when the wording is different.

The two score sets are normalized and combined. Each video is then represented by its strongest matching transcript window.

```text
Video audio
    ↓
Parakeet CTC 0.6B Vietnamese
    ↓
canonical processed transcript
    ↓
60-second overlapping transcript windows
    ↓
┌──────────────────────┐
│ BM25 lexical scores  │
└──────────┬───────────┘
           │
           ├──────────────┐
           │              │
┌──────────▼───────────┐  │
│ E5 semantic scores   │  │
└──────────┬───────────┘  │
           │              │
           └──────┬───────┘
                  ↓
       per-query score normalization
                  ↓
       25% BM25 + 75% E5
                  ↓
          hybrid window scores
                  ↓
      maximum window score per video
                  ↓
      ranked videos + supporting windows
```

This document explains the current system directly. It does not describe the experiments or internal development labels that were used to choose these components.

---

## 2. Transcript Preparation

### 2.1 Speech recognition

The system uses:

```text
nvidia/parakeet-ctc-0.6b-Vietnamese
```

to transcribe Vietnamese speech.

The retrieval input is the **processed transcript view** produced by the canonical transcript-processing pipeline. Raw ASR output is not used as a second retrieval channel in the current system.

### 2.2 Transcript windows

A full video transcript is not indexed as one document. Instead, each video is divided into overlapping temporal windows:

| Setting | Value |
|---|---:|
| Window length | 60 s |
| Stride | 45 s |
| Overlap | 15 s |

The overlap helps preserve spoken evidence that crosses a window boundary.

Window-level indexing also gives the retriever a local unit of evidence. A query can match one relevant minute of a long video without requiring the rest of the transcript to be related.

### 2.3 Eligible retrieval windows

A processed transcript window is eligible for retrieval when:

- the ASR record completed successfully;
- the processed text is nonempty; and
- the processed view has no canonical rejection reason.

Ineligible windows remain in the physical window list so IDs and timestamps stay aligned with the rest of the system, but they do not contribute retrieval evidence.

---

## 3. Text Retrieval

For every natural-language query, the same eligible transcript windows are scored by both BM25 and multilingual E5-large.

### 3.1 BM25 lexical retrieval

BM25 is the lexical component of the system. It is useful when the query and transcript share distinctive words such as names, locations, numbers, or topic-specific terms.

The current BM25 configuration preserves Vietnamese accents.

| Setting | Value |
|---|---|
| Text normalization | `normalize_for_matching` |
| Accent handling | Preserve Vietnamese accents |
| Tokenization | Normalized whitespace tokens |
| `k1` | 1.5 |
| `b` | 0.75 |
| IDF | Positive Okapi-style IDF |
| Query term frequency | Unique query terms |

Only eligible, nonempty transcript windows contribute to the BM25 index.

For each query, BM25 produces one lexical relevance score for every physical transcript window.

### 3.2 Multilingual semantic retrieval

The semantic component uses:

```text
intfloat/multilingual-e5-large-instruct
```

Pinned revision:

```text
274baa43b0e13e37fafa6428dbc7938e62e5c439
```

Queries are encoded with the instruction:

```text
Instruct: Given a detailed description of a target video moment, retrieve transcript passages that are relevant to the described event.
Query: <query>
```

Transcript windows are encoded directly from their processed text without an additional document prefix.

Query and document embeddings are:

1. converted to `float32`;
2. checked for finite values and nonzero norms;
3. explicitly L2-normalized; and
4. compared using normalized dot-product cosine similarity.

Document embeddings can be computed offline and reused. At query time, only the query embedding needs to be generated before similarity scores are calculated against the cached transcript-window embeddings.

### 3.3 Why both retrievers are used

BM25 and E5 solve different parts of the retrieval problem.

BM25 is strong when exact wording matters. E5 is stronger when the query describes the same event using different words from the transcript.

The system therefore treats E5 as the main retrieval signal and BM25 as a smaller lexical correction rather than giving both sources equal influence.

---

## 4. Score Fusion and Video Ranking

### 4.1 Why scores are normalized

Raw BM25 scores and cosine-similarity scores are on different numeric scales, so they cannot be combined directly.

For each query, BM25 scores and E5 scores are normalized **independently** using only eligible windows.

For one score source:

```text
normalized_score = (score - minimum) / (maximum - minimum)
```

where `minimum` and `maximum` are calculated across that source's eligible windows for the current query.

If all eligible scores from a source are effectively identical, their normalized values are set to zero.

Ineligible windows are assigned a final normalized score of zero.

This makes the two retrieval sources comparable while preserving their relative ordering for the current query.

### 4.2 Hybrid window score

Each transcript window receives one final retrieval score:

```text
hybrid_score =
0.25 × normalized_BM25
+
0.75 × normalized_E5
```

The weighting intentionally gives semantic retrieval most of the influence while still allowing exact lexical evidence to adjust the ranking.

There is no fusion between different ASR models and no fusion between raw and processed transcript views.

### 4.3 From windows to videos

A video can contain many transcript windows, but a query may only describe one short part of that video.

The system therefore scores each video using its strongest matching window:

```text
video_score = max(hybrid_score of all windows in the video)
```

Videos are ranked in descending order by this score.

This policy preserves **partial relevance**: one highly relevant transcript window is enough for a video to rank strongly even when most of the video discusses something else.

### 4.4 Supporting temporal evidence

The window that produces the highest hybrid score for a video is also its strongest ASR evidence.

Its `start_s` and `end_s` values provide an initial temporal location that can be passed to later candidate selection, reranking, or multimodal stages.

The current subsystem therefore produces both:

```text
ranked videos
+
supporting transcript windows
```

rather than only a video-level score.

### 4.5 Deterministic ranking

Evaluation treats scores within `1e-12` as tied and assigns the worst rank within that tie group.

When an explicit display order is needed for equal scores, stable secondary identifiers are used so repeated runs remain deterministic.

---

## 5. Data Flow and Caching

### 5.1 Offline work

The expensive corpus-side work is performed before online querying:

```text
video/audio
    ↓
ASR transcription
    ↓
processed transcript windows
    ↓
BM25 corpus statistics
    ↓
E5 document embeddings
    ↓
cached retrieval artifacts
```

Dense document embeddings are reusable across queries because the transcript corpus does not change between searches.

### 5.2 Online work

For a new query:

```text
query
  ├─ tokenize and score with BM25
  └─ encode once with E5
            ↓
      score all transcript windows
            ↓
      normalize both score sets
            ↓
      weighted score fusion
            ↓
      maximum score per video
            ↓
      ranked videos
```

The dense query encoder is the main online neural cost. Score fusion itself is lightweight.

### 5.3 Score caches

Retrieval experiments and validation use two important cache files:

```text
window_scores.npz
score_axes.json
```

`window_scores.npz` stores numeric score matrices.

`score_axes.json` stores the identities needed to interpret those matrices, such as:

- query set;
- retrieval source;
- ASR model;
- transcript view;
- query IDs; and
- window IDs.

A cached score matrix must only be reused when its identity and axes match exactly. The system should never silently repair a mismatch by sorting IDs independently, intersecting two axes, or dropping unmatched windows.

### 5.4 Code and artifact locations

The project separates source code from large generated artifacts.

`CODE_ROOT` refers to the Git repository containing source, configuration, notebooks, and documentation.

`ARTIFACT_ROOT` refers to the external experiment workspace containing large ASR outputs, embeddings, score caches, and generated reports.

The artifact root can be configured through:

```text
AIC_RETRIEVAL_ARTIFACT_ROOT
```

This separation keeps large generated data out of Git while allowing the same source code to run against the shared experiment workspace.

---

## 6. Implementation Map and Current Scope

### 6.1 Core implementation files

| File | Responsibility |
|---|---|
| `configs/retrieval_v2.json` | Model revisions, transcript settings, retrieval parameters, score policies, fusion weights, and artifact paths |
| `src/retrieval_v2.py` | BM25 scoring, dense score bundles, score normalization, and sparse+dense fusion |
| `src/retrieval_backends.py` | Dense model loading, query/document formatting, embedding, normalization, and runtime metadata |
| `src/retrieval_v2_evaluation.py` | Temporal relevance, video ranking, tie handling, and retrieval metrics |
| `src/07_evaluate_retrieval_v2.py` | Configuration validation, cache loading, artifact validation, experiment execution, and report generation |

The modules intentionally separate model inference, retrieval scoring, evaluation policy, and experiment orchestration.

### 6.2 Current system configuration

| Component | Current setting |
|---|---|
| ASR | Parakeet CTC 0.6B Vietnamese |
| Transcript representation | Canonical processed transcript |
| Window length | 60 s |
| Window stride | 45 s |
| Window overlap | 15 s |
| Lexical retrieval | Accent-preserving BM25 |
| Semantic retrieval | Multilingual E5-large-instruct |
| Score normalization | Per-query, eligible-window min-max |
| BM25 contribution | 25% |
| E5 contribution | 75% |
| Video score | Maximum hybrid window score |
| Query representation | Original natural-language query |

### 6.3 What this subsystem currently covers

The ASR retrieval subsystem currently handles:

- Vietnamese speech transcription;
- processed transcript selection;
- overlapping transcript-window construction;
- lexical retrieval;
- semantic retrieval;
- score normalization;
- lexical-semantic fusion;
- global video ranking; and
- retrieval of the strongest supporting transcript window.

The broader video-search pipeline may later add:

- hierarchical candidate retrieval;
- second-stage reranking;
- improved query or transcript representations;
- visual retrieval;
- OCR retrieval;
- multimodal fusion; and
- independent final evaluation of the broader retrieval system.

Those later components can extend this subsystem, but the ASR-text pipeline described here is the current retrieval baseline they should start from.

### 6.4 Holdout validation

After the ASR-text subsystem was frozen, it was evaluated once on `holdout20`. The frozen system achieved **Video R@1 = 0.70**, **Video R@5 = 0.95**, **Video R@10 = 0.95**, **Video MRR = 0.8119**, **Story R@1 = 0.90**, and **Story MRR = 0.9500**.

These results are validation evidence for the frozen subsystem, not a new tuning signal. `holdout20` is now considered exposed and must not be used to modify the ASR model, transcript representation, retrieval weights, or other Stage 1–5 settings. Detailed comparisons and failure analysis are maintained in `docs/results/retrieval-v2-progress.md`.
