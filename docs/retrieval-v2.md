# Retrieval v2

## 1. Purpose and Scope

### 1.1 Goal

Retrieval v2 develops the text-retrieval component of the AIC 2026 video-search pipeline. ASR transcripts provide useful evidence about spoken events, people, locations, numbers, and actions, but earlier evaluation showed that global video identification is substantially harder than locating the relevant moment once the correct video is known.

The goal is therefore to improve retrieval progressively. The work starts with lexical and dense text retrieval, then moves to stronger video-level evidence aggregation, sparse+dense hybrid retrieval, ASR and transcript-view selection, hierarchical retrieval, reranking, query and transcript representation, and finally multimodal integration.

Retrieval v2 is designed as a controlled research pipeline. Each stage should answer one main question while keeping the rest of the retrieval system fixed as much as possible.

### 1.2 Scope

Retrieval v2 covers:

- ASR transcript retrieval
- window-level scoring
- video-level evidence aggregation
- sparse+dense hybrid retrieval
- candidate-video and candidate-moment selection
- candidate reranking
- query and transcript representation
- multimodal integration with visual and OCR evidence

Retrieval v2 does not attempt to redesign the ASR models themselves. Whisper Large-v3 and NVIDIA Parakeet CTC 0.6B Vietnamese are retained as controlled development conditions until the retrieval architecture is strong enough to support a final ASR decision.

### 1.3 Historical baseline

The historical Baseline v1 remains frozen for regression and comparison.

| Component | Baseline v1 |
|---|---|
| Lexical retrieval | Query-fitted character TF-IDF |
| Dense retrieval | `intfloat/multilingual-e5-small` |
| Combination | Fixed 50/50 lexical-semantic score average |
| Video aggregation | Maximum-scoring window |
| Transcript conditions | Whisper / Parakeet, raw / processed |

Retrieval v2 is developed separately so that improvements can be measured against the frozen baseline without silently changing historical behavior.

---

## 2. Evaluation Protocol

### 2.1 Benchmark structure

| Dataset | Videos | Queries | Windows | Purpose |
|---|---:|---:|---:|---|
| Core10 | 10 | 10 | 219 | Regression and implementation checks |
| Extension40 | 40 | 40 | 775 | Development and holdout queries |
| All50 | 50 | 50 | 994 | Full retrieval corpus |

Every development query is retrieved against the complete All50 corpus. This avoids artificially easy evaluation against a reduced set of videos.

### 2.2 Development and holdout

`development20` is used for retrieval architecture, model, and policy selection.

`holdout20` remains closed during development and is evaluated only after the final retrieval configuration has been frozen.

No holdout result should influence:

- lexical-retriever selection
- dense-model selection
- video aggregation
- sparse+dense fusion
- ASR selection
- transcript-view selection
- hierarchical retrieval
- reranking
- query representation
- multimodal integration

### 2.3 Transcript windows

| Setting | Value |
|---|---:|
| Window length | 60 s |
| Stride | 45 s |
| Overlap | 15 s |

Adjacent windows share 15 seconds of audio. Therefore, neighboring high scores should be interpreted as temporal support or persistence, not as independent evidence.

### 2.4 Transcript channels

| Channel | Description |
|---|---|
| Whisper raw | Original Whisper transcript |
| Whisper processed | Conservatively cleaned Whisper transcript |
| Parakeet raw | Original Parakeet transcript |
| Parakeet processed | Conservatively cleaned Parakeet transcript |

These four channels are development conditions, not fusion inputs. They remain independent until Stage 5 selects one ASR and one primary transcript view.

### 2.5 Metrics

#### Video retrieval

Video retrieval is evaluated with:

- R@1
- R@3
- R@5
- R@10
- R@20
- MRR

These metrics measure whether the correct video is ranked near the top of the full 50-video corpus.

#### Story retrieval

Story retrieval is evaluated with:

- R@1
- R@3
- R@5
- R@10
- MRR

These metrics measure whether the correct temporal region is ranked highly within the known correct video.

#### Paired analysis

Aggregate metrics are supplemented with:

- per-query ranks
- better / tie / worse comparisons
- correct-video score margins
- known failure cases

This is important because `development20` contains only 20 queries, so one query changes R@1 by 0.05.

### 2.6 ASR reference policy

Manually verified transcript ground truth is not currently available. Whisper Large-v3 is therefore used only as a frozen pseudo-reference for transcript-agreement diagnostics.

WER and CER against Whisper measure agreement with Whisper rather than absolute transcription accuracy. Downstream retrieval metrics, reliability measurements, and manual inspection remain the main evidence for system selection.

---

## 3. Retrieval Architecture and Current State

### 3.1 Retrieval flow

```text
query
  ↓
query representation
  ↓
lexical and/or dense window retrieval
  ↓
window scores
  ↓
video evidence aggregation
  ↓
video ranking
  ↓
candidate temporal retrieval
  ↓
optional reranking
  ↓
final video / moment result
```

Stage 9 expands the evidence sources beyond ASR transcripts:

```text
ASR + visual + OCR
```

### 3.2 Current selected components

| Component | Current choice | Status |
|---|---|---|
| Lexical retriever | L2 BM25 preserving Vietnamese accents | Selected |
| Dense retriever | D1 multilingual-e5-large-instruct | Selected |
| Video aggregation | Maximum window | Temporary baseline |
| Hybrid policy | Not selected | Stage 4 |
| ASR | Not selected | Stage 5 |
| Transcript view | Not selected | Stage 5 |
| Hierarchical retrieval | Not selected | Stage 6 |
| Reranker | Not selected | Stage 7 |
| Query representation | Original query | Stage 8 |
| Multimodal evidence | ASR only | Stage 9 |

### 3.3 Current score sources

The two score sources carried into Stage 3 are:

```text
Lexical:
L2_bm25_preserving

Dense:
D1_e5_large_instruct
```

Detailed metrics and current progress are maintained separately in `docs/results/retrieval-v2-progress.md`.

---

## 4. Development Principles and Fixed Decisions

### 4.1 Controlled stage design

Each stage should change one major part of the retrieval system.

For example:

```text
Stage 2 changes representation.
Stage 3 changes video aggregation.
Stage 4 changes sparse+dense combination.
```

Avoid changing several major components at once because that makes improvements difficult to interpret.

### 4.2 Aggressive pruning

Each completed stage should reduce the experiment tree. Clearly dominated methods should not automatically be carried into later stages.

Current examples:

```text
Stage 1:
keep BM25 preserving
drop corpus TF-IDF, folded BM25, and BM25 RRF as downstream candidates

Stage 2:
keep E5-large-instruct
retain E5-small only as historical control
drop Qwen3 and BGE-M3 as downstream candidates
```

Historical controls remain available for comparison but do not become active branches.

### 4.3 ASR policy

Whisper and Parakeet remain independent evaluation conditions through Stage 5.

The final system should use one ASR unless later evidence clearly demonstrates that additional ASR complexity is necessary.

### 4.4 Transcript-view policy

Raw and processed transcripts remain diagnostic views until Stage 5.

They are not automatically fused.

### 4.5 Fusion policy

The following are intentionally excluded from the roadmap:

```text
Whisper + Parakeet fusion
raw + processed fusion
all-four-channel fusion
```

Sparse+dense fusion is different because lexical and semantic retrieval represent distinct retrieval mechanisms and will be evaluated explicitly in Stage 4.

### 4.6 Complexity policy

Complexity should be added only when it produces a clear downstream retrieval benefit.

Compute and engineering effort should preferentially be spent on stronger retrieval, temporal evidence, reranking, and multimodal information rather than maintaining redundant transcript channels.

### 4.7 Efficiency tracking

Future candidate components should be evaluated on both retrieval quality and operational cost.

Where applicable, track:

- retrieval metrics
- warm query latency
- offline indexing time
- throughput
- GPU memory
- candidate-set size
- reranking cost

---

## 5. Retrieval v2 Roadmap

### 5.1 Stage 1 – Proper Lexical Retrieval

#### Goal

Establish a reliable lexical retriever for noisy Vietnamese ASR transcripts.

#### Main question

> Does a proper corpus-oriented lexical method improve retrieval over the historical query-fitted TF-IDF implementation?

#### Methods considered

```text
L0 – query-fitted character TF-IDF
L1 – corpus-fitted character TF-IDF
L2 – BM25 preserving Vietnamese accents
L3 – accent-folded BM25
L4 – RRF of preserving and folded BM25
```

Corpus-fitted TF-IDF corrects the historical fitting issue. BM25 is designed for document retrieval and accounts for term rarity, repeated occurrences, and document length. Accent folding was tested because ASR can make diacritic errors, while accent-preserving BM25 tests whether Vietnamese lexical distinctions are more valuable than that additional robustness.

#### Decision

```text
Selected:
L2_bm25_preserving

Retained alternative:
None
```

Accent folding substantially reduced retrieval quality and is not carried forward.

---

### 5.2 Stage 2 – Stronger Dense Retrieval

#### Goal

Improve semantic retrieval beyond multilingual-e5-small.

#### Main question

> Can a stronger multilingual embedding model improve global video discrimination while keeping temporal localization strong?

#### Models evaluated

```text
D0 – multilingual-e5-small
D1 – multilingual-e5-large-instruct
D2 – Qwen3-Embedding-0.6B
D3 – BGE-M3 dense
```

All models were evaluated with the same query set, transcript channels, eligibility rules, window universe, max-window video aggregation, and evaluation metrics.

No BM25 fusion, aggregation change, ASR fusion, transcript-view fusion, reranking, or query rewriting was introduced.

#### Decision

```text
Selected:
D1_e5_large_instruct

Retained alternative:
None
```

E5-large-instruct produced the strongest and most consistent global video retrieval while maintaining practical indexing cost, latency, and GPU usage.

---

### 5.3 Stage 3 – Video and Temporal Evidence Aggregation

#### Goal

Improve how window-level retrieval scores are converted into video-level evidence.

#### Problem

The current system uses:

```text
video score = highest-scoring window
```

This is simple but fragile. A single accidentally high-scoring or noisy transcript window can make an incorrect video outrank the correct one.

#### Main question

> Can temporally supported evidence improve video ranking compared with relying on one maximum-scoring window?

#### Inputs

Stage 3 uses exactly two frozen score sources:

```text
L2_bm25_preserving
D1_e5_large_instruct
```

They should be evaluated independently so that the aggregation policy is not tuned to only one score distribution.

#### Candidate aggregation methods

**P0 – Max**

```text
video score = maximum window score
```

This remains the control.

**P1 – Top-2 mean**

Average the two strongest window scores in each video.

Purpose: reduce dependence on one isolated high score.

**P2 – Top-3 mean**

Average the three strongest window scores.

Purpose: test whether broader repeated support improves robustness.

**P3 – Best adjacent-pair mean**

Order the windows temporally, average every pair of neighboring window scores, and use the strongest pair as the video score.

Purpose: test whether short contiguous temporal support is more reliable than an isolated peak.

**P4 – Best contiguous-triplet mean**

Average every valid run of three consecutive windows and use the strongest triplet as the video score.

Purpose: test whether stronger temporal persistence provides additional robustness.

Because adjacent windows overlap, this evidence should be described as temporal support or persistence rather than independent confirmation. P0–P4 form the complete first-pass Stage 3 experiment. More complex supported-max or event-style aggregation should only be considered if the first run reveals a clear trade-off that motivates a follow-up.

#### What remains fixed

- lexical retriever
- dense retriever
- ASR channels
- transcript views
- query representation
- evaluation protocol

#### Decision

Select one general video aggregation policy if it improves video retrieval consistently across both lexical and dense evidence.

The selected aggregation policy becomes fixed before Stage 4.

---

### 5.4 Stage 4 – Sparse + Dense Hybrid Retrieval

#### Goal

Combine complementary lexical and semantic retrieval evidence.

#### Main question

> Does combining BM25 and E5-large improve retrieval beyond either retriever independently?

#### Inputs

```text
Sparse:
L2_bm25_preserving

Dense:
D1_e5_large_instruct

Aggregation:
selected Stage 3 policy
```

BM25 is strong when a query contains distinctive names, numbers, locations, exact terms, or rare words. Dense retrieval is better when relevant text expresses the same meaning with different wording.

The two methods may therefore recover different correct results.

#### Candidate methods

**H0 – Sparse only**

Control.

**H1 – Dense only**

Control.

**H2 – Reciprocal Rank Fusion**

Combine sparse and dense rankings instead of raw scores.

Primary predefined setting:

```text
RRF k = 60
```

This avoids problems caused by incompatible BM25 and cosine score scales.

**Limited normalized score fusion**

Evaluate only a small predefined set such as:

```text
25% sparse + 75% dense
50% sparse + 50% dense
75% sparse + 25% dense
```

Avoid large weight sweeps on only 20 development queries.

#### What remains fixed

- dense model
- lexical model
- ASR channels
- transcript views
- video aggregation
- query representation

#### Decision

Choose one retrieval architecture:

```text
sparse only
dense only
or
hybrid
```

The result is frozen before Stage 5.

---

### 5.5 Stage 5 – ASR and Text-View Re-evaluation

#### Goal

Reduce four development channels to one practical transcript pipeline.

#### Main question

> After improving retrieval itself, is Whisper still sufficiently better than Parakeet to justify its higher operational cost, and should raw or processed transcripts be retained?

#### Conditions

Evaluate the frozen Stage 4 retriever independently on:

```text
Whisper raw
Whisper processed
Parakeet raw
Parakeet processed
```

This is explicitly a selection stage.

It does not test:

```text
Whisper + Parakeet
raw + processed
all four channels
```

#### Evaluation evidence

Use:

- Video R@1
- Video R@5
- Video R@10
- Video MRR
- Story R@1
- Story MRR
- paired per-query ranks
- catastrophic failures
- ASR inference cost
- retrieval latency

A cheaper ASR may replace a stronger one if downstream retrieval becomes sufficiently close and no systematic failure pattern remains.

The exact engineering thresholds should be frozen in configuration before the final Stage 5 comparison.

#### Output

```text
one ASR
+
one primary transcript view
```

Everything downstream should use that single transcript pipeline.

---

### 5.6 Stage 6 – Hierarchical Retrieval

#### Goal

Separate video retrieval from temporal localization.

#### Motivation

Current retrieval scores transcript windows globally.

A hierarchical system instead performs:

```text
query
  ↓
retrieve candidate videos
  ↓
keep top-K videos
  ↓
search temporal windows only inside those videos
```

#### Main question

> Can separating video selection and temporal localization improve retrieval quality or make later reranking more efficient?

#### Candidate-video stage

Evaluate:

```text
candidate Video Recall@5
candidate Video Recall@10
candidate Video Recall@20
```

The candidate stage must preserve very high recall because a video removed here cannot be recovered later.

#### Temporal stage

Within selected videos:

```text
rank transcript windows
identify candidate moments
return temporal evidence
```

#### Important constraint

Current All50 dense search is already very fast. Hierarchical retrieval is therefore not justified by the current 994-window search latency alone.

It should be retained only if it improves:

- retrieval quality
- reranking efficiency
- future corpus scaling

#### Output

Select:

```text
candidate K
video scoring policy
within-video retrieval policy
```

---

### 5.7 Stage 7 – Candidate Reranking

#### Goal

Use a more expensive model only on a small first-stage candidate set.

#### Main question

> Can richer query-transcript interaction fix difficult ranking errors without paying the cost over the full corpus?

#### Pipeline

```text
first-stage retriever
  ↓
top candidate windows or videos
  ↓
reranker
  ↓
final ordering
```

Potential reranker families include:

- multilingual cross-encoders
- instruction-following rerankers
- other query-document interaction models

The exact model should be selected only when Stage 7 begins.

#### Evaluation

Track:

- retrieval-quality gain
- candidate recall
- reranking latency
- GPU memory
- failure recovery
- sensitivity to candidate K

#### Decision

Retain reranking only if the gain justifies the added online cost.

---

### 5.8 Stage 8 – Query and Transcript Representation Improvements

#### Goal

Improve how information is presented to the retriever without changing the underlying event being searched.

#### Main question

> Are the original natural-language queries and transcript windows the best retrieval representations?

#### Query-side possibilities

Start conservatively:

```text
original query
content-focused query
entity / location / number-focused representation
ASR-oriented query reformulation
```

The original query remains the control.

#### Transcript-side possibilities

Potential experiments include:

```text
sentence cleanup
context expansion
neighbor-window context
structured entity extraction
number normalization
light transcript segmentation
```

Representations must not invent content that is not present in the original query or ASR transcript.

#### Experimental rule

Avoid large prompt searches on `development20`.

Use a small number of predefined representations, clear deterministic rules, and paired evaluation.

#### Output

Select one query representation and, if useful, one transcript representation.

---

### 5.9 Stage 9 – Multimodal Integration

#### Goal

Address queries that ASR alone cannot solve.

#### Motivation

Some queries depend primarily on:

```text
objects
appearance
actions
scene layout
signs
logos
on-screen text
visual events
```

No ASR retriever can recover information that was never spoken.

#### Evidence sources

Stage 9 introduces:

```text
ASR
visual embeddings
OCR
```

#### Architecture

```text
query
  ├── ASR retrieval
  ├── visual retrieval
  └── OCR retrieval
          ↓
    evidence combination
          ↓
       video ranking
          ↓
      temporal result
```

#### Fusion strategy

Start with simple and interpretable methods:

```text
rank fusion
small score-fusion grid
query-type-aware weighting only if clearly justified
```

Avoid complex learned fusion until simpler approaches have been evaluated.

#### Evaluation

Track:

- overall Video R@k
- overall Story R@k
- ASR-heavy queries
- visual-heavy queries
- OCR-heavy queries
- failure recovery
- latency

#### Output

Freeze the final multimodal retrieval architecture.

---

### 5.10 Stage 10 – Freeze and Holdout

#### Goal

Produce one frozen retrieval system and evaluate generalization once.

#### Freeze before holdout

Freeze:

```text
lexical retriever
dense retriever
video aggregation
hybrid policy
ASR
transcript view
hierarchical candidate K
reranker
query representation
multimodal policy
model revisions
source hashes
configs
```

Only after this freeze should `holdout20` be evaluated.

The system must not be modified based on holdout performance.

#### Final outputs

Produce:

```text
final development metrics
holdout metrics
final ablation
runtime summary
failure analysis
frozen reproducibility manifest
```

---

## 6. Artifacts and Reproducibility

### 6.1 Repository and Drive

| Content | Git | Google Drive |
|---|---|---|
| Source code | Yes | Optional mirror |
| Configs | Yes | Optional mirror |
| Benchmark manifests | Yes | Yes |
| Notebooks | Yes | Optional mirror |
| Documentation | Yes | Optional mirror |
| Generated reports | No | Yes |
| ASR outputs | No | Yes |
| Embedding caches | No | Yes |
| Model weights | No | External / cache |
| Audio and video | No | Yes |

Large or generated artifacts remain outside Git. The repository should stay lightweight and focused on code, configs, benchmark definitions, notebooks, and compact documentation.

### 6.2 Experiment manifests

Every significant experiment should preserve:

- exact model revisions
- source hashes
- benchmark identity
- configuration identity
- execution environment
- cache identity
- generated report paths

This allows results to be reproduced without placing large generated artifacts in version control.

### 6.3 Human-readable documentation

The repository uses two human-facing documents:

```text
docs/retrieval-v2.md
Stable protocol, architecture, decisions, and roadmap.

docs/results/retrieval-v2-progress.md
Current dated metrics, findings, selected components, failures, and next steps.
```

Detailed experiment tables, per-query outputs, hashes, and runtime manifests remain in generated reports rather than being duplicated in this document.
