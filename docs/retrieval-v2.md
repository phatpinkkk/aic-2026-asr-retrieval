# Retrieval v2: Methodology and Results

Retrieval v2 is the controlled research programme that produced the current ASR-text retrieval system. This document explains the starting problem, the evaluation protocol, what each of Stages 1–7 changed, what the experiments found, and which conclusions are reliable enough to carry forward.

Implementation details are documented separately in [`asr-retrieval-system.md`](asr-retrieval-system.md).

## 1. Motivation and Starting Point

The historical text-retrieval baseline could often identify a useful transcript region once the correct video was already known, but ranking the correct video against unrelated videos was much less reliable. Retrieval v2 therefore focused first on improving global video retrieval, then on adding stronger candidate reranking without making query cost unbounded.

The historical Baseline v1 used:

| Component | Historical baseline |
|---|---|
| Lexical retrieval | Query-fitted character TF-IDF |
| Dense retrieval | `intfloat/multilingual-e5-small` |
| Combination | Fixed 50/50 lexical-semantic score average |
| Video scoring | Maximum-scoring window |
| Transcript conditions | Whisper / Parakeet, raw / processed |

Query-fitted TF-IDF was useful as a frozen reference but was not a good long-term lexical design because its vocabulary was fitted from the active query set rather than the transcript corpus. The dense model was also relatively small, and equal weighting assumed that lexical and semantic scores should contribute equally even though they have different strengths and numeric scales.

Retrieval v2 did not replace everything at once. Each stage changed one main part of the system, compared a small predeclared set of alternatives, and then froze the reviewed decision before later stages depended on it.

The seven stages were:

```text
Stage 1  lexical retrieval
Stage 2  dense retrieval
Stage 3  window → video scoring
Stage 4  BM25 + E5 combination
Stage 5  ASR and transcript representation
Stage 6  candidate selection for reranking
Stage 7  cross-encoder reranking
```

This staged design matters because the conclusion about one component can depend on the rest of the retrieval architecture. Stage 5 is the clearest example: Parakeet became the selected ASR only after the first-stage retriever had been improved.

## 2. Evaluation Setup and Research Rules

### 2.1 Benchmarks and query chronology

Retrieval v2 uses three related benchmark views:

| Benchmark | Videos | Queries | Windows | Main role |
|---|---:|---:|---:|---|
| Core10 | 10 | 10 | 219 | Regression and implementation checks |
| Extension40 | 40 | 40 | 775 | Main query set |
| All50 | 50 | — | 994 | Retrieval corpus formed from Core10 + Extension40 videos |

The Extension40 queries were originally divided into `development20` and `holdout20`.

Stages 1–5 selected the retrieval architecture using **development20 against the full All50 corpus**. Searching against all 50 videos was important because evaluating a query only against a smaller local set would make global video ranking artificially easy.

`holdout20` remained closed through the Stage 1–5 decisions. After that subsystem was frozen, the holdout was evaluated once. The frozen Stage 1–5 system produced:

| Holdout20 metric | Result |
|---|---:|
| Video R@1 | 0.70 |
| Video R@5 | 0.95 |
| Video R@10 | 0.95 |
| Video MRR | 0.8119 |
| Story R@1 | 0.90 |
| Story MRR | 0.9500 |

Once these results had been observed, `holdout20` was no longer an untouched holdout. Stages 6–7 therefore used **all 40 Extension40 queries** (`extension40_full`) against All50 for development and comparison. The old development/holdout labels were retained only for diagnostic slices; they are not independent validation splits for the later-stage decisions.

### 2.2 What Video and Story metrics mean

The evaluation deliberately separates two problems.

**Video retrieval** asks whether the correct video is ranked near the top of the complete corpus. It is measured with Video R@1, R@3, R@5, R@10, R@20, and MRR.

**Story retrieval** asks whether a relevant moment is ranked highly **inside the known correct video**. A transcript window is considered relevant when it belongs to the correct video and overlaps the query's ±60-second answer zone by at least 30 seconds. Story retrieval is measured with R@1, R@3, R@5, R@10, and MRR.

This distinction is important. A model can be good at locating the moment once it knows the video but still be poor at separating that video from other videos.

MRR is especially useful because it rewards moving the correct item closer to the top even when R@1 does not change. However, aggregate metrics are never interpreted alone. The reports also keep per-query ranks, better/tie/worse counts, rank-1 recoveries and losses, score margins, failure categories, and paired bootstrap intervals.

With only 20 queries in the early development split, one query changes R@1 by five percentage points. That is why paired behavior matters.

### 2.3 Transcript conditions and experimental discipline

Stages 1–5 evaluated four transcript conditions independently:

```text
Whisper raw
Whisper processed
Parakeet raw
Parakeet processed
```

They were diagnostic channels, not four inputs to one fused system. The project intentionally avoided Whisper+Parakeet or raw+processed fusion because the first question was whether one sufficiently good transcript pipeline could support retrieval without extra complexity.

Each stage reused the same physical window axis and the same evaluation rules. Once a method was frozen, later stages depended on that decision through explicit prerequisite checks and cached-artifact identities. If a prerequisite configuration changes, the later stage is expected to fail rather than silently continue with stale results.

## 3. Building the First-Stage Retriever: Stages 1–5

### 3.1 Stage 1 — Replacing query-fitted TF-IDF

The first question was whether the historical lexical component could be replaced by a proper corpus-oriented retriever.

Five methods were compared:

```text
L0  query-fitted character TF-IDF
L1  corpus-fitted character TF-IDF
L2  BM25 preserving Vietnamese accents
L3  accent-folded BM25
L4  RRF of preserving and folded BM25
```

The development results were summarized across the four ASR/view conditions:

| Method | Composite Video RR | Δ vs L0 | Better / Tie / Worse | Composite Story RR |
|---|---:|---:|---:|---:|
| L0 query TF-IDF | 0.5386 | — | — | 0.7235 |
| L1 corpus TF-IDF | 0.5476 | +0.0090 | 8 / 6 / 6 | 0.7389 |
| **L2 BM25, accents preserved** | **0.6064** | **+0.0678** | **9 / 5 / 6** | **0.7839** |
| L3 accent-folded BM25 | 0.4381 | -0.1005 | 7 / 3 / 10 | 0.6797 |
| L4 BM25 RRF | 0.5174 | -0.0211 | 7 / 4 / 9 | 0.7339 |

Corpus-fitted TF-IDF fixed the methodological weakness of query fitting but produced only a small gain. Accent-preserving BM25 was clearly the strongest lexical option.

Accent folding was tested because ASR can make diacritic errors, but the result went in the opposite direction: removing accents destroyed useful lexical distinctions and reduced retrieval quality substantially. Combining the strong and weak BM25 variants with RRF did not recover that loss.

**Decision:** accent-preserving BM25 (`L2_bm25_preserving`).

The main lesson from Stage 1 is straightforward: Vietnamese lexical detail is valuable enough that we should preserve it rather than pre-emptively remove it for noise tolerance.

### 3.2 Stage 2 — Stronger multilingual dense retrieval

Stage 2 asked whether a stronger embedding model could improve semantic retrieval beyond multilingual E5-small.

The tested models were:

```text
D0  multilingual-e5-small
D1  multilingual-e5-large-instruct
D2  Qwen3-Embedding-0.6B
D3  BGE-M3 dense
```

All other major retrieval choices remained fixed.

| Dense model | Composite Video RR | Δ vs D0 | Better / Tie / Worse | 90% bootstrap interval | Composite Story RR |
|---|---:|---:|---:|---:|---:|
| D0 E5-small | 0.5929 | — | — | — | **0.8550** |
| **D1 E5-large-instruct** | **0.7301** | **+0.1372** | **12 / 6 / 2** | **[+0.0569, +0.2151]** | 0.8413 |
| D2 Qwen3 embedding | 0.6569 | +0.0640 | 8 / 6 / 6 | [-0.0471, +0.1817] | 0.8313 |
| D3 BGE-M3 dense | 0.5682 | -0.0247 | 5 / 6 / 9 | [-0.0885, +0.0359] | 0.8472 |

E5-large-instruct produced the strongest video retrieval and the clearest paired improvement. Its 90% bootstrap interval for the Video RR gain over E5-small stayed above zero. Story RR was slightly lower than E5-small, but the global video gain was much larger and was the main bottleneck the project was trying to improve.

**Decision:** `D1_e5_large_instruct`.

This was the largest early gain in Retrieval v2 and established dense semantic retrieval as the main first-stage signal.

### 3.3 Stage 3 — Turning transcript-window scores into video scores

A retrieval model produces a score for each transcript window, but the competition also needs a video ranking. Stage 3 tested whether a video should be represented by its single strongest window or by repeated/neighboring support.

The candidates were:

```text
P0  maximum window score
P1  mean of top 2 windows
P2  mean of top 3 windows
P3  best adjacent-window pair mean
P4  best contiguous 3-window mean
```

The same aggregation policies were tested independently on the selected BM25 and E5 score sources.

| Aggregation | BM25 Video RR | E5 Video RR | Cross-source Video RR |
|---|---:|---:|---:|
| **P0 max** | **0.6064** | 0.7301 | 0.6683 |
| P1 top-2 mean | 0.6030 | **0.7443** | **0.6737** |
| P2 top-3 mean | 0.5647 | 0.6776 | 0.6212 |
| P3 adjacent-2 mean | 0.6032 | 0.7107 | 0.6570 |
| P4 contiguous-3 mean | 0.5749 | 0.6647 | 0.6198 |

Top-2 mean produced a small cross-source increase of about 0.0054, but the improvement was not consistent across both retrieval sources and its paired bootstrap interval crossed zero. More aggressive averaging was clearly harmful.

The behavior is understandable. A query may describe only a brief spoken moment. If one window contains that moment and the neighboring windows discuss something else, averaging several windows dilutes correct evidence. Temporal adjacency also does not guarantee semantic continuity, especially with fixed overlapping windows.

**Decision:** maximum-window scoring (`P0_max`).

The main lesson is that partial relevance should be preserved: one strong local passage can legitimately identify the correct video.

### 3.4 Stage 4 — Combining lexical and semantic retrieval

BM25 and E5 solve different parts of the problem. Stage 4 tested whether combining them improved on either source alone.

Because their raw score scales are different, the weighted methods normalize BM25 and E5 separately for each query over eligible windows before combining them.

| Method | Video MRR | Video R@1 | Story MRR | Δ Video MRR vs dense |
|---|---:|---:|---:|---:|
| H0 BM25 only | 0.6064 | 0.4875 | 0.7839 | -0.1238 |
| H1 E5 only | 0.7301 | 0.6500 | 0.8413 | — |
| H2 RRF | 0.7225 | 0.6125 | 0.8342 | -0.0077 |
| **H3 25% BM25 / 75% E5** | **0.7841** | **0.7125** | **0.8777** | **+0.0539** |
| H4 50/50 | 0.7482 | 0.6500 | 0.8014 | +0.0180 |
| H5 75% BM25 / 25% E5 | 0.6700 | 0.5500 | 0.7976 | -0.0601 |

The selected 25/75 fusion improved both Video and Story MRR over dense-only retrieval. Its Video RR improvement over E5 alone had a positive 90% paired bootstrap interval of approximately `[+0.0014, +0.1106]`.

The weight pattern is also informative. Equal weighting helped less, and making BM25 the larger signal was harmful. Exact lexical evidence is useful, but semantic retrieval should remain dominant.

**Decision:** per-query normalized 25% BM25 + 75% E5 (`H3_norm_25_75`).

### 3.5 Stage 5 — Selecting the ASR and transcript representation

Only after the first-stage retrieval architecture had improved did the project make the final Stage 1–5 ASR decision. This avoided selecting an ASR based on weaknesses that actually belonged to the old retriever.

The frozen Stage 4 system was applied to the four independent ASR/view conditions:

| Channel | Video R@1 | Video R@5 | Video R@10 | Video MRR | Story R@1 | Story MRR |
|---|---:|---:|---:|---:|---:|---:|
| **Parakeet processed** | **0.75** | **0.85** | 0.90 | **0.8102** | 0.80 | 0.8583 |
| Parakeet raw | 0.70 | **0.85** | 0.90 | 0.7807 | **0.85** | **0.8950** |
| Whisper processed | 0.70 | **0.85** | 0.90 | 0.7733 | **0.85** | 0.8837 |
| Whisper raw | 0.70 | 0.80 | 0.90 | 0.7721 | 0.80 | 0.8738 |

Parakeet processed produced the strongest global video retrieval. The processed representation did not maximize every Story metric—Parakeet raw had higher Story MRR—but the project prioritized the larger global-video problem while enforcing predeclared quality gates to avoid unacceptable Story regressions.

Parakeet also had a major operational advantage:

| Benchmark | Whisper RTF | Parakeet RTF | Parakeet speedup |
|---|---:|---:|---:|
| Core10 | 0.3257 | 0.00855 | 38.1× |
| Extension40 | 0.3047 | 0.00901 | 33.8× |

Peak GPU memory was also lower for Parakeet (about 4.92 GB versus about 6.75 GB for Whisper in the measured runs).

**Decision:** Parakeet CTC 0.6B Vietnamese with the processed transcript representation.

After this decision, the first-stage pipeline was:

```text
Parakeet processed transcript
        ↓
accent-preserving BM25
        +
E5-large-instruct
        ↓
per-query eligible-only min-max normalization
        ↓
25% BM25 + 75% E5
        ↓
maximum window score per video
```

On all 40 Extension40 queries, after the one-time holdout exposure, this first-stage system produced:

| Full40 metric | Result |
|---|---:|
| Video R@1 | 0.725 |
| Video R@3 | 0.900 |
| Video R@5 | 0.900 |
| Video R@10 | 0.925 |
| Video R@20 | 0.950 |
| Video MRR | 0.8110 |
| Story R@1 | 0.850 |
| Story R@3 | 0.925 |
| Story R@5 | 0.975 |
| Story R@10 | 1.000 |
| Story MRR | 0.9042 |

## 4. Candidate Selection and Reranking: Stages 6–7

Stages 6 and 7 solve one practical problem: a stronger cross-encoder can improve difficult rankings, but it is too expensive to apply indiscriminately to every transcript window.

### 4.1 Stage 6 — Reducing the reranker workload

Stage 6 did **not** try to improve MRR. It measured how aggressively the 994-window All50 corpus could be reduced while preserving the evidence a later reranker might need.

For every query, the frozen first stage first ranked videos using the maximum hybrid window score. A candidate policy then kept the top `K` videos and the top `M` windows within each selected video.

The main policies were combinations of:

```text
K ∈ {10, 20, 30}
M ∈ {1, 3, 5}
```

A K=40 fallback was also evaluated because none of the original policies reached perfect joint video+window coverage.

The results show separately why both K and M matter:

| Policy | Mean pairs/query | Video candidate recall | Relevant-window recall | Joint recall |
|---|---:|---:|---:|---:|
| K10_M5 | 49.6 | 0.925 | 0.975 | 0.925 |
| K20_M5 | 99.4 | 0.950 | 0.975 | 0.950 |
| **K30_M5** | **149.3** | **1.000** | **0.975** | **0.975** |
| K40_M5 | 197.5 | 1.000 | 0.975 | 0.975 |
| K30_M3 | 89.9 | 1.000 | 0.925 | 0.925 |
| K30_M1 | 30.0 | 1.000 | 0.850 | 0.850 |

Three difficult queries explain the required video breadth:

```text
R2-15  correct video rank 14
R2-1   correct video rank 24
R2-8   correct video rank 27
```

K=10 loses all three. K=20 recovers `R2-15`, and K=30 keeps all correct videos.

Increasing M solves a different problem. At M=1, six queries lose the relevant temporal window even when the correct video is retained. M=3 reduces this to three; M=5 leaves one persistent temporal miss.

The remaining miss is `R2-8`. Its correct video is present at Stage 7, but none of the selected top-five windows satisfies the benchmark temporal relevance rule. Stage 7 can still improve its **video** rank, but perfect joint video+moment coverage is impossible under this candidate pool.

`K30_M5` and `K40_M5` have the same 0.975 joint recall. K40 requires about 32% more pairs, so the additional videos do not buy any benchmark coverage.

**Decision:** `K30_M5`.

This reduces the reranker workload from all 994 windows to about 149 query-window pairs while preserving all 40 correct videos and relevant temporal evidence for 39 of 40 queries.

This result is specific to All50. K30 retains 60% of a 50-video corpus; it would retain only about 2% of a roughly 1,490-video corpus. The correct production K therefore remains an open scaling question.

### 4.2 Stage 7 — Reranking the candidate set

Stage 7 evaluated six local rerankers on the exact same frozen `K30_M5` pool and Tesla T4 environment:

```text
R1  mMARCO MiniLM
R2  GTE multilingual
R3  BGE reranker v2 M3
R4  Qwen3-Reranker-0.6B
R5  Mixedbread mxbai-rerank-base-v2
R6  Jina reranker v2 multilingual
```

All models used the same 512-token maximum input length. The runner performed a compatibility smoke test, a runtime preflight, batch-size fallback if needed, token-length auditing, and then the complete Full40 run. In the final valid environment all six rerankers passed and used batch size 32 without OOM retries.

Three score policies were compared conceptually:

```text
S0  frozen first-stage control

S1  reranker-only candidate score

S2  50% normalized first-stage candidate score
    + 50% normalized reranker score
```

There was no reranker-weight sweep.

The most important finding was not the identity of the best model. It was that **reranker-only scoring was consistently worse than the first-stage control**:

| Reranker-only model | Video MRR |
|---|---:|
| BGE | 0.7833 |
| Qwen3 | 0.7014 |
| Mixedbread | 0.6927 |
| GTE | 0.6322 |
| Jina | 0.5319 |
| MiniLM | 0.4289 |
| **First-stage control** | **0.8110** |

The first stage therefore contains useful lexical+dense evidence that should not be discarded. Cross-encoding works better as a refinement signal.

With the predefined S2 fusion:

| Reranker | Video R@1 | Video MRR | Story MRR | Better / Tie / Worse | p90 | Meets 5 s limit |
|---|---:|---:|---:|---:|---:|---|
| No reranker | 0.725 | 0.8110 | 0.9042 | — | — | Yes |
| MiniLM | 0.625 | 0.7154 | 0.8729 | 5 / 25 / 10 | 0.34 s | Yes |
| GTE | 0.700 | 0.7972 | 0.8988 | 5 / 29 / 6 | 1.04 s | Yes |
| **BGE v2 M3** | **0.750** | **0.8205** | **0.9125** | **6 / 30 / 4** | **2.54 s** | **Yes** |
| Qwen3-0.6B | **0.825** | **0.8642** | 0.8625 | **8 / 28 / 4** | 7.31 s | No |
| Mixedbread | 0.725 | 0.7955 | 0.8896 | 4 / 29 / 7 | 4.57 s | Yes |
| Jina v2 | 0.675 | 0.7543 | 0.9113 | 4 / 25 / 11 | 1.02 s | Yes |

Qwen was the clear video-quality leader. It raised Video R@1 from 0.725 to 0.825 and Video MRR from 0.8110 to 0.8642. It also had five rank-1 recoveries and one rank-1 loss. Its paired 90% bootstrap interval for Video RR improvement was positive, approximately `[+0.0023, +0.1082]`.

However, Qwen had two important costs. Its p90 reranking latency was 7.31 seconds, above the 5-second limit fixed before Stage 7, and Story MRR fell from 0.9042 to 0.8625. It therefore cannot be the operational choice under the frozen runtime rule.

BGE produced a much smaller gain: Video MRR increased by about 0.0095 and Story MRR by about 0.0083. It had six better queries, 30 ties, four worse queries, two rank-1 recoveries, one rank-1 loss, and no large or hard top-1 regressions. Its Video RR bootstrap interval crossed zero, so the quality improvement should be described as modest rather than conclusive.

Its advantage is balance: BGE is the only tested reranker that improves both aggregate Video and Story MRR while staying within the 5-second limit.

**Stage 7 status:** evaluation complete and passed.

**Operational recommendation:** BGE reranker v2 M3 with S2 50/50 fusion.

**Quality-oriented challenger:** Qwen3-0.6B, retained as evidence of the attainable video-quality gain but not operationally eligible under the current latency rule.

The current configuration still marks `selection.stage07_decision` as `pending_review`, so this document deliberately calls BGE a recommendation rather than a formally frozen selection.

## 5. What We Learned and What Remains Uncertain

### 5.1 Current Stage 1–7 design

The research has converged on the following design:

| Component | Current outcome |
|---|---|
| ASR | Parakeet CTC 0.6B Vietnamese |
| Audio windows | 60 s, 45 s stride, 15 s overlap |
| Transcript representation | Processed |
| Lexical retrieval | Accent-preserving BM25 |
| Dense retrieval | E5-large-instruct |
| First-stage normalization | Per-query, eligible-window min-max |
| First-stage fusion | 25% BM25 + 75% E5 |
| Video score | Maximum window score |
| Stage 6 benchmark candidate policy | `K30_M5` |
| Stage 7 candidate fusion | 50% first-stage + 50% reranker |
| Stage 7 operational recommendation | BGE v2 M3, final freeze pending |

Several broader lessons are more important than the method IDs.

First, **dense semantic retrieval provides most of the first-stage strength**, but a small BM25 contribution is useful for exact names, locations, numbers, and unusual words. The best fusion is therefore asymmetric rather than 50/50.

Second, **short local evidence matters**. The max-window policy survived Stage 3 because requiring repeated or neighboring support often diluted the one passage that actually described the target event.

Third, **ASR quality cannot be judged independently of retrieval architecture**. Whisper appeared stronger in some earlier configurations, but once the retriever improved, Parakeet processed became the best global video channel while also being more than 30× faster in the measured ASR runs.

Fourth, **candidate selection and reranking solve different problems**. Stage 6 protects recall and controls compute; Stage 7 changes the ordering of candidates. A candidate that is removed before Stage 7 cannot be recovered by a stronger reranker.

Finally, **the reranker should refine the first stage rather than replace it**. This is the clearest Stage 7 architectural result: all six reranker-only variants lost Video MRR.

### 5.2 Failure patterns

The remaining errors fall into a few useful categories.

A **deep first-stage video error** occurs when the correct video is far down the global ranking. Stage 6 exposed examples at ranks 14, 24, and 27. These are the cases that force candidate-video breadth to increase.

A **candidate-window miss** occurs when the correct video survives but the relevant moment is not among the windows passed to the cross-encoder. `R2-8` is the persistent Stage 6 example under `K30_M5`.

A **reranker failure/regression** occurs when the necessary candidate evidence is present but the reranker still ranks a wrong video above the correct one, or pushes a previously strong result down. Stage 7 tracks these separately from candidate misses so upstream and reranker errors are not confused.

A fourth limitation is outside ranking itself: **the transcript may not contain the evidence required by the query**. ASR retrieval cannot solve information that is only visual or only present as on-screen text. These cases motivate the separate visual/OCR parts of the broader system rather than further tuning of the ASR retriever.

### 5.3 Runtime and statistical caution

The first-stage system is already relatively cheap on All50. E5-large query encoding is measured in tens of milliseconds, while Stage 7 cross-encoding is measured in seconds. If BGE is enabled, reranking becomes the dominant online neural cost.

This makes the size of the candidate pool an operational parameter, not just an accuracy parameter.

The sample size also limits how strongly small differences should be interpreted. BGE's Full40 aggregate improvement is positive, but its paired bootstrap interval crosses zero and its historical `development20` and `holdout20` slices behave differently. The Stage 7 evidence supports BGE as the best balanced operational choice among the tested rerankers, but it does not justify claiming a large or universally stable improvement.

Qwen's video improvement is stronger and more consistent, which tells us that the candidate pool contains recoverable information. Its current problem is the quality/latency/Story trade-off rather than a lack of ranking capability.

### 5.4 Scaling beyond All50

The largest unresolved question is how the candidate layer behaves on the full competition corpus.

Stage 6 selected `K30_M5` on 50 videos. Keeping 30 of 50 videos is generous; keeping 30 of roughly 1,490 is extremely selective. The correct video may fall below rank 30 simply because many more distractors are present.

The full-corpus deployment should therefore remeasure:

```text
first-stage Video Recall@10 / @20 / @30 / @50 / @100
candidate-video recall
relevant-window recall
joint candidate recall
reranker pair count
warm query latency
```

The goal should not be to preserve the literal `K30_M5` rectangle. The goal is to preserve enough video and temporal evidence while keeping the expensive reranker workload bounded.

A possible engineering direction is to give at least one window to a broader set of videos and allocate additional windows to higher-ranked videos under a fixed total pair budget. This is a deployment hypothesis, not a Stage 1–7 experimental result, and it should be validated on full-corpus queries before adoption.

Retrieval v2 stops here at Stage 7. The current work provides a well-tested ASR-text retrieval core, a clear candidate/reranker interface, measured quality/latency trade-offs, and a concrete list of scaling checks that should be completed before the same settings are used unchanged on the full competition corpus.
