# Retrieval v2 Progress Update – 2026-08-10

> **Status:** Stage 1 and Stage 2 complete. Stage 3 is next.  
> **Selected lexical retriever:** BM25 preserving Vietnamese accents  
> **Selected dense retriever:** multilingual-e5-large-instruct  
> **ASR and text-view selection:** Open  
> **Holdout:** Not evaluated

## 1. Overview

### 1.1 Goal

Retrieval v2 aims to improve ASR-based video and temporal retrieval for the AIC 2026 video-search pipeline. The main weakness of the historical baseline was global video discrimination: the system could often localize the relevant moment once the correct video was known, but identifying that video among the full retrieval corpus remained difficult.

The current work therefore focuses first on improving text retrieval quality, then on improving video-level evidence aggregation, before selecting the final ASR and transcript view.

### 1.2 Current status

| Stage | Purpose | Status | Main outcome |
|---|---|---|---|
| Stage 1 | Improve lexical retrieval | Complete | Accent-preserving BM25 selected |
| Stage 2 | Improve dense retrieval | Complete | multilingual-e5-large-instruct selected |
| Stage 3 | Improve video and temporal aggregation | Next | Not started |
| Stage 4 | Combine sparse and dense retrieval | Pending | Not evaluated |
| Stage 5 | Select ASR and transcript view | Pending | Not evaluated |
| Later stages | Hierarchical retrieval, reranking, query representation, multimodal retrieval | Pending | Not evaluated |

### 1.3 Headline result

The strongest current text-only result uses **multilingual-e5-large-instruct with Whisper raw transcripts**.

| Configuration | Video R@1 | Video MRR | Story R@1 | Story MRR |
|---|---:|---:|---:|---:|
| Historical Baseline v1, Whisper raw | 0.60 | 0.6761 | 0.70 | 0.7831 |
| **E5-large-instruct, Whisper raw** | **0.70** | **0.7601** | **0.85** | **0.8779** |
| E5-large-instruct, Parakeet raw | 0.65 | 0.7265 | 0.70 | 0.8333 |

The current best Parakeet result is close to Whisper on video retrieval, but the ASR decision remains open until later retrieval stages are complete.

---

## 2. Evaluation Setup

### 2.1 Development protocol

| Item | Setting |
|---|---|
| Development queries | 20 |
| Retrieval corpus | 50 videos |
| Transcript windows | 994 |
| Window length | 60 s |
| Window stride | 45 s |
| Window overlap | 15 s |
| Holdout queries | 20, kept closed |

All architecture decisions reported here use **development20** against the full **All50** corpus. The holdout set has not been used for model or hyperparameter selection.

### 2.2 Transcript channels

Four transcript channels are evaluated independently during development:

| ASR | View | Meaning |
|---|---|---|
| Whisper large-v3 | raw | Original ASR transcript |
| Whisper large-v3 | processed | Conservatively cleaned transcript |
| Parakeet CTC 0.6B Vietnamese | raw | Original ASR transcript |
| Parakeet CTC 0.6B Vietnamese | processed | Conservatively cleaned transcript |

These channels are used for diagnosis and robustness checks. **They are not fused.**

### 2.3 Metrics

**Video R@1** is the fraction of queries for which the correct video is ranked first.

**Video MRR** is the mean reciprocal rank of the correct video. It rewards systems that place the correct video near the top even when it is not ranked first.

**Story R@1** and **Story MRR** measure temporal localization within the known correct video.

Higher values are better for all reported retrieval metrics. Because development20 contains only 20 queries, one query changes R@1 by 0.05. Paired per-query results are therefore considered together with aggregate metrics.

---

## 3. Progress by Stage

### 3.1 Stage 1 – Lexical Retrieval

#### Question

Does a proper corpus-based lexical retriever improve retrieval over the historical query-fitted TF-IDF baseline?

#### Methods

| ID | Method | Purpose |
|---|---|---|
| L0 | Query-fitted TF-IDF | Historical lexical control |
| L1 | Corpus-fitted TF-IDF | Correct the TF-IDF fitting source |
| L2 | BM25 preserving accents | Standard lexical retrieval while preserving Vietnamese diacritics |
| L3 | BM25 with accent folding | Test robustness to ASR diacritic errors |
| L4 | RRF of L2 and L3 | Test whether the two BM25 rankings are complementary |

#### Results

| Method | Composite Video RR | Δ vs L0 | Better / Tie / Worse | Composite Story RR |
|---|---:|---:|---:|---:|
| L0 Query TF-IDF | 0.5386 | – | – | 0.7235 |
| L1 Corpus TF-IDF | 0.5476 | +0.0090 | 8 / 6 / 6 | 0.7389 |
| **L2 BM25 preserving** | **0.6064** | **+0.0678** | **9 / 5 / 6** | **0.7839** |
| L3 BM25 folded | 0.4381 | -0.1005 | 7 / 3 / 10 | 0.6797 |
| L4 BM25 RRF | 0.5174 | -0.0211 | 7 / 4 / 9 | 0.7339 |

Corpus-fitted TF-IDF was methodologically cleaner than query-fitted TF-IDF, but the gain was small. Accent-preserving BM25 produced the strongest lexical retrieval. Accent folding substantially degraded retrieval, and fusing the strong and weak BM25 variants through RRF also reduced performance.

#### Decision

**Selected lexical retriever:** `L2_bm25_preserving`  
**Retained alternative:** None

Accent-preserving BM25 is the only lexical method carried forward.

### 3.2 Stage 2 – Dense Retrieval

#### Question

Can a stronger multilingual embedding model improve global video discrimination while preserving temporal localization?

#### Methods

| ID | Dense model | Role |
|---|---|---|
| D0 | multilingual-e5-small | Historical dense control |
| D1 | multilingual-e5-large-instruct | Strong E5 candidate |
| D2 | Qwen3-Embedding-0.6B | Modern multilingual candidate |
| D3 | BGE-M3 dense | Modern multilingual candidate |

All models were evaluated independently on the same four ASR and text-view channels. Video aggregation remained fixed at max-window scoring. No lexical+dense fusion, ASR fusion, transcript-view fusion, reranking, or query rewriting was used.

#### Main results

| Dense model | Composite Video RR | Δ vs D0 | Better / Tie / Worse | 90% bootstrap interval | Composite Story RR |
|---|---:|---:|---:|---:|---:|
| D0 E5-small | 0.5929 | – | – | – | **0.8550** |
| **D1 E5-large-instruct** | **0.7301** | **+0.1372** | **12 / 6 / 2** | **[+0.0569, +0.2151]** | 0.8413 |
| D2 Qwen3-0.6B | 0.6569 | +0.0640 | 8 / 6 / 6 | [-0.0471, +0.1817] | 0.8313 |
| D3 BGE-M3 | 0.5682 | -0.0247 | 5 / 6 / 9 | [-0.0885, +0.0359] | 0.8472 |

E5-large-instruct clearly produced the strongest video retrieval. Its composite Video RR improved by **0.1372** over E5-small, while composite Story RR decreased by only **0.0138**. The gain was broad across the development set: 12 queries improved, six tied, and two worsened. The 90% bootstrap interval for the Video RR improvement remained above zero.

#### E5-large-instruct by transcript channel

| Channel | Video R@1 | Video MRR | Story R@1 | Story MRR |
|---|---:|---:|---:|---:|
| Parakeet processed | 0.60 | 0.6808 | 0.75 | 0.8250 |
| **Parakeet raw** | **0.65** | **0.7265** | 0.70 | 0.8333 |
| Whisper processed | 0.65 | 0.7532 | 0.75 | 0.8288 |
| **Whisper raw** | **0.70** | **0.7601** | **0.85** | **0.8779** |

Whisper raw currently gives the strongest overall text-only result. Parakeet raw is close on video retrieval, with a Video R@1 gap of **0.05** and a Video MRR gap of **0.0336** relative to Whisper raw.

One query, `R2-3`, remains a clear systematic regression for E5-large across the transcript channels and should be tracked in later stages. The second aggregate regression, `R2-8`, is small and mixed across channels.

#### Decision

**Selected dense retriever:** `D1_e5_large_instruct`  
**Retained alternative:** None

Qwen3-Embedding-0.6B improved over E5-small but was less consistent and substantially slower. BGE-M3 did not improve video retrieval and is not carried forward.

---

## 4. Current Best Configuration

### 4.1 Selected components

| Component | Current choice | Status |
|---|---|---|
| Lexical retriever | BM25 preserving Vietnamese accents | Selected |
| Dense retriever | multilingual-e5-large-instruct | Selected |
| Video aggregation | Max window | Temporary baseline |
| ASR | Not selected | Open |
| Transcript view | Not selected | Open |
| Sparse+dense fusion | Not evaluated | Pending |
| Reranker | Not evaluated | Pending |
| Visual and OCR evidence | Not integrated | Pending |

### 4.2 Current retrieval performance

The strongest current dense-only configuration is:

| Setting | Value |
|---|---|
| Dense model | multilingual-e5-large-instruct |
| ASR | Whisper large-v3 |
| Transcript view | Raw |
| Video aggregation | Max window |
| Video R@1 | **0.70** |
| Video MRR | **0.7601** |
| Story R@1 | **0.85** |
| Story MRR | **0.8779** |

The strongest Parakeet configuration currently reaches **Video R@1 = 0.65** and **Video MRR = 0.7265** with raw transcripts. This is already stronger than the historical Whisper raw Baseline v1 on both Video R@1 and Video MRR, showing that retrieval improvements can compensate for a meaningful part of the original ASR gap.

The ASR and transcript-view choices are intentionally not frozen yet.

---

## 5. Efficiency and Operational Results

### 5.1 Dense-model efficiency

All Stage 2 runtime measurements were collected on a Tesla T4.

| Dense model | Index throughput | Warm E2E p50 | Warm E2E p90 | Warm QPS | Peak GPU allocated |
|---|---:|---:|---:|---:|---:|
| E5-small | 191.9 docs/s | 10.46 ms | 11.10 ms | 94.4 | 0.85 GiB |
| **E5-large-instruct** | **103.6 docs/s** | **24.82 ms** | **30.10 ms** | **38.7** | **1.18 GiB** |
| Qwen3-0.6B | 7.57 docs/s | 74.22 ms | 94.52 ms | 13.1 | 2.50 GiB |
| BGE-M3 | 20.24 docs/s | 28.06 ms | 35.38 ms | 33.7 | 2.38 GiB |

E5-large-instruct provides the strongest retrieval quality while remaining substantially faster and lighter than Qwen3 and BGE-M3. Its warm median end-to-end query latency is about **24.8 ms**.

Offline encoding of the 3,963 eligible transcript documents took approximately **38.2 s**, corresponding to **103.6 documents/s**.

### 5.2 Where online latency comes from

For E5-large-instruct:

| Component | Median latency |
|---|---:|
| Query encoding | 24.44 ms |
| Similarity search + ranking + video aggregation | ~0.38 ms |
| Warm end-to-end | 24.82 ms |

Neural query encoding dominates online latency. Searching, ranking, and aggregating all 994 transcript windows takes well below 1 ms. Therefore, the current corpus size is not a retrieval-latency bottleneck.

This also means that more informative video aggregation can be explored in Stage 3 without a strong latency concern, provided the implementation remains efficient.

---

## 6. Key Findings and Open Questions

### 6.1 Key findings

1. **Correcting TF-IDF fitting alone was not enough.** Corpus-fitted TF-IDF was cleaner methodologically but only slightly better than the historical lexical control.

2. **Accent-preserving BM25 is the strongest lexical method.** Accent folding removed useful Vietnamese distinctions and substantially reduced retrieval quality.

3. **Dense representation quality has a larger effect on video retrieval.** E5-large-instruct produced a much larger improvement than the lexical changes alone.

4. **E5-large-instruct has the best quality-efficiency trade-off.** It achieved the strongest video retrieval while remaining much faster and lighter than Qwen3 and BGE-M3.

5. **The Whisper-Parakeet gap has narrowed.** With E5-large-instruct and raw transcripts, the Video R@1 gap is 0.05 and the Video MRR gap is 0.0336.

6. **Global video discrimination remains harder than temporal localization.** Story metrics are already strong, while some queries still rank the correct video poorly because an unrelated window receives a higher score.

### 6.2 Open questions

1. Can temporal support across multiple windows improve video ranking over max-window scoring?
2. Does combining BM25 with E5-large-instruct improve retrieval beyond either method alone?
3. After aggregation and hybrid retrieval improve, is Whisper still sufficiently better than Parakeet to justify its higher ASR cost?
4. Should raw or processed transcripts be retained as the primary text view?
5. Which remaining failures require visual or OCR evidence rather than better ASR-based retrieval?
6. Does the known `R2-3` failure improve under better video aggregation or hybrid retrieval?

---

## 7. Next Steps

### 7.1 Planned retrieval stages

| Next stage | Main question | Planned experiment |
|---|---|---|
| Stage 3 | Can stronger video and temporal aggregation improve global ranking? | Compare max, top-k, and temporally supported evidence using BM25 and E5-large |
| Stage 4 | Are lexical and semantic signals complementary? | Combine selected BM25 and E5-large retrieval through controlled hybrid methods |
| Stage 5 | Which ASR and transcript view should remain? | Re-evaluate Whisper raw/processed and Parakeet raw/processed under the improved retriever |
| Later | Can candidate ranking and representation improve further? | Hierarchical retrieval, reranking, query representation |
| Multimodal stage | Which failures require non-ASR evidence? | Integrate visual and OCR retrieval |
| Final | Does the frozen system generalize? | Run holdout20 once after the architecture is frozen |

Stage 3 should use only the two selected retrieval sources:

```text
Lexical: L2_bm25_preserving
Dense:   D1_e5_large_instruct
```

No Qwen3, BGE-M3, accent-folded BM25, ASR fusion, or transcript-view fusion branches are carried forward.

### 7.2 Reproducibility and source reports

Exact model revisions, source hashes, cache identities, environment details, and per-query outputs remain in the stage artifacts under:

```text
reports/retrieval_v2/stage01_lexical/
reports/retrieval_v2/stage02_dense/
```

The main Stage 2 supporting files are:

```text
retrieval_metrics.csv
method_summary.csv
query_comparison.csv
dense_diagnostics.csv
backend_summary.csv
latency_summary.csv
asr_gap_summary.csv
stage_summary.json
retrieval_manifest.json
```

The holdout set remains closed until the retrieval architecture, ASR choice, transcript view, and selection policy are frozen.
