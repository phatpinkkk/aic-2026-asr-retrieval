# Retrieval v2 Progress Update – 2026-08-11

> **Status:** Stages 1–5 complete. Stage 6 is next.  
> **Selected ASR:** Parakeet CTC 0.6B Vietnamese  
> **Selected transcript view:** Processed  
> **Selected retriever:** 25% BM25 + 75% E5-large-instruct  
> **Selected video aggregation:** Max window  
> **Holdout:** Not evaluated

## 1. Overview

### 1.1 Goal

Retrieval v2 develops the ASR-based text retrieval component of the AIC 2026 video-search system. The work began from a baseline that could often localize a relevant transcript window once the correct video was known, but global video identification remained difficult. Stages 1–5 progressively improved lexical retrieval, dense retrieval, video aggregation, sparse+dense fusion, and finally ASR and transcript-view selection.

### 1.2 Current status

| Stage | Purpose | Status | Final decision |
|---|---|---|---|
| Stage 1 | Lexical retrieval | Complete | `L2_bm25_preserving` |
| Stage 2 | Dense retrieval | Complete | `D1_e5_large_instruct` |
| Stage 3 | Video aggregation | Complete | `P0_max` |
| Stage 4 | Sparse+dense hybrid | Complete | `H3_norm_25_75` |
| Stage 5 | ASR and text view | Complete | Parakeet processed |
| Stage 6 | Hierarchical retrieval | Next | Not evaluated |
| Stage 7–10 | Reranking, representation, multimodal retrieval, holdout | Pending | Not evaluated |

### 1.3 Current selected ASR retrieval pipeline

```text
Parakeet CTC 0.6B Vietnamese
        ↓
processed transcript
        ↓
60 s windows / 45 s stride
        ↓
L2 BM25 preserving
        +
D1 E5-large-instruct
        ↓
per-query eligible-only min-max normalization
        ↓
0.25 BM25 + 0.75 E5
        ↓
max window per video
        ↓
video ranking
```

### 1.4 Headline result

| Checkpoint | Video R@1 | Video MRR | Story R@1 | Story MRR |
|---|---:|---:|---:|---:|
| Historical Baseline v1, Whisper raw | 0.60 | 0.6761 | 0.70 | 0.7831 |
| Stage 2 D1, Whisper raw | 0.70 | 0.7601 | 0.85 | 0.8779 |
| **Stage 5 selected pipeline, Parakeet processed** | **0.75** | **0.8102** | **0.80** | **0.8583** |

The final Stage 1–5 ASR retrieval baseline therefore improves global video retrieval substantially over the historical baseline while using the faster Parakeet ASR.

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

All architecture decisions reported here use **development20** against the full **All50** corpus. Holdout20 has not been used for model, weight, view, or architecture selection.

### 2.2 Metrics

**Video R@1** measures how often the correct video is ranked first. **Video R@5** and **Video R@10** measure whether it appears within the top candidate set. **Video MRR** rewards systems that place the correct video near the top even when it is not ranked first.

**Story R@1** and **Story MRR** measure temporal localization within the known correct video. Video metrics therefore measure global video discrimination, while Story metrics measure within-video localization.

Because development20 contains only 20 queries, one query changes R@1 by 0.05. Aggregate metrics are interpreted together with paired per-query ranks, failure cases, and bootstrap intervals.

### 2.3 Development channels

Stages 1–5 evaluated four transcript channels independently:

```text
Whisper raw
Whisper processed
Parakeet raw
Parakeet processed
```

These channels were used for diagnosis and robustness checks and were **never fused**. Stage 5 selected **Parakeet processed**, so later stages should use only that channel unless a later reviewed decision explicitly changes the baseline.

---

## 3. Progress by Stage

### 3.1 Stage 1 – Lexical Retrieval

#### Question

Does a proper lexical retriever improve retrieval over the historical query-fitted TF-IDF control?

#### Results

| Method | Composite Video RR | Δ vs L0 | Better / Tie / Worse | Composite Story RR |
|---|---:|---:|---:|---:|
| L0 Query TF-IDF | 0.5386 | – | – | 0.7235 |
| L1 Corpus TF-IDF | 0.5476 | +0.0090 | 8 / 6 / 6 | 0.7389 |
| **L2 BM25 preserving** | **0.6064** | **+0.0678** | **9 / 5 / 6** | **0.7839** |
| L3 BM25 folded | 0.4381 | -0.1005 | 7 / 3 / 10 | 0.6797 |
| L4 BM25 RRF | 0.5174 | -0.0211 | 7 / 4 / 9 | 0.7339 |

Corpus-fitted TF-IDF was methodologically cleaner than query-fitted TF-IDF, but the gain was small. Accent-preserving BM25 produced the strongest lexical retrieval. Accent folding removed useful distinctions and substantially reduced performance, while RRF with the weaker folded BM25 variant also hurt retrieval.

**Decision:** `L2_bm25_preserving`

**What we learned:** Vietnamese accent preservation is important for lexical retrieval, and BM25 is a substantially stronger sparse baseline than the historical query-fitted TF-IDF setup.

### 3.2 Stage 2 – Dense Retrieval

#### Question

Can a stronger multilingual embedding model improve global video discrimination while preserving temporal localization?

#### Results

| Dense model | Composite Video RR | Δ vs D0 | Better / Tie / Worse | 90% bootstrap interval | Composite Story RR |
|---|---:|---:|---:|---:|---:|
| D0 E5-small | 0.5929 | – | – | – | **0.8550** |
| **D1 E5-large-instruct** | **0.7301** | **+0.1372** | **12 / 6 / 2** | **[+0.0569, +0.2151]** | 0.8413 |
| D2 Qwen3-0.6B | 0.6569 | +0.0640 | 8 / 6 / 6 | [-0.0471, +0.1817] | 0.8313 |
| D3 BGE-M3 | 0.5682 | -0.0247 | 5 / 6 / 9 | [-0.0885, +0.0359] | 0.8472 |

E5-large-instruct produced the strongest video retrieval and the clearest paired improvement. The gain over E5-small was broad across the development set, while the Story RR decrease was small.

At this stage Whisper raw was the strongest individual channel, with Video R@1 = 0.70 and Video MRR = 0.7601. This was treated as an intermediate result rather than a final ASR decision.

**Decision:** `D1_e5_large_instruct`

**What we learned:** Dense representation quality has a large effect on global video retrieval, and E5-large-instruct provides the strongest quality-efficiency trade-off among the tested dense models.

### 3.3 Stage 3 – Video and Temporal Aggregation

#### Question

Can repeated or temporally supported window evidence improve video ranking over max-window scoring?

#### Results

| Method | BM25 MRR | Dense MRR | Cross-source MRR | Δ vs max |
|---|---:|---:|---:|---:|
| **P0 Max** | **0.6064** | 0.7301 | 0.6683 | – |
| P1 Top-2 mean | 0.6030 | **0.7443** | **0.6737** | +0.0054 |
| P2 Top-3 mean | 0.5647 | 0.6776 | 0.6212 | -0.0471 |
| P3 Adjacent-2 | 0.6032 | 0.7107 | 0.6570 | -0.0113 |
| P4 Contiguous-3 | 0.5749 | 0.6647 | 0.6198 | -0.0484 |

Top-2 mean produced a small cross-source gain and showed that repeated evidence can sometimes suppress isolated false-positive peaks. However, the improvement was inconsistent across retrieval sources and transcript views, and its bootstrap interval crossed zero.

Adjacent and contiguous pooling did not generalize. Requiring three strong windows clearly diluted relevant evidence for short or highly localized spoken moments. Fixed temporal adjacency between overlapping ASR windows is therefore not a reliable substitute for semantic event continuity.

**Decision:** `P0_max`

**What we learned:** Strong partial relevance remains important. A single transcript window can legitimately contain most of the useful spoken evidence for a video.

### 3.4 Stage 4 – Sparse + Dense Hybrid Retrieval

#### Question

Does combining lexical and semantic retrieval improve over either source independently?

#### Results

| Method | Video MRR | Video R@1 | Story MRR | Δ Video MRR vs dense |
|---|---:|---:|---:|---:|
| H0 Sparse | 0.6064 | 0.4875 | 0.7839 | -0.1238 |
| H1 Dense | 0.7301 | 0.6500 | 0.8413 | – |
| H2 RRF | 0.7225 | 0.6125 | 0.8342 | -0.0077 |
| **H3 25/75 normalized** | **0.7841** | **0.7125** | **0.8777** | **+0.0539** |
| H4 50/50 | 0.7482 | 0.6500 | 0.8014 | +0.0180 |
| H5 75/25 | 0.6700 | 0.5500 | 0.7976 | -0.0601 |

`H3_norm_25_75` produced the strongest overall retrieval. Its Video RR improvement over dense-only had a positive 90% bootstrap interval of **[+0.0014, +0.1106]**. The result shows that dense retrieval should remain the main signal, while BM25 works best as a smaller lexical correction.

The query-level behavior also clarified several persistent failures. `R2-1` worsened systematically when lexical evidence was added. `R3-2` and `R3-20` benefited strongly from the hybrid. `R2-3` reached strong Story localization but still struggled with global video ranking.

**Decision:** `H3_norm_25_75`

**What we learned:** Sparse and dense evidence are complementary, but the sparse signal should remain secondary. Rank fusion and heavier sparse weighting were less effective than normalized score fusion.

### 3.5 Stage 5 – ASR and Text-View Selection

#### Question

After improving retrieval itself, which ASR and transcript view should feed the final Stage 1–5 text retrieval baseline?

#### Results

| Channel | Video R@1 | Video R@5 | Video R@10 | Video MRR | Story R@1 | Story MRR |
|---|---:|---:|---:|---:|---:|---:|
| **Parakeet processed** | **0.75** | **0.85** | 0.90 | **0.8102** | 0.80 | 0.8583 |
| Parakeet raw | 0.70 | **0.85** | 0.90 | 0.7807 | **0.85** | **0.8950** |
| Whisper processed | 0.70 | **0.85** | 0.90 | 0.7733 | **0.85** | 0.8837 |
| Whisper raw | 0.70 | 0.80 | 0.90 | 0.7721 | 0.80 | 0.8738 |

Parakeet processed produced the strongest global video retrieval. Against Whisper processed, it passed every pre-frozen video and Story quality threshold and introduced no catastrophic same-view ASR regression.

Its operational advantage is also large. The measured Parakeet ASR RTF advantage is **38.1× on Core10** and **33.8× on Extension40**, with lower peak GPU memory.

**Decision:** Parakeet CTC 0.6B Vietnamese with the processed transcript view.

**What we learned:** The earlier Whisper advantage was partly an interaction between transcript output and retrieval architecture. Once sparse and dense retrieval were combined, Parakeet became the strongest video-retrieval channel rather than merely a cheaper approximation.

Stage 5 closes the four-channel diagnostic phase. Stage 6 onward should use only **Parakeet processed**.

---

## 4. Current Frozen ASR Retrieval System

### 4.1 Frozen components

| Component | Frozen choice |
|---|---|
| ASR | Parakeet CTC 0.6B Vietnamese |
| Transcript view | Processed |
| Windowing | 60 s, stride 45 s, overlap 15 s |
| Sparse retriever | `L2_bm25_preserving` |
| Dense retriever | `D1_e5_large_instruct` |
| Score normalization | Per-query, eligible-only min-max |
| Sparse weight | 0.25 |
| Dense weight | 0.75 |
| Video aggregation | `P0_max` |
| Query representation | Original natural-language query |

### 4.2 Development performance

| Metric | Result |
|---|---:|
| Video R@1 | **0.75** |
| Video R@5 | **0.85** |
| Video R@10 | **0.90** |
| Video MRR | **0.8102** |
| Story R@1 | **0.80** |
| Story R@5 | **0.95** |
| Story R@10 | **1.00** |
| Story MRR | **0.8583** |

Implementation details for this frozen Stage 1–5 subsystem are documented separately in `docs/asr-retrieval-system.md`.

---

## 5. Efficiency and Operational Results

### 5.1 Selected ASR cost

| Benchmark | Whisper RTF | Parakeet RTF | Parakeet speedup |
|---|---:|---:|---:|
| Core10 | 0.3257 | 0.00855 | 38.1× |
| Extension40 | 0.3047 | 0.00901 | 33.8× |

Whisper peak GPU memory was approximately **6.75 GB**, compared with approximately **4.92 GB** for Parakeet. Both ASR systems completed the measured workloads without failed or empty windows.

### 5.2 Retrieval cost

For `D1_e5_large_instruct`, median query encoding latency is approximately **24.44 ms**, and the measured dense online reference is approximately **24.8 ms p50** end to end.

The Stage 4 H3 fusion operation itself costs about **0.1 ms per query** in cached-score evaluation. This is a separate measurement scope from the dense online benchmark, so the two values should not be interpreted as one directly measured production latency.

The current system is therefore dominated by dense query encoding rather than score fusion.

---

## 6. Key Findings and Remaining Limitations

### 6.1 Key findings

1. **Dense retrieval produced the largest early improvement.** E5-large-instruct substantially improved global video ranking over the historical dense baseline.
2. **BM25 contributes useful complementary evidence when kept at low weight.** A 25% sparse contribution improves both Video MRR and Story MRR over dense-only retrieval.
3. **Max-window scoring remains the most reliable video aggregation policy.** Requiring multiple strong or adjacent ASR windows often dilutes short but valid evidence.
4. **Fixed temporal adjacency is not semantic event continuity.** Neighboring overlapping ASR windows do not reliably represent one coherent event.
5. **Retrieval architecture changed the ASR conclusion.** Parakeet was behind Whisper under dense-only retrieval but became the strongest video channel after sparse+dense fusion.
6. **Global video discrimination remains the main ASR-text bottleneck.** Temporal localization is often strong even when the correct video is ranked poorly.

### 6.2 Remaining limitations

- `R2-1` remains sensitive to misleading lexical evidence.
- `R2-3` can localize the relevant Story window but still struggles with global video discrimination.
- `R2-8` remains difficult across multiple retrieval configurations.
- Queries that depend mainly on visual appearance, scene text, or OCR cannot be solved reliably by ASR text alone.
- Current architecture decisions are based on development20.
- Holdout20 remains closed.

---

## 7. Next Steps

| Stage | Main question |
|---|---|
| Stage 6 | Does hierarchical video → temporal retrieval improve candidate handling or later reranking? |
| Stage 7 | Can a second-stage reranker correct difficult text-ranking errors? |
| Stage 8 | Can better query or transcript representations improve retrieval? |
| Stage 9 | Which remaining failures require visual and OCR evidence? |
| Stage 10 | Does the fully frozen system generalize to holdout20? |

Stages 6–10 start from one frozen ASR retrieval baseline:

```text
Parakeet processed
+
L2 BM25 preserving
+
D1 E5-large-instruct
+
H3 normalized 25/75 fusion
+
P0 max video aggregation
```

Large per-query outputs, cache identities, source hashes, runtime details, and reproducibility metadata remain in the stage-specific artifacts under `reports/retrieval_v2/` and `cache/retrieval_v2/`.

**Holdout20 remains closed until the full retrieval architecture is frozen.**
