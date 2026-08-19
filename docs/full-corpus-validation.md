# Full-Corpus Production Validation

This document summarizes the final validation of the production ASR retrieval system on the complete competition corpus. It complements [`asr-retrieval-system.md`](asr-retrieval-system.md), which describes the implementation, and [`retrieval-v2.md`](retrieval-v2.md), which documents the research process that produced the selected retrieval design.

The validation was run with [`production/notebooks/06_full_corpus_validation.ipynb`](../production/notebooks/06_full_corpus_validation.ipynb). The goal was not to reopen component selection or tune new parameters. It was to measure how the frozen production system performs when searching the real full corpus rather than the earlier 50-video research corpus.

## 1. Validation Setup

The final release was validated before evaluation, including its file hashes and retrieval artifacts.

| Item | Full-corpus validation |
|---|---:|
| Release | `aic2026-full-20260817-r01` |
| Videos | 1,478 |
| Physical transcript windows | 26,163 |
| Searchable transcript windows | 26,117 |
| E5 embedding shape | 26,117 × 1,024 |
| Evaluation queries | 40 Extension40 queries |
| Target videos present | 40 / 40 |
| Valid answer times | 40 / 40 |
| Release validation | PASS |

The evaluated system is the final production pipeline: processed Parakeet transcripts, BM25 + E5 first-stage retrieval, video-level ranking, bounded candidate selection, BGE v2 M3 reranking, and final score fusion. The evaluation treats this pipeline as one complete system rather than evaluating its components separately.

## 2. Full-Corpus Video Retrieval

The correct video was ranked against all 1,478 videos for every query.

| Metric | Result | Queries retrieved |
|---|---:|---:|
| Video R@1 | 42.5% | 17 / 40 |
| Video R@3 | 47.5% | 19 / 40 |
| Video R@5 | 55.0% | 22 / 40 |
| Video R@10 | 70.0% | 28 / 40 |
| Video R@20 | 75.0% | 30 / 40 |
| Video R@50 | 87.5% | 35 / 40 |
| Video R@100 | 90.0% | 36 / 40 |
| Video MRR | 0.491 | – |
| Median target rank | 4 | – |
| Mean target rank | 44.98 | – |
| Worst target rank | 512 | – |

The median target rank of 4 shows that the correct video is usually placed near the top of the full corpus. The much larger mean rank is caused by a small number of severe misses rather than consistently weak ranking.

The most useful operational figure is **R@50 = 87.5%**. The ASR subsystem retains the correct video within its top 50 for 35 of the 40 evaluation queries, even though it is searching 1,478 videos.

## 3. Temporal Supporting-Window Quality

The validation also checked whether the supporting transcript windows returned for the known target video point to the annotated answer region.

A supporting window is counted as temporally correct when it overlaps the `answer_time ± 60 s` region by at least 30 seconds.

| Metric | Result |
|---|---:|
| Moment Hit@1 | 87.5% |
| Moment Hit@3 | 95.0% |
| Video + Moment @1 | 42.5% |
| Video + Moment @5 | 52.5% |
| Video + Moment @10 | 67.5% |
| Video + Moment @20 | 72.5% |
| Video + Moment @50 | 85.0% |

Temporal localization is substantially stronger than full-corpus video ranking.

Of the 28 queries where the correct video reached the top 10, **27 also returned a correct temporal window**, or **96.4%**. At top 50, 34 of the 35 retrieved target videos also had correct temporal support, or **97.1%**.

This leads to the main qualitative conclusion from the validation:

> **Once the system finds the correct video, it usually also points to the correct part of that video.**

The main weakness is therefore global video ranking against many distractors, not localization inside the target video.

The current Moment Hit metrics are supporting-window metrics for the final production output. They should not be treated as direct replacements for the older Stage 7 Story MRR metrics, which used a different ranking evaluation.

## 4. Comparison with the Earlier Stage 7 Evaluation

Retrieval v2 Stage 7 evaluated the selected reranked system on the same 40 Extension40 queries, but against the much smaller All50 corpus. The production validation searches the complete 1,478-video corpus.

| Metric | Stage 7 research | Full production | Difference |
|---|---:|---:|---:|
| Corpus size | 50 videos | 1,478 videos | 29.6× larger |
| Queries | 40 | 40 | Same |
| Video R@1 | 75.0% | 42.5% | -32.5 percentage points |
| Video R@5 | 90.0% | 55.0% | -35.0 percentage points |
| Video MRR | 0.821 | 0.491 | -0.329 |

The lower production result is important, but it should not be interpreted as evidence that the production implementation became worse. The search problem changed substantially: the system now ranks the target against 1,477 possible distractors instead of 49.

The production candidate policy also differs from the Stage 7 research setup, so this is not a controlled comparison of corpus size alone. The comparison is best understood as a change from research-scale evaluation to deployment-scale evaluation.

The main conclusion is that the earlier 50-video benchmark gave a more optimistic estimate of global video-ranking performance. Full-corpus validation provides the more realistic measure for integration and deployment.

## 5. Failure Pattern

The full-corpus errors are concentrated rather than evenly distributed.

| Failure view | Count |
|---|---:|
| Target outside Top 10 | 12 / 40 |
| Target outside Top 50 | 5 / 40 |
| Top-10 video hit but temporal miss | 1 / 40 |

The five largest video-ranking misses were:

| Query | Target video | Target rank | Top-ranked video |
|---|---|---:|---|
| R2-1 | `K03_V019` | 512 | `K14_V013` |
| R2-8 | `K03_V023` | 488 | `L22_V015` |
| R2-15 | `K01_V018` | 277 | `K18_V025` |
| R2-3 | `K17_V003` | 173 | `K08_V026` |
| R3-15 | `L26_V222` | 81 | `L26_V227` |

These failures do not show one unrelated video repeatedly dominating the rankings, which argues against an obvious global indexing problem.

Many competition queries also contain information that is primarily visual, such as clothing, objects, actions, scene layout, or appearance. ASR cannot directly retrieve evidence that is never spoken. This explains why some queries that are easy for a multimodal system can still be difficult for an ASR-only retriever.

Only one query combined a top-10 target video with a failed top-3 temporal match. This further supports the conclusion that temporal localization is not the main bottleneck.

## 6. Runtime and Stability

Deployment-style latency was measured with final reranked searches returning the top 50 videos. Each of the 40 queries was executed three times after warmup, for 120 measured searches.

| Runtime metric | Result |
|---|---:|
| Mean latency | 2.37 s |
| Median latency | 2.33 s |
| p90 latency | 2.79 s |
| p95 latency | 2.89 s |
| Maximum latency | 3.03 s |
| Candidate pairs | 149–150 |
| Reranker batch size | 32 |
| Peak GPU memory allocated | 3.49 GiB |
| Peak GPU memory reserved | 4.16 GiB |
| Deterministic queries | 40 / 40 |

The latency distribution is stable: the maximum observed search was only about 0.7 seconds slower than the median.

The system also used essentially its full reranking workload on every query, with 149–150 candidate pairs out of the 150-pair limit. The reranker maintained batch size 32 throughout the validation, so no smaller fallback batch was required.

All 40 queries produced identical top-50 video ordering and candidate-pair counts across the three repeated runs. This provides direct evidence that the production search path is deterministic under the tested conditions.

Latency is hardware-dependent, so these values should be interpreted as measurements of the validation environment rather than universal application latency.

## 7. Overall Assessment

The production ASR retrieval system passed full-corpus structural and operational validation.

| Area | Assessment |
|---|---|
| Release integrity | PASS |
| Full-corpus execution | PASS |
| Determinism | PASS |
| Runtime stability | PASS |
| GPU feasibility | PASS |
| Temporal supporting-window quality | Strong |
| ASR-only global video ranking | Moderate |

The final system reaches **70.0% Video R@10** and **87.5% Video R@50** across the complete 1,478-video corpus. Its median target rank is 4, but several large misses reduce MRR to 0.491.

Temporal support is much stronger: **95.0% Moment Hit@3**, and 96.4% of top-10 target-video hits also include a correct temporal supporting window.

The practical interpretation is straightforward. The ASR subsystem is strong at retrieving spoken semantic evidence and locating useful moments once the correct video is found. Its main limitation is ranking videos for queries whose distinguishing information is primarily visual.

For the complete competition system, ASR should therefore be treated as a complementary retrieval signal rather than a standalone solution. Its high Top-50 recall and strong temporal support make it well suited for combination with visual and other multimodal retrieval channels.

The full-corpus validation supports freezing the current ASR retrieval subsystem and moving forward with multimodal integration rather than reopening the completed Retrieval v2 component-selection experiments.
