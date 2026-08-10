# AIC 2026 ASR & Retrieval Research

Research and evaluation repository for ASR-assisted video retrieval in the AI Challenge 2026 pipeline.

The repository studies how spoken content can support video identification and temporal localization, with a focus on Vietnamese ASR, multilingual text retrieval, sparse+dense retrieval, video-level evidence aggregation, and later reranking and multimodal extensions.

The production video-search application is maintained separately in `aic-2026-pipeline`.

## Current Status

The first ASR-text retrieval baseline is now complete.

The current system uses:

```text
video audio
    ↓
Parakeet CTC 0.6B Vietnamese
    ↓
processed transcript
    ↓
60 s overlapping transcript windows
    ↓
BM25 lexical retrieval
       +
multilingual E5-large semantic retrieval
    ↓
per-query score normalization
    ↓
25% BM25 + 75% E5
    ↓
maximum window score per video
    ↓
ranked videos + supporting transcript windows
```

On `development20` against the 50-video corpus, the current ASR-text system reaches:

| Metric | Result |
|---|---:|
| Video R@1 | **0.75** |
| Video R@5 | **0.85** |
| Video R@10 | **0.90** |
| Video MRR | **0.8102** |
| Story R@1 | **0.80** |
| Story MRR | **0.8583** |

These are development results. `holdout20` remains closed until the broader retrieval architecture is frozen.

## Retrieval Research

Retrieval v2 improved the text-retrieval subsystem in several steps:

1. replace the historical query-fitted TF-IDF setup with a proper lexical retriever;
2. strengthen multilingual semantic retrieval;
3. evaluate how transcript-window evidence should be aggregated into video scores;
4. combine lexical and semantic scores using controlled score fusion; and
5. re-evaluate the ASR model and transcript representation under the improved retriever.

The resulting ASR-text subsystem uses accent-preserving BM25 as a lexical signal and `intfloat/multilingual-e5-large-instruct` as the main semantic signal. Dense retrieval carries most of the weight, while BM25 provides a smaller lexical correction.

The next research stages focus on hierarchical retrieval, reranking, query and transcript representation, visual/OCR evidence, and final holdout evaluation.

## Repository Structure

```text
configs/       experiment and retrieval configuration
data/          compact benchmark and reference metadata
notebooks/     stage-specific research notebooks
src/           retrieval, backend, evaluation, and orchestration code
docs/          technical reference and curated research results
reports/       generated reports, local only and ignored by Git
```

Large generated artifacts are stored outside Git.

## Documentation

Start with the document that matches what you need:

- [ASR Retrieval System](docs/asr-retrieval-system.md)  
  Clear implementation reference for the current ASR-text retrieval subsystem: inputs, transcript preparation, BM25, E5-large, score fusion, video scoring, caches, and code ownership.

- [Retrieval v2 Progress](docs/results/retrieval-v2-progress.md)  
  Research results and decisions through ASR and transcript-view selection, including comparisons, failure analysis, efficiency, and remaining limitations.

- [Retrieval v2](docs/retrieval-v2.md)  
  Research methodology, evaluation protocol, experiment design, and roadmap for the broader Retrieval v2 work.

## Development and Evaluation Policy

Architecture selection is performed using `development20` against the full 50-video corpus.

`core10` is used for regression and reproducibility checks.

`holdout20` remains closed until the retrieval architecture and selection policy are frozen. Development channels are evaluated independently and are not fused unless an experiment explicitly defines such a method.

## Data and Artifacts

Git contains source code, configuration, notebooks, compact benchmark metadata, and curated documentation.

Large artifacts remain in the external experiment workspace, including:

- source videos and audio;
- ASR outputs;
- model weights;
- dense embedding caches;
- retrieval score caches; and
- full generated reports.

The retrieval code separates the repository code root from the artifact root so the Git checkout does not need to contain large generated data.

The artifact root can be configured with:

```text
AIC_RETRIEVAL_ARTIFACT_ROOT
```

## Current Scope

The current ASR-text subsystem covers:

- Vietnamese speech transcription;
- processed transcript selection;
- overlapping transcript-window retrieval;
- BM25 lexical matching;
- multilingual dense retrieval;
- normalized sparse+dense score fusion;
- global video ranking; and
- supporting transcript-window retrieval.

The broader AIC retrieval system is still under development. Hierarchical retrieval, reranking, alternative query/transcript representations, visual retrieval, OCR retrieval, multimodal fusion, and final holdout evaluation are handled in later stages.
