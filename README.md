# AIC 2026 ASR & Retrieval Research

Research and evaluation repository for ASR-assisted multilingual video retrieval in the AI Challenge 2026 pipeline.

This repository studies how speech transcripts can contribute to video identification and temporal localization, with emphasis on Vietnamese ASR, multilingual text retrieval, hybrid retrieval, evidence aggregation, and reranking.

The production video-search application is maintained separately in `aic-2026-pipeline`.

## Current Status

Stage 1 ASR evaluation is complete.

Whisper Large-v3 and NVIDIA Parakeet CTC 0.6B Vietnamese are retained as the two ASR evidence sources.

The current research focus is Retrieval v2.

The main Stage 1 finding is that ASR already provides useful temporal evidence inside the correct video, while global video identification remains the larger bottleneck.

## Retrieval v2

The existing retrieval baseline uses:

- query-fitted character TF-IDF
- multilingual E5-small
- fixed lexical-semantic score averaging
- maximum-window video scoring

Retrieval v2 will evaluate:

BM25  
→ stronger multilingual dense retrieval  
→ rank-level hybrid fusion  
→ improved video evidence aggregation  
→ Whisper/Parakeet channel fusion  
→ hierarchical video-to-window retrieval  
→ reranking  
→ controlled query reformulation

## Repository Structure

```text
configs/       experiment configuration
data/          compact benchmark and reference metadata
notebooks/     canonical research notebooks
src/           experiment and evaluation source code
docs/          methodology, decisions, progress, and curated results
reports/       generated reports, local only and ignored by Git
```

## Documentation

- [Project status](docs/project-status.md)
- [Evaluation protocol](docs/evaluation-protocol.md)
- [Retrieval v2 roadmap](docs/retrieval-v2-roadmap.md)
- [Experiment log](docs/experiment-log.md)
- [Research decisions](docs/decisions.md)
- [Artifact storage](docs/artifacts.md)
- [Stage 1 ASR results](docs/results/stage1-asr.md)

## Data and Artifacts

Large generated artifacts are intentionally excluded from Git.

Audio, source videos, ASR outputs, model weights, embedding caches, and the canonical experiment workspace remain on Google Drive.

Full generated reports may be mirrored locally under `reports/`, but stable findings are documented under `docs/results/`.

See [Artifact Storage](docs/artifacts.md).

## Development Policy

Architecture selection is performed using `development20`.

`core10` remains available for regression checks.

`holdout20` remains closed until the Retrieval v2 architecture is frozen.
