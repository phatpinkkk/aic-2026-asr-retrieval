# Research Decisions

This file records durable architectural and experimental decisions.

---

## ADR-001 – Separate Research and Production Repositories

Status: Accepted

The ASR and retrieval research is maintained in `aic-2026-asr-retrieval`.

The production application remains in `aic-2026-pipeline`.

Stable research components may later be integrated into the production
pipeline.

---

## ADR-002 – Retain Whisper and Parakeet

Status: Accepted

Whisper Large-v3 and NVIDIA Parakeet CTC 0.6B Vietnamese are retained as the
two ASR retrieval channels.

Additional ASR model search is suspended.

Reason:

- both provide useful retrieval evidence
- their errors are not identical
- Parakeet has a major operational speed advantage
- Retrieval v2 is now a higher-value research direction

---

## ADR-003 – Whisper Is a Pseudo-Reference

Status: Accepted

Whisper Large-v3 is not treated as transcript ground truth.

Candidate WER and CER against Whisper measure model agreement only.

---

## ADR-004 – Keep Stage 1 Evaluation Frozen

Status: Accepted

The validated Stage 1 evaluator remains unchanged while Retrieval v2 is
developed separately.

---

## ADR-005 – Protect Holdout20

Status: Accepted

Development20 is used for architecture selection.

Holdout20 remains closed until Retrieval v2 is frozen.

---

## ADR-006 – Keep Large Artifacts Outside Git

Status: Accepted

Audio, source videos, ASR outputs, generated reports, model weights,
embedding caches, and other large artifacts remain on Google Drive or local
storage.

Git contains source code, configs, benchmark definitions, notebooks,
documentation, and compact reproducibility metadata.
