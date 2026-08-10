# Experiment Log

Chronological record of important experiments and observations.

Detailed machine-generated outputs remain under the local `reports/`
directory. This document records only stable research findings and decisions.

---

## 2026-08-10 – Stage 1 Development Evaluation

### Experiment

Evaluated Whisper Large-v3 and NVIDIA Parakeet CTC 0.6B Vietnamese using
development20 against the full 50-video retrieval corpus.

### Main Findings

- Story localization is substantially stronger than global video retrieval.
- Whisper and Parakeet are close on conditional story localization.
- Whisper currently provides stronger global video ranking.
- Parakeet is substantially faster and shows fewer reliability problems.
- Raw and processed transcript views make different retrieval errors.
- The current retrieval implementation is now a more important bottleneck
  than ASR model selection.

### Decision

Stop expanding the ASR model search.

Begin Retrieval v2 with:

baseline reproduction
→ proper lexical retrieval
→ stronger multilingual dense retrieval
→ rank fusion
→ improved video evidence aggregation.

---

## 2026-08-10 – Retrieval v2 Repository Setup

### Changes

Created a dedicated local research repository.

Separated:

- version-controlled research assets
- local generated reports
- Google Drive large artifacts

Added project status, evaluation protocol, retrieval roadmap, decision log,
artifact policy, and result documentation.
