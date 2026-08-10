# Stage 1 ASR Evaluation

## Objective

Evaluate whether Vietnamese ASR provides useful evidence for AIC 2026 video
retrieval and determine which ASR systems should continue into retrieval
research.

## Retained Models

- Whisper Large-v3
- NVIDIA Parakeet CTC 0.6B Vietnamese

## Development20 Retrieval

### Whisper Large-v3

Raw:

- Story R@1: 0.70
- Story R@3: 0.80
- Story R@5: 0.90
- Story MRR: 0.7831
- Video R@1: 0.60
- Video R@3: 0.70
- Video R@5: 0.75
- Video MRR: 0.6761

Processed:

- Story R@1: 0.70
- Story R@3: 0.80
- Story R@5: 0.90
- Story MRR: 0.7845
- Video R@1: 0.60
- Video R@3: 0.65
- Video R@5: 0.75
- Video MRR: 0.6610

### Parakeet CTC 0.6B Vietnamese

Raw:

- Story R@1: 0.70
- Story R@3: 0.80
- Story R@5: 0.90
- Story MRR: 0.7770
- Video R@1: 0.40
- Video R@3: 0.70
- Video R@5: 0.75
- Video MRR: 0.5692

Processed:

- Story R@1: 0.70
- Story R@3: 0.75
- Story R@5: 0.85
- Story MRR: 0.7613
- Video R@1: 0.35
- Video R@3: 0.70
- Video R@5: 0.75
- Video MRR: 0.5421

## Reliability

Whisper exhibited:

- invalid native timestamps
- automatically rejected processed windows
- unrelated exact duplicate transcripts
- repeated boilerplate/hallucination phrases across unrelated videos

Parakeet did not exhibit the same exact unrelated duplicate behavior in this
evaluation.

## Operational Performance

Parakeet achieved approximately a 35x real-time-factor speed advantage over
Whisper in the evaluated inference runs and used less peak allocated GPU
memory.

## Interpretation

The strongest finding is not simply that one ASR model wins.

Both models already provide useful within-video temporal evidence.

The much larger weakness is global video discrimination.

Therefore the next research phase focuses on stronger retrieval rather than
additional ASR model search.

## Decision

Retain both Whisper and Parakeet as independent evidence channels for
Retrieval v2.
