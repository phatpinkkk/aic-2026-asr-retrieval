# Data

This directory contains compact, version-controlled data required to define
and reproduce the ASR and retrieval evaluation protocol.

## Tracked data

- `benchmark_manifest.json` – Core10 benchmark definition.
- `selected_cases.json` – selected evaluation cases.
- `windows.json` – fixed ASR window definitions.
- `stage1_extension40/` – Extension40 benchmark and selection metadata.
- `references/` – frozen reference metadata/transcripts used by the Stage 1
  evaluation.

## External data

Large audio files, source videos, ASR outputs, model weights, embedding caches,
and other generated artifacts are not stored in Git.

Their storage locations and reproducibility information are documented in
`docs/artifacts.md`.

The canonical large-artifact workspace remains on Google Drive.
