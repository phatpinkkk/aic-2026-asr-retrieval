# Artifact Storage

Large or generated artifacts are intentionally kept outside Git.

## Storage Policy

| Artifact | Git | Local | Google Drive |
|---|---|---|---|
| Source code | Yes | Yes | Yes |
| Configs | Yes | Yes | Yes |
| Benchmark manifests | Yes | Yes | Yes |
| Notebooks | Yes | Yes | Yes |
| Research documentation | Yes | Yes | Optional |
| Full generated reports | No | Yes | Yes |
| ASR inference outputs | No | Optional | Yes |
| Audio | No | No | Yes |
| Video | No | No | Yes |
| Model weights | No | No | Yes |
| Embedding caches | No | Optional | Yes |

## Canonical Large-Artifact Workspace

Google Drive remains the canonical storage location for the complete
experimental workspace and generated artifacts.

The local repository is a lightweight research and version-control workspace.

## Reports

A local mirror of `reports/` is retained for analysis.

Generated reports are excluded from Git. Stable findings should instead be
summarized under `docs/results/`.

## Reproducibility

Where relevant, experiment manifests should record:

- model identifier
- exact model revision
- benchmark hash
- configuration hash
- source-code version
- artifact hashes
- execution environment
