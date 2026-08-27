# vortex-data/datasets

Tooling for mirroring external Hugging Face Parquet datasets into vortex-data
repositories as Vortex (and re-compressed Parquet) files.

## `scripts/hf-sync.py`

Syncs a list of external Hugging Face dataset repositories into a vortex-data
destination repository. For every selected source Parquet shard, the script maps
it to a deterministic destination path that keeps the source file's name:

```
<upload-prefix>/<format>/<source-repo-slug>/<source path>.{vortex,parquet}
```

then diffs that path against the destination repository's file listing. Files
already present in the destination are skipped; missing files are downloaded
(Xet-accelerated), converted to each requested format, and uploaded in batched
Hub commits. Runs are checkpointed per source repository under `--output-dir`
and are resumable after interruption.

### Sources

Sources come from a JSON file (see [`sources.example.json`](sources.example.json))
and/or a single `--repo`:

```sh
# Diff every configured source against the destination without transferring:
scripts/hf-sync.py --sources sources.json --upload-repo vortex-data/mirror --dry-run

# Full sync of all configured sources:
HF_TOKEN=... scripts/hf-sync.py --sources sources.json \
    --upload-repo vortex-data/mirror --vx /path/to/vx --delete-after-upload

# One-off single repository:
scripts/hf-sync.py --repo HuggingFaceFW/fineweb --prefix data \
    --filter 'sample/10BT/*' --upload-repo vortex-data/mirror --vx /path/to/vx
```

Per-source keys: `repo` (required), `revision`, `prefix`, `include`, `filters`,
`mode` (`all` | `first` | `sample`), `limit`, `target_size`, `seed`.

### Requirements

- `huggingface_hub` with `hf-xet` (all repositories must be Xet-enabled)
- `pyarrow` for the `parquet-zstd6` output
- a `vx` binary (build `vortex-tui` with `unstable_encodings`) for the
  `vortex` and `vortex-compact` outputs
- `HF_TOKEN` with write access to the destination repository

### Tests

```sh
python -m unittest discover -s scripts/tests -v
```
