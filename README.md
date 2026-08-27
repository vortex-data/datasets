# vortex-data/datasets

Tooling for mirroring external Hugging Face Parquet datasets into vortex-data
repositories as Vortex (and re-compressed Parquet) files.

## `scripts/hf-sync.py`

Syncs a list of external Hugging Face dataset repositories into a vortex-data
destination repository, in **two explicit steps with an inspectable plan file
in between**:

```sh
# Step 1: diff every source against the destination and write the work plan.
HF_TOKEN=... scripts/hf-sync.py plan --sources sources.json \
    --upload-repo vortex-data/mirror --plan plan.json

# Inspect (or trim/edit) plan.json — it lists every file that will be synced,
# the formats it is missing, and the exact destination paths.

# Step 2: execute the plan — download, convert, upload.
HF_TOKEN=... scripts/hf-sync.py run --plan plan.json \
    --vx /path/to/vx --delete-after-upload
```

`plan` resolves each source to an immutable commit, maps every selected Parquet
shard to a deterministic destination path that keeps the source file's name:

```
<upload-prefix>/<format>/<source-repo-slug>/<source path>.{vortex,parquet}
```

and diffs that path against the destination repository's listing. Pairs already
present in the destination are recorded under `already_present`; the rest become
work chunks — one per source file, listing the formats still missing and their
destination paths. Nothing is transferred by `plan`.

`run` executes only what the plan contains: each chunk is downloaded
(Xet-accelerated), converted to its missing formats, and uploaded in batched
Hub commits. Runs are checkpointed per source repository under `--output-dir`
and are resumable after interruption; deleting entries from the plan before
`run` skips them entirely.

### Sources

Sources come from a JSON file (see [`sources.example.json`](sources.example.json))
and/or a single `--repo`:

```sh
scripts/hf-sync.py plan --repo HuggingFaceFW/fineweb --prefix data \
    --filter 'sample/10BT/*' --upload-repo vortex-data/mirror
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
