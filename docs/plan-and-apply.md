# Planning and applying a dataset conversion

`hf-sync.py` deliberately separates discovery from execution:

1. `plan` inspects the source and destination repositories and writes a JSON
   action plan. It does not download, convert, or upload dataset files.
2. You review the plan to confirm the immutable revisions, selected files,
   output paths, and commit batches.
3. `apply` executes exactly that plan and records resumable progress in a
   checkpoint.

This split makes a large conversion inspectable before it writes anything to
the destination repository.

## Prerequisites

Install the Python environment from the repository root:

```sh
uv sync --project scripts
```

Set `HF_TOKEN` to a Hugging Face token that can read the source dataset and
write to the destination dataset:

```sh
export HF_TOKEN=hf_...
```

Both repositories must be Xet-enabled. Vortex output also requires a `vx`
binary built from the Vortex repository with the `unstable_encodings` feature.
The `parquet-zstd6` format only requires the Python environment.

## Step 1: create a plan

The following example plans a small FineWeb conversion:

```sh
uv run --project scripts scripts/hf-sync.py plan \
    --repo HuggingFaceFW/fineweb \
    --revision main \
    --prefix sample/10BT \
    --mode first \
    --limit 10 \
    --formats vortex,vortex-compact \
    --upload-repo vortex-data/fineweb \
    --upload-revision main \
    --plan-file plan.json
```

Planning resolves both requested revisions to immutable commit IDs. Later
changes to `main` therefore cannot silently change the source files represented
by the plan.

### Selecting source files

Use one of these selection modes:

- `--mode first --limit N` selects the first `N` matching shards.
- `--mode sample --target-size 10GB --seed 0` selects shards spread across the
  ordered dataset until their approximate total size reaches the target.
- `--mode all` selects every matching shard.

`--prefix` selects the folder to scan. Use `/` for the repository root.
`--include` is matched against paths below that prefix and defaults to
`*.parquet`. Repeat `--filter` to restrict the selection with full repository
path globs:

```sh
--prefix sample --filter 'sample/10BT/*' --filter 'sample/100BT/*'
```

### Selecting output formats

`--formats` accepts a comma-separated subset of:

- `parquet-zstd6`: Parquet rewritten with Zstandard level 6.
- `vortex`: Vortex using the `btrblocks` strategy.
- `vortex-compact`: Vortex using the `compact` strategy.

Destination paths are deterministic:

```text
<upload-prefix>/<format>/<source path with the output extension>
```

Use `--upload-prefix` to place all generated formats below an additional
destination folder.

## Step 2: review the plan

Open `plan.json` and verify:

- `source.revision` and `destination.revision` are the expected immutable
  commits.
- `work_chunks` contains the intended source shards.
- Every action has the expected `create` or `skip` decision and destination
  path.
- `upload_batches` has acceptable commit boundaries and messages.
- `summary` has plausible file, byte, create, and skip counts.

The apply command validates that every create action belongs to exactly one
upload batch. If you edit the plan, keep `work_chunks` and `upload_batches`
consistent. For routine selection changes, generating a new plan is safer than
editing the JSON manually.

## Step 3: apply the plan

Run the reviewed plan:

```sh
uv run --project scripts scripts/hf-sync.py apply plan.json \
    --vx /path/to/vx \
    --output-dir data/fineweb-run \
    --delete-after-upload
```

During execution, the script:

1. validates the plan and destination revision;
2. downloads selected Parquet shards with bounded concurrency;
3. converts the requested formats in parallel;
4. preuploads and commits outputs in the planned batches;
5. writes checkpoint, status, and metrics files under `--output-dir`.

All three stages share one `--workers` pool. A worker can claim any kind of
work. The scheduler drains an upload queue at its high-water mark, otherwise
refills the download buffer, then handles ready uploads, then converts a ready
source shard. Queue claims are atomic, so two workers cannot process the same
file.

Downloads and uploads are real multi-file operations. Their batch sizes start
at the configured initial value, grow while aggregate throughput improves, and
back off after a significant regression. File and byte limits cap every claim.
Upload preupload batches remain separate from planned commit batches: files may
transfer together, but a commit is created only when all members of its planned
batch are ready.

`--delete-after-upload` removes a local output only after its upload is
committed successfully. Source downloads are removed after conversion unless
`--keep-downloads` is set.

### Test without writing to Hugging Face

Use a local destination to exercise downloading and conversion without remote
uploads:

```sh
uv run --project scripts scripts/hf-sync.py apply plan.json \
    --vx /path/to/vx \
    --output-dir data/test-run \
    --upload-local-dir data/test-sink
```

The local sink preserves the same destination path layout as a Hugging Face
dataset repository.

## Resume an interrupted run

Re-run the same apply command with the same `--output-dir`:

```sh
uv run --project scripts scripts/hf-sync.py apply plan.json \
    --vx /path/to/vx \
    --output-dir data/fineweb-run \
    --delete-after-upload
```

The checkpoint reuses completed uploads and valid local encodings. Transfer
tuning such as Xet cache location or concurrency may change between attempts,
but changing the selected files, formats, destination, or encoding-affecting
arguments requires a new output directory or a new plan.

Before resuming, do not remove local outputs that have finished conversion but
have not yet been committed. They are reusable and avoid repeated conversion.

## Operational controls

The most useful apply controls are:

- `--workers` sets the size of the unified worker pool.
- `--download-initial-concurrency` and `--download-max-concurrency` control the
  adaptive number of files in each download batch.
- `--download-buffer-files` and `--download-buffer-size` bound admitted source
  downloads.
- `--upload-workers` and `--upload-max-concurrency` control the adaptive number
  of files in each upload batch.
- `--upload-buffer-files` and `--upload-buffer-size` bound pending output work.
- `--xet-cache`, `--xet-range-gets`, and `--xet-high-performance` tune Xet.
- `--status-interval` controls how frequently live status is written.

Start conservatively. Increase concurrency only when CPU, disk, memory, and
network headroom are visible in the live status and host metrics.

## Generated run files

The output directory contains the durable state needed to inspect or resume a
run:

- `checkpoint.json` records per-file conversion and upload state.
- `status.json` is an atomically replaced live pipeline snapshot.
- `metrics/selection.json` records the effective run configuration.
- final metric reports summarize selected files and generated formats.

Keep the plan and output directory together until the destination has been
verified and no resume is required.
