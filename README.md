# vortex-data/datasets

Tools for converting physical Parquet shards in Hugging Face dataset
repositories to Vortex or recompressed Parquet files.

## Setup

The repository is a `uv` workspace. Install the script dependencies with:

```sh
uv sync --project scripts
```

The Vortex output formats also require a `vx` binary built from Vortex with
the `unstable_encodings` feature.

## Plan a conversion

Planning resolves the source and destination to immutable revisions, discovers
the selected Parquet shards, and writes an action plan without transferring
data:

```sh
HF_TOKEN=... uv run --project scripts scripts/hf-sync.py plan \
    --repo HuggingFaceFW/fineweb \
    --revision main \
    --prefix sample/10BT \
    --upload-repo vortex-data/fineweb \
    --plan-file plan.json
```

Review or edit `plan.json` before applying it.

See [Planning and applying a dataset conversion](docs/plan-and-apply.md) for a
complete walkthrough, plan review checklist, resume behavior, and operational
tuning guidance.

## Apply a plan

```sh
HF_TOKEN=... uv run --project scripts scripts/hf-sync.py apply \
    plan.json --vx /path/to/vx
```

The apply step uses bounded download, conversion, and upload pipelines. It
checkpoints progress under `--output-dir`, supports resuming interrupted runs,
and only removes local outputs after successful upload when
`--delete-after-upload` is set.

Run `uv run --project scripts scripts/hf-sync.py --help` for all options.

## Tests

```sh
uv run --project scripts python -m unittest discover -s scripts/tests -v
```
