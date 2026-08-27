#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Sync Parquet shards from external Hugging Face repositories into a vortex-data repo.

The sync runs in two explicit steps with an inspectable plan file in between:

  plan  Resolve each source repository to an immutable commit, list its Parquet
        shards, diff every (shard, format) pair against the same-named path in
        the destination repository, and write the resulting work chunks to a
        JSON plan file. Nothing is downloaded or uploaded.

  run   Execute a previously written plan: download each planned shard, convert
        it to the formats the plan marked missing, and upload the outputs in
        batched Hub commits. Runs are checkpointed per source and resumable.

Edit or trim the plan file between the two steps to control exactly what runs.
"""

import abc
import argparse
import concurrent.futures
import csv
import datetime
import fnmatch
import hashlib
import json
import os
import random
import subprocess
import sys
import threading
import time
from pathlib import Path


DEFAULT_TARGET_BYTES = 10_000_000_000
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "hf-sync"
DEFAULT_PLAN_PATH = DEFAULT_DATA_DIR / "plan.json"
DEFAULT_XET_CACHE = DEFAULT_DATA_DIR / "xet-cache"
DEFAULT_DOWNLOAD_BUFFER_BYTES = 64_000_000_000
DEFAULT_UPLOAD_BUFFER_BYTES = 128_000_000_000
PARQUET_BATCH_ROWS = 65_536
SUPPORTED_FORMATS = ("parquet-zstd6", "vortex", "vortex-compact")
PLAN_VERSION = 1
LOG_LOCK = threading.Lock()


# --------------------------------------------------------------------------- logging


def safe_print(*values, file=None, **kwargs):
    """Write progress without allowing a detached output pipe to stop a sync."""
    target = file if file is not None else sys.stdout
    try:
        with LOG_LOCK:
            print(*values, file=target, **kwargs)
    except BrokenPipeError:
        replacement = open(os.devnull, "w")
        if target is sys.stdout:
            sys.stdout = replacement
        elif target is sys.stderr:
            sys.stderr = replacement
        else:
            replacement.close()


# ----------------------------------------------------------------------- Hub helpers


def retry_call(operation, attempts, base_delay=1.0):
    """Retry a Hub operation with bounded exponential backoff."""
    for attempt in range(1, attempts + 1):
        try:
            return operation()
        except Exception as error:
            try:
                import httpx
                transport_error = isinstance(error, httpx.TransportError)
            except ImportError:
                transport_error = False
            response = getattr(error, "response", None)
            status = getattr(response, "status_code", None)
            retryable = (isinstance(error, (ConnectionError, TimeoutError)) or transport_error
                         or status in (408, 429) or (status is not None and status >= 500))
            if attempt == attempts or not retryable:
                raise
            time.sleep(base_delay * 2 ** (attempt - 1))


def parse_size(value):
    units = {"b": 1, "kb": 1000, "mb": 1000**2, "gb": 1000**3, "tb": 1000**4,
             "kib": 1024, "mib": 1024**2, "gib": 1024**3, "tib": 1024**4}
    value = value.strip().lower()
    number = value
    unit = "b"
    for suffix in sorted(units, key=len, reverse=True):
        if value.endswith(suffix):
            number = value[:-len(suffix)]
            unit = suffix
            break
    return int(float(number) * units[unit])


def hub_api():
    from huggingface_hub import HfApi

    return HfApi(token=os.environ.get("HF_TOKEN"))


def resolve_dataset_revision(api, repo, revision, attempts, timeout=None):
    info = retry_call(lambda: api.dataset_info(repo, revision=revision, timeout=timeout), attempts)
    if not info.sha:
        raise RuntimeError(f"could not resolve {repo}@{revision} to an immutable commit")
    return info.sha


def require_xet_repository(api, repo, revision, attempts):
    """Fail before transfer if a source or destination repository is not Xet-backed."""
    info = retry_call(
        lambda: api.repo_info(repo, repo_type="dataset", revision=revision,
                              expand=["xetEnabled"]), attempts)
    enabled = getattr(info, "xet_enabled", None)
    if enabled is None:
        enabled = info.__dict__.get("xetEnabled")
    if not enabled:
        raise RuntimeError(f"dataset repository is not Xet-enabled: {repo}@{revision}")


def list_shards(api, repo, revision, prefix, include, filters=(), attempts=3):
    shards = []
    tree = retry_call(
        lambda: list(api.list_repo_tree(repo, path_in_repo=prefix or None, recursive=True,
                                        revision=revision, repo_type="dataset")), attempts)
    for item in tree:
        path = getattr(item, "path", "")
        size = getattr(item, "size", None)
        relative = path.removeprefix(prefix.rstrip("/") + "/")
        filter_match = not filters or any(fnmatch.fnmatch(path, pattern) for pattern in filters)
        if path.endswith(".parquet") and size is not None and fnmatch.fnmatch(relative, include) and filter_match:
            lfs = getattr(item, "lfs", None) or {}
            shards.append({"path": path, "size": int(size), "sha256": lfs.get("sha256")})
    if not shards:
        raise RuntimeError(f"no Parquet shards found under {repo}@{revision}/{prefix}")
    return sorted(shards, key=lambda shard: shard["path"])


def list_repository_files(api, repo, revision, prefix="", attempts=3):
    """Return dataset repository file metadata keyed by path."""
    from huggingface_hub.errors import RemoteEntryNotFoundError

    files = {}
    try:
        tree = retry_call(
            lambda: list(api.list_repo_tree(repo, path_in_repo=prefix or None, recursive=True,
                                            revision=revision, repo_type="dataset")), attempts)
    except RemoteEntryNotFoundError:
        return {}
    for item in tree:
        path = getattr(item, "path", None)
        if path is not None and getattr(item, "size", None) is not None:
            files[path] = {"size": item.size, "oid": getattr(item, "blob_id", None),
                           "lfs": getattr(item, "lfs", None)}
    return files


def download_shard(repo, revision, shard, download_root, attempts, etag_timeout):
    from huggingface_hub import hf_hub_download

    started = time.monotonic()
    path = Path(retry_call(
        lambda: hf_hub_download(repo, shard["path"], repo_type="dataset", revision=revision,
                                local_dir=download_root, etag_timeout=etag_timeout), attempts))
    elapsed = time.monotonic() - started
    actual_size = path.stat().st_size
    if actual_size != shard["size"]:
        raise RuntimeError(f"downloaded {actual_size} bytes for {path}, expected {shard['size']}")
    if shard.get("sha256") and file_sha256(path) != shard["sha256"]:
        path.unlink(missing_ok=True)
        raise RuntimeError(f"SHA-256 mismatch for {shard['path']}")
    return path, elapsed


# ------------------------------------------------------------------------ conversion


def parquet_zstd6(source, destination):
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError("pyarrow is required; install it in the active Python environment") from error
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    parquet = pq.ParquetFile(source)
    started = time.monotonic()
    writer = pq.ParquetWriter(partial, parquet.schema_arrow,
                              compression="zstd", compression_level=6)
    try:
        for batch in parquet.iter_batches(batch_size=PARQUET_BATCH_ROWS):
            writer.write_batch(batch, row_group_size=PARQUET_BATCH_ROWS)
    finally:
        writer.close()
    partial.replace(destination)
    return time.monotonic() - started


def run_vx(vx, source, destination, strategy):
    staging = source.parent / f".{destination.stem}.{strategy}.input.parquet"
    generated = staging.with_suffix(".vortex")
    staging.unlink(missing_ok=True)
    generated.unlink(missing_ok=True)
    os.link(source, staging)
    started = time.monotonic()
    try:
        subprocess.run([str(vx), "convert", str(staging), "--strategy", strategy, "--quiet"], check=True)
        elapsed = time.monotonic() - started
        destination.parent.mkdir(parents=True, exist_ok=True)
        generated.replace(destination)
        return elapsed
    finally:
        staging.unlink(missing_ok=True)
        generated.unlink(missing_ok=True)


def file_sha256(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


# ------------------------------------------------------------------------- uploaders


class Uploader(abc.ABC):
    """Upload backend used after each completed conversion."""

    @abc.abstractmethod
    def upload(self, local_path, destination_path, **metadata):
        """Upload one file and return serializable checkpoint metadata."""

    @abc.abstractmethod
    def config(self):
        """Return stable configuration included in the resumability fingerprint."""

    def existing_files(self, prefix=""):
        """Return existing sink files keyed by destination path."""
        return {}


class HuggingFaceBatchUploader(Uploader):
    """Preupload outputs and commit up to a fixed file count per format."""

    def __init__(self, api, repo, revision, batch_files, totals, attempts=5, timeout=30,
                 batch_bytes=None):
        self.api = api
        self.repo = repo
        self.revision = revision
        self.batch_files = batch_files
        self.totals = dict(totals)
        self.attempts = attempts
        self.timeout = timeout
        self.batch_bytes = batch_bytes
        self.parent_commit = resolve_dataset_revision(api, repo, revision, attempts, timeout)
        self.pending = {fmt: [] for fmt in totals}
        self.lock = threading.Lock()

    def upload(self, local_path, destination_path, *, format_name, ordinal, **metadata):
        from huggingface_hub import CommitOperationAdd

        operation = CommitOperationAdd(path_in_repo=destination_path, path_or_fileobj=local_path)
        started = time.monotonic()
        retry_call(lambda: self.api.preupload_lfs_files(
            self.repo, additions=[operation], repo_type="dataset",
            revision=self.revision, num_threads=1), self.attempts)
        entry = {"operation": operation, "local_path": str(local_path),
                 "hub_path": destination_path, "size_bytes": local_path.stat().st_size,
                 "format": format_name, "ordinal": ordinal}
        with self.lock:
            pending = self.pending[format_name]
            pending.append(entry)
            pending_bytes = sum(item["size_bytes"] for item in pending)
            if (len(pending) >= self.batch_files
                    or (self.batch_bytes is not None and pending_bytes >= self.batch_bytes)):
                return self._flush_format_locked(format_name)
        return {"status": "preuploaded", "hub_path": destination_path,
                "seconds": time.monotonic() - started}

    def flush(self):
        with self.lock:
            results = []
            for format_name in self.pending:
                result = self._flush_format_locked(format_name)
                if result["status"] == "complete":
                    results.append(result)
            return results

    def _flush_format_locked(self, format_name):
        from huggingface_hub import CommitOperationAdd

        batch = self.pending[format_name]
        if not batch:
            return {"status": "empty", "committed": []}
        batch.sort(key=lambda entry: entry["ordinal"])
        start = batch[0]["ordinal"]
        end = batch[-1]["ordinal"]
        total_files = self.totals[format_name]
        message = f"Upload {format_name} files {start}-{end} of {total_files}"
        operations = [entry["operation"] for entry in batch]
        result = None
        for attempt in range(1, self.attempts + 1):
            try:
                result = self.api.create_commit(
                    self.repo, operations=operations, repo_type="dataset",
                    revision=self.revision, parent_commit=self.parent_commit,
                    commit_message=message)
                break
            except Exception:
                remote = list_repository_files(self.api, self.repo, self.revision,
                                               attempts=1)
                committed = all(remote.get(entry["hub_path"], {}).get("size")
                                == entry["size_bytes"] for entry in batch)
                if committed:
                    commit_id = resolve_dataset_revision(
                        self.api, self.repo, self.revision, attempts=1, timeout=self.timeout)
                    result = type("CommitResult", (), {
                        "oid": commit_id,
                        "commit_url": f"https://huggingface.co/datasets/{self.repo}/commit/{commit_id}",
                    })()
                    break
                if attempt == self.attempts:
                    raise
                time.sleep(2 ** (attempt - 1))
                operations = []
                for entry in batch:
                    operation = CommitOperationAdd(path_in_repo=entry["hub_path"],
                                                   path_or_fileobj=entry["local_path"])
                    retry_call(lambda op=operation: self.api.preupload_lfs_files(
                        self.repo, additions=[op], repo_type="dataset",
                        revision=self.revision, num_threads=1), self.attempts)
                    operations.append(operation)
        if result is None:
            raise RuntimeError(f"failed to commit {format_name} files {start}-{end}")
        committed = [{key: entry[key]
                      for key in ("local_path", "hub_path", "size_bytes", "format", "ordinal")}
                     for entry in batch]
        total_bytes = sum(entry["size_bytes"] for entry in batch)
        self.pending[format_name] = []
        self.parent_commit = result.oid
        return {"status": "complete", "url": result.commit_url,
                "commit_id": result.oid, "committed": committed, "size_bytes": total_bytes,
                "format": format_name, "start": start, "end": end,
                "total_files": total_files, "commit_message": message}

    def config(self):
        return {"type": "huggingface-preupload-batch", "repo": self.repo,
                "revision": self.revision, "batch_files": self.batch_files}

    def existing_files(self, prefix=""):
        return list_repository_files(self.api, self.repo, self.revision,
                                     prefix, self.attempts)


class LocalCopyUploader(Uploader):
    """Test backend with the same interface and destination layout as Hub upload."""

    def __init__(self, destination, chunk_size=8 * 1024 * 1024):
        self.destination = destination.resolve()
        self.chunk_size = chunk_size

    def upload(self, local_path, destination_path, **metadata):
        output = self.destination / destination_path
        output.parent.mkdir(parents=True, exist_ok=True)
        partial = output.with_suffix(output.suffix + ".part")
        started = time.monotonic()
        chunks = 0
        bytes_copied = 0
        with local_path.open("rb") as source, partial.open("wb") as sink:
            while True:
                block = source.read(self.chunk_size)
                if not block:
                    break
                sink.write(block)
                chunks += 1
                bytes_copied += len(block)
        partial.replace(output)
        return {"status": "complete", "hub_path": destination_path,
                "url": output.as_uri(), "seconds": time.monotonic() - started,
                "chunks": chunks, "bytes_uploaded": bytes_copied}

    def config(self):
        return {"type": "local-copy", "destination": str(self.destination)}

    def existing_files(self, prefix=""):
        root = self.destination / prefix.strip("/")
        if not root.exists():
            return {}
        return {str(path.relative_to(self.destination)): {"size": path.stat().st_size}
                for path in root.rglob("*") if path.is_file() and not path.name.endswith(".part")}


def upload_then_maybe_delete(uploader, local_path, destination_path, delete_after_upload):
    """Upload one artifact, deleting the local copy only after the sink succeeds."""
    result = uploader.upload(local_path, destination_path)
    if result.get("status") != "complete":
        raise RuntimeError(f"sink did not complete upload of {destination_path}")
    if delete_after_upload:
        local_path.unlink()
        result["local_deleted"] = True
    else:
        result["local_deleted"] = False
    return result


# -------------------------------------------------------------- selection and diffing


def select_evenly(shards, target_bytes, seed):
    """Select seeded positions spread evenly over the complete ordered shard list."""
    average = sum(shard["size"] for shard in shards) / len(shards)
    count = min(len(shards), max(1, round(target_bytes / average)))
    rng = random.Random(seed)
    while True:
        selected = []
        for index in range(count):
            start = index * len(shards) // count
            stop = (index + 1) * len(shards) // count
            selected.append(shards[rng.randrange(start, stop)])
        total = sum(shard["size"] for shard in selected)
        if total >= target_bytes or count == len(shards):
            return selected
        count += 1
        rng = random.Random(seed)


def repo_slug(repo):
    """Stable filesystem/Hub-safe identifier for one source repository."""
    return repo.strip("/").replace("/", "__")


def destination_path(source_repo, source_path, fmt, upload_prefix=""):
    """Map a source Parquet path to its deterministic encoded destination path.

    The destination keeps the source file's own name so a later run can diff by
    name: <upload_prefix>/<format>/<source-repo-slug>/<source path with new suffix>.
    """
    suffix = ".parquet" if fmt == "parquet-zstd6" else ".vortex"
    encoded = str(Path(source_path).with_suffix(suffix))
    parts = [upload_prefix.strip("/"), fmt, repo_slug(source_repo), encoded]
    return "/".join(part.strip("/") for part in parts if part.strip("/"))


def diff_selection(source_repo, selection, formats, existing_sink_files, upload_prefix=""):
    """Split each (shard, format) pair into missing versus already-present.

    A destination file with the same mapped name counts as present regardless of
    size: conversion changes bytes, so name identity is the sync key.
    """
    missing, present = [], []
    for shard in selection:
        for fmt in formats:
            sink_path = destination_path(source_repo, shard["path"], fmt, upload_prefix)
            entry = {"shard": shard, "format": fmt, "sink_path": sink_path,
                     "remote": existing_sink_files.get(sink_path)}
            (present if entry["remote"] is not None else missing).append(entry)
    return missing, present


def select_shards(source, shards):
    if source["mode"] == "all":
        return shards
    if source["mode"] == "first":
        return shards[:source["limit"]]
    return select_evenly(shards, source["target_size"], source["seed"])


# ------------------------------------------------------------- checkpoints and reports


def fits_download_buffer(inflight_bytes, next_bytes, limit_bytes):
    """Allow one oversized shard, but never grow an occupied buffer past its limit."""
    return inflight_bytes == 0 or inflight_bytes + next_bytes <= limit_bytes


def needs_source_download(file_state, outputs):
    """Return whether any output lacks both a remote commit and reusable local encoding."""
    for fmt, local_path in outputs.items():
        output_state = file_state.get("outputs", {}).get(fmt, {})
        upload_complete = output_state.get("upload", {}).get("status") == "complete"
        local_complete = (
            local_path.exists()
            and local_path.stat().st_size == output_state.get("metrics", {}).get("size_bytes")
        )
        encoding_complete = output_state.get("status") == "complete" and local_complete
        if not upload_complete and not encoding_complete:
            return True
    return False


def resumability_config(config):
    """Reduce a run config to the fields that determine checkpoint compatibility.

    A checkpoint stays valid across plan edits (trimmed or re-diffed chunks) and
    operational tuning such as --delete-after-upload: checkpoint states are keyed
    by source path and destination paths are deterministic. Only a different
    pinned source revision or a different destination invalidates it.
    """
    source = config.get("plan_source", {})
    return {"repo": source.get("repo"), "revision": source.get("revision"),
            "destination": config.get("destination")}


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def load_checkpoint(path):
    if not path.exists():
        return {"version": 1, "files": {}}
    with path.open() as source:
        checkpoint = json.load(source)
    if checkpoint.get("version") != 1 or not isinstance(checkpoint.get("files"), dict):
        raise RuntimeError(f"unsupported or invalid checkpoint: {path}")
    return checkpoint


def checkpoint_records(checkpoint):
    records = []
    for source_name in sorted(checkpoint["files"]):
        outputs = checkpoint["files"][source_name].get("outputs", {})
        for fmt in sorted(outputs):
            if outputs[fmt].get("status") == "complete" and outputs[fmt].get("metrics"):
                source_size = checkpoint["files"][source_name]["size"]
                metrics = outputs[fmt]["metrics"]
                records.append({"source": source_name, "format": fmt,
                                "source_parquet_bytes": source_size,
                                "size_bytes": metrics["size_bytes"],
                                "ratio_to_parquet": metrics["size_bytes"] / source_size,
                                "bytes_saved_vs_parquet": source_size - metrics["size_bytes"],
                                **{key: value for key, value in metrics.items()
                                   if key not in ("size_bytes", "ratio_to_download")}})
    return records


def write_reports(checkpoint, output_dir, chunks):
    records = checkpoint_records(checkpoint)
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    with (metrics_dir / "files.jsonl").open("w") as output:
        for record in records:
            output.write(json.dumps(record, sort_keys=True) + "\n")
    fields = ["source", "format", "source_parquet_bytes", "size_bytes", "ratio_to_parquet",
              "bytes_saved_vs_parquet", "seconds", "mb_per_second", "sha256"]
    with (metrics_dir / "files.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: record.get(field) for field in fields} for record in records)
    totals = {}
    source_totals = {}
    for record in records:
        totals[record["format"]] = totals.get(record["format"], 0) + record["size_bytes"]
        source_totals[record["format"]] = (source_totals.get(record["format"], 0)
                                           + record["source_parquet_bytes"])
    comparisons = {
        fmt: {"source_parquet_bytes": source_totals[fmt], "encoded_bytes": size,
              "ratio_to_parquet": size / source_totals[fmt],
              "bytes_saved_vs_parquet": source_totals[fmt] - size}
        for fmt, size in totals.items()
    }
    summary = {"planned_source_bytes": sum(chunk["size"] for chunk in chunks),
               "planned_files": len(chunks), "format_size_bytes": totals,
               "format_comparison": comparisons}
    atomic_json(metrics_dir / "summary.json", summary)
    safe_print(json.dumps(summary, indent=2, sort_keys=True))


def metric(source_name, fmt, path, input_size, seconds):
    size = path.stat().st_size
    return {"source": source_name, "format": fmt, "size_bytes": size,
            "ratio_to_download": size / input_size, "seconds": seconds,
            "mb_per_second": (input_size / 1_000_000 / seconds) if seconds else None,
            "sha256": file_sha256(path)}


# ----------------------------------------------------------------- source definitions


SOURCE_DEFAULTS = {"revision": "main", "prefix": "", "include": "*.parquet",
                   "filters": [], "mode": "all", "limit": 10,
                   "target_size": DEFAULT_TARGET_BYTES, "seed": 0}


def load_sources(args):
    """Build the list of source repositories from --sources and/or --repo."""
    sources = []
    if args.sources:
        with args.sources.open() as handle:
            listed = json.load(handle)
        if not isinstance(listed, list) or not listed:
            raise RuntimeError(f"{args.sources} must contain a non-empty JSON list of sources")
        for index, entry in enumerate(listed):
            if not isinstance(entry, dict) or "repo" not in entry:
                raise RuntimeError(f"source #{index + 1} in {args.sources} needs a 'repo' key")
            unknown = set(entry) - set(SOURCE_DEFAULTS) - {"repo"}
            if unknown:
                raise RuntimeError(f"source {entry['repo']}: unknown keys {sorted(unknown)}")
            source = {**SOURCE_DEFAULTS, **entry}
            if isinstance(source["target_size"], str):
                source["target_size"] = parse_size(source["target_size"])
            sources.append(source)
    if args.repo:
        sources.append({**SOURCE_DEFAULTS, "repo": args.repo, "revision": args.revision,
                        "prefix": args.prefix or "", "include": args.include,
                        "filters": args.filter, "mode": args.mode, "limit": args.limit,
                        "target_size": args.target_size, "seed": args.seed})
    if not sources:
        raise RuntimeError("no sources: pass --sources <file> and/or --repo <repo>")
    seen = set()
    for source in sources:
        source["prefix"] = source["prefix"].strip("/")
        if source["mode"] not in ("sample", "first", "all"):
            raise RuntimeError(f"source {source['repo']}: mode must be sample, first, or all")
        if source["limit"] < 1:
            raise RuntimeError(f"source {source['repo']}: limit must be positive")
        if source["repo"] in seen:
            raise RuntimeError(f"duplicate source repository: {source['repo']}")
        seen.add(source["repo"])
    return sources


# ----------------------------------------------------------------------- step 1: plan


def list_destination_files(api, destination, source_repo, formats, attempts):
    """List destination files under this source's per-format prefixes."""
    prefixes = ["/".join(part for part in (destination["prefix"].strip("/"), fmt,
                                           repo_slug(source_repo)) if part)
                for fmt in formats]
    existing = {}
    if destination.get("repo"):
        for prefix in prefixes:
            existing.update(list_repository_files(
                api, destination["repo"], destination["revision"], prefix, attempts))
    elif destination.get("local_dir"):
        probe = LocalCopyUploader(Path(destination["local_dir"]))
        for prefix in prefixes:
            existing.update(probe.existing_files(prefix))
    return existing


def build_plan(api, sources, destination, formats, attempts, timeout):
    """Resolve, list, and diff every source; return the executable plan."""
    plan_sources = []
    for source in sources:
        source_commit = resolve_dataset_revision(
            api, source["repo"], source["revision"], attempts, timeout)
        shards = list_shards(api, source["repo"], source_commit, source["prefix"],
                             source["include"], source["filters"], attempts)
        selection = select_shards(source, shards)
        existing = list_destination_files(api, destination, source["repo"], formats, attempts)
        missing, present = diff_selection(source["repo"], selection, formats,
                                          existing, destination["prefix"])
        chunks = {}
        for entry in missing:
            shard = entry["shard"]
            chunk = chunks.setdefault(shard["path"], {
                "path": shard["path"], "size": shard["size"], "sha256": shard.get("sha256"),
                "formats": [], "sinks": {}})
            chunk["formats"].append(entry["format"])
            chunk["sinks"][entry["format"]] = entry["sink_path"]
        plan_sources.append({
            "repo": source["repo"], "requested_revision": source["revision"],
            "revision": source_commit, "prefix": source["prefix"],
            "include": source["include"], "filters": source["filters"],
            "mode": source["mode"], "limit": source["limit"],
            "target_bytes": source["target_size"], "seed": source["seed"],
            "selected_files": len(selection),
            "selected_bytes": sum(shard["size"] for shard in selection),
            "already_present": sorted(entry["sink_path"] for entry in present),
            "chunks": [chunks[path] for path in sorted(chunks)],
        })
    return {
        "version": PLAN_VERSION,
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "destination": destination,
        "formats": list(formats),
        "sources": plan_sources,
    }


def summarize_plan(plan):
    for source in plan["sources"]:
        pending_bytes = sum(chunk["size"] for chunk in source["chunks"])
        safe_print(
            f"{source['repo']}@{source['requested_revision']} ({source['revision'][:12]}): "
            f"{source['selected_files']} shards selected, "
            f"{len(source['already_present'])} outputs already in destination, "
            f"{len(source['chunks'])} files to sync ({pending_bytes / 1e9:.2f} GB source)",
            flush=True,
        )


def load_plan(path):
    if not path.exists():
        raise RuntimeError(f"plan not found: {path}; create one with the 'plan' command")
    with path.open() as handle:
        plan = json.load(handle)
    if plan.get("version") != PLAN_VERSION or not isinstance(plan.get("sources"), list):
        raise RuntimeError(f"unsupported or invalid plan: {path}")
    destination = plan.get("destination") or {}
    if not destination.get("repo") and not destination.get("local_dir"):
        raise RuntimeError(f"plan {path} has no destination repo or local_dir")
    formats = plan.get("formats") or []
    if not set(formats) <= set(SUPPORTED_FORMATS):
        raise RuntimeError(f"plan {path} contains unsupported formats: {formats}")
    for source in plan["sources"]:
        for key in ("repo", "revision", "chunks"):
            if key not in source:
                raise RuntimeError(f"plan {path}: source entry missing '{key}'")
        for chunk in source["chunks"]:
            for key in ("path", "size", "formats", "sinks"):
                if key not in chunk:
                    raise RuntimeError(
                        f"plan {path}: chunk for {source['repo']} missing '{key}'")
            if not set(chunk["formats"]) <= set(formats):
                raise RuntimeError(
                    f"plan {path}: chunk {chunk['path']} lists formats outside the plan's")
    return plan


def plan_command(args):
    formats = parse_formats(args.formats)
    sources = load_sources(args)
    destination = {"repo": args.upload_repo,
                   "local_dir": str(args.upload_local_dir) if args.upload_local_dir else None,
                   "revision": args.upload_revision, "prefix": args.upload_prefix}
    api = hub_api()
    for source in sources:
        require_xet_repository(api, source["repo"], source["revision"], args.hub_attempts)
    if args.upload_repo:
        require_xet_repository(api, args.upload_repo, args.upload_revision, args.hub_attempts)
    plan = build_plan(api, sources, destination, formats, args.hub_attempts, args.hub_timeout)
    atomic_json(args.plan, plan)
    summarize_plan(plan)
    total = sum(len(source["chunks"]) for source in plan["sources"])
    safe_print(f"plan written to {args.plan}: {total} files to sync; "
               "inspect or edit it, then execute with the 'run' command", flush=True)
    return 0


# ------------------------------------------------------------------------ step 2: run


def execute_source(api, plan, source, args, vx):
    """Execute one source's planned chunks: download, convert, upload."""
    source_repo = source["repo"]
    source_commit = source["revision"]
    destination = plan["destination"]
    chunks = source["chunks"]
    output_dir = args.output_dir / repo_slug(source_repo)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not chunks:
        safe_print(f"{source_repo}: nothing planned; skipping", flush=True)
        return
    totals = {}
    for chunk in chunks:
        for fmt in chunk["formats"]:
            totals[fmt] = totals.get(fmt, 0) + 1
    if destination.get("repo"):
        uploader = HuggingFaceBatchUploader(
            api, destination["repo"], destination["revision"], args.upload_batch_files,
            totals, args.hub_attempts, args.hub_timeout,
            batch_bytes=max(1, args.upload_buffer_size // len(totals)))
    else:
        uploader = LocalCopyUploader(Path(destination["local_dir"]))
    run_config = {"plan_source": {key: source[key] for key in source if key != "already_present"},
                  "destination": destination,
                  "delete_after_upload": args.delete_after_upload}
    checkpoint_path = output_dir / "checkpoint.json"
    checkpoint = load_checkpoint(checkpoint_path)
    previous_config = checkpoint.get("config")
    if (previous_config is not None
            and resumability_config(previous_config) != resumability_config(run_config)):
        raise RuntimeError(
            f"plan does not match {checkpoint_path}: the source revision or destination "
            "changed; re-plan into the same destination or use another --output-dir")
    checkpoint["config"] = run_config
    checkpoint_lock = threading.Lock()
    atomic_json(checkpoint_path, checkpoint)

    def save_checkpoint():
        with checkpoint_lock:
            atomic_json(checkpoint_path, checkpoint)

    def shard_paths(chunk):
        source_name = chunk["path"]
        short_hash = hashlib.sha256(source_name.encode()).hexdigest()[:10]
        stem = source_name[:-len(".parquet")].replace("/", "__") + "__" + short_hash
        raw_path = output_dir / "downloads" / source_name
        available = {"parquet-zstd6": output_dir / "parquet-zstd6" / f"{stem}.parquet",
                     "vortex": output_dir / "vortex" / f"{stem}.vortex",
                     "vortex-compact": output_dir / "vortex-compact" / f"{stem}.vortex"}
        return raw_path, {fmt: available[fmt] for fmt in chunk["formats"]}

    def chunk_needs_download(chunk):
        _, outputs = shard_paths(chunk)
        return needs_source_download(checkpoint["files"].setdefault(
            chunk["path"], {"size": chunk["size"], "outputs": {}}), outputs)

    missing_chunks = [chunk for chunk in chunks if chunk_needs_download(chunk)]
    download_executor = concurrent.futures.ThreadPoolExecutor(max_workers=args.shard_workers)
    upload_executor = concurrent.futures.ThreadPoolExecutor(max_workers=args.upload_workers)
    pending_uploads = []
    download_futures = {}
    download_sizes = {}
    missing_index = 0

    def schedule_download(chunk):
        download_futures[chunk["path"]] = download_executor.submit(
            download_shard, source_repo, source_commit, chunk,
            output_dir / "downloads", args.hub_attempts, args.hub_timeout)
        download_sizes[chunk["path"]] = chunk["size"]

    def fill_download_window(active_bytes=0):
        nonlocal missing_index
        while len(download_futures) < args.shard_workers:
            if missing_index >= len(missing_chunks):
                break
            next_chunk = missing_chunks[missing_index]
            inflight_bytes = active_bytes + sum(download_sizes.values())
            if not fits_download_buffer(
                    inflight_bytes, next_chunk["size"], args.download_buffer_size):
                break
            schedule_download(next_chunk)
            missing_index += 1

    fill_download_window()

    def apply_committed_batch(result):
        if not result.get("committed"):
            return
        by_path = {entry["hub_path"]: entry for entry in result["committed"]}
        with checkpoint_lock:
            for file_state in checkpoint["files"].values():
                for output_state in file_state.get("outputs", {}).values():
                    sink_path = output_state.get("upload", {}).get("hub_path")
                    entry = by_path.get(sink_path)
                    if entry is None:
                        continue
                    if args.delete_after_upload:
                        Path(entry["local_path"]).unlink(missing_ok=True)
                    output_state["upload"] = {"status": "complete", "hub_path": sink_path,
                                              "url": result["url"],
                                              "commit_id": result.get("commit_id"),
                                              "batch_size_bytes": result["size_bytes"],
                                              "batch_start": result.get("start"),
                                              "batch_end": result.get("end"),
                                              "batch_total_files": result.get("total_files")}
                    output_state["local_deleted"] = args.delete_after_upload
            atomic_json(checkpoint_path, checkpoint)

    for position, chunk in enumerate(chunks, 1):
        source_name = chunk["path"]
        raw, outputs = shard_paths(chunk)
        safe_print(
            f"[{position}/{len(chunks)}] {source_name} ({chunk['size'] / 1e9:.2f} GB, "
            f"formats: {', '.join(chunk['formats'])})",
            flush=True,
        )
        state = checkpoint["files"].setdefault(source_name, {"size": chunk["size"], "outputs": {}})
        completed_outputs = []
        for fmt, destination_file in outputs.items():
            output_state = state["outputs"].get(fmt, {})
            upload_complete = output_state.get("upload", {}).get("status") == "complete"
            if output_state.get("status") == "complete" and upload_complete:
                completed_outputs.append(fmt)
        if len(completed_outputs) == len(outputs):
            state["status"] = "complete"
            save_checkpoint()
            safe_print("  checkpoint complete; nothing to do", flush=True)
            continue
        if source_name in download_futures:
            state["status"] = "downloading"
            save_checkpoint()
            try:
                downloaded_path, download_seconds = download_futures.pop(source_name).result()
                download_sizes.pop(source_name)
                if downloaded_path.resolve() != raw.resolve():
                    raise RuntimeError(f"Hub download returned unexpected path: {downloaded_path}")
                fill_download_window(active_bytes=chunk["size"])
                state["download"] = {"status": "complete", "path": str(raw),
                                     "size_bytes": chunk["size"], "seconds": download_seconds}
                state["status"] = "converting"
                state.pop("error", None)
                save_checkpoint()
                safe_print(
                    f"  downloaded {chunk['size'] / 1e9:.2f} GB in {download_seconds:.1f}s",
                    flush=True,
                )
            except Exception as error:
                state["status"] = "failed"
                state["error"] = f"download: {error}"
                save_checkpoint()
                raise
        else:
            state["status"] = "uploading"
            save_checkpoint()
        all_jobs = {"parquet-zstd6": lambda: parquet_zstd6(raw, outputs["parquet-zstd6"]),
                    "vortex": lambda: run_vx(vx, raw, outputs["vortex"], "btrblocks"),
                    "vortex-compact": lambda: run_vx(vx, raw, outputs["vortex-compact"], "compact")}
        jobs = tuple((fmt, all_jobs[fmt]) for fmt in chunk["formats"])

        def process_format(fmt, job):
            destination_file = outputs[fmt]
            output_state = state["outputs"].setdefault(fmt, {})
            upload_complete = output_state.get("upload", {}).get("status") == "complete"
            local_complete = (destination_file.exists() and
                              destination_file.stat().st_size
                              == output_state.get("metrics", {}).get("size_bytes"))
            if output_state.get("status") == "complete" and upload_complete:
                size = output_state["metrics"]["size_bytes"]
                safe_print(f"  {fmt}: checkpoint complete ({size / 1e9:.2f} GB)", flush=True)
                return
            if not local_complete:
                output_state["status"] = "converting"
                output_state["path"] = str(destination_file)
                save_checkpoint()
                try:
                    seconds = job()
                    output_state["metrics"] = metric(source_name, fmt, destination_file,
                                                     chunk["size"], seconds)
                    output_state["metrics"].pop("source")
                    output_state["metrics"].pop("format")
                    output_state["status"] = "complete"
                    output_state.pop("error", None)
                    save_checkpoint()
                except Exception as error:
                    output_state["status"] = "failed"
                    output_state["error"] = str(error)
                    state["status"] = "failed"
                    state["error"] = f"{fmt}: {error}"
                    save_checkpoint()
                    raise
                safe_print(f"  {fmt}: {destination_file.stat().st_size / 1e9:.2f} GB", flush=True)
            else:
                safe_print(f"  {fmt}: reusing local encoding for upload", flush=True)
            if output_state.get("upload", {}).get("status") != "complete":
                sink_path = chunk["sinks"][fmt]
                output_state["upload"] = {"status": "queued", "hub_path": sink_path}
                save_checkpoint()

                def record_upload_failure(error):
                    with checkpoint_lock:
                        output_state["upload"] = {"status": "failed", "hub_path": sink_path,
                                                  "error": str(error)}
                        state["status"] = "failed"
                        state["error"] = f"upload {fmt}: {error}"
                        atomic_json(checkpoint_path, checkpoint)

                def upload_task():
                    with checkpoint_lock:
                        output_state["upload"] = {"status": "uploading", "hub_path": sink_path}
                        atomic_json(checkpoint_path, checkpoint)
                    if isinstance(uploader, HuggingFaceBatchUploader):
                        try:
                            upload = uploader.upload(destination_file, sink_path,
                                                     format_name=fmt, ordinal=position)
                        except Exception as error:
                            record_upload_failure(error)
                            raise
                        if upload["status"] == "complete":
                            apply_committed_batch(upload)
                            safe_print(f"    committed batch: {upload['url']}", flush=True)
                        else:
                            with checkpoint_lock:
                                output_state["upload"] = upload
                                atomic_json(checkpoint_path, checkpoint)
                            safe_print(f"    preuploaded: {sink_path}", flush=True)
                    else:
                        try:
                            upload = upload_then_maybe_delete(
                                uploader, destination_file, sink_path, args.delete_after_upload)
                        except Exception as error:
                            record_upload_failure(error)
                            raise
                        with checkpoint_lock:
                            output_state["upload"] = upload
                            output_state["local_deleted"] = upload["local_deleted"]
                            atomic_json(checkpoint_path, checkpoint)
                        safe_print(f"    uploaded: {upload['url']}", flush=True)

                pending_uploads.append(upload_executor.submit(upload_task))

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.format_workers) as executor:
            futures = [executor.submit(process_format, fmt, job) for fmt, job in jobs]
            for future in concurrent.futures.as_completed(futures):
                future.result()
        if not args.keep_downloads:
            raw.unlink(missing_ok=True)
            if "download" in state:
                state["download"]["retained"] = False
        elif "download" in state:
            state["download"]["retained"] = True
        state["status"] = "complete"
        state.pop("error", None)
        save_checkpoint()
        fill_download_window()
        max_pending_uploads = args.upload_workers * max(1, len(totals))
        while len(pending_uploads) >= max_pending_uploads:
            pending_uploads.pop(0).result()
    download_executor.shutdown(wait=True, cancel_futures=True)
    for future in pending_uploads:
        future.result()
    upload_executor.shutdown(wait=True, cancel_futures=True)
    if isinstance(uploader, HuggingFaceBatchUploader):
        for final_batch in uploader.flush():
            apply_committed_batch(final_batch)
            safe_print(f"    committed final batch: {final_batch['url']}", flush=True)
    write_reports(checkpoint, output_dir, chunks)


def configure_xet(args):
    if os.environ.get("HF_HUB_DISABLE_XET", "").upper() in {"1", "ON", "YES", "TRUE"}:
        raise RuntimeError("HF_HUB_DISABLE_XET is set; this sync requires Xet")
    args.xet_cache = args.xet_cache.expanduser().resolve()
    args.xet_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_XET_CACHE", str(args.xet_cache))
    try:
        import hf_xet  # noqa: F401
    except ImportError as error:
        raise RuntimeError("hf-xet is required; install it in the active environment") from error
    os.environ.setdefault("HF_XET_NUM_CONCURRENT_RANGE_GETS", str(args.xet_range_gets))
    if args.xet_high_performance:
        os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", str(args.hub_timeout))
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", str(args.hub_timeout))


def run_command(args):
    plan = load_plan(args.plan)
    needs_vortex = any(fmt != "parquet-zstd6"
                       for source in plan["sources"]
                       for chunk in source["chunks"]
                       for fmt in chunk["formats"])
    vx = args.vx.resolve()
    if needs_vortex and not vx.is_file():
        raise RuntimeError(f"vx binary not found: {vx}; build vortex-tui with unstable_encodings")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure_xet(args)
    api = hub_api()
    destination = plan["destination"]
    total = sum(len(source["chunks"]) for source in plan["sources"])
    safe_print(
        f"Executing plan {args.plan} ({plan['created']}): {total} files into "
        f"{destination.get('repo') or destination.get('local_dir')}",
        flush=True,
    )
    safe_print(
        f"Xet: range_gets={args.xet_range_gets}, "
        f"high_performance={args.xet_high_performance}, cache={args.xet_cache}; "
        f"buffers: downloads={args.shard_workers} workers/"
        f"{args.download_buffer_size / 1e9:.1f} GB, uploads={args.upload_workers} workers/"
        f"{args.upload_buffer_size / 1e9:.1f} GB",
        flush=True,
    )
    for source in plan["sources"]:
        execute_source(api, plan, source, args, vx)
    return 0


# ------------------------------------------------------------------------------- CLI


def parse_formats(value):
    formats = tuple(dict.fromkeys(part.strip() for part in value.split(",") if part.strip()))
    if not formats or not set(formats) <= set(SUPPORTED_FORMATS):
        raise RuntimeError("--formats must contain " + ", ".join(SUPPORTED_FORMATS))
    return formats


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    plan = commands.add_parser(
        "plan", help="diff sources against the destination and write the work plan",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    plan.add_argument("--plan", type=Path, default=DEFAULT_PLAN_PATH,
                      help="where to write the plan JSON")
    plan.add_argument("--sources", type=Path,
                      help="JSON file with a list of source repos: "
                           '[{"repo": "org/name", "revision": "main", "prefix": "data", '
                           '"include": "*.parquet", "filters": [], "mode": "all"}]')
    plan.add_argument("--repo", help="single source Hugging Face dataset repository")
    plan.add_argument("--revision", default="main",
                      help="source branch, tag, or commit for --repo (resolved to a commit)")
    plan.add_argument("--prefix", default="",
                      help="source repository folder for --repo; empty scans the root")
    plan.add_argument("--include", default="*.parquet",
                      help="glob matched against paths below the prefix")
    plan.add_argument("--filter", action="append", default=[],
                      help="repeatable full repository-path glob, e.g. 'sample/10BT/*'")
    plan.add_argument("--mode", choices=("sample", "first", "all"), default="all",
                      help="plan every shard, the first --limit shards, or an even sample")
    plan.add_argument("--limit", type=int, default=10,
                      help="number of ordered shards selected by --mode first")
    plan.add_argument("--target-size", type=parse_size, default=DEFAULT_TARGET_BYTES,
                      help="approximate bytes selected by --mode sample")
    plan.add_argument("--seed", type=int, default=0)
    plan.add_argument("--formats", default=",".join(SUPPORTED_FORMATS),
                      help="comma-separated outputs")
    plan.add_argument("--upload-repo",
                      help="vortex-data destination dataset repo diffed against and synced into")
    plan.add_argument("--upload-local-dir", type=Path,
                      help="test destination: a local Hub-shaped fixture directory")
    plan.add_argument("--upload-revision", default="main")
    plan.add_argument("--upload-prefix", default="",
                      help="destination directory within the destination repo")
    plan.add_argument("--hub-attempts", type=int, default=5)
    plan.add_argument("--hub-timeout", type=int, default=30)

    run = commands.add_parser(
        "run", help="execute a previously written plan",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    run.add_argument("--plan", type=Path, default=DEFAULT_PLAN_PATH,
                     help="plan JSON written by the plan command")
    run.add_argument("--output-dir", type=Path, default=DEFAULT_DATA_DIR)
    run.add_argument("--vx", type=Path, default=Path("target/release/vx"))
    run.add_argument("--format-workers", type=int, default=3,
                     help="maximum concurrent encoders per downloaded shard")
    run.add_argument("--shard-workers", type=int, default=2,
                     help="maximum concurrent source downloads")
    run.add_argument("--download-buffer-size", type=parse_size,
                     default=DEFAULT_DOWNLOAD_BUFFER_BYTES,
                     help="maximum source bytes downloaded or in flight")
    run.add_argument("--upload-workers", type=int, default=2,
                     help="maximum concurrent preuploads")
    run.add_argument("--upload-batch-files", type=int, default=100,
                     help="files per format in each Hub commit")
    run.add_argument("--upload-buffer-size", type=parse_size,
                     default=DEFAULT_UPLOAD_BUFFER_BYTES,
                     help="approximate local output bytes awaiting Hub commits, divided "
                          "evenly across formats")
    run.add_argument("--hub-attempts", type=int, default=5)
    run.add_argument("--hub-timeout", type=int, default=30)
    run.add_argument("--xet-range-gets", type=int, default=4,
                     help="concurrent Xet byte ranges downloaded per file")
    run.add_argument("--xet-cache", type=Path, default=DEFAULT_XET_CACHE,
                     help="local Xet chunk/shard cache")
    run.add_argument("--xet-high-performance", action="store_true",
                     help="allow Xet to maximize host CPU, disk, and network utilization")
    run.add_argument("--keep-downloads", action="store_true")
    run.add_argument("--delete-after-upload", action="store_true",
                     help="remove encoded local files only after checkpointed upload success")
    return parser.parse_args(argv)


def validate_args(args):
    if args.command == "plan":
        if args.upload_repo and args.upload_local_dir:
            raise RuntimeError("--upload-repo and --upload-local-dir are mutually exclusive")
        if not args.upload_repo and not args.upload_local_dir:
            raise RuntimeError("pass --upload-repo or --upload-local-dir")
        if args.limit < 1:
            raise RuntimeError("--limit must be positive")
    else:
        if args.format_workers < 1 or args.format_workers > 3:
            raise RuntimeError("--format-workers must be between 1 and 3")
        if args.shard_workers < 1:
            raise RuntimeError("--shard-workers must be positive")
        if args.download_buffer_size < 1:
            raise RuntimeError("--download-buffer-size must be positive")
        if args.upload_workers < 1:
            raise RuntimeError("--upload-workers must be positive")
        if args.upload_batch_files < 1 or args.upload_batch_files > 100:
            raise RuntimeError("--upload-batch-files must be between 1 and 100")
        if args.upload_buffer_size < 1:
            raise RuntimeError("--upload-buffer-size must be positive")
        if args.xet_range_gets < 1:
            raise RuntimeError("--xet-range-gets must be positive")
    if args.hub_attempts < 1:
        raise RuntimeError("--hub-attempts must be positive")
    if args.hub_timeout < 1:
        raise RuntimeError("--hub-timeout must be positive")


def main():
    args = parse_args()
    validate_args(args)
    if args.command == "plan":
        return plan_command(args)
    return run_command(args)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        safe_print(
            "interrupted; partial downloads and completed outputs are resumable",
            file=sys.stderr,
        )
        sys.exit(130)
    except Exception as error:
        safe_print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
