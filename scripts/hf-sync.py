#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Convert physical Parquet shards in a Hugging Face dataset repository.

The source repository, revision, and folder are explicit. Source revisions are pinned to
an immutable commit before bounded download, conversion, and per-format batch upload.
"""

import abc
import argparse
import concurrent.futures
import csv
import fnmatch
import hashlib
import json
import os
import queue
import random
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

DEFAULT_TARGET_BYTES = 10_000_000_000
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "hf-sync"
DEFAULT_XET_CACHE = DEFAULT_DATA_DIR / "xet-cache"
DEFAULT_DOWNLOAD_BUFFER_BYTES = 64_000_000_000
DEFAULT_UPLOAD_BUFFER_BYTES = 128_000_000_000
DEFAULT_STATUS_INTERVAL = 5.0
PARQUET_BATCH_ROWS = 65_536
LOG_LOCK = threading.Lock()
OPERATIONAL_CONFIG_KEYS = {"xet_cache", "xet_high_performance", "xet_range_gets"}
PLAN_VERSION = 2
VX_CPU_SLOTS = None


def safe_print(*values, file=None, **kwargs):
    """Write progress without allowing a detached output pipe to stop conversion."""
    target = file if file is not None else sys.stdout
    try:
        with LOG_LOCK:
            print(*values, file=target, **kwargs)
    except BrokenPipeError:
        # The replacement must stay open because it becomes a process-global stream.
        replacement = open(os.devnull, "w")  # noqa: SIM115
        if target is sys.stdout:
            sys.stdout = replacement
        elif target is sys.stderr:
            sys.stderr = replacement
        else:
            replacement.close()


def retryable_hub_error(error):
    """Recognize transport failures, including transient errors wrapped by Xet."""
    current = error
    messages = []
    seen = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        messages.append(str(current).lower())
        try:
            import httpx

            if isinstance(current, httpx.TransportError):
                return True
        except ImportError:
            pass
        if isinstance(current, (ConnectionError, TimeoutError)):
            return True
        response = getattr(current, "response", None)
        status = getattr(response, "status_code", None)
        if status in (408, 425, 429) or (status is not None and status >= 500):
            return True
        current = current.__cause__ or current.__context__
    message = " ".join(messages)
    return any(
        fragment in message
        for fragment in (
            "timed out",
            "timeout",
            "request body",
            "connection reset",
            "connection aborted",
            "connection closed",
            "broken pipe",
            "temporarily unavailable",
            "internal error",
        )
    )


def retry_call(operation, attempts, base_delay=1.0, max_delay=60.0):
    """Retry transient Hub/Xet operations with bounded jittered exponential backoff."""
    for attempt in range(1, attempts + 1):
        try:
            return operation()
        except Exception as error:
            if attempt == attempts or not retryable_hub_error(error):
                raise
            delay = min(max_delay, base_delay * 2 ** (attempt - 1))
            delay += random.uniform(0, delay * 0.25)
            safe_print(
                f"transient Hub error (attempt {attempt}/{attempts}): {error}; retrying in {delay:.1f}s",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(delay)


def parse_size(value):
    units = {
        "b": 1,
        "kb": 1000,
        "mb": 1000**2,
        "gb": 1000**3,
        "tb": 1000**4,
        "kib": 1024,
        "mib": 1024**2,
        "gib": 1024**3,
        "tib": 1024**4,
    }
    value = value.strip().lower()
    number = value
    unit = "b"
    for suffix in sorted(units, key=len, reverse=True):
        if value.endswith(suffix):
            number = value[: -len(suffix)]
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
        lambda: api.repo_info(repo, repo_type="dataset", revision=revision, expand=["xetEnabled"]), attempts
    )
    enabled = getattr(info, "xet_enabled", None)
    if enabled is None:
        enabled = info.__dict__.get("xetEnabled")
    if not enabled:
        raise RuntimeError(f"dataset repository is not Xet-enabled: {repo}@{revision}")


def list_shards(api, repo, revision, prefix, include, filters=(), attempts=3):
    shards = []
    tree = retry_call(
        lambda: list(
            api.list_repo_tree(
                repo, path_in_repo=prefix or None, recursive=True, revision=revision, repo_type="dataset"
            )
        ),
        attempts,
    )
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


def destination_path(source_path, fmt, upload_prefix=""):
    """Map a source Parquet path to its deterministic encoded repository path."""
    suffix = ".parquet" if fmt == "parquet-zstd6" else ".vortex"
    encoded = str(Path(source_path).with_suffix(suffix))
    parts = [upload_prefix.strip("/"), fmt, encoded]
    return "/".join(part.strip("/") for part in parts if part.strip("/"))


def list_repository_files(api, repo, revision, prefix="", attempts=3):
    """Return dataset repository file metadata keyed by path."""
    from huggingface_hub.errors import RemoteEntryNotFoundError

    files = {}
    try:
        tree = retry_call(
            lambda: list(
                api.list_repo_tree(
                    repo, path_in_repo=prefix or None, recursive=True, revision=revision, repo_type="dataset"
                )
            ),
            attempts,
        )
    except RemoteEntryNotFoundError:
        return {}
    for item in tree:
        path = getattr(item, "path", None)
        if path is not None and getattr(item, "size", None) is not None:
            files[path] = {"size": item.size, "oid": getattr(item, "blob_id", None), "lfs": getattr(item, "lfs", None)}
    return files


def download_shard(repo, revision, shard, download_root, attempts, etag_timeout):
    from huggingface_hub import hf_hub_download

    started = time.monotonic()
    path = Path(
        retry_call(
            lambda: hf_hub_download(
                repo,
                shard["path"],
                repo_type="dataset",
                revision=revision,
                local_dir=download_root,
                etag_timeout=etag_timeout,
            ),
            attempts,
        )
    )
    elapsed = time.monotonic() - started
    actual_size = path.stat().st_size
    if actual_size != shard["size"]:
        raise RuntimeError(f"downloaded {actual_size} bytes for {path}, expected {shard['size']}")
    if shard.get("sha256") and file_sha256(path) != shard["sha256"]:
        path.unlink(missing_ok=True)
        raise RuntimeError(f"SHA-256 mismatch for {shard['path']}")
    return path, elapsed


def parquet_zstd6(source, destination):
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError("pyarrow is required; install it in the active Python environment") from error
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    parquet = pq.ParquetFile(source)
    started = time.monotonic()
    writer = pq.ParquetWriter(partial, parquet.schema_arrow, compression="zstd", compression_level=6)
    try:
        for batch in parquet.iter_batches(batch_size=PARQUET_BATCH_ROWS):
            writer.write_batch(batch, row_group_size=PARQUET_BATCH_ROWS)
    finally:
        writer.close()
    partial.replace(destination)
    return time.monotonic() - started


def run_vx(vx, source, destination, strategy, cpu=None):
    leased_cpu = False
    if cpu is None and VX_CPU_SLOTS is not None:
        cpu = VX_CPU_SLOTS.get()
        leased_cpu = True
    staging = source.parent / f".{destination.stem}.{strategy}.input.parquet"
    generated = staging.with_suffix(".vortex")
    staging.unlink(missing_ok=True)
    generated.unlink(missing_ok=True)
    os.link(source, staging)
    started = time.monotonic()
    try:
        command = [str(vx), "convert", str(staging), "--strategy", strategy, "--quiet"]
        if cpu is not None:
            command = ["taskset", "--cpu-list", str(cpu), *command]
        environment = os.environ.copy()
        environment["TOKIO_WORKER_THREADS"] = "1"
        subprocess.run(command, check=True, env=environment)
        elapsed = time.monotonic() - started
        destination.parent.mkdir(parents=True, exist_ok=True)
        generated.replace(destination)
        return elapsed
    finally:
        staging.unlink(missing_ok=True)
        generated.unlink(missing_ok=True)
        if leased_cpu:
            VX_CPU_SLOTS.put(cpu)


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

    def __init__(
        self, api, repo, revision, batch_files, totals, attempts=5, timeout=30, batch_bytes=None, planned_batches=None
    ):
        self.api = api
        self.repo = repo
        self.revision = revision
        self.batch_files = batch_files
        self.totals = dict(totals)
        self.attempts = attempts
        self.timeout = timeout
        self.batch_bytes = batch_bytes
        self.planned_batches = {batch["id"]: batch for batch in (planned_batches or [])}
        self.batch_for_path = {
            path: batch["id"] for batch in self.planned_batches.values() for path in batch["destination_paths"]
        }
        self.parent_commit = resolve_dataset_revision(api, repo, revision, attempts, timeout)
        self.pending = (
            {batch_id: [] for batch_id in self.planned_batches} if self.planned_batches else {fmt: [] for fmt in totals}
        )
        self.lock = threading.Lock()

    def upload(self, local_path, destination_path, *, format_name, ordinal, **metadata):
        from huggingface_hub import CommitOperationAdd

        operation = CommitOperationAdd(path_in_repo=destination_path, path_or_fileobj=local_path)
        started = time.monotonic()
        retry_call(
            lambda: self.api.preupload_lfs_files(
                self.repo, additions=[operation], repo_type="dataset", revision=self.revision, num_threads=1
            ),
            self.attempts,
        )
        entry = {
            "operation": operation,
            "local_path": str(local_path),
            "hub_path": destination_path,
            "size_bytes": local_path.stat().st_size,
            "format": format_name,
            "ordinal": ordinal,
        }
        with self.lock:
            pending_key = self.batch_for_path.get(destination_path, format_name)
            if self.planned_batches and pending_key == format_name:
                raise RuntimeError(f"destination is not present in the action plan: {destination_path}")
            pending = self.pending[pending_key]
            pending.append(entry)
            if self.planned_batches:
                expected = len(self.planned_batches[pending_key]["destination_paths"])
                if len(pending) == expected:
                    return self._flush_format_locked(pending_key)
                return {"status": "preuploaded", "hub_path": destination_path, "seconds": time.monotonic() - started}
            pending_bytes = sum(item["size_bytes"] for item in pending)
            if len(pending) >= self.batch_files or (self.batch_bytes is not None and pending_bytes >= self.batch_bytes):
                return self._flush_format_locked(format_name)
        return {"status": "preuploaded", "hub_path": destination_path, "seconds": time.monotonic() - started}

    def flush(self):
        with self.lock:
            results = []
            for format_name in self.pending:
                result = self._flush_format_locked(format_name)
                if result["status"] == "complete":
                    results.append(result)
            return results

    def _flush_format_locked(self, pending_key):
        from huggingface_hub import CommitOperationAdd

        batch = self.pending[pending_key]
        if not batch:
            return {"status": "empty", "committed": []}
        batch.sort(key=lambda entry: entry["ordinal"])
        planned = self.planned_batches.get(pending_key)
        format_name = planned["format"] if planned else pending_key
        start = planned["start"] if planned else batch[0]["ordinal"]
        end = planned["end"] if planned else batch[-1]["ordinal"]
        total_files = planned["total_files"] if planned else self.totals[format_name]
        message = planned["commit_message"] if planned else f"Upload {format_name} files {start}-{end} of {total_files}"
        operations = [entry["operation"] for entry in batch]
        result = None
        for attempt in range(1, self.attempts + 1):
            try:
                result = self.api.create_commit(
                    self.repo,
                    operations=operations,
                    repo_type="dataset",
                    revision=self.revision,
                    parent_commit=self.parent_commit,
                    commit_message=message,
                )
                break
            except Exception:
                remote = list_repository_files(self.api, self.repo, self.revision, attempts=1)
                committed = all(remote.get(entry["hub_path"], {}).get("size") == entry["size_bytes"] for entry in batch)
                if committed:
                    commit_id = resolve_dataset_revision(
                        self.api, self.repo, self.revision, attempts=1, timeout=self.timeout
                    )
                    result = type(
                        "CommitResult",
                        (),
                        {
                            "oid": commit_id,
                            "commit_url": f"https://huggingface.co/datasets/{self.repo}/commit/{commit_id}",
                        },
                    )()
                    break
                if attempt == self.attempts:
                    raise
                time.sleep(2 ** (attempt - 1))
                operations = []
                for entry in batch:
                    operation = CommitOperationAdd(path_in_repo=entry["hub_path"], path_or_fileobj=entry["local_path"])
                    retry_call(
                        lambda op=operation: self.api.preupload_lfs_files(
                            self.repo, additions=[op], repo_type="dataset", revision=self.revision, num_threads=1
                        ),
                        self.attempts,
                    )
                    operations.append(operation)
        if result is None:
            raise RuntimeError(f"failed to commit {format_name} files {start}-{end}")
        committed = [
            {key: entry[key] for key in ("local_path", "hub_path", "size_bytes", "format", "ordinal")}
            for entry in batch
        ]
        total_bytes = sum(entry["size_bytes"] for entry in batch)
        self.pending[pending_key] = []
        self.parent_commit = result.oid
        return {
            "status": "complete",
            "url": result.commit_url,
            "commit_id": result.oid,
            "committed": committed,
            "size_bytes": total_bytes,
            "format": format_name,
            "start": start,
            "end": end,
            "total_files": total_files,
            "commit_message": message,
        }

    def config(self):
        return {
            "type": "huggingface-preupload-batch",
            "repo": self.repo,
            "revision": self.revision,
            "batch_files": self.batch_files,
        }

    def existing_files(self, prefix=""):
        return list_repository_files(self.api, self.repo, self.revision, prefix, self.attempts)


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
        return {
            "status": "complete",
            "hub_path": destination_path,
            "url": output.as_uri(),
            "seconds": time.monotonic() - started,
            "chunks": chunks,
            "bytes_uploaded": bytes_copied,
        }

    def config(self):
        return {"type": "local-copy", "destination": str(self.destination)}

    def existing_files(self, prefix=""):
        root = self.destination / prefix.strip("/")
        if not root.exists():
            return {}
        return {
            str(path.relative_to(self.destination)): {"size": path.stat().st_size}
            for path in root.rglob("*")
            if path.is_file() and not path.name.endswith(".part")
        }


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


def file_sha256(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def fits_download_buffer(inflight_bytes, next_bytes, limit_bytes):
    """Allow one oversized shard, but never grow an occupied buffer past its limit."""
    return inflight_bytes == 0 or inflight_bytes + next_bytes <= limit_bytes


def adapt_concurrency(current, maximum, recent_rates, rate):
    """Adjust bounded concurrency from a short throughput history."""
    previous = sum(recent_rates) / len(recent_rates) if recent_rates else 0
    recent_rates.append(rate)
    if not previous or rate >= previous * 0.95:
        return min(maximum, current + 1)
    if len(recent_rates) == recent_rates.maxlen and rate < previous * 0.75:
        return max(1, current // 2)
    return current


def completed_futures(futures):
    """Yield (key, future) in completion order, removing each from the input mapping."""
    while futures:
        done, _ = concurrent.futures.wait(futures.values(), return_when=concurrent.futures.FIRST_COMPLETED)
        for completed in done:
            key = next(key for key, future in futures.items() if future is completed)
            futures.pop(key)
            yield key, completed


def needs_source_download(file_state, outputs):
    """Return whether any output lacks both a remote commit and reusable local encoding."""
    for fmt, local_path in outputs.items():
        output_state = file_state.get("outputs", {}).get(fmt, {})
        upload_complete = output_state.get("upload", {}).get("status") == "complete"
        local_complete = local_path.exists() and local_path.stat().st_size == output_state.get("metrics", {}).get(
            "size_bytes"
        )
        encoding_complete = output_state.get("status") == "complete" and local_complete
        if not upload_complete and not encoding_complete:
            return True
    return False


def resumability_config(config):
    """Remove transfer-tuning fields that do not affect selected files or encoded bytes."""
    return {key: value for key, value in config.items() if key not in OPERATIONAL_CONFIG_KEYS}


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


class LiveStatus:
    """Periodically publish an atomic, machine-readable scheduler snapshot."""

    def __init__(self, path, interval, snapshot):
        self.path = path
        self.interval = interval
        self.snapshot = snapshot
        self.started = time.monotonic()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name="status-writer", daemon=True)

    def start(self):
        self.thread.start()

    def write(self, final=False):
        value = self.snapshot()
        value.update(
            schema_version=1, elapsed_seconds=time.monotonic() - self.started, updated_at_unix=time.time(), final=final
        )
        atomic_json(self.path, value)

    def _run(self):
        while not self.stop_event.wait(self.interval):
            self.write()

    def close(self):
        self.stop_event.set()
        self.thread.join()
        self.write(final=True)


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
            if outputs[fmt].get("status") == "complete":
                source_size = checkpoint["files"][source_name]["size"]
                metrics = outputs[fmt]["metrics"]
                records.append(
                    {
                        "source": source_name,
                        "format": fmt,
                        "source_parquet_bytes": source_size,
                        "size_bytes": metrics["size_bytes"],
                        "ratio_to_parquet": metrics["size_bytes"] / source_size,
                        "bytes_saved_vs_parquet": source_size - metrics["size_bytes"],
                        **{
                            key: value
                            for key, value in metrics.items()
                            if key not in ("size_bytes", "ratio_to_download")
                        },
                    }
                )
    return records


def write_reports(checkpoint, output_dir, selection):
    records = checkpoint_records(checkpoint)
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    with (metrics_dir / "files.jsonl").open("w") as output:
        for record in records:
            output.write(json.dumps(record, sort_keys=True) + "\n")
    fields = [
        "source",
        "format",
        "source_parquet_bytes",
        "size_bytes",
        "ratio_to_parquet",
        "bytes_saved_vs_parquet",
        "seconds",
        "mb_per_second",
        "sha256",
    ]
    with (metrics_dir / "files.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: record.get(field) for field in fields} for record in records)
    totals = {}
    source_totals = {}
    for record in records:
        totals[record["format"]] = totals.get(record["format"], 0) + record["size_bytes"]
        source_totals[record["format"]] = source_totals.get(record["format"], 0) + record["source_parquet_bytes"]
    comparisons = {
        fmt: {
            "source_parquet_bytes": source_totals[fmt],
            "encoded_bytes": size,
            "ratio_to_parquet": size / source_totals[fmt],
            "bytes_saved_vs_parquet": source_totals[fmt] - size,
        }
        for fmt, size in totals.items()
    }
    summary = {
        "selected_source_bytes": sum(item["size"] for item in selection),
        "selected_files": len(selection),
        "format_size_bytes": totals,
        "format_comparison": comparisons,
    }
    atomic_json(metrics_dir / "summary.json", summary)
    safe_print(json.dumps(summary, indent=2, sort_keys=True))


def metric(source_name, fmt, path, input_size, seconds):
    size = path.stat().st_size
    return {
        "source": source_name,
        "format": fmt,
        "size_bytes": size,
        "ratio_to_download": size / input_size,
        "seconds": seconds,
        "mb_per_second": (input_size / 1_000_000 / seconds) if seconds else None,
        "sha256": file_sha256(path),
    }


def add_selection_arguments(parser):
    parser.add_argument(
        "--mode",
        choices=("sample", "first", "all"),
        default="sample",
        help="sample evenly, take the first --limit shards, or stream every shard",
    )
    parser.add_argument("--limit", type=int, default=10, help="number of ordered shards selected by --mode first")
    parser.add_argument("--target-size", type=parse_size, default=DEFAULT_TARGET_BYTES)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repo", required=True, help="source Hugging Face dataset repository")
    parser.add_argument("--revision", required=True, help="source branch, tag, or commit (resolved before planning)")
    parser.add_argument("--prefix", required=True, help="source repository folder; use / to scan the repository root")
    parser.add_argument(
        "--include", default="*.parquet", help="glob matched against paths below --prefix (default: *.parquet)"
    )
    parser.add_argument(
        "--filter", action="append", default=[], help="repeatable full repository-path glob, e.g. 'sample/10BT/*'"
    )
    parser.add_argument(
        "--formats",
        default="parquet-zstd6,vortex,vortex-compact",
        help="comma-separated outputs: parquet-zstd6,vortex,vortex-compact",
    )
    parser.add_argument("--upload-repo", required=True, help="destination Hugging Face dataset repository")
    parser.add_argument("--upload-revision", default="main")
    parser.add_argument("--upload-prefix", default="")
    parser.add_argument(
        "--upload-batch-files", type=int, default=100, help="maximum actions in each planned Hugging Face commit"
    )
    parser.add_argument(
        "--upload-batch-size",
        type=parse_size,
        default=100_000_000_000,
        help="maximum planned source bytes represented by one target commit",
    )
    parser.add_argument("--hub-attempts", type=int, default=8)
    parser.add_argument("--hub-timeout", type=int, default=30)


def add_operational_arguments(parser):
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--vx", type=Path, default=Path("target/release/vx"))
    parser.add_argument("--format-workers", type=int, default=3)
    parser.add_argument(
        "--transcode-workers",
        type=int,
        default=max(1, os.cpu_count() or 1),
        help="maximum single-core transcodes active across downloaded shards",
    )
    parser.add_argument("--shard-workers", type=int, default=2, help="initial downloads (deprecated alias)")
    parser.add_argument("--download-initial-concurrency", type=int)
    parser.add_argument("--download-max-concurrency", type=int, default=8)
    parser.add_argument("--download-buffer-files", type=int, default=100)
    parser.add_argument("--download-buffer-size", type=parse_size, default=DEFAULT_DOWNLOAD_BUFFER_BYTES)
    parser.add_argument("--upload-workers", type=int, default=2)
    parser.add_argument("--upload-max-concurrency", type=int, default=8)
    parser.add_argument("--upload-local-dir", type=Path, help="copy outputs to a local sink instead of Hugging Face")
    parser.add_argument("--upload-buffer-files", type=int, default=100)
    parser.add_argument("--upload-batch-files", type=int, default=100)
    parser.add_argument("--upload-buffer-size", type=parse_size, default=DEFAULT_UPLOAD_BUFFER_BYTES)
    parser.add_argument("--hub-attempts", type=int, default=8)
    parser.add_argument("--hub-timeout", type=int, default=30)
    parser.add_argument("--xet-range-gets", type=int, default=4)
    parser.add_argument("--xet-cache", type=Path, default=DEFAULT_XET_CACHE)
    parser.add_argument("--xet-high-performance", action="store_true")
    parser.add_argument("--keep-downloads", action="store_true")
    parser.add_argument("--delete-after-upload", action="store_true")
    parser.add_argument("--status-interval", type=float, default=DEFAULT_STATUS_INTERVAL)
    parser.add_argument("--detached", action="store_true")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan", help="discover differences and write an action plan")
    add_selection_arguments(plan)
    plan.add_argument("--plan-file", type=Path, required=True)
    apply = commands.add_parser("apply", help="execute a previously reviewed action plan")
    apply.add_argument("plan_file", type=Path)
    add_operational_arguments(apply)
    return parser.parse_args()


def parse_formats(value):
    formats = tuple(dict.fromkeys(part.strip() for part in value.split(",") if part.strip()))
    supported = {"parquet-zstd6", "vortex", "vortex-compact"}
    if not formats or not set(formats) <= supported:
        raise RuntimeError("--formats must contain parquet-zstd6, vortex, and/or vortex-compact")
    return formats


def select_shards(shards, mode, limit, target_bytes, seed):
    if limit < 1:
        raise RuntimeError("--limit must be positive")
    if mode == "all":
        return shards
    if mode == "first":
        return shards[:limit]
    return select_evenly(shards, target_bytes, seed)


def plan_remote_metadata(remote):
    lfs = remote.get("lfs") or {}
    if not isinstance(lfs, dict):
        lfs = getattr(lfs, "__dict__", {})
    return {"size": remote.get("size"), "oid": remote.get("oid"), "sha256": lfs.get("sha256")}


def create_action_plan(api, args):
    formats = parse_formats(args.formats)
    if args.upload_batch_files < 1 or args.upload_batch_files > 100:
        raise RuntimeError("--upload-batch-files must be between 1 and 100")
    upload_batch_size = getattr(args, "upload_batch_size", 100_000_000_000)
    if upload_batch_size < 1:
        raise RuntimeError("--upload-batch-size must be positive")
    prefix = args.prefix.strip("/")
    require_xet_repository(api, args.repo, args.revision, args.hub_attempts)
    require_xet_repository(api, args.upload_repo, args.upload_revision, args.hub_attempts)
    source_commit = resolve_dataset_revision(api, args.repo, args.revision, args.hub_attempts, args.hub_timeout)
    destination_commit = resolve_dataset_revision(
        api, args.upload_repo, args.upload_revision, args.hub_attempts, args.hub_timeout
    )
    shards = list_shards(api, args.repo, source_commit, prefix, args.include, args.filter, args.hub_attempts)
    selection = select_shards(shards, args.mode, args.limit, args.target_size, args.seed)
    existing = {}
    for fmt in formats:
        format_prefix = "/".join(part for part in (args.upload_prefix.strip("/"), fmt) if part)
        existing.update(
            list_repository_files(api, args.upload_repo, destination_commit, format_prefix, args.hub_attempts)
        )
    chunks = []
    counts = {"create": 0, "skip": 0}
    for ordinal, shard in enumerate(selection, 1):
        actions = []
        for fmt in formats:
            sink_path = destination_path(shard["path"], fmt, args.upload_prefix)
            remote = existing.get(sink_path)
            action = {"action": "skip" if remote else "create", "format": fmt, "destination_path": sink_path}
            if remote:
                action["existing"] = plan_remote_metadata(remote)
            counts[action["action"]] += 1
            actions.append(action)
        chunks.append({"ordinal": ordinal, "source": shard, "actions": actions})
    upload_batches = []
    for fmt in formats:
        format_actions = [
            (chunk, action)
            for chunk in chunks
            for action in chunk["actions"]
            if action["format"] == fmt and action["action"] == "create"
        ]
        total = len(format_actions)
        groups = []
        group = []
        group_bytes = 0
        for member in format_actions:
            source_bytes = member[0]["source"]["size"]
            if group and (len(group) >= args.upload_batch_files or group_bytes + source_bytes > upload_batch_size):
                groups.append(group)
                group = []
                group_bytes = 0
            group.append(member)
            group_bytes += source_bytes
        if group:
            groups.append(group)
        offset = 0
        for members in groups:
            start = offset + 1
            end = offset + len(members)
            batch_id = f"{fmt}-{start}-{end}"
            message = f"Upload {fmt} files {start}-{end} of {total}"
            for _, action in members:
                action["upload_batch"] = batch_id
                action["commit_message"] = message
            upload_batches.append(
                {
                    "id": batch_id,
                    "format": fmt,
                    "start": start,
                    "end": end,
                    "total_files": total,
                    "commit_message": message,
                    "source_ordinals": [chunk["ordinal"] for chunk, _ in members],
                    "destination_paths": [action["destination_path"] for _, action in members],
                    "planned_source_bytes": sum(chunk["source"]["size"] for chunk, _ in members),
                }
            )
            offset = end
    return {
        "version": PLAN_VERSION,
        "kind": "hf-sync-plan",
        "created_at_unix_seconds": time.time(),
        "source": {
            "repo": args.repo,
            "requested_revision": args.revision,
            "revision": source_commit,
            "prefix": prefix,
            "include": args.include,
            "filters": args.filter,
        },
        "destination": {
            "repo": args.upload_repo,
            "requested_revision": args.upload_revision,
            "revision": destination_commit,
            "prefix": args.upload_prefix.strip("/"),
        },
        "selection": {"mode": args.mode, "limit": args.limit, "target_bytes": args.target_size, "seed": args.seed},
        "formats": list(formats),
        "work_chunks": chunks,
        "upload_batches": upload_batches,
        "summary": {
            "source_files": len(selection),
            "source_bytes": sum(shard["size"] for shard in selection),
            "create_actions": counts["create"],
            "skip_actions": counts["skip"],
        },
    }


def load_action_plan(path):
    with path.open() as source:
        plan = json.load(source)
    if (
        plan.get("version") != PLAN_VERSION
        or plan.get("kind") != "hf-sync-plan"
        or not isinstance(plan.get("work_chunks"), list)
    ):
        raise RuntimeError(f"unsupported or invalid action plan: {path}")
    batches = plan.get("upload_batches")
    if not isinstance(batches, list):
        raise RuntimeError(f"unsupported or invalid action plan: {path}")  # noqa: TRY004
    batch_paths = {}
    for batch in batches:
        batch_id = batch.get("id")
        if not batch_id or batch_id in batch_paths or not batch.get("commit_message"):
            raise RuntimeError(f"unsupported or invalid upload batch in {path}")
        for destination in batch.get("destination_paths", []):
            if destination in batch_paths:
                raise RuntimeError(f"destination appears in multiple upload batches: {destination}")
            batch_paths[destination] = batch_id
    create_paths = set()
    for chunk in plan["work_chunks"]:
        if not isinstance(chunk.get("source"), dict) or not isinstance(chunk.get("actions"), list):
            raise RuntimeError(f"unsupported or invalid action plan: {path}")  # noqa: TRY004
        for action in chunk["actions"]:
            if action.get("action") not in ("create", "skip"):
                raise RuntimeError(f"unsupported action in {path}: {action.get('action')}")
            if action["action"] == "create":
                destination = action.get("destination_path")
                create_paths.add(destination)
                if action.get("upload_batch") != batch_paths.get(destination):
                    raise RuntimeError(f"create action has no matching upload batch: {destination}")
    if create_paths != set(batch_paths):
        raise RuntimeError(f"upload batches do not match create actions in {path}")
    return plan


def apply_plan_arguments(args, plan):
    source = plan["source"]
    destination = plan["destination"]
    selection = plan["selection"]
    args.repo = source["repo"]
    args.revision = source["revision"]
    args.prefix = source["prefix"]
    args.include = source["include"]
    args.filter = source["filters"]
    args.upload_repo = destination["repo"]
    args.upload_revision = destination["requested_revision"]
    args.upload_prefix = destination["prefix"]
    args.formats = ",".join(plan["formats"])
    args.mode = selection["mode"]
    args.limit = selection["limit"]
    args.target_size = selection["target_bytes"]
    args.seed = selection["seed"]
    return [chunk["source"] for chunk in plan["work_chunks"]]


def main():
    global VX_CPU_SLOTS
    args = parse_args()
    if args.command == "plan":
        plan = create_action_plan(hub_api(), args)
        atomic_json(args.plan_file, plan)
        safe_print(json.dumps(plan["summary"], indent=2, sort_keys=True))
        safe_print(f"Plan written to {args.plan_file}")
        return 0
    plan = load_action_plan(args.plan_file)
    selection = apply_plan_arguments(args, plan)
    if args.format_workers < 1 or args.format_workers > 3:
        raise RuntimeError("--format-workers must be between 1 and 3")
    if args.transcode_workers < 1:
        raise RuntimeError("--transcode-workers must be positive")
    if args.shard_workers < 1:
        raise RuntimeError("--shard-workers must be positive")
    if args.download_initial_concurrency is None:
        args.download_initial_concurrency = args.shard_workers
    if not 1 <= args.download_initial_concurrency <= args.download_max_concurrency:
        raise RuntimeError("download concurrency must satisfy 1 <= initial <= max")
    if args.download_buffer_size < 1:
        raise RuntimeError("--download-buffer-size must be positive")
    if args.download_buffer_files < 1:
        raise RuntimeError("--download-buffer-files must be positive")
    if args.upload_workers < 1:
        raise RuntimeError("--upload-workers must be positive")
    if not args.upload_workers <= args.upload_max_concurrency:
        raise RuntimeError("upload concurrency must satisfy 1 <= initial <= max")
    if args.status_interval <= 0:
        raise RuntimeError("--status-interval must be positive")
    if args.upload_buffer_files < 1:
        raise RuntimeError("--upload-buffer-files must be positive")
    if args.upload_batch_files < 1 or args.upload_batch_files > 100:
        raise RuntimeError("--upload-batch-files must be between 1 and 100")
    if args.upload_buffer_size < 1:
        raise RuntimeError("--upload-buffer-size must be positive")
    if args.hub_attempts < 1:
        raise RuntimeError("--hub-attempts must be positive")
    if args.hub_timeout < 1:
        raise RuntimeError("--hub-timeout must be positive")
    if args.xet_range_gets < 1:
        raise RuntimeError("--xet-range-gets must be positive")
    formats = parse_formats(args.formats)
    args.prefix = args.prefix.strip("/")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.detached or not sys.stdout.isatty():
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    if os.environ.get("HF_HUB_DISABLE_XET", "").upper() in {"1", "ON", "YES", "TRUE"}:
        raise RuntimeError("HF_HUB_DISABLE_XET is set; this converter requires Xet")
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
    vx = args.vx.resolve()
    if not vx.is_file():
        raise RuntimeError(f"vx binary not found: {vx}; build vortex-tui with unstable_encodings")
    available_cpus = sorted(os.sched_getaffinity(0))
    VX_CPU_SLOTS = queue.Queue()
    for cpu in available_cpus[: min(args.transcode_workers, len(available_cpus))]:
        VX_CPU_SLOTS.put(cpu)
    api = hub_api()
    require_xet_repository(api, args.repo, args.revision, args.hub_attempts)
    if args.upload_local_dir is None:
        require_xet_repository(api, args.upload_repo, args.upload_revision, args.hub_attempts)
    safe_print(
        f"Xet enabled: range_gets={args.xet_range_gets}, "
        f"high_performance={args.xet_high_performance}, cache={args.xet_cache}",
        flush=True,
    )
    safe_print(
        f"Transfer buffers: downloads={args.download_initial_concurrency}-"
        f"{args.download_max_concurrency} workers/"
        f"{args.download_buffer_size / 1e9:.1f} GB, uploads={args.upload_workers}-"
        f"{args.upload_max_concurrency} workers/"
        f"{args.upload_buffer_size / 1e9:.1f} GB",
        flush=True,
    )
    source_commit = plan["source"]["revision"]
    checkpoint_path = args.output_dir / "checkpoint.json"
    checkpoint = load_checkpoint(checkpoint_path)
    current_destination = (
        resolve_dataset_revision(api, args.upload_repo, args.upload_revision, args.hub_attempts, args.hub_timeout)
        if args.upload_local_dir is None
        else plan["destination"]["revision"]
    )
    resumable_commits = {
        output.get("upload", {}).get("commit_id")
        for state in checkpoint.get("files", {}).values()
        for output in state.get("outputs", {}).values()
    }
    resumable_commits.discard(None)
    if current_destination != plan["destination"]["revision"] and current_destination not in resumable_commits:
        raise RuntimeError(
            f"stale plan: destination {args.upload_repo}@{args.upload_revision} changed from "
            f"{plan['destination']['revision']} to {current_destination}; create a new plan"
        )
    create_totals = {
        fmt: sum(
            action["action"] == "create"
            for chunk in plan["work_chunks"]
            for action in chunk["actions"]
            if action["format"] == fmt
        )
        for fmt in formats
    }
    if args.upload_local_dir is not None:
        uploader = LocalCopyUploader(args.upload_local_dir)
    else:
        uploader = HuggingFaceBatchUploader(
            api,
            args.upload_repo,
            args.upload_revision,
            args.upload_batch_files,
            create_totals,
            args.hub_attempts,
            args.hub_timeout,
            batch_bytes=max(1, args.upload_buffer_size // len(formats)),
            planned_batches=plan.get("upload_batches", []),
        )
    run_config = {
        "repo": args.repo,
        "requested_revision": args.revision,
        "revision": source_commit,
        "prefix": args.prefix,
        "include": args.include,
        "filters": args.filter,
        "mode": args.mode,
        "limit": args.limit,
        "target_bytes": args.target_size,
        "formats": list(formats),
        "xet_range_gets": args.xet_range_gets,
        "xet_high_performance": args.xet_high_performance,
        "xet_cache": str(args.xet_cache),
        "seed": args.seed,
        "uploader": uploader.config() if uploader else None,
        "upload_prefix": args.upload_prefix,
        "delete_after_upload": args.delete_after_upload,
        "files": selection,
    }
    previous_config = checkpoint.get("config")
    if previous_config is not None and resumability_config(previous_config) != resumability_config(run_config):
        raise RuntimeError(
            f"arguments do not match {checkpoint_path}; resume with the same arguments or use another --output-dir"
        )
    checkpoint["config"] = run_config
    checkpoint_lock = threading.Lock()
    existing_sink_files = {
        action["destination_path"]: action["existing"]
        for chunk in plan["work_chunks"]
        for action in chunk["actions"]
        if action["action"] == "skip"
    }
    for shard in selection:
        state = checkpoint["files"].setdefault(shard["path"], {"size": shard["size"], "outputs": {}})
        for fmt in formats:
            sink_path = destination_path(shard["path"], fmt, args.upload_prefix)
            remote = existing_sink_files.get(sink_path)
            if remote is None:
                continue
            output_state = state["outputs"].setdefault(fmt, {})
            output_state["status"] = "complete"
            output_state["upload"] = {
                "status": "complete",
                "hub_path": sink_path,
                "size_bytes": remote.get("size"),
                "discovered": True,
            }
            output_state.setdefault(
                "metrics",
                {
                    "size_bytes": remote.get("size"),
                    "ratio_to_download": remote.get("size") / shard["size"] if remote.get("size") else None,
                    "seconds": None,
                    "mb_per_second": None,
                    "sha256": remote.get("oid"),
                },
            )
    atomic_json(checkpoint_path, checkpoint)
    atomic_json(args.output_dir / "metrics" / "selection.json", run_config)

    def save_checkpoint():
        with checkpoint_lock:
            atomic_json(checkpoint_path, checkpoint)

    def shard_paths(shard):
        source_name = shard["path"]
        short_hash = hashlib.sha256(source_name.encode()).hexdigest()[:10]
        stem = source_name[: -len(".parquet")].replace("/", "__") + "__" + short_hash
        raw_path = args.output_dir / "downloads" / source_name
        available = {
            "parquet-zstd6": args.output_dir / "parquet-zstd6" / f"{stem}.parquet",
            "vortex": args.output_dir / "vortex" / f"{stem}.vortex",
            "vortex-compact": args.output_dir / "vortex-compact" / f"{stem}.vortex",
        }
        return raw_path, {fmt: available[fmt] for fmt in formats}

    def shard_needs_download(shard):
        _, outputs = shard_paths(shard)
        return needs_source_download(checkpoint["files"][shard["path"]], outputs)

    missing_shards = [shard for shard in selection if shard_needs_download(shard)]
    download_executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=min(args.download_max_concurrency, max(1, len(missing_shards)))
    )
    upload_executor = concurrent.futures.ThreadPoolExecutor(max_workers=args.upload_max_concurrency)
    pending_uploads = deque()
    upload_control = threading.Condition()
    upload_active = 0
    upload_concurrency = args.upload_workers
    recent_upload_rates = deque(maxlen=4)
    download_futures = {}
    download_sizes = {}
    missing_index = 0
    download_concurrency = args.download_initial_concurrency
    recent_download_rates = deque(maxlen=4)
    pipeline_started = time.monotonic()
    transfer_totals = {
        "download_bytes": 0,
        "download_seconds": 0.0,
        "upload_bytes": 0,
        "upload_seconds": 0.0,
        "download_failures": 0,
        "upload_failures": 0,
    }
    transfer_totals_lock = threading.Lock()

    def run_adaptive_upload(operation, size_bytes):
        nonlocal upload_active, upload_concurrency
        with upload_control:
            while upload_active >= upload_concurrency:
                upload_control.wait()
            upload_active += 1
        started = time.monotonic()
        try:
            return operation()
        finally:
            elapsed = max(time.monotonic() - started, 0.001)
            rate = size_bytes / elapsed
            with transfer_totals_lock:
                transfer_totals["upload_bytes"] += size_bytes
                transfer_totals["upload_seconds"] += elapsed
            with upload_control:
                upload_concurrency = adapt_concurrency(
                    upload_concurrency, args.upload_max_concurrency, recent_upload_rates, rate
                )
                upload_active -= 1
                upload_control.notify_all()

    def schedule_download(shard):
        download_futures[shard["path"]] = download_executor.submit(
            download_shard,
            args.repo,
            source_commit,
            shard,
            args.output_dir / "downloads",
            args.hub_attempts,
            args.hub_timeout,
        )
        download_sizes[shard["path"]] = shard["size"]

    def fill_download_window(active_bytes=0):
        nonlocal missing_index
        while len(download_futures) < min(
            download_concurrency, args.download_max_concurrency, args.download_buffer_files
        ):
            if missing_index >= len(missing_shards):
                break
            next_shard = missing_shards[missing_index]
            inflight_bytes = active_bytes + sum(download_sizes.values())
            if not fits_download_buffer(inflight_bytes, next_shard["size"], args.download_buffer_size):
                break
            schedule_download(next_shard)
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
                    output_state["upload"] = {
                        "status": "complete",
                        "hub_path": sink_path,
                        "url": result["url"],
                        "commit_id": result.get("commit_id"),
                        "batch_size_bytes": result["size_bytes"],
                        "batch_start": result.get("start"),
                        "batch_end": result.get("end"),
                        "batch_total_files": result.get("total_files"),
                    }
                    output_state["local_deleted"] = args.delete_after_upload
            atomic_json(checkpoint_path, checkpoint)

    def process_downloaded_shard(position, shard, source_name, raw, outputs, state):
        all_jobs = {
            "parquet-zstd6": lambda: parquet_zstd6(raw, outputs["parquet-zstd6"]),
            "vortex": lambda: run_vx(vx, raw, outputs["vortex"], "btrblocks"),
            "vortex-compact": lambda: run_vx(vx, raw, outputs["vortex-compact"], "compact"),
        }
        jobs = tuple((fmt, all_jobs[fmt]) for fmt in formats)

        def process_format(fmt, job):
            destination = outputs[fmt]
            output_state = state["outputs"].setdefault(fmt, {})
            upload_complete = output_state.get("upload", {}).get("status") == "complete"
            local_complete = destination.exists() and destination.stat().st_size == output_state.get("metrics", {}).get(
                "size_bytes"
            )
            durable_complete = upload_complete if uploader else local_complete
            complete = output_state.get("status") == "complete" and durable_complete
            if complete:
                size = output_state["metrics"]["size_bytes"]
                location = "Hub" if not destination.exists() else "local"
                safe_print(
                    f"  {fmt}: checkpoint complete ({size / 1e9:.2f} GB, {location})",
                    flush=True,
                )
                return
            if not local_complete:
                output_state["status"] = "converting"
                output_state["path"] = str(destination)
                with checkpoint_lock:
                    atomic_json(checkpoint_path, checkpoint)
                try:
                    seconds = job()
                    output_state["metrics"] = metric(source_name, fmt, destination, shard["size"], seconds)
                    output_state["metrics"].pop("source")
                    output_state["metrics"].pop("format")
                    output_state["status"] = "complete"
                    output_state.pop("error", None)
                    with checkpoint_lock:
                        atomic_json(checkpoint_path, checkpoint)
                except Exception as error:
                    output_state["status"] = "failed"
                    output_state["error"] = str(error)
                    state["status"] = "failed"
                    state["error"] = f"{fmt}: {error}"
                    with checkpoint_lock:
                        atomic_json(checkpoint_path, checkpoint)
                    raise
                safe_print(
                    f"  {fmt}: {destination.stat().st_size / 1e9:.2f} GB",
                    flush=True,
                )
            else:
                safe_print(f"  {fmt}: reusing local encoding for upload", flush=True)
            if uploader and output_state.get("upload", {}).get("status") != "complete":
                sink_path = destination_path(source_name, fmt, args.upload_prefix)
                output_state["upload"] = {"status": "queued", "hub_path": sink_path}
                with checkpoint_lock:
                    atomic_json(checkpoint_path, checkpoint)

                def record_upload_failure(error):
                    with transfer_totals_lock:
                        transfer_totals["upload_failures"] += 1
                    with checkpoint_lock:
                        output_state["upload"] = {"status": "failed", "hub_path": sink_path, "error": str(error)}
                        state["status"] = "failed"
                        state["error"] = f"upload {fmt}: {error}"
                        atomic_json(checkpoint_path, checkpoint)

                def do_upload_task():
                    with checkpoint_lock:
                        output_state["upload"] = {"status": "uploading", "hub_path": sink_path}
                        atomic_json(checkpoint_path, checkpoint)
                    if isinstance(uploader, HuggingFaceBatchUploader):
                        try:
                            upload = uploader.upload(destination, sink_path, format_name=fmt, ordinal=position)
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
                                uploader, destination, sink_path, args.delete_after_upload
                            )
                        except Exception as error:
                            record_upload_failure(error)
                            raise
                        with checkpoint_lock:
                            output_state["upload"] = upload
                            output_state["local_deleted"] = upload["local_deleted"]
                            atomic_json(checkpoint_path, checkpoint)
                        safe_print(f"    uploaded: {upload['url']}", flush=True)

                def upload_task():
                    return run_adaptive_upload(do_upload_task, destination.stat().st_size)

                pending_uploads.append(upload_executor.submit(upload_task))

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.format_workers) as executor:
            futures = [executor.submit(process_format, fmt, job) for fmt, job in jobs]
            for future in concurrent.futures.as_completed(futures):
                future.result()
        if not args.keep_downloads:
            raw.unlink(missing_ok=True)
            if "download" in state:
                state["download"]["retained"] = False
        else:
            if "download" in state:
                state["download"]["retained"] = True
        state["status"] = "complete"
        state.pop("error", None)
        save_checkpoint()

    transcode_slots = max(1, args.transcode_workers // max(1, len(formats)))
    conversion_executor = concurrent.futures.ThreadPoolExecutor(max_workers=transcode_slots)
    conversion_futures = []

    def submit_conversion(position, shard, raw, outputs, state):
        source_name = shard["path"]
        state["status"] = "converting"
        state.pop("error", None)
        save_checkpoint()
        conversion_futures.append(
            conversion_executor.submit(process_downloaded_shard, position, shard, source_name, raw, outputs, state)
        )

    # Admit resumable local sources immediately. Network downloads below are admitted in
    # completion order, so one slow low-ordinal shard cannot strand ready CPU work.
    positions = {shard["path"]: position for position, shard in enumerate(selection, 1)}
    shards_by_name = {shard["path"]: shard for shard in selection}
    for position, shard in enumerate(selection, 1):
        source_name = shard["path"]
        raw, outputs = shard_paths(shard)
        safe_print(
            f"[{position}/{len(selection)}] {source_name} ({shard['size'] / 1e9:.2f} GB)",
            flush=True,
        )
        state = checkpoint["files"].setdefault(source_name, {"size": shard["size"], "outputs": {}})
        completed_outputs = []
        for fmt, destination in outputs.items():
            output_state = state["outputs"].get(fmt, {})
            upload_complete = output_state.get("upload", {}).get("status") == "complete"
            local_complete = destination.exists() and destination.stat().st_size == output_state.get("metrics", {}).get(
                "size_bytes"
            )
            durable_complete = upload_complete if uploader else local_complete
            if output_state.get("status") == "complete" and durable_complete:
                completed_outputs.append(fmt)
        if len(completed_outputs) == len(outputs):
            state["status"] = "complete"
            save_checkpoint()
            safe_print(
                "  file checkpoint/sink scan complete; no download or conversion needed",
                flush=True,
            )
            continue
        if source_name not in download_futures and not shard_needs_download(shard):
            submit_conversion(position, shard, raw, outputs, state)

    def status_snapshot():
        with checkpoint_lock:
            files = list(checkpoint["files"].values())
            outputs = [output for state in files for output in state.get("outputs", {}).values()]
        source_done = sum(state.get("status") == "complete" for state in files)
        converted = sum(output.get("status") == "complete" for output in outputs)
        uploaded = sum(output.get("upload", {}).get("status") == "complete" for output in outputs)
        preuploaded = sum(output.get("upload", {}).get("status") == "preuploaded" for output in outputs)
        with transfer_totals_lock:
            totals = dict(transfer_totals)
        elapsed = max(time.monotonic() - pipeline_started, 0.001)
        conversion_source_bytes = sum(
            state["size"]
            for state in files
            for output in state.get("outputs", {}).values()
            if output.get("status") == "complete" and output.get("metrics", {}).get("seconds")
        )
        return {
            "queues": {
                "download": {
                    "waiting_items": len(missing_shards) - missing_index,
                    "active_items": len(download_futures),
                    "reserved_bytes": sum(download_sizes.values()),
                },
                "transcode": {
                    "waiting_items": sum(not future.running() and not future.done() for future in conversion_futures),
                    "active_items": sum(future.running() for future in conversion_futures),
                    "succeeded_items": sum(
                        future.done() and future.exception() is None for future in conversion_futures
                    ),
                    "active_cpu_slots": args.transcode_workers - VX_CPU_SLOTS.qsize(),
                },
                "upload": {
                    "waiting_items": sum(not future.running() and not future.done() for future in pending_uploads),
                    "active_items": upload_active,
                    "succeeded_items": sum(future.done() and future.exception() is None for future in pending_uploads),
                    "concurrency": upload_concurrency,
                },
            },
            "progress": {
                "source_complete": source_done,
                "source_total": len(selection),
                "outputs_converted": converted,
                "outputs_uploaded": uploaded,
                "outputs_preuploaded": preuploaded,
            },
            "limits": {
                "download_files": args.download_buffer_files,
                "download_bytes": args.download_buffer_size,
                "upload_files": args.upload_buffer_files,
                "upload_bytes": args.upload_buffer_size,
            },
            "throughput": {
                "download_effective_bytes_per_second": totals["download_bytes"] / elapsed,
                "download_worker_bytes_per_second": totals["download_bytes"] / max(totals["download_seconds"], 0.001),
                "conversion_source_bytes_per_second": conversion_source_bytes / elapsed,
                "upload_effective_bytes_per_second": totals["upload_bytes"] / elapsed,
                "upload_worker_bytes_per_second": totals["upload_bytes"] / max(totals["upload_seconds"], 0.001),
            },
            "failures": {"download": totals["download_failures"], "upload": totals["upload_failures"]},
        }

    status = LiveStatus(args.output_dir / "status.json", args.status_interval, status_snapshot)
    status.start()
    try:
        for source_name, completed in completed_futures(download_futures):
            download_sizes.pop(source_name)
            shard = shards_by_name[source_name]
            raw, outputs = shard_paths(shard)
            state = checkpoint["files"][source_name]
            try:
                downloaded_path, download_seconds = completed.result()
                if downloaded_path.resolve() != raw.resolve():
                    raise RuntimeError(f"Hub download returned unexpected path: {downloaded_path}")
                state["download"] = {
                    "status": "complete",
                    "path": str(raw),
                    "size_bytes": shard["size"],
                    "seconds": download_seconds,
                }
                rate = shard["size"] / max(download_seconds, 0.001)
                download_concurrency = adapt_concurrency(
                    download_concurrency, args.download_max_concurrency, recent_download_rates, rate
                )
                with transfer_totals_lock:
                    transfer_totals["download_bytes"] += shard["size"]
                    transfer_totals["download_seconds"] += download_seconds
                submit_conversion(positions[source_name], shard, raw, outputs, state)
                safe_print(
                    f"[{positions[source_name]}/{len(selection)}] downloaded {source_name} in {download_seconds:.1f}s",
                    flush=True,
                )
            except Exception as error:
                with transfer_totals_lock:
                    transfer_totals["download_failures"] += 1
                state["status"] = "failed"
                state["error"] = f"download: {error}"
                save_checkpoint()
                raise
            fill_download_window()
            while len(pending_uploads) >= args.upload_buffer_files:
                pending_uploads.popleft().result()
    finally:
        status.write()
    for future in concurrent.futures.as_completed(conversion_futures):
        future.result()
    conversion_executor.shutdown(wait=True, cancel_futures=True)
    download_executor.shutdown(wait=True, cancel_futures=True)
    for future in pending_uploads:
        future.result()
    upload_executor.shutdown(wait=True, cancel_futures=True)
    if isinstance(uploader, HuggingFaceBatchUploader):
        for final_batch in uploader.flush():
            apply_committed_batch(final_batch)
            safe_print(f"    committed final batch: {final_batch['url']}", flush=True)
    write_reports(checkpoint, args.output_dir, selection)
    status.close()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        safe_print(
            "interrupted; partial downloads and completed outputs are resumable",
            file=sys.stderr,
        )
        sys.exit(130)
    except Exception as error:  # noqa: BLE001
        safe_print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
