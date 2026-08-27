# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "hf-sync.py"
SPEC = importlib.util.spec_from_file_location("hf_sync", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class SafePrintTest(unittest.TestCase):
    def test_closed_stdout_cannot_stop_sync(self):
        class BrokenStream:
            def write(self, _value):
                raise BrokenPipeError("detached terminal")

            def flush(self):
                raise BrokenPipeError("detached terminal")

        original = sys.stdout
        replacement = None
        try:
            sys.stdout = BrokenStream()
            MODULE.safe_print("progress", flush=True)
            replacement = sys.stdout
            self.assertIsNot(replacement, original)
            self.assertFalse(isinstance(replacement, BrokenStream))
        finally:
            sys.stdout = original
            if replacement is not None and replacement is not original:
                replacement.close()


class HubHelperTest(unittest.TestCase):
    def test_requires_xet_enabled_repository(self):
        class FakeApi:
            def __init__(self, enabled):
                self.enabled = enabled

            def repo_info(self, *args, **kwargs):
                return types.SimpleNamespace(xet_enabled=self.enabled)

        MODULE.require_xet_repository(FakeApi(True), "owner/repo", "main", 1)
        with self.assertRaisesRegex(RuntimeError, "not Xet-enabled"):
            MODULE.require_xet_repository(FakeApi(False), "owner/repo", "main", 1)

    def test_retry_call_retries_transient_failures_only(self):
        calls = []

        def transient_operation():
            calls.append(None)
            if len(calls) < 3:
                raise TimeoutError("temporary")
            return "complete"

        self.assertEqual(
            MODULE.retry_call(transient_operation, attempts=3, base_delay=0), "complete")
        self.assertEqual(len(calls), 3)

        with self.assertRaises(ValueError):
            MODULE.retry_call(lambda: (_ for _ in ()).throw(ValueError("permanent")),
                              attempts=3, base_delay=0)

    def test_parse_size(self):
        self.assertEqual(MODULE.parse_size("10GB"), 10_000_000_000)
        self.assertEqual(MODULE.parse_size("1kib"), 1024)
        self.assertEqual(MODULE.parse_size("512"), 512)


class DestinationPathTest(unittest.TestCase):
    def test_paths_keep_source_name_and_namespace_by_repo(self):
        self.assertEqual(
            MODULE.destination_path("org/name", "data/part-0.parquet", "vortex"),
            "vortex/org__name/data/part-0.vortex")
        self.assertEqual(
            MODULE.destination_path("org/name", "data/part-0.parquet", "parquet-zstd6", "mirror"),
            "mirror/parquet-zstd6/org__name/data/part-0.parquet")


class DiffSelectionTest(unittest.TestCase):
    def test_splits_missing_and_present_by_destination_name(self):
        selection = [{"path": "data/a.parquet", "size": 10},
                     {"path": "data/b.parquet", "size": 20}]
        existing = {"vortex/org__name/data/a.vortex": {"size": 5}}
        missing, present = MODULE.diff_selection(
            "org/name", selection, ("vortex",), existing)
        self.assertEqual([entry["sink_path"] for entry in present],
                         ["vortex/org__name/data/a.vortex"])
        self.assertEqual([entry["sink_path"] for entry in missing],
                         ["vortex/org__name/data/b.vortex"])

    def test_every_format_is_diffed_independently(self):
        selection = [{"path": "a.parquet", "size": 10}]
        existing = {"vortex/org__name/a.vortex": {"size": 5}}
        missing, present = MODULE.diff_selection(
            "org/name", selection, ("vortex", "parquet-zstd6"), existing)
        self.assertEqual(len(present), 1)
        self.assertEqual([entry["format"] for entry in missing], ["parquet-zstd6"])


class SourcesTest(unittest.TestCase):
    def _args(self, **overrides):
        argv = ["plan", "--upload-repo", "vortex-data/mirror"]
        for key, value in overrides.items():
            argv.extend([key, value])
        return MODULE.parse_args(argv)

    def test_loads_sources_file_with_defaults(self):
        with tempfile.TemporaryDirectory() as scratch:
            listing = Path(scratch) / "sources.json"
            listing.write_text(json.dumps(
                [{"repo": "org/a"},
                 {"repo": "org/b", "mode": "sample", "target_size": "1GB"}]))
            sources = MODULE.load_sources(self._args(**{"--sources": str(listing)}))
        self.assertEqual([source["repo"] for source in sources], ["org/a", "org/b"])
        self.assertEqual(sources[0]["mode"], "all")
        self.assertEqual(sources[1]["target_size"], 1_000_000_000)

    def test_rejects_duplicate_and_unknown_keys(self):
        with tempfile.TemporaryDirectory() as scratch:
            listing = Path(scratch) / "sources.json"
            listing.write_text(json.dumps([{"repo": "org/a"}, {"repo": "org/a"}]))
            with self.assertRaisesRegex(RuntimeError, "duplicate"):
                MODULE.load_sources(self._args(**{"--sources": str(listing)}))
            listing.write_text(json.dumps([{"repo": "org/a", "bogus": 1}]))
            with self.assertRaisesRegex(RuntimeError, "unknown keys"):
                MODULE.load_sources(self._args(**{"--sources": str(listing)}))

    def test_single_repo_flags_still_work(self):
        sources = MODULE.load_sources(self._args(**{"--repo": "org/a", "--prefix": "/data/"}))
        self.assertEqual(sources[0]["repo"], "org/a")
        self.assertEqual(sources[0]["prefix"], "data")

    def test_requires_at_least_one_source(self):
        with self.assertRaisesRegex(RuntimeError, "no sources"):
            MODULE.load_sources(self._args())


class FakeApi:
    """Hub stub: one source shard, empty destination."""

    def __init__(self, shards=None):
        self.shards = shards or [types.SimpleNamespace(path="data/a.parquet", size=10, lfs=None)]

    def dataset_info(self, repo, revision=None, timeout=None):
        return types.SimpleNamespace(sha="deadbeefdeadbeef")

    def repo_info(self, repo, **kwargs):
        return types.SimpleNamespace(xet_enabled=True)

    def list_repo_tree(self, repo, **kwargs):
        return list(self.shards)


class PlanTest(unittest.TestCase):
    def _plan(self, tmp):
        sources = MODULE.load_sources(MODULE.parse_args(
            ["plan", "--repo", "org/a", "--upload-local-dir", str(Path(tmp) / "sink")]))
        destination = {"repo": None, "local_dir": str(Path(tmp) / "sink"),
                       "revision": "main", "prefix": ""}
        return MODULE.build_plan(FakeApi(), sources, destination,
                                 ("vortex", "parquet-zstd6"), attempts=1, timeout=1)

    def test_build_plan_lists_missing_chunks_with_sinks(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = self._plan(tmp)
        source = plan["sources"][0]
        self.assertEqual(source["revision"], "deadbeefdeadbeef")
        self.assertEqual(len(source["chunks"]), 1)
        chunk = source["chunks"][0]
        self.assertEqual(chunk["path"], "data/a.parquet")
        self.assertEqual(sorted(chunk["formats"]), ["parquet-zstd6", "vortex"])
        self.assertEqual(chunk["sinks"]["vortex"], "vortex/org__a/data/a.vortex")
        self.assertEqual(source["already_present"], [])

    def test_build_plan_skips_files_already_in_destination(self):
        with tempfile.TemporaryDirectory() as tmp:
            existing = Path(tmp) / "sink" / "vortex" / "org__a" / "data" / "a.vortex"
            existing.parent.mkdir(parents=True)
            existing.write_bytes(b"x")
            plan = self._plan(tmp)
        chunk = plan["sources"][0]["chunks"][0]
        self.assertEqual(chunk["formats"], ["parquet-zstd6"])
        self.assertEqual(plan["sources"][0]["already_present"],
                         ["vortex/org__a/data/a.vortex"])

    def test_plan_round_trips_through_load_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = self._plan(tmp)
            path = Path(tmp) / "plan.json"
            MODULE.atomic_json(path, plan)
            loaded = MODULE.load_plan(path)
            self.assertEqual(loaded, json.loads(json.dumps(plan)))

    def test_load_plan_rejects_bad_plans(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "plan.json"
            with self.assertRaisesRegex(RuntimeError, "plan not found"):
                MODULE.load_plan(path)
            path.write_text(json.dumps({"version": 99, "sources": []}))
            with self.assertRaisesRegex(RuntimeError, "unsupported or invalid"):
                MODULE.load_plan(path)
            path.write_text(json.dumps({"version": 1, "sources": [],
                                        "destination": {}, "formats": []}))
            with self.assertRaisesRegex(RuntimeError, "no destination"):
                MODULE.load_plan(path)
            path.write_text(json.dumps({
                "version": 1, "destination": {"local_dir": "/tmp/sink"},
                "formats": ["vortex"], "created": "now",
                "sources": [{"repo": "org/a", "revision": "abc",
                             "chunks": [{"path": "a.parquet", "size": 1,
                                         "formats": ["parquet-zstd6"], "sinks": {}}]}]}))
            with self.assertRaisesRegex(RuntimeError, "outside the plan"):
                MODULE.load_plan(path)


class LocalCopyUploaderTest(unittest.TestCase):
    def test_round_trips_and_lists_existing_files(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            payload = root / "input.bin"
            payload.write_bytes(b"vortex" * 1024)
            uploader = MODULE.LocalCopyUploader(root / "sink")
            result = uploader.upload(payload, "vortex/org__name/a.vortex")
            self.assertEqual(result["status"], "complete")
            listed = uploader.existing_files("vortex/org__name")
            self.assertEqual(list(listed), ["vortex/org__name/a.vortex"])
            self.assertEqual(listed["vortex/org__name/a.vortex"]["size"], payload.stat().st_size)


class BufferTest(unittest.TestCase):
    def test_allows_one_oversized_shard_only_when_empty(self):
        self.assertTrue(MODULE.fits_download_buffer(0, 100, 10))
        self.assertFalse(MODULE.fits_download_buffer(1, 100, 10))
        self.assertTrue(MODULE.fits_download_buffer(4, 6, 10))


if __name__ == "__main__":
    unittest.main()
