import copy
import ast
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if importlib.util.find_spec("requests") is None:
    # Graph tests never use HTTP; deployment still uses baseline requests.
    sys.modules["requests"] = types.ModuleType("requests")
import h3_ref2va_generation as production
from h3_addons import AddonUnavailable, MOTION_REVISION, Options, REF2VA_TURBO, adapt_ref2va, duration_plan
from h3_context_store import ContextStore


class AddonTests(unittest.TestCase):
    def setUp(self):
        self.request = {"prompt": "Keep driving", "duration_seconds": 5, "quality": "480p", "aspect_ratio": "9:16"}
        self.prepared = {"images": [{"name": "face.png"}], "videos": [{"name": "tail.mp4", "has_audio": True}],
                         "audios": [{"name": "master.wav"}], "counts": {"images": 1, "videos": 1, "audios": 1}}
        with patch.object(production, "h3_ref2va_status", return_value={"ready": True}):
            self.graph, self.settings = production.build_ref2va_workflow(self.request, self.prepared)

    def adapt(self, options, **kw):
        return adapt_ref2va(self.graph, self.settings, options, job_id="new-job", **kw)

    def test_normal_returns_original_graph_and_settings(self):
        graph, settings = self.adapt(Options())
        self.assertIs(graph, self.graph)
        self.assertIs(settings, self.settings)
        self.assertEqual(graph["h3_sampler_select"]["inputs"]["sampler_name"], "res_multistep")
        self.assertEqual(settings["steps"], 20)

    def test_first_motion_clip_saves_pair_without_trimming(self):
        original = copy.deepcopy(self.graph)
        graph, settings = self.adapt(Options(motion=True))
        self.assertEqual(graph["h3_context_save"]["inputs"]["latent"], ["h3_sample", 0])
        self.assertNotIn("h3_context_trim", graph)
        self.assertEqual(settings["addons"]["trimmed_frames"], 0)
        self.assertEqual(self.graph, original)

    def test_motion_preserves_all_references_and_trims_both_streams(self):
        manifest = {"job_id": "old-job", "width": 480, "height": 832, "fps": 24,
                    "workflow": "ref2va", "motion_revision": MOTION_REVISION}
        graph, settings = self.adapt(Options(motion=True, source_job_id="old-job"),
                                    source_manifest=manifest, staged_latent="h3_addons/new-job/source.safetensors")
        self.assertEqual(graph["h3_ref2va"], self.graph["h3_ref2va"])
        self.assertEqual(graph["h3_create_video"]["inputs"]["audio"], ["h3_context_trim", 1])
        self.assertEqual(graph["h3_context_trim"]["inputs"]["match_tail"], True)
        self.assertEqual(settings["addons"]["delivered_frames"], 102)
        self.assertAlmostEqual(settings["actual_duration_seconds"], 4.25)

    def test_fifteen_seconds_does_not_promise_fifteen_added(self):
        plan = duration_plan(362, continuation=True)
        self.assertAlmostEqual(plan["delivered_duration_seconds"], 340 / 24)
        self.assertLess(plan["delivered_duration_seconds"], 15)

    def test_resolution_mismatch_and_missing_context_fail(self):
        with self.assertRaises(AddonUnavailable):
            self.adapt(Options(motion=True, source_job_id="old-job"))
        with self.assertRaises(AddonUnavailable):
            self.adapt(Options(motion=True, source_job_id="old-job"), staged_latent="source",
                       source_manifest={"width": 768})

    def test_ref2va_balanced_and_missing_lora_fail(self):
        with self.assertRaises(AddonUnavailable):
            self.adapt(Options(speed="turbo_balanced"))
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(AddonUnavailable):
                self.adapt(Options(speed="turbo_fast"), model_root=Path(folder))

    def test_turbo_matches_official_graph_and_keeps_references(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "loras").mkdir()
            with (root / "loras" / REF2VA_TURBO).open("wb") as f:
                f.truncate(1_000_000_000)  # sparse fixture, not a model download
            graph, settings = self.adapt(Options(speed="turbo_fast"), model_root=root)
        self.assertEqual(graph["h3_ref2va"], self.graph["h3_ref2va"])
        self.assertEqual(graph["h3_turbo_shift"]["inputs"]["shift_video"], 12)
        self.assertEqual(graph["h3_scheduler"]["inputs"]["steps"], 4)
        self.assertEqual(graph["h3_sampler_select"]["inputs"]["sampler_name"], "euler")
        self.assertEqual(settings["steps"], 4)
        self.assertEqual(self.settings["steps"], 20)

    def test_invalid_context_ids_and_flags_are_rejected(self):
        for data in [{"motion_context": {"enabled": True, "source_job_id": "../escape"}},
                     {"motion_context": {"enabled": "false"}},
                     {"motion_context": {"enabled": True, "preserve_audio": False}}]:
            with self.assertRaises(AddonUnavailable):
                Options.from_input(data)

    def test_context_commit_stage_and_integrity(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            store = ContextStore(root / "volume", root / "output")
            _, settings = self.adapt(Options(motion=True))
            target = store.output / "h3_addons/new-job/clip_00001.safetensors"
            target.parent.mkdir(parents=True)
            target.write_bytes(b"paired-latent-fixture")
            store.commit("new-job", settings, settings["frames"])
            manifest, staged = store.stage("new-job", "next-job")
            self.assertEqual(manifest["width"], 480)
            self.assertTrue((store.output / staged).is_file())
            with self.assertRaises(AddonUnavailable):
                store.commit("new-job", settings, settings["frames"])
            (store.root / "new-job/context.safetensors").write_bytes(b"corrupt")
            with self.assertRaises(AddonUnavailable):
                store.read("new-job")

    def test_interrupted_context_has_no_committed_manifest(self):
        with tempfile.TemporaryDirectory() as folder:
            store = ContextStore(Path(folder) / "volume", Path(folder) / "output")
            _, settings = self.adapt(Options(motion=True))
            with self.assertRaises(AddonUnavailable):
                store.commit("new-job", settings, settings["frames"])
            self.assertFalse((store.root / "new-job/manifest.json").exists())

    def test_actual_baseline_preflight_does_not_queue_generation(self):
        # Execute the inspected production function itself, replacing only IO.
        source = Path(production.__file__).with_name("handler.py").read_text()
        function = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "run_h3_ref2va_job")
        response = unittest.mock.Mock()
        response.json.return_value = {n["class_type"]: {} for n in self.graph.values()}
        queue = unittest.mock.Mock(side_effect=AssertionError("preflight queued a GPU job"))
        cleanup = unittest.mock.Mock()
        scope = {"time": time, "requests": types.SimpleNamespace(get=lambda *a, **k: response),
                 "COMFY_HOST": "test:8188", "wait_for_comfy": lambda: True,
                 "prepare_ref2va_references": lambda *a: self.prepared,
                 "build_ref2va_workflow": lambda *a: (self.graph, self.settings),
                 "validate_ref2va_workflow": production.validate_ref2va_workflow,
                 "public_prepared_summary": production.public_prepared_summary,
                 "queue_workflow": queue, "cleanup_ref2va_paths": cleanup}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "production-preflight", "exec"), scope)
        result = scope["run_h3_ref2va_job"]({"id": "preflight"}, self.request, preflight_only=True)
        self.assertTrue(result["ready"])
        queue.assert_not_called()
        cleanup.assert_called_once_with(self.prepared)


class CandidateHandlerTests(unittest.TestCase):
    def setUp(self):
        self.baseline = types.ModuleType("h3_baseline_handler")
        self.baseline.handler = unittest.mock.Mock(return_value={"baseline": True})
        self.baseline.is_ref2va_request = lambda data: data.get("workflow") == "ref2va"
        self.baseline.run_h3_ref2va_job = unittest.mock.Mock()
        sys.modules["h3_baseline_handler"] = self.baseline
        sys.modules["runpod"] = types.ModuleType("runpod")
        import h3_addon_handler
        h3_addon_handler.baseline = self.baseline
        self.handler = h3_addon_handler.handler

    def test_default_delegates_exact_job_to_original_handler(self):
        job = {"id": "one", "input": {"prompt": "hello"}}
        self.assertEqual(self.handler(job), {"baseline": True})
        self.baseline.handler.assert_called_once_with(job)

    def test_paid_generation_blocked_and_no_baseline_gpu_call(self):
        with patch.dict(os.environ, {"MIRRORVIDGEN_H3_ADDONS_TEST": "false"}):
            result = self.handler({"id": "one", "input": {"workflow": "ref2va", "generation_speed": "turbo_fast"}})
        self.assertIn("disabled", result["error"])
        self.baseline.run_h3_ref2va_job.assert_not_called()

    def test_combined_test_is_blocked_independently(self):
        with patch.dict(os.environ, {"MIRRORVIDGEN_H3_ADDONS_TEST": "true", "MIRRORVIDGEN_H3_ADDONS_COMBINED_TEST": "false"}):
            result = self.handler({"id": "one", "input": {"workflow": "ref2va", "generation_speed": "turbo_fast", "motion_context": {"enabled": True}}})
        self.assertIn("independently", result["error"])
        self.baseline.run_h3_ref2va_job.assert_not_called()

    def test_fl2va_never_routes_to_addon_or_retired_backend(self):
        result = self.handler({"id": "one", "input": {"engine": "freevideo_h3", "mode": "h3_t2v", "motion_context": {"enabled": True}}})
        self.assertIn("FreeVideo", result["error"])
        self.baseline.handler.assert_not_called()


if __name__ == "__main__":
    unittest.main()
