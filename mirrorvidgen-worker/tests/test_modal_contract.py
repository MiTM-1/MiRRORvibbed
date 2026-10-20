import ast
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
MODAL_APP = ROOT / "modal_mirrorvidgen.py"
DOCKERFILE = ROOT / "mirrorvidgen-worker" / "Dockerfile.modal"


class ModalContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = MODAL_APP.read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source)
        cls.dockerfile = DOCKERFILE.read_text(encoding="utf-8")

    def test_exact_gpu_and_scale_to_zero_are_pinned(self):
        self.assertIn('GPU_KIND = "RTX-PRO-6000"', self.source)
        self.assertGreaterEqual(self.source.count("min_containers=0"), 3)
        self.assertGreaterEqual(self.source.count("max_containers=1"), 3)
        self.assertGreaterEqual(self.source.count("scaledown_window=2"), 3)

    def test_existing_worker_contract_is_preserved(self):
        self.assertIn("from handler import handler", self.source)
        self.assertIn('{"id": job_id, "input": payload}', self.source)
        self.assertIn("COPY mirrorvision_dispatch.py /handler.py", self.dockerfile)
        self.assertIn("COPY mirrorvision_dispatch.py /mirrorvision_dispatch.py", self.dockerfile)
        self.assertIn("COPY handler.py /h3_baseline_handler.py", self.dockerfile)
        self.assertIn("import h3_baseline_handler as baseline", (ROOT / "mirrorvidgen-worker" / "h3_addon_handler.py").read_text(encoding="utf-8"))
        self.assertIn("cp /h3_addon_handler.py /h3_addon_handler_existing.py", self.dockerfile)
        self.assertIn("context_dir=WORKER_CONTEXT", self.source)

    def test_volume_keeps_runpod_compatible_layout(self):
        self.assertIn('VOLUME_ROOT = Path("/runpod-volume")', self.source)
        self.assertIn("MIRRORVIDGEN_MODEL_ROOT=/runpod-volume/models", self.dockerfile)
        self.assertIn("FREEVIDEO_HOME=/runpod-volume/freevideo-h3", self.dockerfile)

    def test_native_and_h3_work_are_separate(self):
        names = {
            node.name
            for node in self.tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        self.assertTrue({"native_job", "h3_job", "h3_bootstrap_job", "maintenance_job", "submit", "status"}.issubset(names))
        self.assertIn('{"h3_runtime_audit", "h3_ref2va_status"}', self.source)
        self.assertIn('command == "h3-audit"', self.source)
        self.assertIn('command == "h3-bootstrap"', self.source)
        self.assertIn("timeout=7200", self.source)
        self.assertIn("timeout=2100", self.source)
        self.assertIn("The bounded bootstrap function only accepts h3_install", self.source)
        self.assertIn("def _ensure_comfy()", self.source)
        self.assertIn('"/comfyui/main.py"', self.source)
        self.assertIn("if _requires_comfy(payload):", self.source)
        self.assertIn('"h3_reference", "ref2va"', self.source)
        self.assertIn('action in {"h3_comfy_runtime_audit", "h3_ref2va_schema_audit", "h3_ref2va_preflight"}', self.source)
        dispatcher = (ROOT / "mirrorvidgen-worker" / "mirrorvision_dispatch.py").read_text(encoding="utf-8")
        self.assertIn('action == "mirrorvision_health"', dispatcher)
        self.assertIn('{"action": "mirrorvision_health"}', self.source)


if __name__ == "__main__":
    unittest.main()

