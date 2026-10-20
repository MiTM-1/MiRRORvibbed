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
        self.assertIn("COPY handler.py /h3_handler_existing.py", self.dockerfile)

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
        self.assertTrue({"native_job", "h3_job", "submit", "status"}.issubset(names))


if __name__ == "__main__":
    unittest.main()

