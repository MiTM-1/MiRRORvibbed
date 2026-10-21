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

    def test_modal_gvisor_uses_conservative_rss_memory_guard(self):
        patch = (ROOT / "mirrorvidgen-worker" / "patch_freevideo_modal.py").read_text(encoding="utf-8")
        self.assertIn("FREEVIDEO_ALLOW_RSS_GUARD", patch)
        self.assertIn("RSS-fallback", patch)
        self.assertIn("rss if rss_fallback", patch)
        self.assertIn("COPY patch_freevideo_modal.py", self.dockerfile)
        self.assertIn("/tmp/patch_freevideo_modal.py", self.dockerfile)
        self.assertIn("ENV FREEVIDEO_ALLOW_RSS_GUARD=1", self.dockerfile)

    def test_modal_volume_uses_image_python_without_hardlinks(self):
        setup = (ROOT / "mirrorvidgen-worker" / "h3-setup.sh").read_text(encoding="utf-8")
        self.assertIn("FREEVIDEO_PYTHON=/opt/venv/bin/python", self.dockerfile)
        self.assertIn("python3.12-dev", self.dockerfile)
        self.assertIn("FREEVIDEO_UV_LINK_MODE=copy", self.dockerfile)
        self.assertIn("FREEVIDEO_USE_PIP=1", self.dockerfile)
        self.assertIn("Applying Modal Volume Python-runtime compatibility fix", setup)
        self.assertIn("requested = self.env.get('FREEVIDEO_PYTHON')", setup)
        self.assertIn("[requested, '-m', 'venv', '--copies', folder]", setup)
        self.assertIn("Using verified Modal image Python", setup)
        self.assertIn("FREEVIDEO_USE_PIP", setup)
        self.assertIn("args = [executable, '-m', 'pip'", setup)
        self.assertIn("Applying Modal model-support installer compatibility fix", setup)
        self.assertIn("if env.get('FREEVIDEO_USE_PIP') == '1' else", setup)
        self.assertIn("Pinned FreeVideo model-support installer changed", setup)
        self.assertIn("UV_LINK_MODE=os.environ.get('FREEVIDEO_UV_LINK_MODE'", setup)

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
        self.assertIn('"h3_runtime_audit"', self.source)
        self.assertIn('"h3_ref2va_status"', self.source)
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

    def test_private_site_media_requests_use_server_only_bearer(self):
        renderer = (ROOT / "mirrorvidgen-worker" / "mirrorvision_renderer.py").read_text(encoding="utf-8")
        dispatcher = (ROOT / "mirrorvidgen-worker" / "mirrorvision_dispatch.py").read_text(encoding="utf-8")
        for source in (renderer, dispatcher):
            self.assertIn("MIRRORVISION_SITE_ACCESS_BEARER", source)
            self.assertIn("OAI-Sites-Authorization", source)
        self.assertIn("site_headers('video/mp4')", renderer)
        self.assertIn("_site_headers('image/png')", dispatcher)

    def test_private_preview_api_is_proxy_authenticated_and_gpu_locked(self):
        self.assertIn('@modal.fastapi_endpoint(method="POST", requires_proxy_auth=True)', self.source)
        self.assertIn('def api(request: dict)', self.source)
        self.assertIn('"gpuSubmissionEnabled": False', self.source)
        self.assertIn('"GPU submissions are disabled on the isolated Modal website preview."', self.source)
        self.assertIn('str(payload.get("action") or "").strip().lower() != "mirrorvision_render"', self.source)
        self.assertIn('candidate.hostname == allowed.hostname', self.source)
        self.assertIn('modal.FunctionCall.from_id(call_id).cancel()', self.source)
        self.assertIn('_notify_callback(webhook, state)', self.source)
        self.assertIn('"capabilities", "mirrorvision_health", "h3_runtime_audit", "h3_ref2va_status"', self.source)
        self.assertIn('"https://mirrorvidgen-modal-preview.mitm1.chatgpt.site"', self.source)


if __name__ == "__main__":
    unittest.main()

