import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import h3_addon_compare as comparison

class Response:
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def raise_for_status(self): pass
    def iter_content(self, *args): return iter([b"oversize"])

class ComparisonGuardTests(unittest.TestCase):
    def test_requires_explicit_approval(self):
        with patch.dict(comparison.os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "approved budget"):
                comparison.main()

    def test_hash_known_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            file = Path(folder) / "weights"
            file.write_bytes(b"abc")
            self.assertEqual(comparison.digest(file), hashlib.sha256(b"abc").hexdigest())

    def test_never_overwrites_existing_download_target(self):
        with tempfile.TemporaryDirectory() as folder:
            file = Path(folder) / "existing"
            file.write_bytes(b"original")
            with patch.object(comparison.requests, "get", return_value=Response()):
                with self.assertRaises(FileExistsError):
                    comparison.download("https://example.test", file, 100)
            self.assertEqual(file.read_bytes(), b"original")

    def test_download_size_guard(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(comparison.requests, "get", return_value=Response()):
                with self.assertRaisesRegex(RuntimeError, "size guard"):
                    comparison.download("https://example.test", Path(folder) / "partial", 2)
