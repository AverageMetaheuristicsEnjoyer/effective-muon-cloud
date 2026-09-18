import base64
import gzip
import hashlib
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class ExportTests(unittest.TestCase):
    def test_reordered_chunks_and_incomplete_export(self):
        script = Path(__file__).resolve().parents[1] / "scripts/collect_layerwise_export.py"
        raw = b'{"status":"complete"}\n'
        encoded = base64.b64encode(gzip.compress(raw)).decode()
        parts = [encoded[:20], encoded[20:]]
        digest = hashlib.sha256(raw).hexdigest()
        lines = [f"prefix [1,0]<stdout>:RESULT_CHUNK result.json {i} 2 {digest} {part}\n"
                 for i, part in enumerate(parts)]
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "results"
            command = [sys.executable, str(script), str(destination)]
            completed = subprocess.run(command, input="".join(reversed(lines)), text=True, capture_output=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual((destination / "result.json").read_bytes(), raw)
            incomplete = subprocess.run(command, input=lines[0], text=True, capture_output=True)
            self.assertNotEqual(incomplete.returncode, 0)
            self.assertIn("Incomplete export", incomplete.stderr)
            self.assertEqual((destination / "result.json").read_bytes(), raw)


if __name__ == "__main__":
    unittest.main()
