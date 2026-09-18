import contextlib
import importlib.util
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


spec = importlib.util.spec_from_file_location(
    "scaling", Path(__file__).resolve().parents[1] / "scripts/benchmark_layerwise_scaling.py"
)
scaling = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scaling)


class ReuseTests(unittest.TestCase):
    def test_reuse_and_recover_incomplete_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old = root / "old"
            old.mkdir()
            dest = root / "new"
            argv = ["scaling", "--group", "small", "--riemannian",
                    "--retraction-implementation", "cholesky", "--output-dir", str(dest)]
            capture = io.StringIO()
            with patch.object(sys, "argv", argv + ["--dry-run"]), contextlib.redirect_stdout(capture):
                scaling.main()
            cases = json.loads(capture.getvalue())
            for case in cases:
                args = dict(tf32=False, optimizer_batch_size=4)
                tokens = iter(case["command"][2:])
                for flag in tokens:
                    args[flag[2:].replace("-", "_")] = (
                        True if flag in ("--liger", "--stable-grad-buffers") else next(tokens)
                    )
                (old / Path(case["output"]).name).write_text(json.dumps(dict(status="complete", args=args)))
            first = old / Path(cases[0]["output"]).name
            first.write_text("{")
            second = old / Path(cases[1]["output"]).name
            second.unlink()
            third = old / Path(cases[2]["output"]).name
            result = json.loads(third.read_text())
            result["status"] = "oom"
            third.write_text(json.dumps(result))

            def run(command, timeout):
                Path(command[command.index("--output") + 1]).write_text('{"status": "complete"}')
                return subprocess.CompletedProcess(command, 0)

            with patch.object(sys, "argv", argv + ["--reuse-results", str(old)]), \
                    patch.object(scaling.subprocess, "run", side_effect=run) as runner, \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(scaling.main(), 0)
            self.assertEqual(runner.call_count, 2)
            manifest = json.loads((dest / "manifest-small.json").read_text())
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(sum("reused_from" in c for c in manifest["cases"]), 88)
            self.assertEqual(first.read_text(), "{")
            self.assertEqual((dest / third.name).read_bytes(), third.read_bytes())

            result["args"]["width"] = 123
            third.write_text(json.dumps(result))
            argv[argv.index(str(dest))] = str(root / "mismatch")
            with patch.object(sys, "argv", argv + ["--reuse-results", str(old)]), \
                    patch.object(scaling.subprocess, "run", side_effect=run), \
                    contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "mismatched"):
                scaling.main()


if __name__ == "__main__":
    unittest.main()
