import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class RepairSftBuilderCase(unittest.TestCase):
    def test_selects_only_verified_rollouts_and_splits_repositories(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tasks = root / "tasks.jsonl"
            rollouts = root / "rollouts.jsonl"
            output, manifest = root / "train.jsonl", root / "manifest.json"
            tasks.write_text("\n".join(json.dumps({"id": f"t{i}", "repo": f"org/r{i}", "issue": "parser fails on empty input"}) for i in range(5)) + "\n")
            rollouts.write_text("\n".join(json.dumps({"id": f"t{i}", "resolved": i != 0, "test_exit_code": 0, "trace": "inspect then test", "patch": "diff --git a/x b/x"}) for i in range(5)) + "\n")
            result = subprocess.run([sys.executable, "tools/build_repair_sft.py", "--tasks", str(tasks), "--rollouts", str(rollouts), "--output", str(output), "--manifest", str(manifest), "--expected-public-tasks", "5"], cwd=ROOT, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            summary = json.loads(manifest.read_text())
            self.assertEqual(summary["selected_verified_rollouts"], 4)
            self.assertEqual(summary["patterns"], {"parsing_validation": 4})
            self.assertTrue(output.read_text().strip())
            self.assertTrue(output.with_name("train.holdout.jsonl").read_text().strip())
