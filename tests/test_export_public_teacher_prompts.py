import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ExportPublicTeacherPromptsCase(unittest.TestCase):
    def test_omits_gold_patch_and_tests_from_teacher_input(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task_file = root / "tasks.jsonl"
            task_file.write_text(json.dumps({
                "instance_id": "rich_123",
                "repo": "Textualize/rich",
                "base_commit": "a" * 40,
                "problem_statement": "Fix the formatter edge case",
                "hints_text": "Check the console implementation",
                "patch": "GOLD_PATCH_SECRET",
                "test_patch": "GOLD_TEST_SECRET",
            }) + "\n", encoding="utf-8")
            output = root / "blind.jsonl"
            result = subprocess.run(
                [sys.executable, "tools/export_public_teacher_prompts.py", "--tasks", str(task_file), "--output", str(output)],
                cwd=ROOT, text=True, capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            exported = json.loads(output.read_text(encoding="utf-8"))
            self.assertTrue(exported["blind_to_reference"])
            self.assertEqual(exported["snapshot"], "snapshots/rich_123.tgz")
            self.assertNotIn("patch", exported)
            self.assertNotIn("test_patch", exported)
            self.assertNotIn("GOLD_PATCH_SECRET", json.dumps(exported))
            self.assertNotIn("GOLD_TEST_SECRET", json.dumps(exported))


if __name__ == "__main__":
    unittest.main()
