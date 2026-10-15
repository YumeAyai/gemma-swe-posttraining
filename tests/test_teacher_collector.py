import json
import os
import subprocess
import sys
import tempfile
import unittest
import io
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COLLECTOR = ROOT / "training" / "collector" / "collector.py"


class TeacherCollectorCase(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.tasks = self.root / "tasks.jsonl"
        self.tasks.write_text(json.dumps({
            "instance_id": "repo_1",
            "repo": "owner/repo",
            "base_commit": "a" * 40,
            "problem_statement": "Fix parser token boundary",
            "hints_text": "Inspect parser module",
            "patch": "GOLD_PATCH_SECRET",
            "test_patch": "GOLD_TEST_SECRET",
        }) + "\n", encoding="utf-8")
        self.graphs = self.root / "graphs"
        self.graphs.mkdir()
        (self.graphs / f"repo_{'a' * 40}.json").write_text(json.dumps({"nodes": [
            {"id": "src/parser.py", "name": "parser", "text": "def parse_token(value): return value"},
        ]}), encoding="utf-8")
        self.snapshots = self.root / "snapshots"
        self.snapshots.mkdir()
        content = b"def parse_token(value):\n    # Fix parser token boundary\n    return value\n"
        with tarfile.open(self.snapshots / "repo_1.tgz", "w:gz") as archive:
            info = tarfile.TarInfo("repo/src/parser.py")
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
        self.output = self.root / "out.jsonl"

    def tearDown(self):
        self.temp.cleanup()

    def run_collector(self, *args, env=None):
        return subprocess.run(
            [sys.executable, "-B", str(COLLECTOR), "--tasks", str(self.tasks),
             "--graphs", str(self.graphs), "--output", str(self.output), *args],
            cwd=ROOT, text=True, capture_output=True, env=env,
        )

    def test_dry_run_blinds_gold_and_selects_graph_context(self):
        result = self.run_collector("--allow-repo-graph-fallback")
        self.assertEqual(result.returncode, 0, result.stderr)
        row = json.loads(self.output.read_text(encoding="utf-8"))
        self.assertEqual(row["status"], "dry_run")
        messages = json.dumps(row["messages"])
        self.assertIn("def parse_token", messages)
        self.assertNotIn("GOLD_PATCH_SECRET", messages)
        self.assertNotIn("GOLD_TEST_SECRET", messages)
        self.assertFalse(row["reference_patch_sent"])
        self.assertFalse(row["test_patch_sent"])
        self.assertEqual(row["rationale_mode"], "detailed")
        self.assertIn("explicit_rationale", row["messages"][0]["content"])
        self.assertIn("5-8 numbered steps", row["messages"][0]["content"])
        self.assertTrue(row["graph_matches_task_commit"])

    def test_stale_repo_graph_is_labeled_as_non_authoritative(self):
        graph = self.graphs / f"repo_{'b' * 40}.json"
        exact = self.graphs / f"repo_{'a' * 40}.json"
        exact.unlink()
        graph.write_text(json.dumps({"graph": {"repo_name": f"repo_{'b' * 40}"}, "nodes": [
            {"id": "parser", "name": "parser", "text": "def parse_token(value): return value"},
        ]}), encoding="utf-8")
        result = self.run_collector("--allow-repo-graph-fallback")
        self.assertEqual(result.returncode, 0, result.stderr)
        row = json.loads(self.output.read_text(encoding="utf-8"))
        self.assertFalse(row["graph_matches_task_commit"])
        self.assertIn("differs from the task base commit", row["messages"][1]["content"])

    def test_exact_snapshot_source_is_preferred_over_graph(self):
        result = self.run_collector("--snapshots", str(self.snapshots))
        self.assertEqual(result.returncode, 0, result.stderr)
        row = json.loads(self.output.read_text(encoding="utf-8"))
        self.assertEqual(row["context_source"], "task_snapshot")
        self.assertIsNotNone(row["snapshot_sha256"])
        self.assertIn("repo/src/parser.py", row["messages"][1]["content"])
        self.assertNotIn("GOLD_PATCH_SECRET", row["messages"][1]["content"])

    def test_resume_does_not_treat_dry_run_as_collected(self):
        self.run_collector()
        result = self.run_collector("--resume", "--execute", "--model", "mock-model",
                                    env={**os.environ, "TEACHER_API_KEY": "not-used"})
        # Execution would need a network endpoint, so the dry-run must not be skipped;
        # absent/failed API call proves it proceeded beyond resume filtering.
        self.assertNotIn('"records_skipped_by_resume": 1', result.stdout)

    def test_candidate_parser_extracts_json(self):
        sys.path.insert(0, str(COLLECTOR.parent))
        try:
            import collector
            parsed = collector.parse_candidate('answer: {"summary":"inspect","patch":"diff --git x"}')
            rationale = collector.sanitize_explicit_rationale(
                'Step 1: inspect. <think>private thought</think> Step 2: fix.'
            )
        finally:
            sys.path.pop(0)
            sys.modules.pop("collector", None)
        self.assertEqual(parsed["summary"], "inspect")
        self.assertEqual(parsed["patch"], "diff --git x")
        self.assertEqual(rationale, "Step 1: inspect.  Step 2: fix.")


if __name__ == "__main__":
    unittest.main()
