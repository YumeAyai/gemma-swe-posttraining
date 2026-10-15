import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def candidate(record_id, task_id, repo, status="passed", allowed=None, **kwargs):
    return {
        "record_id": record_id,
        "task": {
            "task_id": task_id,
            "repo": repo,
            "prompt": "Fix parser behavior for empty input",
            "repo_snapshot": f"snapshots/{task_id}",
            "test_command": ["python", "-m", "pytest", "tests/test_parser.py", "-q"],
        },
        "source": {
            "kind": "student_rollout",
            "reference": f"run/{record_id}",
            "license": "MIT",
            "rights_review": "approved",
            "allowed_modes": allowed or ["sft", "grpo", "preference"],
            "reasoning_visibility": "observable_trace",
        },
        "messages": [
            {"role": "user", "content": "Fix parser behavior for empty input"},
            {"role": "assistant", "content": f"observable actions and patch {record_id}"},
        ],
        "explicit_rationale": "The issue identifies an empty-input boundary; update its handling.",
        "patch": f"diff --git a/parser.py b/parser.py\n+safe_patch_{record_id}",
        "verification": {
            "status": status,
            "patch_replayable": status == "passed",
            "test_exit_code": 0 if status == "passed" else 1,
            "failure_mode": "tests_failed" if status != "passed" else "",
        },
        "evaluation_only": False,
        **kwargs,
    }


class PreparePosttrainDataCase(unittest.TestCase):
    def test_filters_by_rights_and_verification_splits_by_repo(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot_root = root / "snapshots"
            for index in range(4):
                (snapshot_root / f"task-{index}").mkdir(parents=True)
            records = []
            for index in range(4):
                records.append(candidate(f"ok-{index}", f"task-{index}", f"org/repo-{index}"))
            records.append(candidate("bad-0", "task-0", "org/repo-0", status="failed"))
            pending = candidate("pending-0", "task-0", "org/repo-0")
            pending["source"]["rights_review"] = "pending"
            records.append(pending)
            source = root / "input.jsonl"
            source.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
            output = root / "out"
            result = subprocess.run(
                [sys.executable, "tools/prepare_posttrain_data.py", "--input", str(source),
                 "--output-dir", str(output), "--data-root", str(root), "--validation-fraction", "0.25"],
                cwd=ROOT, text=True, capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            train = [json.loads(line) for line in (output / "sft_train.jsonl").read_text().splitlines()]
            validation = [json.loads(line) for line in (output / "sft_validation.jsonl").read_text().splitlines()]
            grpo = [json.loads(line) for line in (output / "grpo_train.jsonl").read_text().splitlines()]
            pairs = [json.loads(line) for line in (output / "preference_pairs.jsonl").read_text().splitlines()]

            self.assertEqual(manifest["rights_approved_records"], 5)
            self.assertEqual(manifest["sft_train_records"] + manifest["sft_validation_records"], 4)
            self.assertTrue(all(item["explicit_rationale"] for item in train + validation))
            self.assertFalse({item["repo"] for item in train} & {item["repo"] for item in validation})
            self.assertEqual(len(grpo), len({item["task_id"] for item in grpo}))
            self.assertEqual(len(pairs), 1)
            self.assertEqual(pairs[0]["task_id"], "task-0")

    def test_rejects_path_traversal_in_grpo_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "snapshots").mkdir()
            rows = [candidate(f"r{i}", f"t{i}", f"org/r{i}") for i in range(3)]
            rows[0]["task"]["repo_snapshot"] = "../outside"
            input_path = root / "input.jsonl"
            input_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            result = subprocess.run(
                [sys.executable, "tools/prepare_posttrain_data.py", "--input", str(input_path),
                 "--output-dir", str(root / "out"), "--data-root", str(root)],
                cwd=ROOT, text=True, capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            grpo = [json.loads(line) for line in (root / "out/grpo_train.jsonl").read_text().splitlines()]
            self.assertNotIn("t0", {item["task_id"] for item in grpo})

    def test_rejects_private_or_evaluation_training_examples(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [candidate(f"r{i}", f"t{i}", f"org/r{i}") for i in range(3)]
            rows[0]["evaluation_only"] = True
            rows[1]["source"]["reasoning_visibility"] = "private_hidden_reasoning"
            input_path = root / "input.jsonl"
            input_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            result = subprocess.run(
                [sys.executable, "tools/prepare_posttrain_data.py", "--input", str(input_path),
                 "--output-dir", str(root / "out")],
                cwd=ROOT, text=True, capture_output=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("evaluation/hidden examples are prohibited", result.stderr)

    def test_task_split_stratifies_multiple_tasks_within_four_repositories(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [
                candidate(f"r{repo}-{task}", f"task-{repo}-{task}", f"org/repo-{repo}")
                for repo in range(4) for task in range(5)
            ]
            input_path = root / "input.jsonl"
            input_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            result = subprocess.run(
                [sys.executable, "tools/prepare_posttrain_data.py", "--input", str(input_path),
                 "--output-dir", str(root / "out"), "--split-strategy", "task"],
                cwd=ROOT, text=True, capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = json.loads((root / "out/manifest.json").read_text(encoding="utf-8"))
            train = [json.loads(line) for line in (root / "out/sft_train.jsonl").read_text().splitlines()]
            validation = [json.loads(line) for line in (root / "out/sft_validation.jsonl").read_text().splitlines()]
            self.assertEqual(manifest["split_strategy"], "task")
            self.assertFalse(manifest["repo_disjoint"])
            self.assertTrue({row["repo"] for row in train} & {row["repo"] for row in validation})
            self.assertFalse({row["task_id"] for row in train} & {row["task_id"] for row in validation})
            self.assertEqual(len(train) + len(validation), 20)


if __name__ == "__main__":
    unittest.main()
