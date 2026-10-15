#!/usr/bin/env python3
"""Export blind teacher prompts from Kaggle's public 129-task training set.

Reference patches and test patches are intentionally omitted from every output row.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import PurePosixPath
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    seen = set()
    with args.tasks.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            task = json.loads(line)
            task_id = task.get("instance_id")
            repo = task.get("repo")
            commit = task.get("base_commit")
            issue = task.get("problem_statement")
            if not all(isinstance(value, str) and value.strip() for value in (task_id, repo, commit, issue)):
                raise ValueError(f"Task row {line_number} lacks instance_id/repo/base_commit/problem_statement")
            if task_id in seen:
                raise ValueError(f"Duplicate instance_id: {task_id}")
            seen.add(task_id)
            snapshot = PurePosixPath("snapshots") / f"{task_id}.tgz"
            if snapshot.is_absolute() or any(part in {"", ".", ".."} for part in snapshot.parts):
                raise ValueError(f"Unsafe instance_id for snapshot path: {task_id!r}")
            hints = task.get("hints_text") if isinstance(task.get("hints_text"), str) else ""
            user_prompt = (
                "Fix the reported software issue in the provided repository snapshot. "
                "Use the available repository tools to inspect, edit, and run focused verification. "
                "Return a minimal unified diff and a concise observable action trace.\n\n"
                f"Repository: {repo}\nBase commit: {commit}\n\nIssue:\n{issue}"
            )
            if hints.strip():
                user_prompt += f"\n\nMaintainer hints:\n{hints}"
            rows.append({
                "task_id": task_id,
                "repo": repo,
                "base_commit": commit,
                "snapshot": snapshot.as_posix(),
                "prompt": [{"role": "user", "content": user_prompt}],
                "source_kind": "competition_public_training_task",
                "blind_to_reference": True,
            })

    if not rows:
        raise ValueError("No task rows found")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(json.dumps({
        "task_count": len(rows),
        "input_sha256": hashlib.sha256(args.tasks.read_bytes()).hexdigest(),
        "output": str(args.output),
        "reference_patch_fields_exported": False,
        "test_patch_fields_exported": False,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
