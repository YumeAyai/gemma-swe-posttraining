#!/usr/bin/env python3
"""Create a trace-first LoRA SFT set from public, patch-verified repair rollouts.

The competition's hidden tasks must never be used here.  This utility deliberately
requires the task list and rollouts to be supplied separately, keeps repositories
disjoint between train and holdout, and accepts only a rollout with a verified
patch.  It does not invent a solution trace from an issue title.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


PATTERNS = (
    ("test_failure", r"\b(test|pytest|assert|regression|spec)\b"),
    ("api_contract", r"\b(api|argument|parameter|return|response|compatib)"),
    ("parsing_validation", r"\b(parse|parser|validat|serializ|json|yaml|unicode)"),
    ("edge_case", r"\b(edge|empty|null|none|zero|boundary|overflow|missing)"),
    ("state_lifecycle", r"\b(cache|state|thread|async|race|close|cleanup|retry)"),
    ("dependency_build", r"\b(import|dependency|package|build|install|version)"),
)


def rows(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    yield line_number, json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON on {path}:{line_number}") from exc


def text(value: object) -> str:
    if isinstance(value, str):
        return value.strip()
    return ""


def classify(issue: str, patch: str) -> str:
    haystack = f"{issue}\n{patch}".lower()
    for name, expression in PATTERNS:
        if re.search(expression, haystack):
            return name
    return "implementation_logic"


def render_trace(rollout: dict, pattern: str) -> str:
    trace = text(rollout.get("trace")) or text(rollout.get("assistant"))
    patch = text(rollout.get("patch"))
    if not trace or not patch:
        raise ValueError("A selected rollout needs non-empty trace/assistant and patch")
    return (
        "<repair_plan>\n"
        f"pattern={pattern}\n"
        "Inspect the reported behavior, identify the smallest affected contract, "
        "make one minimal implementation change, and verify it with the recorded test.\n"
        "</repair_plan>\n"
        f"{trace}\n\n<final_patch>\n{patch}\n</final_patch>"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True, help="Public task metadata JSONL")
    parser.add_argument("--rollouts", type=Path, required=True, help="Patch rollout JSONL")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-public-tasks", type=int, default=175)
    args = parser.parse_args()

    tasks: dict[str, dict] = {}
    for line, task in rows(args.tasks):
        task_id = text(task.get("task_id") or task.get("instance_id") or task.get("id"))
        repository = text(task.get("repo") or task.get("repository"))
        issue = text(task.get("issue") or task.get("problem_statement") or task.get("prompt"))
        if not task_id or not repository or not issue:
            raise ValueError(f"Task {args.tasks}:{line} needs id, repo, and issue/prompt")
        if task_id in tasks:
            raise ValueError(f"Duplicate task id: {task_id}")
        tasks[task_id] = {"repo": repository, "issue": issue}
    if len(tasks) != args.expected_public_tasks:
        raise ValueError(f"Expected {args.expected_public_tasks} public tasks, found {len(tasks)}")

    candidates: dict[str, list[dict]] = defaultdict(list)
    for line, rollout in rows(args.rollouts):
        task_id = text(rollout.get("task_id") or rollout.get("instance_id") or rollout.get("id"))
        if task_id not in tasks:
            raise ValueError(f"Rollout {args.rollouts}:{line} references unknown task {task_id!r}")
        if rollout.get("resolved") is not True or rollout.get("test_exit_code") != 0:
            continue
        if not text(rollout.get("patch")):
            continue
        candidates[task_id].append(rollout)

    selected = []
    for task_id, successful in sorted(candidates.items()):
        # Prefer a short verified trajectory: this repairs failure modes without
        # distilling redundant exploration or a missing final patch.
        rollout = min(successful, key=lambda item: len(text(item.get("trace")) or text(item.get("assistant"))))
        task = tasks[task_id]
        pattern = classify(task["issue"], text(rollout.get("patch")))
        selected.append({
            "task_id": task_id,
            "repo": task["repo"],
            "pattern": pattern,
            "messages": [
                {"role": "user", "content": task["issue"]},
                {"role": "assistant", "content": render_trace(rollout, pattern)},
            ],
        })
    if not selected:
        raise ValueError("No patch-verified successful rollouts were available")

    # Hold out whole repositories; otherwise near-duplicate issues can leak.
    repositories = sorted({record["repo"] for record in selected})
    if len(repositories) < 2:
        raise ValueError("Need verified rollouts from at least two repositories for a leakage-safe holdout")
    holdout_repositories = set(repositories[::5])
    if len(holdout_repositories) == len(repositories):
        holdout_repositories = {repositories[0]}
    train, holdout = [], []
    for record in selected:
        (holdout if record["repo"] in holdout_repositories else train).append(record)
    if not train or not holdout:
        raise ValueError("Repository split produced an empty partition")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in train), encoding="utf-8")
    holdout_path = args.output.with_name(args.output.stem + ".holdout.jsonl")
    holdout_path.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in holdout), encoding="utf-8")
    manifest = {
        "source": "public tasks plus patch-verified rollouts; hidden evaluation excluded",
        "public_task_count": len(tasks), "selected_verified_rollouts": len(selected),
        "train_examples": len(train), "holdout_examples": len(holdout),
        "patterns": dict(sorted(Counter(item["pattern"] for item in selected).items())),
        "train_path": str(args.output), "holdout_path": str(holdout_path),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
