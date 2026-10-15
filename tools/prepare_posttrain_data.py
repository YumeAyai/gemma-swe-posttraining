#!/usr/bin/env python3
"""Filter provenance-rich repair candidates into leakage-safe SFT/GRPO manifests.

Input is canonical JSONL (see training/README.md). This tool never executes a
repository test command; GRPO execution belongs in a separately sandboxed runner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


SOURCE_KINDS = {"student_rollout", "authorized_teacher", "open_patch", "synthetic_solver"}
VISIBLE_REASONING = {"observable_trace", "authorized_explicit_rationale", "none"}
MODES = {"sft", "grpo", "preference"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected a JSON object at {path}:{line_no}")
            records.append(value)
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in records),
        encoding="utf-8",
    )


def validate_record(row: dict[str, Any], seen_ids: set[str]) -> list[str]:
    errors = []
    rid = row.get("record_id")
    if not isinstance(rid, str) or not rid.strip():
        errors.append("record_id is required")
    elif rid in seen_ids:
        errors.append(f"duplicate record_id: {rid}")
    else:
        seen_ids.add(rid)

    task = row.get("task")
    if not isinstance(task, dict) or not all(
        isinstance(task.get(key), str) and task[key].strip() for key in ("task_id", "repo", "prompt")
    ):
        errors.append("task.task_id, task.repo, and task.prompt are required strings")

    source = row.get("source")
    if not isinstance(source, dict):
        errors.append("source metadata is required")
    else:
        if source.get("kind") not in SOURCE_KINDS:
            errors.append(f"source.kind must be one of {sorted(SOURCE_KINDS)}")
        for key in ("reference", "license", "rights_review"):
            if not isinstance(source.get(key), str) or not source[key].strip():
                errors.append(f"source.{key} is required (use 'unknown' if not yet established)")
        if source.get("rights_review") not in {"approved", "pending", "excluded"}:
            errors.append("source.rights_review must be approved, pending, or excluded")
        if source.get("reasoning_visibility") not in VISIBLE_REASONING:
            errors.append(f"source.reasoning_visibility must be one of {sorted(VISIBLE_REASONING)}")
        allowed = source.get("allowed_modes")
        if not isinstance(allowed, list) or any(mode not in MODES for mode in allowed):
            errors.append(f"source.allowed_modes must be a list drawn from {sorted(MODES)}")

    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        errors.append("messages must be a non-empty chat-message list")
    elif not all(
        isinstance(message, dict)
        and message.get("role") in {"system", "user", "assistant", "tool"}
        and isinstance(message.get("content"), str)
        for message in messages
    ):
        errors.append("each message needs an allowed role and string content")
    elif not any(message["role"] == "user" for message in messages) or not any(
        message["role"] == "assistant" for message in messages
    ):
        errors.append("messages need at least one user and one assistant turn")

    verification = row.get("verification")
    if not isinstance(verification, dict):
        errors.append("verification metadata is required")
    elif verification.get("status") not in {"passed", "failed", "timeout", "unverified"}:
        errors.append("verification.status must be passed/failed/timeout/unverified")

    if not isinstance(row.get("patch"), str) or not row["patch"].strip():
        errors.append("patch must contain the candidate unified diff (empty patches are not training targets)")

    if not isinstance(row.get("evaluation_only"), bool):
        errors.append("evaluation_only must be explicitly true/false")
    elif row["evaluation_only"] or row.get("benchmark_split") in {"test", "hidden", "evaluation"}:
        errors.append("evaluation/hidden examples are prohibited from training inputs")
    return errors


def repo_partitions(repos: set[str], validation_fraction: float) -> tuple[set[str], set[str]]:
    ordered = sorted(repos, key=lambda repo: hashlib.sha256(repo.encode("utf-8")).hexdigest())
    if len(ordered) < 2:
        raise ValueError("At least two distinct repositories are required for a repo-disjoint holdout")
    holdout_count = min(len(ordered) - 1, max(1, round(len(ordered) * validation_fraction)))
    return set(ordered[holdout_count:]), set(ordered[:holdout_count])


def task_partitions(records: list[dict[str, Any]], validation_fraction: float) -> tuple[set[str], set[str]]:
    by_repo: dict[str, set[str]] = defaultdict(set)
    task_repo: dict[str, str] = {}
    for row in records:
        task = row["task"]
        task_id, repo = task["task_id"], task["repo"]
        if task_id in task_repo and task_repo[task_id] != repo:
            raise ValueError(f"Task {task_id} appears under multiple repositories")
        task_repo[task_id] = repo
        by_repo[repo].add(task_id)
    if len(task_repo) < 2:
        raise ValueError("At least two distinct tasks are required for a task-level holdout")
    train, validation = set(), set()
    for repo, task_ids in sorted(by_repo.items()):
        ordered = sorted(task_ids, key=lambda task_id: hashlib.sha256(f"{repo}/{task_id}".encode("utf-8")).hexdigest())
        if len(ordered) < 2:
            train.update(ordered)
            continue
        count = min(len(ordered) - 1, max(1, round(len(ordered) * validation_fraction)))
        validation.update(ordered[:count])
        train.update(ordered[count:])
    return train, validation


def safe_snapshot(root: Path, relative: Any) -> Path | None:
    if not isinstance(relative, str) or not relative.strip():
        return None
    candidate = Path(relative)
    if candidate.is_absolute() or any(part in {"", ".", ".."} for part in candidate.parts):
        return None
    try:
        resolved_root = root.resolve(strict=True)
        resolved = (resolved_root / candidate).resolve(strict=True)
    except (OSError, RuntimeError):
        return None
    if not resolved.is_relative_to(resolved_root) or not resolved.is_dir():
        return None
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Canonical candidate JSONL")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, help="Trusted local root for GRPO repository snapshots")
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--split-strategy", choices=("repo", "task"), default="repo")
    args = parser.parse_args()
    if not 0.05 <= args.validation_fraction <= 0.5:
        raise ValueError("validation-fraction must be between 0.05 and 0.5")

    records = read_jsonl(args.input)
    seen: set[str] = set()
    errors = [(row.get("record_id", f"row-{index}"), validate_record(row, seen)) for index, row in enumerate(records, 1)]
    errors = [(record_id, problems) for record_id, problems in errors if problems]
    if errors:
        preview = "; ".join(f"{record_id}: {', '.join(problems)}" for record_id, problems in errors[:12])
        raise ValueError(f"Input validation failed for {len(errors)} record(s): {preview}")

    usable = [
        row for row in records
        if row["source"]["rights_review"] == "approved"
        and row["source"]["reasoning_visibility"] != "private_hidden_reasoning"
        and row["source"]["rights_review"] != "excluded"
    ]
    repos = {row["task"]["repo"] for row in usable}
    train_repos: set[str] = set()
    validation_repos: set[str] = set()
    if args.split_strategy == "repo":
        train_repos, validation_repos = repo_partitions(repos, args.validation_fraction)
        train_tasks = {row["task"]["task_id"] for row in usable if row["task"]["repo"] in train_repos}
        validation_tasks = {row["task"]["task_id"] for row in usable if row["task"]["repo"] in validation_repos}
    else:
        train_tasks, validation_tasks = task_partitions(usable, args.validation_fraction)
    sft_train, sft_validation, grpo_train = [], [], []
    grpo_task_ids: set[str] = set()
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in usable:
        source, task, verification = row["source"], row["task"], row["verification"]
        split = "train" if task["task_id"] in train_tasks else "validation"
        if "sft" in source["allowed_modes"] and verification["status"] == "passed" and verification.get("patch_replayable") is True:
            item = {
                "record_id": row["record_id"],
                "task_id": task["task_id"],
                "repo": task["repo"],
                "messages": row["messages"],
                "explicit_rationale": row.get("explicit_rationale", ""),
                "patch": row["patch"],
            }
            (sft_train if split == "train" else sft_validation).append(item)
        if split == "train" and "grpo" in source["allowed_modes"] and args.data_root and task["task_id"] not in grpo_task_ids:
            snapshot = safe_snapshot(args.data_root, task.get("repo_snapshot"))
            command = task.get("test_command")
            if snapshot and isinstance(command, list) and command and all(isinstance(arg, str) for arg in command):
                grpo_train.append({
                    "task_id": task["task_id"],
                    "prompt": [{"role": "user", "content": task["prompt"]}],
                    "repo_snapshot": snapshot.relative_to(args.data_root.resolve(strict=True)).as_posix(),
                    "test_command": command,
                    "source_record_id": row["record_id"],
                })
                grpo_task_ids.add(task["task_id"])
        if "preference" in source["allowed_modes"]:
            grouped[(task["task_id"], task["repo"])].append(row)

    preference_pairs = []
    for (task_id, repo), candidates in sorted(grouped.items()):
        chosen = [row for row in candidates if row["verification"]["status"] == "passed" and row["verification"].get("patch_replayable") is True]
        rejected = [row for row in candidates if row["verification"]["status"] in {"failed", "timeout"}]
        if not chosen or not rejected:
            continue
        prompt_sets = {tuple((m["role"], m["content"]) for m in row["messages"] if m["role"] == "user") for row in candidates}
        if len(prompt_sets) != 1:
            continue
        preference_pairs.append({
            "task_id": task_id,
            "repo": repo,
            "chosen": chosen[0]["messages"],
            "rejected": rejected[0]["messages"],
            "chosen_record_id": chosen[0]["record_id"],
            "rejected_record_id": rejected[0]["record_id"],
        })

    out = args.output_dir
    write_jsonl(out / "sft_train.jsonl", sft_train)
    write_jsonl(out / "sft_validation.jsonl", sft_validation)
    write_jsonl(out / "grpo_train.jsonl", grpo_train)
    write_jsonl(out / "preference_pairs.jsonl", preference_pairs)
    manifest = {
        "schema_version": "1.0",
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "input_records": len(records),
        "rights_approved_records": len(usable),
        "excluded_or_pending_records": len(records) - len(usable),
        "approved_by_source_kind": {
            kind: sum(row["source"]["kind"] == kind for row in usable)
            for kind in sorted(SOURCE_KINDS)
        },
        "split_strategy": args.split_strategy,
        "repo_disjoint": args.split_strategy == "repo",
        "train_repositories": sorted(train_repos),
        "validation_repositories": sorted(validation_repos),
        "train_tasks": len(train_tasks),
        "validation_tasks": len(validation_tasks),
        "sft_train_records": len(sft_train),
        "sft_validation_records": len(sft_validation),
        "grpo_train_tasks": len(grpo_train),
        "preference_pairs": len(preference_pairs),
        "outputs": {
            "sft_train": str(out / "sft_train.jsonl"),
            "sft_validation": str(out / "sft_validation.jsonl"),
            "grpo_train": str(out / "grpo_train.jsonl"),
            "preference_pairs": str(out / "preference_pairs.jsonl"),
        },
        "note": "No tests were executed. Run GRPO only in an isolated sandbox with approved training repositories.",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
