#!/usr/bin/env python3
"""Collect blind, short-form repair candidates from an OpenAI-compatible teacher API.

This is data collection only: it never applies patches or executes repository code.
The response's public content field is stored; provider reasoning/private fields are ignored.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import tarfile
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any


TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")
STOPWORDS = {
    "about", "after", "again", "also", "because", "before", "being", "between", "could",
    "does", "during", "each", "from", "have", "into", "issue", "just", "make", "more",
    "most", "other", "problem", "should", "some", "such", "than", "that", "their", "them",
    "there", "these", "they", "this", "those", "through", "under", "using", "very", "what",
    "when", "where", "which", "while", "with", "would", "your", "return", "function", "value",
}

SYSTEM_PROMPT = """You are a senior software engineer producing a candidate repair for supervised training.
Use only the issue and code excerpts supplied. Do not claim to have run tests or inspected files not shown.
Return exactly one JSON object with keys: summary, explicit_rationale, patch, confidence, missing_context.
summary: one-sentence diagnosis.
explicit_rationale: a clear, user-facing step-by-step explanation of the evidence, diagnosis,
and chosen repair. Make each step concrete and grounded in supplied context. This is an
explicit rationale requested for training, not private hidden chain-of-thought or token-level
internal reasoning. Never claim it is a verbatim dump of hidden reasoning.
patch: a minimal unified git diff, or an empty string if the supplied context is insufficient.
confidence: a number from 0 to 1. missing_context: a short string or empty string.
Do not include tests, gold answers, or invented tool observations."""

RATIONALE_INSTRUCTIONS = {
    "none": "explicit_rationale: empty string.",
    "brief": "explicit_rationale: 2-4 concise numbered steps, prioritizing evidence and decision.",
    "detailed": "explicit_rationale: 5-8 numbered steps covering evidence, localization, root cause, minimal change, and a verification plan (do not say tests ran).",
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_tasks(path: Path) -> tuple[list[dict[str, Any]], str]:
    raw = path.read_bytes()
    tasks = []
    seen: set[str] = set()
    for line_no, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        task = json.loads(line)
        task_id = task.get("instance_id")
        required = (task_id, task.get("repo"), task.get("base_commit"), task.get("problem_statement"))
        if not all(isinstance(item, str) and item.strip() for item in required):
            raise ValueError(f"Task row {line_no} lacks required public fields")
        if task_id in seen:
            raise ValueError(f"Duplicate task id: {task_id}")
        seen.add(task_id)
        tasks.append(task)
    if not tasks:
        raise ValueError("No tasks found")
    return tasks, sha256_bytes(raw)


def graph_file(graph_root: Path | None, task: dict[str, Any], allow_repo_fallback: bool = False) -> Path | None:
    if not graph_root:
        return None
    task_id = task["instance_id"]
    repo_short = task["repo"].rstrip("/").split("/")[-1]
    candidates = (
        graph_root / f"{task_id}.json",
        graph_root / f"{repo_short}_{task['base_commit']}.json",
    )
    exact = next((path for path in candidates if path.is_file()), None)
    if exact or not allow_repo_fallback:
        return exact
    repo_graphs = sorted(graph_root.glob(f"{repo_short}_*.json"))
    return repo_graphs[0] if len(repo_graphs) == 1 else None


def relevant_graph_context(path: Path | None, issue: str, max_nodes: int, max_chars: int) -> tuple[str, str | None]:
    if path is None:
        return "", None
    data = json.loads(path.read_text(encoding="utf-8"))
    nodes = data.get("nodes", [])
    if not isinstance(nodes, list):
        return "", sha256_bytes(path.read_bytes())

    issue_terms = {term.lower() for term in TOKEN_RE.findall(issue) if term.lower() not in STOPWORDS}
    if not issue_terms:
        return "", sha256_bytes(path.read_bytes())
    doc_freq = Counter()
    node_tokens = []
    for node in nodes:
        searchable = f"{node.get('name', '')} {node.get('id', '')} {node.get('text', '')}".lower()
        terms = {term.lower() for term in TOKEN_RE.findall(searchable)}
        node_tokens.append(terms)
        doc_freq.update(issue_terms & terms)
    n_docs = max(1, len(nodes))
    scored = []
    for index, node in enumerate(nodes):
        overlap = issue_terms & node_tokens[index]
        if not overlap:
            continue
        name_terms = {term.lower() for term in TOKEN_RE.findall(f"{node.get('name', '')} {node.get('id', '')}")}
        score = sum(math.log(1 + n_docs / (1 + doc_freq[term])) for term in overlap)
        score += 2.0 * len(overlap & name_terms)
        text = node.get("text") if isinstance(node.get("text"), str) else ""
        if text.strip():
            scored.append((score, str(node.get("id", node.get("name", index))), text))
    scored.sort(key=lambda item: (-item[0], item[1]))
    blocks = []
    used = 0
    for _, name, source in scored[:max_nodes]:
        block = f"### {name}\n```python\n{source[:5000]}\n```"
        if used + len(block) > max_chars:
            break
        blocks.append(block)
        used += len(block)
    return "\n\n".join(blocks), sha256_bytes(path.read_bytes())


def relevant_snapshot_context(path: Path, issue: str, max_files: int, max_chars: int) -> tuple[str, str]:
    """Read bounded source excerpts from an exact task archive without extracting it."""
    archive_hash = sha256_file(path)
    issue_terms = {term.lower() for term in TOKEN_RE.findall(issue) if term.lower() not in STOPWORDS}
    if not issue_terms:
        return "", archive_hash
    extensions = {".py", ".pyi", ".js", ".jsx", ".ts", ".tsx", ".go", ".rs", ".java", ".c", ".h", ".cpp", ".hpp", ".rb", ".php"}
    excluded = {".git", ".venv", "venv", "node_modules", "vendor", "dist", "build", "__pycache__"}
    scored = []
    try:
        with tarfile.open(path, "r:gz") as archive:
            for member in archive:
                if not member.isfile() or member.size <= 0 or member.size > 1_000_000:
                    continue
                rel = PurePosixPath(member.name)
                if rel.is_absolute() or any(part in {"", ".", ".."} for part in rel.parts):
                    continue
                if any(part.lower() in excluded for part in rel.parts) or rel.suffix.lower() not in extensions:
                    continue
                if any(part.lower() in {"test", "tests", "__tests__"} or part.lower().startswith("test_") or part.lower().endswith("_test.py") for part in rel.parts):
                    continue
                handle = archive.extractfile(member)
                if handle is None:
                    continue
                try:
                    source = handle.read(1_000_001).decode("utf-8", errors="replace")
                finally:
                    handle.close()
                term_counts = Counter(term.lower() for term in TOKEN_RE.findall(source))
                path_terms = {term.lower() for term in TOKEN_RE.findall(rel.as_posix())}
                term_counts.update({term: 3 for term in issue_terms & path_terms})
                overlap = issue_terms & set(term_counts)
                if not overlap:
                    continue
                score = sum(min(term_counts[term], 12) for term in overlap)
                score += 4.0 * len(overlap & path_terms)
                scored.append((score, rel.as_posix(), member))
    except (tarfile.TarError, OSError) as exc:
        raise ValueError(f"Cannot read task snapshot {path.name}: {exc}") from exc
    scored.sort(key=lambda item: (-item[0], item[1]))
    blocks, used = [], 0
    with tarfile.open(path, "r:gz") as archive:
        for _, name, member in scored[:max_files]:
            handle = archive.extractfile(member)
            if handle is None:
                continue
            try:
                source = handle.read(1_000_001).decode("utf-8", errors="replace")
            finally:
                handle.close()
            source_lines = source.splitlines()
            if len(source) <= 6000:
                excerpt = source
            else:
                line_scores = [len(issue_terms & {term.lower() for term in TOKEN_RE.findall(line)}) for line in source_lines]
                ranked_lines = sorted(range(len(line_scores)), key=lambda index: (-line_scores[index], index))
                windows = []
                for index in ranked_lines:
                    if line_scores[index] == 0:
                        break
                    start, end = max(0, index - 4), min(len(source_lines), index + 5)
                    if any(start <= right and left <= end for left, right in windows):
                        continue
                    windows.append((start, end))
                    preview = "\n...\n".join(
                        f"# lines {left + 1}-{right}\n" + "\n".join(source_lines[left:right])
                        for left, right in sorted(windows)
                    )
                    if len(preview) >= 5000:
                        break
                excerpt = preview[:6000]
            language = PurePosixPath(name).suffix.lstrip(".") or "text"
            block = f"### {name}\n```{language}\n{excerpt}\n```"
            if used + len(block) > max_chars:
                break
            blocks.append(block)
            used += len(block)
    return "\n\n".join(blocks), archive_hash


def make_messages(task: dict[str, Any], context: str, rationale_mode: str,
                  graph_revision: str | None = None,
                  context_source: str | None = None) -> list[dict[str, str]]:
    issue = task["problem_statement"]
    hints = task.get("hints_text")
    content = (
        f"Repository: {task['repo']}\nBase commit: {task['base_commit']}\n\n"
        f"Issue:\n{issue}"
    )
    if isinstance(hints, str) and hints.strip():
        content += f"\n\nMaintainer hints:\n{hints}"
    if context_source == "task_snapshot":
        content += f"\n\nContext source: exact task snapshot for {task['instance_id']} at base commit {task['base_commit']}.\n"
    elif graph_revision:
        content += f"\n\nIndexed graph snapshot: {graph_revision}.\n"
        if graph_revision != f"{task['repo'].rstrip('/').split('/')[-1]}_{task['base_commit']}":
            content += "WARNING: indexed graph revision differs from the task base commit; treat excerpts only as location hints and do not assume code is identical.\n"
    content += "\nCandidate source excerpts (graph retrieval; verify relevance and do not assume complete context):\n"
    content += context if context else "[No indexed source excerpts available.]"
    system_prompt = SYSTEM_PROMPT.replace(
        "explicit rationale requested for training, not private hidden chain-of-thought or token-level\ninternal reasoning. Never claim it is a verbatim dump of hidden reasoning.",
        "explicit rationale requested for training, not private hidden chain-of-thought or token-level\ninternal reasoning. Never claim it is a verbatim dump of hidden reasoning.\n" + RATIONALE_INSTRUCTIONS[rationale_mode],
    )
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": content}]


def api_chat(base_url: str, api_key: str, model: str, messages: list[dict[str, str]], temperature: float,
             max_tokens: int, timeout: int) -> dict[str, Any]:
    url = base_url.rstrip("/") + "/chat/completions"
    body = json.dumps({
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }).encode("utf-8")
    request = urllib.request.Request(url, data=body, headers={
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def parse_candidate(content: str) -> dict[str, Any]:
    try:
        start, end = content.index("{"), content.rindex("}") + 1
        value = json.loads(content[start:end])
        if not isinstance(value, dict):
            raise ValueError("response JSON is not an object")
        return value
    except (ValueError, json.JSONDecodeError):
        patch = ""
        match = re.search(r"(?ms)^diff --git .*?(?=\Z)", content)
        if match:
            patch = match.group(0).strip()
        return {"summary": "", "patch": patch, "confidence": None, "missing_context": "response was not valid JSON"}


def sanitize_explicit_rationale(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    # Some compatible endpoints place private reasoning markers inside content.
    value = re.sub(r"(?is)<(think|analysis|reasoning)>.*?</\1\s*>", "", value)
    value = re.sub(r"(?i)</?(think|analysis|reasoning)>", "", value)
    return value.strip()


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--graphs", type=Path)
    parser.add_argument("--snapshots", type=Path, help="Optional directory containing exact <instance_id>.tgz task snapshots")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=os.environ.get("TEACHER_MODEL", ""))
    parser.add_argument("--base-url", default=os.environ.get("TEACHER_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--api-key-env", default="TEACHER_API_KEY")
    parser.add_argument("--task-id", action="append", default=[])
    parser.add_argument("--limit", type=int, default=4, help="Default smoke limit; pass 129 explicitly for all tasks")
    parser.add_argument("--samples-per-task", type=int, default=1)
    parser.add_argument("--max-nodes", type=int, default=10)
    parser.add_argument("--max-files", type=int, default=10)
    parser.add_argument("--max-context-chars", type=int, default=24000)
    parser.add_argument("--allow-repo-graph-fallback", action="store_true",
                        help="Use the sole graph for a repository when task commit has no exact index; marked as stale context")
    parser.add_argument("--max-tokens", type=int, default=2500)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--rationale-mode", choices=tuple(RATIONALE_INSTRUCTIONS), default="detailed",
                        help="Collect an explicit user-facing rationale; never provider-private reasoning")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--retry-5xx", type=int, default=2)
    parser.add_argument("--delay-seconds", type=float, default=0.5)
    parser.add_argument("--execute", action="store_true", help="Without this flag, only writes blind prompts as a dry run")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.limit < 1 or args.samples_per_task < 1 or args.max_nodes < 1 or args.max_files < 1 or args.max_context_chars < 1:
        raise ValueError("limit, samples-per-task, max-nodes, max-files, and max-context-chars must be positive")
    if not 0 <= args.temperature <= 2 or args.max_tokens < 1 or args.timeout < 1:
        raise ValueError("invalid temperature/max-tokens/timeout")

    tasks, dataset_hash = read_tasks(args.tasks)
    if args.task_id:
        wanted = set(args.task_id)
        tasks = [task for task in tasks if task["instance_id"] in wanted]
        missing = wanted - {task["instance_id"] for task in tasks}
        if missing:
            raise ValueError(f"Unknown task ids: {sorted(missing)}")
    tasks = tasks[:args.limit]
    if not tasks:
        raise ValueError("No tasks selected")

    api_key = os.environ.get(args.api_key_env, "")
    if args.execute and (not args.model or not api_key):
        raise ValueError(f"Set --model/TEACHER_MODEL and secret env var {args.api_key_env} before --execute")
    if args.execute and not (args.base_url.startswith("https://") or "localhost" in args.base_url or "127.0.0.1" in args.base_url):
        raise ValueError("Teacher endpoint must use HTTPS unless it is a localhost endpoint")

    completed: set[tuple[str, int, str]] = set()
    if args.resume and args.output.is_file():
        with args.output.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                old = json.loads(line)
                if (old.get("status") == "collected" and old.get("dataset_sha256") == dataset_hash
                        and old.get("teacher_model") == args.model):
                    completed.add((old.get("task_id", ""), old.get("sample_index", -1), old.get("prompt_sha256", "")))

    calls, skipped = 0, 0
    for task in tasks:
        snapshot_path = args.snapshots / f"{task['instance_id']}.tgz" if args.snapshots else None
        if snapshot_path is not None and snapshot_path.is_file():
            context, snapshot_hash = relevant_snapshot_context(
                snapshot_path, task["problem_statement"], args.max_files, args.max_context_chars
            )
            graph_path, graph_hash, context_source = None, None, "task_snapshot"
        else:
            graph_path = graph_file(args.graphs, task, args.allow_repo_graph_fallback)
            context, graph_hash = relevant_graph_context(
                graph_path, task["problem_statement"], args.max_nodes, args.max_context_chars
            )
            snapshot_hash, context_source = None, "graph" if graph_path else "none"
        graph_revision = None
        if graph_path:
            graph_data = json.loads(graph_path.read_text(encoding="utf-8"))
            graph_revision = graph_data.get("graph", {}).get("repo_name", graph_path.stem)
        messages = make_messages(task, context, args.rationale_mode, graph_revision, context_source)
        prompt_hash = sha256_bytes(json.dumps(messages, sort_keys=True, ensure_ascii=False).encode("utf-8"))
        for sample_index in range(args.samples_per_task):
            if (task["instance_id"], sample_index, prompt_hash) in completed:
                skipped += 1
                continue
            base_row = {
                "task_id": task["instance_id"],
                "repo": task["repo"],
                "base_commit": task["base_commit"],
                "snapshot": f"snapshots/{task['instance_id']}.tgz",
                "dataset_sha256": dataset_hash,
                "graph_sha256": graph_hash,
                "snapshot_sha256": snapshot_hash,
                "context_source": context_source,
                "graph_revision": graph_revision,
                "graph_matches_task_commit": graph_revision == f"{task['repo'].rstrip('/').split('/')[-1]}_{task['base_commit']}" if graph_revision else None,
                "prompt_sha256": prompt_hash,
                "teacher_model": args.model or None,
                "teacher_endpoint": args.base_url,
                "rationale_mode": args.rationale_mode,
                "sample_index": sample_index,
                "blind_to_reference": True,
                "reference_patch_sent": False,
                "test_patch_sent": False,
                "rights_review": "pending",
            }
            if not args.execute:
                append_jsonl(args.output, {**base_row, "status": "dry_run", "messages": messages})
                calls += 1
                continue

            response = None
            for attempt in range(args.retry_5xx + 1):
                try:
                    response = api_chat(args.base_url, api_key, args.model, messages,
                                        args.temperature, args.max_tokens, args.timeout)
                    break
                except urllib.error.HTTPError as exc:
                    if exc.code < 500 or attempt >= args.retry_5xx:
                        raise RuntimeError(f"Teacher HTTP error {exc.code} (response body omitted)") from None
                    time.sleep(min(2 ** attempt, 8))
                except (TimeoutError, urllib.error.URLError) as exc:
                    raise RuntimeError(f"Teacher request failed ({type(exc).__name__}); not auto-retried to avoid duplicate billing") from None
            choice = (response or {}).get("choices", [{}])[0]
            message = choice.get("message", {})
            content = message.get("content") if isinstance(message, dict) else None
            if not isinstance(content, str):
                content = ""
            candidate = parse_candidate(content)
            usage = response.get("usage", {}) if isinstance(response, dict) else {}
            append_jsonl(args.output, {
                **base_row,
                "status": "collected",
                "summary": candidate.get("summary", ""),
                "explicit_rationale": sanitize_explicit_rationale(candidate.get("explicit_rationale", "")),
                "candidate_patch": candidate.get("patch", ""),
                "confidence": candidate.get("confidence"),
                "missing_context": candidate.get("missing_context", ""),
                "teacher_response_sha256": sha256_bytes(content.encode("utf-8")),
                "finish_reason": choice.get("finish_reason"),
                "usage": usage,
            })
            calls += 1
            if args.delay_seconds > 0:
                time.sleep(args.delay_seconds)

    print(json.dumps({
        "selected_tasks": len(tasks),
        "samples_per_task": args.samples_per_task,
        "records_written": calls,
        "records_skipped_by_resume": skipped,
        "api_calls": calls if args.execute else 0,
        "dry_run": not args.execute,
        "output": str(args.output),
        "dataset_sha256": dataset_hash,
        "warning": "Candidates are unverified and rights_review=pending; do not train from this file directly.",
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
