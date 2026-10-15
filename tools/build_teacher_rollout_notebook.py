#!/usr/bin/env python3
"""Derive a teacher-rollout notebook from the current Kaggle harness notebook."""

from __future__ import annotations

import argparse
import ast
import json
import pprint
import re
from pathlib import Path


def source(cell: dict) -> str:
    return "".join(cell.get("source", []))


def set_source(cell: dict, value: str) -> None:
    cell["source"] = value.splitlines(keepends=True)


def literal_assignment(text: str, name: str):
    tree = ast.parse(text)
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            return node, ast.literal_eval(node.value)
    raise ValueError(f"Missing literal assignment: {name}")


def replace_assignment(text: str, name: str, value) -> str:
    node, _ = literal_assignment(text, name)
    lines = text.splitlines(keepends=True)
    replacement = f"{name} = {pprint.pformat(value, width=100, sort_dicts=False)}\n"
    return "".join(lines[: node.lineno - 1]) + replacement + "".join(lines[node.end_lineno :])


def code_cell(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


def markdown_cell(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def indented(text: str) -> str:
    return "".join("    " + line if line.strip() else line for line in text.splitlines(keepends=True))


def build(source_path: Path, output_path: Path) -> None:
    notebook = json.loads(source_path.read_text(encoding="utf-8"))
    cells = notebook["cells"]
    if len(cells) < 15:
        raise ValueError("Source notebook does not match the expected current-harness layout")

    set_source(cells[0], """# Gemma SWE — Teacher Rollout with the Competition Harness

This notebook runs the current competition harness against public development tasks, but routes the declared Gemma model alias to a user-configured LiteLLM-compatible teacher endpoint. The agent prompt, tools, tool schemas, sandbox, graph/embedding integration, patch extraction, and verification remain the current harness. Only the model backend changes.

The default cohort is the four local pilot tasks. Set `SELECTED_TASK_IDS = None` to run all 129 public tasks. External rollout is opt-in and disabled by default. Kaggle Secrets must contain `TEACHER_API_KEY`; set `TEACHER_MODEL_SPEC` and optionally `TEACHER_API_BASE` in the model cell. Only send public task inputs to a provider whose terms permit this use. Hidden chain-of-thought is not requested or exported; visible assistant/tool messages and verification artifacts are retained.
""")

    # The source notebook embeds a complete declarative harness. Remove student LoRA
    # files so the teacher runs the same tools/prompts without inheriting the student's adapter.
    cell4 = source(cells[4])
    _, text_assets = literal_assignment(cell4, "TEXT_ASSETS")
    clean_assets = {key: value for key, value in text_assets.items() if not key.startswith("adapters/")}
    for key in ("agent.yaml", "sub_agents/code_analyzer.yaml"):
        if key in clean_assets:
            clean_assets[key] = re.sub(r"(?m)^adapter:\s*[^\n]+\n", "", clean_assets[key])
    cell4 = replace_assignment(cell4, "TEXT_ASSETS", clean_assets)
    cell4 = replace_assignment(cell4, "BINARY_ASSETS_B64", {})
    cell4 = cell4.replace('RUN_LABEL = "bounded_exploration_6min_129"', 'RUN_LABEL = "teacher_rollout_same_harness_v1"')
    set_source(cells[4], cell4)

    set_source(cells[9], "## 5. Run the Selected Public Teacher Tasks\n")
    set_source(cells[10], """## 6. Rollout Contract and Artifacts

- Default cohort: `requests_7433`, `httpx_3672`, `fastapi_15745`, and `rich_3064`. Set `SELECTED_TASK_IDS = None` to select all public tasks in dataset order.
- Per-task limits and prompt are inherited from the current six-minute harness configuration; the notebook does not silently expand the student's budget.
- Each task runs in the existing isolated harness sandbox. The runner records real tool calls/results, patch output, test result, duration, and usage where available, with a checkpoint after each completed task.
- The output zip contains sanitized visible trajectories and evaluation artifacts. It omits private reasoning fields and does not include the competition's hidden evaluation set.
- This is a data-collection notebook, not a competition submission notebook. It does not build `submission.zip`.
""")

    # Remove the old model-serving cell and route the fixed declared alias to LiteLLM.
    set_source(cells[8], """import os
from adk_submission import discover_declared_models
from google.adk.models.lite_llm import LiteLlm

TARGET_MODEL_NAME = 'gemma-4-31b-it-qat-w4a16-ct'
declared_models = sorted(discover_declared_models(AGENT_DIR))
if declared_models != [TARGET_MODEL_NAME]:
    raise ValueError(f'Expected the current harness model alias {TARGET_MODEL_NAME}; found {declared_models}')
declared_model = declared_models[0]

# Example: openai/gpt-... or anthropic/claude-...; use a provider/model string supported by LiteLLM.
TEACHER_MODEL_SPEC = os.environ.get('TEACHER_MODEL_SPEC', '').strip()
TEACHER_API_BASE = os.environ.get('TEACHER_API_BASE', '').strip() or None
TEACHER_API_KEY = os.environ.get('TEACHER_API_KEY', '').strip()
if not TEACHER_API_KEY:
    try:
        from kaggle_secrets import UserSecretsClient
        TEACHER_API_KEY = UserSecretsClient().get_secret('TEACHER_API_KEY').strip()
    except Exception:
        TEACHER_API_KEY = ''

COLLECT_TEACHER_ROLLOUTS = False  # Explicitly enable after reviewing provider, cost, and data-sharing terms.
TEACHER_READY = bool(TEACHER_MODEL_SPEC and TEACHER_API_KEY)
models = {}
if TEACHER_READY:
    teacher_llm = LiteLlm(
        model=TEACHER_MODEL_SPEC,
        api_key=TEACHER_API_KEY,
        api_base=TEACHER_API_BASE,
        num_retries=2,
    )
    # Keep the benchmark's declared model alias, tool descriptions, and agent tree unchanged.
    models = {declared_model: teacher_llm}
    print(f'Teacher configured: {TEACHER_MODEL_SPEC}; API key is loaded but never displayed.')
else:
    print('Dry-run setup only. Add TEACHER_API_KEY as a Kaggle Secret and set TEACHER_MODEL_SPEC to enable collection.')

# No adapters are loaded for the teacher. Agent YAML still declares the allowed Gemma alias;
# the evaluator's model registry routes that alias to the teacher endpoint above.
adapters = {}
""")

    original_eval = source(cells[12])
    original_eval = original_eval.replace(
        "# Fixed cohort: first N tasks in the published dataset order.\nEVAL_TASK_LIMIT = 129\nSAMPLE_TASKS = tasks[:EVAL_TASK_LIMIT]",
        """# Select four pilot tasks by default; set to None for the full public 129-task cohort.
SELECTED_TASK_IDS = ['requests_7433', 'httpx_3672', 'fastapi_15745', 'rich_3064']
if SELECTED_TASK_IDS is None:
    SAMPLE_TASKS = list(tasks)
else:
    task_by_id = {task.instance_id: task for task in tasks}
    missing_ids = [task_id for task_id in SELECTED_TASK_IDS if task_id not in task_by_id]
    if missing_ids:
        raise ValueError(f'Selected task ids are absent from the attached public dataset: {missing_ids}')
    SAMPLE_TASKS = [task_by_id[task_id] for task_id in SELECTED_TASK_IDS]
EVAL_TASK_LIMIT = len(SAMPLE_TASKS)""",
    )
    original_eval = original_eval.replace("'model_input': str(TRAIN_MODEL_PATH)", "'teacher_model': TEACHER_MODEL_SPEC")
    original_eval = original_eval.replace("'selection': 'first 129 tasks in published dataset order'", "'selection': [task.instance_id for task in SAMPLE_TASKS]")
    set_source(
        cells[12],
        "if COLLECT_TEACHER_ROLLOUTS and not TEACHER_READY:\n"
        "    raise RuntimeError('Set TEACHER_MODEL_SPEC and add TEACHER_API_KEY to Kaggle Secrets before enabling collection.')\n"
        "if not COLLECT_TEACHER_ROLLOUTS:\n"
        "    print('Teacher calls are disabled. Review the configuration above, then set COLLECT_TEACHER_ROLLOUTS = True and rerun this cell.')\n"
        "else:\n" + indented(original_eval),
    )

    export_source = source(cells[13])
    set_source(
        cells[13],
        "if COLLECT_TEACHER_ROLLOUTS:\n" + indented(export_source) +
        "else:\n    print('No rollout artifacts exported because collection is disabled.')\n",
    )

    # Replace submission-zip instructions with an explicit data-bundle handoff.
    set_source(cells[15], "## 7. Download the Teacher Rollout Bundle\n")
    set_source(cells[16], """from IPython.display import display, FileLink

teacher_bundle = WORKING_DIR / f'{RUN_LABEL}_observable_logs.zip'
if teacher_bundle.is_file():
    print(f'Teacher rollout bundle ready: {teacher_bundle} ({teacher_bundle.stat().st_size / (1024 * 1024):.2f} MiB)')
    display(FileLink(str(teacher_bundle)))
else:
    print('No bundle yet. Enable collection and rerun the selected tasks first.')
""")

    # The generated source notebook may contain stale outputs from prior evaluations.
    # Keep the teacher notebook clean and remove the duplicate/outdated contract cell.
    cells.pop(14)  # The source notebook has a duplicated observable-export cell.
    cells.pop(11)  # Drop its stale duplicate contract cell.
    for cell in cells:
        if cell.get("cell_type") == "code":
            cell["execution_count"] = None
            cell["outputs"] = []

    kaggle = notebook.setdefault("metadata", {}).setdefault("kaggle", {})
    kaggle["accelerator"] = "none"
    kaggle["isGpuEnabled"] = False
    kaggle["isInternetEnabled"] = True  # Required only for the explicitly enabled teacher API calls.
    kaggle["dataSources"] = [
        item for item in kaggle.get("dataSources", [])
        if item.get("sourceType") != "modelInstanceVersion"
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"Wrote {output_path} ({len(cells)} cells)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Current six-minute observable harness notebook")
    parser.add_argument("--output", type=Path, default=Path("notebooks/teacher-rollout-same-harness.ipynb"))
    args = parser.parse_args()
    build(args.source, args.output)


if __name__ == "__main__":
    main()
