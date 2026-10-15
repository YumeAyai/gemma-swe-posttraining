import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "teacher-rollout-same-harness.ipynb"


def test_teacher_notebook_is_cpu_only_and_all_cells_parse():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    kaggle = notebook["metadata"]["kaggle"]
    assert kaggle["accelerator"] == "none"
    assert kaggle["isGpuEnabled"] is False
    assert kaggle["isInternetEnabled"] is True
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            assert cell["outputs"] == []
            assert cell["execution_count"] is None
            ast.parse("".join(cell["source"]), filename=f"cell-{index}")


def test_teacher_reuses_harness_and_external_collection_is_opt_in():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    sources = ["".join(cell.get("source", [])) for cell in notebook["cells"]]
    all_source = "\n".join(sources)
    model_cell = next(source for source in sources if "from google.adk.models.lite_llm import LiteLlm" in source)
    evaluation_cell = next(source for source in sources if "task_by_id = {task.instance_id" in source)

    assert "COLLECT_TEACHER_ROLLOUTS = False" in model_cell
    assert "models = {declared_model: teacher_llm}" in model_cell
    assert "Evaluator(eval_config)" in all_source
    assert "run_command" in all_source and "submit_patch" in all_source
    assert all(task_id in evaluation_cell for task_id in (
        "requests_7433", "httpx_3672", "fastapi_15745", "rich_3064"
    ))
    assert "if SELECTED_TASK_IDS is None" in evaluation_cell
    assert "adapter_model.safetensors" not in all_source
    assert "remove_private_reasoning" in all_source
    assert "shutil.make_archive" not in all_source
