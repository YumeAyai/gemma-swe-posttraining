import ast
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "train-gemma-sft-grpo-kaggle.ipynb"


class TrainingNotebookCase(unittest.TestCase):
    def test_kaggle_metadata_remains_cpu_neutral_and_cells_parse(self):
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        kaggle = notebook["metadata"]["kaggle"]
        self.assertEqual(kaggle["accelerator"], "none")
        self.assertIs(kaggle["isGpuEnabled"], False)
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                ast.parse("".join(cell["source"]), filename=f"cell-{index}")

    def test_contains_opt_in_sft_and_grpo_with_visible_rationale_path(self):
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
        self.assertIn("RUN_SFT = False", source)
        self.assertIn("RUN_GRPO = False", source)
        self.assertIn("explicit_rationale", source)
        self.assertIn("TRUST_REPOSITORY_TESTS = False", source)


if __name__ == "__main__":
    unittest.main()
