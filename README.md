# Gemma SWE Post-Training Toolkit

An auditable workflow for collecting authorized software-repair candidates, validating provenance and patches, exporting leakage-safe SFT/GRPO manifests, and running an opt-in Kaggle training notebook.

This repository contains tooling and schemas only. It does not include competition task data, repository snapshots, model weights, adapters, API credentials, or teacher outputs. Review provider terms, repository licenses, and competition rules before collecting or training on any data.

## Components

- `training/collector/collector.py`: dry-run-first collector for explicit, user-facing teacher rationales and candidate patches. It does not execute tools or repository code.
- `notebooks/teacher-rollout-same-harness.ipynb`: opt-in pilot notebook that routes the current benchmark model alias to a teacher API while retaining the competition harness, tools, isolated task sandbox, patch extraction, and test verification.
- `tools/export_public_teacher_prompts.py`: exports blind prompts without reference patches or test patches.
- `tools/prepare_posttrain_data.py`: validates canonical records, applies rights/evaluation gates, creates splits, and exports SFT, preference, and GRPO manifests. It does not execute tests.
- `tools/build_repair_sft.py`: legacy/simple builder for verified repair rollouts.
- `notebooks/train-gemma-sft-grpo-kaggle.ipynb`: opt-in SFT/GRPO notebook. Kaggle metadata is CPU-only (`accelerator: none`, `isGpuEnabled: false`); choose hardware manually in Kaggle UI when needed.
- `training/record.schema.json`: canonical record schema.

Start with [the workflow and safety guide](training/README.md), including provider/data-sharing cautions for teacher rollouts.

## Local checks

```bash
uv run --with pytest pytest -q
```

The tests use temporary fixtures and do not call a teacher API or execute task repositories.
