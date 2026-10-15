#!/usr/bin/env python3
"""Build the Kaggle SFT→GRPO notebook with CPU-only Kaggle metadata."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "train-gemma-sft-grpo-kaggle.ipynb"


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


cells = [
    markdown("""# Gemma 4 SWE：Teacher 数据 → LoRA SFT → GRPO

这本 notebook 是训练端，不调用外部教师 API，也不负责收集数据。先在外置 `uv` collector 采集显式 rationale/patch 候选，再完成来源审核、补丁重放与测试验证，使用 `prepare_posttrain_data.py` 导出合格的 `sft_train.jsonl`、`sft_validation.jsonl` 和 `grpo_train.jsonl`，最后把这些文件及可信仓库快照作为 Kaggle 输入挂载。

**硬件设置：**Notebook JSON 的 Kaggle `accelerator` 固定为 `none`、`isGpuEnabled=false`。如需实际训练，请在 Kaggle UI 手动选择 L4x4 并重启 session；不要改 notebook 中的 GPU 声明。无 GPU 时可执行数据检查，但不能训练 31B。

训练分两阶段，默认均关闭：SFT 学习已经验证的可观察修复过程；GRPO 只对训练 split 生成多个补丁，用受限测试奖励继续优化。先运行小样本，不要将未验证 teacher 候选直接训练。"""),
    markdown("## 1. 配置输入、模型与实验开关\n\n将训练 manifest 和快照数据集附加到 Kaggle notebook 后，修改以下路径。`MODEL_PATH` 指向比赛允许的 Gemma 4 31B QAT base。SFT/GRPO 参数是可复现起点，不代表已经在 Kaggle GPU 上验证过。"""),
    code("""import importlib.util
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path, PurePosixPath

INPUT_ROOT = Path('/kaggle/input')
WORK_DIR = Path('/kaggle/working/gemma_sft_grpo')
WORK_DIR.mkdir(parents=True, exist_ok=True)

MODEL_ID = 'google/gemma-4-31b-it-qat-w4a16-ct'
MODEL_PATH = Path('/kaggle/input/models/google/gemma-4/other/gemma-4-31b-it-qat-w4a16-ct/2')
SFT_TRAIN_PATH = Path('/kaggle/input/YOUR_POSTTRAIN_DATA/sft_train.jsonl')
SFT_VALIDATION_PATH = Path('/kaggle/input/YOUR_POSTTRAIN_DATA/sft_validation.jsonl')
GRPO_TRAIN_PATH = Path('/kaggle/input/YOUR_POSTTRAIN_DATA/grpo_train.jsonl')
SNAPSHOT_ROOT = Path('/kaggle/input/YOUR_VERIFIED_SNAPSHOTS')
WHEELHOUSE = Path('/kaggle/input/YOUR_TRAINING_WHEELHOUSE')

RUN_SFT = False
RUN_GRPO = False
SFT_MAX_EXAMPLES = 0       # 0 = all approved train rows
GRPO_MAX_TASKS = 8         # start small; expand after reward smoke test
TRUST_REPOSITORY_TESTS = False  # explicit opt-in: tests execute code from snapshots
SEED = 3407

SFT_OUTPUT = WORK_DIR / 'sft_adapter'
GRPO_OUTPUT = WORK_DIR / 'grpo_adapter'
ADAPTER_FOR_GRPO = SFT_OUTPUT
TRAINING_LOG = WORK_DIR / 'training_log.json'

random.seed(SEED)
print('Python:', sys.version.split()[0])
print('Model present:', MODEL_PATH.exists(), MODEL_PATH)
print('Train inputs present:', SFT_TRAIN_PATH.is_file(), SFT_VALIDATION_PATH.is_file(), GRPO_TRAIN_PATH.is_file())
print('Output directory:', WORK_DIR)
"""),
    markdown("## 2. 环境与硬件预检\n\n当前 Kaggle 环境应连接好比赛模型与兼容的训练 wheelhouse。此处只从挂载 wheelhouse 离线安装，不访问 PyPI。环境包版本变化较快，先检查再决定是否安装。GPU 由 UI 手动选择，本 notebook 不设置 GPU 元数据。"""),
    code("""required = ('torch', 'transformers', 'peft', 'trl', 'datasets', 'accelerate')
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing and WHEELHOUSE.is_dir():
    wheels = sorted(str(path) for path in WHEELHOUSE.glob('*.whl'))
    if not wheels:
        raise RuntimeError(f'No wheel files found in {WHEELHOUSE}')
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps', *wheels], check=True)
    importlib.invalidate_caches()
    missing = [name for name in required if importlib.util.find_spec(name) is None]
print('Missing training packages:', missing)
if missing:
    print('Attach a version-compatible offline training wheelhouse; no internet install is attempted.')

if importlib.util.find_spec('torch'):
    import torch
    print('CUDA available:', torch.cuda.is_available())
    print('Visible GPU count:', torch.cuda.device_count())
    if torch.cuda.is_available():
        print('GPU names:', [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
        print('BF16 supported:', torch.cuda.is_bf16_supported())
else:
    torch = None
"""),
    markdown("## 3. 审核 SFT 与 GRPO 输入\n\nSFT JSONL 每行至少有 `messages`（包含 user 与 assistant）；assistant 可包含经审核的 `explicit_rationale` 和最终 patch。GRPO JSONL 每行需要 `prompt`、`repo_snapshot`（相对 `SNAPSHOT_ROOT` 的目录）和 `test_command`（参数数组，非 shell 字符串）。训练输入应来自 canonical 导出器，且只含 rights-approved、非评测任务。"""),
    code("""def read_jsonl(path):
    if not path.is_file():
        return []
    rows = []
    with path.open(encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f'{path}:{line_number} is not a JSON object')
            rows.append(row)
    return rows

def validate_sft_rows(rows, path):
    for index, row in enumerate(rows, 1):
        messages = row.get('messages')
        if not isinstance(messages, list) or not any(m.get('role') == 'user' for m in messages if isinstance(m, dict)):
            raise ValueError(f'{path}:{index}: messages must include a user message')
        if not any(m.get('role') == 'assistant' for m in messages if isinstance(m, dict)):
            raise ValueError(f'{path}:{index}: messages must include an assistant target')
        if row.get('evaluation_only') is True or row.get('benchmark_split') in {'test', 'hidden', 'evaluation'}:
            raise ValueError(f'{path}:{index}: evaluation/hidden row cannot be used for training')

def messages_for_sft(row):
    messages = [dict(message) for message in row['messages']]
    rationale = row.get('explicit_rationale', '')
    if isinstance(rationale, str) and rationale.strip() and not any(
        rationale.strip() in str(message.get('content', '')) for message in messages
        if message.get('role') == 'assistant'
    ):
        assistant = next((message for message in reversed(messages) if message.get('role') == 'assistant'), None)
        if assistant is not None:
            patch = row.get('patch', assistant.get('content', ''))
            assistant['content'] = f"Reasoning (explicit rationale):\\n{rationale.strip()}\\n\\nPatch:\\n{patch}"
    return messages

sft_train_rows = read_jsonl(SFT_TRAIN_PATH)
sft_validation_rows = read_jsonl(SFT_VALIDATION_PATH)
validate_sft_rows(sft_train_rows, SFT_TRAIN_PATH) if sft_train_rows else None
validate_sft_rows(sft_validation_rows, SFT_VALIDATION_PATH) if sft_validation_rows else None
grpo_rows = read_jsonl(GRPO_TRAIN_PATH)
print({'sft_train': len(sft_train_rows), 'sft_validation': len(sft_validation_rows), 'grpo': len(grpo_rows)})
if not sft_train_rows:
    print('SFT is not runnable until the approved training JSONL is attached and SFT_TRAIN_PATH is updated.')
"""),
    markdown("## 4. SFT：用已验证轨迹训练 LoRA\n\n默认关闭。建议第一轮 1 epoch、短上下文、rank 16、覆盖 attention 与 MLP 投影层；完成后检查可训练参数量、loss、adapter 文件与 holdout 表现。这里保留 rationale，是因为输入 messages 中含有经审核的显式理由；绝不把 provider 私有隐藏 reasoning 当成标签。"""),
    code("""def require_training_runtime():
    absent = [name for name in required if importlib.util.find_spec(name) is None]
    if absent:
        raise RuntimeError(f'Missing packages: {absent}; attach compatible offline wheels first.')
    if not MODEL_PATH.is_dir():
        raise FileNotFoundError(f'Attach the official Gemma 4 model input: {MODEL_PATH}')
    if torch is None or not torch.cuda.is_available():
        raise RuntimeError('No CUDA GPU visible. Select the intended accelerator in Kaggle UI, restart, and rerun.')

if RUN_SFT:
    require_training_runtime()
    if not sft_train_rows:
        raise ValueError('No approved SFT training rows loaded.')
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForMultimodalLM, AutoProcessor
    from trl import SFTConfig, SFTTrainer
    import inspect

    if SFT_MAX_EXAMPLES:
        sft_train_rows = sft_train_rows[:SFT_MAX_EXAMPLES]
    train_ds = Dataset.from_list([{'messages': messages_for_sft(row)} for row in sft_train_rows])
    eval_ds = Dataset.from_list([{'messages': messages_for_sft(row)} for row in sft_validation_rows]) if sft_validation_rows else None
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    processor = AutoProcessor.from_pretrained(str(MODEL_PATH), local_files_only=True)
    model = AutoModelForMultimodalLM.from_pretrained(
        str(MODEL_PATH), dtype=dtype, device_map='auto', low_cpu_mem_usage=True, local_files_only=True
    )
    model.config.use_cache = False
    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias='none',
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
        task_type='CAUSAL_LM', inference_mode=False,
    )
    train_values = {
        'output_dir': str(SFT_OUTPUT), 'num_train_epochs': 1,
        'per_device_train_batch_size': 1, 'gradient_accumulation_steps': 8,
        'gradient_checkpointing': True, 'gradient_checkpointing_kwargs': {'use_reentrant': False},
        'learning_rate': 5e-5, 'warmup_ratio': 0.03, 'logging_steps': 1,
        'save_strategy': 'epoch', 'save_total_limit': 2, 'report_to': 'none',
        'bf16': dtype == torch.bfloat16, 'fp16': dtype == torch.float16,
        'remove_unused_columns': False, 'max_length': 4096,
        'eval_strategy': 'epoch' if eval_ds is not None else 'no',
        'dataset_kwargs': {'skip_prepare_dataset': True}, 'seed': SEED,
    }
    config_sig = inspect.signature(SFTConfig.__init__)
    accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in config_sig.parameters.values())
    supported = train_values if accepts_kwargs else {k: v for k, v in train_values.items() if k in config_sig.parameters}
    sft_args = SFTConfig(**supported)
    trainer = SFTTrainer(
        model=model, args=sft_args, train_dataset=train_ds, eval_dataset=eval_ds,
        processing_class=processor, peft_config=lora,
    )
    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in trainer.model.parameters())
    print(f'trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.4f}%)')
    train_result = trainer.train()
    trainer.save_model(str(SFT_OUTPUT))
    (WORK_DIR / 'sft_metrics.json').write_text(json.dumps({
        'metrics': train_result.metrics, 'history': trainer.state.log_history,
        'train_rows': len(train_ds), 'validation_rows': len(eval_ds) if eval_ds else 0,
        'model_id': MODEL_ID, 'seed': SEED,
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    for filename in ('adapter_config.json', 'adapter_model.safetensors'):
        if not (SFT_OUTPUT / filename).is_file():
            raise RuntimeError(f'SFT output missing {filename}')
    print('SFT adapter saved:', SFT_OUTPUT)
    del trainer, model, processor, train_ds, eval_ds
    torch.cuda.empty_cache()
else:
    print('SFT skipped; set RUN_SFT=True after checking all inputs and GPU selection.')
"""),
    markdown("## 5. GRPO：从 SFT adapter 做补丁级在线奖励\n\nGRPO 不是把 gold patch 当作标准答案直接训练：每题采样多个 unified diff，经安全路径检查、patch apply 和指定测试，按 fail→pass 给予主奖励。由于测试会执行仓库代码，本阶段默认关闭，并需要同时把 `TRUST_REPOSITORY_TESTS=True`；只对来源可信、审过命令的训练快照运行。测试超时/基础测试本来通过/补丁非法分别记日志，不能误记为修复成功。"""),
    code("""def completion_to_text(completion):
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        return '\\n'.join(str(item.get('content', '')) for item in completion if isinstance(item, dict))
    if isinstance(completion, dict):
        return str(completion.get('content', ''))
    return str(completion)

def patch_paths_are_safe(patch):
    paths = []
    for line in patch.splitlines():
        if line.startswith('diff --git '):
            paths.extend(line[len('diff --git '):].split())
        elif line.startswith(('+++ ', '--- ')):
            paths.append(line[4:].split('\\t', 1)[0])
    changed = set()
    for value in paths:
        if value == '/dev/null':
            continue
        value = value[2:] if value.startswith(('a/', 'b/')) else value
        parsed = PurePosixPath(value)
        if parsed.is_absolute() or any(part in {'', '.', '..'} for part in parsed.parts):
            return False
        changed.add(parsed)
    for path in changed:
        parts = [part.lower() for part in path.parts]
        if any(part in {'test', 'tests', '__tests__'} or part.startswith('test_') or part.endswith('_test.py') for part in parts):
            return False
    return bool(changed)

def normalize_grpo_rows(rows):
    normalized = []
    root = SNAPSHOT_ROOT.resolve(strict=True)
    for index, row in enumerate(rows, 1):
        prompt = row.get('prompt')
        if isinstance(prompt, str):
            prompt = [{'role': 'user', 'content': prompt}]
        if not isinstance(prompt, list) or not prompt:
            raise ValueError(f'GRPO row {index}: prompt must be text or messages')
        rel = row.get('repo_snapshot')
        if not isinstance(rel, str):
            raise ValueError(f'GRPO row {index}: repo_snapshot must be a relative path')
        rel_path = PurePosixPath(rel)
        if rel_path.is_absolute() or any(part in {'', '.', '..'} for part in rel_path.parts):
            raise ValueError(f'GRPO row {index}: unsafe snapshot path')
        repo = (root / Path(*rel_path.parts)).resolve(strict=True)
        if not repo.is_dir() or not repo.is_relative_to(root):
            raise ValueError(f'GRPO row {index}: snapshot escapes SNAPSHOT_ROOT')
        command = row.get('test_command')
        if not isinstance(command, list) or not command or not all(isinstance(arg, str) for arg in command):
            raise ValueError(f'GRPO row {index}: test_command must be argv list')
        if Path(command[0]).name not in {'python', 'python3', 'pytest'}:
            raise ValueError(f'GRPO row {index}: executable not in allowlist')
        if not any(isinstance(item, dict) and item.get('role') == 'system' for item in prompt):
            prompt = [{'role': 'system', 'content': 'Solve the software issue. Return only a minimal unified git diff. Do not explain outside the diff, edit tests, or weaken verification.'}] + prompt
        normalized.append({'prompt': prompt, 'repo_path': str(repo), 'test_command': command,
                           'task_id': str(row.get('task_id', index))})
    return normalized

GRPO_REWARD_EVENTS = []
GRPO_TEST_TIMEOUT = 60

def isolated_test(repo_dir, command):
    env = {'PATH': os.environ.get('PATH', ''), 'HOME': str(repo_dir), 'PIP_NO_INDEX': '1',
           'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONNOUSERSITE': '1'}
    return subprocess.run(command, cwd=repo_dir, env=env, capture_output=True, text=True,
                          timeout=GRPO_TEST_TIMEOUT, shell=False, check=False)

def code_patch_reward(completions, repo_path=None, test_command=None, **kwargs):
    rewards = []
    repo_paths = repo_path if isinstance(repo_path, list) else [repo_path] * len(completions)
    commands = test_command if isinstance(test_command, list) and test_command and isinstance(test_command[0], list) else [test_command] * len(completions)
    for completion, source_path, command in zip(completions, repo_paths, commands):
        reward, detail = -0.2, 'invalid_or_unverified'
        try:
            source = Path(str(source_path)).resolve(strict=True)
            if not source.is_dir() or not source.is_relative_to(INPUT_ROOT.resolve()):
                raise ValueError('GRPO repositories must be mounted read-only under /kaggle/input')
            if any(p.is_symlink() for p in source.rglob('*')):
                raise ValueError('symlinks in source snapshots are not accepted')
            if not isinstance(command, list) or not command or Path(command[0]).name not in {'python', 'python3', 'pytest'}:
                raise ValueError('test command violates executable allowlist')
            patch = completion_to_text(completion).strip()
            if not patch or len(patch) > 200000 or not patch_paths_are_safe(patch):
                raise ValueError('empty, oversized, or unsafe patch')
            with tempfile.TemporaryDirectory(prefix='gemma-grpo-') as temp_dir:
                temp = Path(temp_dir)
                baseline = temp / 'baseline'
                shutil.copytree(source, baseline, ignore=shutil.ignore_patterns('.git', '.venv', '__pycache__', '.pytest_cache', 'node_modules', '.tox'))
                base_result = isolated_test(baseline, command)
                if base_result.returncode == 0:
                    reward, detail = 0.0, 'tests_already_pass_on_base'
                else:
                    candidate = temp / 'candidate'
                    shutil.copytree(baseline, candidate)
                    patch_file = temp / 'candidate.patch'
                    patch_file.write_text(patch + '\\n', encoding='utf-8')
                    checked = subprocess.run(['git', 'apply', '--check', str(patch_file)], cwd=candidate,
                                             capture_output=True, text=True, timeout=10, shell=False)
                    if checked.returncode:
                        reward, detail = -0.2, 'patch_not_applicable'
                    else:
                        applied = subprocess.run(['git', 'apply', str(patch_file)], cwd=candidate,
                                                 capture_output=True, text=True, timeout=10, shell=False)
                        if applied.returncode:
                            reward, detail = -0.2, 'patch_apply_failed'
                        else:
                            test_result = isolated_test(candidate, command)
                            reward = 1.0 if test_result.returncode == 0 else 0.0
                            detail = 'tests_pass' if reward else f'tests_fail:{test_result.returncode}'
        except subprocess.TimeoutExpired:
            reward, detail = 0.0, 'test_timeout'
        except Exception as exc:
            reward, detail = -0.2, f'{type(exc).__name__}:{exc}'
        rewards.append(float(reward))
        GRPO_REWARD_EVENTS.append({'reward': float(reward), 'detail': detail})
        del GRPO_REWARD_EVENTS[:-500]
    return rewards

if RUN_GRPO:
    if not TRUST_REPOSITORY_TESTS:
        raise RuntimeError('Review snapshots/tests, then explicitly set TRUST_REPOSITORY_TESTS=True.')
    require_training_runtime()
    if not grpo_rows:
        raise ValueError('No GRPO rows loaded.')
    from datasets import Dataset
    from peft import PeftModel
    from transformers import AutoModelForMultimodalLM, AutoProcessor
    from trl import GRPOConfig, GRPOTrainer
    import inspect

    normalized = normalize_grpo_rows(grpo_rows)
    if GRPO_MAX_TASKS:
        normalized = random.Random(SEED).sample(normalized, min(GRPO_MAX_TASKS, len(normalized)))
    if len(normalized) < 2:
        raise ValueError('GRPO requires at least two tasks for grouped samples.')
    if not (ADAPTER_FOR_GRPO / 'adapter_config.json').is_file():
        raise FileNotFoundError('Train SFT first or set ADAPTER_FOR_GRPO to a reviewed adapter directory.')
    grpo_ds = Dataset.from_list(normalized)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    processor = AutoProcessor.from_pretrained(str(MODEL_PATH), local_files_only=True)
    base_model = AutoModelForMultimodalLM.from_pretrained(str(MODEL_PATH), dtype=dtype, device_map='auto',
                                                          low_cpu_mem_usage=True, local_files_only=True)
    model = PeftModel.from_pretrained(base_model, str(ADAPTER_FOR_GRPO), is_trainable=True)
    model.config.use_cache = False
    grpo_values = {
        'output_dir': str(GRPO_OUTPUT), 'learning_rate': 1e-6,
        'per_device_train_batch_size': 1, 'gradient_accumulation_steps': 1,
        'num_generations': 2, 'max_prompt_length': 2048, 'max_completion_length': 1024,
        'max_steps': 8, 'beta': 0.0, 'logging_steps': 1, 'save_strategy': 'no',
        'bf16': dtype == torch.bfloat16, 'gradient_checkpointing': True,
        'report_to': 'none', 'remove_unused_columns': False, 'use_vllm': False, 'seed': SEED,
    }
    cfg_sig = inspect.signature(GRPOConfig.__init__)
    cfg_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in cfg_sig.parameters.values())
    grpo_values = grpo_values if cfg_kwargs else {k: v for k, v in grpo_values.items() if k in cfg_sig.parameters}
    grpo_args = GRPOConfig(**grpo_values)
    trainer_kwargs = {'model': model, 'args': grpo_args, 'train_dataset': grpo_ds,
                      'reward_funcs': code_patch_reward, 'processing_class': processor.tokenizer}
    trainer_sig = inspect.signature(GRPOTrainer.__init__)
    if not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in trainer_sig.parameters.values()):
        trainer_kwargs = {k: v for k, v in trainer_kwargs.items() if k in trainer_sig.parameters}
    trainer = GRPOTrainer(**trainer_kwargs)
    result = trainer.train()
    trainer.save_model(str(GRPO_OUTPUT))
    if not (GRPO_OUTPUT / 'adapter_model.safetensors').is_file():
        raise RuntimeError('GRPO did not write adapter_model.safetensors')
    (WORK_DIR / 'grpo_metrics.json').write_text(json.dumps({
        'metrics': getattr(result, 'metrics', {}), 'reward_events': GRPO_REWARD_EVENTS,
        'tasks': [row['task_id'] for row in normalized], 'seed': SEED,
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    print('GRPO adapter saved:', GRPO_OUTPUT)
else:
    print('GRPO skipped; enable only after SFT adapter and sandboxed-trust review are ready.')
"""),
    markdown("## 6. 训练后检查与产物\n\n只将 adapter 与 metrics 作为训练产物保存。将 adapter 接回比赛 harness 前，确认 `adapter_config.json` 的 base model 与比赛指定模型一致，权重非空、可加载，并在相同 prompt/harness/预算上做成对评测。不要把训练 notebook 的 SFT/GRPO loss 当作修复成功率。"""),
    code("""artifacts = []
for adapter_dir in (SFT_OUTPUT, GRPO_OUTPUT):
    if adapter_dir.is_dir():
        artifacts.extend(str(path) for path in adapter_dir.iterdir() if path.is_file())
print('Training artifacts currently present:', artifacts)
if TRAINING_LOG.exists():
    print(TRAINING_LOG.read_text(encoding='utf-8')[:4000])
"""),
    markdown("""## 7. 执行顺序与已知边界

1. 外置 collector 以 `--rationale-mode detailed` 采集 explicit rationale 与 patch 候选；保存 teacher/model/prompt/data hash。
2. 审核来源与服务条款；对每个候选独立重放补丁，验证基线测试 fail、补丁后目标测试 pass；生成 canonical records。
3. 运行 `tools/prepare_posttrain_data.py`，使用 task split（官方 129 题只有四个仓库，不能伪称 repo-disjoint holdout），审阅 train/validation 清单。
4. 在 Kaggle UI 手动选择 GPU，先启用 `RUN_SFT=True` 训练一个 epoch；检查 adapter 后再配好快照与测试启动 GRPO 小跑。
5. GRPO 的 pytest/测试代码仍会在 notebook worker 中执行；虽然限制到 `/kaggle/input` 快照、禁 shell、限时并拒绝 symlink，这不等价于 OS 级隔离容器。只对自己审过的可信仓库/测试显式启用。

本 notebook 的 metadata 明确保持 Kaggle accelerator=`none`、`isGpuEnabled=false`，与项目规则一致；GPU 由运行者在 Kaggle UI 设置。训练阶段未在本地执行，因为本机没有 Kaggle 模型挂载、完整 129 快照与目标 L4x4 环境。"""),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kaggle": {
            "accelerator": "none",
            "dataSources": [
                {"sourceType": "competition", "sourceId": 149921},
                {"sourceType": "modelInstanceVersion", "sourceId": 897258},
            ],
            "isInternetEnabled": False,
            "language": "python",
            "sourceType": "notebook",
            "isGpuEnabled": False,
        },
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12.0"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(OUT)
