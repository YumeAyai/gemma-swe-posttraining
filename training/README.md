# LoRA SFT → GRPO 数据与训练体系

## 目标

训练对象不是“所有看起来像 CoT 的文本”，而是 Gemma 在这场代码修复任务里可重复执行的策略：读 issue、定位代码、选择工具、形成最小改动、运行相关验证、根据反馈继续或提交。训练成功标准是同预算下的修复通过率，而不是训练 loss 或轨迹长度。

Qwen/Nemotron 笔记可迁移的是数据筛选和实验纪律：保留模型自己的有效轨迹；从同题失败中找具体故障；teacher 修复必须独立于 gold patch/test；对长而错、循环、无补丁轨迹做负例；按任务/仓库隔离验证集；早停看端到端指标。笔记里的模型、题型、奖励和步数仅是来源实验结果，不直接当作 Gemma/SWE 的默认超参。

## 数据从哪里来

优先建候选池，再经统一格式和门禁筛选；文献笔记本身用于设计策略，不作为 assistant 目标文本。

| 来源 | 可贡献的信号 | 入库条件 |
| --- | --- | --- |
| 自己的公开训练任务 rollout | 真实工具选择、观察、编辑、验证和失败行为 | 非比赛评测任务；保留工具 I/O；成功 SFT 必须补丁可重放且测试通过 |
| 授权 teacher rollout | 针对学生失败的诊断和可执行修复轨迹 | teacher/数据使用权明确；teacher 看不到参考 patch、hidden tests 或答案标签；独立验证后再标注 |
| 有许可证的公开 PR/issue 数据 | issue→补丁、代码上下文、历史修复模式 | 逐仓库记录许可证；训练/验证仓库隔离；确认测试与上下文可重放 |
| 合成任务 / 变异任务 | 边界条件、回归、API 契约等稀缺覆盖 | 从真实许可代码生成；自动验证 mutation 和修复；记录生成器版本、随机种子与基线 |
| 同题多 rollout | 成功/失败对比、GRPO 组内奖励、偏好对 | 同一初始状态、同一工具和预算；保存每条原始轨迹及验证结果 |

Claude 或其他闭源 teacher 的使用不能仅凭传闻推断允许或禁止。当前比赛说明明确要求提交固定 Gemma base model、LoRA 可选；这并不等同于对离线 teacher 数据使用权的完整授权。正式采集前逐项核对 teacher 服务条款、仓库许可证、Kaggle 规则及数据是否含不可公开内容。不要收集或导出运行时私有隐藏思维；仅使用服务明确提供且允许训练的显式 rationale，或可观察的工具调用/工具返回。

官方 `tasks.jsonl` 的 129 道题是**公开训练/开发集**，带 `patch` 与 `test_patch`，可用于本地训练和验证；提交评分会换成约 120 道隐藏题。Kaggle Getting Started 还说明隐藏集来自私有仓库，主办方在论坛确认训练集仓库不出现在测试集。不要把隐藏评分日志、隐藏题或推测出来的隐藏标签混入训练。来源：[官方 Getting Started 数据说明](https://www.kaggle.com/code/ryanholbrook/getting-started-gemma-4-developer-agent/input)、[Kaggle Staff 说明](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/747248)。

### 用 teacher 处理这 129 题

对每题先冻结 `base_commit` 和工具 schema。Teacher 只收到 issue、hints 和仓库快照，不给 `patch`/`test_patch`；采集多条可观察轨迹（工具名、参数、返回摘要、编辑 diff、测试选择、最终 diff），原始输出按 task/teacher/prompt 版本保存。随后独立做两段验证：`test_patch` 在原始基线上应失败；teacher patch 应能应用并让验证测试通过。验证结果再用公开 `patch` 做行为对照/错误定位，不将参考答案提前泄露给 teacher。

先从官方 JSONL 导出 blind prompt manifest（字段层面剔除 reference patch/test patch）：

```bash
uv run python -B tools/export_public_teacher_prompts.py \
  --tasks gemma4/tasks.jsonl \
  --output data/candidates/public129_blind_prompts.jsonl
```

这一步只准备 teacher 输入和快照相对路径，不会调用模型。实际 teacher rollout 需要在可访问完整 129 个快照的训练/推理环境里运行；本地工作区目前只留 4 个 snapshot tarball。

项目还提供独立、无第三方依赖的 `uv` teacher candidate collector。它按 issue 词项从官方预索引 graph 中取少量代码节点，形成有界上下文；默认只写 dry-run prompt，不发送 API 请求。建议先审查 4 题的输入，再显式打开调用：

```bash
cd training/collector
uv sync
export TEACHER_API_KEY='由你的密钥管理器注入，不要写入 notebook/仓库'
export TEACHER_MODEL='你的教师模型 ID'

# 演练：默认只选前 4 题，不发生网络/API 调用
uv run collector.py \
  --tasks ../../gemma4/tasks.jsonl \
  --graphs ../../gemma4/graph \
  --output ../../data/candidates/teacher-dry-run.jsonl \
  --allow-repo-graph-fallback

# 确认 prompt 后才真实调用；先小样本测质量与费用
uv run collector.py \
  --tasks ../../gemma4/tasks.jsonl \
  --graphs ../../gemma4/graph \
  --snapshots ../../gemma4/snapshots \
  --output ../../data/candidates/teacher-candidates.jsonl \
  --task-id fastapi_15745 --task-id httpx_3672 \
  --task-id requests_7433 --task-id rich_3064 \
  --limit 4 --samples-per-task 1 --rationale-mode detailed --execute

# 预算/质量确认后，显式扩大到 129 题；2 samples/task 即最多 258 次 API 调用
uv run collector.py \
  --tasks ../../gemma4/tasks.jsonl \
  --graphs ../../gemma4/graph \
  --snapshots ../../gemma4/snapshots \
  --output ../../data/candidates/teacher-candidates.jsonl \
  --limit 129 --samples-per-task 2 --rationale-mode detailed --allow-repo-graph-fallback --execute --resume
```

可用 `--rationale-mode brief|detailed|none` 控制**教师明确输出的、面向用户的逐步理由**，默认 `detailed`；结果字段叫 `explicit_rationale`。这让采集接口能拿到用于 SFT 的显式过程监督，但它不是 provider 私有隐藏思维链的逐 token dump。可用 `--base-url` 指向兼容 OpenAI Chat Completions 的 HTTPS 服务，`--api-key-env` 指定密钥环境变量名。每条结果记录模型/API 用量、数据/图/prompt 哈希、候选补丁、显式理由和 `missing_context`；不读取 provider 私有 reasoning 字段、不保留原始响应正文（仅留其哈希），并过滤 rationale 中常见的隐藏推理标记；也不声称执行过工具或测试。`--resume` 仅跳过同一数据、模型、任务、样本序号及 prompt 哈希下已经成功采集的记录，dry-run 与失败项不会误标完成。断网/超时不自动重试，避免重复计费；HTTP 5xx 有限重试。

任务 commit 与每仓库唯一 graph 索引不一致时，默认不提供 graph 代码上下文；显式打开 `--allow-repo-graph-fallback` 才回退到该仓库索引，并在输入/JSONL 中标注索引 revision 与 `graph_matches_task_commit=false`。这些摘录只能帮助定位，不能当成题目快照源代码；teacher patch 必须在对应的确切 task snapshot 上独立重放和验证。

注意：当前 collector 是“盲态 patch/简短策略摘要候选”采集，不是 agent 沙箱 rollout；它不会对仓库执行 `read_file`/`edit_file`/测试，也不会生成真实工具调用轨迹。产物全部是未验证候选，`rights_review=pending`，不能直接喂训练。需要先在隔离环境重放补丁、运行目标测试，并通过来源/条款审核，再转换成 canonical record。要采集实际工具轨迹，应另接有隔离容器的 harness runner；不可把模型自述当成真实工具观察。

每道题可产出三类不同监督，不要混成一条：

1. **策略 SFT**：实际工具轨迹 + 验证通过的 patch，教定位、工具选择、最小编辑和验证闭环。
2. **Patch SFT**：issue/上下文到 reference patch，作为单独消融；它有监督密度，但不教工具链，且容易受题面/测试不足影响。
3. **同题对比/GRPO**：同一初始状态多次 rollout，pass patch 为正；不适用 patch、测试失败、无编辑、重复探索、超时分别标注。GRPO 优化器只看到 train tasks；按 heldout task 评估。

129 题只有 4 个公开仓库，因此初期验证应按 repo 分层做 task-level split/CV，避免验证集只剩某一个 repo；同时加入来源独立、许可明确的外部 repo holdout 测泛化。不能把 task-level validation 描述成 repo-disjoint。最终决定训练配比后可以用完整 129 题重训，但最终指标必须来自完全独立的数据。

Claude/闭源 teacher 能否用于 Kaggle 训练不能凭传闻下结论。比赛当前公开材料指定 Gemma 变体作为提交模型；外部 teacher API 是否满足外部数据可访问性/条款/获奖开源义务，需要主办方澄清。先用本地/比赛允许的 Gemma teacher 跑通数据闭环，并将 Claude 轨迹标记为 `rights_review=pending`，在确认前不会进入训练导出。

## 标准样本合同

每个 canonical 候选是一行 JSONL，完整字段见 `record.schema.json`。如含显式教师理由，单独存 `explicit_rationale` 并在 SFT assistant target 中保留；它必须是可见、获准使用的面向用户解释，不是 provider 私有 reasoning 字段。关键字段示例：

```json
{
  "record_id": "source/task/rollout-03",
  "task": {
    "task_id": "task-03",
    "repo": "owner/project@commit-sha",
    "prompt": "Issue and bounded source context",
    "repo_snapshot": "snapshots/task-03",
    "test_command": ["python", "-m", "pytest", "tests/test_issue.py", "-q"]
  },
  "source": {
    "kind": "student_rollout",
    "reference": "run-id/trajectory-id",
    "license": "MIT",
    "rights_review": "approved",
    "allowed_modes": ["sft", "grpo", "preference"],
    "reasoning_visibility": "observable_trace"
  },
  "messages": [
    {"role": "user", "content": "Issue and repository context"},
    {"role": "assistant", "content": "Observable plan/tool trace and final unified diff"}
  ],
  "patch": "diff --git a/src/parser.py b/src/parser.py\\n...",
  "verification": {
    "status": "passed",
    "patch_replayable": true,
    "test_exit_code": 0,
    "verifier_ref": "container-image-and-command-digest"
  },
  "evaluation_only": false
}
```

`rights_review=approved` 是人工审核结论，不可由脚本自动推断。未知来源先标 `pending`，不会进入导出集。原始轨迹与加工产物应分目录、只追加保存，并在实验 manifest 记录输入文件 SHA-256；训练脚本只读导出集。

## 门禁与导出

```bash
uv run python -B tools/prepare_posttrain_data.py \
  --input data/candidates/canonical.jsonl \
  --output-dir data/processed/run-001 \
  --data-root data/snapshots \
  --validation-fraction 0.2 \
  --split-strategy task
```

脚本负责格式/来源字段校验、排除 pending/标记为评测的数据、固定切分、SFT 正例筛选、同题偏好对生成，以及 GRPO 任务清单导出。对这 129 题使用 `--split-strategy task`，且 canonical rows 显式标 `evaluation_only=false`；外部语料是否属于许可训练集需单独审核。脚本不会执行仓库测试命令。SFT 正例必须 `status=passed` 且 `patch_replayable=true`；GRPO 任务只从 train task 导出，快照路径必须在 `--data-root` 下且不含 `..`。实际运行测试前，GRPO worker 必须使用无网络、非特权、资源/时间受限的独立容器；不能在信任边界外直接执行样本给出的命令。

## 训练顺序

1. **基线与数据审计**：先测无 LoRA 的工具行为和任务通过率；统计来源、任务类别、重复率、轨迹长度、成功率、首个有效读取/编辑时间、空补丁/超时/循环。没有数据清单和任务级 baseline，不启动训练。
2. **SFT**：先用通过验证的短而完整策略轨迹 warm start；逐步加入有信息量的失败修复轨迹和授权 teacher 数据。每次只加一个来源/数据配比变量。必要时保留长而正确轨迹，不以“短”为唯一质量标准。按仓库 holdout 比较任务 pass rate、工具调用、首次编辑、patch replay 与 wall-clock；保存多个 checkpoint，按 holdout 选择而非最后一步。
3. **GRPO**：仅在同一任务能生成多个 rollout 且奖励器经过人工对照/回归测试后启动。基础奖励以 patch 可应用、目标测试由 fail→pass 为主；测试原本通过、命令超时、环境错误、拒绝执行、改测试/CI/依赖规避等情况应分别记录，不能把“测试没失败”都算正奖励。过程奖励只能小于真实修复奖励，不能鼓励为拿分而过早编辑或盲目调用工具。
4. **LoRA 检查**：在装入 harness 前检查 adapter 的 base model、target module/layer 覆盖、可训练参数数目、文件哈希和 `lora_B` 非零；跑 adapter load smoke test。已知当前旧 `main_lora`/`tool_lora` 为相同权重且 B 为零，不能把它们当成有效训练结果或 SFT 起点。
5. **配对消融**：base、SFT、SFT+GRPO 使用相同 prompt/harness/采样/任务顺序/预算。每阶段都有独立命名、seed、数据 hash、adapter hash、训练配置、checkpoint、运行日志和分任务结果；最终排行榜分数只作外部确认，不替代本地诊断。

## 首轮数据策略（假设，等待数据盘点验证）

按“一个例子是否教会可泛化动作”采样，不按 teacher 名气采样。首轮以可重放成功轨迹和真实公开 PR 修复对为主体；teacher 修复用来补学生具体失败类型；合成数据补长尾边界和 API 约束；同题失败轨迹优先留作 GRPO/preference，而不是作为正向 SFT。所有比例在 manifest 显示，并做按来源的 leave-one-source-out 消融。暂不写死各来源百分比，先统计可用样本质量与重叠后再定。

## 当前限制

- 所有 Kaggle notebook metadata 的 accelerator 固定为 `none`、`isGpuEnabled=false`；需要 GPU 时由运行者在 Kaggle UI 手动选择。配套训练 notebook 不会自动声明 GPU，也不自动启用训练；见 [`notebooks/train-gemma-sft-grpo-kaggle.ipynb`](../notebooks/train-gemma-sft-grpo-kaggle.ipynb)。
- 当前仓库的 `gemma4/tasks.jsonl` 有 129 条官方训练题，但本地只保留 4 个仓库快照；全量 teacher rollout 与 GRPO 需要 Kaggle 数据输入。尚无获批的闭源 teacher 数据，也没有经过隔离容器验证的独立 GRPO 执行器。因此这次建立的是数据合同与安全导出闭环，不声称已经训练出新 adapter。
- 现有 `tools/build_repair_sft.py` 是早期原型；在切换到这里的 canonical JSONL 前，不把它的输出直接当成正式训练数据。
