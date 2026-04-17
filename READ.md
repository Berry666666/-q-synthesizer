# Q Synthesizer 详细说明（READ）

本文档面向数据工程、训练工程和评测工程，目标是把本项目的长程轨迹数据生产流程讲清楚：

- 这条管线为什么能产出可训练数据。
- 每个阶段具体做了什么、输出了什么。
- 如何开启外部大模型增强（特别是子目标表达增强）。
- 如何做质量门禁、去重均衡、切分和复现。

## 1. 你会得到什么

该工程用于批量生成长程任务训练数据，核心收益如下：

- 任务是长程结构，不是单轮问答。
- 数据含完整执行轨迹，不只是静态问题文本。
- 数据含动态事件纠偏链路，适合训练 replan 能力。
- 自带质量门禁与分布控制，可直接进入训练流水线。
- 自动产出 train/val/test 与 summary，便于版本化管理。

## 2. 项目关键文件

- `configs/default_profiles.json`：复杂度档位、默认阈值、基础域配置。
- `configs/industry_catalog.json`：行业扩展词库与约束素材。
- `src/q_synth/synthesizer.py`：单样本生成器（Q、A、task_graph、quality）。
- `scripts/run_trajectory_pipeline.py`：主流水线（生成、筛选、去重、均衡、切分、摘要）。
- `scripts/inspect_q_dataset.py`：产物体检脚本。
- `data/`：输出目录。

## 3. 一图看全流程

主流程可以拆为 7 个阶段：

1. 配置加载与行业合并
2. 候选样本生成（Q/QA + task_graph + meta）
3. 轨迹构建（plan + tool_execution + correction + turns）
4. 轨迹评分与门禁过滤
5. 去重与均衡抽样
6. train/val/test 切分
7. summary 与产物清单输出

执行入口：

```bash
python3 scripts/run_trajectory_pipeline.py \
  --profile mixed \
  --num-samples 300 \
  --candidate-multiplier 5 \
  --min-quality 0.76 \
  --min-plan-phases 3 \
  --min-tool-steps 10 \
  --min-corrections 1 \
  --min-trajectory-score 0.68 \
  --min-tool-coverage 0.35 \
  --min-phase-execution-coverage 0.70 \
  --min-industries 10 \
  --split-ratios 0.8,0.1,0.1 \
  --save-stage-artifacts \
  --output data/trajectory_mixed_300.jsonl
```

## 4. 生成模式与策略矩阵

### 4.1 Q 生成模式

- `--q-generation-mode rule`：规则模板生成 Q。
- `--q-generation-mode llm`：外部 LLM 直接生成 Q。
- `--q-generation-mode hybrid`：规则草稿 + LLM 重写 Q。

### 4.2 子目标生成模式

- `--subgoal-generation-mode rule`：子目标表达使用规则分支。
- `--subgoal-generation-mode llm`：子目标表达由外部 LLM 生成。
- `--subgoal-generation-mode hybrid`：规则骨架 + LLM 重写子目标表达。
- `--subgoal-generation-mode auto`：
  - 若提供了 LLM 参数（`--llm-base-url` + `--llm-model`），优先走 LLM 子目标分支。
  - 若未提供 LLM 参数：当 Q 为 rule 时走 rule；否则走 hybrid。

### 4.3 子目标“固定模板”现状说明

当前代码已经去掉初始化时硬编码的 12 条动作词。现在行为是：

- LLM/hybrid 子目标模式：草稿不依赖固定动作词，最终表达由模型生成。
- rule 子目标模式：若配置了 `defaults.fallback_action_pool` 才使用动作池；否则使用通用草稿表达。

这保证了你可以真正把子目标表达交给外部模型，不再被固定词表锚定。

### 4.4 LLM 开关与最小必需参数

当任一分支使用 LLM（Q 或子目标）时，至少需要：

- `--llm-base-url`
- `--llm-model`
- `--llm-api-key`（或环境变量 `--llm-api-key-env`）

如果以上参数缺失，流水线会在启动阶段直接报错并退出。

## 5. 每个阶段到底做了什么

### 5.1 阶段 A：配置加载与行业合并

输入：

- `configs/default_profiles.json`
- `configs/industry_catalog.json`

流程：

- 加载基础 domains。
- 将行业扩展映射为统一结构并合并。
- 校验最小字段：`org_pool`、`focus_pool`、`tool_pool` 至少 4 项。

产出：

- 更广覆盖的可采样 domain 集。

### 5.2 阶段 B：候选样本生成

由 `QSynthesizer.generate_one()` 生成单条候选，含：

- `Q`：任务问题。
- `A`：参考规划（若非 `--q-only`）。
- `task_graph`：子目标节点与依赖边。
- `context`：约束、事件、工具等结构化上下文。
- `quality`：基础质量评分。

候选池目标大小：

- `target_candidates = num_samples * candidate_multiplier`（下限为 `num_samples`）。

基础门禁：

- `quality.score >= threshold`
- `feasible_hours = true`
- `feasible_budget = true`

### 5.3 阶段 C：轨迹构建

对每个通过基础门禁的候选生成 `trajectory`：

1. `long_range_plan`
   - 按 layer 切分阶段。
   - 每阶段输出周窗、里程碑、关键工具、准入/退出条件。

2. `tool_execution`
   - 按子目标展开工具执行步骤。
   - 状态为 `partial` 时自动插入修复步骤，形成闭环。

3. `correction_trace`
   - 由动态事件驱动纠偏链路（诊断 -> 重规划 -> 验证/回滚）。

4. `turns`
   - 多角色回合串联（planner/executor/critic），用于训练对话式轨迹能力。

### 5.4 阶段 D：轨迹指标与评分

先计算 `trajectory_metrics`：

- `plan_phases`
- `tool_steps`
- `correction_steps`
- `turns`
- `tool_coverage`
- `phase_execution_coverage`
- `correction_action_coverage`
- `correction_link_coverage`

再计算 `trajectory_quality.score`，分量权重为：

- phase_depth: 0.16
- execution_density: 0.22
- correction_density: 0.18
- tool_coverage: 0.14
- phase_execution_coverage: 0.14
- turn_richness: 0.10
- correction_link_coverage: 0.06

### 5.5 阶段 E：轨迹门禁

样本必须同时满足：

- `plan_phases >= min_plan_phases`
- `tool_steps >= min_tool_steps`
- `correction_steps >= min_corrections`
- `tool_coverage >= min_tool_coverage`
- `phase_execution_coverage >= min_phase_execution_coverage`
- `trajectory_quality.score >= min_trajectory_score`

### 5.6 阶段 F：去重与均衡抽样

去重：

- 精确签名去重：`signature_hash`。
- 近重复去重：`dedup_text` 的 shingle Jaccard。

均衡：

- 先满足 `min_industries`。
- 行业 cap：`max_per_industry`（0 时自动估算）。
- focus cap：`max_per_focus`（0 时自动估算）。
- round-robin 继续填充直到达到目标样本。

综合排序分：

- `combined_score = 0.65 * quality + 0.35 * trajectory_quality`

### 5.7 阶段 G：切分与摘要

切分默认开启：

- 按 `(domain, profile)` 分组，尽量保持组内均衡。
- 默认比例 `0.8,0.1,0.1`。

摘要包含：

- 规模统计、拒绝统计、过滤条件。
- 质量与轨迹评分统计。
- 行业与 focus 分布。
- split 分布。
- 产物行数、大小、sha256。

## 6. 评分公式（便于训练侧调参）

### 6.1 基础质量 quality.score

来自 `QSynthesizer._quality_score()`：

- subgoal_complexity: 0.24
- dependency_depth: 0.24
- constraint_density: 0.20
- dynamic_events: 0.16
- tool_diversity: 0.16

若不可行（工时或预算不满足）则额外罚分 0.20。

### 6.2 轨迹质量 trajectory_quality.score

见上节阶段 D，强调执行密度和纠偏密度。

建议理解为：

- 基础质量回答“任务构造是否难而可解”。
- 轨迹质量回答“执行过程是否足够长程、可观测、可纠偏”。

## 7. 子目标外部大模型专题（重点）

### 7.1 为什么要单独做子目标 LLM

仅让 Q 走 LLM 仍可能出现子目标表达趋同。子目标单独接入 LLM 能显著提升：

- 动作表达多样性。
- 行业语义贴合度。
- 任务执行语言的自然度。

### 7.2 子目标 LLM 输出契约

要求模型只输出 JSON：

```json
{
  "subgoals": [
    {"id": "SG-01", "title": "...", "objective": "..."}
  ]
}
```

系统会强制保留以下结构字段，不允许被模型篡改：

- `id`
- `layer`
- `dependencies`
- `required_tools`

模型只改：

- `title`
- `objective`

### 7.3 子目标参数（独立于 Q 参数）

- `--subgoal-llm-temperature`
- `--subgoal-llm-top-p`
- `--subgoal-llm-max-tokens`
- `--subgoal-llm-max-retries`
- `--subgoal-llm-fallback-to-rule`

如果不显式设置这些参数，会回退使用通用 LLM 参数。

### 7.4 失败与回退行为

- 未开回退：子目标 LLM 失败会直接中断样本生成。
- 开启 `--subgoal-llm-fallback-to-rule`：失败时回到规则子目标。

可在样本 `meta` 中追踪：

- `subgoal_generation_mode`
- `subgoal_source`（如 `llm-generate` / `llm-hybrid` / `rule-fallback`）

## 8. Seed 泛化与可复现

参数：

- `--seed`
- `--seed-mode`：`single` / `cycle` / `hash`
- `--seed-pool`
- `--seed-step`

模式说明：

- `single`：固定 seed。
- `cycle`：`seed = pool[idx] + attempt * seed_step`。
- `hash`：基于 `(base, base_seed, attempt, profile, seed_step)` 派生哈希 seed。

推荐：

- 训练集生成用 `hash + 多 seed_pool`。
- 小规模调试用 `single`。

每条样本会记录 `meta.sample_seed`，可用于追溯与复现。

## 9. 输出结构详解

单条样本核心字段：

- `id`, `created_at`, `profile`, `domain`, `industry`
- `Q`, `A`（可选）
- `quality`
- `context`
- `meta`
- `task_graph`
- `trajectory`
- `trajectory_metrics`
- `trajectory_quality`
- `pipeline_version`

训练侧优先关注：

- `task_graph.nodes/edges`
- `trajectory.long_range_plan`
- `trajectory.tool_execution`
- `trajectory.correction_trace`
- `trajectory.turns`

## 10. 参数分组速查

### 10.1 规模与候选池

- `--num-samples`
- `--candidate-multiplier`
- `--max-retries-per-sample`
- `--max-generation-attempts`

### 10.2 LLM（通用）

- `--q-generation-mode`
- `--llm-base-url`
- `--llm-model`
- `--llm-api-key`
- `--llm-api-key-env`
- `--llm-timeout-sec`
- `--llm-temperature`
- `--llm-top-p`
- `--llm-max-tokens`
- `--llm-max-retries`
- `--llm-fallback-to-rule`

### 10.3 子目标 LLM（专用）

- `--subgoal-generation-mode`
- `--subgoal-llm-temperature`
- `--subgoal-llm-top-p`
- `--subgoal-llm-max-tokens`
- `--subgoal-llm-max-retries`
- `--subgoal-llm-fallback-to-rule`

### 10.4 轨迹门禁

- `--min-plan-phases`
- `--min-tool-steps`
- `--min-corrections`
- `--min-trajectory-score`
- `--min-tool-coverage`
- `--min-phase-execution-coverage`

### 10.5 去重与均衡

- `--near-dup-threshold`
- `--min-industries`
- `--max-per-industry`
- `--max-per-focus`

### 10.6 切分与中间产物

- `--split-ratios`
- `--disable-splits`
- `--save-stage-artifacts`
- `--stage-dir`

## 11. 推荐命令模板

### 11.1 标准训练集（默认推荐）

```bash
python3 scripts/run_trajectory_pipeline.py \
  --profile mixed \
  --num-samples 5000 \
  --candidate-multiplier 5 \
  --min-quality 0.76 \
  --min-plan-phases 3 \
  --min-tool-steps 10 \
  --min-corrections 1 \
  --min-trajectory-score 0.68 \
  --min-tool-coverage 0.35 \
  --min-phase-execution-coverage 0.70 \
  --min-industries 10 \
  --seed-mode hash \
  --seed-pool 42,2026,4096,8192 \
  --output data/trajectory_mixed_5k.jsonl
```

### 11.2 严格高质量集

```bash
python3 scripts/run_trajectory_pipeline.py \
  --profile mixed \
  --num-samples 3000 \
  --candidate-multiplier 7 \
  --min-quality 0.80 \
  --min-plan-phases 4 \
  --min-tool-steps 12 \
  --min-corrections 2 \
  --min-trajectory-score 0.74 \
  --min-tool-coverage 0.45 \
  --min-phase-execution-coverage 0.80 \
  --near-dup-threshold 0.84 \
  --min-industries 12 \
  --output data/trajectory_mixed_strict_3k.jsonl
```

### 11.3 Q+子目标都用 LLM 增强

```bash
python3 scripts/run_trajectory_pipeline.py \
  --q-generation-mode hybrid \
  --subgoal-generation-mode hybrid \
  --llm-base-url https://api.openai.com/v1 \
  --llm-model gpt-4o-mini \
  --llm-api-key "$OPENAI_API_KEY" \
  --llm-fallback-to-rule \
  --subgoal-llm-fallback-to-rule \
  --profile mixed \
  --num-samples 3000 \
  --output data/trajectory_hybrid_3k.jsonl
```

### 11.4 仅子目标用 LLM（Q 保持 rule）

```bash
python3 scripts/run_trajectory_pipeline.py \
  --q-generation-mode rule \
  --subgoal-generation-mode auto \
  --llm-base-url https://api.openai.com/v1 \
  --llm-model gpt-4o-mini \
  --llm-api-key "$OPENAI_API_KEY" \
  --subgoal-llm-fallback-to-rule \
  --profile mixed \
  --num-samples 2000 \
  --output data/trajectory_subgoal_llm_2k.jsonl
```

## 12. 验收与体检清单

运行后建议检查：

1. 基础规模：`samples_selected` 是否达标。
2. 覆盖度：`unique_industries` 是否满足预期。
3. 轨迹密度：`avg_tool_steps`、`avg_correction_steps` 是否足够。
4. 可执行性：`avg_tool_coverage`、`avg_phase_execution_coverage` 是否过低。
5. LLM 生效：检查 `q_generation` 与 `subgoal_generation` 模块。
6. 样本级追踪：抽样查看 `meta.q_source` 与 `meta.subgoal_source`。

体检命令：

```bash
python3 scripts/inspect_q_dataset.py --input data/trajectory_mixed_300.jsonl
```

## 13. 常见问题与排障

### 13.1 selected 小于目标样本

优先顺序：

1. 提高 `candidate-multiplier`。
2. 放宽门禁（`min-trajectory-score`、`min-tool-coverage`、`min-phase-execution-coverage`）。
3. 放宽近重复阈值（提高 `near-dup-threshold`）。

### 13.2 行业覆盖不足

处理方式：

1. 降低 `min-industries`。
2. 扩充 `industry_catalog.json`。
3. 增大候选池与生成尝试上限。

### 13.3 LLM 报错或超时

处理方式：

1. 检查 `--llm-base-url`、`--llm-model`、`--llm-api-key`。
2. 先把 `--llm-timeout-sec` 调大到 90~120。
3. 开启 `--llm-fallback-to-rule` 和/或 `--subgoal-llm-fallback-to-rule`。

### 13.4 split 分布抖动明显

处理方式：

1. 增大总样本规模。
2. 调整 `split-ratios`。
3. 观察 `(domain, profile)` 是否极端稀疏。

## 14. 复现与版本固化

建议固定以下信息：

1. 命令行参数（完整保存）。
2. summary 文件。
3. artifacts 里的 sha256。
4. Git commit id。

最小复现流程：

1. 拉取同一 commit。
2. 使用相同参数重新运行。
3. 对比 summary 与 artifacts 哈希。

## 15. 版本说明

当前主线：

- `pipeline_version = trajectory_pipeline_v2`

相对早期版本主要增强：

- 轨迹质量评分门禁
- 更完整执行与纠偏字段
- 去重与均衡抽样增强
- 全局配额切分
- 产物哈希清单

