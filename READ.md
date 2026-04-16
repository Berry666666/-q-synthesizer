# Q Synthesizer 详细说明（READ）

本文档面向“数据工程与训练工程”场景，重点讲清楚本项目的长程轨迹数据生产管线（pipeline）是如何工作的、每一步会产出什么、如何调参、如何保障质量与覆盖。

## 1. 项目目标

该工程用于批量生成可训练的长程任务数据，核心特点如下：

- 任务具备长程规划结构（多阶段、多依赖、多约束）。
- 轨迹包含工具执行链路（tool execution trace）。
- 轨迹包含纠偏闭环（correction trace / replan）。
- 通过质量门禁、去重与行业均衡，保证数据可用性和覆盖度。
- 自动产出 train/val/test 切分与摘要报告，支持大规模训练前检查。

## 2. 目录与关键文件

- `configs/default_profiles.json`：基础复杂度配置、原始行业域模板、质量阈值。
- `configs/industry_catalog.json`：行业扩展配置（用于提升行业覆盖广度）。
- `src/q_synth/synthesizer.py`：基础任务实例生成器（Q/可选A + 任务图 + context）。
- `scripts/run_trajectory_pipeline.py`：精细化主管线（v2）。
- `scripts/inspect_q_dataset.py`：数据体检脚本（质量、分布、轨迹指标）。
- `data/`：输出目录（jsonl 数据与 summary）。

## 3. 管线总览

v2 管线可以拆成 7 个阶段：

1. 配置加载与领域合并
2. 候选样本生成（Q/QA + task_graph + context）
3. 轨迹构建（规划、执行、纠偏、对话回合）
4. 轨迹质量评估与门禁过滤
5. 去重与均衡抽样（行业 + focus）
6. 训练切分（train/val/test）
7. 摘要汇总与产物清单（含 sha256）

执行入口：

```bash
python3 scripts/run_trajectory_pipeline.py \
  --q-generation-mode rule \
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

`--q-generation-mode` 支持：

- `rule`：纯规则模板生成 Q（默认）。
- `llm`：调用外部 OpenAI 兼容 API 生成全新 Q。
- `hybrid`：先规则生成草稿，再用 LLM 重写增强多样性。

## 4. 各阶段细节

### 4.1 配置加载与领域合并

输入：

- 基础配置 `default_profiles.json`
- 行业扩展 `industry_catalog.json`

逻辑：

- 读取基础 domains。
- 将行业扩展映射为标准 domain 结构后合并。
- 对行业最小字段做校验：`org_pool`、`focus_pool`、`tool_pool` 必须至少 4 项。

结果：

- 形成更广行业覆盖的 domains 集合，用于后续采样。

### 4.2 候选样本生成

由 `QSynthesizer.generate_one()` 产生单条候选样本，核心字段：

- `Q`：长程任务问题。
- `A`：参考规划答案（当未启用 `--q-only`）。
- `task_graph.nodes/edges`：子目标图与依赖。
- `context`：约束、事件、可用工具等结构化上下文。
- `quality`：基础任务质量分（复杂度+可行性）。

候选池规模由以下参数控制：

- `num_samples * candidate_multiplier`

Q 文本生成由 `--q-generation-mode` 控制：

- `rule`：使用 `synthesizer.py` 内置模板渲染。
- `llm`：将结构化上下文发送到外部 LLM，要求输出 JSON `{\"Q\":\"...\"}`。
- `hybrid`：将规则草稿与结构化上下文一并发送给 LLM 重写。

LLM 必需参数：

- `--llm-base-url`
- `--llm-model`
- `--llm-api-key`（或环境变量 `--llm-api-key-env`，默认 `OPENAI_API_KEY`）

可调参数：

- `--llm-temperature`
- `--llm-top-p`
- `--llm-max-tokens`
- `--llm-timeout-sec`
- `--llm-max-retries`
- `--llm-fallback-to-rule`（LLM失败时回退规则分支）

基础门禁：

- `quality.score >= quality_threshold`
- 可行性通过：`feasible_hours` 且 `feasible_budget`

### 4.3 轨迹构建（核心）

每个通过基础门禁的候选样本，会构建 `trajectory`。

#### A. `long_range_plan`

- 按 task_graph 的 layer 划分阶段。
- 阶段字段包含：
  - `phase_id`
  - `phase_goal`
  - `weeks`（阶段周区间）
  - `subgoal_ids`
  - `key_tools`
  - `milestones`
  - `risk_signal`
  - `entry_criteria` / `exit_criteria`

#### B. `tool_execution`

- 按每个子目标展开工具执行步骤。
- 每步包含：
  - `step_id`
  - `phase_id`
  - `subgoal_id`
  - `tool`
  - `action`
  - `status`（success/partial/recovered）
  - `expected_signal` / `observed_signal`
  - `evidence_fields`

- 当状态是 `partial` 时，自动补一条修复执行步骤，形成执行闭环。

#### C. `correction_trace`

- 基于动态事件生成纠偏记录。
- 每条纠偏包含：
  - `trigger_event`
  - `phase_id`
  - `diagnosis`
  - `affected_step_ids`
  - `replan_actions`
  - `rollback_guard`
  - `verification`
  - `window_weeks`

这部分保证“事件触发 -> 诊断 -> 重规划 -> 验证/回退条件”完整存在。

#### D. `turns`

- 组织成多角色回合轨迹：planner/executor/critic。
- 回合类型覆盖：
  - `long_range_planning`
  - `phase_brief`
  - `tool_execution`
  - `trajectory_correction`
  - `replan_decision`

### 4.4 轨迹指标与质量评分

会先生成 `trajectory_metrics`，包括：

- `plan_phases`
- `tool_steps`
- `correction_steps`
- `turns`
- `tool_coverage`
- `phase_execution_coverage`
- `correction_action_coverage`
- `correction_link_coverage`

然后生成 `trajectory_quality.score`（0~1），由多个分量加权得到：

- 阶段深度
- 执行密度
- 纠偏密度
- 工具覆盖
- 阶段执行覆盖
- 回合丰富度
- 纠偏链路覆盖

### 4.5 轨迹门禁

样本必须同时满足：

- `plan_phases >= min_plan_phases`
- `tool_steps >= min_tool_steps`
- `correction_steps >= min_corrections`
- `tool_coverage >= min_tool_coverage`
- `phase_execution_coverage >= min_phase_execution_coverage`
- `trajectory_quality.score >= min_trajectory_score`

### 4.6 去重与均衡抽样

目标：避免重复、提升覆盖、控制行业/主题偏斜。

去重策略：

- 精确签名去重：`signature_hash`
- 近重复去重：基于 `dedup_text` + shingle Jaccard

均衡策略：

- 最小行业覆盖：`min_industries`
- 行业上限：`max_per_industry`
- 焦点上限：`max_per_focus`
- 先满足覆盖，再 round-robin 填充

综合排序分：

- `combined_score = 0.65 * quality + 0.35 * trajectory_quality`

### 4.7 数据切分（train/val/test）

默认启用切分（可 `--disable-splits` 关闭）。

机制：

- 先按 `(domain, profile)` 分组。
- 全局配额控制目标样本数（避免 test 过小）。
- 在满足全局配额前提下尽量保持组内均衡。

默认比例：

- `0.8,0.1,0.1`

### 4.8 摘要与产物清单

summary 文件会包含：

- 生成规模、拒绝统计、过滤条件
- 质量均值/最值
- 行业分布与 focus 分布
- split 分布
- 每个产物文件的：
  - 行数
  - 字节大小
  - sha256

## 5. 输出数据结构说明

单条样本典型顶层字段：

- `id`
- `created_at`
- `profile`
- `domain`
- `industry`
- `Q`
- `A`（可选）
- `quality`
- `context`
- `meta`
- `task_graph`
- `trajectory`
- `trajectory_metrics`
- `trajectory_quality`
- `pipeline_version`

其中，真正用于“长程轨迹学习”的关键是：

- `task_graph`（目标与依赖）
- `trajectory.long_range_plan`（阶段计划）
- `trajectory.tool_execution`（执行日志）
- `trajectory.correction_trace`（纠偏日志）
- `trajectory.turns`（多角色回合）

## 6. 运行配置建议

### 6.1 标准训练集（推荐默认）

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
  --split-ratios 0.8,0.1,0.1 \
  --output data/trajectory_mixed_5k.jsonl
```

### 6.2 严格高质集（更难、更干净）

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

### 6.4 LLM 增强版（提升Q泛化性）

```bash
python3 scripts/run_trajectory_pipeline.py \
  --q-generation-mode hybrid \
  --llm-base-url https://api.openai.com/v1 \
  --llm-model gpt-4o-mini \
  --llm-api-key "$OPENAI_API_KEY" \
  --llm-temperature 0.9 \
  --llm-top-p 0.95 \
  --llm-max-tokens 2200 \
  --llm-fallback-to-rule \
  --profile mixed \
  --num-samples 3000 \
  --candidate-multiplier 6 \
  --min-quality 0.80 \
  --min-trajectory-score 0.74 \
  --output data/trajectory_hybrid_3k.jsonl
```

### 6.3 仅生成 Q（无 A）

```bash
python3 scripts/run_trajectory_pipeline.py \
  --profile mixed \
  --num-samples 2000 \
  --q-only \
  --output data/trajectory_q_only_2k.jsonl
```

## 7. 验证与体检

```bash
python3 scripts/inspect_q_dataset.py --input data/trajectory_mixed_300.jsonl
```

重点关注：

- `avg_quality`
- `avg_trajectory_quality`
- `avg_tool_coverage`
- `avg_phase_execution_coverage`
- `trajectory_schema_distribution`
- `domain_distribution`

## 8. 常见问题与排查

### 8.1 样本不够（selected < num_samples）

处理方式：

- 提高 `candidate-multiplier`
- 放宽门禁：`min-trajectory-score` / `min-tool-coverage` / `min-phase-execution-coverage`
- 放宽去重：提高 `near-dup-threshold`

### 8.2 行业覆盖不足

处理方式：

- 降低 `min-industries`
- 扩展 `industry_catalog.json`
- 调高候选池规模

### 8.3 split 分布不理想

处理方式：

- 调整 `split-ratios`
- 提高总样本规模（小样本时分布抖动更大）

## 9. 复现清单

1. 安装依赖。
2. 准备配置文件（基础配置 + 行业目录）。
3. 运行 `run_trajectory_pipeline.py`。
4. 用 `inspect_q_dataset.py` 检查指标。
5. 查看 summary 中的 artifacts 哈希，固化版本。

## 10. 版本说明

当前主线版本：

- `pipeline_version = trajectory_pipeline_v2`

相对旧版提升：

- 轨迹质量评分门禁
- 更细执行与纠偏字段
- 更强去重和均衡
- 全局配额切分
- 产物哈希清单

