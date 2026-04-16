# Q Synthesizer for Long-Horizon Tasks

该工程用于在服务器上批量合成高质量长程任务问题（Q），可选同时生成“参考长程规划答案（A）”。

详细数据生产管线说明见 `READ.md`。

## 1) 项目结构

- `configs/default_profiles.json`：任务域模板、复杂度档位、质量阈值
- `src/q_synth/synthesizer.py`：Q 合成与质检核心逻辑
- `scripts/generate_q_dataset.py`：批量生成主脚本
- `scripts/generate_shards.py`：分片批量生成脚本（适合大规模）
- `scripts/inspect_q_dataset.py`：结果统计脚本
- `scripts/select_curated_samples.py`：按质量筛选 + 去重 + 导出精选集
- `scripts/run_trajectory_pipeline.py`：精细化长程轨迹管线（规划 + 工具执行 + 纠偏 + 去重均衡 + split）
- `configs/industry_catalog.json`：行业扩展配置（用于提升行业覆盖）
- `data/`：输出目录

`run_trajectory_pipeline.py` 支持两类 Q 生成分支：

- `rule`：规则与模板生成（默认）
- `llm` / `hybrid`：调用外部 OpenAI 兼容 API 生成或重写 Q

## 2) 快速开始

### 2.1 生成 1,000 条 hard 档 QA

python3 scripts/generate_q_dataset.py \
  --profile hard \
  --num-samples 1000 \
  --output data/q_qa_hard_1k.jsonl \
  --seed 42

### 2.2 只生成 Q（不带 A）

python3 scripts/generate_q_dataset.py \
  --profile hard \
  --num-samples 1000 \
  --q-only \
  --output data/q_only_hard_1k.jsonl

### 2.3 按混合难度生成

python3 scripts/generate_q_dataset.py \
  --profile mixed \
  --num-samples 5000 \
  --output data/q_qa_mixed_5k.jsonl

### 2.4 查看统计摘要

python3 scripts/inspect_q_dataset.py --input data/q_qa_hard_1k.jsonl

### 2.5 分片批量生成（大规模）

python3 scripts/generate_shards.py \
  --profile mixed \
  --num-shards 16 \
  --samples-per-shard 10000 \
  --output-dir data/shards \
  --base-seed 2026 \
  --q-only

### 2.6 导出精选 100 条（高质量 + 去重）

python3 scripts/select_curated_samples.py \
  --inputs data/q_qa_hard_1200.jsonl \
  --output data/q_qa_hard_curated_100.jsonl \
  --top-k 100 \
  --min-quality 0.75 \
  --near-dup-threshold 0.88 \
  --max-per-domain 40

python3 scripts/inspect_q_dataset.py --input data/q_qa_hard_curated_100.jsonl

### 2.7 生成轨迹数据（v2精细化pipeline）

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

python3 scripts/inspect_q_dataset.py --input data/trajectory_mixed_300.jsonl

### 2.8 使用外部大模型生成Q（提升泛化）

```bash
python3 scripts/run_trajectory_pipeline.py \
  --q-generation-mode llm \
  --llm-base-url https://api.openai.com/v1 \
  --llm-model gpt-4o-mini \
  --llm-api-key "$OPENAI_API_KEY" \
  --llm-temperature 0.9 \
  --llm-top-p 0.95 \
  --llm-max-tokens 2200 \
  --num-samples 200 \
  --output data/trajectory_llm_200.jsonl
```

如果你希望“先按规则构图，再用 LLM 重写问题文本”，可使用：

```bash
python3 scripts/run_trajectory_pipeline.py \
  --q-generation-mode hybrid \
  --llm-base-url https://api.openai.com/v1 \
  --llm-model gpt-4o-mini \
  --llm-api-key "$OPENAI_API_KEY" \
  --llm-fallback-to-rule \
  --num-samples 200 \
  --output data/trajectory_hybrid_200.jsonl
```

该命令会同时产出：

- `data/trajectory_mixed_300.jsonl`
- `data/trajectory_mixed_300.train.jsonl`
- `data/trajectory_mixed_300.val.jsonl`
- `data/trajectory_mixed_300.test.jsonl`
- `data/trajectory_mixed_300.summary.json`

## 3) 数据格式

每行一个 JSON 对象，典型字段：

- `id`: 样本ID
- `profile`: 难度档位（easy/medium/hard/expert）
- `domain`: 任务域
- `Q`: 复杂任务问题
- `A`: 参考长程规划（当未启用 `--q-only`）
- `quality`: 质量分与分解指标
- `meta`: 样本元信息
- `task_graph`: 子目标图（节点和依赖边）
- `trajectory`: 轨迹结构（`long_range_plan` / `tool_execution` / `correction_trace` / `turns`）
- `trajectory_metrics`: 轨迹可执行性指标（工具覆盖率、阶段执行覆盖率等）
- `trajectory_quality`: 轨迹质量评分（用于门禁筛选）

## 4) 质量保障机制

- 先构造任务图（子目标、依赖、事件、约束）再渲染自然语言
- 进行可解性检查：工时容量、预算可行性
- 进行复杂度评分：子目标数量、依赖深度、约束密度、事件数量、工具多样性
- 进行轨迹质量评分：阶段深度、执行密度、纠偏密度、工具覆盖、阶段执行覆盖
- 去重策略：精确签名 + 近重复Jaccard去重
- 分布控制：行业覆盖约束 + focus上限控制 + round-robin均衡抽样
- 自动输出 train/val/test 三路数据切分与分布统计

## 5) 扩展建议

1. 扩展 `configs/default_profiles.json` 的 domains，增加行业覆盖。
2. 在 `synthesizer.py` 增加“对抗式约束冲突检测”与“语义去重”。
3. 将 `A` 替换为你自己的规划器输出，形成 teacher-forcing 训练语料。
