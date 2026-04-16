# Q 合成器服务器部署说明

## 部署位置

- /root/shared-nvme/q-synthesizer

## 环境要求

- Python >= 3.10
- 无第三方依赖（标准库即可）

## 1) 进入目录

cd /root/shared-nvme/q-synthesizer

## 2) 语法检查

python3 -m py_compile src/q_synth/*.py scripts/*.py

## 3) 快速生成（20条验证）

python3 scripts/generate_q_dataset.py \
  --profile hard \
  --num-samples 20 \
  --output data/q_qa_hard_20.jsonl \
  --seed 42

## 4) 只生成Q（不含A）

python3 scripts/generate_q_dataset.py \
  --profile hard \
  --num-samples 2000 \
  --q-only \
  --output data/q_only_hard_2k.jsonl

## 5) 分片批量生成（推荐）

python3 scripts/generate_shards.py \
  --profile mixed \
  --num-shards 16 \
  --samples-per-shard 10000 \
  --output-dir data/shards \
  --base-seed 2026 \
  --q-only

说明：
- 每个 shard 独立输出一个 jsonl。
- 汇总信息写入 data/shards/shard_summary.json。

## 6) 质量检查

python3 scripts/inspect_q_dataset.py --input data/q_qa_hard_20.jsonl

## 7) 输出格式

每行一个JSON对象，关键字段：
- id
- profile
- domain
- Q
- A（若未开启 q-only）
- quality（质量分与可解性）
- meta
- task_graph

## 8) 大规模建议（5B-10B token）

1. 先生成 10万条做抽样质检。
2. 通过后扩大到分片并行（每片 1万-5万条）。
3. 先用 q-only 生成主语料，再按比例补充带 A 的教师数据。
4. 每个 shard 固定 seed，确保可复现。
