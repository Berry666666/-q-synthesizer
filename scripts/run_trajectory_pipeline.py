#!/usr/bin/env python3
import argparse
import copy
import hashlib
import json
import math
import os
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from q_synth import QSynthesizer, load_config  # noqa: E402

PIPELINE_VERSION = "trajectory_pipeline_v2"

COMMON_SOFT_PREFERENCES = [
    "优先选择可复用且可模板化的动作。",
    "优先减少跨团队等待时间。",
    "优先保留扩展与回滚空间。",
    "优先提高关键链路可观测性。",
]

COMMON_DELIVERABLES = [
    "给出阶段里程碑、负责人和验收标准。",
    "提供风险清单、触发信号与纠偏动作。",
    "明确工具调用节奏与数据观测口径。",
    "给出回滚条件、止损方案和复盘机制。",
    "给出资源分配表与优先级重排原则。",
]

COMMON_NOISE = [
    "近期管理层在讨论品牌传播方案。",
    "行政正在推进办公流程优化。",
    "团队希望统一跨部门周会模板。",
    "公司在征集下季度分享主题。",
]

COMMON_HARD = [
    "所有关键动作必须保留审计日志。",
    "高风险节点必须有人工复核兜底。",
    "任何阶段都必须保留回滚路径。",
    "关键数据口径变更必须先评审后执行。",
]

COMMON_EVENTS = [
    "关键依赖服务出现间歇性超时。",
    "核心成员临时离岗导致排期波动。",
    "外部政策或规则在中途发生变更。",
    "高峰流量超预期导致容量紧张。",
]

RISK_SIGNALS = [
    "预算消耗偏离阈值",
    "关键路径延迟累计",
    "高优先级告警持续触发",
    "跨团队依赖阻塞",
    "模型或系统性能退化",
]

STATUS_POOL = ["success", "partial", "recovered"]
STATUS_WEIGHTS = [0.70, 0.20, 0.10]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate fine-grained long-horizon trajectory dataset pipeline.")
    p.add_argument("--config", default=str(ROOT / "configs" / "default_profiles.json"))
    p.add_argument("--industry-catalog", default=str(ROOT / "configs" / "industry_catalog.json"))
    p.add_argument("--profile", default="mixed", choices=["easy", "medium", "hard", "expert", "mixed"])
    p.add_argument("--num-samples", type=int, default=300)
    p.add_argument("--candidate-multiplier", type=int, default=5)
    p.add_argument("--output", default=str(ROOT / "data" / "trajectory_dataset.jsonl"))
    p.add_argument("--summary-output", default="")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--q-only", action="store_true")
    p.add_argument("--require-A", action="store_true", help="Only keep records with oracle A.")
    p.add_argument("--min-quality", type=float, default=None)
    p.add_argument("--max-retries-per-sample", type=int, default=60)
    p.add_argument("--max-generation-attempts", type=int, default=30000)

    p.add_argument("--q-generation-mode", type=str, default="rule", choices=["rule", "llm", "hybrid"])
    p.add_argument("--llm-base-url", type=str, default="", help="OpenAI-compatible base URL, e.g. https://api.openai.com/v1")
    p.add_argument("--llm-model", type=str, default="", help="Model name for LLM Q generation")
    p.add_argument("--llm-api-key", type=str, default="", help="API key (if empty, read from --llm-api-key-env)")
    p.add_argument("--llm-api-key-env", type=str, default="OPENAI_API_KEY")
    p.add_argument("--llm-timeout-sec", type=float, default=60.0)
    p.add_argument("--llm-temperature", type=float, default=0.9)
    p.add_argument("--llm-top-p", type=float, default=0.95)
    p.add_argument("--llm-max-tokens", type=int, default=2200)
    p.add_argument("--llm-max-retries", type=int, default=2)
    p.add_argument("--llm-fallback-to-rule", action="store_true", help="Fallback to template Q when LLM call fails")

    p.add_argument("--min-industries", type=int, default=10)
    p.add_argument("--max-per-industry", type=int, default=0, help="0 means auto cap.")
    p.add_argument("--max-per-focus", type=int, default=0, help="0 means auto cap.")

    p.add_argument("--near-dup-threshold", type=float, default=0.88)
    p.add_argument("--min-plan-phases", type=int, default=3)
    p.add_argument("--min-tool-steps", type=int, default=10)
    p.add_argument("--min-corrections", type=int, default=1)
    p.add_argument("--min-trajectory-score", type=float, default=0.68)
    p.add_argument("--min-tool-coverage", type=float, default=0.35)
    p.add_argument("--min-phase-execution-coverage", type=float, default=0.70)
    p.add_argument("--max-tools-per-subgoal", type=int, default=2)
    p.add_argument("--max-corrections", type=int, default=5)

    p.add_argument("--split-ratios", default="0.8,0.1,0.1", help="Comma-separated train,val,test ratios.")
    p.add_argument("--disable-splits", action="store_true")
    p.add_argument("--save-stage-artifacts", action="store_true")
    p.add_argument("--stage-dir", default="")
    return p.parse_args()


def parse_split_ratios(raw: str) -> Tuple[float, float, float]:
    parts = [x.strip() for x in raw.split(",") if x.strip()]
    if len(parts) != 3:
        raise ValueError("--split-ratios must have exactly 3 values, e.g. 0.8,0.1,0.1")

    vals = [float(x) for x in parts]
    if any(v < 0 for v in vals):
        raise ValueError("--split-ratios values must be non-negative")

    total = sum(vals)
    if total <= 0:
        raise ValueError("--split-ratios sum must be positive")

    return vals[0] / total, vals[1] / total, vals[2] / total


def build_llm_cfg(args: argparse.Namespace) -> Dict[str, Any]:
    mode = str(args.q_generation_mode or "rule").lower()
    if mode == "rule":
        return {}

    api_key = str(args.llm_api_key or "").strip()
    if not api_key:
        api_key = str(os.getenv(str(args.llm_api_key_env), "")).strip()

    base_url = str(args.llm_base_url or "").strip()
    model = str(args.llm_model or "").strip()

    missing = []
    if not base_url:
        missing.append("--llm-base-url")
    if not model:
        missing.append("--llm-model")
    if not api_key:
        missing.append("--llm-api-key or env " + str(args.llm_api_key_env))

    if missing:
        raise ValueError("LLM mode requires: " + ", ".join(missing))

    return {
        "base_url": base_url.rstrip("/"),
        "model": model,
        "api_key": api_key,
        "timeout_sec": float(args.llm_timeout_sec),
        "temperature": float(args.llm_temperature),
        "top_p": float(args.llm_top_p),
        "max_tokens": int(args.llm_max_tokens),
        "max_retries": int(args.llm_max_retries),
        "fallback_to_rule": bool(args.llm_fallback_to_rule),
    }


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[^\u4e00-\u9fff0-9a-z]", "", text)
    return text


def build_shingles(text: str, k: int = 4, max_chars: int = 1800) -> set:
    text = normalize_text(text)[:max_chars]
    if len(text) <= k:
        return {text} if text else set()
    return {text[i : i + k] for i in range(len(text) - k + 1)}


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    union = len(a | b)
    if union == 0:
        return 0.0
    return len(a & b) / union


def load_catalog(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Industry catalog not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    industries = data.get("industries", [])
    if not isinstance(industries, list):
        raise ValueError("industry_catalog.json must contain key 'industries' as list")
    return industries


def to_domain(seed: Dict[str, Any]) -> Dict[str, Any]:
    name = str(seed.get("name", "未命名行业"))
    hard_constraints = list(seed.get("hard_constraint_pool", []))
    events = list(seed.get("dynamic_event_pool", []))

    domain = {
        "name": name,
        "org_pool": list(seed.get("org_pool", [])),
        "focus_pool": list(seed.get("focus_pool", [])),
        "background_templates": [
            "你负责{org}的关键项目，当前主线是{focus}，并且需要在有限资源下保持稳定、合规和可追踪。",
            "{org}正在推进{focus}，项目跨团队依赖复杂，要求你给出可执行且可纠偏的长程方案。",
            "围绕{focus}，{org}近期暴露出执行断点与资源冲突，需要你重构端到端推进路径。",
        ],
        "goal_templates": [
            "制定可执行的长程规划，覆盖准备、执行、监控、纠偏与复盘。",
            "输出兼顾效率、稳定与风险约束的阶段化推进方案。",
            "给出可落地的跨团队协同路线，并明确每阶段门禁与回滚策略。",
        ],
        "tool_pool": list(seed.get("tool_pool", [])),
        "hard_constraint_pool": hard_constraints + COMMON_HARD,
        "soft_preference_pool": list(seed.get("soft_preference_pool", [])) + COMMON_SOFT_PREFERENCES,
        "dynamic_event_pool": events + COMMON_EVENTS,
        "deliverable_requirements": list(seed.get("deliverable_requirements", [])) + COMMON_DELIVERABLES,
        "noise_pool": list(seed.get("noise_pool", [])) + COMMON_NOISE,
    }

    required_keys = ["org_pool", "focus_pool", "tool_pool"]
    for k in required_keys:
        if len(domain[k]) < 4:
            raise ValueError(f"Industry '{name}' has too few items in {k}; requires >= 4")

    return domain


def merge_domains(base_domains: List[Dict[str, Any]], extra_domains: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged = copy.deepcopy(base_domains)
    names = {str(d.get("name", "")) for d in merged}
    for d in extra_domains:
        if d["name"] not in names:
            merged.append(d)
            names.add(d["name"])
    return merged


def pick_profile(profile_arg: str, rnd: random.Random) -> str:
    if profile_arg != "mixed":
        return profile_arg
    pool = ["medium", "hard", "expert"]
    weights = [0.20, 0.55, 0.25]
    x = rnd.random()
    cum = 0.0
    for p, w in zip(pool, weights):
        cum += w
        if x <= cum:
            return p
    return pool[-1]


def weighted_pick(items: List[str], weights: List[float], rnd: random.Random) -> str:
    x = rnd.random()
    cum = 0.0
    for item, w in zip(items, weights):
        cum += w
        if x <= cum:
            return item
    return items[-1]


def phase_tool_union(nodes: List[Dict[str, Any]]) -> List[str]:
    tools = []
    seen = set()
    for n in nodes:
        for t in n.get("required_tools", []):
            if t not in seen:
                seen.add(t)
                tools.append(t)
    return tools


def ensure_week_window(start: int, end: int, max_week: int) -> Tuple[int, int]:
    start = max(1, start)
    end = min(max_week, max(start, end))
    return start, end


def build_trajectory(record: Dict[str, Any], rnd: random.Random, max_tools_per_subgoal: int, max_corrections: int) -> Dict[str, Any]:
    meta = record.get("meta", {})
    context = record.get("context", {})
    nodes = list(record.get("task_graph", {}).get("nodes", []))

    nodes_by_layer: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for n in nodes:
        nodes_by_layer[int(n.get("layer", 0))].append(n)

    sorted_layers = sorted(nodes_by_layer.keys())
    if not sorted_layers:
        sorted_layers = [0]

    timeline_weeks = int(meta.get("timeline_weeks", 8))
    phase_count = len(sorted_layers)
    phase_span = max(1, timeline_weeks // max(1, phase_count))

    # 1) Long-range planning block.
    long_range_plan = []
    phase_to_week: Dict[str, Tuple[int, int]] = {}
    for i, layer in enumerate(sorted_layers):
        phase_nodes = nodes_by_layer[layer]
        start_week = i * phase_span + 1
        end_week = timeline_weeks if i == phase_count - 1 else (i + 1) * phase_span
        start_week, end_week = ensure_week_window(start_week, end_week, timeline_weeks)

        phase_id = f"P{i + 1:02d}"
        phase_tools = phase_tool_union(phase_nodes)
        if not phase_tools:
            phase_tools = list(context.get("available_tools", []))[:2]

        milestones = [n.get("title", "未命名子任务") for n in phase_nodes[:3]]
        risk_signal = rnd.choice(RISK_SIGNALS)

        long_range_plan.append(
            {
                "phase_id": phase_id,
                "phase_goal": f"推进第{i + 1}阶段里程碑，收敛关键不确定性并保障约束满足。",
                "weeks": [start_week, end_week],
                "subgoal_ids": [n.get("id", "") for n in phase_nodes],
                "key_tools": phase_tools,
                "milestones": milestones,
                "risk_signal": risk_signal,
                "entry_criteria": "上阶段关键依赖满足，资源与权限完成校验。",
                "exit_criteria": "阶段里程碑完成且风险状态可控，允许进入下一阶段。",
            }
        )
        phase_to_week[phase_id] = (start_week, end_week)

    # 2) Tool execution trace.
    tool_execution = []
    step_idx = 1
    phase_step_ids: Dict[str, List[str]] = defaultdict(list)
    all_tools = list(context.get("available_tools", []))

    for i, layer in enumerate(sorted_layers):
        phase_nodes = nodes_by_layer[layer]
        phase_id = f"P{i + 1:02d}"

        for node in phase_nodes:
            tools = list(node.get("required_tools", []))
            if not tools:
                tools = all_tools[:]
            if not tools:
                continue

            use_cnt = min(max(1, max_tools_per_subgoal), len(tools))
            chosen = tools[:use_cnt]
            for tool in chosen:
                status = weighted_pick(STATUS_POOL, STATUS_WEIGHTS, rnd)
                observed_signal = {
                    "success": "关键指标达到阶段阈值，子目标进入可验收状态。",
                    "partial": "结果部分达标，仍有关键约束未完全满足。",
                    "recovered": "执行过程触发异常，已通过备选路径恢复。",
                }[status]

                step = {
                    "step_id": f"TX-{step_idx:03d}",
                    "phase_id": phase_id,
                    "subgoal_id": node.get("id", ""),
                    "tool": tool,
                    "intent": node.get("title", "执行子任务"),
                    "action": f"调用{tool}完成子任务推进，并同步记录关键指标。",
                    "expected_signal": "获得可验证的中间结果，并更新风险状态。",
                    "observed_signal": observed_signal,
                    "status": status,
                    "evidence_fields": ["owner", "timestamp", "input_snapshot", "output_metric"],
                }
                tool_execution.append(step)
                phase_step_ids[phase_id].append(step["step_id"])
                step_idx += 1

                # If partial, add one follow-up remediation step.
                if status == "partial":
                    backup_tool = None
                    for t in all_tools:
                        if t != tool:
                            backup_tool = t
                            break
                    if backup_tool is None:
                        backup_tool = "人工复核流程"

                    step = {
                        "step_id": f"TX-{step_idx:03d}",
                        "phase_id": phase_id,
                        "subgoal_id": node.get("id", ""),
                        "tool": backup_tool,
                        "intent": f"修复{node.get('title', '执行子任务')}中的偏差",
                        "action": f"通过{backup_tool}补充校验并闭环问题。",
                        "expected_signal": "偏差收敛，风险回到阈值内。",
                        "observed_signal": "补救动作完成，阶段状态恢复。",
                        "status": "success",
                        "evidence_fields": ["owner", "timestamp", "fix_plan", "verification_result"],
                    }
                    tool_execution.append(step)
                    phase_step_ids[phase_id].append(step["step_id"])
                    step_idx += 1

    # 3) Correction trace.
    dynamic_events = list(context.get("dynamic_events", []))
    if not dynamic_events:
        dynamic_events = [
            {
                "id": "EV-00",
                "trigger_phase": 1,
                "description": "出现跨团队阻塞与资源冲突。",
                "impact_level": "medium",
                "required_action": "重排优先级并启用应急容量。",
            }
        ]

    correction_trace = []
    corr_idx = 1
    for ev in dynamic_events[: max(1, max_corrections)]:
        trigger_phase = int(ev.get("trigger_phase", 1))
        phase_id = f"P{max(1, trigger_phase):02d}"
        impact = str(ev.get("impact_level", "medium"))

        affected_steps = phase_step_ids.get(phase_id, [])[:3]
        if not affected_steps:
            # fallback: attach from any phase
            for k in sorted(phase_step_ids.keys()):
                affected_steps = phase_step_ids[k][:2]
                if affected_steps:
                    phase_id = k
                    break

        correction_trace.append(
            {
                "correction_id": f"CR-{corr_idx:03d}",
                "trigger_event": ev.get("description", "突发事件"),
                "phase_id": phase_id,
                "impact_level": impact,
                "diagnosis": "定位偏差来源，区分容量瓶颈、依赖阻塞与约束冲突。",
                "affected_step_ids": affected_steps,
                "replan_actions": [
                    "冻结非关键任务并释放资源给关键路径。",
                    "更新里程碑顺序，优先保护硬约束与主目标。",
                    "增加监控频率并设置异常升级阈值。",
                ],
                "rollback_guard": "若纠偏后关键指标连续两个观测窗口未恢复，立即回退到前一稳定版本。",
                "verification": "检查阶段达成率、预算消耗与风险敞口是否回归阈值内。",
                "window_weeks": list(phase_to_week.get(phase_id, (1, timeline_weeks))),
            }
        )
        corr_idx += 1

    # 4) Conversation turns stitched from planning/execution/correction.
    turns = [
        {
            "turn_id": "T-001",
            "role": "planner",
            "type": "long_range_planning",
            "content": f"已形成{len(long_range_plan)}阶段长程计划，覆盖关键依赖、工具与里程碑。",
        }
    ]

    t_idx = 2
    for p in long_range_plan[: min(6, len(long_range_plan))]:
        turns.append(
            {
                "turn_id": f"T-{t_idx:03d}",
                "role": "planner",
                "type": "phase_brief",
                "content": f"{p['phase_id']}目标={p['phase_goal']}，关键信号={p['risk_signal']}。",
                "ref_phase_id": p["phase_id"],
            }
        )
        t_idx += 1

    for tx in tool_execution[: min(12, len(tool_execution))]:
        turns.append(
            {
                "turn_id": f"T-{t_idx:03d}",
                "role": "executor",
                "type": "tool_execution",
                "content": f"在{tx['phase_id']}执行{tx['tool']}，状态={tx['status']}。",
                "tool": tx["tool"],
                "ref_step_id": tx["step_id"],
            }
        )
        t_idx += 1

    for cr in correction_trace[: min(4, len(correction_trace))]:
        turns.append(
            {
                "turn_id": f"T-{t_idx:03d}",
                "role": "critic",
                "type": "trajectory_correction",
                "content": f"触发事件: {cr['trigger_event']}，执行纠偏并重排优先级。",
                "ref_correction_id": cr["correction_id"],
            }
        )
        t_idx += 1
        turns.append(
            {
                "turn_id": f"T-{t_idx:03d}",
                "role": "planner",
                "type": "replan_decision",
                "content": "已根据纠偏结果更新阶段门禁与资源配比，继续推进主路径。",
                "ref_correction_id": cr["correction_id"],
            }
        )
        t_idx += 1

    return {
        "schema": "long_horizon_trajectory_v2",
        "pipeline_version": PIPELINE_VERSION,
        "long_range_plan": long_range_plan,
        "tool_execution": tool_execution,
        "correction_trace": correction_trace,
        "turns": turns,
    }


def trajectory_metrics(record: Dict[str, Any], traj: Dict[str, Any]) -> Dict[str, Any]:
    plan_phases = len(traj.get("long_range_plan", []))
    tool_steps = len(traj.get("tool_execution", []))
    correction_steps = len(traj.get("correction_trace", []))
    turns = len(traj.get("turns", []))

    available_tools = list(record.get("context", {}).get("available_tools", []))
    used_tools = {str(x.get("tool", "")) for x in traj.get("tool_execution", []) if x.get("tool")}
    tool_coverage = len(used_tools) / len(available_tools) if available_tools else 1.0

    plan_phase_ids = {str(x.get("phase_id", "")) for x in traj.get("long_range_plan", [])}
    exec_phase_ids = {str(x.get("phase_id", "")) for x in traj.get("tool_execution", [])}
    if plan_phase_ids:
        phase_execution_coverage = len(exec_phase_ids & plan_phase_ids) / len(plan_phase_ids)
    else:
        phase_execution_coverage = 0.0

    correction_with_actions = 0
    correction_with_links = 0
    for c in traj.get("correction_trace", []):
        if c.get("replan_actions"):
            correction_with_actions += 1
        if c.get("affected_step_ids"):
            correction_with_links += 1

    correction_action_coverage = correction_with_actions / correction_steps if correction_steps else 0.0
    correction_link_coverage = correction_with_links / correction_steps if correction_steps else 0.0

    return {
        "plan_phases": plan_phases,
        "tool_steps": tool_steps,
        "correction_steps": correction_steps,
        "turns": turns,
        "tool_coverage": round(tool_coverage, 4),
        "phase_execution_coverage": round(phase_execution_coverage, 4),
        "correction_action_coverage": round(correction_action_coverage, 4),
        "correction_link_coverage": round(correction_link_coverage, 4),
    }


def trajectory_quality(metrics: Dict[str, Any]) -> Dict[str, Any]:
    plan_phases = int(metrics.get("plan_phases", 0))
    tool_steps = int(metrics.get("tool_steps", 0))
    correction_steps = int(metrics.get("correction_steps", 0))
    turns = int(metrics.get("turns", 0))

    s_phase_depth = min(1.0, plan_phases / 6.0)
    s_exec_density = min(1.0, tool_steps / max(1.0, plan_phases * 3.0))
    s_correction_density = min(1.0, correction_steps / max(1.0, plan_phases / 2.0))
    s_tool_coverage = float(metrics.get("tool_coverage", 0.0))
    s_phase_exec = float(metrics.get("phase_execution_coverage", 0.0))
    s_turn_richness = min(1.0, turns / 14.0)
    s_corr_link = float(metrics.get("correction_link_coverage", 0.0))

    score = (
        s_phase_depth * 0.16
        + s_exec_density * 0.22
        + s_correction_density * 0.18
        + s_tool_coverage * 0.14
        + s_phase_exec * 0.14
        + s_turn_richness * 0.10
        + s_corr_link * 0.06
    )

    return {
        "score": round(max(0.0, min(1.0, score)), 4),
        "components": {
            "phase_depth": round(s_phase_depth, 4),
            "execution_density": round(s_exec_density, 4),
            "correction_density": round(s_correction_density, 4),
            "tool_coverage": round(s_tool_coverage, 4),
            "phase_execution_coverage": round(s_phase_exec, 4),
            "turn_richness": round(s_turn_richness, 4),
            "correction_link_coverage": round(s_corr_link, 4),
        },
    }


def pass_trajectory_gate(metrics: Dict[str, Any], t_quality: Dict[str, Any], args: argparse.Namespace) -> bool:
    return (
        int(metrics.get("plan_phases", 0)) >= args.min_plan_phases
        and int(metrics.get("tool_steps", 0)) >= args.min_tool_steps
        and int(metrics.get("correction_steps", 0)) >= args.min_corrections
        and float(metrics.get("tool_coverage", 0.0)) >= args.min_tool_coverage
        and float(metrics.get("phase_execution_coverage", 0.0)) >= args.min_phase_execution_coverage
        and float(t_quality.get("score", 0.0)) >= args.min_trajectory_score
    )


def dedup_text(record: Dict[str, Any]) -> str:
    q = str(record.get("Q", ""))
    traj = record.get("trajectory", {})
    phase_text = " ".join(str(x.get("phase_goal", "")) for x in traj.get("long_range_plan", [])[:6])
    tool_text = " ".join(str(x.get("tool", "")) for x in traj.get("tool_execution", [])[:16])
    corr_text = " ".join(str(x.get("trigger_event", "")) for x in traj.get("correction_trace", [])[:4])
    return " ".join([q, phase_text, tool_text, corr_text])


def signature_hash(record: Dict[str, Any]) -> str:
    text = dedup_text(record)
    return hashlib.sha1(normalize_text(text).encode("utf-8")).hexdigest()


def allocate_counts(total: int, ratios: Tuple[float, float, float]) -> List[int]:
    base = [int(math.floor(total * r)) for r in ratios]
    remain = total - sum(base)

    frac = [(i, total * ratios[i] - base[i]) for i in range(3)]
    frac.sort(key=lambda x: x[1], reverse=True)
    for i in range(remain):
        base[frac[i % 3][0]] += 1
    return base


def split_records(records: List[Dict[str, Any]], ratios: Tuple[float, float, float], seed: int) -> Dict[str, List[Dict[str, Any]]]:
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for rec in records:
        key = (str(rec.get("domain", "unknown")), str(rec.get("profile", "unknown")))
        groups[key].append(rec)

    rnd = random.Random(seed)
    splits = {"train": [], "val": [], "test": []}
    target_train, target_val, target_test = allocate_counts(len(records), ratios)
    targets = {"train": target_train, "val": target_val, "test": target_test}

    # group-local counters to keep each group spread across splits when possible.
    group_split_counts: Dict[Tuple[str, str], Dict[str, int]] = defaultdict(lambda: {"train": 0, "val": 0, "test": 0})

    items_list = list(groups.items())
    items_list.sort(key=lambda x: len(x[1]), reverse=True)

    for key, items in items_list:
        rnd.shuffle(items)
        for rec in items:
            remaining = {k: targets[k] - len(splits[k]) for k in targets.keys()}
            available = [k for k, v in remaining.items() if v > 0]
            if not available:
                available = ["train", "val", "test"]

            def rank(split_name: str) -> Tuple[float, float, float]:
                rem = float(remaining.get(split_name, 0))
                grp = float(group_split_counts[key][split_name])
                return rem, -grp, rnd.random()

            chosen = max(available, key=rank)
            splits[chosen].append(rec)
            group_split_counts[key][chosen] += 1

    rnd.shuffle(splits["train"])
    rnd.shuffle(splits["val"])
    rnd.shuffle(splits["test"])
    return splits


def write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def dedup_and_balance(candidates: List[Dict[str, Any]], args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    if not candidates:
        return [], {}

    candidates.sort(
        key=lambda x: (float(x.get("_combined_score", 0.0)), float(x.get("quality", {}).get("score", 0.0))),
        reverse=True,
    )

    by_domain: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for rec in candidates:
        by_domain[str(rec.get("domain", "unknown"))].append(rec)

    domain_keys = sorted(by_domain.keys(), key=lambda d: by_domain[d][0].get("_combined_score", 0.0), reverse=True)
    min_industries = min(args.min_industries, len(domain_keys), args.num_samples)

    domain_cap = args.max_per_industry
    if domain_cap <= 0:
        domain_cap = math.ceil(args.num_samples / max(1, min_industries)) + 2

    focus_cap = args.max_per_focus
    if focus_cap <= 0:
        focus_cap = max(4, math.ceil(args.num_samples / 12))

    selected: List[Dict[str, Any]] = []
    domain_counter: Counter = Counter()
    focus_counter: Counter = Counter()

    selected_hash = set()
    selected_shingles: List[set] = []
    skipped = Counter()

    def can_add(rec: Dict[str, Any]) -> bool:
        sig = signature_hash(rec)
        if sig in selected_hash:
            skipped["exact_dup"] += 1
            return False

        txt = dedup_text(rec)
        sh = build_shingles(txt)
        for prev in selected_shingles:
            if jaccard(sh, prev) >= args.near_dup_threshold:
                skipped["near_dup"] += 1
                return False

        d = str(rec.get("domain", "unknown"))
        if domain_counter[d] >= domain_cap:
            skipped["domain_cap"] += 1
            return False

        focus = str(rec.get("meta", {}).get("focus", "unknown"))
        if focus_counter[focus] >= focus_cap:
            skipped["focus_cap"] += 1
            return False

        selected_hash.add(sig)
        selected_shingles.append(sh)
        domain_counter[d] += 1
        focus_counter[focus] += 1
        return True

    # Pass 1: enforce minimum industry coverage.
    covered = set()
    for d in domain_keys:
        if len(covered) >= min_industries:
            break
        queue = by_domain[d]
        i = 0
        while i < len(queue):
            rec = queue[i]
            if can_add(rec):
                selected.append(rec)
                covered.add(d)
                queue.pop(i)
                break
            i += 1

    # Pass 2: balanced round-robin fill.
    while len(selected) < args.num_samples:
        progressed = False
        for d in domain_keys:
            if len(selected) >= args.num_samples:
                break
            queue = by_domain[d]
            i = 0
            while i < len(queue):
                rec = queue[i]
                if can_add(rec):
                    selected.append(rec)
                    queue.pop(i)
                    progressed = True
                    break
                i += 1

        if not progressed:
            pending = sum(len(v) for v in by_domain.values())
            if pending > 0 and domain_cap < args.num_samples:
                domain_cap += 1
                skipped["domain_cap_relax"] += 1
                continue
            break

    return selected, dict(skipped)


def build_distribution(records: List[Dict[str, Any]], key_path: Tuple[str, ...]) -> Dict[str, int]:
    c = Counter()
    for rec in records:
        cur: Any = rec
        for k in key_path:
            if isinstance(cur, dict):
                cur = cur.get(k)
            else:
                cur = None
                break
        c[str(cur if cur is not None else "unknown")] += 1
    return dict(c)


def stage_dir_path(args: argparse.Namespace, output_path: Path) -> Path:
    if args.stage_dir:
        return Path(args.stage_dir)
    return output_path.parent / f"stages_{output_path.stem}"


def main() -> None:
    args = parse_args()

    if args.q_only and args.require_A:
        raise ValueError("--q-only 与 --require-A 不能同时使用")

    llm_cfg = build_llm_cfg(args)
    split_ratios = parse_split_ratios(args.split_ratios)
    rnd = random.Random(args.seed)

    base_cfg = load_config(args.config)
    catalog = load_catalog(args.industry_catalog)
    extra_domains = [to_domain(item) for item in catalog]

    cfg = copy.deepcopy(base_cfg)
    cfg["domains"] = merge_domains(base_cfg.get("domains", []), extra_domains)

    if args.min_industries > len(cfg.get("domains", [])):
        raise ValueError(
            f"--min-industries={args.min_industries} exceeds available domains={len(cfg.get('domains', []))}"
        )

    synth = QSynthesizer(cfg, seed=args.seed)

    target_candidates = max(args.num_samples * args.candidate_multiplier, args.num_samples)
    candidates: List[Dict[str, Any]] = []
    attempts = 0
    profile_counter = Counter()

    rejected_quality = 0
    rejected_trajectory = 0

    while len(candidates) < target_candidates and attempts < args.max_generation_attempts:
        attempts += 1
        profile = pick_profile(args.profile, rnd)
        quality_threshold = synth.get_quality_threshold(profile, args.min_quality)

        accepted = None
        for _ in range(args.max_retries_per_sample):
            rec = synth.generate_one(
                profile=profile,
                q_only=args.q_only,
                q_generation_mode=args.q_generation_mode,
                llm_cfg=llm_cfg,
            )
            score = float(rec.get("quality", {}).get("score", 0.0))
            feas = rec.get("quality", {}).get("feasibility", {})
            if score >= quality_threshold and feas.get("feasible_hours") and feas.get("feasible_budget"):
                accepted = rec
                break

        if accepted is None:
            rejected_quality += 1
            continue

        if args.require_A and "A" not in accepted:
            rejected_quality += 1
            continue

        traj = build_trajectory(
            accepted,
            rnd=rnd,
            max_tools_per_subgoal=args.max_tools_per_subgoal,
            max_corrections=args.max_corrections,
        )
        metrics = trajectory_metrics(accepted, traj)
        t_quality = trajectory_quality(metrics)

        if not pass_trajectory_gate(metrics, t_quality, args):
            rejected_trajectory += 1
            continue

        accepted["industry"] = accepted.get("domain", "unknown")
        accepted["trajectory"] = traj
        accepted["trajectory_metrics"] = metrics
        accepted["trajectory_quality"] = t_quality
        accepted["pipeline_version"] = PIPELINE_VERSION

        q_score = float(accepted.get("quality", {}).get("score", 0.0))
        t_score = float(t_quality.get("score", 0.0))
        accepted["_combined_score"] = round(q_score * 0.65 + t_score * 0.35, 4)

        candidates.append(accepted)
        profile_counter[profile] += 1

    if not candidates:
        raise RuntimeError("No valid candidates generated. Try lowering quality or trajectory thresholds.")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.save_stage_artifacts:
        sdir = stage_dir_path(args, output_path)
        sdir.mkdir(parents=True, exist_ok=True)
        write_jsonl(sdir / "stage1_candidates.jsonl", candidates)

    selected, skip_stats = dedup_and_balance(candidates, args)

    if len(selected) < args.num_samples:
        raise RuntimeError(
            f"Only selected {len(selected)} samples, below target {args.num_samples}. "
            "Try increasing --candidate-multiplier or lowering filters."
        )

    selected = selected[: args.num_samples]

    for rec in selected:
        rec.pop("_combined_score", None)

    write_jsonl(output_path, selected)

    split_files: Dict[str, str] = {}
    split_stats: Dict[str, Dict[str, Any]] = {}

    if not args.disable_splits:
        splits = split_records(selected, split_ratios, args.seed + 17)

        train_path = output_path.with_suffix(".train.jsonl")
        val_path = output_path.with_suffix(".val.jsonl")
        test_path = output_path.with_suffix(".test.jsonl")

        write_jsonl(train_path, splits["train"])
        write_jsonl(val_path, splits["val"])
        write_jsonl(test_path, splits["test"])

        split_files = {
            "train": str(train_path),
            "val": str(val_path),
            "test": str(test_path),
        }

        split_stats = {
            "train": {
                "rows": len(splits["train"]),
                "industry_distribution": build_distribution(splits["train"], ("domain",)),
                "profile_distribution": build_distribution(splits["train"], ("profile",)),
            },
            "val": {
                "rows": len(splits["val"]),
                "industry_distribution": build_distribution(splits["val"], ("domain",)),
                "profile_distribution": build_distribution(splits["val"], ("profile",)),
            },
            "test": {
                "rows": len(splits["test"]),
                "industry_distribution": build_distribution(splits["test"], ("domain",)),
                "profile_distribution": build_distribution(splits["test"], ("profile",)),
            },
        }

    if args.save_stage_artifacts:
        sdir = stage_dir_path(args, output_path)
        write_jsonl(sdir / "stage2_selected.jsonl", selected)

    summary_path = Path(args.summary_output) if args.summary_output else output_path.with_suffix(".summary.json")

    quality_scores = [float(x.get("quality", {}).get("score", 0.0)) for x in selected]
    t_quality_scores = [float(x.get("trajectory_quality", {}).get("score", 0.0)) for x in selected]
    plan_phases = [int(x.get("trajectory_metrics", {}).get("plan_phases", 0)) for x in selected]
    tool_steps = [int(x.get("trajectory_metrics", {}).get("tool_steps", 0)) for x in selected]
    corr_steps = [int(x.get("trajectory_metrics", {}).get("correction_steps", 0)) for x in selected]
    tool_covs = [float(x.get("trajectory_metrics", {}).get("tool_coverage", 0.0)) for x in selected]
    phase_covs = [float(x.get("trajectory_metrics", {}).get("phase_execution_coverage", 0.0)) for x in selected]

    domain_dist = build_distribution(selected, ("domain",))
    focus_dist = build_distribution(selected, ("meta", "focus"))

    artifact_manifest: Dict[str, Dict[str, Any]] = {}
    tracked_files = [output_path]
    if split_files:
        tracked_files.extend([Path(split_files["train"]), Path(split_files["val"]), Path(split_files["test"])])

    for p in tracked_files:
        artifact_manifest[str(p)] = {
            "rows": sum(1 for _ in p.open("r", encoding="utf-8")),
            "size_bytes": p.stat().st_size,
            "sha256": file_sha256(p),
        }

    summary = {
        "pipeline_version": PIPELINE_VERSION,
        "q_generation": {
            "mode": args.q_generation_mode,
            "llm_base_url": llm_cfg.get("base_url", "") if llm_cfg else "",
            "llm_model": llm_cfg.get("model", "") if llm_cfg else "",
            "llm_timeout_sec": llm_cfg.get("timeout_sec", 0) if llm_cfg else 0,
            "llm_temperature": llm_cfg.get("temperature", 0) if llm_cfg else 0,
            "llm_top_p": llm_cfg.get("top_p", 0) if llm_cfg else 0,
            "llm_max_tokens": llm_cfg.get("max_tokens", 0) if llm_cfg else 0,
            "llm_max_retries": llm_cfg.get("max_retries", 0) if llm_cfg else 0,
            "llm_fallback_to_rule": llm_cfg.get("fallback_to_rule", False) if llm_cfg else False,
            "llm_api_key_env": args.llm_api_key_env,
        },
        "config": args.config,
        "industry_catalog": args.industry_catalog,
        "domains_available": len(cfg.get("domains", [])),
        "samples_requested": args.num_samples,
        "samples_selected": len(selected),
        "candidate_pool_target": target_candidates,
        "candidate_pool_actual": len(candidates),
        "generation_attempts": attempts,
        "profile_arg": args.profile,
        "profile_distribution": dict(profile_counter),
        "quality_avg": round(sum(quality_scores) / len(quality_scores), 4) if quality_scores else 0.0,
        "quality_min": round(min(quality_scores), 4) if quality_scores else 0.0,
        "quality_max": round(max(quality_scores), 4) if quality_scores else 0.0,
        "trajectory_quality_avg": round(sum(t_quality_scores) / len(t_quality_scores), 4) if t_quality_scores else 0.0,
        "trajectory_quality_min": round(min(t_quality_scores), 4) if t_quality_scores else 0.0,
        "trajectory_quality_max": round(max(t_quality_scores), 4) if t_quality_scores else 0.0,
        "avg_plan_phases": round(sum(plan_phases) / len(plan_phases), 2) if plan_phases else 0.0,
        "avg_tool_steps": round(sum(tool_steps) / len(tool_steps), 2) if tool_steps else 0.0,
        "avg_correction_steps": round(sum(corr_steps) / len(corr_steps), 2) if corr_steps else 0.0,
        "avg_tool_coverage": round(sum(tool_covs) / len(tool_covs), 4) if tool_covs else 0.0,
        "avg_phase_execution_coverage": round(sum(phase_covs) / len(phase_covs), 4) if phase_covs else 0.0,
        "unique_industries": len(domain_dist),
        "industry_distribution": domain_dist,
        "focus_distribution": focus_dist,
        "skip_stats": skip_stats,
        "rejected_quality": rejected_quality,
        "rejected_trajectory": rejected_trajectory,
        "filters": {
            "min_quality": args.min_quality,
            "q_generation_mode": args.q_generation_mode,
            "min_plan_phases": args.min_plan_phases,
            "min_tool_steps": args.min_tool_steps,
            "min_corrections": args.min_corrections,
            "min_trajectory_score": args.min_trajectory_score,
            "min_tool_coverage": args.min_tool_coverage,
            "min_phase_execution_coverage": args.min_phase_execution_coverage,
            "near_dup_threshold": args.near_dup_threshold,
            "min_industries": args.min_industries,
            "max_per_industry": args.max_per_industry,
            "max_per_focus": args.max_per_focus,
            "max_tools_per_subgoal": args.max_tools_per_subgoal,
            "max_corrections": args.max_corrections,
        },
        "split_ratios": {
            "train": split_ratios[0],
            "val": split_ratios[1],
            "test": split_ratios[2],
        },
        "split_files": split_files,
        "split_stats": split_stats,
        "artifacts": artifact_manifest,
        "output": str(output_path),
    }

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
