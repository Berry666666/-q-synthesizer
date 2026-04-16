#!/usr/bin/env python3
import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description="Inspect generated Q/QA dataset stats.")
    p.add_argument("--input", type=str, required=True)
    args = p.parse_args()

    path = Path(args.input)
    if not path.exists():
        raise FileNotFoundError(path)

    n = 0
    quality = []
    q_len = []
    a_len = []
    profile = Counter()
    domain = Counter()
    trajectory_rows = 0
    plan_phases = []
    tool_steps = []
    correction_steps = []
    trajectory_quality = []
    tool_coverage = []
    phase_execution_coverage = []
    trajectory_schema = Counter()

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n += 1
            obj = json.loads(line)
            quality.append(float(obj.get("quality", {}).get("score", 0)))
            q_len.append(len(obj.get("Q", "")))
            if "A" in obj:
                a_len.append(len(obj.get("A", "")))
            profile[str(obj.get("profile", ""))] += 1
            domain[str(obj.get("domain", ""))] += 1

            traj = obj.get("trajectory")
            if isinstance(traj, dict):
                trajectory_rows += 1
                plan_phases.append(len(traj.get("long_range_plan", [])))
                tool_steps.append(len(traj.get("tool_execution", [])))
                correction_steps.append(len(traj.get("correction_trace", [])))
                trajectory_schema[str(traj.get("schema", "unknown"))] += 1

                t_quality = obj.get("trajectory_quality", {})
                if isinstance(t_quality, dict):
                    trajectory_quality.append(float(t_quality.get("score", 0.0)))

                t_metrics = obj.get("trajectory_metrics", {})
                if isinstance(t_metrics, dict):
                    tool_coverage.append(float(t_metrics.get("tool_coverage", 0.0)))
                    phase_execution_coverage.append(float(t_metrics.get("phase_execution_coverage", 0.0)))

    summary = {
        "rows": n,
        "avg_quality": round(sum(quality) / n, 4) if n else 0,
        "avg_q_chars": round(sum(q_len) / n, 2) if n else 0,
        "avg_a_chars": round(sum(a_len) / len(a_len), 2) if a_len else 0,
        "profile_distribution": dict(profile),
        "domain_distribution": dict(domain),
        "trajectory_rows": trajectory_rows,
        "avg_plan_phases": round(sum(plan_phases) / len(plan_phases), 2) if plan_phases else 0,
        "avg_tool_steps": round(sum(tool_steps) / len(tool_steps), 2) if tool_steps else 0,
        "avg_correction_steps": round(sum(correction_steps) / len(correction_steps), 2) if correction_steps else 0,
        "avg_trajectory_quality": round(sum(trajectory_quality) / len(trajectory_quality), 4) if trajectory_quality else 0,
        "avg_tool_coverage": round(sum(tool_coverage) / len(tool_coverage), 4) if tool_coverage else 0,
        "avg_phase_execution_coverage": round(sum(phase_execution_coverage) / len(phase_execution_coverage), 4)
        if phase_execution_coverage
        else 0,
        "trajectory_schema_distribution": dict(trajectory_schema),
    }

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
