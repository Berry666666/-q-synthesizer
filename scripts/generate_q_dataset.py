#!/usr/bin/env python3
import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from q_synth import QSynthesizer, load_config  # noqa: E402


def pick_profile(profile_arg: str, rnd_choice_counter: Counter) -> str:
    if profile_arg != "mixed":
        return profile_arg
    pool = ["medium", "hard", "expert"]
    weights = [0.20, 0.55, 0.25]

    # Lightweight weighted choice without external dependency.
    import random

    x = random.random()
    cum = 0.0
    for p, w in zip(pool, weights):
        cum += w
        if x <= cum:
            rnd_choice_counter[p] += 1
            return p
    rnd_choice_counter[pool[-1]] += 1
    return pool[-1]


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate high-quality long-horizon Q (or QA) dataset.")
    p.add_argument("--config", type=str, default=str(ROOT / "configs" / "default_profiles.json"))
    p.add_argument("--profile", type=str, default="hard", choices=["easy", "medium", "hard", "expert", "mixed"])
    p.add_argument("--num-samples", type=int, default=100)
    p.add_argument("--output", type=str, default=str(ROOT / "data" / "q_dataset.jsonl"))
    p.add_argument("--summary-output", type=str, default="")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--q-only", action="store_true", help="Only generate Q, skip oracle A.")
    p.add_argument("--min-quality", type=float, default=None, help="Override profile quality threshold.")
    p.add_argument("--max-retries-per-sample", type=int, default=40)
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    config = load_config(args.config)
    synthesizer = QSynthesizer(config=config, seed=args.seed)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary_path = Path(args.summary_output) if args.summary_output else output_path.with_suffix(".summary.json")

    generated: List[Dict] = []
    profile_counter = Counter()
    decision_counter = Counter()

    import random

    random.seed(args.seed)

    for i in range(args.num_samples):
        target_profile = pick_profile(args.profile, profile_counter)
        threshold = synthesizer.get_quality_threshold(target_profile, args.min_quality)

        accepted = None
        for _ in range(args.max_retries_per_sample):
            rec = synthesizer.generate_one(profile=target_profile, q_only=args.q_only)
            score = float(rec["quality"]["score"])
            feasible_hours = rec["quality"]["feasibility"]["feasible_hours"]
            feasible_budget = rec["quality"]["feasibility"]["feasible_budget"]
            if score >= threshold and feasible_hours and feasible_budget:
                accepted = rec
                break

        if accepted is None:
            raise RuntimeError(
                f"Failed to generate sample #{i + 1} meeting quality threshold {threshold} within {args.max_retries_per_sample} retries"
            )

        generated.append(accepted)
        decision_counter[target_profile] += 1

    with output_path.open("w", encoding="utf-8") as f:
        for rec in generated:
            f.write(synthesizer.to_jsonl_line(rec) + "\n")

    avg_score = sum(float(r["quality"]["score"]) for r in generated) / len(generated)
    avg_len_q = sum(len(r["Q"]) for r in generated) / len(generated)

    summary = {
        "config": args.config,
        "profile_arg": args.profile,
        "samples": len(generated),
        "output": str(output_path),
        "avg_quality_score": round(avg_score, 4),
        "avg_q_chars": round(avg_len_q, 2),
        "profile_distribution": dict(decision_counter),
        "profile_sampling_counter": dict(profile_counter),
    }

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
