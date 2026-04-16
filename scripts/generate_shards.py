#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from q_synth import QSynthesizer, load_config  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate sharded Q/QA dataset for large-scale training.")
    p.add_argument("--config", type=str, default=str(ROOT / "configs" / "default_profiles.json"))
    p.add_argument("--profile", type=str, default="hard", choices=["easy", "medium", "hard", "expert", "mixed"])
    p.add_argument("--num-shards", type=int, default=8)
    p.add_argument("--samples-per-shard", type=int, default=5000)
    p.add_argument("--output-dir", type=str, default=str(ROOT / "data" / "shards"))
    p.add_argument("--base-seed", type=int, default=42)
    p.add_argument("--q-only", action="store_true")
    p.add_argument("--min-quality", type=float, default=None)
    p.add_argument("--max-retries-per-sample", type=int, default=60)
    return p


def pick_profile(profile_arg: str, rnd):
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


def main() -> None:
    args = build_arg_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    import random

    overall = {
        "profile_arg": args.profile,
        "num_shards": args.num_shards,
        "samples_per_shard": args.samples_per_shard,
        "total_samples": 0,
        "avg_quality_per_shard": {},
        "files": [],
    }

    for shard_id in range(args.num_shards):
        seed = args.base_seed + shard_id * 997
        rnd = random.Random(seed)
        cfg = load_config(args.config)
        synth = QSynthesizer(cfg, seed=seed)

        shard_path = output_dir / f"q_dataset_shard_{shard_id:03d}.jsonl"
        scores = []

        with shard_path.open("w", encoding="utf-8") as f:
            for i in range(args.samples_per_shard):
                profile = pick_profile(args.profile, rnd)
                threshold = synth.get_quality_threshold(profile, args.min_quality)

                accepted = None
                for _ in range(args.max_retries_per_sample):
                    rec = synth.generate_one(profile=profile, q_only=args.q_only)
                    score = float(rec["quality"]["score"])
                    feas = rec["quality"]["feasibility"]
                    if score >= threshold and feas["feasible_hours"] and feas["feasible_budget"]:
                        accepted = rec
                        break

                if accepted is None:
                    raise RuntimeError(
                        f"Shard {shard_id}: sample {i + 1} failed to meet threshold {threshold}"
                    )

                scores.append(float(accepted["quality"]["score"]))
                f.write(json.dumps(accepted, ensure_ascii=False) + "\n")

        avg_score = sum(scores) / len(scores)
        overall["total_samples"] += len(scores)
        overall["avg_quality_per_shard"][str(shard_id)] = round(avg_score, 4)
        overall["files"].append(str(shard_path))

        print(
            json.dumps(
                {
                    "shard": shard_id,
                    "seed": seed,
                    "samples": len(scores),
                    "avg_quality": round(avg_score, 4),
                    "file": str(shard_path),
                },
                ensure_ascii=False,
            )
        )

    summary_path = output_dir / "shard_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(overall, f, ensure_ascii=False, indent=2)

    print(json.dumps({"summary": str(summary_path), **overall}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
