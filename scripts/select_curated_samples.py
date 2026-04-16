#!/usr/bin/env python3
import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", "", text)
    # Keep chinese, letters, digits.
    text = re.sub(r"[^\u4e00-\u9fff0-9a-z]", "", text)
    return text


def build_shingles(text: str, k: int = 4, max_chars: int = 1400) -> set:
    text = normalize_text(text)[:max_chars]
    if len(text) <= k:
        return {text} if text else set()
    return {text[i : i + k] for i in range(len(text) - k + 1)}


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    if union == 0:
        return 0.0
    return inter / union


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Select high-quality, deduplicated curated samples.")
    p.add_argument("--inputs", nargs="+", required=True, help="Input JSONL files")
    p.add_argument("--output", required=True, help="Curated output JSONL")
    p.add_argument("--summary-output", default="", help="Summary JSON path")
    p.add_argument("--top-k", type=int, default=100)
    p.add_argument("--min-quality", type=float, default=0.74)
    p.add_argument("--near-dup-threshold", type=float, default=0.88)
    p.add_argument("--max-per-domain", type=int, default=40)
    p.add_argument("--require-A", action="store_true", help="Only keep samples with A field")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    candidates: List[Dict] = []
    source_counter = Counter()

    for inp in args.inputs:
        path = Path(inp)
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                score = float(obj.get("quality", {}).get("score", 0.0))
                if score < args.min_quality:
                    continue
                if args.require_A and "A" not in obj:
                    continue
                obj["_score"] = score
                obj["_source"] = str(path)
                candidates.append(obj)
                source_counter[str(path)] += 1

    if not candidates:
        raise RuntimeError("No candidates found after filtering. Consider lowering --min-quality.")

    # Sort by quality first, then by Q length to prefer richer tasks.
    candidates.sort(key=lambda x: (x["_score"], len(x.get("Q", ""))), reverse=True)

    selected: List[Dict] = []
    selected_exact_hashes = set()
    selected_shingles: List[set] = []
    domain_counter = defaultdict(int)

    skipped_exact_dup = 0
    skipped_near_dup = 0
    skipped_domain_cap = 0

    for item in candidates:
        if len(selected) >= args.top_k:
            break

        domain = str(item.get("domain", "unknown"))
        if domain_counter[domain] >= args.max_per_domain:
            skipped_domain_cap += 1
            continue

        q_text = item.get("Q", "")
        norm = normalize_text(q_text)
        exact_h = hashlib.sha1(norm.encode("utf-8")).hexdigest()
        if exact_h in selected_exact_hashes:
            skipped_exact_dup += 1
            continue

        sh = build_shingles(q_text)
        is_near_dup = False
        for prev in selected_shingles:
            if jaccard(sh, prev) >= args.near_dup_threshold:
                is_near_dup = True
                break
        if is_near_dup:
            skipped_near_dup += 1
            continue

        selected_exact_hashes.add(exact_h)
        selected_shingles.append(sh)
        domain_counter[domain] += 1

        item.pop("_score", None)
        item.pop("_source", None)
        selected.append(item)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for obj in selected:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    if not args.summary_output:
        summary_path = out_path.with_suffix(".summary.json")
    else:
        summary_path = Path(args.summary_output)

    quality_values = [float(x.get("quality", {}).get("score", 0.0)) for x in selected]

    summary = {
        "inputs": args.inputs,
        "filtered_candidates": len(candidates),
        "selected": len(selected),
        "top_k": args.top_k,
        "min_quality": args.min_quality,
        "near_dup_threshold": args.near_dup_threshold,
        "max_per_domain": args.max_per_domain,
        "require_A": args.require_A,
        "skipped_exact_dup": skipped_exact_dup,
        "skipped_near_dup": skipped_near_dup,
        "skipped_domain_cap": skipped_domain_cap,
        "quality_avg": round(sum(quality_values) / len(quality_values), 4) if quality_values else 0.0,
        "quality_min": round(min(quality_values), 4) if quality_values else 0.0,
        "quality_max": round(max(quality_values), 4) if quality_values else 0.0,
        "domain_distribution": dict(Counter(str(x.get("domain", "unknown")) for x in selected)),
    }

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
