#!/usr/bin/env python3
"""Recompute ThreadWorkspace metrics using the memory-performance workbook contract."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


TABLE_CATEGORIES = (
    (1, "MultiHop"),
    (2, "Temporal"),
    (3, "OpenDomain"),
    (4, "SingleHop"),
)
ALL_CATEGORIES = (*TABLE_CATEGORIES, (5, "Adversarial"))


def normalize(text: Any) -> str:
    return str(text).lower()


def simple_tokens(text: Any) -> list[str]:
    return re.sub(r"[.,!?]", " ", normalize(text)).split()


def f1_score(prediction: Any, reference: Any) -> float:
    prediction_tokens = set(simple_tokens(prediction))
    reference_tokens = set(simple_tokens(reference))
    if not prediction_tokens or not reference_tokens:
        return 0.0
    common = len(prediction_tokens & reference_tokens)
    precision = common / len(prediction_tokens)
    recall = common / len(reference_tokens)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def bleu1_score(prediction: Any, reference: Any) -> float:
    prediction_tokens = simple_tokens(prediction)
    reference_tokens = simple_tokens(reference)
    if not prediction_tokens or not reference_tokens:
        return 0.0
    overlap = sum(
        min(prediction_tokens.count(token), reference_tokens.count(token))
        for token in set(prediction_tokens)
    )
    precision = overlap / len(prediction_tokens)
    brevity = 1.0 if len(prediction_tokens) >= len(reference_tokens) else math.exp(
        1 - len(reference_tokens) / len(prediction_tokens)
    )
    return precision * brevity


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarize(rows: list[dict[str, Any]], token_summary: dict[str, Any]) -> dict[str, Any]:
    buckets: dict[int, dict[str, list[float]]] = defaultdict(
        lambda: {
            "f1": [],
            "bleu1": [],
            "stored_f1": [],
            "stored_bleu1": [],
            "token_cost": [],
            "judge": [],
        }
    )
    for row in rows:
        category = int(row["category"])
        bucket = buckets[category]
        prediction = row.get("answer", "")
        reference = row.get("reference", "")
        bucket["f1"].append(f1_score(prediction, reference))
        bucket["bleu1"].append(bleu1_score(prediction, reference))
        stored = row.get("metrics", {})
        bucket["stored_f1"].append(float(stored.get("f1", 0.0)))
        bucket["stored_bleu1"].append(float(stored.get("bleu1", 0.0)))
        token_cost = row.get("token_cost", row.get("retrieved_context_tokens"))
        if isinstance(token_cost, (int, float)):
            bucket["token_cost"].append(float(token_cost))
        judge_score = stored.get("llm_judge_score")
        if isinstance(judge_score, (int, float)):
            bucket["judge"].append(float(judge_score))

    per_category: dict[str, dict[str, Any]] = {}
    category_names = dict((*TABLE_CATEGORIES, (5, "Adversarial")))
    for category, name in category_names.items():
        bucket = buckets[category]
        per_category[str(category)] = {
            "name": name,
            "count": len(bucket["f1"]),
            "f1": mean(bucket["f1"]),
            "bleu1": mean(bucket["bleu1"]),
            "token_cost_mean": mean(bucket["token_cost"]),
            "llm_judge_score": mean(bucket["judge"]),
            "max_abs_f1_recompute_diff": max(
                (abs(a - b) for a, b in zip(bucket["f1"], bucket["stored_f1"])),
                default=0.0,
            ),
            "max_abs_bleu1_recompute_diff": max(
                (abs(a - b) for a, b in zip(bucket["bleu1"], bucket["stored_bleu1"])),
                default=0.0,
            ),
        }

    four_category_f1 = mean([per_category[str(category)]["f1"] for category, _ in TABLE_CATEGORIES])
    four_category_bleu1 = mean([per_category[str(category)]["bleu1"] for category, _ in TABLE_CATEGORIES])
    question_count = len(rows)
    row_token_costs = [
        float(row.get("token_cost", row.get("retrieved_context_tokens")))
        for row in rows
        if isinstance(row.get("token_cost", row.get("retrieved_context_tokens")), (int, float))
    ]
    row_judge_scores = [
        float(row["metrics"]["llm_judge_score"])
        for row in rows
        if isinstance((row.get("metrics") or {}).get("llm_judge_score"), (int, float))
    ]
    context_summary = token_summary.get("token_cost") or token_summary.get("paper_like_retrieved_context") or {}
    token_total = sum(row_token_costs) if row_token_costs else int(context_summary.get("total_tokens", 0) or 0)
    token_questions = len(row_token_costs) if row_token_costs else int(context_summary.get("questions", 0) or question_count)
    token_average = token_total / token_questions if token_questions else 0.0
    macro_judge = mean([per_category[str(category)]["llm_judge_score"] for category, _ in TABLE_CATEGORIES])
    main_table = {
        name: {
            "f1": round(per_category[str(category)]["f1"] * 100, 2),
            "bleu": round(per_category[str(category)]["bleu1"] * 100, 2),
        }
        for category, name in TABLE_CATEGORIES
    }
    main_table["Average"] = {
        "f1": round(four_category_f1 * 100, 2),
        "bleu": round(four_category_bleu1 * 100, 2),
        "llm_judge_score": round(macro_judge * 100, 2),
    }
    return {
        "questions": question_count,
        "category_mapping": {str(category): name for category, name in (*TABLE_CATEGORIES, (5, "Adversarial"))},
        "per_category": per_category,
        "main_table_percent": main_table,
        "macro_average_f1": four_category_f1,
        "macro_average_bleu1": four_category_bleu1,
        "adversarial_excluded_from_four_category_average": True,
        "token_cost": {
            "definition": "final retrieved context tokens sent to the answer model; QA instructions, completion, maintenance, and judge calls excluded",
            "questions": token_questions,
            "total_tokens": token_total,
            "tokens_per_question": token_average,
            "source": "detailed_results.token_cost" if row_token_costs else "token_summary.token_cost",
        },
        "llm_judge": {
            "definition": "binary semantic correctness score from the matched judge",
            "mean": mean(row_judge_scores),
            "count": len(row_judge_scores),
            "failures": sum(1 for row in rows if row.get("judge_error") or (row.get("metrics") or {}).get("llm_judge_error")),
        },
    }


def render_markdown(report: dict[str, Any], model: str, method: str, elapsed: str) -> str:
    table = report["main_table_percent"]
    headers = ["Model", "Method", "Jop Num"]
    values = [model, method, str(report["questions"])]
    for _, name in TABLE_CATEGORIES:
        headers.extend((f"{name} F1", f"{name} BLEU"))
        values.extend((f"{table[name]['f1']:.2f}", f"{table[name]['bleu']:.2f}"))
    headers.extend(("Average F1", "Average BLEU", "LLM Judge", "Token Cost", "Local Time"))
    values.extend(
        (
            f"{table['Average']['f1']:.2f}",
            f"{table['Average']['bleu']:.2f}",
            f"{table['Average']['llm_judge_score']:.2f}",
            f"{report['token_cost']['tokens_per_question']:.2f} tok/q",
            elapsed or "N/A",
        )
    )
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        "| " + " | ".join(values) + " |",
        "",
        "Metric contract: F1 and smoothed BLEU-1; Average is the macro mean over categories 1-4.",
        "Category 5 (Adversarial) is reported separately and is excluded from that four-category Average.",
        "Token Cost is the final retrieved-context token count sent to the answer model; QA instructions, completion, maintenance, and judge calls are excluded.",
    ]
    adversarial = report["per_category"]["5"]
    lines.append(
        f"Adversarial (n={adversarial['count']}): F1={adversarial['f1'] * 100:.2f}, "
        f"BLEU-1={adversarial['bleu1'] * 100:.2f}, "
        f"LLM Judge={adversarial['llm_judge_score'] * 100:.2f}, "
        f"Token Cost={adversarial['token_cost_mean']:.2f} tok/q."
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--token-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="Qwen3.8-27B")
    parser.add_argument("--method", default="TrawMem")
    parser.add_argument("--elapsed", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_payload = json.loads(args.results.read_text(encoding="utf-8"))
    rows = result_payload.get("detailed_results", [])
    if not rows:
        raise ValueError(f"No detailed_results found in {args.results}")
    token_summary = json.loads(args.token_summary.read_text(encoding="utf-8"))
    report = summarize(rows, token_summary)
    report["model"] = args.model
    report["method"] = args.method
    report["local_time"] = args.elapsed
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "excel_metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "excel_metrics.md").write_text(render_markdown(report, args.model, args.method, args.elapsed), encoding="utf-8")
    print(render_markdown(report, args.model, args.method, args.elapsed), end="")


if __name__ == "__main__":
    main()
