"""Merge benchmark shards without rerunning the model."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter
from pathlib import Path


def _p95(values: list[float]) -> float | None:
    if not values:
        return None
    return sorted(values)[max(0, math.ceil(0.95 * len(values)) - 1)]


_TOKEN_FIELDS = ("calls", "prompt_tokens", "completion_tokens", "total_tokens")


def _row_identity(row: dict) -> tuple[int, int | str]:
    """Use the indexed benchmark identity before legacy display names."""
    raw_index = row.get("sample_idx")
    if isinstance(raw_index, int) and not isinstance(raw_index, bool):
        return (0, raw_index)
    if isinstance(raw_index, str):
        normalized = raw_index.strip()
        if normalized and normalized.lstrip("-").isdigit():
            return (0, int(normalized))
    return (1, str(row.get("sample_id", "")))


def _row_key(row: dict) -> tuple[tuple[int, int | str], int]:
    return (_row_identity(row), int(row.get("question_index", -1)))


def _sum_buckets(buckets: list[dict]) -> dict[str, int]:
    return {
        field: sum(int(bucket.get(field, 0) or 0) for bucket in buckets)
        for field in _TOKEN_FIELDS
    }


def _merge_token_usage(payloads: list[dict], question_count: int, contexts: list[int]) -> dict:
    """Add token ledgers across shards without dropping stage information."""
    stage_names = ("maintenance", "retrieval", "answer", "judge")

    def bucket_for(payload: dict, name: str) -> dict | None:
        usage = payload.get("summary", {}).get("token_usage", {})
        value = usage.get(name)
        if not isinstance(value, dict):
            return None
        if set(_TOKEN_FIELDS).issubset(value):
            return value
        # LoCoMo keeps its separate judge client's full summary under the
        # top-level ``judge`` key; its candidate_total bucket is the compact
        # form needed for the merged ledger.
        if name == "judge" and set(_TOKEN_FIELDS).issubset(value.get("candidate_total_token_cost", {})):
            return value["candidate_total_token_cost"]
        return None

    stage_buckets = {
        name: [
            bucket
            for payload in payloads
            if (bucket := bucket_for(payload, name)) is not None
        ]
        for name in stage_names
    }
    stages = {
        name: _sum_buckets(buckets)
        for name, buckets in stage_buckets.items()
        if buckets
    }
    retrieval = stages.get("retrieval", {field: 0 for field in _TOKEN_FIELDS})
    answer = stages.get("answer", {field: 0 for field in _TOKEN_FIELDS})
    maintenance = stages.get("maintenance", {field: 0 for field in _TOKEN_FIELDS})
    judge = stages.get("judge", {field: 0 for field in _TOKEN_FIELDS})
    online = {
        field: retrieval[field] + answer[field]
        for field in _TOKEN_FIELDS
    }
    total = {
        field: maintenance[field] + online[field]
        for field in _TOKEN_FIELDS
    }
    usage = {
        "definition": {
            "candidate_online_token_cost": "retrieval + answer; maintenance/build excluded",
            "candidate_total_token_cost": "maintenance + retrieval + answer",
            "token_units": "prompt + completion tokens",
            "merge": "sum of successful calls across all shards",
        },
        "by_stage": stages,
        "maintenance": maintenance,
        "retrieval": retrieval,
        "answer": answer,
        "judge": judge,
        "candidate_online_token_cost": online,
        "candidate_total_token_cost": total,
        "question_count": question_count,
        "candidate_online_average_total_tokens_per_question": (
            online["total_tokens"] / question_count if question_count else None
        ),
        "candidate_total_average_tokens_per_question": (
            total["total_tokens"] / question_count if question_count else None
        ),
    }
    if contexts:
        context_total = sum(contexts)
        context_summary = {
            "definition": "final retrieved context tokens, excluding QA instructions and completion",
            "questions": len(contexts),
            "total_tokens": context_total,
            "average_tokens_per_question": context_total / len(contexts),
        }
        usage["paper_like_retrieved_context"] = context_summary
        usage["token_cost"] = dict(context_summary)
    return usage


def merge(inputs: list[Path], output: Path, benchmark: str) -> None:
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in inputs]
    rows = [row for payload in payloads for row in payload.get("detailed_results", [])]
    rows.sort(key=_row_key)
    status_counts = Counter(
        str(row.get("evaluation_status", "unknown")) for row in rows
    )
    answer_failure_count = sum(
        count
        for status, count in status_counts.items()
        if status in {"answer_error", "question_error"}
    )
    seen: set[tuple[tuple[int, int | str], int]] = set()
    duplicates: list[tuple[tuple[int, int | str], int]] = []
    for row in rows:
        key = _row_key(row)
        if key in seen:
            duplicates.append(key)
        seen.add(key)
    if duplicates:
        preview = ", ".join(f"{sample}/{question}" for sample, question in duplicates[:5])
        raise ValueError(
            "Refusing to merge duplicate questions across shards; "
            f"inspect {preview}"
        )
    build_stats = [stat for payload in payloads for stat in payload.get("summary", {}).get("build_stats", [])]
    metric_names = sorted({name for row in rows for name in (row.get("metrics") or {})})
    metrics = {
        name: statistics.fmean(row["metrics"][name] for row in rows if name in (row.get("metrics") or {}))
        for name in metric_names
        if any(name in (row.get("metrics") or {}) for row in rows)
    }
    category_metrics = {}
    categories = sorted(
        {str(row.get("category")) for row in rows if row.get("category") not in (None, "")},
        key=lambda value: (not value.isdigit(), int(value) if value.isdigit() else value),
    )
    for category in categories:
        category_rows = [row for row in rows if str(row.get("category")) == category]
        category_metric_names = sorted({
            name
            for row in category_rows
            for name in (row.get("metrics") or {})
        })
        category_metrics[category] = {
            "num_questions": len(category_rows),
            "metrics": {
                name: statistics.fmean(
                    row["metrics"][name]
                    for row in category_rows
                    if name in (row.get("metrics") or {})
                )
                for name in category_metric_names
            },
        }
    retrieval = [float(row.get("retrieval_time", 0.0)) for row in rows]
    answer = [float(row.get("answer_time", 0.0)) for row in rows]
    total = [float(row.get("total_time", 0.0)) for row in rows]
    contexts = [row.get("token_cost") for row in rows if row.get("token_cost") is not None]
    calls = [float(row.get("llm_calls_query", 0.0)) for row in rows]
    retrieval_calls = [float(row.get("retrieval_llm_calls", 0.0)) for row in rows]
    judge_calls = [float(row.get("llm_judge_calls", 0.0)) for row in rows]
    memory_sizes = [
        float(stat["memory_size_bytes"])
        for stat in build_stats
        if isinstance(stat.get("memory_size_bytes"), (int, float))
    ]
    token_usage = _merge_token_usage(payloads, len(rows), [int(value) for value in contexts])
    summary = {
        "benchmark": benchmark,
        "num_samples": len(build_stats),
        "num_questions": len(rows),
        "evaluation_status_counts": dict(status_counts),
        "failed_questions": sum(
            count for status, count in status_counts.items() if status != "ok"
        ),
        "answer_success_rate": (
            (len(rows) - answer_failure_count) / len(rows) if rows else 0.0
        ),
        "evaluation_success_rate": (
            status_counts.get("ok", 0) / len(rows) if rows else 0.0
        ),
        "aggregate_metrics": metrics,
        "aggregate_metrics_by_category": category_metrics,
        "avg_retrieval_time": statistics.fmean(retrieval) if retrieval else 0.0,
        "p50_retrieval_time": statistics.median(retrieval) if retrieval else 0.0,
        "p95_retrieval_time": _p95(retrieval),
        "avg_answer_time": statistics.fmean(answer) if answer else 0.0,
        "p50_answer_time": statistics.median(answer) if answer else 0.0,
        "p95_answer_time": _p95(answer),
        "avg_total_query_time": statistics.fmean(total) if total else 0.0,
        "p50_total_query_time": statistics.median(total) if total else 0.0,
        "p95_total_query_time": _p95(total),
        "avg_context_tokens": statistics.fmean(contexts) if contexts else None,
        "avg_query_llm_calls": statistics.fmean(calls) if calls else 0.0,
        "avg_retrieval_llm_calls": statistics.fmean(retrieval_calls) if retrieval_calls else 0.0,
        "avg_judge_llm_calls": statistics.fmean(judge_calls) if judge_calls else 0.0,
        "avg_build_seconds": (
            statistics.fmean(float(stat.get("build_seconds", 0.0)) for stat in build_stats)
            if build_stats else 0.0
        ),
        "avg_memory_size_bytes": statistics.fmean(memory_sizes) if memory_sizes else None,
        "build_stats": build_stats,
        "token_usage": token_usage,
        "protocol": "caption/text-only for Mem-Gallery; no raw image encoder",
        "shard_inputs": [str(path) for path in inputs],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "method": "TrawMem",
                "config": {"benchmark": benchmark},
                "summary": summary,
                "build_stats": build_stats,
                "detailed_results": rows,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("inputs", nargs="+", type=Path)
    args = parser.parse_args()
    if not all(path.is_file() for path in args.inputs):
        missing = [str(path) for path in args.inputs if not path.is_file()]
        raise SystemExit(f"missing shard results: {missing}")
    merge(args.inputs, Path(args.output), args.benchmark)


if __name__ == "__main__":
    main()
