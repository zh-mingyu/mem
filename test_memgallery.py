"""TrawMem evaluation for MemGallery.

The LoCoMo evaluator has a deliberately LoCoMo-specific data model. This file
keeps the MemGallery loader separate, but uses the same TrawMem system and
records the same quality/cost fields.
Mem-Gallery is evaluated with the textual/caption protocol: TrawMem currently
does not contain an image encoder, so an image path alone is never treated as
visual evidence.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from main import TrawMemSystem
from trawmem.core.models.memory_entry import Dialogue
from trawmem.core.prompts import (
    BINARY_JUDGE_SYSTEM_PROMPT,
    build_binary_judge_prompt,
)


_WORD_RE = re.compile(r"[A-Za-z0-9]+")


def _tokens(text: Any) -> list[str]:
    return [x.lower() for x in _WORD_RE.findall(str(text or ""))]


def token_f1(prediction: str, reference: str) -> float:
    pred = _tokens(prediction)
    ref = _tokens(reference)
    if not pred or not ref:
        return 0.0
    common = set(pred) & set(ref)
    if not common:
        return 0.0
    precision = len(common) / len(set(pred))
    recall = len(common) / len(set(ref))
    return 2 * precision * recall / (precision + recall)


def bleu1(prediction: str, reference: str) -> float:
    pred = _tokens(prediction)
    ref = _tokens(reference)
    if not pred or not ref:
        return 0.0
    overlap = sum(min(pred.count(token), ref.count(token)) for token in set(pred))
    precision = overlap / len(pred)
    brevity = 1.0 if len(pred) >= len(ref) else math.exp(1 - len(ref) / len(pred))
    return precision * brevity


def directory_size_bytes(path: str | Path) -> int:
    root = Path(path)
    if not root.exists():
        return 0
    if root.is_file():
        return root.stat().st_size
    return sum(
        child.stat().st_size
        for child in root.rglob("*")
        if child.is_file() and not child.is_symlink()
    )


def _memgallery_identity(index: int, source_file: Path) -> str:
    """Use dataset position, not the repeated display character name, as ID."""
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", source_file.stem).strip("_.-") or "sample"
    return f"memgallery_{index:02d}_{stem}"


def _judge(
    system: TrawMemSystem,
    question: str,
    reference: str,
    prediction: str,
    *,
    benchmark: str = "",
    category: str = "",
    question_type: str = "",
    abstention: bool = False,
) -> float:
    """One binary semantic judge call, recorded as stage=judge."""
    # An abstention item's reference is an explanation, and some releases may
    # leave it empty.  The dedicated rubric can still judge whether the model
    # explicitly declined an unsupported question.
    if not prediction or (not reference and not abstention):
        return 0.0
    messages = [
        {"role": "system", "content": BINARY_JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": build_binary_judge_prompt(
            question,
            reference,
            prediction,
            category=category or None,
            question_type=question_type or None,
            benchmark=benchmark or None,
            abstention=abstention,
        )},
    ]
    response = system.llm_client.chat_completion(
        messages, temperature=0.0, max_tokens=8, stage="judge"
    )
    return 1.0 if _parse_binary_judge_response(response) else 0.0


def _parse_binary_judge_response(response: Any) -> bool:
    """Parse the binary judge without accepting an ambiguous verdict.

    OpenAI-compatible models occasionally append punctuation or a short
    explanation despite the one-token contract.  We accept that harmless
    formatting variation only when the first line starts with an unambiguous
    ``YES`` or ``NO`` token; a response whose first token is neither verdict
    remains an evaluation error rather than a silent zero.
    """
    text = str(response or "").strip().lower()
    if not text:
        raise RuntimeError("judge_response_invalid: expected YES or NO")
    first_line = text.splitlines()[0].strip()
    first_line = re.sub(r"^[`*_ \t]+|[`*_ \t]+$", "", first_line)
    match = re.match(r"^(yes|no)\b", first_line)
    if match is None:
        raise RuntimeError("judge_response_invalid: expected YES or NO")
    return match.group(1) == "yes"


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        return ", ".join(_text(item) for item in value)
    return json.dumps(value, ensure_ascii=False)


def _as_text_list(value: Any) -> list[str]:
    """Represent scalar-or-list metadata without losing item boundaries."""
    if value is None:
        return []
    values = value if isinstance(value, (list, tuple)) else [value]
    result = []
    for item in values:
        text = _text(item).strip()
        if text:
            result.append(text)
    return result


def _image_annotations(image_ids: Any, captions: Any) -> list[str]:
    """Serialize image metadata that a text-only model can actually inspect."""
    ids = _as_text_list(image_ids)
    caption_values = _as_text_list(captions)
    count = max(len(ids), len(caption_values))
    annotations = []
    for index in range(count):
        image_id = ids[index] if index < len(ids) else ""
        caption = caption_values[index] if index < len(caption_values) else ""
        fields = []
        if image_id:
            fields.append(f"id: {image_id}")
        if caption:
            fields.append(f"caption: {caption}")
        if fields:
            annotations.append("[Image " + "; ".join(fields) + "]")
    return annotations


def _turns(
    raw: Any,
    *,
    prefix: str = "",
    session_id: str = "",
    timestamp: str = "",
) -> list[dict[str, Any]]:
    """Normalize common conversation encodings into TrawMem turns."""
    if isinstance(raw, dict):
        # Session-style mapping with session_N keys.
        session_keys = [k for k in raw if str(k).startswith("session_") and not str(k).endswith("_date_time")]
        if session_keys:
            out: list[dict[str, Any]] = []
            for key in sorted(session_keys, key=lambda k: int(str(k).split("_")[-1]) if str(k).split("_")[-1].isdigit() else str(k)):
                date = raw.get(f"{key}_date_time", "")
                sid = str(raw.get(f"{key}_id") or key)
                out.extend(
                    _turns(
                        raw[key],
                        prefix=f"{prefix}{key}/",
                        session_id=sid,
                        timestamp=str(date or ""),
                    )
                )
            return out
        # A single mapping may itself be one turn.
        raw = [raw]
    if not isinstance(raw, list):
        return []
    out = []
    ordinal = 0
    for item in raw:
        if isinstance(item, str):
            ordinal += 1
            out.append({
                "speaker": "User",
                "text": item,
                "source_id": f"{prefix}{ordinal}",
                "session_id": session_id,
                "timestamp": timestamp,
            })
            continue
        if not isinstance(item, dict):
            continue
        # Image metadata is retained as text evidence. The identifier matters
        # for visual-search questions, while the caption is the only visual
        # signal available to this text-only protocol.
        caption = (
            item.get("blip_caption")
            or item.get("image_caption")
            or item.get("image_captions")
            or item.get("caption")
        )
        image_id = item.get("image_id") or item.get("image_ids")
        image = item.get("image") or item.get("image_path") or item.get("img_url")
        base = item.get("text") or item.get("content") or item.get("message")
        if base is None:
            base = item.get("user_message")
        if base is None:
            base = ""
        text = _text(base)
        image_prefix = " ".join(_image_annotations(image_id, caption))
        if image_prefix:
            text = f"{image_prefix} {text}".strip()
        elif image and not text:
            # Do not claim image understanding from an opaque path.
            text = f"[Image unavailable to text-only TrawMem: {_text(image)}]"
        speaker = item.get("speaker") or item.get("role") or item.get("author")
        if not speaker:
            speaker = "User" if item.get("user_message") is not None else "Assistant"
        speaker = {"user": "User", "assistant": "Assistant", "human": "User", "ai": "Assistant"}.get(str(speaker).lower(), str(speaker))
        ordinal += 1
        source = item.get("dia_id") or item.get("id") or item.get("turn_id") or f"{prefix}{ordinal}"
        out.append({
            "speaker": speaker,
            "text": text,
            "timestamp": item.get("timestamp") or item.get("time") or item.get("date") or timestamp,
            "source_id": str(source),
            "session_id": str(item.get("session_id") or item.get("session") or session_id),
        })
        # MemBench-like paired turns may contain an assistant response too.
        if item.get("assistant_message") is not None:
            ordinal += 1
            out.append({
                "speaker": "Assistant",
                "text": _text(item["assistant_message"]),
                "timestamp": item.get("timestamp") or item.get("time") or timestamp,
                "source_id": f"{prefix}{ordinal}",
                "session_id": str(item.get("session_id") or item.get("session") or session_id),
            })
    return out


def _dialogue_from_turns(turns: Iterable[dict[str, Any]]) -> list[Dialogue]:
    out = []
    for index, item in enumerate(turns, 1):
        out.append(Dialogue(
            dialogue_id=index,
            speaker=str(item.get("speaker", "User")),
            content=str(item.get("text", "")),
            timestamp=str(item.get("timestamp", "")) or None,
            source_id=str(item.get("source_id", index)),
            session_id=str(item.get("session_id", "S0")),
            turn_index=index,
        ))
    return out


def _memgallery_turns(
    conversation_data: Any,
    *,
    character_profile: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Normalize the official Mem-Gallery session/dialogue schema.

    Mem-Gallery stores one dialogue item as ``user`` and ``assistant`` fields.
    Image captions are retained as text because this TrawMem protocol is
    text-only.
    """
    if not isinstance(conversation_data, list):
        return []
    profile_name = (character_profile or {}).get("name")
    user_speaker = f"user ({profile_name})" if profile_name else "user"
    turns: list[dict[str, Any]] = []
    for session_index, session_data in enumerate(conversation_data):
        if not isinstance(session_data, dict):
            continue
        session_id = str(session_data.get("session_id") or f"session_{session_index}")
        timestamp = _text(session_data.get("date"))
        dialogues = session_data.get("dialogues", [])
        if not isinstance(dialogues, list):
            continue
        for dialogue_index, dialogue in enumerate(dialogues):
            if not isinstance(dialogue, dict):
                continue
            user_text = _text(dialogue.get("user")).strip()
            assistant_text = _text(dialogue.get("assistant")).strip()
            image_ids = dialogue.get("image_id") or dialogue.get("image_ids")
            caption = dialogue.get("image_caption")
            image = _text(dialogue.get("input_image")).strip()
            image_prefix = " ".join(_image_annotations(image_ids, caption))
            if image_prefix:
                user_text = f"{image_prefix} {user_text}".strip()
            elif image and not user_text:
                user_text = f"[Image unavailable to text-only TrawMem: {image}]"
            source_base = str(dialogue.get("round") or dialogue_index)
            if user_text:
                turns.append({
                    "speaker": user_speaker,
                    "text": user_text,
                    "source_id": f"s{session_index}/r{source_base}/user",
                    "session_id": session_id,
                    "timestamp": timestamp,
                })
            if assistant_text:
                turns.append({
                    "speaker": "assistant",
                    "text": assistant_text,
                    "source_id": f"s{session_index}/r{source_base}/assistant",
                    "session_id": session_id,
                    "timestamp": timestamp,
                })
    return turns


def _find_conversation(record: dict[str, Any]) -> Any:
    for key in ("conversation", "dialogue", "messages", "history", "sessions", "turns"):
        if key in record:
            return record[key]
    return []


def _find_qas(record: dict[str, Any]) -> list[dict[str, Any]]:
    qas = (
        record.get("qa")
        or record.get("qas")
        or record.get("qa_pairs")
        or record.get("questions")
        or record.get("queries")
        or record.get("annotations")
    )
    if qas is None and "question" in record:
        qas = [record]
    if isinstance(qas, dict):
        qas = [qas]
    return qas if isinstance(qas, list) else []


def load_memgallery(path: str) -> list[dict[str, Any]]:
    root = Path(path)
    if root.is_dir():
        # The official release keeps the 20 benchmark dialogues under
        # ``dialog/``.  Prefer that directory so result/metadata JSON files
        # are never mistaken for benchmark samples.
        dialog_root = root / "dialog"
        search_root = dialog_root if dialog_root.is_dir() else root
        files = sorted(search_root.glob("*.json")) + sorted(search_root.glob("*.jsonl"))
        if not files:
            files = sorted(search_root.rglob("*.json")) + sorted(search_root.rglob("*.jsonl"))
        if not files:
            raise ValueError(f"Mem-Gallery directory has no JSON/JSONL files: {root}")
        file_records: list[tuple[Path, Any]] = []
        for file in files:
            if file.suffix == ".jsonl":
                for line in file.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        file_records.append((file, json.loads(line)))
            else:
                file_records.append((file, json.loads(file.read_text(encoding="utf-8"))))
    else:
        file_records = [(root, json.loads(root.read_text(encoding="utf-8")))]

    samples = []
    for index, (source_file, blob) in enumerate(file_records):
        # Official Mem-Gallery schema: one JSON object per topic file.
        if isinstance(blob, dict) and (
            "multi_session_dialogues" in blob or "human-annotated QAs" in blob
        ):
            profile = blob.get("character_profile") or {}
            turns = _memgallery_turns(
                blob.get("multi_session_dialogues", []),
                character_profile=profile,
            )
            qas = []
            for qa in blob.get("human-annotated QAs", []) or []:
                if not isinstance(qa, dict):
                    continue
                question = _text(qa.get("question")).strip()
                if not question:
                    continue
                category = _text(qa.get("point")).strip()
                qas.append({
                    "question": question,
                    "reference": _text(qa.get("answer")).strip(),
                    "category": category,
                    "question_type": category,
                    "query_image_caption": _text(
                        qa.get("image_caption")
                        or qa.get("question_image_caption")
                        or qa.get("query_image_caption")
                    ).strip(),
                    "session_id": qa.get("session_id"),
                    "clue": qa.get("clue", []),
                })
            if turns and qas:
                sample_id = _memgallery_identity(index, source_file)
                samples.append({"sample_id": str(sample_id), "turns": turns, "qas": qas})
            continue

        # Keep compatibility with older/local generic releases.
        raw = blob
        if isinstance(raw, dict):
            raw = raw.get("data") or raw.get("samples") or raw.get("records") or raw.get("dialogs") or raw.get("dialogues")
        if not isinstance(raw, list):
            raw = [raw] if isinstance(raw, dict) else []
        for record in raw:
            if not isinstance(record, dict):
                continue
            sample_index = len(samples)
            turns = _turns(_find_conversation(record), prefix=f"d{sample_index}/")
            qas = []
            for qa in _find_qas(record):
                if not isinstance(qa, dict):
                    continue
                qas.append({
                    "question": _text(qa.get("question") or qa.get("query")),
                    "reference": _text(qa.get("answer") or qa.get("ground_truth") or qa.get("gold_answer")),
                    "category": _text(qa.get("category") or qa.get("task_type") or qa.get("type")),
                    "question_type": _text(qa.get("category") or qa.get("task_type") or qa.get("type")),
                    "query_image_caption": _text(
                        qa.get("image_caption")
                        or qa.get("question_image_caption")
                        or qa.get("query_image_caption")
                    ).strip(),
                })
            if turns and qas:
                # Generic/local releases may not have a stable id either;
                # retain the same indexed identity contract as the official
                # topic-file schema.
                samples.append({"sample_id": _memgallery_identity(sample_index, root), "turns": turns, "qas": qas})

    if not samples:
        raise ValueError("No Mem-Gallery samples found; inspect the file schema before running")
    return samples


def _category_for_prompt(qtype: str) -> str:
    """Normalize benchmark task labels without conflating their semantics."""
    value = str(qtype or "").strip().lower()
    if "temporal" in value or value == "tr":
        return "temporal"
    if "conflict" in value or value == "cd":
        return "conflict"
    if ("visual" in value and "search" in value) or value == "vs":
        return "visual-search"
    if value == "vr" or ("visual" in value and "reason" in value):
        return "visual-reasoning"
    if (
        "test-time" in value
        or "test time" in value
        or value in {"ttl", "test_time_learning"}
    ):
        return "test-time-learning"
    if "factual" in value or "retrieval" in value or value == "fr":
        return "factual-retrieval"
    if "knowledge" in value or "update" in value or value == "kr":
        return "knowledge-update"
    if "preference" in value or "personal" in value:
        return "preference"
    if "abstention" in value or "unanswerable" in value or value == "ar":
        return "abstention"
    # Keep unknown task labels on the generic grading path.
    if (
        "multi-hop" in value
        or "multi_hop" in value
        or "entity" in value
        or value == "mr"
    ):
        return "multi-hop"
    return "general"


def _question_for_prompt(qa: dict[str, Any]) -> str:
    """Add a benchmark-provided query-image caption without changing the QA."""
    question = _text(qa.get("question")).strip()
    caption = _text(qa.get("query_image_caption")).strip()
    if not caption:
        return question
    return f"{question}\n[Query image caption: {caption}]"


def _write_partial(
    result_file: str,
    benchmark: str,
    shard_index: int,
    num_shards: int,
    rows: list[dict[str, Any]],
    build_stats: list[dict[str, Any]],
    completed_sample_indices: set[int],
) -> None:
    import config
    partial_path = Path(result_file).with_name(Path(result_file).stem + ".partial.json")
    temp_path = partial_path.with_suffix(partial_path.suffix + ".tmp")
    payload = {
        "checkpoint": {
            "version": 1,
            "benchmark": benchmark,
            "sample_identity_version": 2 if benchmark == "memgallery" else 1,
            "shard_index": shard_index,
            "num_shards": num_shards,
            "model": getattr(config, "LLM_MODEL", ""),
            "embedding_model": getattr(config, "EMBEDDING_MODEL", ""),
        },
        "completed_sample_indices": sorted(completed_sample_indices),
        "build_stats": build_stats,
        "detailed_results": rows,
    }
    temp_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    temp_path.replace(partial_path)


def run(args: argparse.Namespace) -> None:
    samples = load_memgallery(args.dataset)
    if args.num_samples is not None:
        samples = samples[: args.num_samples]
    shard_samples = [(i, s) for i, s in enumerate(samples) if i % args.num_shards == args.shard_index]
    if not shard_samples:
        raise ValueError(f"Shard {args.shard_index}/{args.num_shards} has no samples")

    import config
    tokenizer = None
    tokenizer_path = getattr(config, "TOKENIZER_MODEL_PATH", None)
    tokenizer_encoding = getattr(config, "TOKENIZER_ENCODING", None)
    if tokenizer_path:
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True, trust_remote_code=True)
        except Exception:
            from tokenizers import Tokenizer
            tokenizer = Tokenizer.from_file(str(Path(tokenizer_path) / "tokenizer.json"))
    if tokenizer is None and tokenizer_encoding:
        import tiktoken

        encoding = tiktoken.get_encoding(tokenizer_encoding)

        class _TiktokenAdapter:
            def encode(self, text, add_special_tokens=False):
                del add_special_tokens
                return encoding.encode(str(text), disallowed_special=())

        tokenizer = _TiktokenAdapter()
        print(f"Using context tokenizer encoding: {tokenizer_encoding}")

    system = TrawMemSystem(clear_db=True)
    rows = []
    build_stats = []
    completed_sample_indices: set[int] = set()
    partial_path = Path(args.result_file).with_name(
        Path(args.result_file).stem + ".partial.json"
    )
    if partial_path.is_file():
        try:
            payload = json.loads(partial_path.read_text(encoding="utf-8"))
            checkpoint = payload.get("checkpoint") or {}
            compatible = (
                checkpoint.get("version") == 1
                and checkpoint.get("benchmark") == args.benchmark
                and checkpoint.get("sample_identity_version", 1)
                == (2 if args.benchmark == "memgallery" else 1)
                and checkpoint.get("shard_index") == args.shard_index
                and checkpoint.get("num_shards") == args.num_shards
                and checkpoint.get("model", getattr(config, "LLM_MODEL", "")) == getattr(config, "LLM_MODEL", "")
                and checkpoint.get("embedding_model", getattr(config, "EMBEDDING_MODEL", "")) == getattr(config, "EMBEDDING_MODEL", "")
            )
            if compatible and isinstance(payload.get("detailed_results"), list):
                rows = list(payload["detailed_results"])
                build_stats = list(payload.get("build_stats") or [])
                completed_sample_indices = {
                    int(index) for index in payload.get("completed_sample_indices", [])
                }
                if not completed_sample_indices:
                    completed_sample_indices = {
                        int(row["sample_idx"])
                        for row in rows
                        if row.get("sample_idx") is not None
                    }
                print(
                    f"Resuming {args.benchmark} shard {args.shard_index}/{args.num_shards}: "
                    f"{len(completed_sample_indices)} completed samples",
                    flush=True,
                )
            else:
                print(f"Ignoring incompatible checkpoint: {partial_path}", flush=True)
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
            print(f"Ignoring unreadable checkpoint {partial_path}: {error}", flush=True)
    started = time.time()
    for global_index, sample in shard_samples:
        if global_index in completed_sample_indices:
            print(f"Skipping completed sample {global_index}", flush=True)
            continue
        system.vector_store.clear()
        build_start = time.time()
        with system.llm_client.usage_scope(sample_idx=global_index, phase="build"):
            system.add_dialogues(_dialogue_from_turns(sample["turns"]))
            system.finalize()
        build_seconds = time.time() - build_start
        db_path = getattr(system.vector_store, "db_path", None)
        build_stats.append({
            "sample_idx": global_index,
            "build_seconds": build_seconds,
            "raw_turn_count": len(sample["turns"]),
            "memory_count": len(system.get_all_memories()),
            "memory_size_bytes": directory_size_bytes(db_path) if db_path else None,
            "memory_size_source": "on_disk_db_path_bytes",
        })
        for qidx, qa in enumerate(sample["qas"]):
            question = qa["question"]
            prompt_question = _question_for_prompt(qa)
            reference = qa["reference"]
            qtype = qa.get("question_type", "")
            normalized_qtype = _category_for_prompt(qtype)
            abstention = bool(qa.get("abstention", False)) or normalized_qtype == "abstention"
            with system.llm_client.usage_scope(sample_idx=global_index, question_idx=qidx, question=prompt_question):
                retrieval_start = time.time()
                contexts = system.hybrid_retriever.retrieve_workspace(prompt_question)
                # The benchmark marks abstention by question id.  Preserve
                # that signal for answer generation without changing the
                # retrieval query or the persistent memory representation.
                if normalized_qtype != "general" or abstention:
                    contexts.question_type = "abstention" if abstention else normalized_qtype
                retrieval_time = time.time() - retrieval_start
                context_text = system.answer_generator._format_contexts(contexts)
                if tokenizer is not None:
                    encoded = tokenizer.encode(context_text, add_special_tokens=False)
                    context_tokens = len(encoded.ids) if hasattr(encoded, "ids") else len(encoded)
                else:
                    context_tokens = None
                answer_start = time.time()
                answer_error = None
                try:
                    answer = system.answer_generator.generate_answer(prompt_question, contexts)
                except Exception as error:
                    answer = ""
                    answer_error = f"{type(error).__name__}: {error}"
                answer_time = time.time() - answer_start
                judge = None
                judge_error = None
                if answer_error is None and args.llm_judge:
                    try:
                        judge = _judge(
                            system,
                            prompt_question,
                            reference,
                            answer,
                            benchmark=args.benchmark,
                            category=qa.get("category", ""),
                            question_type=("abstention" if abstention else normalized_qtype),
                            abstention=abstention,
                        )
                    except Exception as error:
                        judge_error = f"{type(error).__name__}: {error}"
                calls = system.llm_client.question_call_count(global_index, qidx)
            evaluation_status = "ok"
            if answer_error:
                evaluation_status = "answer_error"
                # An unavailable answer is an end-to-end zero, but remains
                # visible through the status and error fields below.
                metrics = {"f1": 0.0, "bleu1": 0.0}
            else:
                metrics = {"f1": token_f1(answer, reference), "bleu1": bleu1(answer, reference)}
            if judge_error:
                evaluation_status = "judge_error"
            if judge is not None:
                metrics["llm_judge_score"] = judge
            rows.append({
                "evaluation_status": evaluation_status,
                "benchmark": args.benchmark,
                "sample_id": sample["sample_id"],
                "sample_idx": global_index,
                "question_index": qidx,
                "question": question,
                "prompt_question": prompt_question,
                "answer": answer,
                "reference": reference,
                "category": qa.get("category", ""),
                "question_type": qtype,
                "abstention": abstention,
                "metrics": metrics,
                "answer_error": answer_error,
                "judge_error": judge_error,
                "retrieval_time": retrieval_time,
                "answer_time": answer_time,
                "total_time": retrieval_time + answer_time,
                "retrieved_context_tokens": context_tokens,
                "token_cost": context_tokens,
                "llm_calls_query": calls,
                "retrieval_llm_calls": max(0, calls - 1),
                "llm_judge_calls": 1 if args.llm_judge else 0,
                "build_seconds": build_seconds,
                "memory_size_bytes": build_stats[-1]["memory_size_bytes"],
                "memory_size_source": "on_disk_db_path_bytes",
            })
            print(
                f"[{args.benchmark}] sample={global_index} q={qidx + 1}/{len(sample['qas'])} "
                f"status={evaluation_status}",
                flush=True,
            )
        completed_sample_indices.add(global_index)
        _write_partial(
            args.result_file,
            args.benchmark,
            args.shard_index,
            args.num_shards,
            rows,
            build_stats,
            completed_sample_indices,
        )

    token_usage = system.llm_client.usage_summary(question_count=len(rows))
    context_values = [r["token_cost"] for r in rows if r["token_cost"] is not None]
    memory_values = [
        float(stat["memory_size_bytes"])
        for stat in build_stats
        if isinstance(stat.get("memory_size_bytes"), (int, float))
    ]
    answer_times = [r["answer_time"] for r in rows]
    fields = ("f1", "bleu1", "llm_judge_score")
    aggregate = {name: statistics.fmean([r["metrics"][name] for r in rows if name in r["metrics"]]) for name in fields if any(name in r["metrics"] for r in rows)}
    status_counts = Counter(str(row.get("evaluation_status", "unknown")) for row in rows)
    answer_failure_count = sum(
        count
        for status, count in status_counts.items()
        if status in {"answer_error", "question_error"}
    )
    summary = {
        "benchmark": args.benchmark,
        "num_samples": len(shard_samples),
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
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "elapsed_seconds": time.time() - started,
        "aggregate_metrics": aggregate,
        "avg_retrieval_time": statistics.fmean(r["retrieval_time"] for r in rows) if rows else 0.0,
        "p50_retrieval_time": statistics.median(r["retrieval_time"] for r in rows) if rows else 0.0,
        "p95_retrieval_time": sorted(r["retrieval_time"] for r in rows)[max(0, math.ceil(.95 * len(rows)) - 1)] if rows else 0.0,
        "avg_answer_time": statistics.fmean(answer_times) if answer_times else 0.0,
        "p50_answer_time": statistics.median(answer_times) if answer_times else 0.0,
        "p95_answer_time": sorted(answer_times)[max(0, math.ceil(.95 * len(answer_times)) - 1)] if answer_times else 0.0,
        "avg_total_query_time": statistics.fmean(r["total_time"] for r in rows) if rows else 0.0,
        "p50_total_query_time": statistics.median(r["total_time"] for r in rows) if rows else 0.0,
        "p95_total_query_time": sorted(r["total_time"] for r in rows)[max(0, math.ceil(.95 * len(rows)) - 1)] if rows else 0.0,
        "avg_context_tokens": statistics.fmean(context_values) if context_values else None,
        "avg_query_llm_calls": statistics.fmean(r["llm_calls_query"] for r in rows) if rows else 0.0,
        "avg_judge_llm_calls": statistics.fmean(r["llm_judge_calls"] for r in rows) if rows else 0.0,
        "avg_build_seconds": statistics.fmean(stat["build_seconds"] for stat in build_stats) if build_stats else 0.0,
        "avg_memory_size_bytes": statistics.fmean(memory_values) if memory_values else None,
        "context_tokenizer": str(
            getattr(config, "TOKENIZER_MODEL_PATH", None)
            or getattr(config, "TOKENIZER_ENCODING", None)
        ),
        "build_stats": build_stats,
        "token_usage": token_usage,
        "protocol": "caption/text-only for Mem-Gallery; no raw image encoder",
    }
    output = Path(args.result_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"method": "TrawMem", "summary": summary, "detailed_results": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    summary_path = getattr(config, "TOKEN_USAGE_SUMMARY_PATH", None)
    if summary_path:
        Path(summary_path).write_text(json.dumps(token_usage, indent=2), encoding="utf-8")
    print(json.dumps({"benchmark": args.benchmark, "questions": len(rows), "metrics": aggregate}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=("memgallery",), required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--result-file", required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--num-samples", type=int)
    parser.add_argument("--llm-judge", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
