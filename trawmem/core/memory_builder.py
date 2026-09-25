"""Build provenance-preserving interaction threads.

The persistent unit is an *event thread*: a short, contiguous interaction
episode with one routing synopsis and an ordered ledger of the original turns.
The language model proposes boundaries and the synopsis, but it does not
rewrite turns into answer-ready semantic memories. A later query-conditioned
workspace chooses and relates ledger nodes for the question at hand.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Sequence

from trawmem.core.models.memory_entry import Dialogue, InteractionThread, ThreadNode
from trawmem.core.prompts import (
    THREAD_BUILDER_SYSTEM_PROMPT,
    build_thread_builder_prompt,
)
from trawmem.core.settings import settings as config
from trawmem.core.thread_store import ThreadStore

if TYPE_CHECKING:
    from trawmem.core.utils.llm_client import LLMClient


class MemoryBuilder:
    """Construct contiguous event threads without semantic memory views."""

    def __init__(
        self,
        llm_client: LLMClient,
        vector_store: ThreadStore,
        window_size: Optional[int] = None,
        enable_parallel_processing: bool = True,
        max_parallel_workers: int = 3,
    ) -> None:
        self.llm_client = llm_client
        self.vector_store = vector_store
        self.window_size = int(window_size or getattr(config, "WINDOW_SIZE", 20))
        # A session is an input boundary, not necessarily a memory boundary.
        # The LLM may discover finer event boundaries; this cap is a recovery
        # guard and keeps a single head from becoming an entire conversation.
        self.thread_turn_limit = max(
            1,
            int(getattr(config, "THREAD_TURN_LIMIT", 10)),
        )
        self.enable_parallel_processing = enable_parallel_processing
        self.max_parallel_workers = max_parallel_workers
        self.dialogue_buffer: List[Dialogue] = []
        self.processed_count = 0
        self.thread_count = 0

    def add_dialogue(self, dialogue: Dialogue, auto_process: bool = True) -> None:
        self.dialogue_buffer.append(dialogue)
        if auto_process and len(self.dialogue_buffer) >= self.window_size:
            self.process_window()

    def add_dialogues(self, dialogues: Sequence[Dialogue], auto_process: bool = True) -> None:
        self.dialogue_buffer.extend(dialogues)
        # Dataset sessions are kept intact until finalize so the segmentation
        # call can see a complete local episode. Anonymous streams retain the
        # old bounded-window behavior.
        if auto_process and not any(dialogue.session_id for dialogue in dialogues):
            while len(self.dialogue_buffer) >= self.window_size:
                self.process_window()

    def process_window(self) -> None:
        if not self.dialogue_buffer:
            return
        window = self.dialogue_buffer[: self.window_size]
        self.dialogue_buffer = self.dialogue_buffer[self.window_size :]
        self._store_groups(self._split_groups(window))

    def process_remaining(self) -> None:
        if not self.dialogue_buffer:
            return
        groups = self._split_groups(self.dialogue_buffer)
        self.dialogue_buffer = []
        self._store_groups(groups)

    def _store_groups(self, groups: Sequence[List[Dialogue]]) -> None:
        threads: List[InteractionThread] = []
        for group in groups:
            if not group:
                continue
            built = self._build_threads(group, self.thread_count)
            threads.extend(built)
            self.thread_count += len(built)
            # A source turn is processed once even if the recovery path makes
            # more than one event thread from the containing episode.
            self.processed_count += len(group)
        if threads:
            self.vector_store.add_threads(threads)

    def _split_groups(self, dialogues: Sequence[Dialogue]) -> List[List[Dialogue]]:
        """Split only at explicit session/input boundaries.

        Fine-grained event segmentation happens in :meth:`_build_threads` so a
        batch boundary never becomes a semantic memory category.
        """
        groups: List[List[Dialogue]] = []
        current: List[Dialogue] = []
        current_session: Optional[str] = None
        for dialogue in dialogues:
            session_id = dialogue.session_id
            boundary = bool(
                current
                and session_id
                and current_session
                and session_id != current_session
            )
            if boundary or (len(current) >= self.window_size and not session_id):
                groups.append(current)
                current = []
                current_session = None
            current.append(dialogue)
            current_session = session_id or current_session
        if current:
            groups.append(current)
        return groups

    def _build_threads(
        self,
        dialogues: List[Dialogue],
        ordinal_start: int,
    ) -> List[InteractionThread]:
        """Segment one input episode and materialize its event threads.

        The model is asked for a list of contiguous source-turn spans. The
        parser treats those spans as routing/continuity scaffolding only. Any
        omitted or malformed span is recovered as a raw contiguous chunk, so
        construction cannot silently lose evidence.
        """
        payload = self._extract_thread_bundle(dialogues)
        specs = self._normalise_specs(payload, dialogues)
        if not specs:
            specs = [
                {"dialogues": list(chunk), "address_synopsis": ""}
                for chunk in self._chunk_dialogues(dialogues)
            ]

        source_order = {item.source_turn_id: index for index, item in enumerate(dialogues)}
        covered: set[str] = set()
        result: List[InteractionThread] = []
        for spec in specs:
            raw_ids = [
                source_id
                for source_id in spec.get("source_turn_ids", [])
                if source_id in source_order and source_id not in covered
            ]
            if raw_ids:
                indices = sorted(source_order[source_id] for source_id in raw_ids)
                runs = self._contiguous_runs(indices)
                chunks = [[dialogues[index] for index in run] for run in runs]
            else:
                # Every source id in this spec was already claimed by an
                # earlier valid thread. Do not materialize overlapping spans a
                # second time.
                if spec.get("source_turn_ids"):
                    continue
                chunks = [list(spec.get("dialogues") or [])]

            for chunk in chunks:
                if not chunk:
                    continue
                # A malformed model response must not create a giant head.
                for bounded in self._chunk_dialogues(chunk):
                    bounded_ids = {item.source_turn_id for item in bounded}
                    bounded_spec = dict(spec)
                    bounded_spec["source_turn_ids"] = list(bounded_ids)
                    thread = self._materialize_thread(
                        bounded,
                        bounded_spec,
                        ordinal_start + len(result),
                    )
                    result.append(thread)
                    covered.update(bounded_ids)

        # Recover every source turn not claimed by a valid model span. This is
        # also what makes the builder robust to partial JSON from an LLM.
        missing_indices = [
            index
            for index, item in enumerate(dialogues)
            if item.source_turn_id not in covered
        ]
        for run in self._contiguous_runs(missing_indices):
            for chunk in self._chunk_dialogues([dialogues[index] for index in run]):
                result.append(
                    self._materialize_thread(
                        chunk,
                        {"address_synopsis": ""},
                        ordinal_start + len(result),
                    )
                )

        return result

    def _chunk_dialogues(self, dialogues: Sequence[Dialogue]) -> Iterable[List[Dialogue]]:
        limit = max(1, self.thread_turn_limit)
        for start in range(0, len(dialogues), limit):
            chunk = list(dialogues[start : start + limit])
            if chunk:
                yield chunk

    @staticmethod
    def _contiguous_runs(indices: Sequence[int]) -> List[List[int]]:
        if not indices:
            return []
        runs: List[List[int]] = [[int(indices[0])]]
        for index in indices[1:]:
            index = int(index)
            if index == runs[-1][-1] + 1:
                runs[-1].append(index)
            else:
                runs.append([index])
        return runs

    def _normalise_specs(
        self,
        payload: Dict[str, Any],
        dialogues: List[Dialogue],
    ) -> List[Dict[str, Any]]:
        """Accept the bundle schema and the old single-thread test schema."""
        has_bundle_schema = isinstance(payload, dict) and isinstance(
            payload.get("threads"), list
        )
        raw_specs: Any = payload.get("threads") if has_bundle_schema else None
        if not isinstance(raw_specs, list):
            if isinstance(payload, dict) and (
                isinstance(payload.get("nodes"), list)
                or payload.get("address_synopsis")
                or payload.get("thread_head")
            ):
                raw_specs = [payload]
            else:
                raw_specs = []

        known_ids = {item.source_turn_id for item in dialogues}
        by_id = {item.source_turn_id: item for item in dialogues}
        specs: List[Dict[str, Any]] = []
        for raw in raw_specs:
            if not isinstance(raw, dict):
                continue
            nodes = raw.get("nodes") if isinstance(raw.get("nodes"), list) else []
            source_ids = [
                str(value)
                for value in (raw.get("source_turn_ids") or [])
                if str(value) in known_ids
            ]
            if not source_ids:
                source_ids = [
                    str(value)
                    for node in nodes
                    if isinstance(node, dict)
                    for value in (node.get("source_turn_ids") or [])
                    if str(value) in known_ids
                ]
            # A single legacy node without provenance is assigned by position
            # only when there is no better source-turn signal.
            if not source_ids and nodes:
                source_ids = [dialogues[0].source_turn_id]
            # The pre-bundle schema represented one complete input episode.
            # Keep that compatibility behavior so a partial node list still
            # receives raw provenance for every turn in the episode.
            if not has_bundle_schema:
                source_ids = [item.source_turn_id for item in dialogues]
            if not source_ids:
                continue
            specs.append(
                {
                    "address_synopsis": str(
                        raw.get("address_synopsis") or raw.get("thread_head") or ""
                    ).strip(),
                    "source_turn_ids": list(dict.fromkeys(source_ids)),
                    "nodes": nodes,
                    "dialogues": [by_id[item] for item in source_ids if item in by_id],
                }
            )
        return specs

    def _materialize_thread(
        self,
        dialogues: List[Dialogue],
        spec: Dict[str, Any],
        ordinal: int,
    ) -> InteractionThread:
        source_ids = [dialogue.source_turn_id for dialogue in dialogues]
        session_id = next((dialogue.session_id for dialogue in dialogues if dialogue.session_id), None)
        thread_key = "|".join([str(session_id or ""), *source_ids])
        thread_id = "thread-" + hashlib.sha1(thread_key.encode("utf-8")).hexdigest()[:16]
        head = str(spec.get("address_synopsis") or "").strip()
        if not head:
            head = self._fallback_head(dialogues)

        # The bank keeps an ordered ledger of the original turns. Any optional
        # ``nodes`` returned by older prompts are used only for provenance and
        # boundaries; generated prose is deliberately ignored so construction
        # cannot freeze a query-specific interpretation.
        nodes = self._materialize_ledger(dialogues, spec.get("nodes"), thread_id)
        nodes.sort(key=lambda node: (node.ordinal, node.node_id))
        for index, node in enumerate(nodes):
            node.ordinal = index
            node.node_id = f"{thread_id}:n{index + 1}"

        timestamps = [dialogue.timestamp for dialogue in dialogues if dialogue.timestamp]
        return InteractionThread(
            thread_id=thread_id,
            head=head,
            session_id=session_id,
            ordinal=ordinal,
            start_timestamp=timestamps[0] if timestamps else None,
            end_timestamp=timestamps[-1] if timestamps else None,
            # The bank stores one routing synopsis plus an ordered ledger.
            # Roles and query-specific relations are created only in workspace.
            nodes=nodes,
        )

    def _extract_thread_bundle(self, dialogues: List[Dialogue]) -> Dict[str, Any]:
        dialogue_text = "\n".join(
            f"[{dialogue.source_turn_id}] {dialogue}" for dialogue in dialogues
        )
        prompt = build_thread_builder_prompt(dialogue_text)
        messages = [
            {
                "role": "system",
                "content": THREAD_BUILDER_SYSTEM_PROMPT,
            },
            {"role": "user", "content": prompt},
        ]
        try:
            response_format = (
                {"type": "json_object"}
                if getattr(config, "USE_JSON_FORMAT", False)
                else None
            )
            response = self.llm_client.chat_completion(
                messages,
                temperature=0.0,
                response_format=response_format,
                stage="maintenance.event_thread_builder",
            )
            data = self.llm_client.extract_json(response)
            return data if isinstance(data, dict) else {}
        except Exception as error:
            raise RuntimeError(
                f"thread segmentation failed; refusing raw-turn fallback: {error}"
            ) from error

    @staticmethod
    def _materialize_ledger(
        dialogues: List[Dialogue],
        raw_nodes: Any,
        thread_id: str,
    ) -> List[ThreadNode]:
        """Materialize one node per original turn, preserving exact text.

        Older extraction prompts sometimes returned node prose.  We only use
        their source ids as an optional ordering hint; the persisted text is
        always reconstructed from ``Dialogue`` so no model-generated summary
        becomes a permanent memory view.
        """
        metadata_by_source: Dict[str, Dict[str, Any]] = {}
        if isinstance(raw_nodes, list):
            for index, item in enumerate(raw_nodes):
                if not isinstance(item, dict):
                    continue
                source_ids = [str(value) for value in (item.get("source_turn_ids") or [])]
                for source_id in source_ids:
                    metadata_by_source.setdefault(source_id, {
                        "ordinal": int(item.get("ordinal", index) or index),
                        "timestamp": item.get("timestamp"),
                        "speakers": item.get("speakers") or [],
                    })

        nodes: List[ThreadNode] = []
        for index, dialogue in enumerate(dialogues):
            metadata = metadata_by_source.get(dialogue.source_turn_id, {})
            speakers = [
                str(value)
                for value in (metadata.get("speakers") or [dialogue.speaker])
                if str(value).strip()
            ]
            nodes.append(
                ThreadNode(
                    node_id=f"{thread_id}:raw{index + 1}",
                    text=f"{dialogue.speaker}: {dialogue.content}",
                    source_turn_ids=[dialogue.source_turn_id],
                    ordinal=index,
                    timestamp=dialogue.timestamp or (
                        str(metadata.get("timestamp"))
                        if metadata.get("timestamp")
                        else None
                    ),
                    speakers=speakers,
                )
            )
        return nodes

    @staticmethod
    def _fallback_head(dialogues: List[Dialogue]) -> str:
        snippets = [
            dialogue.content.strip()
            for dialogue in dialogues[:3]
            if dialogue.content.strip()
        ]
        return "Event trajectory: " + " ".join(snippets)[:640]
