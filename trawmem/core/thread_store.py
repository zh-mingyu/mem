"""Persistent thread-head index and local link traversal."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import heapq
import json
import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:  # LanceDB is used in the full experiment environment.
    import lancedb
    import pyarrow as pa
    _HAS_LANCEDB = True
except ImportError:  # The lightweight local test runtime has no LanceDB.
    lancedb = None
    pa = None
    _HAS_LANCEDB = False

from trawmem.core.models.memory_entry import (
    InteractionThread,
    MemoryEntry,
    ThreadLink,
    ThreadNode,
)
from trawmem.core.settings import settings as config
from trawmem.core.utils.embedding import EmbeddingModel


@dataclass
class ThreadRoute:
    thread: InteractionThread
    score: float
    rank: int


@dataclass
class ThreadCandidate:
    thread: InteractionThread
    route_score: float
    hop: int
    path: List[str]


class ThreadStore:
    """Store heads globally; load node/link payloads only for routed threads."""

    def __init__(
        self,
        db_path: Optional[str] = None,
        embedding_model: Optional[EmbeddingModel] = None,
        table_name: Optional[str] = None,
    ) -> None:
        self.db_path = db_path or getattr(config, "THREAD_LANCEDB_PATH", "./thread_lancedb_data")
        self.table_name = table_name or getattr(config, "THREAD_TABLE_NAME", "interaction_threads")
        self.head_table_name = f"{self.table_name}_heads"
        self.payload_table_name = f"{self.table_name}_payloads"
        self.embedding_model = embedding_model or EmbeddingModel()
        self._rows: List[Dict[str, Any]] = []
        self._payload_rows: List[Dict[str, Any]] = []
        self._payload_cache: Dict[str, Dict[str, Any]] = {}
        self._thread_cache: Dict[str, InteractionThread] = {}
        self._head_cache: Dict[str, InteractionThread] = {}
        self.last_route_comparisons = 0
        self.last_expansion_edges = 0
        self._use_lancedb = bool(_HAS_LANCEDB)
        if self._use_lancedb:
            self._is_cloud_storage = self.db_path.startswith(("gs://", "s3://", "az://"))
            if self._is_cloud_storage:
                self.db = lancedb.connect(self.db_path)
            else:
                os.makedirs(self.db_path, exist_ok=True)
                self.db = lancedb.connect(self.db_path)
            self._init_tables()
        else:
            os.makedirs(self.db_path, exist_ok=True)
            self._fallback_path = os.path.join(self.db_path, f"{self.table_name}.json")
            self._load_fallback()

    def _init_tables(self) -> None:
        head_schema = pa.schema(
            [
                pa.field("thread_id", pa.string()),
                pa.field("head", pa.string()),
                pa.field("session_id", pa.string()),
                pa.field("ordinal", pa.int64()),
                pa.field("start_timestamp", pa.string()),
                pa.field("end_timestamp", pa.string()),
                pa.field("links_json", pa.string()),
                pa.field("vector", pa.list_(pa.float32(), self.embedding_model.dimension)),
            ]
        )
        payload_schema = pa.schema(
            [
                pa.field("thread_id", pa.string()),
                pa.field("nodes_json", pa.string()),
            ]
        )
        table_names = set(self.db.table_names())
        if self.head_table_name not in table_names:
            self.table = self.db.create_table(self.head_table_name, schema=head_schema)
        else:
            self.table = self.db.open_table(self.head_table_name)
        if self.payload_table_name not in table_names:
            self.payload_table = self.db.create_table(
                self.payload_table_name,
                schema=payload_schema,
            )
        else:
            self.payload_table = self.db.open_table(self.payload_table_name)

    def _load_fallback(self) -> None:
        try:
            with open(self._fallback_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            self._rows = payload if isinstance(payload, list) else []
            self._payload_rows = []
            self._payload_cache = {}
            self._thread_cache = {}
            self._head_cache = {}
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            self._rows = []

    def _save_fallback(self) -> None:
        temp_path = self._fallback_path + ".tmp"
        with open(temp_path, "w", encoding="utf-8") as handle:
            json.dump(self._rows, handle, ensure_ascii=True)
        os.replace(temp_path, self._fallback_path)

    def count(self) -> int:
        return self.table.count_rows() if self._use_lancedb else len(self._rows)

    def clear(self) -> None:
        if self._use_lancedb:
            for name in (self.head_table_name, self.payload_table_name):
                if name in self.db.table_names():
                    self.db.drop_table(name)
            self._init_tables()
            self._payload_cache = {}
            self._thread_cache = {}
            self._head_cache = {}
        else:
            self._rows = []
            self._payload_cache = {}
            self._thread_cache = {}
            self._head_cache = {}
            self._save_fallback()

    def add_threads(self, threads: Sequence[InteractionThread]) -> None:
        if not threads:
            return
        existing = {thread.thread_id: thread for thread in self.get_all_threads()}
        incoming = [thread.copy(deep=True) for thread in threads]
        heads = [thread.routing_text for thread in incoming]
        head_vectors = self.embedding_model.encode_documents(heads)
        # A ledger turn is often anaphoric ("yesterday", "she did it", ...).
        # Blend its raw vector with a parent-trajectory-conditioned vector. The
        # raw component preserves concrete facts while the contextual component
        # resolves the turn's event without a second global retrieval index.
        raw_node_texts = [
            node.text
            for thread in incoming
            for node in thread.nodes
        ]
        contextual_node_texts = [
            self._node_routing_text(thread, node)
            for thread in incoming
            for node in thread.nodes
        ]
        node_vectors = []
        if raw_node_texts:
            all_vectors = self.embedding_model.encode_documents(
                raw_node_texts + contextual_node_texts
            )
            node_count = len(raw_node_texts)
            node_vectors = [
                self._blend_vectors(
                    self._as_list(all_vectors[index]),
                    self._as_list(all_vectors[node_count + index]),
                )
                for index in range(node_count)
            ]
        node_index = 0
        for thread, head_vector in zip(incoming, head_vectors):
            thread.embedding = [float(value) for value in self._as_list(head_vector)]
            for node in thread.nodes:
                if len(node_vectors):
                    node.embedding = [float(value) for value in node_vectors[node_index]]
                node_index += 1
            existing[thread.thread_id] = thread
        materialized = sorted(
            existing.values(),
            key=lambda thread: (thread.ordinal, thread.thread_id),
        )
        # Connectivity is a write-time graph operation.  It is derived from
        # the already-computed head vectors and session order, never from a
        # keyword/entity posting list.  The graph is traversed locally after
        # one head route at query time.
        self._build_thread_links(
            materialized,
            max_neighbors=int(getattr(config, "THREAD_GRAPH_NEIGHBORS", 3)),
            min_similarity=float(getattr(config, "THREAD_GRAPH_MIN_SIM", 0.25)),
        )
        self._thread_cache = {
            thread.thread_id: thread.copy(deep=True)
            for thread in materialized
        }
        self._head_cache = {
            thread.thread_id: InteractionThread(
                thread_id=thread.thread_id,
                head=thread.head,
                session_id=thread.session_id,
                ordinal=thread.ordinal,
                start_timestamp=thread.start_timestamp,
                end_timestamp=thread.end_timestamp,
                links=[link.copy(deep=True) for link in thread.links],
                embedding=list(thread.embedding),
            )
            for thread in materialized
        }
        rows = [self._thread_to_head_row(thread) for thread in materialized]
        payload_rows = [self._thread_to_payload_row(thread) for thread in materialized]
        if self._use_lancedb:
            for name in (self.head_table_name, self.payload_table_name):
                if name in self.db.table_names():
                    self.db.drop_table(name)
            self._init_tables()
            self._payload_cache = {}
            self.table.add(rows)
            self.payload_table.add(payload_rows)
        else:
            # The dependency-free fallback keeps one JSON file for portability;
            # the production LanceDB path physically separates heads and nodes.
            self._rows = [self._thread_to_row(thread) for thread in materialized]
            self._save_fallback()

    def route(self, address: str, top_k: int = 4) -> Tuple[List[ThreadRoute], List[float]]:
        """Route one address to heads and return its query vector."""
        if self.count() == 0:
            return [], []
        query_vector = self.embedding_model.encode_single(address, is_query=True)
        query_values = self._as_list(query_vector)
        if self._use_lancedb:
            raw = self.table.search(query_values).limit(max(1, int(top_k))).to_list()
            scored = [
                (row, 1.0 / (1.0 + max(0.0, float(row.get("_distance", 0.0)))))
                for row in raw
            ]
        else:
            raw = []
            for row in self._rows:
                score = self._cosine(query_values, row.get("vector") or [])
                raw.append((row, (score + 1.0) / 2.0))
            scored = raw
            scored.sort(key=lambda item: item[1], reverse=True)
            scored = scored[: max(1, int(top_k))]
        self.last_route_comparisons = self.count()
        return [
            ThreadRoute(self._hydrate_thread(row), float(score), rank)
            for rank, (row, score) in enumerate(scored, 1)
        ], [float(value) for value in query_values]

    @staticmethod
    def _as_list(value: Any) -> List[float]:
        """Convert numpy/torch/list embedding outputs to plain floats."""
        if hasattr(value, "tolist"):
            value = value.tolist()
        return list(value)

    @staticmethod
    def _node_routing_text(thread: InteractionThread, node: ThreadNode) -> str:
        timestamp = f" [{node.timestamp}]" if node.timestamp else ""
        return f"{thread.head}{timestamp}\n{node.text}"[:4000]

    @staticmethod
    def _blend_vectors(raw: Sequence[float], contextual: Sequence[float]) -> List[float]:
        if len(raw) != len(contextual):
            return [float(value) for value in raw]
        values = [0.6 * float(left) + 0.4 * float(right) for left, right in zip(raw, contextual)]
        norm = math.sqrt(sum(value * value for value in values))
        return [value / norm for value in values] if norm else values

    def get_all_threads(self) -> List[InteractionThread]:
        if not self._use_lancedb:
            if not self._thread_cache and self._rows:
                self._thread_cache = {
                    thread.thread_id: thread
                    for thread in (self._row_to_thread(row) for row in self._rows)
                }
            return [thread.copy(deep=True) for thread in self._thread_cache.values()]
        if self._thread_cache:
            return [thread.copy(deep=True) for thread in self._thread_cache.values()]
        head_rows = self.table.to_arrow().to_pylist()
        payload_rows = self.payload_table.to_arrow().to_pylist()
        payload_by_id = {str(row["thread_id"]): row for row in payload_rows}
        threads = [
            self._hydrate_thread(row, payload_by_id.get(str(row["thread_id"])))
            for row in head_rows
        ]
        self._thread_cache = {thread.thread_id: thread for thread in threads}
        self._head_cache = {
            thread.thread_id: self._row_to_head_thread(self._thread_to_head_row(thread))
            for thread in threads
        }
        return [thread.copy(deep=True) for thread in threads]

    def get_all_entries(self) -> List[MemoryEntry]:
        entries: List[MemoryEntry] = []
        for thread in self.get_all_threads():
            for node in thread.nodes:
                entries.append(MemoryEntry(
                    entry_id=node.node_id,
                    lossless_restatement=node.text,
                    timestamp=node.timestamp,
                    source_turn_ids=node.source_turn_ids,
                    thread_id=thread.thread_id,
                ))
        return entries

    def expand(self, routes: Sequence[ThreadRoute], max_hops: int = 2, max_threads: int = 8) -> List[ThreadCandidate]:
        """Traverse routed links with best-first score propagation.

        FIFO traversal made the result depend on route order: a weak early
        neighbor could consume the expansion budget before a stronger path was
        inspected.  The workspace remains local and bounded, but candidates
        are now admitted by propagated route score.
        """
        if not routes:
            return []
        # Seed the cache with routed heads, then fetch only the head metadata
        # named by a traversed edge. This avoids a first-query scan of the
        # complete head table; payloads are hydrated only for reached threads.
        headers = self._head_cache
        for route in routes:
            headers.setdefault(
                route.thread.thread_id,
                self._row_to_head_thread(self._thread_to_head_row(route.thread)),
            )
        frontier: List[Tuple[float, int, int, str, int, ThreadCandidate]] = []
        push_order = 0
        self.last_expansion_edges = 0
        best: Dict[str, ThreadCandidate] = {}
        for route in routes:
            candidate = ThreadCandidate(
                self._hydrate_thread_from_header(route.thread),
                route.score,
                0,
                [route.thread.thread_id],
            )
            previous = best.get(route.thread.thread_id)
            if previous is None or candidate.route_score > previous.route_score:
                best[route.thread.thread_id] = candidate
                heapq.heappush(
                    frontier,
                    (-candidate.route_score, candidate.hop, route.rank, candidate.thread.thread_id, push_order, candidate),
                )
                push_order += 1
        max_threads = max(1, int(max_threads))
        while frontier:
            _neg_score, _hop, _rank, _thread_id, _push_order, current = heapq.heappop(frontier)
            incumbent = best.get(current.thread.thread_id)
            if incumbent is None or incumbent.path != current.path or incumbent.route_score > current.route_score + 1e-12:
                continue
            if current.hop >= max(0, int(max_hops)):
                continue
            for link in sorted(
                current.thread.links,
                key=lambda item: (-float(item.weight), item.target_thread_id),
            ):
                self.last_expansion_edges += 1
                target_header = headers.get(link.target_thread_id)
                if target_header is None:
                    target_header = self._load_head_thread(link.target_thread_id)
                    if target_header is not None:
                        headers[target_header.thread_id] = target_header
                target = (
                    self._hydrate_thread_from_header(target_header)
                    if target_header is not None
                    else None
                )
                if target is None:
                    continue
                next_score = current.route_score * 0.72 + 0.28 * float(link.weight)
                previous = best.get(target.thread_id)
                if previous is not None and previous.route_score >= next_score:
                    continue
                candidate = ThreadCandidate(target, next_score, current.hop + 1, current.path + [target.thread_id])
                best[target.thread_id] = candidate
                heapq.heappush(
                    frontier,
                    (-candidate.route_score, candidate.hop, 999, candidate.thread.thread_id, push_order, candidate),
                )
                push_order += 1
        route_order = {route.thread.thread_id: route.rank for route in routes}
        # A local expansion may add context, but it must never discard a
        # globally routed seed.  Preserve all seeds first, then spend the
        # remaining budget on the strongest propagated neighbors.
        routed = [
            item
            for item in best.values()
            if item.thread.thread_id in route_order
        ]
        routed.sort(key=lambda item: route_order[item.thread.thread_id])
        routed_ids = {item.thread.thread_id for item in routed}
        neighbors = sorted(
            (
                item
                for item in best.values()
                if item.thread.thread_id not in routed_ids
            ),
            key=lambda item: (
                -item.route_score,
                item.hop,
                item.thread.ordinal,
                item.thread.thread_id,
            ),
        )
        return (routed + neighbors)[:max_threads]

    @staticmethod
    def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
        if not left or not right or len(left) != len(right):
            return 0.0
        dot = sum(float(a) * float(b) for a, b in zip(left, right))
        left_norm = math.sqrt(sum(float(a) * float(a) for a in left))
        right_norm = math.sqrt(sum(float(b) * float(b) for b in right))
        return dot / (left_norm * right_norm) if left_norm and right_norm else 0.0

    @staticmethod
    def _build_thread_links(
        threads: List[InteractionThread],
        max_neighbors: int = 3,
        min_similarity: float = 0.25,
    ) -> None:
        """Build a sparse episode graph from dense heads and chronology.

        This is deliberately an offline graph-construction step.  It does not
        create parallel retrieval channels.  At inference
        time only edges reachable from routed heads are inspected.
        """
        for thread in threads:
            thread.links = []
        max_neighbors = max(1, int(max_neighbors))
        min_similarity = max(-1.0, min(1.0, float(min_similarity)))
        pair_scores: Dict[str, List[Tuple[float, InteractionThread]]] = {
            thread.thread_id: [] for thread in threads
        }
        # Chronology is defined within an input session. Using global ordinal
        # adjacency would accidentally connect the last event of one session
        # to the first event of another when batches are interleaved.
        by_session: Dict[str, List[InteractionThread]] = {}
        for thread in threads:
            if thread.session_id:
                by_session.setdefault(thread.session_id, []).append(thread)

        def add_edge(left: InteractionThread, right: InteractionThread, relation: str, weight: float) -> None:
            # The same pair can receive a chronology edge and a dense edge;
            # retain the stronger interpretation once.
            for existing in left.links:
                if existing.target_thread_id == right.thread_id:
                    existing.weight = max(existing.weight, weight)
                    if existing.relation != "chronology-next" and relation == "chronology-next":
                        existing.relation = relation
                    return
            left.links.append(ThreadLink(
                target_thread_id=right.thread_id,
                relation=relation,
                weight=float(weight),
            ))

        chronology_pairs: set[Tuple[str, str]] = set()
        for session_threads in by_session.values():
            ordered = sorted(session_threads, key=lambda item: (item.ordinal, item.thread_id))
            for left, right in zip(ordered, ordered[1:]):
                chronology_pairs.add((left.thread_id, right.thread_id))
                chronology_pairs.add((right.thread_id, left.thread_id))

        for index, left in enumerate(threads):
            for right in threads[index + 1:]:
                adjacent = (left.thread_id, right.thread_id) in chronology_pairs
                similarity = ThreadStore._cosine(left.embedding, right.embedding)
                if adjacent:
                    add_edge(left, right, "chronology-next", 0.65)
                    add_edge(right, left, "chronology-next", 0.65)
                if similarity >= min_similarity:
                    pair_scores[left.thread_id].append((similarity, right))
                    pair_scores[right.thread_id].append((similarity, left))

        for thread in threads:
            neighbors = sorted(
                pair_scores[thread.thread_id],
                key=lambda item: (-item[0], item[1].ordinal, item[1].thread_id),
            )[:max_neighbors]
            for similarity, target in neighbors:
                weight = (similarity + 1.0) / 2.0
                add_edge(thread, target, "head-neighbor", weight)
                add_edge(target, thread, "head-neighbor", weight)

    @staticmethod
    def _thread_to_head_row(thread: InteractionThread) -> Dict[str, Any]:
        return {
            "thread_id": thread.thread_id,
            "head": thread.head,
            "session_id": thread.session_id or "",
            "ordinal": int(thread.ordinal),
            "start_timestamp": thread.start_timestamp or "",
            "end_timestamp": thread.end_timestamp or "",
            "links_json": json.dumps([link.dict() for link in thread.links], ensure_ascii=True),
            "vector": list(thread.embedding),
        }

    @staticmethod
    def _thread_to_payload_row(thread: InteractionThread) -> Dict[str, Any]:
        return {
            "thread_id": thread.thread_id,
            "nodes_json": json.dumps(
                [node.dict() for node in thread.nodes],
                ensure_ascii=True,
            ),
        }

    @staticmethod
    def _thread_to_row(thread: InteractionThread) -> Dict[str, Any]:
        """Compatibility row for the dependency-free JSON backend."""
        row = ThreadStore._thread_to_head_row(thread)
        row["nodes_json"] = json.dumps(
            [node.dict() for node in thread.nodes],
            ensure_ascii=True,
        )
        return row

    @staticmethod
    def _row_to_thread(row: Dict[str, Any]) -> InteractionThread:
        def parse_json(value: Any, default: Any) -> Any:
            if isinstance(value, str):
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    return default
            return value if value is not None else default

        return InteractionThread(
            thread_id=str(row["thread_id"]),
            head=str(row.get("head") or ""),
            session_id=str(row.get("session_id") or "") or None,
            ordinal=int(row.get("ordinal") or 0),
            start_timestamp=str(row.get("start_timestamp") or "") or None,
            end_timestamp=str(row.get("end_timestamp") or "") or None,
            nodes=[ThreadNode(**item) for item in parse_json(row.get("nodes_json"), [])],
            links=[ThreadLink(**item) for item in parse_json(row.get("links_json"), [])],
            embedding=[float(value) for value in (row.get("vector") or [])],
        )

    @staticmethod
    def _row_to_head_thread(row: Dict[str, Any]) -> InteractionThread:
        """Decode a LanceDB head row without loading node payloads."""
        def parse_json(value: Any, default: Any) -> Any:
            if isinstance(value, str):
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    return default
            return value if value is not None else default

        return InteractionThread(
            thread_id=str(row["thread_id"]),
            head=str(row.get("head") or ""),
            session_id=str(row.get("session_id") or "") or None,
            ordinal=int(row.get("ordinal") or 0),
            start_timestamp=str(row.get("start_timestamp") or "") or None,
            end_timestamp=str(row.get("end_timestamp") or "") or None,
            nodes=[],
            links=[ThreadLink(**item) for item in parse_json(row.get("links_json"), [])],
            embedding=[float(value) for value in (row.get("vector") or [])],
        )

    def _load_payload_row(self, thread_id: str) -> Optional[Dict[str, Any]]:
        if not self._use_lancedb:
            for row in self._rows:
                if str(row.get("thread_id")) == thread_id:
                    return row
            return None
        if thread_id in self._payload_cache:
            return self._payload_cache[thread_id]
        row: Optional[Dict[str, Any]] = None
        try:
            escaped = thread_id.replace("'", "''")
            row_list = (
                self.payload_table.search()
                .where(f"thread_id = '{escaped}'", prefilter=True)
                .limit(1)
                .to_list()
            )
            if row_list:
                row = row_list[0]
        except Exception:
            # Older LanceDB releases do not support a filter-only search. The
            # fallback is still payload-only and keeps the head ANN table lean.
            try:
                for candidate in self.payload_table.to_arrow().to_pylist():
                    if str(candidate.get("thread_id")) == thread_id:
                        row = candidate
                        break
            except Exception:
                row = None
        if row is not None:
            self._payload_cache[thread_id] = row
        return row

    def _load_head_thread(self, thread_id: str) -> Optional[InteractionThread]:
        """Load one lightweight head row for a graph edge target."""
        if thread_id in self._head_cache:
            return self._head_cache[thread_id]
        row: Optional[Dict[str, Any]] = None
        if not self._use_lancedb:
            for candidate in self._rows:
                if str(candidate.get("thread_id")) == thread_id:
                    row = candidate
                    break
        else:
            try:
                escaped = thread_id.replace("'", "''")
                rows = (
                    self.table.search()
                    .where(f"thread_id = '{escaped}'", prefilter=True)
                    .limit(1)
                    .to_list()
                )
                if rows:
                    row = rows[0]
            except Exception:
                # Keep compatibility with LanceDB versions without a
                # filter-only search; this fallback is used only for one edge
                # target at a time and still avoids hydrating node payloads.
                try:
                    for candidate in self.table.to_arrow().to_pylist():
                        if str(candidate.get("thread_id")) == thread_id:
                            row = candidate
                            break
                except Exception:
                    row = None
        if row is None:
            return None
        header = self._row_to_head_thread(row)
        self._head_cache[thread_id] = header
        return header

    def _hydrate_thread(
        self,
        head_row: Dict[str, Any],
        payload_row: Optional[Dict[str, Any]] = None,
    ) -> InteractionThread:
        if not self._use_lancedb:
            thread_id = str(head_row.get("thread_id"))
            cached = self._thread_cache.get(thread_id)
            return cached.copy(deep=True) if cached is not None else self._row_to_thread(head_row)
        header = self._row_to_head_thread(head_row)
        payload_row = payload_row or self._load_payload_row(header.thread_id)
        if payload_row:
            raw_nodes = payload_row.get("nodes_json")
            if isinstance(raw_nodes, str):
                try:
                    raw_nodes = json.loads(raw_nodes)
                except json.JSONDecodeError:
                    raw_nodes = []
            header.nodes = [ThreadNode(**item) for item in (raw_nodes or [])]
        return header

    def _hydrate_thread_from_header(self, header: InteractionThread) -> InteractionThread:
        if not self._use_lancedb:
            cached = self._thread_cache.get(header.thread_id)
            return cached.copy(deep=True) if cached is not None else header
        if header.nodes:
            return header
        row = self._load_payload_row(header.thread_id)
        if row:
            raw_nodes = row.get("nodes_json")
            if isinstance(raw_nodes, str):
                try:
                    raw_nodes = json.loads(raw_nodes)
                except json.JSONDecodeError:
                    raw_nodes = []
            header.nodes = [ThreadNode(**item) for item in (raw_nodes or [])]
        return header
