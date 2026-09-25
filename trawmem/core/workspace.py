"""Query-conditioned roles over a locally routed memory graph.

There is one global addressing operation over event-thread heads. A query-local
functional role field is computed after routing and local graph expansion. In
the canonical ``full`` scope, the frozen field conditions local transport,
graph coverage, greedy evidence admission, and answer readout; the optional
``selection-only`` scope is retained for controlled comparisons. Roles are
ephemeral workspace state: they are recomputed for every query and never
written into the persistent bank. Requirements are never converted into
searches, and no persistent memory category is assumed.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
import hashlib
import math
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from pydantic import BaseModel, Field

from trawmem.core.models.memory_entry import InteractionThread
from trawmem.core.thread_store import ThreadCandidate


ROLE_NAMES = ("anchor", "bridge", "context")


class WorkspaceEvidence(BaseModel):
    """A memory node activated for one question."""

    evidence_id: str
    thread_id: str
    node_id: str
    text: str
    thread_head: str = ""
    source_turn_ids: List[str] = Field(default_factory=list)
    contribution: str = "support"
    # ``role`` is a query-local functional assignment.  ``contribution`` is
    # retained as a compatibility alias for older serialized workspaces.
    role: str = "support"
    role_distribution: Dict[str, float] = Field(default_factory=dict)
    role_fit: float = 0.0
    route_score: float = 0.0
    node_score: float = 0.0
    activation_score: float = 0.0
    marginal_gain: float = 0.0
    thread_rank: int = 0
    ordinal: int = 0
    timestamp: Optional[str] = None


class WorkspaceRelation(BaseModel):
    source_evidence_id: str
    target_evidence_id: str
    relation_type: str
    anchors: List[str] = Field(default_factory=list)


class MemoryWorkspace(BaseModel):
    """Ephemeral query-conditioned state handed to the answer model."""

    workspace_id: str
    query: str
    address: str = ""
    question_type: str = "general"
    complexity: str = "LOW"
    evidence: List[WorkspaceEvidence] = Field(default_factory=list)
    relations: List[WorkspaceRelation] = Field(default_factory=list)
    sufficient: bool = False
    # Runtime selection includes bridge bonuses and early stopping; keep the
    # label neutral instead of implying an end-to-end submodular guarantee.
    coverage_mode: str = "capacitated-greedy-heuristic"
    source_candidate_count: int = 0
    routed_thread_count: int = 0
    expanded_thread_count: int = 0
    node_candidate_count: int = 0
    selected_thread_count: int = 0
    selection_budget: int = 0
    selection_coverage: float = 0.0
    selection_rounds: int = 0
    routed_source_turn_ids: List[str] = Field(default_factory=list)
    expanded_source_turn_ids: List[str] = Field(default_factory=list)
    selected_activation_mass: float = 0.0
    activation_entropy: float = 0.0
    activation_iterations: int = 0
    transport_flow_units: int = 0
    transport_cost: float = 0.0
    graph_vertex_count: int = 0
    graph_edge_count: int = 0
    global_route_calls: int = 1
    address_planner_calls: int = 0
    indexed_thread_count: int = 0
    local_edge_visits: int = 0
    # Route diagnostics for the query-conditioned probe gate.
    route_probe_count: int = 0
    route_budget: int = 0
    expansion_budget: int = 0
    adaptive_route: bool = False
    routed_thread_ids: List[str] = Field(default_factory=list)
    expanded_thread_ids: List[str] = Field(default_factory=list)
    selected_node_ids: List[str] = Field(default_factory=list)
    role_mode: str = "none"
    role_frozen: bool = False
    role_demand: Dict[str, float] = Field(default_factory=dict)
    role_entropy: float = 0.0
    role_assignment_count: int = 0

    @property
    def entries(self) -> List[WorkspaceEvidence]:
        return list(self.evidence)

    def __len__(self) -> int:
        return len(self.evidence)

    def __bool__(self) -> bool:
        return bool(self.evidence)


class ThreadWorkspaceBuilder:
    """Allocate query supply to evidence nodes, then select a covered workspace.

    Each routed thread head receives query-conditioned supply proportional to
    its route score. A node is a demand site with bounded capacity. The local
    graph induces a shortest-path cost between every supplied head and demand
    node; a deterministic one-pass soft-cheapest heuristic allocates mass
    using path, query-mismatch, and (when enabled) role costs. It is not an
    optimizer for the capacitated min-cost-flow problem, nor a global ranking
    prior or spreading-activation process. The resulting node flow is used as
    the query-conditioned evidence mass.

    Evidence selection maximizes a facility-location coverage function

        F_q(S) = sum_v mass_q(v) max_{u in S} K_q(u, v),

    where ``K_q`` is a fixed non-negative local graph kernel. ``F_q`` is
    monotone submodular; the standard ``1-1/e`` statement applies only to the
    fixed base objective under a full cardinality budget. The runtime selector
    additionally uses query, role, and thread terms, a relational bridge
    bonus, and early stopping; no end-to-end approximation guarantee is made
    for that augmented procedure. No per-requirement search or DP stage is
    involved.
    """

    def __init__(
        self,
        max_evidence: int = 16,
        min_evidence: int = 1,
        max_relations: int = 12,
        restart_probability: float = 0.25,
        mass_threshold: float = 0.70,
        tolerance: float = 1e-8,
        max_iterations: int = 80,
        selection_epsilon: float = 1e-9,
        thread_diversity_weight: float = 0.05,
        query_prior_weight: float = 0.55,
        query_selection_weight: float = 0.52,
        node_seed_weight: float = 0.80,
        flow_temperature: float = 0.35,
        flow_mismatch_weight: float = 1.25,
        role_aware: bool = False,
        role_scope: str = "full",
        role_transport_weight: float = 0.22,
        role_selection_weight: float = 0.10,
        role_temperature: float = 0.65,
    ) -> None:
        self.max_evidence = max(1, int(max_evidence))
        self.min_evidence = max(1, int(min_evidence))
        self.max_relations = max(1, int(max_relations))
        # Kept in the constructor for callers of the older API. The active
        # transport path does not use a restart process.
        self.mass_threshold = max(0.01, min(1.0, float(mass_threshold)))
        self.tolerance = max(0.0, float(tolerance))
        # max_iterations is retained only for API compatibility; the current
        # path is a single sorted allocation pass.
        self.selection_epsilon = max(0.0, float(selection_epsilon))
        # A partition-coverage term is still monotone submodular. It prevents
        # a long local thread from crowding out a connected second thread.
        self.thread_diversity_weight = max(0.0, min(0.9, float(thread_diversity_weight)))
        # Direct node affinity is a query prior, not a second global retrieval
        # channel: it is evaluated only on the already-induced workspace.
        # query_prior_weight belonged to the retired spreading-activation
        # implementation. Direct query matching is controlled by the
        # selection weight below and is evaluated on the induced candidates.
        self.query_selection_weight = max(0.0, min(0.9, float(query_selection_weight)))
        self.node_seed_weight = max(0.05, min(0.95, float(node_seed_weight)))
        self.head_seed_weight = max(0.05, 1.0 - self.node_seed_weight)
        self.flow_temperature = max(0.05, float(flow_temperature))
        self.flow_mismatch_weight = max(0.0, float(flow_mismatch_weight))
        self.role_aware = bool(role_aware)
        normalized_role_scope = str(role_scope or "full").strip().lower().replace("_", "-")
        if normalized_role_scope not in {"full", "selection-only"}:
            raise ValueError(
                "role_scope must be either 'full' or 'selection-only', "
                f"got {role_scope!r}"
            )
        self.role_scope = normalized_role_scope
        self.role_transport_enabled = self.role_aware and self.role_scope == "full"
        self.role_kernel_enabled = self.role_aware and self.role_scope == "full"
        self.role_prompt_enabled = self.role_aware and self.role_scope == "full"
        self.role_transport_weight = max(0.0, float(role_transport_weight))
        self.role_selection_weight = max(0.0, min(0.45, float(role_selection_weight)))
        self.role_temperature = max(0.05, float(role_temperature))

    @staticmethod
    def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
        if not left or not right or len(left) != len(right):
            return 0.0
        dot = sum(float(a) * float(b) for a, b in zip(left, right))
        left_norm = math.sqrt(sum(float(a) * float(a) for a in left))
        right_norm = math.sqrt(sum(float(b) * float(b) for b in right))
        if not left_norm or not right_norm:
            return 0.0
        return max(-1.0, min(1.0, dot / (left_norm * right_norm)))

    @staticmethod
    def _temporal_signal(text: str) -> float:
        """Estimate whether a node explicitly exposes temporal language.

        This is a local observability feature, not a retrieval query.  It lets
        the role-conditioned kernel preserve turns that carry relative dates
        even when their semantic embeddings are dissimilar to the answer
        turn (for example, a response saying ``the Friday before ...``).
        """
        value = str(text or "").lower()
        markers = re.findall(
            r"\b(?:yesterday|today|tomorrow|last|next|before|after|"
            r"week|weekend|month|year|monday|tuesday|wednesday|thursday|"
            r"friday|saturday|sunday|january|february|march|april|may|"
            r"june|july|august|september|october|november|december|\d{4})\b",
            value,
        )
        return min(1.0, 0.5 * len(markers))

    @staticmethod
    def _normalise(values: Dict[str, float]) -> Dict[str, float]:
        total = sum(max(0.0, value) for value in values.values())
        if not total:
            size = max(1, len(values))
            return {key: 1.0 / size for key in values}
        return {key: max(0.0, value) / total for key, value in values.items()}

    @staticmethod
    def _softmax(values: Sequence[float], temperature: float = 1.0) -> List[float]:
        """Stable softmax used for the ephemeral role field."""
        if not values:
            return []
        scale = max(0.05, float(temperature))
        peak = max(float(value) for value in values)
        exponentials = [
            math.exp((float(value) - peak) / scale)
            for value in values
        ]
        total = sum(exponentials) or 1.0
        return [value / total for value in exponentials]

    @staticmethod
    def _role_query_signals(query: str) -> Dict[str, float]:
        """Extract structural task signals without using benchmark labels."""
        text = str(query or "").lower()
        temporal = float(bool(re.search(
            r"\b(when|what date|what year|which year|how long|how many years|"
            r"what month|what day|before|after|during|first time|last time|"
            r"recently|yesterday|tomorrow)\b",
            text,
        )))
        relational = float(bool(re.search(
            r"\b(both|each other|between|shared|relationship|compare|"
            r"difference|similar|same|because|what led|in common)\b",
            text,
        )))
        return {
            "temporal": temporal,
            "relational": relational,
            "compound": float(len(re.findall(r"\b\w+\b", text)) >= 12),
        }

    @classmethod
    def _role_demand(cls, query: str) -> Dict[str, float]:
        """Return the task's desired functional mix over the three roles."""
        signals = cls._role_query_signals(query)
        raw = {
            "anchor": 1.0,
            "bridge": 0.30 + 0.85 * signals["relational"] + 0.25 * signals["compound"],
            "context": 0.22 + 0.80 * signals["temporal"] + 0.25 * signals["relational"],
        }
        return cls._normalise(raw)

    def _assign_roles(
        self,
        candidates: Sequence[Dict[str, Any]],
        expanded: Sequence[ThreadCandidate],
        adjacency: Dict[str, Dict[str, float]],
        query: str,
    ) -> Tuple[Dict[str, float], float]:
        """Assign a frozen, query-local functional role distribution.

        The scores are structural rather than semantic memory categories:
        direct query affinity supports ``anchor``; cross-thread connectivity
        supports ``bridge``; local chronology and temporal observability
        support ``context``.  The resulting posterior is computed once per
        query and then consumed by transport and readout.
        """
        demand = self._role_demand(query)
        included = {item.thread.thread_id for item in expanded}
        link_strength: Dict[str, float] = defaultdict(float)
        for item in expanded:
            link_strength[item.thread.thread_id] = sum(
                max(0.0, float(link.weight))
                for link in item.thread.links
                if link.target_thread_id in included
            )
        max_link = max(link_strength.values(), default=0.0) or 1.0
        by_thread: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for candidate in candidates:
            by_thread[candidate["thread_id"]].append(candidate)

        role_entropy = 0.0
        for candidate in candidates:
            nodes = by_thread[candidate["thread_id"]]
            neighbours = [
                other for other in nodes
                if abs(int(other["ordinal"]) - int(candidate["ordinal"])) == 1
            ]
            continuity = max(
                (max(0.0, self._cosine(
                    candidate.get("node_embedding") or [],
                    other.get("node_embedding") or [],
                )) for other in neighbours),
                default=0.15,
            )
            ordinals = [int(item["ordinal"]) for item in nodes]
            boundary = float(
                int(candidate["ordinal"]) in {min(ordinals), max(ordinals)}
            )
            node_temporal = max(
                self._temporal_signal(candidate.get("text", "")),
                0.45 if candidate.get("timestamp") else 0.0,
            )
            direct = max(0.0, min(1.0, (
                0.75 * float(candidate.get("query_affinity", 0.0))
                + 0.25 * float(candidate.get("route_score", 0.0))
            )))
            bridge = max(0.0, min(1.0, (
                0.70 * link_strength.get(candidate["thread_id"], 0.0) / max_link
                + 0.20 * boundary
                + 0.10 * (1.0 - direct)
            )))
            context = max(0.0, min(1.0, (
                0.65 * continuity
                + 0.20 * (1.0 - direct)
                + 0.15 * node_temporal
            )))
            logits = [
                2.20 * direct + 0.35 * demand["anchor"],
                1.45 * bridge + 0.85 * demand["bridge"] * bridge
                + 0.12 * (1.0 - direct),
                1.30 * context + 0.85 * demand["context"] * node_temporal
                + 0.12 * (1.0 - direct),
            ]
            distribution = self._softmax(logits, self.role_temperature)
            role_distribution = {
                role: round(float(value), 8)
                for role, value in zip(ROLE_NAMES, distribution)
            }
            role_fit = sum(
                role_distribution[role] * demand[role]
                for role in ROLE_NAMES
            )
            role_index = max(
                range(len(ROLE_NAMES)),
                key=lambda index: (distribution[index], -index),
            )
            candidate["role"] = ROLE_NAMES[role_index]
            candidate["role_distribution"] = role_distribution
            candidate["role_fit"] = max(0.0, min(1.0, float(role_fit)))
            candidate["role_demand"] = dict(demand)
            candidate["role_features"] = {
                "direct": round(direct, 8),
                "mediation": round(bridge, 8),
                "continuity": round(context, 8),
            }
            role_entropy -= sum(
                value * math.log(value)
                for value in distribution
                if value > 0.0
            )
        mean_entropy = role_entropy / max(1, len(candidates))
        return demand, mean_entropy

    @staticmethod
    def _infer_question_type(query: str) -> Tuple[str, str]:
        """Infer rendering/organization hints without creating a search query.

        The evaluator's categories are not exposed to the system.  These
        conservative lexical cues only choose how an already-built workspace
        is presented to the answer model; they never alter the global route or
        add a retrieval channel.
        """
        text = str(query or "").lower()
        if re.search(
            r"\b(when|what date|what year|which year|how long|how many years|"
            r"what month|what day|first time|last time|recently)\b",
            text,
        ):
            return "temporal", "MEDIUM"
        if (
            re.search(r"\bhow many times\b", text)
            or re.search(
                r"\bwhat (activities|events|books|places|types|kinds|names|"
                r"pets|instruments|artists|bands|symbols|ways|things|subjects)\b",
                text,
            )
            or re.search(r"\bwhere (has|have|did)\b", text)
        ):
            return "set-valued", "HIGH"
        if (
            re.search(r"\bboth\b|\bin common\b|\beach other\b", text)
            or re.search(r"\bwhat .* and .* (both|have|did)\b", text)
        ):
            return "multi-hop", "HIGH"
        if re.search(
            r"\b(would|could|likely|might|personality|political|future|"
            r"interested|prefer|open to|potentially)\b",
            text,
        ):
            return "open-domain", "HIGH"
        return "general", "LOW"

    @classmethod
    def _plan_state(
        cls,
        plan: Optional[Dict[str, Any]],
        query: str = "",
    ) -> Tuple[str, str]:
        """Read metadata for rendering; never derive a retrieval budget."""
        plan = plan if isinstance(plan, dict) else {}
        question_type = str(plan.get("question_type") or "general").strip().lower()
        complexity = str(plan.get("complexity") or "LOW").upper()
        if question_type in ("", "general"):
            return cls._infer_question_type(query)
        return question_type, complexity

    def build(
        self,
        query: str,
        address: str,
        query_vector: Sequence[float],
        routes: Sequence[Any],
        expanded: Sequence[ThreadCandidate],
        plan: Optional[Dict[str, Any]] = None,
    ) -> MemoryWorkspace:
        question_type, complexity = self._plan_state(plan, query)
        key = "|".join([query, address, *(item.thread.thread_id for item in expanded)])
        workspace_id = "ws-" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
        candidates = self._flatten(expanded, query_vector)
        routed_sources = sorted(
            {
                source_id
                for route in routes
                for node in route.thread.nodes
                for source_id in node.source_turn_ids
            }
        )
        expanded_sources = sorted(
            {
                source_id
                for item in expanded
                for node in item.thread.nodes
                for source_id in node.source_turn_ids
            }
        )
        # The role demand is query-local workspace state. Compute it even
        # when routing yields no candidate so diagnostics remain explicit.
        role_demand: Dict[str, float] = (
            self._role_demand(query) if self.role_aware else {}
        )
        role_entropy = 0.0
        base = dict(
            workspace_id=workspace_id,
            query=query,
            address=address,
            question_type=question_type,
            complexity=complexity,
            role_mode=(
                "selection-only"
                if self.role_aware and self.role_scope == "selection-only"
                else "one-shot" if self.role_aware else "legacy"
            ),
            role_frozen=bool(self.role_aware),
            source_candidate_count=len(expanded),
            routed_thread_count=len(routes),
            expanded_thread_count=len(expanded),
            selection_budget=self.max_evidence,
            routed_source_turn_ids=routed_sources,
            expanded_source_turn_ids=expanded_sources,
            global_route_calls=1,
            routed_thread_ids=[route.thread.thread_id for route in routes],
            expanded_thread_ids=[item.thread.thread_id for item in expanded],
        )
        if not candidates:
            return MemoryWorkspace(
                **base,
                sufficient=False,
                role_demand=role_demand,
            )

        adjacency, seed = self._activation_graph(expanded, candidates)
        if self.role_aware:
            role_demand, role_entropy = self._assign_roles(
                candidates,
                expanded,
                adjacency,
                query,
            )
        else:
            # Preserve the legacy path exactly: role labels remain a display
            # annotation and cannot change transport or selection.
            for candidate in candidates:
                candidate["role_fit"] = 1.0
        node_mass, iterations, edge_visits, flow_units, flow_cost = (
            self._transport_activation(adjacency, seed, candidates)
        )
        for candidate in candidates:
            candidate["activation_score"] = node_mass[candidate["node_id"]]

        kernel = self._kernel(
            candidates,
            expanded,
            question_type=question_type,
            role_demand=role_demand,
        )
        selected, coverage, gains = self._select(
            candidates,
            kernel,
            question_type,
            role_demand=role_demand,
        )
        # Keep the optimizer's stable gain order for compatibility with the
        # answer model calibration.  A later renderer can group turns for a
        # diagnostic view without changing the evidence order used in scoring.
        selected_with_gains = list(zip(selected, gains))
        evidence = [
            WorkspaceEvidence(
                evidence_id=f"W{index}",
                thread_id=item["thread_id"],
                node_id=item["node_id"],
                text=item["text"],
                thread_head=item["thread_head"],
                source_turn_ids=item["source_turn_ids"],
                contribution=(
                    self._role(item, selected, index, question_type)
                    if self.role_prompt_enabled
                    else self._legacy_contribution(item, selected, index, question_type)
                ),
                role=str(
                    item.get("role")
                    or self._legacy_contribution(item, selected, index, question_type)
                ),
                role_distribution={
                    str(key): round(float(value), 8)
                    for key, value in (item.get("role_distribution") or {}).items()
                },
                role_fit=round(float(item.get("role_fit", 0.0)), 8),
                route_score=round(item["route_score"], 6),
                node_score=round(item["dense_score"], 6),
                activation_score=round(item["activation_score"], 6),
                marginal_gain=round(gain, 6),
                thread_rank=item["thread_rank"],
                ordinal=item["ordinal"],
                timestamp=item["timestamp"],
            )
            for index, (item, gain) in enumerate(selected_with_gains, 1)
        ]
        relations = self._relations(evidence, expanded)
        entropy = -sum(value * math.log(value) for value in node_mass.values() if value > 0.0)
        selected_activation_mass = sum(item["activation_score"] for item in selected)
        sufficient = bool(evidence) and coverage >= self.mass_threshold
        return MemoryWorkspace(
            **base,
            evidence=evidence,
            relations=relations,
            sufficient=sufficient,
            node_candidate_count=len(candidates),
            selected_thread_count=len({item["thread_id"] for item, _ in selected_with_gains}),
            selection_coverage=round(coverage, 6),
            selection_rounds=len(selected),
            selected_activation_mass=round(selected_activation_mass, 6),
            activation_entropy=round(entropy, 6),
            activation_iterations=iterations,
            transport_flow_units=flow_units,
            transport_cost=round(flow_cost, 6),
            graph_vertex_count=len(adjacency),
            graph_edge_count=sum(len(edges) for edges in adjacency.values()) // 2,
            local_edge_visits=edge_visits,
            selected_node_ids=[item["node_id"] for item, _ in selected_with_gains],
            role_demand=role_demand,
            role_entropy=round(role_entropy, 8),
            role_assignment_count=len(candidates) if self.role_aware else 0,
        )

    @staticmethod
    def _timestamp_key(value: Any) -> Tuple[int, str]:
        """Return a stable sortable key for ISO-like conversation dates."""
        text = str(value or "")
        if not text:
            return (1, "")
        try:
            return (0, datetime.fromisoformat(text.replace("Z", "+00:00")).isoformat())
        except (TypeError, ValueError):
            return (0, text)

    @classmethod
    def _presentation_order(
        cls,
        selected: Sequence[Dict[str, Any]],
        gains: Sequence[float],
        question_type: str,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """Order evidence for reading while retaining the greedy gains."""
        pairs = list(zip(selected, gains))
        if not pairs:
            return []

        if "temporal" in str(question_type).lower():
            return sorted(
                pairs,
                key=lambda pair: (
                    cls._timestamp_key(pair[0].get("timestamp")),
                    int(pair[0].get("thread_rank", 0)),
                    int(pair[0].get("ordinal", 0)),
                    -float(pair[0].get("query_affinity", 0.0)),
                    str(pair[0].get("node_id", "")),
                ),
            )

        # A thread's best node is a robust local relevance estimate.  Using
        # max rather than a sum avoids rewarding long, noisy trajectories.
        thread_strength: Dict[str, float] = {}
        thread_rank: Dict[str, int] = {}
        for item, _gain in pairs:
            thread_id = str(item.get("thread_id", ""))
            strength = (
                0.55 * float(item.get("query_affinity", 0.0))
                + 0.30 * float(item.get("activation_score", 0.0))
                + 0.15 * float(item.get("route_score", 0.0))
            )
            thread_strength[thread_id] = max(thread_strength.get(thread_id, 0.0), strength)
            thread_rank[thread_id] = min(
                thread_rank.get(thread_id, 10**9),
                int(item.get("thread_rank", 0)),
            )
        return sorted(
            pairs,
            key=lambda pair: (
                -thread_strength.get(str(pair[0].get("thread_id", "")), 0.0),
                thread_rank.get(str(pair[0].get("thread_id", "")), 10**9),
                int(pair[0].get("ordinal", 0)),
                -float(pair[0].get("query_affinity", 0.0)),
                str(pair[0].get("node_id", "")),
            ),
        )

    def _flatten(
        self,
        expanded: Sequence[ThreadCandidate],
        query_vector: Sequence[float],
    ) -> List[Dict[str, Any]]:
        candidates: List[Dict[str, Any]] = []
        for thread_rank, item in enumerate(expanded, 1):
            thread: InteractionThread = item.thread
            for node in thread.nodes:
                dense = (self._cosine(query_vector, node.embedding) + 1.0) / 2.0
                candidates.append(
                    {
                        "thread_id": thread.thread_id,
                        "node_id": node.node_id,
                        "text": node.text,
                        "thread_head": thread.head,
                        "source_turn_ids": list(node.source_turn_ids),
                        "ordinal": int(node.ordinal),
                        "timestamp": node.timestamp,
                        "dense_score": dense,
                        "route_score": max(0.0, min(1.0, item.route_score)),
                        "thread_rank": thread_rank,
                        "activation_score": 0.0,
                        "query_affinity": dense,
                        # Used only by the local coverage kernel. Keeping the
                        # vector here lets the selector distinguish content
                        # that is actually observable from content that is
                        # merely adjacent in the thread graph.
                        "node_embedding": list(node.embedding),
                    }
                )
        return candidates

    @staticmethod
    def _head_vertex(thread_id: str) -> str:
        return f"h:{thread_id}"

    @staticmethod
    def _node_vertex(node_id: str) -> str:
        return f"n:{node_id}"

    def _activation_graph(
        self,
        expanded: Sequence[ThreadCandidate],
        candidates: Sequence[Dict[str, Any]],
    ) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float]]:
        adjacency: Dict[str, Dict[str, float]] = defaultdict(dict)
        seed: Dict[str, float] = {}
        by_thread: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        item_by_thread = {item.thread.thread_id: item for item in expanded}
        included = set(item_by_thread)

        for candidate in candidates:
            by_thread[candidate["thread_id"]].append(candidate)

        def connect(left: str, right: str, weight: float) -> None:
            weight = max(1e-9, float(weight))
            adjacency[left][right] = max(weight, adjacency[left].get(right, 0.0))
            adjacency[right][left] = max(weight, adjacency[right].get(left, 0.0))

        for thread_id, nodes in by_thread.items():
            nodes.sort(key=lambda item: (item["ordinal"], item["node_id"]))
            head_vertex = self._head_vertex(thread_id)
            adjacency.setdefault(head_vertex, {})
            route_score = item_by_thread[thread_id].route_score
            seed[head_vertex] = self.head_seed_weight * max(1e-6, route_score)
            for node in nodes:
                node_vertex = self._node_vertex(node["node_id"])
                adjacency.setdefault(node_vertex, {})
                connect(head_vertex, node_vertex, 0.35)
                seed[node_vertex] = (
                    self.node_seed_weight
                    * max(1e-6, route_score)
                    * max(1e-6, node["dense_score"])
                    / max(1, len(nodes))
                )
            for left, right in zip(nodes, nodes[1:]):
                connect(
                    self._node_vertex(left["node_id"]),
                    self._node_vertex(right["node_id"]),
                    1.0,
                )

        for item in expanded:
            left = self._head_vertex(item.thread.thread_id)
            for link in item.thread.links:
                if link.target_thread_id in included:
                    connect(
                        left,
                        self._head_vertex(link.target_thread_id),
                        max(0.05, link.weight),
                    )
        return dict(adjacency), self._normalise(seed)

    @staticmethod
    def _edge_cost(weight: float) -> float:
        """Convert a positive local affinity into a non-negative path cost."""
        return max(0.0, -math.log(max(1e-9, min(1.0, float(weight)))))

    def _shortest_path_costs(
        self,
        adjacency: Dict[str, Dict[str, float]],
        source: str,
    ) -> Tuple[Dict[str, float], int]:
        """Dijkstra costs on the induced graph, with deterministic tie order."""
        import heapq

        distances = {vertex: math.inf for vertex in adjacency}
        distances[source] = 0.0
        queue: List[Tuple[float, str]] = [(0.0, source)]
        relaxations = 0
        while queue:
            distance, vertex = heapq.heappop(queue)
            if distance > distances.get(vertex, math.inf) + self.tolerance:
                continue
            for target, weight in sorted((adjacency.get(vertex) or {}).items()):
                candidate = distance + self._edge_cost(weight)
                if candidate + self.tolerance < distances.get(target, math.inf):
                    distances[target] = candidate
                    heapq.heappush(queue, (candidate, target))
                    relaxations += 1
        return distances, relaxations

    def _transport_activation(
        self,
        adjacency: Dict[str, Dict[str, float]],
        seed: Dict[str, float],
        candidates: Sequence[Dict[str, Any]],
    ) -> Tuple[Dict[str, float], int, int, int, float]:
        """Solve a bounded source-to-node capacitated transport relaxation.

        The induced graph is converted to a metric closure using Dijkstra.
        Head supply is normalized route mass; each evidence node has capacity
        ``1 / B`` where ``B`` is the evidence budget. A single pass over
        sorted head-node pairs applies a cost-dependent soft allocation. This
        is a bounded heuristic, not an exact min-cost-flow solver. The
        returned flow is normalized and used only as a mass measure; the
        separate greedy selector enforces the final at-most-``B`` evidence
        budget.
        """
        node_vertices = [self._node_vertex(item["node_id"]) for item in candidates]
        head_vertices = sorted(vertex for vertex in seed if vertex.startswith("h:"))
        if not node_vertices or not head_vertices:
            return ({item["node_id"]: 0.0 for item in candidates}, 0, 0, 0, 0.0)

        distances: Dict[str, Dict[str, float]] = {}
        relaxations = 0
        for head in head_vertices:
            distances[head], count = self._shortest_path_costs(adjacency, head)
            relaxations += count

        budget = max(1, min(self.max_evidence, len(candidates)))
        node_capacity = 1.0 / budget
        remaining_supply = {head: max(0.0, float(seed.get(head, 0.0))) for head in head_vertices}
        flow = {item["node_id"]: 0.0 for item in candidates}
        pair_costs: List[Tuple[float, str, str]] = []
        for head in head_vertices:
            for item in candidates:
                node_id = item["node_id"]
                distance = distances[head].get(self._node_vertex(node_id), math.inf)
                if not math.isfinite(distance):
                    continue
                mismatch = self.flow_mismatch_weight * (
                    1.0 - max(0.0, min(1.0, float(item.get("dense_score", 0.0))))
                )
                role_penalty = self.role_transport_weight * (
                    1.0 - max(0.0, min(1.0, float(item.get("role_fit", 1.0))))
                ) if self.role_transport_enabled else 0.0
                pair_costs.append((distance + mismatch + role_penalty, head, node_id))
        pair_costs.sort(key=lambda value: (value[0], value[1], value[2]))

        total_cost = 0.0
        flow_units = 0
        for cost, head, node_id in pair_costs:
            supply = remaining_supply.get(head, 0.0)
            capacity = node_capacity - flow[node_id]
            if supply <= self.selection_epsilon or capacity <= self.selection_epsilon:
                continue
            amount = min(supply, capacity)
            # Temperature turns a hard cheapest-pair tie into a stable,
            # bounded soft allocation while preserving the cost ordering.
            amount *= math.exp(-max(0.0, cost) / self.flow_temperature)
            amount = min(amount, supply, capacity)
            if amount <= self.selection_epsilon:
                continue
            remaining_supply[head] -= amount
            flow[node_id] += amount
            total_cost += amount * cost
            flow_units += 1

        if sum(flow.values()) <= self.selection_epsilon:
            fallback = max(candidates, key=lambda item: float(item.get("dense_score", 0.0)))
            flow[fallback["node_id"]] = 1.0
            total_cost = 1.0 - float(fallback.get("dense_score", 0.0))
            flow_units = 1
        node_mass = self._normalise(flow)
        return node_mass, relaxations, relaxations, flow_units, total_cost

    def _kernel(
        self,
        candidates: Sequence[Dict[str, Any]],
        expanded: Sequence[ThreadCandidate],
        question_type: str = "general",
        role_demand: Optional[Dict[str, float]] = None,
    ) -> List[List[float]]:
        """Build a query-role-conditioned graph kernel for facility coverage.

        For temporal questions, graph proximity alone is not evidence
        observability: selecting a question turn does not reveal the relative
        time expression in an adjacent response.  We therefore use

            K(u, v) = G(u, v) * max(0, cos(e_u, e_v)),

        for temporal evidence. Compositional questions retain G because local
        continuity itself is useful bridge coverage. Thus K_q changes with the
        query role while the persistent graph does not. Every instantiated
        kernel is fixed and non-negative, so facility location remains
        monotone submodular with the same greedy approximation guarantee.
        """
        size = len(candidates)
        kernel = [[0.0 for _ in range(size)] for _ in range(size)]
        demand = (
            self._normalise(role_demand or {})
            if self.role_kernel_enabled
            else {}
        )
        # Blend role compatibility into local graph coverage. The gate stays
        # in [1-lambda, 1], preserving non-negativity and the fixed-kernel
        # monotone-submodular guarantee.
        role_blend = (
            min(0.35, max(0.0, self.role_selection_weight))
            if demand
            else 0.0
        )
        by_thread: Dict[str, List[int]] = defaultdict(list)
        for index, candidate in enumerate(candidates):
            by_thread[candidate["thread_id"]].append(index)
            kernel[index][index] = 1.0

        link_weight: Dict[Tuple[str, str], float] = {}
        for item in expanded:
            for link in item.thread.links:
                key = (item.thread.thread_id, link.target_thread_id)
                link_weight[key] = max(link_weight.get(key, 0.0), float(link.weight))

        temporal_observability = "temporal" in str(question_type).lower()
        for left in range(size):
            for right in range(left + 1, size):
                left_item = candidates[left]
                right_item = candidates[right]
                weight = 0.0
                observable = max(
                    0.0,
                    self._cosine(
                        left_item.get("node_embedding") or [],
                        right_item.get("node_embedding") or [],
                    ),
                )
                if temporal_observability:
                    # Embedding cosine alone can erase a temporally decisive
                    # turn whose wording is lexically different.  The floor
                    # is still query-role-conditioned and non-negative, so
                    # the facility objective remains monotone submodular.
                    lexical_signal = max(
                        self._temporal_signal(left_item.get("text", "")),
                        self._temporal_signal(right_item.get("text", "")),
                    )
                    observability_gate = 0.25 + 0.75 * max(
                        observable,
                        0.35 * lexical_signal,
                    )
                else:
                    observability_gate = 1.0
                if left_item["thread_id"] == right_item["thread_id"]:
                    gap = abs(left_item["ordinal"] - right_item["ordinal"])
                    weight = (0.35 ** max(1, gap)) * observability_gate
                else:
                    weight = 0.45 * observability_gate * max(
                        link_weight.get(
                            (left_item["thread_id"], right_item["thread_id"]),
                            0.0,
                        ),
                        link_weight.get(
                            (right_item["thread_id"], left_item["thread_id"]),
                            0.0,
                        ),
                    )
                if weight > 0.0 and role_blend:
                    left_profile = self._role_profile(left_item)
                    right_profile = self._role_profile(right_item)
                    overlap = sum(
                        demand[role]
                        * min(left_profile[role], right_profile[role])
                        for role in ROLE_NAMES
                    )
                    role_gate = (1.0 - role_blend) + role_blend * max(
                        0.0,
                        min(1.0, overlap),
                    )
                    weight *= role_gate
                if weight > 0.0:
                    kernel[left][right] = weight
                    kernel[right][left] = weight
        return kernel

    def _select(
        self,
        candidates: Sequence[Dict[str, Any]],
        kernel: Sequence[Sequence[float]],
        question_type: str = "general",
        role_demand: Optional[Dict[str, float]] = None,
    ) -> Tuple[List[Dict[str, Any]], float, List[float]]:
        """Greedy coverage with a fixed-kernel submodular base objective.

        The runtime path may add bridge bonuses, multi-thread constraints, and
        early stopping, so this method itself is not covered by the base
        facility-location approximation guarantee.
        """
        if not candidates:
            return [], 0.0, []
        activation = [max(0.0, float(item["activation_score"])) for item in candidates]
        # The direct relevance term is modular, so adding it to the
        # facility-location objective preserves monotone submodularity while
        # making answer-bearing ledger nodes competitive with continuity-only
        # nodes.
        demand = self._normalise(role_demand or {}) if self.role_aware else {}
        role_weight = self.role_selection_weight if demand else 0.0
        facility_weight = max(
            0.0,
            1.0
            - self.thread_diversity_weight
            - self.query_selection_weight
            - role_weight,
        )
        thread_mass: Dict[str, float] = defaultdict(float)
        for item, mass in zip(candidates, activation):
            thread_mass[item["thread_id"]] += mass
        total_thread_mass = sum(thread_mass.values()) or 1.0
        query_target = sum(
            sorted(
                (max(0.0, float(item.get("query_affinity", 0.0))) for item in candidates),
                reverse=True,
            )[: self.max_evidence]
        ) or 1.0
        covered = [0.0 for _ in candidates]
        role_covered = {role: 0.0 for role in ROLE_NAMES}
        selected_indices: List[int] = []
        selected_threads: set[str] = set()
        gains: List[float] = []
        total_coverage = 0.0
        candidate_thread_count = len({item["thread_id"] for item in candidates})
        # Relational questions often need two separate trajectories.  Treat
        # this as a tiny partition-coverage constraint: once the best anchor
        # is chosen, admit one connected alternative before the mass stopping
        # rule can terminate the workspace.  It prevents a high-similarity
        # single thread from hiding the second hop.
        question_mode = str(question_type).lower()
        require_two_threads = (
            ("multi" in question_mode or "set-valued" in question_mode)
            and candidate_thread_count >= 2
        )

        while len(selected_indices) < self.max_evidence:
            best_index: Optional[int] = None
            best_gain = -1.0
            for index in range(len(candidates)):
                if index in selected_indices:
                    continue
                facility_gain = 0.0
                row = kernel[index]
                for target, mass in enumerate(activation):
                    facility_gain += mass * max(0.0, row[target] - covered[target])
                gain = facility_weight * facility_gain
                gain += self.query_selection_weight * max(
                    0.0,
                    float(candidates[index].get("query_affinity", 0.0)),
                )
                if role_weight:
                    profile = self._role_profile(candidates[index])
                    role_gain = sum(
                        demand[role]
                        * max(0.0, profile[role] - role_covered[role])
                        for role in ROLE_NAMES
                    )
                    gain += role_weight * role_gain
                if candidates[index]["thread_id"] not in selected_threads:
                    gain += self.thread_diversity_weight * (
                        thread_mass[candidates[index]["thread_id"]] / total_thread_mass
                    )
                    if require_two_threads and selected_threads:
                        # Prefer a candidate that is locally connected to an
                        # already selected trajectory.  The kernel is the
                        # same non-negative graph coverage used by the
                        # facility-location objective.
                        bridge = max(
                            (kernel[index][chosen] for chosen in selected_indices),
                            default=0.0,
                        )
                        gain += 0.35 * bridge
                # Stable tie-breaks favor routed heads and chronology.
                tie = (
                    candidates[index]["thread_rank"],
                    -candidates[index].get("query_affinity", 0.0),
                    -candidates[index]["dense_score"],
                    candidates[index]["ordinal"],
                    candidates[index]["node_id"],
                )
                best_tie = (
                    candidates[best_index]["thread_rank"],
                    -candidates[best_index].get("query_affinity", 0.0),
                    -candidates[best_index]["dense_score"],
                    candidates[best_index]["ordinal"],
                    candidates[best_index]["node_id"],
                ) if best_index is not None else None
                if best_index is None or gain > best_gain + self.selection_epsilon or (
                    abs(gain - best_gain) <= self.selection_epsilon and tie < best_tie
                ):
                    best_index = index
                    best_gain = gain
            if best_index is None or best_gain <= self.selection_epsilon:
                break
            selected_indices.append(best_index)
            selected_threads.add(candidates[best_index]["thread_id"])
            gains.append(best_gain)
            for target in range(len(candidates)):
                covered[target] = max(
                    covered[target],
                    kernel[best_index][target],
                )
            if role_weight:
                profile = self._role_profile(candidates[best_index])
                for role in ROLE_NAMES:
                    role_covered[role] = max(
                        role_covered[role],
                        profile[role],
                    )
            total_coverage = sum(
                mass * value for mass, value in zip(activation, covered)
            )
            query_coverage = min(
                1.0,
                sum(
                    max(0.0, float(candidates[index].get("query_affinity", 0.0)))
                    for index in selected_indices
                )
                / query_target,
            )
            represented_mass = sum(
                thread_mass[thread_id] for thread_id in selected_threads
            ) / total_thread_mass
            role_coverage = (
                sum(demand[role] * role_covered[role] for role in ROLE_NAMES)
                if role_weight
                else 0.0
            )
            total_coverage = (
                facility_weight * total_coverage
                + self.query_selection_weight * query_coverage
                + self.thread_diversity_weight * represented_mass
                + role_weight * role_coverage
            )
            if (
                len(selected_indices) >= self.min_evidence
                and (not require_two_threads or len(selected_threads) >= 2)
                and total_coverage >= self.mass_threshold
            ):
                break

        selected = [candidates[index] for index in selected_indices]
        return selected, min(1.0, total_coverage), gains

    @staticmethod
    def _role(
        candidate: Dict[str, Any],
        selected: Sequence[Dict[str, Any]],
        index: int,
        question_type: str,
    ) -> str:
        # Role-aware candidates already carry the frozen query-local
        # assignment. Legacy candidates retain the historical post-hoc labels
        # so the default path remains reproducible.
        assigned = str(candidate.get("role") or "").strip()
        if assigned in ROLE_NAMES and candidate.get("role_distribution"):
            return assigned
        return ThreadWorkspaceBuilder._legacy_contribution(
            candidate, selected, index, question_type
        )

    @staticmethod
    def _legacy_contribution(
        candidate: Dict[str, Any],
        selected: Sequence[Dict[str, Any]],
        index: int,
        question_type: str,
    ) -> str:
        """Return the historical post-selection label used by the answer prompt."""
        if "temporal" in question_type.lower() and candidate["timestamp"]:
            return "temporal-anchor"
        if index == 1:
            return "anchor"
        first = selected[0]
        if candidate["thread_id"] != first["thread_id"]:
            return "cross-thread-support"
        if any(
            previous["thread_id"] == candidate["thread_id"]
            and abs(previous["ordinal"] - candidate["ordinal"]) == 1
            for previous in selected[: index - 1]
        ):
            return "bridge" if "multi" in question_type.lower() else "continuation"
        return "support"

    @staticmethod
    def _role_profile(candidate: Dict[str, Any]) -> Dict[str, float]:
        """Return a normalized ephemeral role posterior for one node."""
        raw = candidate.get("role_distribution") or {}
        values = {
            role: max(0.0, float(raw.get(role, 0.0)))
            for role in ROLE_NAMES
        }
        total = sum(values.values())
        if total <= 0.0:
            return {role: 1.0 / len(ROLE_NAMES) for role in ROLE_NAMES}
        return {role: values[role] / total for role in ROLE_NAMES}

    def _relations(
        self,
        evidence: Sequence[WorkspaceEvidence],
        expanded: Sequence[ThreadCandidate],
    ) -> List[WorkspaceRelation]:
        relations: List[WorkspaceRelation] = []
        by_thread_link = {
            (item.thread.thread_id, link.target_thread_id): link.relation
            for item in expanded
            for link in item.thread.links
        }
        for left_index, left in enumerate(evidence):
            for right in evidence[left_index + 1 :]:
                relation_type: Optional[str] = None
                if left.thread_id == right.thread_id and abs(left.ordinal - right.ordinal) == 1:
                    relation_type = "thread-continuation"
                elif (left.thread_id, right.thread_id) in by_thread_link:
                    relation_type = by_thread_link[(left.thread_id, right.thread_id)]
                elif (right.thread_id, left.thread_id) in by_thread_link:
                    relation_type = by_thread_link[(right.thread_id, left.thread_id)]
                if relation_type:
                    relations.append(
                        WorkspaceRelation(
                            source_evidence_id=left.evidence_id,
                            target_evidence_id=right.evidence_id,
                            relation_type=relation_type,
                        )
                    )
                if len(relations) >= self.max_relations:
                    return relations
        return relations


def build_thread_workspace(
    query: str,
    address: str,
    query_vector: Sequence[float],
    routes: Sequence[Any],
    expanded: Sequence[ThreadCandidate],
    plan: Optional[Dict[str, Any]] = None,
    *,
    max_evidence: int = 16,
    min_evidence: int = 1,
    max_relations: int = 12,
    restart_probability: float = 0.25,
    mass_threshold: float = 0.70,
    query_prior_weight: float = 0.55,
    query_selection_weight: float = 0.52,
    node_seed_weight: float = 0.80,
    thread_diversity_weight: float = 0.05,
    flow_temperature: float = 0.35,
    flow_mismatch_weight: float = 1.25,
    role_aware: bool = False,
    role_scope: str = "full",
    role_transport_weight: float = 0.22,
    role_selection_weight: float = 0.10,
    role_temperature: float = 0.65,
) -> MemoryWorkspace:
    return ThreadWorkspaceBuilder(
        max_evidence=max_evidence,
        min_evidence=min_evidence,
        max_relations=max_relations,
        restart_probability=restart_probability,
        mass_threshold=mass_threshold,
        query_prior_weight=query_prior_weight,
        query_selection_weight=query_selection_weight,
        node_seed_weight=node_seed_weight,
        thread_diversity_weight=thread_diversity_weight,
        flow_temperature=flow_temperature,
        flow_mismatch_weight=flow_mismatch_weight,
        role_aware=role_aware,
        role_scope=role_scope,
        role_transport_weight=role_transport_weight,
        role_selection_weight=role_selection_weight,
        role_temperature=role_temperature,
    ).build(query, address, query_vector, routes, expanded, plan)
