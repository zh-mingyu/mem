"""Thread-addressed retrieval and workspace assembly."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from trawmem.core.models.memory_entry import MemoryEntry
from trawmem.core.prompts import (
    ADDRESS_PLANNER_SYSTEM_PROMPT,
    build_address_planner_prompt,
)
from trawmem.core.settings import settings as config
from trawmem.core.thread_store import ThreadStore
if TYPE_CHECKING:
    from trawmem.core.utils.llm_client import LLMClient
from trawmem.core.workspace import MemoryWorkspace, build_thread_workspace


class ThreadWorkspaceRetriever:
    """Route once to thread heads, then assemble evidence locally.

    The class name remains for API compatibility, but there is deliberately no
    three-channel candidate union and no per-requirement retrieval.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        vector_store: ThreadStore,
        enable_planning: Optional[bool] = None,
        enable_reflection: bool = False,
        max_reflection_rounds: int = 0,
        enable_parallel_retrieval: bool = True,
        max_retrieval_workers: int = 1,
    ) -> None:
        self.llm_client = llm_client
        self.vector_store = vector_store
        self.thread_route_top_k = int(getattr(config, "THREAD_ROUTE_TOP_K", 16))
        # Probe a small tail once, then admit it only when the query signals
        # that one routed trajectory is unlikely to cover the answer. This
        # keeps the global operation single-pass while making the route budget
        # query-conditioned.
        # Keep the probe budget independent from the core route cap.  The
        # canonical snapshot deliberately probes 12 heads while retaining a
        # configurable core ceiling of 16; using ``max`` here silently
        # changes both the reported budget and the adaptive-tail behavior.
        self.thread_route_probe_top_k = max(
            1,
            int(getattr(config, "THREAD_ROUTE_PROBE_TOP_K", 12)),
        )
        self.thread_route_probe_margin = max(
            0.0,
            float(getattr(config, "THREAD_ROUTE_PROBE_MARGIN", 0.02)),
        )
        self.thread_max_hops = int(getattr(config, "THREAD_MAX_HOPS", 3))
        self.thread_max_expanded = int(getattr(config, "THREAD_MAX_EXPANDED", 32))
        # ``M_adapt`` is an explicit query-conditioned cap and may be lower
        # than the ordinary cap to avoid admitting a wider, noisier frontier.
        self.thread_adaptive_max_expanded = max(
            1,
            int(getattr(config, "THREAD_ADAPTIVE_MAX_EXPANDED", 24)),
        )
        self.workspace_max_evidence = int(getattr(config, "WORKSPACE_MAX_EVIDENCE", 16))
        self.workspace_min_evidence = int(getattr(config, "WORKSPACE_MIN_EVIDENCE", 1))
        self.workspace_max_relations = int(getattr(config, "WORKSPACE_MAX_RELATIONS", 12))
        self.workspace_activation_mass = float(
            getattr(config, "WORKSPACE_ACTIVATION_MASS", 0.70)
        )
        self.workspace_query_selection_weight = float(
            getattr(config, "WORKSPACE_QUERY_SELECTION_WEIGHT", 0.52)
        )
        self.workspace_node_seed_weight = float(
            getattr(config, "WORKSPACE_NODE_SEED_WEIGHT", 0.80)
        )
        self.workspace_thread_diversity_weight = float(
            getattr(config, "WORKSPACE_THREAD_DIVERSITY_WEIGHT", 0.05)
        )
        self.workspace_flow_temperature = float(
            getattr(config, "WORKSPACE_FLOW_TEMPERATURE", 0.35)
        )
        self.workspace_flow_mismatch_weight = float(
            getattr(config, "WORKSPACE_FLOW_MISMATCH_WEIGHT", 1.25)
        )
        self.workspace_role_aware = bool(
            getattr(config, "WORKSPACE_ROLE_AWARE", False)
        )
        self.workspace_role_scope = str(
            getattr(config, "WORKSPACE_ROLE_SCOPE", "full")
        )
        self.workspace_role_transport_weight = float(
            getattr(config, "WORKSPACE_ROLE_TRANSPORT_WEIGHT", 0.22)
        )
        self.workspace_role_selection_weight = float(
            getattr(config, "WORKSPACE_ROLE_SELECTION_WEIGHT", 0.10)
        )
        self.workspace_role_temperature = float(
            getattr(config, "WORKSPACE_ROLE_TEMPERATURE", 0.65)
        )
        self.enable_planning = (
            bool(enable_planning)
            if enable_planning is not None
            else bool(getattr(config, "ENABLE_THREAD_ADDRESS_PLANNER", False))
        )
        # Reflection is intentionally not part of the main method: it would
        # turn local assembly back into repeated query-conditioned retrieval.
        self.enable_reflection = False

    def retrieve_workspace(
        self,
        query: str,
        enable_reflection: Optional[bool] = None,
    ) -> MemoryWorkspace:
        plan = self._plan_address(query) if self.enable_planning else {}
        address = str(plan.get("address") or query).strip() or query
        probed_routes, query_vector = self.vector_store.route(
            address,
            self.thread_route_probe_top_k,
        )
        adaptive_route = self._needs_broad_route(query)
        if adaptive_route:
            core_routes = probed_routes[: self.thread_route_top_k]
            tail_score = core_routes[-1].score if core_routes else 0.0
            # A probe route is admitted only when its head score is close to
            # the core boundary. Low-confidence tails remain available to the
            # one-pass probe for diagnostics but cannot dilute the workspace.
            probe_routes = [
                route
                for route in probed_routes[self.thread_route_top_k :]
                if route.score + self.thread_route_probe_margin >= tail_score
            ]
            routes = core_routes + probe_routes
            expanded_budget = self.thread_adaptive_max_expanded
        else:
            routes = probed_routes[: self.thread_route_top_k]
            expanded_budget = self.thread_max_expanded
        expanded = self.vector_store.expand(
            routes,
            max_hops=self.thread_max_hops,
            max_threads=expanded_budget,
        )
        workspace = build_thread_workspace(
            query=query,
            address=address,
            query_vector=query_vector,
            routes=routes,
            expanded=expanded,
            plan=plan,
            max_evidence=self.workspace_max_evidence,
            min_evidence=self.workspace_min_evidence,
            max_relations=self.workspace_max_relations,
            mass_threshold=self.workspace_activation_mass,
            query_selection_weight=self.workspace_query_selection_weight,
            node_seed_weight=self.workspace_node_seed_weight,
            thread_diversity_weight=self.workspace_thread_diversity_weight,
            flow_temperature=self.workspace_flow_temperature,
            flow_mismatch_weight=self.workspace_flow_mismatch_weight,
            role_aware=self.workspace_role_aware,
            role_scope=self.workspace_role_scope,
            role_transport_weight=self.workspace_role_transport_weight,
            role_selection_weight=self.workspace_role_selection_weight,
            role_temperature=self.workspace_role_temperature,
        )
        workspace.address_planner_calls = 1 if self.enable_planning else 0
        workspace.indexed_thread_count = self.vector_store.count()
        workspace.local_edge_visits += self.vector_store.last_expansion_edges
        workspace.route_probe_count = len(probed_routes)
        workspace.adaptive_route = adaptive_route
        workspace.route_budget = len(routes)
        workspace.expansion_budget = expanded_budget
        print(
            f"[ThreadWorkspace] routed={len(routes)} expanded={len(expanded)} "
            f"nodes={workspace.node_candidate_count} evidence={len(workspace.evidence)} "
            f"mass={workspace.selected_activation_mass:.3f}"
        )
        return workspace

    @staticmethod
    def _needs_broad_route(query: str) -> bool:
        """Detect structural breadth without predicting a benchmark label.

        The gate uses only query form: explicit relations, open-ended
        inference, or a long compound constraint. It is intentionally
        conservative for short factual questions, where extra trajectories
        mostly dilute the fixed answer context.
        """
        text = str(query or "").lower()
        words = re.findall(r"\b\w+\b", text)
        relation = bool(re.search(
            r"\b(both|in common|each other|between|shared|relationship|"
            r"compare|difference|similar|same|after|before|because|"
            r"what led|in light of|considering)\b",
            text,
        ))
        open_inference = bool(re.search(
            r"\b(would|could|likely|might|personality|political|future|"
            r"interested|prefer|open to|potentially|appropriate|advice|"
            r"suggest|recommend)\b",
            text,
        ))
        compound = len(words) >= 11 and bool(re.search(
            r"\b(and|or|while|that|from|during|after|before)\b", text
        ))
        # Temporal questions benefit from a slightly wider one-pass probe:
        # the answer may live in a neighboring session whose head is just
        # below the core route boundary.  This remains a bounded route
        # expansion, not a second retrieval pass.
        temporal = bool(re.search(
            r"\b(when|what date|what year|which year|how long|how many years|"
            r"what month|what day|first time|last time|recently)\b",
            text,
        ))
        return relation or open_inference or compound or temporal

    def retrieve(self, query: str, enable_reflection: Optional[bool] = None) -> List[MemoryEntry]:
        """Compatibility API returning the selected workspace nodes as flat items."""
        workspace = self.retrieve_workspace(query, enable_reflection=enable_reflection)
        return [
            MemoryEntry(
                entry_id=item.node_id,
                lossless_restatement=item.text,
                timestamp=item.timestamp,
                source_turn_ids=item.source_turn_ids,
                thread_id=item.thread_id,
            )
            for item in workspace.evidence
        ]

    def _plan_address(self, query: str) -> Dict[str, Any]:
        """Create one address plus rendering diagnostics; never emit search queries."""
        prompt = build_address_planner_prompt(query)
        messages = [
            {"role": "system", "content": ADDRESS_PLANNER_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        try:
            response_format = {"type": "json_object"} if getattr(config, "USE_JSON_FORMAT", False) else None
            response = self.llm_client.chat_completion(
                messages,
                temperature=0.0,
                response_format=response_format,
                stage="retrieval.thread_address",
            )
            plan = self.llm_client.extract_json(response)
            if not isinstance(plan, dict):
                raise ValueError("thread address planner must return an object")
            return plan
        except Exception as error:
            print(f"[ThreadWorkspace] address planning failed; using question: {error}")
            return {
                "address": query,
                "question_type": "general",
                "complexity": "LOW",
                "temporal_direction": "both",
            }


# Compatibility name used by the copied evaluator and main entry point.
HybridRetriever = ThreadWorkspaceRetriever
