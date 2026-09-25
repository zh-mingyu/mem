"""Dependency-light tests for the active thread workspace path."""

import json
import tempfile

import numpy as np

from trawmem.core.memory_builder import MemoryBuilder
from trawmem.core.models.memory_entry import Dialogue, InteractionThread, ThreadNode
from trawmem.core.thread_store import ThreadStore
from trawmem.core.hybrid_retriever import ThreadWorkspaceRetriever
from trawmem.core.workspace import ThreadWorkspaceBuilder, build_thread_workspace
from trawmem.core.answer_generator import AnswerGenerator


class FakeEmbedding:
    dimension = 4

    def encode_documents(self, texts):
        return np.asarray(
            [
                [float("road" in text.lower()), float("accident" in text.lower()),
                 float("future" in text.lower()), 1.0]
                for text in texts
            ],
            dtype="float32",
        )

    def encode_single(self, text, is_query=False):
        text = text.lower()
        return np.asarray(
            [float("road" in text), float("accident" in text), float("future" in text), 1.0],
            dtype="float32",
        )


class FakeLLM:
    def chat_completion(self, *_args, **_kwargs):
        return json.dumps({
            "thread_head": "Caroline support group and education",
            "nodes": [{
                "text": "Caroline attended an LGBTQ support group.",
                "source_turn_ids": ["D1:3"],
                "ordinal": 0,
            }],
        })

    @staticmethod
    def extract_json(response):
        return json.loads(response)


class BundleLLM:
    def chat_completion(self, *_args, **_kwargs):
        return json.dumps({
            "threads": [
                {
                    "address_synopsis": "support group disclosure",
                    "source_turn_ids": ["D2:1", "D2:2"],
                    "nodes": [{
                        "text": "Caroline attended a support group.",
                        "source_turn_ids": ["D2:1"],
                        "ordinal": 0,
                    }],
                },
                {
                    "address_synopsis": "education plan",
                    "source_turn_ids": ["D2:3", "D2:4"],
                    "nodes": [{
                        "text": "Caroline plans further education.",
                        "source_turn_ids": ["D2:3"],
                        "ordinal": 0,
                    }],
                },
            ]
        })

    @staticmethod
    def extract_json(response):
        return json.loads(response)


class NoopPlannerLLM:
    """The default path must not call the retrieval-time planner."""

    def chat_completion(self, *_args, **_kwargs):
        raise AssertionError("retrieval-time planning should be disabled")

    @staticmethod
    def extract_json(response):
        return json.loads(response)


def _threads():
    return [
        InteractionThread(
            thread_id="t1",
            head="Melanie road trip and accident",
            session_id="D1",
            ordinal=0,
            nodes=[
                ThreadNode(
                    node_id="t1:n1",
                    text="Melanie took a road trip with her son.",
                    source_turn_ids=["D1:1"],
                    ordinal=0,
                    timestamp="2023-06-10",
                    speakers=["Melanie"],
                ),
                ThreadNode(
                    node_id="t1:n2",
                    text="Melanie's son was injured in a car accident.",
                    source_turn_ids=["D1:2"],
                    ordinal=1,
                    timestamp="2023-06-12",
                    speakers=["Melanie"],
                ),
            ],
        ),
        InteractionThread(
            thread_id="t2",
            head="Melanie future road trip",
            session_id="D2",
            ordinal=1,
            nodes=[ThreadNode(
                node_id="t2:n1",
                text="Melanie would consider another road trip.",
                source_turn_ids=["D2:1"],
                ordinal=0,
                speakers=["Melanie"],
            )],
        ),
    ]


def test_thread_builder_preserves_missing_turn_provenance():
    with tempfile.TemporaryDirectory() as directory:
        store = ThreadStore(directory, embedding_model=FakeEmbedding(), table_name="threads")
        builder = MemoryBuilder(FakeLLM(), store, window_size=20)
        builder.add_dialogues([
            Dialogue(dialogue_id=1, speaker="Caroline", content="group", source_id="D1:3", session_id="D1"),
            Dialogue(dialogue_id=2, speaker="Caroline", content="education", source_id="D1:4", session_id="D1"),
        ])
        builder.process_remaining()
        thread = store.get_all_threads()[0]
        nodes = thread.nodes
        assert {source for node in nodes for source in node.source_turn_ids} == {"D1:3", "D1:4"}
        assert nodes[0].text == "Caroline: group"
        assert nodes[1].text == "Caroline: education"
        assert not hasattr(thread, "continuity_keys")
        assert all(not hasattr(node, "continuity_keys") for node in nodes)


def test_event_bundle_creates_multiple_contiguous_threads():
    with tempfile.TemporaryDirectory() as directory:
        store = ThreadStore(directory, embedding_model=FakeEmbedding(), table_name="threads")
        builder = MemoryBuilder(BundleLLM(), store, window_size=20)
        builder.add_dialogues([
            Dialogue(dialogue_id=1, speaker="Caroline", content="group", source_id="D2:1", session_id="D2"),
            Dialogue(dialogue_id=2, speaker="Caroline", content="support", source_id="D2:2", session_id="D2"),
            Dialogue(dialogue_id=3, speaker="Caroline", content="school", source_id="D2:3", session_id="D2"),
            Dialogue(dialogue_id=4, speaker="Caroline", content="course", source_id="D2:4", session_id="D2"),
        ])
        builder.process_remaining()
        threads = store.get_all_threads()
        assert len(threads) == 2
        assert [thread.session_id for thread in threads] == ["D2", "D2"]
        assert [
            source_id
            for thread in threads
            for node in thread.nodes
            for source_id in node.source_turn_ids
        ] == ["D2:1", "D2:2", "D2:3", "D2:4"]
        assert threads[0].links[0].relation == "chronology-next"


def test_local_workspace_selects_connected_multi_evidence():
    with tempfile.TemporaryDirectory() as directory:
        store = ThreadStore(directory, embedding_model=FakeEmbedding(), table_name="threads")
        store.add_threads(_threads())
        routes, vector = store.route("accident", top_k=1)
        expanded = store.expand(routes, max_hops=2, max_threads=4)
        workspace = build_thread_workspace(
            "Why might Melanie take another road trip after the accident?",
            "another road trip after accident",
            vector,
            routes,
            expanded,
            {
                "question_type": "multi-hop",
                "complexity": "HIGH",
            },
            max_evidence=4,
        )
        assert len(workspace.evidence) >= 2
        assert {source for item in workspace.evidence for source in item.source_turn_ids} >= {"D1:2", "D2:1"}
        assert workspace.sufficient
        assert workspace.coverage_mode == "capacitated-greedy-heuristic"
        assert workspace.activation_iterations > 0
        assert workspace.transport_flow_units > 0
        assert workspace.transport_cost >= 0.0
        assert workspace.graph_edge_count > 0
        assert workspace.global_route_calls == 1
        assert not hasattr(workspace, "requirements")
        assert not hasattr(workspace.evidence[0], "requirement_ids")
        assert workspace.selection_budget == 4
        assert workspace.selection_rounds == len(workspace.evidence)


def test_same_node_role_changes_with_query():
    with tempfile.TemporaryDirectory() as directory:
        store = ThreadStore(directory, embedding_model=FakeEmbedding(), table_name="threads")
        store.add_threads(_threads())
        routes, vector = store.route("accident", top_k=1)
        expanded = store.expand(routes, max_hops=0, max_threads=1)
        direct = build_thread_workspace(
            "What happened to the son?", "son accident", vector, routes, expanded,
            {"question_type": "single-hop"},
            max_evidence=2,
        )
        temporal = build_thread_workspace(
            "When did the accident happen?", "time of accident", vector, routes, expanded,
            {"question_type": "temporal"},
            max_evidence=2,
        )
        direct_item = next(item for item in direct.evidence if item.node_id == "t1:n2")
        temporal_item = next(item for item in temporal.evidence if item.node_id == "t1:n2")
        assert direct_item.contribution != temporal_item.contribution
    assert temporal_item.contribution == "temporal-anchor"


def test_role_demand_recognizes_all_documented_temporal_forms():
    for query in (
        "What month did the accident happen?",
        "What day did the accident happen?",
        "How many years later did it happen?",
    ):
        demand = ThreadWorkspaceBuilder._role_demand(query)
        assert demand["context"] > demand["anchor"]


def test_one_shot_roles_are_query_local_and_change_local_assembly():
    """One-shot roles must be active workspace state, not display metadata."""
    with tempfile.TemporaryDirectory() as directory:
        store = ThreadStore(directory, embedding_model=FakeEmbedding(), table_name="threads")
        store.add_threads(_threads())
        routes, vector = store.route("accident", top_k=1)
        expanded = store.expand(routes, max_hops=0, max_threads=1)
        direct = build_thread_workspace(
            "What happened to the son?",
            "son accident",
            vector,
            routes,
            expanded,
            {"question_type": "single-hop"},
            max_evidence=2,
            min_evidence=2,
            mass_threshold=1.0,
            role_aware=True,
        )
        temporal = build_thread_workspace(
            "When did the accident happen?",
            "time of accident",
            vector,
            routes,
            expanded,
            {"question_type": "temporal"},
            max_evidence=2,
            min_evidence=2,
            mass_threshold=1.0,
            role_aware=True,
        )
    assert direct.role_mode == "one-shot"
    assert direct.role_frozen
    assert direct.role_assignment_count == 2
    direct_item = next(item for item in direct.evidence if item.node_id == "t1:n2")
    temporal_item = next(item for item in temporal.evidence if item.node_id == "t1:n2")
    assert abs(sum(direct_item.role_distribution.values()) - 1.0) < 1e-6
    assert abs(sum(temporal_item.role_distribution.values()) - 1.0) < 1e-6
    assert temporal_item.role_distribution["context"] > direct_item.role_distribution["context"]

    candidates = [
        {
            "thread_id": "t1", "node_id": "n-anchor", "ordinal": 0,
            "thread_rank": 1, "dense_score": 0.7, "query_affinity": 0.7,
            "activation_score": 0.5, "node_embedding": [1.0, 0.0],
            "role_distribution": {"anchor": 1.0, "bridge": 0.0, "context": 0.0},
        },
        {
            "thread_id": "t1", "node_id": "n-context", "ordinal": 1,
            "thread_rank": 1, "dense_score": 0.7, "query_affinity": 0.7,
            "activation_score": 0.5, "node_embedding": [0.0, 1.0],
            "role_distribution": {"anchor": 0.0, "bridge": 0.0, "context": 1.0},
        },
    ]
    legacy = ThreadWorkspaceBuilder(max_evidence=1)
    role_aware = ThreadWorkspaceBuilder(
        max_evidence=1,
        role_aware=True,
        role_selection_weight=0.30,
        query_selection_weight=0.30,
        thread_diversity_weight=0.0,
    )
    legacy_kernel = legacy._kernel(candidates, [], question_type="general")
    role_kernel = role_aware._kernel(
        candidates,
        [],
        question_type="general",
        role_demand={"anchor": 1.0, "bridge": 0.0, "context": 0.0},
    )
    assert role_kernel[0][1] < legacy_kernel[0][1]
    selected, _coverage, _gains = role_aware._select(
        candidates,
        role_kernel,
        role_demand={"anchor": 0.0, "bridge": 0.0, "context": 1.0},
    )
    assert selected[0]["node_id"] == "n-context"


def test_role_state_never_mutates_persistent_thread_bank():
    """Role assignments belong to one workspace and cannot leak into storage."""
    with tempfile.TemporaryDirectory() as directory:
        store = ThreadStore(directory, embedding_model=FakeEmbedding(), table_name="threads")
        store.add_threads(_threads())
        before = [thread.model_dump(mode="json") for thread in store.get_all_threads()]

        routes, vector = store.route("accident", top_k=1)
        expanded = store.expand(routes, max_hops=1, max_threads=2)
        workspace = build_thread_workspace(
            "When did the accident happen?",
            "accident date",
            vector,
            routes,
            expanded,
            {"question_type": "temporal"},
            max_evidence=3,
            min_evidence=1,
            role_aware=True,
        )

        after = [thread.model_dump(mode="json") for thread in store.get_all_threads()]
        assert before == after
        assert workspace.role_mode == "one-shot"
        assert workspace.role_frozen
        assert workspace.evidence
        assert all("role" not in node for thread in after for node in thread["nodes"])
        assert all("role_distribution" not in node for thread in after for node in thread["nodes"])


def test_selection_only_roles_leave_transport_kernel_and_prompt_legacy():
    candidates = [
        {
            "thread_id": "t1", "node_id": "n-anchor", "ordinal": 0,
            "thread_rank": 1, "dense_score": 0.7, "query_affinity": 0.7,
            "activation_score": 0.5, "node_embedding": [1.0, 0.0],
            "timestamp": None,
            "role_fit": 0.2,
            "role": "anchor",
            "role_distribution": {"anchor": 1.0, "bridge": 0.0, "context": 0.0},
        },
        {
            "thread_id": "t1", "node_id": "n-context", "ordinal": 1,
            "thread_rank": 1, "dense_score": 0.7, "query_affinity": 0.7,
            "activation_score": 0.5, "node_embedding": [0.0, 1.0],
            "timestamp": None,
            "role_fit": 0.8,
            "role": "context",
            "role_distribution": {"anchor": 0.0, "bridge": 0.0, "context": 1.0},
        },
    ]
    legacy = ThreadWorkspaceBuilder(max_evidence=1)
    selection_only = ThreadWorkspaceBuilder(
        max_evidence=1,
        role_aware=True,
        role_scope="selection-only",
        role_selection_weight=0.30,
        query_selection_weight=0.30,
        thread_diversity_weight=0.0,
    )
    demand = {"anchor": 0.0, "bridge": 0.0, "context": 1.0}
    legacy_kernel = legacy._kernel(candidates, [], question_type="general")
    isolated_kernel = selection_only._kernel(
        candidates, [], question_type="general", role_demand=demand
    )
    assert isolated_kernel == legacy_kernel

    adjacency = {
        "h:t1": {"n:n-anchor": 1.0, "n:n-context": 1.0},
        "n:n-anchor": {"h:t1": 1.0},
        "n:n-context": {"h:t1": 1.0},
    }
    seed = {"h:t1": 1.0}
    legacy_mass = legacy._transport_activation(adjacency, seed, candidates)[0]
    isolated_mass = selection_only._transport_activation(adjacency, seed, candidates)[0]
    assert isolated_mass == legacy_mass

    selected, _coverage, _gains = selection_only._select(
        candidates, isolated_kernel, role_demand=demand
    )
    assert selected[0]["node_id"] == "n-context"
    assert selection_only._legacy_contribution(
        candidates[1], candidates, 1, "general"
    ) == "anchor"


def test_retriever_routes_once_and_assembles_locally():
    with tempfile.TemporaryDirectory() as directory:
        store = ThreadStore(directory, embedding_model=FakeEmbedding(), table_name="threads")
        store.add_threads(_threads())
        retriever = ThreadWorkspaceRetriever(
            NoopPlannerLLM(),
            store,
            enable_planning=False,
        )
        workspace = retriever.retrieve_workspace("What happened in the accident?")
        assert workspace.global_route_calls == 1
        assert workspace.address_planner_calls == 0
        assert workspace.evidence
        assert not hasattr(workspace, "requirements")
        assert all(not hasattr(item, "matched_terms") for item in workspace.evidence)


def test_retriever_preserves_explicit_probe_and_adaptive_caps():
    """Configured route/expansion budgets must not be silently rewritten."""
    retriever = ThreadWorkspaceRetriever(llm_client=None, vector_store=None)
    assert retriever.thread_route_top_k == 16
    assert retriever.thread_route_probe_top_k == 12
    assert retriever.thread_max_expanded == 32
    assert retriever.thread_adaptive_max_expanded == 24


def test_locomo_calendar_hint_uses_session_anchor_without_iso_format():
    hint = AnswerGenerator._calendar_hints(
        "I painted that lake sunrise last year.",
        "1:56 pm on 8 May, 2023",
    )
    assert '"last year" -> 2022' in hint
    friday = AnswerGenerator._calendar_hints(
        "I went there last Friday.",
        "12:00 pm on 15 July, 2023",
    )
    assert "Friday before 15 July 2023" in friday


def test_graph_coverage_requires_content_observability():
    candidates = [
        {
            "thread_id": "t1", "node_id": "n1", "ordinal": 0,
            "node_embedding": [1.0, 0.0],
        },
        {
            "thread_id": "t1", "node_id": "n2", "ordinal": 1,
            "node_embedding": [0.0, 1.0],
        },
        {
            "thread_id": "t1", "node_id": "n3", "ordinal": 2,
            "node_embedding": [1.0, 0.0],
        },
    ]
    builder = build_thread_workspace.__globals__["ThreadWorkspaceBuilder"]()
    kernel = builder._kernel(
        candidates,
        [],
        question_type="temporal",
    )
    # Temporal observability has a non-zero floor so lexical date turns are
    # not discarded solely because their embeddings are orthogonal.
    assert kernel[0][1] > 0.0
    assert kernel[0][2] > 0.0
    compositional = builder._kernel(
        candidates,
        [],
        question_type="set-valued",
    )
    assert compositional[0][1] > 0.0


def test_structured_answer_is_normalized_to_plain_text():
    assert AnswerGenerator._answer_text(["clarinet", "violin"], "") == "clarinet, violin"
    assert AnswerGenerator._normalise_count_answer(
        "3", "How many screenplays has Joanna written?"
    ) == "three"
    assert AnswerGenerator._normalise_count_answer(
        "2023", "When did Joanna write it?"
    ) == "2023"


def test_temporal_hint_preserves_relative_phrase_and_resolves_year():
    hint = AnswerGenerator._calendar_hints(
        "I joined the group the Friday before 15 July 2023.",
        "12:00 pm on 20 July, 2023",
    )
    assert "preserve phrase" in hint
    assert "14 July 2023" in hint
    assert AnswerGenerator._normalise_temporal_answer(
        "last year", 'calendar-hint="last year" -> 2022'
    ) == "2022"
    assert AnswerGenerator._normalise_temporal_answer(
        "14 July 2023",
        'calendar-hint="the Friday before 15 July 2023" -> preserve phrase; '
        'resolved date=14 July 2023',
    ) == "the Friday before 15 July 2023"
    month_hint = AnswerGenerator._calendar_hints(
        "Our group performs next month.",
        "10:00 am on 20 January, 2023",
    )
    assert '"next month" -> February 2023' in month_hint
    assert AnswerGenerator._normalise_temporal_answer(
        "next month", month_hint
    ) == "February 2023"
    week_hint = AnswerGenerator._calendar_hints(
        "I had the picnic last week.",
        "1:00 pm on 6 July, 2023",
    )
    assert AnswerGenerator._normalise_temporal_answer(
        "29 June 2023", week_hint
    ) == "the week before 6 July 2023"


def test_temporal_canonicalization_uses_query_anchored_evidence():
    from trawmem.core.workspace import MemoryWorkspace, WorkspaceEvidence

    workspace = MemoryWorkspace(
        workspace_id="ws-test",
        query="When did Caroline have a picnic?",
        question_type="temporal",
        evidence=[
            WorkspaceEvidence(
                evidence_id="W1", thread_id="t1", node_id="n1",
                text="Gina designed hoodies last week.",
                timestamp="21 June 2023",
            ),
            WorkspaceEvidence(
                evidence_id="W2", thread_id="t2", node_id="n2",
                text="Caroline had a picnic last week.",
                timestamp="6 July 2023",
            ),
        ],
    )
    assert AnswerGenerator._normalise_workspace_temporal_answer(
        "29 June 2023",
        workspace.query,
        workspace,
    ) == "the week before 6 July 2023"


if __name__ == "__main__":
    test_thread_builder_preserves_missing_turn_provenance()
    test_event_bundle_creates_multiple_contiguous_threads()
    test_local_workspace_selects_connected_multi_evidence()
    test_same_node_role_changes_with_query()
    test_one_shot_roles_are_query_local_and_change_local_assembly()
    test_role_state_never_mutates_persistent_thread_bank()
    test_selection_only_roles_leave_transport_kernel_and_prompt_legacy()
    test_retriever_routes_once_and_assembles_locally()
    test_locomo_calendar_hint_uses_session_anchor_without_iso_format()
    test_graph_coverage_requires_content_observability()
    test_structured_answer_is_normalized_to_plain_text()
    print("thread workspace tests passed")
