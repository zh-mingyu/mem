"""The evaluator must never turn failed LLM calls into benchmark answers."""

from types import SimpleNamespace

import pytest

from trawmem.core.answer_generator import AnswerGenerationError, AnswerGenerator
from trawmem.core.models.memory_entry import MemoryEntry
from trawmem.core.memory_builder import MemoryBuilder
from trawmem.core.models.memory_entry import Dialogue


class FailingLLM:
    def chat_completion(self, *_args, **_kwargs):
        raise RuntimeError("simulated API outage")


def test_answer_generation_raises_after_api_failure():
    generator = AnswerGenerator(FailingLLM())
    context = [MemoryEntry(lossless_restatement="A fact.")]

    with pytest.raises(AnswerGenerationError, match="answer generation failed"):
        generator.generate_answer("What happened?", context)


def test_category5_failure_never_returns_gold_abstention():
    from test_locomo10 import LoCoMoTester

    tester = object.__new__(LoCoMoTester)
    tester.system = SimpleNamespace(
        llm_client=FailingLLM(),
        answer_generator=SimpleNamespace(_format_contexts=lambda _contexts: "A fact."),
    )

    with pytest.raises(AnswerGenerationError):
        tester.generate_category5_answer("Unsupported?", [], "Invented answer")


def test_judge_failure_is_not_scored_as_zero():
    from test_locomo10 import calculate_metrics

    class FailingJudge:
        def chat_completion(self, *_args, **_kwargs):
            raise RuntimeError("judge outage")

    with pytest.raises(RuntimeError, match="LLM judge failed"):
        calculate_metrics(
            "answer",
            "reference",
            question="Question?",
            judge_client=FailingJudge(),
            use_llm_judge=True,
        )


def test_binary_judge_parser_accepts_only_an_unambiguous_first_verdict():
    from test_memgallery import _parse_binary_judge_response

    assert _parse_binary_judge_response("YES.") is True
    assert _parse_binary_judge_response("no\nbrief explanation") is False
    with pytest.raises(RuntimeError, match="expected YES or NO"):
        _parse_binary_judge_response("The answer is probably yes")


def test_thread_segmentation_failure_does_not_silently_use_raw_chunks():
    builder = MemoryBuilder(FailingLLM(), object())
    with pytest.raises(RuntimeError, match="refusing raw-turn fallback"):
        builder._extract_thread_bundle([
            Dialogue(dialogue_id=1, speaker="User", content="fact", source_id="D1:1")
        ])
