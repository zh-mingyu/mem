from types import SimpleNamespace

from trawmem.core.utils import llm_client as llm_module


class _FakeCompletions:
    def __init__(self, key, calls, kwargs_seen=None):
        self.key = key
        self.calls = calls
        self.kwargs_seen = kwargs_seen

    def create(self, **kwargs):
        self.calls.append((self.key, kwargs["model"]))
        if self.kwargs_seen is not None:
            self.kwargs_seen.append(dict(kwargs))
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
            usage=SimpleNamespace(
                prompt_tokens=1,
                completion_tokens=1,
                total_tokens=2,
            ),
        )


class _FakeOpenAI:
    calls = []
    kwargs_seen = []
    failures = set()

    def __init__(self, *, base_url, api_key):
        self.chat = SimpleNamespace(
            completions=_FakeCompletions(api_key, self.calls, self.kwargs_seen)
        )


def test_comma_separated_keys_rotate_without_logging_secrets(monkeypatch):
    _FakeOpenAI.calls = []
    _FakeOpenAI.kwargs_seen = []
    _FakeOpenAI.failures = set()
    monkeypatch.setattr(llm_module, "OpenAI", _FakeOpenAI)
    client = llm_module.LLMClient(
        api_key=" key-one, key-two, key-one ",
        model="demo-model",
        base_url="https://example.invalid/v1",
        enable_thinking=False,
        use_streaming=False,
    )

    assert client.chat_completion([{"role": "user", "content": "a"}]) == "ok"
    assert client.chat_completion([{"role": "user", "content": "b"}]) == "ok"
    assert _FakeOpenAI.calls == [
        ("key-one", "demo-model"),
        ("key-two", "demo-model"),
    ]
    assert all(
        "key-one" not in str(event) and "key-two" not in str(event)
        for event in client.usage_events()
    )


def test_failed_key_falls_through_to_next_key(monkeypatch):
    class _FailingCompletions(_FakeCompletions):
        def create(self, **kwargs):
            self.calls.append((self.key, kwargs["model"]))
            if self.key == "bad-key":
                raise RuntimeError("authentication failed")
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
                usage=None,
            )

    class _FailingOpenAI:
        calls = []

        def __init__(self, *, base_url, api_key):
            self.chat = SimpleNamespace(
                completions=_FailingCompletions(api_key, self.calls)
            )

    monkeypatch.setattr(llm_module, "OpenAI", _FailingOpenAI)
    client = llm_module.LLMClient(
        api_key="bad-key,good-key",
        model="demo-model",
        base_url="https://example.invalid/v1",
        enable_thinking=False,
        use_streaming=False,
    )

    assert client.chat_completion(
        [{"role": "user", "content": "a"}], max_retries=1
    ) == "ok"
    assert _FailingOpenAI.calls == [
        ("bad-key", "demo-model"),
        ("good-key", "demo-model"),
    ]


def test_gpt5_uses_modern_completion_limit_and_omits_temperature(monkeypatch):
    _FakeOpenAI.calls = []
    _FakeOpenAI.kwargs_seen = []
    monkeypatch.setattr(llm_module, "OpenAI", _FakeOpenAI)
    client = llm_module.LLMClient(
        api_key="key",
        model="provider/gpt-5.6-luna",
        base_url="https://example.invalid/v1",
        enable_thinking=False,
        use_streaming=False,
    )

    assert client.chat_completion(
        [{"role": "user", "content": "a"}],
        temperature=0.1,
        max_tokens=256,
    ) == "ok"
    request = _FakeOpenAI.kwargs_seen[0]
    assert request["max_completion_tokens"] == 256
    assert "max_tokens" not in request
    assert "temperature" not in request


def test_non_reasoning_model_keeps_legacy_sampling_fields(monkeypatch):
    _FakeOpenAI.calls = []
    _FakeOpenAI.kwargs_seen = []
    monkeypatch.setattr(llm_module, "OpenAI", _FakeOpenAI)
    client = llm_module.LLMClient(
        api_key="key",
        model="deepseek/deepseek-v4-flash-0731",
        base_url="https://example.invalid/v1",
        enable_thinking=False,
        use_streaming=False,
    )

    assert client.chat_completion(
        [{"role": "user", "content": "a"}],
        temperature=0.1,
        max_tokens=256,
    ) == "ok"
    request = _FakeOpenAI.kwargs_seen[0]
    assert request["max_tokens"] == 256
    assert request["temperature"] == 0.1
    assert "max_completion_tokens" not in request
