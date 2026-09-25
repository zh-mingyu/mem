"""
TrawMem runtime settings.

Resolution order for each attribute:
  1. User's top-level ``config.py`` (if importable on sys.path)
  2. Environment variable of the same name
  3. Built-in default
"""
import os


_DEFAULTS = {
    "OPENAI_API_KEY": "",
    "OPENAI_BASE_URL": None,
    "OPENROUTER_API_KEY": "",
    "OPENROUTER_BASE_URL": None,
    "LLM_MODEL": "gpt-4.1-mini",
    "TOKENIZER_MODEL_PATH": None,
    "EMBEDDING_MODEL": "Qwen/Qwen3-Embedding-0.6B",
    "EMBEDDING_DIMENSION": 1024,
    "EMBEDDING_CONTEXT_LENGTH": 32768,
    "ENABLE_THINKING": False,
    "USE_STREAMING": True,
    "USE_JSON_FORMAT": True,
    "TEMPERATURE": 0.3,
    "MAX_TOKENS": 4096,
    "WINDOW_SIZE": 20,
    "THREAD_TURN_LIMIT": 10,
    "THREAD_LANCEDB_PATH": "./thread_lancedb_data",
    "THREAD_TABLE_NAME": "event_threads_v3",
    "THREAD_ROUTE_TOP_K": 16,
    "THREAD_ROUTE_PROBE_TOP_K": 12,
    "THREAD_ROUTE_PROBE_MARGIN": 0.02,
    "THREAD_MAX_HOPS": 3,
    "THREAD_MAX_EXPANDED": 32,
    "THREAD_ADAPTIVE_MAX_EXPANDED": 24,
    "THREAD_GRAPH_NEIGHBORS": 5,
    "THREAD_GRAPH_MIN_SIM": 0.25,
    "ENABLE_THREAD_ADDRESS_PLANNER": False,
    "ENABLE_PARALLEL_PROCESSING": True,
    "MAX_PARALLEL_WORKERS": 16,
    "ENABLE_PARALLEL_RETRIEVAL": True,
    "MAX_RETRIEVAL_WORKERS": 8,
    "ENABLE_MEMORY_WORKSPACE": True,
    "WORKSPACE_MAX_EVIDENCE": 16,
    "WORKSPACE_MIN_EVIDENCE": 1,
    "WORKSPACE_MAX_RELATIONS": 12,
    # This role snapshot matches the controlled Mass70 evaluation.
    "WORKSPACE_ACTIVATION_MASS": 0.70,
    "WORKSPACE_QUERY_SELECTION_WEIGHT": 0.52,
    "WORKSPACE_NODE_SEED_WEIGHT": 0.80,
    "WORKSPACE_THREAD_DIVERSITY_WEIGHT": 0.05,
    "WORKSPACE_FLOW_TEMPERATURE": 0.35,
    "WORKSPACE_FLOW_MISMATCH_WEIGHT": 1.25,
    # Query-local one-shot roles are enabled in the canonical snapshot. The
    # full scope lets the frozen posterior condition every local workspace
    # stage; selection-only remains an explicit ablation.
    "WORKSPACE_ROLE_AWARE": True,
    "WORKSPACE_ROLE_SCOPE": "full",
    "WORKSPACE_ROLE_TRANSPORT_WEIGHT": 0.22,
    "WORKSPACE_ROLE_SELECTION_WEIGHT": 0.10,
    "WORKSPACE_ROLE_TEMPERATURE": 0.65,
}


def _coerce(default, raw):
    if default is None or isinstance(default, str):
        return raw
    if isinstance(default, bool):
        return raw.strip().lower() in ("1", "true", "yes", "on")
    if isinstance(default, int):
        return int(raw)
    if isinstance(default, float):
        return float(raw)
    return raw


class Settings:
    def __init__(self):
        try:
            import config as _user_config
        except ImportError:
            _user_config = None
        self._user_config = _user_config

    def __getattr__(self, name):
        # __getattr__ only fires when normal lookup fails, so no recursion via _user_config.
        user_cfg = self.__dict__.get("_user_config")
        if user_cfg is not None and hasattr(user_cfg, name):
            return getattr(user_cfg, name)
        env_val = os.getenv(name)
        if name in _DEFAULTS:
            if env_val is not None:
                return _coerce(_DEFAULTS[name], env_val)
            return _DEFAULTS[name]
        if env_val is not None:
            return env_val
        raise AttributeError(
            f"Setting {name!r} is not defined. Set it in config.py or as env var."
        )


settings = Settings()
