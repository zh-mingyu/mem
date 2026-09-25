"""
Utils package
"""
try:
    from .llm_client import LLMClient
except ImportError:  # Optional in the dependency-light structural test runtime.
    LLMClient = None
from .embedding import EmbeddingModel

__all__ = ['LLMClient', 'EmbeddingModel']
