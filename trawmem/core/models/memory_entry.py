"""Data structures for thread-addressed conversational memory."""

from typing import List, Optional
import uuid

from pydantic import BaseModel, Field


class ThreadNode(BaseModel):
    """One original dialogue turn in an interaction-thread ledger."""

    node_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str
    source_turn_ids: List[str] = Field(default_factory=list)
    ordinal: int = 0
    timestamp: Optional[str] = None
    speakers: List[str] = Field(default_factory=list)
    # Nodes are embedded/scored only after their parent thread has been routed.
    embedding: List[float] = Field(default_factory=list)


class ThreadLink(BaseModel):
    """A persistent edge for local traversal between interaction threads."""

    target_thread_id: str
    relation: str
    weight: float = 0.0
    anchors: List[str] = Field(default_factory=list)


class InteractionThread(BaseModel):
    """Persistent memory unit addressed by its head and expanded locally."""

    thread_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    head: str
    session_id: Optional[str] = None
    ordinal: int = 0
    start_timestamp: Optional[str] = None
    end_timestamp: Optional[str] = None
    nodes: List[ThreadNode] = Field(default_factory=list)
    links: List[ThreadLink] = Field(default_factory=list)
    embedding: List[float] = Field(default_factory=list)

    @property
    def routing_text(self) -> str:
        """One query-independent trajectory representation for global routing.

        A compact LLM synopsis can omit the names, dates, and concrete objects
        that appear in the original ledger.  The synopsis and bounded raw-turn
        anchors are therefore embedded together as one representation.  This
        is still one persistent thread view, rather than separate semantic,
        keyword, and structured indexes.
        """
        parts = [self.head.strip()]
        for node in sorted(self.nodes, key=lambda item: (item.ordinal, item.node_id)):
            timestamp = f" [{node.timestamp}]" if node.timestamp else ""
            speakers = ",".join(node.speakers)
            speaker_prefix = f" {speakers}:" if speakers else ""
            parts.append(f"{timestamp}{speaker_prefix} {node.text}".strip())
        # Bound only the routing document; the payload ledger remains lossless.
        return "\n".join(part for part in parts if part)[:6000]


class Dialogue(BaseModel):
    """Original dialogue turn with dataset-level provenance."""

    dialogue_id: int
    speaker: str
    content: str
    timestamp: Optional[str] = None
    source_id: Optional[str] = None
    session_id: Optional[str] = None
    turn_index: Optional[int] = None

    @property
    def source_turn_id(self) -> str:
        return str(self.source_id or self.dialogue_id)

    def __str__(self) -> str:
        time_str = f"[{self.timestamp}] " if self.timestamp else ""
        return f"{time_str}{self.speaker}: {self.content}"


class MemoryEntry(BaseModel):
    """Compatibility view for answer/evaluator integrations.

    The thread path does not use this flat object as a persistent index; it is
    created only when a caller requests a flat compatibility result.
    """

    entry_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    lossless_restatement: str
    timestamp: Optional[str] = None
    source_turn_ids: List[str] = Field(default_factory=list)
    thread_id: Optional[str] = None
