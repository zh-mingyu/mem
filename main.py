"""TrawMem system entry point."""
from typing import List, Optional
from trawmem.core.models.memory_entry import Dialogue, MemoryEntry
from trawmem.core.utils.llm_client import LLMClient
from trawmem.core.utils.embedding import EmbeddingModel
from trawmem.core.thread_store import ThreadStore
from trawmem.core.memory_builder import MemoryBuilder
from trawmem.core.hybrid_retriever import ThreadWorkspaceRetriever
from trawmem.core.answer_generator import AnswerGenerator
from trawmem.core.settings import settings as config


class TrawMemSystem:
    """
    Query-conditioned thread-workspace memory system.

    Pipeline: interaction-thread construction -> one head route -> local graph
    expansion -> connected evidence activation -> answer synthesis.
    """
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        db_path: Optional[str] = None,
        table_name: Optional[str] = None,
        clear_db: bool = False,
        enable_thinking: Optional[bool] = None,
        use_streaming: Optional[bool] = None,
        enable_planning: Optional[bool] = None,
        enable_parallel_processing: Optional[bool] = None,
        max_parallel_workers: Optional[int] = None,
        enable_parallel_retrieval: Optional[bool] = None,
        max_retrieval_workers: Optional[int] = None
    ):
        """
        Initialize system

        Args:
        - api_key: OpenAI API key
        - model: LLM model name
        - base_url: Custom OpenAI base URL (for compatible APIs)
        - db_path: Database path
        - table_name: Memory table name (for parallel processing)
        - clear_db: Whether to clear existing database
        - enable_thinking: Enable deep thinking mode (for Qwen and compatible models)
        - use_streaming: Enable streaming responses
        - enable_planning: Enable one-address planning (None=use config default)
        - enable_parallel_processing: Enable parallel processing for memory building (None=use config default)
        - max_parallel_workers: Maximum number of parallel workers for memory building (None=use config default)
        - enable_parallel_retrieval: Enable parallel processing for retrieval queries (None=use config default)
        - max_retrieval_workers: Maximum number of parallel workers for retrieval (None=use config default)
        """
        print("=" * 60)
        print("Initializing ThreadWorkspace System")
        print("=" * 60)

        # Initialize core components
        self.llm_client = LLMClient(
            api_key=api_key,
            model=model,
            base_url=base_url,
            enable_thinking=enable_thinking,
            use_streaming=use_streaming
        )
        self.embedding_model = EmbeddingModel()
        self.vector_store = ThreadStore(
            db_path=db_path,
            embedding_model=self.embedding_model,
            table_name=table_name
        )

        if clear_db:
            print("\nClearing existing database...")
            self.vector_store.clear()

        # Initialize three major modules
        self.memory_builder = MemoryBuilder(
            llm_client=self.llm_client,
            vector_store=self.vector_store,
            enable_parallel_processing=enable_parallel_processing,
            max_parallel_workers=max_parallel_workers
        )

        self.hybrid_retriever = ThreadWorkspaceRetriever(
            llm_client=self.llm_client,
            vector_store=self.vector_store,
            enable_planning=enable_planning,
            enable_parallel_retrieval=enable_parallel_retrieval,
            max_retrieval_workers=max_retrieval_workers
        )

        self.answer_generator = AnswerGenerator(
            llm_client=self.llm_client
        )

        print("\nSystem initialization complete!")
        print("=" * 60)

    def add_dialogue(self, speaker: str, content: str, timestamp: Optional[str] = None):
        """
        Add a single dialogue

        Args:
        - speaker: Speaker name
        - content: Dialogue content
        - timestamp: Timestamp (ISO 8601 format)
        """
        dialogue_id = self.memory_builder.processed_count + len(self.memory_builder.dialogue_buffer) + 1
        dialogue = Dialogue(
            dialogue_id=dialogue_id,
            speaker=speaker,
            content=content,
            timestamp=timestamp
        )
        self.memory_builder.add_dialogue(dialogue)

    def add_dialogues(self, dialogues: List[Dialogue]):
        """
        Batch add dialogues

        Args:
        - dialogues: List of dialogues
        """
        self.memory_builder.add_dialogues(dialogues)

    def finalize(self):
        """
        Finalize dialogue input, process any remaining buffer (safety check)
        Note: In parallel mode, remaining dialogues are already processed
        """
        self.memory_builder.process_remaining()

    def ask(self, question: str) -> str:
        """
        Ask question - Core Q&A interface

        Args:
        - question: User question

        Returns:
        - Answer
        """
        print("\n" + "=" * 60)
        print(f"Question: {question}")
        print("=" * 60)

        # Stage 3: query-conditioned retrieval workspace
        if getattr(config, "ENABLE_MEMORY_WORKSPACE", True):
            contexts = self.hybrid_retriever.retrieve_workspace(question)
        else:
            contexts = self.hybrid_retriever.retrieve(question)

        # Generate answer from retrieved context C_q
        answer = self.answer_generator.generate_answer(question, contexts)

        print("\nAnswer:")
        print(answer)
        print("=" * 60 + "\n")

        return answer

    def get_all_memories(self) -> List[MemoryEntry]:
        """
        Get all memory entries (for debugging)
        """
        return self.vector_store.get_all_entries()

    def print_memories(self):
        """
        Print all memory entries (for debugging)
        """
        memories = self.get_all_memories()
        print("\n" + "=" * 60)
        print(f"All Memory Entries ({len(memories)} total)")
        print("=" * 60)

        for i, memory in enumerate(memories, 1):
            print(f"\n[Entry {i}]")
            print(f"ID: {memory.entry_id}")
            print(f"Restatement: {memory.lossless_restatement}")
            if memory.timestamp:
                print(f"Time: {memory.timestamp}")
            if memory.thread_id:
                print(f"Thread: {memory.thread_id}")
            if memory.source_turn_ids:
                print(f"Source turns: {', '.join(memory.source_turn_ids)}")

        print("\n" + "=" * 60)


# Convenience function
def create_system(
    clear_db: bool = False,
    enable_planning: Optional[bool] = None,
    enable_parallel_processing: Optional[bool] = None,
    max_parallel_workers: Optional[int] = None,
    enable_parallel_retrieval: Optional[bool] = None,
    max_retrieval_workers: Optional[int] = None
) -> TrawMemSystem:
    """
    Create a TrawMem system instance.
    """
    return TrawMemSystem(
        clear_db=clear_db,
        enable_planning=enable_planning,
        enable_parallel_processing=enable_parallel_processing,
        max_parallel_workers=max_parallel_workers,
        enable_parallel_retrieval=enable_parallel_retrieval,
        max_retrieval_workers=max_retrieval_workers
    )


if __name__ == "__main__":
    print("Running TrawMem quick test...")

    system = create_system(clear_db=True)
    print(f"Using embedding model: {system.memory_builder.vector_store.embedding_model.model_name}")
    print(f"Model type: {system.memory_builder.vector_store.embedding_model.model_type}")

    # Add some test dialogues
    system.add_dialogue("Alice", "Bob, let's meet at Starbucks tomorrow at 2pm to discuss the new product", "2025-11-15T14:30:00")
    system.add_dialogue("Bob", "Okay, I'll prepare the materials", "2025-11-15T14:31:00")
    system.add_dialogue("Alice", "Remember to bring the market research report from last time", "2025-11-15T14:32:00")

    # Finalize input
    system.finalize()

    # View memories
    system.print_memories()

    print("\nTesting address routing and local activation...")
    system.ask("When will Alice and Bob meet?")
    
    print("\nTesting an unsupported question...")
    question = "What is Alice's favorite food?"
    if getattr(config, "ENABLE_MEMORY_WORKSPACE", True):
        contexts = system.hybrid_retriever.retrieve_workspace(question)
    else:
        contexts = system.hybrid_retriever.retrieve(question)
    answer = system.answer_generator.generate_answer(question, contexts)
    print(f"\nQuestion: {question}")
    print(f"Answer: {answer}")
    
    print("\nQuick test completed.")
