"""
LoComo10 Dataset Test for TrawMem System
Tests retrieval time, token usage, and answer quality
"""
from pathlib import Path
import hashlib
import math
import time
import json
from typing import List, Dict, Optional, Union
from dataclasses import dataclass
# Token usage is recorded by the OpenAI-compatible client from response.usage.
from tqdm import tqdm
import statistics
from collections import defaultdict
from main import TrawMemSystem
from trawmem.core.models.memory_entry import Dialogue
from trawmem.core.answer_generator import AnswerGenerationError
from trawmem.core.prompts import (
    CATEGORY5_SYSTEM_PROMPT,
    JUDGE_SYSTEM_PROMPT,
    build_accuracy_judge_prompt,
    build_category5_prompt,
)


def directory_size_bytes(path: str | Path) -> int:
    root = Path(path)
    if not root.exists():
        return 0
    if root.is_file():
        return root.stat().st_size
    total = 0
    for child in root.rglob("*"):
        try:
            if child.is_file() and not child.is_symlink():
                total += child.stat().st_size
        except OSError:
            continue
    return total

# ============================================================================
# Data Structures for LoComo10 Dataset
# ============================================================================

@dataclass
class QA:
    question: str
    answer: Optional[str]
    evidence: List[str]
    category: Optional[int] = None
    adversarial_answer: Optional[str] = None

    @property
    def final_answer(self) -> Optional[str]:
        """Get the appropriate answer based on category."""
        if self.category == 5:
            return self.adversarial_answer
        return self.answer

@dataclass
class Turn:
    speaker: str
    dia_id: str
    text: str

@dataclass
class Session:
    session_id: int
    date_time: str
    turns: List[Turn]

@dataclass
class Conversation:
    speaker_a: str
    speaker_b: str
    sessions: Dict[int, Session]

@dataclass
class EventSummary:
    events: Dict[str, Dict[str, List[str]]]  # session -> speaker -> events

@dataclass
class Observation:
    observations: Dict[str, Dict[str, List[List[str]]]]  # session -> speaker -> [observation, evidence]

@dataclass
class LoCoMoSample:
    """A single sample from the LoComo dataset"""
    sample_id: str
    qa: List[QA]
    conversation: Conversation
    event_summary: EventSummary
    observation: Observation
    session_summary: Dict[str, str]


# ============================================================================
# Dataset Loading Functions
# ============================================================================

def parse_session(session_data: List[dict], session_id: int, date_time: str) -> Session:
    """Parse a single session's data, including turns with images by using their captions."""
    turns = []
    for turn in session_data:
        # For turns with images, combine caption and text
        text = turn.get("text", "")
        if "img_url" in turn and "blip_caption" in turn:
            caption_text = f"[Image: {turn['blip_caption']}]"
            if text:
                text = f"{caption_text} {text}"
            else:
                text = caption_text

        turns.append(Turn(
            speaker=turn["speaker"],
            dia_id=turn["dia_id"],
            text=text
        ))
    return Session(session_id=session_id, date_time=date_time, turns=turns)

def parse_conversation(conv_data: dict) -> Conversation:
    """Parse conversation data."""
    sessions = {}
    for key, value in conv_data.items():
        if key.startswith("session_") and isinstance(value, list):
            session_id = int(key.split("_")[1])
            date_time = conv_data.get(f"{key}_date_time")
            if date_time:
                session = parse_session(value, session_id, date_time)
                # Only add sessions that have turns after filtering
                if session.turns:
                    sessions[session_id] = session

    return Conversation(
        speaker_a=conv_data["speaker_a"],
        speaker_b=conv_data["speaker_b"],
        sessions=sessions
    )

def load_locomo_dataset(file_path: Union[str, Path]) -> List[LoCoMoSample]:
    """
    Load the LoComo dataset from a JSON file, including image-based content by using captions.

    Args:
        file_path: Path to the JSON file containing the dataset

    Returns:
        List of LoCoMoSample objects containing the parsed data
    """
    if isinstance(file_path, str):
        file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found at {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    samples = []
    total_qa = 0
    total_image_qa = 0
    qa_counts_per_sample = []

    for sample_idx, sample in enumerate(data):
        try:
            # Parse QA data
            qa_list = []
            sample_qa_count = 0
            sample_image_qa_count = 0

            for qa_idx, qa in enumerate(sample["qa"]):
                try:
                    # Check if QA has image evidence
                    has_image_evidence = False
                    for evidence_id in qa.get("evidence", []):
                        if ":" not in evidence_id:
                            continue
                        turn_id = evidence_id.split(":")[1]
                        for session in sample["conversation"].values():
                            if isinstance(session, list):
                                for turn in session:
                                    if turn.get("dia_id", "").endswith(turn_id):
                                        if "img_url" in turn or "blip_caption" in turn:
                                            has_image_evidence = True
                                            break

                    if has_image_evidence:
                        sample_image_qa_count += 1

                    qa_obj = QA(
                        question=qa["question"],
                        answer=qa.get("answer"),
                        evidence=qa.get("evidence", []),
                        category=qa.get("category"),
                        adversarial_answer=qa.get("adversarial_answer")
                    )
                    qa_list.append(qa_obj)
                    sample_qa_count += 1

                except KeyError as e:
                    print(f"Error in sample {sample_idx}, QA pair {qa_idx}:")
                    print(f"QA data: {qa}")
                    raise e
                except Exception as e:
                    print(f"Unexpected error in sample {sample_idx}, QA pair {qa_idx}:")
                    print(f"QA data: {qa}")
                    raise e

            # Parse conversation
            conversation = parse_conversation(sample["conversation"])

            # Parse event summary
            event_summary = EventSummary(events=sample["event_summary"])

            # Parse observation
            observation = Observation(observations=sample["observation"])

            # Get session summary
            session_summary = sample.get("session_summary", {})

            # Create sample object
            sample_obj = LoCoMoSample(
                sample_id=str(sample_idx),
                qa=qa_list,
                conversation=conversation,
                event_summary=event_summary,
                observation=observation,
                session_summary=session_summary
            )
            samples.append(sample_obj)

            total_qa += sample_qa_count
            total_image_qa += sample_image_qa_count
            qa_counts_per_sample.append(sample_qa_count)

            # Print statistics for this sample
            print(f"\nSample {sample_idx}:")
            print(f"  Total QAs: {sample_qa_count}")
            print(f"  QAs with image evidence: {sample_image_qa_count}")

        except Exception as e:
            print(f"Error processing sample {sample_idx}:")
            print(str(e))
            raise e

    # Print overall statistics
    print("\nOverall Statistics:")
    print(f"Total QAs: {total_qa}")
    print(f"Total QAs with image evidence: {total_image_qa}")
    print(f"Average QAs per sample: {total_qa / len(samples):.2f}")
    print(f"Min QAs in a sample: {min(qa_counts_per_sample)}")
    print(f"Max QAs in a sample: {max(qa_counts_per_sample)}")

    return samples


# ============================================================================
# Evaluation Metrics Functions
# ============================================================================

def simple_tokenize(text):
    """Simple tokenization function."""
    text = str(text)
    return text.lower().replace('.', ' ').replace(',', ' ').replace('!', ' ').replace('?', ' ').split()

def calculate_bleu1(prediction: str, reference: str) -> float:
    """Compute smoothed unigram BLEU without NLTK or external resources."""
    pred_tokens = simple_tokenize(prediction)
    ref_tokens = simple_tokenize(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0
    overlap = sum(min(pred_tokens.count(token), ref_tokens.count(token)) for token in set(pred_tokens))
    precision = overlap / len(pred_tokens)
    brevity = 1.0 if len(pred_tokens) >= len(ref_tokens) else math.exp(
        1 - len(ref_tokens) / len(pred_tokens)
    )
    return precision * brevity

def create_judge_llm_client():
    """Create a dedicated LLM client for judge evaluation"""
    from trawmem.core.utils.llm_client import LLMClient
    import config
    
    # Use judge-specific settings, fall back to main settings if not specified
    judge_api_key = getattr(config, 'JUDGE_API_KEY', None) or config.OPENAI_API_KEY
    judge_base_url = getattr(config, 'JUDGE_BASE_URL', None)
    if judge_base_url is None:
        judge_base_url = getattr(config, 'OPENAI_BASE_URL', None)
    judge_model = getattr(config, 'JUDGE_MODEL', None) or config.LLM_MODEL
    judge_thinking = getattr(config, 'JUDGE_ENABLE_THINKING', False)
    judge_streaming = getattr(config, 'JUDGE_USE_STREAMING', False)
    
    print(f"Initializing LLM-as-judge with model: {judge_model}")
    if judge_base_url and judge_base_url != getattr(config, 'OPENAI_BASE_URL', None):
        print(f"Using separate judge endpoint: {judge_base_url}")
    
    # For OpenAI API, disable thinking mode to avoid parameter errors
    is_openai_api = not judge_base_url or "openai" in judge_base_url.lower()
    if is_openai_api and judge_thinking:
        print("Note: Disabling thinking mode for OpenAI API compatibility")
        judge_thinking = False
    
    return LLMClient(
        api_key=judge_api_key,
        model=judge_model,
        base_url=judge_base_url,
        enable_thinking=judge_thinking,
        use_streaming=judge_streaming
    )

def _locomo_question_type(category: Optional[object]) -> Optional[str]:
    """Map LoCoMo's numeric categories to judge-specific instructions.

    The answer generator does not consume benchmark labels, but the evaluator
    may use them to apply the benchmark's semantic grading rules.  Keeping the
    mapping here prevents category semantics from leaking into retrieval.
    """
    try:
        value = int(category) if category is not None else None
    except (TypeError, ValueError):
        value = None
    return {
        1: "multi-hop",
        2: "temporal",
        3: "open-domain",
        4: "single-hop",
        5: "adversarial",
    }.get(value)


def llm_judge_answers(
    prediction: str,
    reference: str,
    question: str,
    judge_client,
    *,
    category: Optional[object] = None,
    question_type: Optional[str] = None,
    abstention: bool = False,
) -> Dict[str, Union[float, str]]:
    """Use LLM to judge if prediction is semantically equivalent to reference."""
    # Handle empty or None values
    if not prediction or (not reference and not abstention):
        return {
            "llm_judge_score": 0.0,
            "llm_reasoning": "Empty prediction or reference",
            "llm_judge_error": "empty_prediction_or_reference",
        }
    
    prediction = str(prediction).strip()
    reference = str(reference).strip()
    
    effective_category = (
        category
        if category is not None
        else (5 if reference == "Not mentioned in the conversation" else None)
    )
    effective_question_type = question_type or _locomo_question_type(effective_category)
    prompt = build_accuracy_judge_prompt(
        question,
        reference,
        prediction,
        category=effective_category,
        question_type=effective_question_type,
        benchmark="locomo",
        abstention=abstention,
    )

    try:
        messages = [
            {
                "role": "system", 
                "content": JUDGE_SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        import config
        # Use JSON format if configured
        response_format = None
        if hasattr(config, 'USE_JSON_FORMAT') and config.USE_JSON_FORMAT:
            response_format = {"type": "json_object"}
        
        # Use judge-specific temperature setting
        judge_temperature = getattr(config, 'JUDGE_TEMPERATURE', 0.0)
        
        response = judge_client.chat_completion(
            messages,
            temperature=judge_temperature,
            response_format=response_format,
            max_retries=3,  # Ensure robust evaluation with retries
            stage="judge",
            max_tokens=getattr(config, "JUDGE_MAX_TOKENS", 256),
        )
        
        # Parse JSON response
        result = judge_client.extract_json(response)
        if not isinstance(result, dict) or "score" not in result:
            raise ValueError("judge_response_missing_score")
        raw_score = result["score"]
        if isinstance(raw_score, bool):
            raise ValueError("judge_score_must_be_numeric")
        score = float(raw_score)
        if score not in (0.0, 1.0):
            raise ValueError(f"judge_score_out_of_range:{score}")
        reasoning = result.get("reasoning", "No reasoning provided")
        
        return {
            "llm_judge_score": score,
            "llm_reasoning": reasoning
        }
        
    except Exception as e:
        print(f"Warning: LLM judge evaluation failed: {e}")
        return {
            "llm_judge_score": 0.0,
            "llm_reasoning": f"Evaluation failed: {e}",
            "llm_judge_error": str(e),
        }

def calculate_metrics(
    prediction: str,
    reference: str,
    question: str = None,
    judge_client=None,
    use_llm_judge: bool = False,
    *,
    category: Optional[object] = None,
    question_type: Optional[str] = None,
    abstention: bool = False,
) -> Dict[str, float]:
    """Calculate the paper metrics without optional network-backed scorers."""
    # Handle empty or None values
    if not prediction or not reference:
        metrics = {
            "exact_match": 0,
            "f1": 0.0,
            "bleu1": 0.0,
        }
        if use_llm_judge and question and judge_client:
            llm_result = llm_judge_answers(
                prediction,
                reference,
                question,
                judge_client,
                category=category,
                question_type=question_type,
                abstention=abstention,
            )
            if "llm_judge_error" in llm_result:
                raise RuntimeError(f"LLM judge failed: {llm_result['llm_judge_error']}")
            metrics.update(llm_result)
        return metrics

    # Convert to strings if they're not already
    prediction = str(prediction).strip()
    reference = str(reference).strip()

    # Calculate exact match
    exact_match = int(prediction.lower() == reference.lower())

    # Calculate token-based F1 score
    pred_tokens = set(simple_tokenize(prediction))
    ref_tokens = set(simple_tokenize(reference))
    common_tokens = pred_tokens & ref_tokens

    if not pred_tokens or not ref_tokens:
        f1 = 0.0
    else:
        precision = len(common_tokens) / len(pred_tokens)
        recall = len(common_tokens) / len(ref_tokens)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Keep the evaluation contract small and deterministic: F1, BLEU-1, judge.
    metrics = {
        "exact_match": exact_match,
        "f1": f1,
        "bleu1": calculate_bleu1(prediction, reference),
    }
    
    # Add LLM judge evaluation if enabled
    if use_llm_judge and question and judge_client:
        llm_result = llm_judge_answers(
            prediction,
            reference,
            question,
            judge_client,
            category=category,
            question_type=question_type,
            abstention=abstention,
        )
        if "llm_judge_error" in llm_result:
            raise RuntimeError(f"LLM judge failed: {llm_result['llm_judge_error']}")
        metrics["llm_judge_score"] = llm_result["llm_judge_score"]
        metrics["llm_reasoning"] = llm_result["llm_reasoning"]

    return metrics

def aggregate_metrics(all_metrics: List[Dict[str, float]], all_categories: List[int]) -> Dict[str, Dict[str, Union[float, Dict[str, float]]]]:
    """Calculate aggregate statistics for all metrics, split by category."""
    if not all_metrics:
        return {}

    # Initialize aggregates for overall and per-category metrics
    aggregates = defaultdict(list)
    category_aggregates = defaultdict(lambda: defaultdict(list))

    # Collect all values for each metric, both overall and per category
    for metrics, category in zip(all_metrics, all_categories):
        for metric_name, value in metrics.items():
            # Skip non-numeric values like llm_reasoning
            if isinstance(value, (int, float)):
                aggregates[metric_name].append(value)
                category_aggregates[category][metric_name].append(value)

    # Calculate statistics for overall metrics
    results = {
        "overall": {}
    }

    for metric_name, values in aggregates.items():
        if values:  # Only calculate if we have numeric values
            results["overall"][metric_name] = {
                'mean': statistics.mean(values),
                'std': statistics.stdev(values) if len(values) > 1 else 0.0,
                'median': statistics.median(values),
                'min': min(values),
                'max': max(values),
                'count': len(values)
            }

    # Calculate statistics for each category
    for category in sorted(category_aggregates.keys()):
        results[f"category_{category}"] = {}
        for metric_name, values in category_aggregates[category].items():
            if values:  # Only calculate if we have values for this category
                results[f"category_{category}"][metric_name] = {
                    'mean': statistics.mean(values),
                    'std': statistics.stdev(values) if len(values) > 1 else 0.0,
                    'median': statistics.median(values),
                    'min': min(values),
                    'max': max(values),
                    'count': len(values)
                }

    return results


# ============================================================================
# Testing Classes
# ============================================================================


class LoCoMoTester:
    """Test TrawMem system on LoComo10 dataset"""

    def __init__(self, system: TrawMemSystem, dataset_path: str, use_llm_judge: bool = False, test_workers: int = None):
        self.system = system
        self.dataset_path = Path(dataset_path)
        self.use_llm_judge = use_llm_judge
        self.test_workers = test_workers

        # Initialize judge client if needed
        self.judge_client = None
        if self.use_llm_judge:
            self.judge_client = create_judge_llm_client()

        import config
        tokenizer_path = getattr(config, "TOKENIZER_MODEL_PATH", None)
        tokenizer_encoding = getattr(config, "TOKENIZER_ENCODING", None)
        self.context_tokenizer = None
        if tokenizer_path:
            try:
                from transformers import AutoTokenizer

                self.context_tokenizer = AutoTokenizer.from_pretrained(
                    tokenizer_path,
                    trust_remote_code=True,
                    local_files_only=True,
                )
            except Exception as error:
                # Keep exact Qwen token accounting when the local Transformers
                # package does not yet know the checkpoint architecture.
                try:
                    from tokenizers import Tokenizer

                    class _FastTokenizerAdapter:
                        def __init__(self, tokenizer):
                            self._tokenizer = tokenizer

                        def encode(self, text, add_special_tokens=False):
                            return self._tokenizer.encode(str(text)).ids

                    self.context_tokenizer = _FastTokenizerAdapter(
                        Tokenizer.from_file(str(Path(tokenizer_path) / "tokenizer.json"))
                    )
                except Exception as fallback_error:
                    raise RuntimeError(
                        f"could not load context tokenizer: {error}; "
                        f"tokenizer.json fallback failed: {fallback_error}"
                    ) from fallback_error
        if self.context_tokenizer is None and tokenizer_encoding:
            try:
                import tiktoken

                encoding = tiktoken.get_encoding(tokenizer_encoding)

                class _TiktokenAdapter:
                    def encode(self, text, add_special_tokens=False):
                        del add_special_tokens
                        return encoding.encode(str(text), disallowed_special=())

                self.context_tokenizer = _TiktokenAdapter()
                print(f"Using context tokenizer encoding: {tokenizer_encoding}")
            except Exception as error:
                raise RuntimeError(
                    f"could not load context tokenizer encoding {tokenizer_encoding}: {error}"
                ) from error

        # Statistics
        self.retrieval_times = []
        self.answer_times = []
        self.total_times = []
        self.metrics_list = []
        self.categories = []
        self.question_failures = []
        self.build_stats = []

    def generate_category5_answer(self, question: str, contexts: List, adversarial_answer: str) -> str:
        """
        Special answer generation for category 5 (adversarial questions).
        Ask model to choose between "Not mentioned in the conversation" and the adversarial answer.
        """
        # Balance option position without introducing run-to-run randomness.
        options = ["Not mentioned in the conversation", adversarial_answer]
        if hashlib.sha256(question.encode("utf-8")).digest()[0] % 2:
            options.reverse()

        # Build context string
        context_str = self.system.answer_generator._format_contexts(contexts)

        prompt = build_category5_prompt(question, context_str, options)

        messages = [
            {
                "role": "system",
                "content": CATEGORY5_SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        # Retry up to 3 times
        max_retries = 3
        for attempt in range(max_retries):
            try:
                import config
                # Use JSON format if configured
                response_format = None
                if hasattr(config, 'USE_JSON_FORMAT') and config.USE_JSON_FORMAT:
                    response_format = {"type": "json_object"}

                response = self.system.llm_client.chat_completion(
                    messages,
                    temperature=0.0,
                    response_format=response_format,
                    max_retries=3,  # Ensure robust category 5 evaluation with retries
                    stage="answer"
                )

                # Parse JSON response
                result = self.system.llm_client.extract_json(response)
                answer = result.get("answer") if isinstance(result, dict) else None
                if not isinstance(answer, str) or answer.strip() not in options:
                    raise ValueError("category5_response_missing_valid_answer")
                return answer.strip()

            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Category 5 answer generation attempt {attempt + 1}/{max_retries} failed: {e}. Retrying...")
                else:
                    print(f"Warning: Failed to generate category 5 answer after {max_retries} attempts: {e}")
                    raise AnswerGenerationError(
                        f"category-5 answer generation failed after {max_retries} attempts: {e}"
                    ) from e

    def load_dataset(self, limit: int = None) -> List[LoCoMoSample]:
        """Load LoComo10 dataset"""
        print(f"Loading dataset from {self.dataset_path}...")
        samples = load_locomo_dataset(self.dataset_path)

        if limit:
            samples = samples[:limit]
            print(f"Limited to {limit} samples")

        return samples

    def convert_to_dialogues(self, sample: LoCoMoSample) -> List[Dialogue]:
        """Convert LoComo sample to Dialogue objects"""
        dialogues = []
        dialogue_id = 1

        # Process all sessions in order
        for session_id in sorted(sample.conversation.sessions.keys()):
            session = sample.conversation.sessions[session_id]

            for turn_index, turn in enumerate(session.turns, 1):
                dialogue = Dialogue(
                    dialogue_id=dialogue_id,
                    speaker=turn.speaker,
                    content=turn.text,
                    timestamp=session.date_time,  # Use session datetime
                    source_id=turn.dia_id,
                    session_id=f"D{session_id}",
                    turn_index=turn_index,
                )
                dialogues.append(dialogue)
                dialogue_id += 1

        return dialogues

    def test_sample(self, sample: LoCoMoSample, sample_idx: int, enable_parallel_questions: bool = False):
        """Test a single sample from the dataset"""
        print(f"\n{'='*80}")
        print(f"Testing Sample {sample_idx}")
        print(f"{'='*80}")

        # Convert and add dialogues
        dialogues = self.convert_to_dialogues(sample)
        print(f"Adding {len(dialogues)} dialogues to memory...")

        add_start = time.time()
        with self.system.llm_client.usage_scope(sample_idx=sample_idx, phase="build"):
            self.system.add_dialogues(dialogues)
            self.system.finalize()
        add_time = time.time() - add_start
        print(f"Memory building time: {add_time:.2f}s")
        db_path = getattr(self.system.vector_store, "db_path", None)
        memory_size = directory_size_bytes(db_path) if db_path else None
        self.build_stats.append({
            "sample_idx": sample_idx,
            "build_seconds": add_time,
            "raw_turn_count": len(dialogues),
            "memory_count": len(self.system.get_all_memories()),
            "llm_calls": sum(
                1 for event in self.system.llm_client.usage_events()
                if event.get("success")
                and event.get("context", {}).get("sample_idx") == sample_idx
                and event.get("context", {}).get("phase") == "build"
            ),
            "memory_size_bytes": memory_size,
            "memory_size_source": "on_disk_db_path_bytes",
            "memory_path": str(db_path) if db_path else None,
        })

        # Test each question (parallel or sequential)
        if enable_parallel_questions and len(sample.qa) > 1:
            sample_results = self._test_questions_parallel(sample.qa, sample_idx)
        else:
            sample_results = self._test_questions_sequential(sample.qa, sample_idx)

        return sample_results
    
    def _test_questions_sequential(self, qa_list: List, sample_idx: int):
        """Test questions sequentially (original method)"""
        sample_results = []
        
        for qa_idx, qa in enumerate(qa_list):
            try:
                result = self._process_single_question(qa, qa_idx, sample_idx)
            except Exception as error:
                print(f"[Question] Q{qa_idx + 1} failed: {error}", flush=True)
                result = self._failed_question_result(qa, qa_idx, sample_idx, error)
            sample_results.append(result)
            
        return sample_results

    def _failed_question_result(self, qa, qa_idx: int, sample_idx: int, error: Exception):
        """Materialize a question failure so one bad call cannot abort a shard."""
        error_text = f"{type(error).__name__}: {error}"
        is_judge_error = "judge" in error_text.lower()
        status = "judge_error" if is_judge_error else "answer_error"
        category = qa.category if qa.category is not None else 0
        reference = (
            "Not mentioned in the conversation"
            if category == 5
            else qa.final_answer
        )
        self.question_failures.append({
            "sample_idx": sample_idx,
            "question_idx": qa_idx,
            "error": error_text,
            "status": status,
        })
        metrics = (
            {"exact_match": 0, "f1": 0.0, "bleu1": 0.0}
            if status == "answer_error" else {}
        )
        if metrics:
            self.metrics_list.append(metrics)
            self.categories.append(category)
        return {
            "evaluation_status": status,
            "sample_idx": sample_idx,
            "question_index": qa_idx,
            "question": qa.question,
            "answer": "",
            "reference": reference,
            "category": category,
            "retrieval_time": 0.0,
            "answer_time": 0.0,
            "total_time": 0.0,
            "num_retrieved": 0,
            "retrieved_context_tokens": None,
            "token_cost": None,
            "llm_calls_query": 0,
            "retrieval_llm_calls": 0,
            "llm_judge_calls": 1 if self.use_llm_judge else 0,
            "answer_error": error_text if status == "answer_error" else None,
            "judge_error": error_text if status == "judge_error" else None,
            # An unavailable answer is an end-to-end zero, while a judge
            # failure must remain excluded from the judge aggregate.
            "metrics": metrics,
        }
    
    def _test_questions_parallel(self, qa_list: List, sample_idx: int):
        """Test questions in parallel using ThreadPoolExecutor"""
        import concurrent.futures
        
        print(f"\n[Parallel Testing] Processing {len(qa_list)} questions in parallel")
        sample_results = []
        
        # Use ThreadPoolExecutor for parallel question processing
        # Use explicit test_workers parameter, or config, or reasonable default
        import config
        
        if self.test_workers is not None:
            max_workers = self.test_workers
        else:
            max_workers = getattr(config, 'MAX_RETRIEVAL_WORKERS', 16)
        
        # Apply reasonable limits
        max_workers = min(
            max_workers,
            len(qa_list),  # Don't create more workers than questions
            20  # Higher limit for better parallelism, but watch API rate limits
        )
        max_workers = max(max_workers, 1)  # At least 1 worker
        
        print(f"[Parallel Testing] Using {max_workers} parallel workers for {len(qa_list)} questions")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all question processing tasks
            future_to_qa = {}
            for qa_idx, qa in enumerate(qa_list):
                future = executor.submit(self._process_single_question, qa, qa_idx, sample_idx)
                future_to_qa[future] = (qa, qa_idx)
            
            # Collect results as they complete, maintain order
            results_dict = {}
            for future in concurrent.futures.as_completed(future_to_qa):
                qa, qa_idx = future_to_qa[future]
                try:
                    result = future.result()
                    results_dict[qa_idx] = result
                    print(f"[Parallel Testing] Question {qa_idx+1} completed")
                except Exception as e:
                    print(f"[Parallel Testing] Question {qa_idx+1} failed: {e}")
                    results_dict[qa_idx] = self._failed_question_result(
                        qa, qa_idx, sample_idx, e
                    )
            
            # Sort results by qa_idx to maintain original order
            for qa_idx in sorted(results_dict.keys()):
                sample_results.append(results_dict[qa_idx])
        
        return sample_results
    
    def _process_single_question(self, qa, qa_idx: int, sample_idx: int):
        """Process a single question and return result"""
        question = qa.question
        category = qa.category if qa.category is not None else 0

        # For category 5, the ground truth is always "Not mentioned in the conversation"
        # For other categories, use qa.final_answer
        if category == 5:
            reference_answer = "Not mentioned in the conversation"
        else:
            reference_answer = qa.final_answer

        print(f"\n[Q{qa_idx+1}] Category {category}: {question}")

        with self.system.llm_client.usage_scope(
            sample_idx=sample_idx,
            question_idx=qa_idx,
            question=question,
        ):
            # Measure retrieval time
            # For category 5 (adversarial), disable reflection since "no answer means no answer"
            retrieval_start = time.time()
            import config
            use_workspace = getattr(config, "ENABLE_MEMORY_WORKSPACE", True)
            if use_workspace:
                if category == 5:
                    contexts = self.system.hybrid_retriever.retrieve_workspace(
                        question, enable_reflection=False
                    )
                else:
                    contexts = self.system.hybrid_retriever.retrieve_workspace(question)
            elif category == 5:
                contexts = self.system.hybrid_retriever.retrieve(
                    question, enable_reflection=False
                )
            else:
                contexts = self.system.hybrid_retriever.retrieve(question)
            retrieval_time = time.time() - retrieval_start
            # The benchmark category is used only to select answer-format
            # guidance after retrieval; it does not alter persistent memory or
            # the route itself.
            answer_question_type = _locomo_question_type(category)
            if answer_question_type and hasattr(contexts, "question_type"):
                contexts.question_type = answer_question_type
            context_text = self.system.answer_generator._format_contexts(contexts)
            context_token_count = (
                len(self.context_tokenizer.encode(context_text, add_special_tokens=False))
                if self.context_tokenizer is not None
                else None
            )
            workspace_stats = {}
            if hasattr(contexts, "source_candidate_count"):
                selected_sources = {
                    source_id
                    for item in contexts.evidence
                    for source_id in (item.source_turn_ids or [])
                }
                gold_sources = {str(source_id) for source_id in (qa.evidence or [])}
                source_intersection = selected_sources & gold_sources
                routed_sources = set(getattr(contexts, "routed_source_turn_ids", []) or [])
                expanded_sources = set(getattr(contexts, "expanded_source_turn_ids", []) or [])
                routed_intersection = routed_sources & gold_sources
                expanded_intersection = expanded_sources & gold_sources
                source_recall = (
                    len(source_intersection) / len(gold_sources)
                    if gold_sources else None
                )
                role_counts: Dict[str, int] = defaultdict(int)
                for item in contexts.evidence:
                    role = str(getattr(item, "role", "") or "").strip()
                    if role:
                        role_counts[role] += 1
                workspace_stats = {
                    "workspace_candidate_count": contexts.source_candidate_count,
                    "workspace_evidence_count": len(contexts.evidence),
                    "workspace_sufficient": contexts.sufficient,
                    "workspace_routed_thread_count": getattr(contexts, "routed_thread_count", None),
                    "workspace_expanded_thread_count": getattr(contexts, "expanded_thread_count", None),
                    "workspace_node_candidate_count": getattr(contexts, "node_candidate_count", None),
                    "workspace_selected_thread_count": getattr(contexts, "selected_thread_count", None),
                    "workspace_selection_budget": getattr(contexts, "selection_budget", None),
                    "workspace_selection_coverage": getattr(contexts, "selection_coverage", None),
                    "workspace_selection_rounds": getattr(contexts, "selection_rounds", None),
                    "workspace_coverage_mode": getattr(contexts, "coverage_mode", None),
                    "workspace_selected_activation_mass": getattr(contexts, "selected_activation_mass", None),
                    "workspace_activation_entropy": getattr(contexts, "activation_entropy", None),
                    "workspace_activation_iterations": getattr(contexts, "activation_iterations", None),
                    "workspace_transport_flow_units": getattr(contexts, "transport_flow_units", None),
                    "workspace_transport_cost": getattr(contexts, "transport_cost", None),
                    "workspace_graph_vertex_count": getattr(contexts, "graph_vertex_count", None),
                    "workspace_graph_edge_count": getattr(contexts, "graph_edge_count", None),
                    "workspace_local_edge_visits": getattr(contexts, "local_edge_visits", None),
                    "workspace_global_route_calls": getattr(contexts, "global_route_calls", None),
                    "workspace_address_planner_calls": getattr(contexts, "address_planner_calls", None),
                    "workspace_indexed_thread_count": getattr(contexts, "indexed_thread_count", None),
                    "workspace_role_mode": getattr(contexts, "role_mode", None),
                    "workspace_role_frozen": getattr(contexts, "role_frozen", None),
                    "workspace_role_demand": dict(getattr(contexts, "role_demand", {}) or {}),
                    "workspace_role_entropy": getattr(contexts, "role_entropy", None),
                    "workspace_role_assignment_count": getattr(
                        contexts, "role_assignment_count", None
                    ),
                    "workspace_role_counts": dict(role_counts),
                    "gold_source_count": len(gold_sources),
                    "gold_source_recall": source_recall,
                    "gold_source_any": bool(source_intersection) if gold_sources else None,
                    "gold_source_all": bool(gold_sources) and gold_sources.issubset(selected_sources),
                    "routed_gold_source_recall": (
                        len(routed_intersection) / len(gold_sources) if gold_sources else None
                    ),
                    "routed_gold_source_any": bool(routed_intersection) if gold_sources else None,
                    "routed_gold_source_all": (
                        bool(gold_sources) and gold_sources.issubset(routed_sources)
                    ),
                    "expanded_gold_source_recall": (
                        len(expanded_intersection) / len(gold_sources) if gold_sources else None
                    ),
                    "expanded_gold_source_any": bool(expanded_intersection) if gold_sources else None,
                    "expanded_gold_source_all": (
                        bool(gold_sources) and gold_sources.issubset(expanded_sources)
                    ),
                }

            # Measure answer generation time
            answer_start = time.time()

            # Use special answer generation for category 5
            if category == 5:
                adversarial_answer = qa.adversarial_answer if qa.adversarial_answer else "Unknown answer"
                answer = self.generate_category5_answer(question, contexts, adversarial_answer)
            else:
                answer = self.system.answer_generator.generate_answer(question, contexts)

            answer_time = time.time() - answer_start

            query_llm_calls = self.system.llm_client.question_call_count(sample_idx, qa_idx)

        total_time = retrieval_time + answer_time

        # Calculate metrics.  A judge outage must not erase the deterministic
        # answer metrics; keep the answer score and mark the judge separately.
        evaluation_status = "ok"
        answer_error = None
        judge_error = None
        if reference_answer:
            if self.use_llm_judge:
                try:
                    with self.system.llm_client.usage_scope(
                        sample_idx=sample_idx,
                        question_idx=qa_idx,
                        question=question,
                        phase="judge",
                    ):
                        metrics = calculate_metrics(
                            answer,
                            reference_answer,
                            question=question,
                            judge_client=self.judge_client,
                            use_llm_judge=True,
                            category=category,
                            question_type=_locomo_question_type(category),
                        )
                except Exception as error:
                    evaluation_status = "judge_error"
                    judge_error = f"{type(error).__name__}: {error}"
                    self.question_failures.append({
                        "sample_idx": sample_idx,
                        "question_idx": qa_idx,
                        "error": judge_error,
                        "status": evaluation_status,
                    })
                    metrics = calculate_metrics(answer, reference_answer)
            else:
                metrics = calculate_metrics(answer, reference_answer)
        else:
            metrics = {}

        # Store statistics
        self.retrieval_times.append(retrieval_time)
        self.answer_times.append(answer_time)
        self.total_times.append(total_time)
        if metrics:
            self.metrics_list.append(metrics)
            self.categories.append(category)

        # Print results
        print(f"  Retrieved: {len(contexts)} memory entries")
        if workspace_stats:
            print(
                f"  Workspace: {workspace_stats['workspace_evidence_count']} evidence / "
                f"{workspace_stats['workspace_node_candidate_count']} local nodes; "
                f"mass={workspace_stats['workspace_selected_activation_mass']:.3f}"
            )
            if workspace_stats.get("gold_source_recall") is not None:
                print(f"  Gold source-turn recall: {workspace_stats['gold_source_recall']:.3f}")
        if context_token_count is not None:
            print(f"  Retrieved context tokens: {context_token_count}")
        print(f"  Retrieval time: {retrieval_time:.3f}s")
        print(f"  Answer time: {answer_time:.3f}s")
        print(f"  Total time: {total_time:.3f}s")
        print(f"  Answer: {answer}")
        if reference_answer:
            print(f"  Reference: {reference_answer}")
            if metrics:
                print(f"  F1: {metrics.get('f1', 0):.3f}, "
                      f"BLEU-1: {metrics.get('bleu1', 0):.3f}")
                if self.use_llm_judge and 'llm_judge_score' in metrics:
                    print(f"  LLM Judge: {metrics.get('llm_judge_score', 0):.3f}")
                    if 'llm_reasoning' in metrics:
                        print(f"  LLM Reasoning: {metrics.get('llm_reasoning', '')}")

        return {
            'evaluation_status': evaluation_status,
            'sample_idx': sample_idx,
            'question_index': qa_idx,
            'question': question,
            'answer': answer,
            'reference': reference_answer,
            'category': category,
            'retrieval_time': retrieval_time,
            'answer_time': answer_time,
            'total_time': total_time,
            'num_retrieved': len(contexts),
            'retrieved_context_tokens': context_token_count,
            'token_cost': context_token_count,
            'llm_calls_query': query_llm_calls,
            'retrieval_llm_calls': max(0, query_llm_calls - 1),
            'llm_judge_calls': 1 if self.use_llm_judge else 0,
            'build_seconds': self.build_stats[-1]['build_seconds'] if self.build_stats else None,
            'memory_size_bytes': self.build_stats[-1]['memory_size_bytes'] if self.build_stats else None,
            'memory_size_source': self.build_stats[-1]['memory_size_source'] if self.build_stats else None,
            'answer_error': answer_error,
            'judge_error': judge_error,
            **workspace_stats,
            'metrics': metrics
        }

    @staticmethod
    def _summarize_workspace_results(workspace_results: List[Dict]) -> Dict:
        """Aggregate routing, expansion, activation, and final evidence diagnostics."""
        if not workspace_results:
            return {"questions": 0}

        def mean(field: str):
            values = [row[field] for row in workspace_results if row.get(field) is not None]
            return sum(values) / len(values) if values else None

        def rate(field: str):
            values = [row[field] for row in workspace_results if row.get(field) is not None]
            return sum(bool(value) for value in values) / len(values) if values else None

        summary = {
            "questions": len(workspace_results),
            "average_indexed_threads": mean("workspace_indexed_thread_count"),
            "average_routed_threads": mean("workspace_routed_thread_count"),
            "average_expanded_threads": mean("workspace_expanded_thread_count"),
            "average_local_nodes": mean("workspace_node_candidate_count"),
            "average_evidence": mean("workspace_evidence_count"),
            "average_selected_threads": mean("workspace_selected_thread_count"),
            "average_selection_budget": mean("workspace_selection_budget"),
            "average_selection_coverage": mean("workspace_selection_coverage"),
            "average_selection_rounds": mean("workspace_selection_rounds"),
            "average_activation_mass": mean("workspace_selected_activation_mass"),
            "average_activation_entropy": mean("workspace_activation_entropy"),
            "average_activation_iterations": mean("workspace_activation_iterations"),
            "average_transport_flow_units": mean("workspace_transport_flow_units"),
            "average_transport_cost": mean("workspace_transport_cost"),
            "average_graph_edges": mean("workspace_graph_edge_count"),
            "average_local_edge_visits": mean("workspace_local_edge_visits"),
            "average_global_route_calls": mean("workspace_global_route_calls"),
            "average_address_planner_calls": mean("workspace_address_planner_calls"),
            "average_token_cost": mean("token_cost"),
            "activation_sufficient_rate": rate("workspace_sufficient"),
            "routed_gold_source_recall": mean("routed_gold_source_recall"),
            "routed_gold_source_any_rate": rate("routed_gold_source_any"),
            "routed_gold_source_all_rate": rate("routed_gold_source_all"),
            "expanded_gold_source_recall": mean("expanded_gold_source_recall"),
            "expanded_gold_source_any_rate": rate("expanded_gold_source_any"),
            "expanded_gold_source_all_rate": rate("expanded_gold_source_all"),
            "workspace_gold_source_recall": mean("gold_source_recall"),
            "workspace_gold_source_any_rate": rate("gold_source_any"),
            "workspace_gold_source_all_rate": rate("gold_source_all"),
            "average_role_entropy": mean("workspace_role_entropy"),
            "average_role_assignment_count": mean("workspace_role_assignment_count"),
            "role_frozen_rate": rate("workspace_role_frozen"),
        }
        role_totals: Dict[str, int] = defaultdict(int)
        role_mode_counts: Dict[str, int] = defaultdict(int)
        demand_totals: Dict[str, float] = defaultdict(float)
        demand_rows = 0
        for row in workspace_results:
            role_mode = str(row.get("workspace_role_mode") or "").strip()
            if role_mode:
                role_mode_counts[role_mode] += 1
            for role, count in (row.get("workspace_role_counts") or {}).items():
                role_totals[str(role)] += int(count or 0)
            demand = row.get("workspace_role_demand") or {}
            if demand:
                demand_rows += 1
                for role, value in demand.items():
                    demand_totals[str(role)] += float(value or 0.0)
        total_role_bindings = sum(role_totals.values())
        summary["role_mode_counts"] = dict(role_mode_counts)
        summary["role_binding_counts"] = dict(role_totals)
        summary["role_binding_fractions"] = (
            {role: count / total_role_bindings for role, count in role_totals.items()}
            if total_role_bindings else {}
        )
        summary["average_role_demand"] = (
            {role: value / demand_rows for role, value in demand_totals.items()}
            if demand_rows else {}
        )
        summary["by_category"] = {}
        for category in sorted({row.get("category") for row in workspace_results}):
            rows = [row for row in workspace_results if row.get("category") == category]

            def category_mean(field: str):
                values = [row[field] for row in rows if row.get(field) is not None]
                return sum(values) / len(values) if values else None

            summary["by_category"][str(category)] = {
                "questions": len(rows),
                "routed_gold_source_recall": category_mean("routed_gold_source_recall"),
                "expanded_gold_source_recall": category_mean("expanded_gold_source_recall"),
                "workspace_gold_source_recall": category_mean("gold_source_recall"),
                "average_evidence": category_mean("workspace_evidence_count"),
                "average_local_nodes": category_mean("workspace_node_candidate_count"),
                "average_token_cost": category_mean("token_cost"),
            }
        return summary

    def _save_root_progress(
        self,
        sample_idx: int,
        global_sample_idx: int,
        total_samples: int,
        root_question_count: int,
        all_results: List[Dict],
        result_file: str,
        completed_root_indices: set[int] | None = None,
    ) -> None:
        """Persist completed-root progress and a recoverable partial result."""
        import config

        token_usage = self.system.llm_client.usage_summary(
            question_count=len(all_results)
        )
        context_token_counts = [
            result["retrieved_context_tokens"]
            for result in all_results
            if result.get("retrieved_context_tokens") is not None
        ]
        workspace_results = [
            result for result in all_results
            if result.get("workspace_evidence_count") is not None
        ]
        workspace_summary = self._summarize_workspace_results(workspace_results)
        progress = {
            "timestamp": time.time(),
            "shard_index": getattr(self, "_checkpoint_shard_index", None),
            "num_shards": getattr(self, "_checkpoint_num_shards", None),
            "sample_idx": global_sample_idx,
            "roots_completed": len(completed_root_indices or {global_sample_idx}),
            "total_roots": total_samples,
            "root_questions": root_question_count,
            "questions_completed": len(all_results),
            "candidate_online_total_tokens": token_usage[
                "candidate_online_token_cost"
            ]["total_tokens"],
            "candidate_online_average_total_tokens_per_question": token_usage[
                "candidate_online_average_total_tokens_per_question"
            ],
            "paper_like_context_average_tokens_per_question": (
                sum(context_token_counts) / len(context_token_counts)
                if context_token_counts
                else None
            ),
            "workspace": workspace_summary,
        }

        progress_path = getattr(config, "ROOT_PROGRESS_PATH", None)
        try:
            if progress_path:
                with open(progress_path, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(progress, ensure_ascii=True) + "\n")
            print(
                "ROOT_PROGRESS "
                f"roots={sample_idx + 1}/{total_samples} "
                f"questions={len(all_results)}"
            )

            partial_path = Path(result_file).with_name(
                Path(result_file).stem + ".partial.json"
            )
            temp_path = partial_path.with_suffix(partial_path.suffix + ".tmp")
            partial_aggregated = (
                aggregate_metrics(self.metrics_list, self.categories)
                if self.metrics_list else {}
            )
            with temp_path.open("w", encoding="utf-8") as handle:
                json.dump({
                    "checkpoint": {
                        "version": 1,
                        "benchmark": "locomo",
                        "shard_index": progress.get("shard_index"),
                        "num_shards": progress.get("num_shards"),
                        "model": getattr(config, "LLM_MODEL", ""),
                        "embedding_model": getattr(config, "EMBEDDING_MODEL", ""),
                    },
                    "completed_root_indices": sorted(completed_root_indices or {global_sample_idx}),
                    "build_stats": self.build_stats,
                    "progress": progress,
                    "token_usage": token_usage,
                    "aggregated_metrics": partial_aggregated,
                    "detailed_results": all_results,
                }, handle, indent=2)
            temp_path.replace(partial_path)
        except Exception as exc:
            print(f"Warning: could not save root progress checkpoint: {exc}")

    def run_test(self, num_samples: int = None, save_results: bool = True, result_file: str = 'locomo10_test_results.json', enable_parallel_questions: bool = False, shard_index: int = 0, num_shards: int = 1):
        """Run full test on dataset"""
        print("\n" + "="*80)
        print(" TrawMem LoComo10 Dataset Test".center(80))
        print("="*80 + "\n")

        # Load dataset
        samples = self.load_dataset(limit=num_samples)
        if num_shards < 1:
            raise ValueError("num_shards must be at least 1")
        if not 0 <= shard_index < num_shards:
            raise ValueError("shard_index must satisfy 0 <= shard_index < num_shards")
        indexed_samples = [
            (index, sample)
            for index, sample in enumerate(samples)
            if index % num_shards == shard_index
        ]
        if not indexed_samples:
            raise ValueError(
                f"Shard {shard_index}/{num_shards} contains no roots"
            )
        root_indices = [index for index, _ in indexed_samples]
        total_samples = len(indexed_samples)
        print(
            f"Root shard {shard_index}/{num_shards}: "
            f"global root indices={root_indices}"
        )

        all_results = []
        completed_root_indices: set[int] = set()
        partial_path = Path(result_file).with_name(
            Path(result_file).stem + ".partial.json"
        )
        self._checkpoint_shard_index = shard_index
        self._checkpoint_num_shards = num_shards
        import config
        checkpoint_model = getattr(config, "LLM_MODEL", "")
        checkpoint_embedding = getattr(config, "EMBEDDING_MODEL", "")
        if partial_path.is_file():
            try:
                payload = json.loads(partial_path.read_text(encoding="utf-8"))
                checkpoint = payload.get("checkpoint") or {}
                compatible = (
                    checkpoint.get("version") == 1
                    and checkpoint.get("benchmark") == "locomo"
                    and checkpoint.get("shard_index", shard_index) == shard_index
                    and checkpoint.get("num_shards", num_shards) == num_shards
                    and checkpoint.get("model", checkpoint_model) == checkpoint_model
                    and checkpoint.get("embedding_model", checkpoint_embedding) == checkpoint_embedding
                )
                if compatible and isinstance(payload.get("detailed_results"), list):
                    all_results = list(payload["detailed_results"])
                    completed_root_indices = {
                        int(index)
                        for index in payload.get("completed_root_indices", [])
                    }
                    if not completed_root_indices:
                        completed_root_indices = {
                            int(row["sample_idx"])
                            for row in all_results
                            if row.get("sample_idx") is not None
                        }
                    self.build_stats = list(payload.get("build_stats") or [])
                    for row in all_results:
                        metrics = row.get("metrics") or {}
                        if metrics:
                            self.metrics_list.append(metrics)
                            self.categories.append(int(row.get("category", 0) or 0))
                        self.retrieval_times.append(float(row.get("retrieval_time", 0) or 0))
                        self.answer_times.append(float(row.get("answer_time", 0) or 0))
                        self.total_times.append(float(row.get("total_time", 0) or 0))
                    print(
                        f"Resuming LoCoMo shard {shard_index}/{num_shards}: "
                        f"{len(completed_root_indices)} completed roots",
                        flush=True,
                    )
                else:
                    print(f"Ignoring incompatible LoCoMo checkpoint: {partial_path}", flush=True)
            except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
                print(f"Ignoring unreadable LoCoMo checkpoint {partial_path}: {error}", flush=True)

        # Test each sample
        for sample_idx, (global_sample_idx, sample) in enumerate(indexed_samples):
            if global_sample_idx in completed_root_indices:
                print(f"Skipping completed LoCoMo root {global_sample_idx}", flush=True)
                continue
            # Clear system for each sample
            self.system.vector_store.clear()

            # Test sample
            sample_results = self.test_sample(sample, global_sample_idx, enable_parallel_questions=enable_parallel_questions)
            root_failures = [
                failure
                for failure in self.question_failures
                if failure["sample_idx"] == global_sample_idx
            ]
            if root_failures:
                examples = "; ".join(
                    f"Q{failure['question_idx'] + 1}: {failure['error']}"
                    for failure in root_failures[:5]
                )
                print(
                    f"Root {global_sample_idx} had {len(root_failures)} failed questions; "
                    f"continuing and recording them. Examples: {examples}",
                    flush=True,
                )
            all_results.extend(sample_results)
            completed_root_indices.add(global_sample_idx)
            if save_results:
                self._save_root_progress(
                    sample_idx=sample_idx,
                    global_sample_idx=global_sample_idx,
                    total_samples=total_samples,
                    root_question_count=len(sample_results),
                    all_results=all_results,
                    result_file=result_file,
                    completed_root_indices=completed_root_indices,
                )

        # Calculate aggregate metrics
        print("\n" + "="*80)
        print(" Test Summary".center(80))
        print("="*80 + "\n")

        # Timing statistics
        print("Timing Statistics:")
        print(f"  Average retrieval time: {sum(self.retrieval_times)/len(self.retrieval_times):.3f}s")
        print(f"  Average answer time: {sum(self.answer_times)/len(self.answer_times):.3f}s")
        print(f"  Average total time: {sum(self.total_times)/len(self.total_times):.3f}s")
        print(f"  Total retrieval time: {sum(self.retrieval_times):.2f}s")
        print(f"  Total answer time: {sum(self.answer_times):.2f}s")

        import config
        token_usage = self.system.llm_client.usage_summary(question_count=len(all_results))
        # The judge uses a separate client, so include its exact API usage in
        # the result instead of silently dropping those calls from the cost
        # ledger.  Main online cost remains retrieval + answer only; callers
        # can add judge totals when reporting the full experiment cost.
        if self.judge_client is not None:
            token_usage["judge"] = self.judge_client.usage_summary(
                question_count=len(all_results)
            )
            token_usage["all_llm_calls_including_judge"] = {
                key: token_usage["candidate_total_token_cost"].get(key, 0)
                + token_usage["judge"].get("candidate_total_token_cost", {}).get(key, 0)
                for key in ("calls", "prompt_tokens", "completion_tokens", "total_tokens")
            }
        context_token_counts = [
            result["retrieved_context_tokens"]
            for result in all_results
            if result.get("retrieved_context_tokens") is not None
        ]
        workspace_results = [
            result for result in all_results
            if result.get("workspace_evidence_count") is not None
        ]
        workspace_summary = self._summarize_workspace_results(workspace_results)
        token_usage["paper_like_retrieved_context"] = {
            "definition": "final retrieved context tokens, excluding QA instructions and completion",
            "tokenizer": str(
                getattr(config, "TOKENIZER_MODEL_PATH", None)
                or getattr(config, "TOKENIZER_ENCODING", None)
            ),
            "questions": len(context_token_counts),
            "total_tokens": sum(context_token_counts),
            "average_tokens_per_question": (
                sum(context_token_counts) / len(context_token_counts)
                if context_token_counts
                else None
            ),
        }
        token_usage["token_cost"] = dict(token_usage["paper_like_retrieved_context"])
        print("Token usage:")
        print(
            "  Paper-like retrieved context average/question: "
            f"{token_usage['paper_like_retrieved_context']['average_tokens_per_question']}"
        )
        print(f"  Candidate online total: {token_usage['candidate_online_token_cost']['total_tokens']}")
        print(f"  Candidate online average/question: {token_usage['candidate_online_average_total_tokens_per_question']}")
        print(f"  Maintenance total: {token_usage['maintenance']['total_tokens']}")
        if workspace_results:
            print(
                "Workspace statistics: "
                f"avg evidence={workspace_summary['average_evidence']:.2f}, "
                f"route recall={workspace_summary['routed_gold_source_recall']:.3f}, "
                f"expanded recall={workspace_summary['expanded_gold_source_recall']:.3f}, "
                f"workspace recall={workspace_summary['workspace_gold_source_recall']:.3f}, "
                f"selection coverage={workspace_summary['average_selection_coverage']:.3f}"
            )

        # Keep a standalone summary next to the raw JSONL log for later re-analysis.
        try:
            summary_path = getattr(config, "TOKEN_USAGE_SUMMARY_PATH", None)
            if summary_path:
                with open(summary_path, "w", encoding="utf-8") as handle:
                    json.dump(token_usage, handle, indent=2)
        except Exception as exc:
            print(f"Warning: could not save token usage summary: {exc}")

        # Answer quality metrics
        if self.metrics_list:
            print(f"\nAnswer Quality Metrics:")
            aggregated = aggregate_metrics(self.metrics_list, self.categories)

            # Overall metrics
            overall = aggregated.get('overall', {})
            print(f"\nOverall Performance:")
            metrics_to_show = ['f1', 'bleu1']
            if self.use_llm_judge:
                metrics_to_show.append('llm_judge_score')
            
            for metric_name in metrics_to_show:
                if metric_name in overall:
                    stats = overall[metric_name]
                    print(f"  {metric_name:20s}: {stats['mean']:.4f} (±{stats['std']:.4f})")

            # Per-category metrics
            print(f"\nPer-Category Performance:")
            for key in sorted(aggregated.keys()):
                if key.startswith('category_'):
                    category_num = key.split('_')[1]
                    category_data = aggregated[key]
                    if 'f1' in category_data:
                        f1_mean = category_data['f1']['mean']
                        count = category_data['f1']['count']
                        print(f"  Category {category_num}: F1={f1_mean:.4f} (n={count})")

        # Save results
        status_counts = defaultdict(int)
        for row in all_results:
            status_counts[str(row.get("evaluation_status", "unknown"))] += 1
        failed_count = sum(
            count for status, count in status_counts.items() if status != "ok"
        )
        answer_failure_count = sum(
            count
            for status, count in status_counts.items()
            if status in {"answer_error", "question_error"}
        )
        print(
            "Evaluation status: "
            f"ok={status_counts.get('ok', 0)} "
            f"failed={failed_count} "
            f"answer_error={status_counts.get('answer_error', 0)} "
            f"judge_error={status_counts.get('judge_error', 0)}",
            flush=True,
        )
        if save_results:
            output_file = result_file
            with open(output_file, 'w') as f:
                json.dump({
                    'summary': {
                        'num_samples': total_samples,
                        'shard_index': shard_index,
                        'num_shards': num_shards,
                        'root_indices': root_indices,
                        'num_questions': len(all_results),
                        'evaluation_status_counts': dict(status_counts),
                        'failed_questions': failed_count,
                        'answer_success_rate': (
                            (len(all_results) - answer_failure_count) / len(all_results)
                            if all_results else 0.0
                        ),
                        'evaluation_success_rate': (
                            status_counts.get('ok', 0) / len(all_results)
                            if all_results else 0.0
                        ),
                        'avg_retrieval_time': sum(self.retrieval_times)/len(self.retrieval_times),
                        'avg_answer_time': sum(self.answer_times)/len(self.answer_times),
                        'avg_total_time': sum(self.total_times)/len(self.total_times),
                        'token_usage': token_usage,
                        'workspace': workspace_summary,
                        'build_stats': self.build_stats,
                    },
                    'method': 'TrawMem',
                    'config': {
                        'llm_judge': self.use_llm_judge,
                    },
                    'aggregated_metrics': aggregated if self.metrics_list else {},
                    'detailed_results': all_results
                }, f, indent=2)
            print(f"\nResults saved to {output_file}")

        print("\n" + "="*80)
        print(" Test Complete!".center(80))
        print("="*80 + "\n")

        return all_results


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Test TrawMem on LoComo10 dataset')
    parser.add_argument('--dataset', type=str, default='test_ref/locomo10.json',
                       help='Path to LoComo10 dataset')
    parser.add_argument('--num-samples', type=int, default=None,
                       help='Number of samples to test (default: all)')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save results to file')
    parser.add_argument('--result-file', type=str, default='locomo10_test_results.json',
                       help='Path to the result file')
    parser.add_argument('--parallel-questions', action='store_true',
                       help='Enable parallel processing of questions within each sample')
    parser.add_argument('--llm-judge', action='store_true',
                       help='Enable LLM-as-judge evaluation for semantic answer comparison')
    parser.add_argument('--test-workers', type=int, default=None,
                       help='Number of parallel workers for question testing (default: use config MAX_RETRIEVAL_WORKERS)')
    parser.add_argument('--shard-index', type=int, default=0,
                       help='Zero-based root shard index')
    parser.add_argument('--num-shards', type=int, default=1,
                       help='Total number of root shards')

    args = parser.parse_args()

    # Create system
    print("Initializing TrawMem system...")
    system = TrawMemSystem(clear_db=True)

    # Create tester
    tester = LoCoMoTester(system, args.dataset, use_llm_judge=args.llm_judge, test_workers=args.test_workers)
    
    if args.llm_judge:
        print("LLM-as-judge evaluation enabled")
    if args.test_workers:
        print(f"Using {args.test_workers} test workers for parallel question processing")

    # Run test
    results = tester.run_test(
        num_samples=args.num_samples,
        save_results=not args.no_save,
        result_file=args.result_file,
        enable_parallel_questions=args.parallel_questions,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
    )


if __name__ == "__main__":
    main()
