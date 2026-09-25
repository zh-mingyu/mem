"""
LLM Client - Handles all LLM interactions
"""
import json
import os
import threading
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from typing import List, Dict, Any, Optional, Iterator
from openai import OpenAI
from trawmem.core.settings import settings as config


_USAGE_CONTEXT: ContextVar[Dict[str, Any]] = ContextVar(
    "trawmem_usage_context", default={}
)


class LLMClient:
    """
    Unified LLM client interface
    """
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        enable_thinking: Optional[bool] = None,
        use_streaming: Optional[bool] = None
    ):
        self.api_key = api_key or config.OPENAI_API_KEY
        self.model = model or config.LLM_MODEL
        self.base_url = base_url or config.OPENAI_BASE_URL
        self.enable_thinking = enable_thinking if enable_thinking is not None else config.ENABLE_THINKING
        self.use_streaming = use_streaming if use_streaming is not None else config.USE_STREAMING
        self.api_keys = list(dict.fromkeys(
            key.strip() for key in str(self.api_key).split(",") if key.strip()
        ))
        if not self.api_keys:
            raise ValueError("At least one OpenAI API key is required")
        self._client_lock = threading.Lock()
        self._client_index = 0

        self.usage_log_path = getattr(config, "TOKEN_USAGE_LOG_PATH", None)
        if self.usage_log_path:
            os.makedirs(os.path.dirname(self.usage_log_path) or ".", exist_ok=True)
        self._usage_lock = threading.Lock()
        self._usage_events: List[Dict[str, Any]] = []

        # Initialize one client per key for direct round-robin rotation.
        if self.base_url:
            print(f"Using custom OpenAI base URL: {self.base_url}")

        if self.enable_thinking:
            print(f"Deep thinking mode enabled")

        self._clients = [
            OpenAI(base_url=self.base_url, api_key=key)
            for key in self.api_keys
        ]
        # Preserve the historical attribute for external callers.
        self.client = self._clients[0]
        if len(self._clients) > 1:
            print(f"OpenAI key rotation enabled: {len(self._clients)} keys")

    def _next_client(self):
        """Return the next client and its non-secret one-based slot number."""
        with self._client_lock:
            slot = self._client_index % len(self._clients)
            self._client_index += 1
        return self._clients[slot], slot + 1

    @staticmethod
    def _is_reasoning_model(model: object) -> bool:
        """Whether *model* uses the GPT-5/o-series chat-completions contract.

        GPT-5-compatible endpoints reject the legacy ``temperature`` and
        ``max_tokens`` request fields.  Model ids may be provider-qualified
        (for example ``openai/gpt-5.6-luna``), so inspect only the final path
        component.  This mirrors the login-node relay's rewrite rule while
        also making direct/API calls safe when they bypass that relay.
        """
        name = str(model or "").strip().lower().rsplit("/", 1)[-1]
        return name.startswith(("gpt-5", "gpt5", "o1", "o3", "o4"))

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        response_format: Optional[Dict[str, str]] = None,
        max_retries: int = 3,
        stage: str = "unspecified",
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Standard chat completion with optional thinking mode and retry mechanism
        """
        reasoning_model = self._is_reasoning_model(self.model)
        kwargs = {
            "model": self.model,
            "messages": messages,
        }
        # GPT-5/o-series models use provider-controlled sampling and reject
        # the legacy temperature field.  Keep the old behavior for all other
        # OpenAI-compatible models so existing Qwen/DeepSeek runs are stable.
        if not reasoning_model:
            kwargs["temperature"] = temperature

        if response_format:
            kwargs["response_format"] = response_format
        if max_tokens is not None:
            kwargs["max_completion_tokens" if reasoning_model else "max_tokens"] = int(max_tokens)

        # Enable thinking mode if configured (for Qwen and compatible models only)
        # Only add enable_thinking parameter for Qwen API (identified by base_url)
        is_qwen_api = self.base_url and "dashscope.aliyuncs.com" in self.base_url
        is_qwen_model = "qwen" in (self.model or "").lower()
        
        if is_qwen_api:
            # Qwen API requires explicit enable_thinking parameter
            # - Streaming + thinking: enable_thinking=True
            # - Non-streaming: enable_thinking=False (required, not optional)
            # - JSON format: enable_thinking=False (incompatible with thinking mode)
            if self.use_streaming and self.enable_thinking and not response_format:
                kwargs["extra_body"] = {"enable_thinking": True}
            else:
                # Explicitly set to False for non-streaming calls or JSON format
                kwargs["extra_body"] = {"enable_thinking": False}
        elif is_qwen_model:
            kwargs["extra_body"] = {
                "chat_template_kwargs": {
                    "enable_thinking": bool(self.enable_thinking and not response_format)
                }
            }
        # For OpenAI and other APIs, don't add extra_body parameters

        # Retry mechanism
        last_exception = None
        total_attempts = max(max_retries, len(self._clients))
        for attempt in range(total_attempts):
            client, key_slot = self._next_client()
            try:
                # Use streaming if configured
                if self.use_streaming:
                    kwargs["stream"] = True
                    # vLLM/OpenAI sends usage in the final stream chunk only when
                    # explicitly requested.  Keep streaming behavior unchanged.
                    kwargs["stream_options"] = {"include_usage": True}
                    content, usage = self._handle_streaming_response(client, **kwargs)
                    self._record_usage(
                        messages=messages,
                        response_text=content,
                        usage=usage,
                        stage=stage,
                        attempt=attempt + 1,
                        streaming=True,
                    )
                    return content
                else:
                    response = client.chat.completions.create(**kwargs)
                    content = response.choices[0].message.content
                    self._record_usage(
                        messages=messages,
                        response_text=content,
                        usage=getattr(response, "usage", None),
                        stage=stage,
                        attempt=attempt + 1,
                        streaming=False,
                    )
                    return content
                
                # kwargs["stream"] = True
                # return self._handle_streaming_response(**kwargs)
                    
            except Exception as e:
                last_exception = e
                self._record_usage(
                    messages=messages,
                    response_text="",
                    usage=None,
                    stage=stage,
                    attempt=attempt + 1,
                    success=False,
                    error=str(e),
                    streaming=self.use_streaming,
                )
                if attempt < total_attempts - 1:
                    wait_time = (
                        min(1.0, 0.2 * (attempt + 1))
                        if len(self._clients) > 1
                        else (2 ** attempt)
                    )
                    print(
                        f"LLM API call failed (attempt {attempt + 1}/{total_attempts}, "
                        f"key slot {key_slot}/{len(self._clients)}): {e}"
                    )
                    print(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    print(f"LLM API call failed after {total_attempts} attempts: {e}")
        
        # If all retries failed, raise the last exception.
        raise last_exception

    def _handle_streaming_response(self, client, **kwargs):
        """
        Handle streaming response and collect full content
        """
        full_content = []
        usage = None
        stream = client.chat.completions.create(**kwargs)

        # for chunk in stream:
        #     if chunk.choices is not None:
        #         print(chunk.choices[0].delta.content)
        
        # print('---------')

        for chunk in stream:
            if getattr(chunk, "usage", None) is not None:
                usage = chunk.usage
            # print(chunk)
            # fix list index out of range
            choices = getattr(chunk, "choices", None) or []
            if len(choices) > 0 and choices[0].delta.content is not None:
                content = choices[0].delta.content
                full_content.append(content)
                # print(full_content)
                # Optional: print streaming content in real-time
                # print(content, end='', flush=True)
        # print(full_content)
        print()
        return ''.join(full_content), usage

    @contextmanager
    def usage_scope(self, **context: Any) -> Iterator[None]:
        """Attach run/question metadata to all calls in the current thread."""
        previous = _USAGE_CONTEXT.get()
        merged = dict(previous)
        merged.update({key: value for key, value in context.items() if value is not None})
        token = _USAGE_CONTEXT.set(merged)
        try:
            yield
        finally:
            _USAGE_CONTEXT.reset(token)

    def usage_context(self) -> Dict[str, Any]:
        """Return the current thread's run/question metadata."""
        return dict(_USAGE_CONTEXT.get())

    @staticmethod
    def _usage_values(usage: Any) -> Optional[Dict[str, int]]:
        if usage is None:
            return None
        def value(name: str):
            if isinstance(usage, dict):
                return usage.get(name)
            return getattr(usage, name, None)
        prompt = value("prompt_tokens")
        completion = value("completion_tokens")
        total = value("total_tokens")
        if prompt is None and completion is None and total is None:
            return None
        prompt = int(prompt or 0)
        completion = int(completion or 0)
        total = int(total if total is not None else prompt + completion)
        return {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": total,
        }

    @staticmethod
    def _estimate_tokens(messages: List[Dict[str, str]], response_text: str) -> Dict[str, int]:
        """Conservative fallback only; estimates are explicitly marked as such."""
        prompt_chars = sum(len(str(message.get("content", ""))) for message in messages)
        completion_chars = len(response_text or "")
        prompt = max(1, round(prompt_chars / 4)) if prompt_chars else 0
        completion = max(1, round(completion_chars / 4)) if completion_chars else 0
        return {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
        }

    def _record_usage(
        self,
        messages: List[Dict[str, str]],
        response_text: str,
        usage: Any,
        stage: str,
        attempt: int,
        success: bool = True,
        error: Optional[str] = None,
        streaming: bool = False,
    ) -> None:
        exact_usage = self._usage_values(usage)
        if exact_usage is None:
            token_values = self._estimate_tokens(messages, response_text)
            usage_source = "char_estimate"
        else:
            token_values = exact_usage
            usage_source = "api"
        event = {
            "event_id": str(uuid.uuid4()),
            "timestamp": time.time(),
            "model": self.model,
            "request_type": "chat.completions",
            "stage": stage,
            "context": dict(_USAGE_CONTEXT.get()),
            "attempt": attempt,
            "streaming": streaming,
            "success": success,
            "usage_source": usage_source,
            "usage_exact": usage_source == "api",
            "prompt_chars": sum(len(str(message.get("content", ""))) for message in messages),
            "completion_chars": len(response_text or ""),
            **token_values,
        }
        if error:
            event["error"] = error
        with self._usage_lock:
            self._usage_events.append(event)
            if self.usage_log_path:
                with open(self.usage_log_path, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(event, ensure_ascii=True) + "\n")

    def usage_summary(self, question_count: Optional[int] = None) -> Dict[str, Any]:
        """Return raw and candidate cost views without discarding alternate readings."""
        with self._usage_lock:
            events = list(self._usage_events)
        stages: Dict[str, Dict[str, int]] = {}
        exact_calls = estimated_calls = 0
        for event in events:
            if event.get("success"):
                bucket = stages.setdefault(event["stage"], {
                    "calls": 0, "prompt_tokens": 0,
                    "completion_tokens": 0, "total_tokens": 0,
                })
                bucket["calls"] += 1
                for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                    bucket[key] += int(event.get(key, 0))
                if event.get("usage_exact"):
                    exact_calls += 1
                else:
                    estimated_calls += 1

        def totals(prefixes):
            selected = [bucket for name, bucket in stages.items()
                        if any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes)]
            return {
                "calls": sum(item["calls"] for item in selected),
                "prompt_tokens": sum(item["prompt_tokens"] for item in selected),
                "completion_tokens": sum(item["completion_tokens"] for item in selected),
                "total_tokens": sum(item["total_tokens"] for item in selected),
            }

        maintenance = totals(("maintenance",))
        retrieval = totals(("retrieval",))
        answer = totals(("answer",))
        judge = totals(("judge",))
        online = {
            key: retrieval[key] + answer[key]
            for key in ("calls", "prompt_tokens", "completion_tokens", "total_tokens")
        }
        denominator = question_count or 0
        return {
            "definition": {
                "candidate_online_token_cost": "retrieval + answer; maintenance/build excluded",
                "candidate_total_token_cost": "maintenance + retrieval + answer",
                "token_units": "prompt + completion tokens",
                "exactness": "api usage is exact; char_estimate is fallback only",
            },
            "calls_recorded": len(events),
            "successful_calls": sum(1 for event in events if event.get("success")),
            "failed_attempts_recorded": sum(1 for event in events if not event.get("success")),
            "exact_calls": exact_calls,
            "estimated_calls": estimated_calls,
            "by_stage": stages,
            "maintenance": maintenance,
            "retrieval": retrieval,
            "answer": answer,
            "judge": judge,
            "candidate_online_token_cost": online,
            "candidate_total_token_cost": {
                key: maintenance[key] + online[key]
                for key in ("calls", "prompt_tokens", "completion_tokens", "total_tokens")
            },
            "question_count": denominator,
            "candidate_online_average_total_tokens_per_question": (
                online["total_tokens"] / denominator if denominator else None
            ),
            "candidate_total_average_tokens_per_question": (
                (maintenance["total_tokens"] + online["total_tokens"]) / denominator
                if denominator else None
            ),
        }

    def usage_events(self) -> List[Dict[str, Any]]:
        """Return a snapshot of recorded calls for per-question accounting."""
        with self._usage_lock:
            return [dict(event) for event in self._usage_events]

    def question_call_count(self, sample_idx: int, question_idx: int) -> int:
        """Count successful retrieval/answer calls for one query, excluding judge."""
        return sum(
            1
            for event in self.usage_events()
            if event.get("success")
            and event.get("context", {}).get("sample_idx") == sample_idx
            and event.get("context", {}).get("question_idx") == question_idx
            and (
                str(event.get("stage", "")).startswith("retrieval")
                or str(event.get("stage", "")) == "answer"
            )
        )

    def extract_json(self, text: str) -> Any:
        """
        Extract JSON from LLM response with robust parsing
        Supports multiple formats:
        1. Pure JSON
        2. ```json ... ```
        3. ``` ... ``` (generic code block)
        4. JSON embedded in text with common prefixes
        5. Multiple JSON objects (returns first valid one)
        """
        if not text or not text.strip():
            raise ValueError("Empty response received")

        text = text.strip()

        # Remove common LLM prefixes/suffixes
        common_prefixes = [
            "Here's the JSON:",
            "Here is the JSON:",
            "The JSON is:",
            "JSON:",
            "Result:",
            "Output:",
            "Answer:",
        ]
        for prefix in common_prefixes:
            if text.lower().startswith(prefix.lower()):
                text = text[len(prefix):].strip()

        # Try direct parsing first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try extracting JSON from ```json ... ``` block
        if "```json" in text.lower():
            # Case insensitive search for ```json
            start_marker = "```json"
            start_idx = text.lower().find(start_marker)
            if start_idx != -1:
                start = start_idx + len(start_marker)
                # Find the closing ```
                end = text.find("```", start)
                if end != -1:
                    json_str = text[start:end].strip()
                    try:
                        return json.loads(json_str)
                    except json.JSONDecodeError as e:
                        # Try to clean up common issues
                        json_str = self._clean_json_string(json_str)
                        try:
                            return json.loads(json_str)
                        except json.JSONDecodeError:
                            pass

        # Try extracting from generic ``` ... ``` code block
        if "```" in text:
            start = text.find("```") + 3
            # Skip language identifier if present
            newline = text.find("\n", start)
            if newline != -1 and newline - start < 20:
                start = newline + 1
            end = text.find("```", start)
            if end != -1:
                json_str = text[start:end].strip()
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    # Try to clean up
                    json_str = self._clean_json_string(json_str)
                    try:
                        return json.loads(json_str)
                    except json.JSONDecodeError:
                        pass

        # Try finding balanced JSON object/array by scanning for { or [
        for start_char in ['{', '[']:
            result = self._extract_balanced_json(text, start_char)
            if result is not None:
                return result

        # Last resort: try to find any JSON-like structure and clean it
        for start_char in ['{', '[']:
            start_idx = text.find(start_char)
            if start_idx != -1:
                # Extract a large chunk and try to parse
                chunk = text[start_idx:]
                cleaned = self._clean_json_string(chunk)
                try:
                    return json.loads(cleaned)
                except json.JSONDecodeError:
                    pass

        raise ValueError(f"Failed to extract valid JSON from response. First 300 chars: {text[:300]}...")

    def _clean_json_string(self, json_str: str) -> str:
        """
        Clean common issues in JSON strings from LLM output
        """
        # Remove trailing commas before } or ]
        import re
        json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)

        # Remove comments (// and /* */)
        json_str = re.sub(r'//.*?$', '', json_str, flags=re.MULTILINE)
        json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)

        return json_str.strip()

    def _extract_balanced_json(self, text: str, start_char: str) -> Any:
        """
        Extract a balanced JSON object or array starting with start_char
        """
        end_char = '}' if start_char == '{' else ']'
        start_idx = text.find(start_char)

        if start_idx == -1:
            return None

        # Track depth to find matching closing bracket
        depth = 0
        in_string = False
        escape_next = False

        for i in range(start_idx, len(text)):
            char = text[i]

            # Handle string escaping
            if escape_next:
                escape_next = False
                continue

            if char == '\\':
                escape_next = True
                continue

            # Handle strings (don't count brackets inside strings)
            if char == '"':
                in_string = not in_string
                continue

            if in_string:
                continue

            # Count depth
            if char == start_char:
                depth += 1
            elif char == end_char:
                depth -= 1
                if depth == 0:
                    json_str = text[start_idx:i+1]
                    try:
                        return json.loads(json_str)
                    except json.JSONDecodeError:
                        # Try cleaning and parsing again
                        cleaned = self._clean_json_string(json_str)
                        try:
                            return json.loads(cleaned)
                        except json.JSONDecodeError:
                            # Continue searching for next occurrence
                            break

        return None
