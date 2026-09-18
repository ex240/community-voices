"""Injectable text generation. The OpenAI client is created only when constructed."""

from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

WRITING_TOKEN_NOTE = "Writing-model tokens only; embedding usage is not included."
RAG_LATENCY_LABEL = "End-to-end report latency"
BASELINE_LATENCY_LABEL = "Baseline generation latency"


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


class TextGenerator(Protocol):
    model: str

    def complete(self, *, instructions: str, input_text: str) -> str:
        """Return model text. Callers treat the result as untrusted structured output."""


class OpenAIWriter:
    """Thin wrapper around the official OpenAI Responses API."""

    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        temperature: float,
        timeout_seconds: int,
    ) -> None:
        if not api_key.strip():
            raise ValueError("OpenAI API key is not configured")
        from openai import OpenAI

        self.model = model
        self.last_usage: TokenUsage | None = None
        self._temperature = temperature
        self._client = OpenAI(api_key=api_key, timeout=timeout_seconds)

    def complete(self, *, instructions: str, input_text: str) -> str:
        response = self._client.responses.create(
            model=self.model,
            instructions=instructions,
            input=input_text,
            temperature=self._temperature,
            text={"format": {"type": "json_object"}},
        )
        self.last_usage = usage_from_response(response)
        text = (response.output_text or "").strip()
        if not text:
            raise RuntimeError("The writing model returned empty text.")
        return text


def usage_from_response(response: object) -> TokenUsage | None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    input_tokens = getattr(usage, "input_tokens", None)
    output_tokens = getattr(usage, "output_tokens", None)
    total_tokens = getattr(usage, "total_tokens", None)
    if input_tokens is None and output_tokens is None and total_tokens is None:
        return None
    return TokenUsage(
        input_tokens=_optional_int(input_tokens),
        output_tokens=_optional_int(output_tokens),
        total_tokens=_optional_int(total_tokens),
    )


def merge_usage(*usages: TokenUsage | None) -> TokenUsage | None:
    present = [item for item in usages if item is not None]
    if not present:
        return None
    return TokenUsage(
        input_tokens=_sum_optional(item.input_tokens for item in present),
        output_tokens=_sum_optional(item.output_tokens for item in present),
        total_tokens=_sum_optional(item.total_tokens for item in present),
    )


def timed_complete(
    writer: TextGenerator,
    *,
    instructions: str,
    input_text: str,
) -> tuple[str, float, TokenUsage | None]:
    started = time.perf_counter()
    text = writer.complete(instructions=instructions, input_text=input_text)
    elapsed = time.perf_counter() - started
    return text, elapsed, getattr(writer, "last_usage", None)


def format_run_metrics(
    *,
    latency_seconds: float | None,
    total_tokens: int | None,
    latency_label: str,
) -> str:
    latency = "unavailable" if latency_seconds is None else f"{latency_seconds}s"
    tokens = "unavailable" if total_tokens is None else str(total_tokens)
    return f"{latency_label}: {latency}. {WRITING_TOKEN_NOTE} Total: {tokens}."


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _sum_optional(values: Iterable[int | None]) -> int | None:
    nums = [item for item in values if item is not None]
    if not nums:
        return None
    return int(sum(nums))
