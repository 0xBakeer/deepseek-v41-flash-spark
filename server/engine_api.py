"""Engine interface for the OpenAI-compatible server.

The server talks to an inference engine through the small abstract class below.
The real DeepSeek-V4.1-Flash engine lives in ``engine/v41_engine.py`` (repo
root) and is imported lazily by ``app.py --engine v41``; ``MockEngine`` is a
tokenizer-only stand-in so the HTTP layer can be exercised without a GPU.

Contract (see ``Engine``):

* ``generate`` is a generator that yields *bursts* of newly generated token ids.
  Speculative decoding (DSpark) typically produces several tokens per step, so
  a burst is ``list[int]`` of length >= 1. The engine should stop after it has
  emitted a token in ``stop_token_ids`` (the stop token itself may be included
  in the last burst; the server truncates at the first stop id it sees) and it
  must stop after ``max_tokens`` tokens at the latest. After an EOS/stop id or
  ``max_tokens`` the server drains the generator (up to 4 more bursts, ignored)
  so an engine whose loop ends there gets to run its epilogue (stats). On a
  stop *string* or a client disconnect the server closes the generator
  instead -- ``GeneratorExit`` is raised at the pending ``yield`` -- so
  cache/arena cleanup belongs in a ``try/finally`` around the decode loop and
  the engine must be ready for the next request afterwards.
* Optional ``context_margin`` (int, default 8 as read by the server): tokens
  the engine needs beyond ``len(prompt_ids) + max_tokens`` (DSpark draft
  block); the server clamps ``max_tokens`` so that
  ``prompt + max_tokens + margin <= max_context``.
* The server serialises calls: at most one ``generate`` runs at any time.
* ``stats`` returns whatever the engine wants to expose about the last
  generation (tok/s, draft acceptance rate, expert cache hit rate ...). It is
  passed through verbatim as ``x_engine_stats`` in responses.
"""

from __future__ import annotations

import json
import random
import re
import time
from abc import ABC, abstractmethod
from typing import Callable, Iterator, List, Optional, Set


class Engine(ABC):
    """Single-sequence text generation engine."""

    #: id of ``<｜end▁of▁sentence｜>`` (1 for DeepSeek-V4.1).
    eos_token_id: int = 1
    #: maximum prompt + completion length the engine accepts.
    max_context: int = 32768

    @abstractmethod
    def generate(
        self,
        prompt_ids: List[int],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop_token_ids: Set[int],
        seed: Optional[int],
    ) -> Iterator[List[int]]:
        """Yield bursts of newly generated token ids for ``prompt_ids``."""

    def stats(self) -> dict:
        """Engine statistics for the most recent generation (free-form)."""
        return {}

    def close(self) -> None:
        """Release resources; called once at server shutdown."""


# The mock reply, split into the pieces the mock engine assembles depending on
# what the prompt asks for.
MOCK_REASONING = (
    "The user is talking to a mock engine. I should answer briefly, "
    "mention that this is canned text, and keep the tone friendly."
)
MOCK_CONTENT = (
    "Hello from the mock engine! This is a canned reply.\n\n"
    "It streams in small bursts of one to four tokens so the SSE path, "
    "the incremental detokenizer and the reasoning/content split can be "
    "tested end to end without a GPU."
)
MOCK_TOOL_CONTENT = "Let me look that up."

_TOOL_SCHEMAS_HEADING = "### Available Tool Schemas\n\n"


class MockEngine(Engine):
    """Tokenizer-only engine that replays a canned reply in random bursts.

    * If the prompt ends with ``<think>`` the reply starts with a reasoning
      block terminated by ``</think>``; if it ends with ``</think>`` (chat
      mode) the reply is content only.
    * If the prompt carries a V4.1 tool schema block the reply is a valid DSML
      tool call against the first listed tool (first declared parameter,
      string-typed), so tool-call parsing can be verified.
    * Bursts have 1-4 tokens and the sequence ends with ``eos_token_id``.
    """

    def __init__(
        self,
        encode: Callable[[str], List[int]],
        decode: Callable[[List[int]], str],
        *,
        eos_token_id: int = 1,
        think_start_id: int = 128821,
        think_end_id: int = 128822,
        max_context: int = 32768,
        burst_delay_s: float = 0.0,
    ) -> None:
        self._encode = encode
        self._decode = decode
        self.eos_token_id = eos_token_id
        self.think_start_id = think_start_id
        self.think_end_id = think_end_id
        self.max_context = max_context
        self.burst_delay_s = burst_delay_s
        self._last_stats: dict = {"engine": "mock"}

    # -- helpers ---------------------------------------------------------
    @staticmethod
    def _first_tool(prompt_text: str):
        """Return (name, first_param_name) of the first tool schema, or None."""
        pos = prompt_text.rfind(_TOOL_SCHEMAS_HEADING)
        if pos < 0:
            return None
        block = prompt_text[pos + len(_TOOL_SCHEMAS_HEADING):]
        first_line = block.split("\n", 1)[0]
        try:
            schema = json.loads(first_line)
            name = schema["name"]
            props = (schema.get("parameters") or {}).get("properties") or {}
            param = next(iter(props), "query")
            return name, param
        except Exception:
            m = re.search(r'"name":\s*"([^"]+)"', first_line)
            return (m.group(1), "query") if m else None

    def _reply_text(self, prompt_ids: List[int]) -> str:
        thinking = bool(prompt_ids) and prompt_ids[-1] == self.think_start_id
        prompt_text = self._decode(prompt_ids)
        tool = self._first_tool(prompt_text)
        parts = []
        if thinking:
            parts.append(MOCK_REASONING + "</think>")
        if tool is not None:
            name, param = tool
            parts.append(MOCK_TOOL_CONTENT)
            parts.append(
                "\n\n<｜DSML｜ calls>\n"
                f'<｜DSML｜ invoke name="{name}">\n'
                f'<｜DSML｜ parameter name="{param}" string="true">Berlin</｜DSML｜ parameter>\n'
                "</｜DSML｜ invoke>\n"
                "</｜DSML｜ calls>"
            )
        else:
            parts.append(MOCK_CONTENT)
        return "".join(parts)

    # -- Engine ----------------------------------------------------------
    def generate(
        self,
        prompt_ids: List[int],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop_token_ids: Set[int],
        seed: Optional[int],
    ) -> Iterator[List[int]]:
        ids = self._encode(self._reply_text(prompt_ids)) + [self.eos_token_id]
        ids = ids[:max_tokens]
        rng = random.Random(seed if seed is not None else 1234)
        t0 = time.perf_counter()
        emitted = 0
        i = 0
        try:
            while i < len(ids):
                burst = ids[i:i + rng.randint(1, 4)]
                i += len(burst)
                emitted += len(burst)
                if self.burst_delay_s:
                    time.sleep(self.burst_delay_s)
                yield burst
                if any(t in stop_token_ids for t in burst):
                    break
        finally:
            # The server closes the generator as soon as it has seen a stop id
            # or a stop string, so stats must be finalised here, not after the
            # loop. A real engine should do the same (and abort its sequence).
            dt = max(time.perf_counter() - t0, 1e-9)
            self._last_stats = {
                "engine": "mock",
                "tok_per_s": round(emitted / dt, 1),
                "acceptance_rate": 1.0,
                "expert_cache_hit_rate": 1.0,
                "generated_tokens": emitted,
                "prompt_tokens": len(prompt_ids),
            }

    def stats(self) -> dict:
        return dict(self._last_stats)
