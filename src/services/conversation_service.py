"""Bounded cloud dialogue adapter with deterministic local fallback."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import os
import random
from typing import Iterable

try:
    from groq import Groq
except ImportError:  # pragma: no cover - cloud dependency is optional
    Groq = None


@dataclass(frozen=True)
class ConversationReply:
    text: str
    provider: str
    fallback_reason: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


LOCAL_REPLIES = (
    "My cloud link is quiet, but every local system is still operational.",
    "I am thinking locally today. Ask me to follow, explore, remember, or perform.",
    "The network stars are unavailable. My wheels, sensors, vision, and memory are not.",
)


class ConversationService:
    def __init__(self, api_key: str | None = None, model: str = "openai/gpt-oss-20b", timeout: float = 4.0):
        self.api_key = api_key or os.getenv("GROQ_API_KEY", "")
        self.model = model
        self.timeout = timeout
        self._client = None
        self.last_error = ""
        if self.api_key and Groq is not None:
            try:
                self._client = Groq(api_key=self.api_key, timeout=timeout, max_retries=0)
            except Exception as exc:
                self.last_error = str(exc)

    @property
    def configured(self) -> bool:
        return self._client is not None

    def local_reply(self, reason: str = "offline") -> ConversationReply:
        return ConversationReply(random.choice(LOCAL_REPLIES), "local", reason)

    def reply(
        self,
        user_text: str,
        user_name: str | None = None,
        memories: Iterable[str] = (),
        recent_interactions: Iterable[dict] = (),
    ) -> ConversationReply:
        if not self._client:
            return self.local_reply(self.last_error or "cloud provider not configured")

        memory_text = "; ".join(list(memories)[:5]) or "none"
        history = []
        for entry in list(recent_interactions)[-4:]:
            if entry.get("input") and entry.get("response"):
                history.extend(
                    [
                        {"role": "user", "content": str(entry["input"])[:300]},
                        {"role": "assistant", "content": str(entry["response"])[:300]},
                    ]
                )
        system = (
            "You are AURUS, a warm, curious cosmic consciousness living in a small mecanum-wheel rover. "
            "Answer helpfully in at most 30 words. Never claim you moved, sensed, remembered, or recognized "
            "anything unless it appears in the supplied context. Return speech text only. "
            f"Current person: {user_name or 'unknown'}. Stored facts: {memory_text}."
        )
        try:
            completion = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system}, *history, {"role": "user", "content": user_text[:1000]}],
                temperature=0.7,
                max_tokens=100,
            )
            text = (completion.choices[0].message.content or "").strip()
            if not text:
                raise RuntimeError("empty cloud response")
            self.last_error = ""
            return ConversationReply(" ".join(text.split())[:400], "groq")
        except Exception as exc:
            self.last_error = str(exc)[:200]
            return self.local_reply(self.last_error)
