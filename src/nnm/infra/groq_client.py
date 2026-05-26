from __future__ import annotations
from dataclasses import dataclass

import httpx
import structlog

log = structlog.get_logger()


class GroqError(RuntimeError):
    pass


@dataclass
class GroqClient:
    api_key: str
    model: str
    base_url: str = "https://api.groq.com/openai/v1"
    timeout: float = 60.0

    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> str:
        if not self.api_key:
            raise GroqError("NNM_GROQ_API_KEY 가 비어 있습니다. .env 에 키를 채워주세요.")
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        payload: dict[str, object] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(url, json=payload, headers=headers)
        if resp.status_code >= 400:
            log.warning("groq.error", status=resp.status_code, body=resp.text[:500])
            raise GroqError(f"Groq API {resp.status_code}: {resp.text[:300]}")
        data = resp.json()
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            raise GroqError(f"unexpected Groq response shape: {data}") from exc
