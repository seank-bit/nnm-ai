"""AWS Bedrock 비동기 클라이언트 — Converse API + Bedrock API 키 (bearer token).

GroqClient 와 동일한 `chat(messages, temperature, max_tokens)` 인터페이스를 제공해
metrics.py 등 호출부 변경 없이 swap 가능.
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx
import structlog

log = structlog.get_logger()


class BedrockError(RuntimeError):
    pass


@dataclass
class BedrockClient:
    api_key: str
    model: str
    region: str = "ap-northeast-2"
    timeout: float = 60.0

    @property
    def _url(self) -> str:
        return (
            f"https://bedrock-runtime.{self.region}.amazonaws.com"
            f"/model/{self.model}/converse"
        )

    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> str:
        if not self.api_key:
            raise BedrockError("AWS_BEDROCK_API_KEY 가 비어 있습니다.")

        # OpenAI 스타일 messages → Bedrock Converse 포맷
        # OpenAI : {"role": "system|user|assistant", "content": "..."}
        # Bedrock: system 배열 별도 + messages 는 user/assistant 만, content 는 블록 배열
        system_parts: list[dict[str, str]] = []
        bedrock_messages: list[dict[str, object]] = []
        for m in messages:
            role = m["role"]
            content = m["content"]
            if role == "system":
                system_parts.append({"text": content})
            elif role in ("user", "assistant"):
                bedrock_messages.append(
                    {"role": role, "content": [{"text": content}]}
                )

        payload: dict[str, object] = {
            "messages": bedrock_messages,
            "inferenceConfig": {
                "temperature": temperature,
                "maxTokens": max_tokens or 2048,
            },
        }
        if system_parts:
            payload["system"] = system_parts

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(self._url, json=payload, headers=headers)
        if resp.status_code >= 400:
            log.warning(
                "bedrock.error", status=resp.status_code, body=resp.text[:500],
            )
            raise BedrockError(
                f"Bedrock API {resp.status_code}: {resp.text[:300]}"
            )
        data = resp.json()
        try:
            return data["output"]["message"]["content"][0]["text"]
        except (KeyError, IndexError, TypeError) as exc:
            raise BedrockError(f"unexpected response shape: {data}") from exc
