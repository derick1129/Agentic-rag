"""OpenAI embedding and generation providers."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from openai import AsyncOpenAI

from adaptive.interfaces import EmbeddingProvider, Generator

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pydantic import BaseModel

logger = logging.getLogger(__name__)


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """Embedding provider that uses OpenAI's embedding API."""

    def __init__(
        self,
        api_key: str,
        api_base: str = "https://api.openai.com/v1",
        model: str = "text-embedding-3-small",
        dimensions: int = 1536,
        input_type: str | None = None,
    ) -> None:
        self.client = AsyncOpenAI(api_key=api_key, base_url=api_base)
        self.model = model
        self._dimensions = dimensions
        self._input_type = input_type

    async def embed(
        self, texts: Sequence[str], *, input_type: str | None = None
    ) -> list[list[float]]:
        kwargs: dict[str, Any] = {"input": texts, "model": self.model}
        # Request an explicit output width so embeddings match the configured
        # schema dimensions. OpenAI text-embedding-3 models honor this; NVIDIA
        # NIMS dynamic-width models honor it via their Matryoshka support.
        if self._dimensions:
            kwargs["dimensions"] = self._dimensions
        effective = input_type or self._input_type
        # NVIDIA embedding models (e.g. llama-nemotron-embed-vl) require the
        # asymmetric "input_type" flag (passage for indexing, query for
        # retrieval) and accept it via extra_body. It is only sent for NIMS
        # models since generic OpenAI embeddings have no such parameter.
        is_nims = "nvidia" in self.model.lower() or "nemotron" in self.model.lower()
        if effective and is_nims:
            kwargs["extra_body"] = {"input_type": effective}
        response = await self.client.embeddings.create(**kwargs)
        # The response data is a list of objects with an embedding attribute
        return [item.embedding for item in response.data]

    @property
    def dimensions(self) -> int:
        return self._dimensions


class OpenAIGenerationProvider(Generator):
    """Generation provider that uses OpenAI's completion API."""

    def __init__(
        self,
        api_key: str,
        api_base: str = "https://api.openai.com/v1",
        model: str = "gpt-4o-mini",
    ) -> None:
        self.client = AsyncOpenAI(api_key=api_key, base_url=api_base)
        self.model = model
        self.last_usage: dict[str, Any] = {}

    async def generate(self, prompt: str, max_tokens: int, temperature: float = 0.0) -> str:
        # Use the OpenAI-compatible chat completions endpoint. Legacy
        # /completions is not served by hosted providers such as NVIDIA NIMS.
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        if response.usage:
            self.last_usage = {
                "tokens": response.usage.total_tokens,
                "total_tokens": response.usage.total_tokens,
            }
        return (response.choices[0].message.content or "").strip()

    async def generate_structured(
        self, prompt: str, schema: type[BaseModel], max_tokens: int
    ) -> BaseModel:
        # First attempt: OpenAI structured outputs API (.beta.chat.completions.parse)
        try:
            parsed_response = await self.client.beta.chat.completions.parse(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format=schema,
                max_tokens=max_tokens,
            )
            if parsed_response.usage:
                self.last_usage = {
                    "tokens": parsed_response.usage.total_tokens,
                    "total_tokens": parsed_response.usage.total_tokens,
                }
            parsed = parsed_response.choices[0].message.parsed
            if parsed is not None:
                return parsed
        except Exception:
            # Fallback for providers (e.g. NVIDIA NIM / local proxies) without .beta parse
            logger.debug("Structured parse failed; falling back to JSON completion", exc_info=True)

        # Fallback: Instruct model to return JSON adhering to schema
        schema_json = json.dumps(schema.model_json_schema())
        system_instruction = (
            "You are a helpful assistant. You must respond with a JSON object strictly "
            f"conforming to this JSON schema:\n{schema_json}\n"
            "Do not include any explanation or markdown formatting outside the JSON."
        )
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
                temperature=0.0,
                response_format={"type": "json_object"},
            )
        except Exception:
            # If response_format={"type": "json_object"} is unsupported, try standard call
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
                temperature=0.0,
            )

        if response.usage:
            self.last_usage = {
                "tokens": response.usage.total_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        content = response.choices[0].message.content or "{}"
        content = content.strip()
        if content.startswith("```"):
            lines = content.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            content = "\n".join(lines).strip()

        return schema.model_validate_json(content)
