import re
import time
from dataclasses import dataclass

_HF_BASE = "https://api-inference.huggingface.co/models/{model_id}/v1"


@dataclass
class ChatResponse:
    answer: str
    think: str | None
    tokens_used: int


def _split_think(raw: str) -> tuple[str | None, str]:
    m = re.search(r"<think>(.*?)</think>\s*", raw, flags=re.DOTALL)
    if m:
        return m.group(1).strip(), raw[m.end():].strip()
    return None, raw.strip()


class ModelClient:
    def __init__(
        self,
        provider: str,
        model_id: str,
        api_key: str,
        endpoint_url: str | None = None,
    ):
        self.provider = provider
        self.model_id = model_id
        self._endpoint_url = endpoint_url
        self._api_key = api_key
        self._openai_client = None
        self._anthropic_client = None

    def _get_openai_client(self):
        if self._openai_client is None:
            from openai import OpenAI

            base_url = self._resolve_base_url()
            self._openai_client = OpenAI(base_url=base_url, api_key=self._api_key)
        return self._openai_client

    def _resolve_base_url(self) -> str | None:
        if self.provider == "hf":
            return _HF_BASE.format(model_id=self.model_id)
        if self.provider == "endpoint":
            if not self._endpoint_url:
                raise ValueError("provider=endpoint requires --endpoint <url>")
            url = self._endpoint_url.rstrip("/")
            if not url.endswith("/v1"):
                url = url + "/v1"
            return url
        if self.provider == "openai":
            return None  # SDK default
        raise ValueError(f"Unknown provider: {self.provider}")

    def _get_anthropic_client(self):
        if self._anthropic_client is None:
            import anthropic

            self._anthropic_client = anthropic.Anthropic(api_key=self._api_key)
        return self._anthropic_client

    def chat(
        self,
        messages: list[dict],
        max_tokens: int = 500,
        temperature: float = 0.0,
        enable_thinking: bool = False,
    ) -> ChatResponse:
        if self.provider == "anthropic":
            return self._chat_anthropic(messages, max_tokens, temperature)
        return self._chat_openai(messages, max_tokens, temperature, enable_thinking)

    def _chat_openai(
        self,
        messages: list[dict],
        max_tokens: int,
        temperature: float,
        enable_thinking: bool,
    ) -> ChatResponse:
        client = self._get_openai_client()
        kwargs: dict = {
            "model": self.model_id,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if self.provider != "openai":
            kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": enable_thinking}}
        if enable_thinking:
            kwargs["max_tokens"] = max(max_tokens, 3000)

        resp = client.chat.completions.create(**kwargs)
        raw = resp.choices[0].message.content or ""
        think, answer = _split_think(raw)
        tokens = resp.usage.total_tokens if resp.usage else 0
        return ChatResponse(answer=answer, think=think, tokens_used=tokens)

    def _chat_anthropic(
        self,
        messages: list[dict],
        max_tokens: int,
        temperature: float,
    ) -> ChatResponse:
        client = self._get_anthropic_client()
        system_msgs = [m for m in messages if m["role"] == "system"]
        user_msgs = [m for m in messages if m["role"] != "system"]
        system_text = system_msgs[0]["content"] if system_msgs else None

        # Opus 4+ models deprecate temperature (they use extended thinking internally)
        _no_temp_models = {"claude-opus-4-7", "claude-opus-4-6"}
        kwargs: dict = {
            "model": self.model_id,
            "max_tokens": max_tokens,
            "messages": user_msgs,
        }
        if self.model_id not in _no_temp_models:
            kwargs["temperature"] = temperature
        if system_text:
            kwargs["system"] = system_text

        resp = client.messages.create(**kwargs)
        raw = resp.content[0].text if resp.content else ""
        think, answer = _split_think(raw)
        tokens = (resp.usage.input_tokens or 0) + (resp.usage.output_tokens or 0)
        return ChatResponse(answer=answer, think=think, tokens_used=tokens)

    def health_check(self) -> tuple[bool, float, str]:
        """Returns (ok, latency_s, detail)."""
        try:
            start = time.monotonic()
            resp = self.chat(
                [{"role": "user", "content": "Reply with the single word: OK"}],
                max_tokens=10,
                temperature=0.0,
            )
            latency = time.monotonic() - start
            ok = bool(resp.answer)
            return ok, latency, resp.answer[:80]
        except Exception as exc:
            return False, 0.0, str(exc)
