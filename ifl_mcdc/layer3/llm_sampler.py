"""
LLM 採樣器：呼叫 LLM 後端，解析 JSON，驗證，重試。

TC-U-47: 第一次成功回傳
TC-U-48: 解析 markdown 包裝的 JSON
TC-U-49: 第一次回傳壞 JSON → 第二次成功
TC-U-50: 全部重試失敗 → LLMSamplingError
TC-U-51: token_log 記錄每次嘗試 / 指數退避
"""
from __future__ import annotations

import json
import re
import time
from abc import ABC, abstractmethod

from ifl_mcdc.exceptions import LLMSamplingError
from ifl_mcdc.layer3.domain_validator import DomainValidator
from ifl_mcdc.models.validation import ValidationResult


# ─────────────────────────────────────────────
# LLM 後端抽象介面
# ─────────────────────────────────────────────


class LLMBackend(ABC):
    """LLM 後端統一介面。"""

    @abstractmethod
    def complete(self, prompt: str, max_tokens: int = 512) -> str:
        """呼叫 LLM 並回傳完成文字。"""
        ...


class OpenAIBackend(LLMBackend):
    """OpenAI ChatCompletion 後端。"""

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str = "",
        temperature: float = 0.3,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.temperature = temperature

    def complete(self, prompt: str, max_tokens: int = 512) -> str:
        """呼叫 OpenAI API。需安裝 openai 套件並設定 OPENAI_API_KEY。"""
        import openai
        client = openai.OpenAI(api_key=self.api_key or None)
        resp = client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content or ""


class AnthropicBackend(LLMBackend):
    """Anthropic Messages 後端。"""

    def __init__(self, model: str = "claude-sonnet-4-6", api_key: str = "") -> None:
        self.model = model
        self.api_key = api_key

    def complete(self, prompt: str, max_tokens: int = 512) -> str:
        """呼叫 Anthropic API。需安裝 anthropic 套件並設定 ANTHROPIC_API_KEY。"""
        import anthropic
        client = anthropic.Anthropic(api_key=self.api_key or None)
        msg = client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        block = msg.content[0]
        return block.text if hasattr(block, "text") else ""


class MockLLMBackend(LLMBackend):
    """測試用 Mock 後端，按順序回傳預設回應或拋出例外。"""

    def __init__(self, responses: list[str | BaseException]) -> None:
        self._responses = list(responses)
        self._index = 0

    def complete(self, prompt: str, max_tokens: int = 512) -> str:
        if self._index >= len(self._responses):
            raise LLMSamplingError("MockLLMBackend: 回應列表已耗盡")
        resp = self._responses[self._index]
        self._index += 1
        if isinstance(resp, BaseException):
            raise resp
        return resp


# ─────────────────────────────────────────────
# LLM 採樣器
# ─────────────────────────────────────────────


class LLMSampler:
    """呼叫 LLM 後端，重試解析，驗證輸出，記錄 token 消耗。"""

    MAX_RETRIES: int = 3
    RETRY_DELAY: float = 2.0

    def __init__(
        self,
        backend: LLMBackend,
        validator: DomainValidator,
    ) -> None:
        self.backend = backend
        self.validator = validator
        self.token_log: list[dict[str, object]] = []

    def sample(self, prompt: str) -> tuple[dict[str, object], ValidationResult]:
        """最多重試 MAX_RETRIES 次，回傳第一個通過驗證的 (data, result) 對。

        Args:
            prompt: Gap-Guided Prompt。

        Returns:
            (parsed_dict, ValidationResult) 元組，ValidationResult.passed=True。

        Raises:
            LLMSamplingError: 全部重試失敗。
        """
        current_prompt = prompt
        last_error: str = ""

        for attempt in range(1, self.MAX_RETRIES + 1):
            # 第 2 次起：退避等待
            if attempt > 1:
                time.sleep(self.RETRY_DELAY * (attempt - 1))
                current_prompt = self._build_retry_prompt(prompt, last_error)

            t_start = time.time()
            try:
                raw = self.backend.complete(current_prompt)
            except Exception as exc:
                elapsed = time.time() - t_start
                last_error = str(exc)
                self.token_log.append(
                    {
                        "attempt": attempt,
                        "elapsed": elapsed,
                        "est_tokens": 0,
                    }
                )
                continue

            elapsed = time.time() - t_start
            est_tokens = len(raw) // 4  # 粗略估算
            self.token_log.append(
                {
                    "attempt": attempt,
                    "elapsed": elapsed,
                    "est_tokens": est_tokens,
                }
            )

            data, parse_error = self._parse_json(raw)
            if data is None:
                last_error = f"JSON 解析失敗：{parse_error}"
                continue

            result = self.validator.validate(json.dumps(data))
            if result.passed:
                return data, result

            last_error = result.to_corrective_prompt()

        raise LLMSamplingError(
            f"LLM 採樣失敗，已重試 {self.MAX_RETRIES} 次。最後錯誤：{last_error}"
        )

    @staticmethod
    def _parse_json(raw: str) -> tuple[dict[str, object] | None, str | None]:
        """嘗試從 LLM 回應中解析 JSON 物件。

        步驟：
          1. 移除 ```json ... ``` 和 ``` 包裝
          2. json.loads(cleaned)
          3. 失敗 → 用 regex 找 { ... }（re.DOTALL）再試

        Returns:
            (dict, None) 成功；(None, error_str) 失敗。
        """
        # 移除 markdown code block
        cleaned = re.sub(r"```json\s*", "", raw)
        cleaned = re.sub(r"```\s*", "", cleaned)
        cleaned = cleaned.strip()

        try:
            data = json.loads(cleaned)
            if isinstance(data, dict):
                return data, None
            return None, f"期望 JSON 物件，得到 {type(data).__name__}"
        except json.JSONDecodeError:
            pass

        # 用 regex 提取 { ... }
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
                if isinstance(data, dict):
                    return data, None
            except json.JSONDecodeError as exc:
                return None, str(exc)

        return None, f"無法從回應中提取 JSON 物件：{raw[:80]!r}"

    @staticmethod
    def _build_retry_prompt(original: str, error: str) -> str:
        """建構重試提示詞。"""
        return (
            f"上一次你的回應有以下問題，請重新生成：\n"
            f"{error}\n\n"
            f"請僅輸出合法的 JSON 物件，不要有任何其他文字。\n\n"
            f"原始需求：\n"
            f"{original}"
        )
