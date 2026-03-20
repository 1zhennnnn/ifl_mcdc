"""
Layer 3 LLMSampler 單元測試。

TC-U-47: 第一次成功回傳
TC-U-48: 解析 markdown 包裝的 JSON
TC-U-49: 第一次回傳壞 JSON → 第二次成功
TC-U-50: 全部重試失敗 → LLMSamplingError
TC-U-51: token_log 記錄每次嘗試 / 指數退避
"""
from __future__ import annotations

import json
import time
from unittest.mock import patch

import pytest

from ifl_mcdc.exceptions import LLMSamplingError
from ifl_mcdc.layer3.domain_validator import DomainValidator
from ifl_mcdc.layer3.llm_sampler import LLMSampler, MockLLMBackend
from ifl_mcdc.models.validation import DomainRule


def _sampler_no_rules(responses: list[str | BaseException]) -> LLMSampler:
    """建立不帶任何領域規則的採樣器（接受所有合法 JSON）。"""
    validator = DomainValidator(rules=[])
    backend = MockLLMBackend(responses)
    return LLMSampler(backend=backend, validator=validator)


def _sampler_with_age(responses: list[str | BaseException]) -> LLMSampler:
    """建立帶 age 驗證規則的採樣器。"""
    rules = [
        DomainRule(
            field="age",
            description="年齡必須在 0～130 之間",
            validator=lambda v: isinstance(v, int) and not isinstance(v, bool) and 0 <= v <= 130,
        )
    ]
    validator = DomainValidator(rules=rules)
    backend = MockLLMBackend(responses)
    return LLMSampler(backend=backend, validator=validator)


# ─────────────────────────────────────────────
# TC-U-47：第一次成功回傳
# ─────────────────────────────────────────────


def test_success_first_try():  # type: ignore[no-untyped-def]
    """TC-U-47：第一次回傳合法 JSON → 直接成功，不重試。"""
    payload = {"age": 30, "egg_allergy": False}
    sampler = _sampler_no_rules([json.dumps(payload)])

    result_data, vr = sampler.sample("test prompt")

    assert result_data == payload
    assert vr.passed is True
    assert len(sampler.token_log) == 1, "只應有 1 次嘗試記錄"


# ─────────────────────────────────────────────
# TC-U-48：解析 markdown 包裝的 JSON
# ─────────────────────────────────────────────


def test_parse_markdown_wrapped():  # type: ignore[no-untyped-def]
    """TC-U-48：LLM 回傳 ```json {...} ``` 包裝時應正確解析。"""
    payload = {"age": 25, "high_risk": True}
    raw = f"```json\n{json.dumps(payload)}\n```"
    sampler = _sampler_no_rules([raw])

    result_data, vr = sampler.sample("test prompt")

    assert result_data == payload
    assert vr.passed is True


# ─────────────────────────────────────────────
# TC-U-49：第一次回傳壞 JSON → 第二次成功
# ─────────────────────────────────────────────


def test_retry_on_bad_json():  # type: ignore[no-untyped-def]
    """TC-U-49：第一次回傳壞 JSON，第二次成功。"""
    good = {"age": 50}
    sampler = _sampler_no_rules(["not json at all", json.dumps(good)])

    # 攔截 time.sleep，避免測試等待
    with patch("ifl_mcdc.layer3.llm_sampler.time.sleep"):
        result_data, vr = sampler.sample("test prompt")

    assert result_data == good
    assert vr.passed is True
    assert len(sampler.token_log) == 2, "應有 2 次嘗試記錄"


# ─────────────────────────────────────────────
# TC-U-50：全部重試失敗 → LLMSamplingError
# ─────────────────────────────────────────────


def test_all_retries_fail():  # type: ignore[no-untyped-def]
    """TC-U-50：MAX_RETRIES 次全部失敗 → 拋出 LLMSamplingError。"""
    sampler = _sampler_no_rules(["bad1", "bad2", "bad3"])

    with patch("ifl_mcdc.layer3.llm_sampler.time.sleep"):
        with pytest.raises(LLMSamplingError):
            sampler.sample("test prompt")

    assert len(sampler.token_log) == LLMSampler.MAX_RETRIES, (
        f"應有 {LLMSampler.MAX_RETRIES} 次嘗試記錄"
    )


# ─────────────────────────────────────────────
# TC-U-51：token_log 記錄每次嘗試
# ─────────────────────────────────────────────


def test_token_log_records_attempts():  # type: ignore[no-untyped-def]
    """TC-U-51：token_log 應記錄 attempt、elapsed、est_tokens 三個欄位。"""
    payload = {"age": 40}
    sampler = _sampler_no_rules([json.dumps(payload)])
    sampler.sample("test prompt")

    assert len(sampler.token_log) == 1
    entry = sampler.token_log[0]
    assert "attempt" in entry, "token_log 應含 attempt 欄位"
    assert "elapsed" in entry, "token_log 應含 elapsed 欄位"
    assert "est_tokens" in entry, "token_log 應含 est_tokens 欄位"
    assert entry["attempt"] == 1


# ─────────────────────────────────────────────
# TC-U-51：指數退避
# ─────────────────────────────────────────────


def test_exponential_backoff():  # type: ignore[no-untyped-def]
    """TC-U-51：第 2 次 sleep(RETRY_DELAY*1)，第 3 次 sleep(RETRY_DELAY*2)。"""
    good = {"age": 30}
    # 第 1 次壞、第 2 次壞、第 3 次好
    sampler = _sampler_no_rules(["bad", "bad", json.dumps(good)])

    sleep_calls: list[float] = []
    with patch(
        "ifl_mcdc.layer3.llm_sampler.time.sleep",
        side_effect=lambda s: sleep_calls.append(s),
    ):
        result_data, _ = sampler.sample("test prompt")

    assert result_data == good
    # 第 2 次（attempt=2）sleep RETRY_DELAY*1；第 3 次（attempt=3）sleep RETRY_DELAY*2
    assert len(sleep_calls) == 2, f"應有 2 次 sleep 呼叫，實際 {sleep_calls}"
    assert sleep_calls[0] == sampler.retry_delay * 1, (
        f"第 2 次 sleep 應為 {sampler.retry_delay}，實際 {sleep_calls[0]}"
    )
    assert sleep_calls[1] == sampler.retry_delay * 2, (
        f"第 3 次 sleep 應為 {sampler.retry_delay * 2}，實際 {sleep_calls[1]}"
    )
