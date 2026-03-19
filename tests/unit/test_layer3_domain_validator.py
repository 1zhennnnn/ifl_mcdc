"""
Layer 3 DomainValidator 單元測試。

TC-U-41: 拒絕負數年齡
TC-U-42: 拒絕超過 130 的年齡
TC-U-43: 邊界值 0 和 130 通過
TC-U-44: 拒絕負數距上次接種天數
TC-U-45: 拒絕非布林型高風險/過敏標記
TC-U-46: 無效 JSON 時 passed=False
"""
from __future__ import annotations

import json

from ifl_mcdc.layer3.domain_validator import DomainValidator


def _make_validator() -> DomainValidator:
    return DomainValidator()


def _json(**kwargs: object) -> str:
    return json.dumps(kwargs)


# ─────────────────────────────────────────────
# TC-U-41：拒絕負數年齡
# ─────────────────────────────────────────────


def test_reject_negative_age():  # type: ignore[no-untyped-def]
    """TC-U-41：age=-1 應被拒絕。"""
    validator = _make_validator()
    result = validator.validate(_json(age=-1, egg_allergy=False))

    assert result.passed is False
    fields = [v.field for v in result.violations]
    assert "age" in fields, f"violations 應包含 age，實際：{fields}"


# ─────────────────────────────────────────────
# TC-U-42：拒絕超過 130 的年齡
# ─────────────────────────────────────────────


def test_reject_over_age():  # type: ignore[no-untyped-def]
    """TC-U-42：age=131 應被拒絕。"""
    validator = _make_validator()
    result = validator.validate(_json(age=131, egg_allergy=False))

    assert result.passed is False
    fields = [v.field for v in result.violations]
    assert "age" in fields


# ─────────────────────────────────────────────
# TC-U-43：邊界值 0 和 130 通過
# ─────────────────────────────────────────────


def test_accept_boundary_age_0():  # type: ignore[no-untyped-def]
    """TC-U-43：age=0 應通過驗證。"""
    validator = _make_validator()
    result = validator.validate(_json(age=0, egg_allergy=False))

    age_violations = [v for v in result.violations if v.field == "age"]
    assert len(age_violations) == 0, f"age=0 不應有違規：{age_violations}"


def test_accept_boundary_age_130():  # type: ignore[no-untyped-def]
    """TC-U-43：age=130 應通過驗證。"""
    validator = _make_validator()
    result = validator.validate(_json(age=130, egg_allergy=False))

    age_violations = [v for v in result.violations if v.field == "age"]
    assert len(age_violations) == 0, f"age=130 不應有違規：{age_violations}"


# ─────────────────────────────────────────────
# TC-U-44：拒絕負數距上次接種天數
# ─────────────────────────────────────────────


def test_reject_negative_days():  # type: ignore[no-untyped-def]
    """TC-U-44：days_since_last=-1 應被拒絕。"""
    validator = _make_validator()
    result = validator.validate(_json(age=30, days_since_last=-1, egg_allergy=False))

    assert result.passed is False
    fields = [v.field for v in result.violations]
    assert "days_since_last" in fields


# ─────────────────────────────────────────────
# TC-U-45：拒絕非布林型高風險/過敏標記
# ─────────────────────────────────────────────


def test_reject_non_bool():  # type: ignore[no-untyped-def]
    """TC-U-45：high_risk="yes" 和 egg_allergy=1 應被拒絕。"""
    validator = _make_validator()

    # high_risk 為字串
    result1 = validator.validate(_json(age=30, high_risk="yes", egg_allergy=False))
    assert result1.passed is False
    fields1 = [v.field for v in result1.violations]
    assert "high_risk" in fields1, f"high_risk='yes' 應有違規，實際：{fields1}"

    # egg_allergy 為整數（1 不是 bool）
    result2 = validator.validate(_json(age=30, high_risk=True, egg_allergy=1))
    assert result2.passed is False
    fields2 = [v.field for v in result2.violations]
    assert "egg_allergy" in fields2, f"egg_allergy=1 應有違規，實際：{fields2}"


# ─────────────────────────────────────────────
# TC-U-46：無效 JSON 時 passed=False
# ─────────────────────────────────────────────


def test_invalid_json():  # type: ignore[no-untyped-def]
    """TC-U-46：無效 JSON 字串應回傳 passed=False。"""
    validator = _make_validator()

    result = validator.validate("not a json string {{{")

    assert result.passed is False, "無效 JSON 應 passed=False"
    assert len(result.violations) > 0, "應有至少一個違規記錄"
