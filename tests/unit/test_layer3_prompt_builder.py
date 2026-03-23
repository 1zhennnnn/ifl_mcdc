"""
Layer 3 PromptConstructor 單元測試。

TC-U-37: bound_specs 全部出現在 §3
TC-U-38: 超過 MAX_TOKENS 時截斷 §1 的 source_context
TC-U-39: §3 和 §4 截斷後完整保留
TC-U-40: F2T/T2F 方向描述正確
"""
from __future__ import annotations

from ifl_mcdc.layer1.ast_parser import ASTParser
from ifl_mcdc.layer3.prompt_builder import PromptConstructor
from ifl_mcdc.models.coverage_matrix import GapEntry
from ifl_mcdc.models.smt_models import BoundSpec


def _make_gap(cond_id: str, direction: str) -> GapEntry:
    return GapEntry(
        condition_id=cond_id,
        flip_direction=direction,
        missing_pair_type="unique_cause",
        estimated_difficulty=0.5,
    )


def _make_bound_specs() -> list[BoundSpec]:
    return [
        BoundSpec(var_name="age", var_type="int", interval=(18.0, 38.0), valid_set=None),
        BoundSpec(var_name="egg_allergy", var_type="bool", interval=None, valid_set=frozenset({False})),
    ]


def _parse_first(expr: str):  # type: ignore[no-untyped-def]
    parser = ASTParser()
    nodes = parser.parse_source(f"if {expr}: pass\n")
    return nodes[0]


# ─────────────────────────────────────────────
# TC-U-37：bound_specs 全部出現在 §3
# ─────────────────────────────────────────────


def test_contains_all_bound_specs():  # type: ignore[no-untyped-def]
    """TC-U-37：每個 BoundSpec 的變數名稱都出現在 §3 提示詞中。"""
    dn = _parse_first("age >= 18 and not egg_allergy")
    bound_specs = _make_bound_specs()
    gap = _make_gap(dn.condition_set.conditions[0].cond_id, "F2T")

    builder = PromptConstructor()
    prompt = builder.build(
        decision_node=dn,
        gap=gap,
        bound_specs=bound_specs,
        func_signature="def check(age: int, egg_allergy: bool) -> bool",
        domain_context="疫苗接種資格",
    )

    assert "【約束條件】" in prompt, "§3 標題應為【約束條件】"
    for bs in bound_specs:
        assert bs.var_name in prompt, f"BoundSpec {bs.var_name} 的變數名稱應出現在提示詞中"
    # int 型應有 「必須在...到...之間」描述
    assert "18" in prompt and "38" in prompt, "age 的範圍邊界應出現在提示詞中"
    # bool 型應有具體值描述
    assert "False" in prompt, "egg_allergy 的約束值 False 應出現在提示詞中"


# ─────────────────────────────────────────────
# TC-U-38：超過 MAX_TOKENS 時截斷 §1 的 source_context
# ─────────────────────────────────────────────


def test_token_limit():  # type: ignore[no-untyped-def]
    """TC-U-38：超長 source_context 被截斷，輸出不超過 MAX_TOKENS。"""
    dn = _parse_first("age >= 18 and not egg_allergy")
    # 注入超長 source_context
    dn.source_context = "x" * 3000  # 遠超過 MAX_TOKENS=2048

    bound_specs = _make_bound_specs()
    gap = _make_gap(dn.condition_set.conditions[0].cond_id, "F2T")

    builder = PromptConstructor()
    prompt = builder.build(
        decision_node=dn,
        gap=gap,
        bound_specs=bound_specs,
        func_signature="def check(age: int, egg_allergy: bool) -> bool",
        domain_context="疫苗接種",
    )

    assert len(prompt) <= PromptConstructor.MAX_TOKENS, (
        f"輸出長度 {len(prompt)} 應 ≤ MAX_TOKENS={PromptConstructor.MAX_TOKENS}"
    )


# ─────────────────────────────────────────────
# TC-U-39：§3 和 §4 截斷後完整保留
# ─────────────────────────────────────────────


def test_section_preservation():  # type: ignore[no-untyped-def]
    """TC-U-39：即使截斷，§3（臨床情境引導）和 §4（輸出格式）必須完整保留。"""
    dn = _parse_first("age >= 18 and not egg_allergy")
    dn.source_context = "y" * 3000  # 超長

    bound_specs = _make_bound_specs()
    gap = _make_gap(dn.condition_set.conditions[0].cond_id, "F2T")

    builder = PromptConstructor()
    prompt = builder.build(
        decision_node=dn,
        gap=gap,
        bound_specs=bound_specs,
        func_signature="def check(age: int, egg_allergy: bool) -> bool",
        domain_context="疫苗",
    )

    # §3 標題必須存在
    assert "【約束條件】" in prompt, "§3 標題必須保留"
    # §4 標題必須存在
    assert "【輸出格式】" in prompt, "§4 標題必須保留"
    # 每個 BoundSpec 的變數名稱必須存在
    for bs in bound_specs:
        assert bs.var_name in prompt, f"{bs.var_name} 的變數名稱必須出現在截斷後的提示詞中"


# ─────────────────────────────────────────────
# TC-U-40：F2T / T2F 方向描述正確
# ─────────────────────────────────────────────


def test_f2t_direction():  # type: ignore[no-untyped-def]
    """TC-U-40：F2T 缺口應在 §2 中描述值為 "True"。"""
    dn = _parse_first("age >= 18")
    gap = _make_gap(dn.condition_set.conditions[0].cond_id, "F2T")
    bound_specs = [BoundSpec(var_name="age", var_type="int", interval=(18.0, 28.0), valid_set=None)]

    builder = PromptConstructor()
    prompt = builder.build(
        decision_node=dn,
        gap=gap,
        bound_specs=bound_specs,
        func_signature="def check(age: int) -> bool",
    )

    assert "True" in prompt, "F2T 缺口應描述目標值為 True"
    # §2 區段必須存在
    assert "【目標缺口】" in prompt


def test_t2f_direction():  # type: ignore[no-untyped-def]
    """TC-U-40：T2F 缺口應在 §2 中描述值為 "False"。"""
    dn = _parse_first("age >= 18")
    gap = _make_gap(dn.condition_set.conditions[0].cond_id, "T2F")
    bound_specs = [BoundSpec(var_name="age", var_type="int", interval=(0.0, 17.0), valid_set=None)]

    builder = PromptConstructor()
    prompt = builder.build(
        decision_node=dn,
        gap=gap,
        bound_specs=bound_specs,
        func_signature="def check(age: int) -> bool",
    )

    assert "False" in prompt, "T2F 缺口應描述目標值為 False"
