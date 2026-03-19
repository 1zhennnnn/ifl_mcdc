"""
Layer 1 ASTParser 單元測試。

TC-U-01: 基本 if 節點識別
TC-U-02: 巢狀 if 的兩層識別
TC-U-03: IfExp 行內三元表達式識別
TC-U-04: 語法錯誤原始碼的錯誤處理
"""
from __future__ import annotations

import pytest

from ifl_mcdc.exceptions import ASTParseError
from ifl_mcdc.layer1.ast_parser import ASTParser


# ─────────────────────────────────────────────
# TC-U-01：基本 if 節點識別
# ─────────────────────────────────────────────


def test_parse_vaccine_logic(vaccine_source_path):  # type: ignore[no-untyped-def]
    """TC-U-01：疫苗邏輯只有一個頂層 if，k=4 個原子條件。"""
    parser = ASTParser()
    nodes = parser.parse_file(vaccine_source_path)

    assert len(nodes) == 1, f"預期 1 個 DecisionNode，得到 {len(nodes)}"
    dn = nodes[0]
    assert dn.node_type == "If"
    assert dn.node_id == "D1"
    assert dn.line_no > 0
    # 疫苗邏輯含巢狀 (age>=18 and high_risk)，分解後為 5 個原子條件：
    # age>=65, age>=18, high_risk, days_since_last>180, egg_allergy
    assert dn.condition_set.k == 5, f"預期 k=5，得到 {dn.condition_set.k}"

    cond_ids = [c.cond_id for c in dn.condition_set.conditions]
    assert cond_ids == ["D1.c1", "D1.c2", "D1.c3", "D1.c4", "D1.c5"]


# ─────────────────────────────────────────────
# TC-U-02：巢狀 if 的兩層識別
# ─────────────────────────────────────────────


def test_parse_nested_if(tmp_path):  # type: ignore[no-untyped-def]
    """TC-U-02：含兩層巢狀 if，應回傳 2 個 DecisionNode。"""
    code = "def f(x, y):\n    if x > 0:\n        if y > 0:\n            pass\n"
    src_file = tmp_path / "nested_if.py"
    src_file.write_text(code, encoding="utf-8")

    parser = ASTParser()
    nodes = parser.parse_file(src_file)

    assert len(nodes) == 2, f"預期 2 個 DecisionNode，得到 {len(nodes)}"
    assert nodes[0].line_no < nodes[1].line_no, "外層行號應小於內層"
    assert nodes[0].node_id == "D1"
    assert nodes[1].node_id == "D2"


def test_generic_visit_for_nested(tmp_path):  # type: ignore[no-untyped-def]
    """確認 generic_visit 正確處理巢狀 if（三層）。"""
    code = (
        "def f(a, b, c):\n"
        "    if a:\n"
        "        if b:\n"
        "            if c:\n"
        "                pass\n"
    )
    src_file = tmp_path / "triple_nested.py"
    src_file.write_text(code, encoding="utf-8")

    parser = ASTParser()
    nodes = parser.parse_file(src_file)

    assert len(nodes) == 3
    line_nos = [n.line_no for n in nodes]
    assert line_nos == sorted(line_nos), "行號應由小到大排列"


# ─────────────────────────────────────────────
# TC-U-03：IfExp 行內三元表達式識別
# ─────────────────────────────────────────────


def test_parse_if_exp():  # type: ignore[no-untyped-def]
    """TC-U-03：識別 IfExp（三元表達式）節點。"""
    code = "def f(x, y):\n    result = x if x > 0 else y\n"

    parser = ASTParser()
    nodes = parser.parse_source(code)

    assert any(n.node_type == "IfExp" for n in nodes), (
        "應識別到 IfExp 節點"
    )


# ─────────────────────────────────────────────
# TC-U-04：語法錯誤原始碼的錯誤處理
# ─────────────────────────────────────────────


def test_parse_syntax_error(tmp_path):  # type: ignore[no-untyped-def]
    """TC-U-04：語法錯誤時應拋出 ASTParseError，訊息含行號資訊。"""
    bad_code = "if x >\n"
    src_file = tmp_path / "bad.py"
    src_file.write_text(bad_code, encoding="utf-8")

    parser = ASTParser()
    with pytest.raises(ASTParseError) as exc_info:
        parser.parse_file(src_file)

    msg = str(exc_info.value)
    # 訊息應包含行號資訊
    assert any(c.isdigit() for c in msg), f"錯誤訊息應含行號，得到：{msg!r}"


def test_parse_syntax_error_source():  # type: ignore[no-untyped-def]
    """語法錯誤時 parse_source 也應拋出 ASTParseError。"""
    parser = ASTParser()
    with pytest.raises(ASTParseError):
        parser.parse_source("def f(\n")
