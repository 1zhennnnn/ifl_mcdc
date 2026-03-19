"""
Layer 1 MCDCCoverageEngine 單元測試。

TC-U-14: 損失函數初始值計算（loss == 8 for k=4）
TC-U-15: 有效獨立對被正確識別，loss 下降
TC-U-16: 遮罩條件不被誤計為獨立對
TC-U-17: 增量更新 L(X) 遞減
TC-U-18: 100% 覆蓋後 L(X)=0，coverage_ratio=1.0
"""
from __future__ import annotations

import pytest

from ifl_mcdc.layer1.ast_parser import ASTParser
from ifl_mcdc.layer1.coverage_engine import MCDCCoverageEngine
from ifl_mcdc.models.coverage_matrix import MCDCMatrix
from ifl_mcdc.models.decision_node import ConditionSet
from ifl_mcdc.models.probe_record import ProbeLog, ProbeRecord


# ─────────────────────────────────────────────
# 輔助函式：建立疫苗邏輯的 ConditionSet
# ─────────────────────────────────────────────

VACCINE_EXPR = (
    "(age >= 65 or (age >= 18 and high_risk)) and (days_since_last > 180) and not egg_allergy"
)


def _vaccine_cond_set() -> ConditionSet:
    """解析疫苗邏輯，回傳 ConditionSet（k=4 或更多，視 decompose 結果）。"""
    code = f"if {VACCINE_EXPR}: pass\n"
    parser = ASTParser()
    nodes = parser.parse_source(code)
    return nodes[0].condition_set


def _make_log(entries: list[dict]) -> ProbeLog:
    """建立 ProbeLog，entries 為 dict(test_id, cond_id, value, decision)。"""
    log = ProbeLog()
    for e in entries:
        log.append(ProbeRecord(
            test_id=e["test_id"],
            cond_id=e["cond_id"],
            value=e["value"],
            decision=e["decision"],
        ))
    return log


def _make_simple_cond_set() -> ConditionSet:
    """建立簡單的 k=4 疫苗 ConditionSet，使用固定 4 條件的邏輯。

    使用 (c1 or c2) and c3 and c4 使矩陣結構固定。
    """
    code = "if (a or b) and c and d: pass\n"
    parser = ASTParser()
    nodes = parser.parse_source(code)
    return nodes[0].condition_set


# ─────────────────────────────────────────────
# TC-U-14：損失函數初始值計算
# ─────────────────────────────────────────────


def test_initial_loss_is_2k():  # type: ignore[no-untyped-def]
    """TC-U-14：空 ProbeLog，k=4，loss == 8，coverage_ratio == 0.0。"""
    cond_set = _make_simple_cond_set()
    assert cond_set.k == 4

    engine = MCDCCoverageEngine()
    empty_log = ProbeLog()
    matrix = engine.build_matrix(cond_set, empty_log)

    assert matrix.compute_loss() == 8, f"預期 loss=8，得到 {matrix.compute_loss()}"
    assert matrix.coverage_ratio == 0.0, f"預期 coverage_ratio=0.0，得到 {matrix.coverage_ratio}"


# ─────────────────────────────────────────────
# TC-U-15：有效獨立對被正確識別
# ─────────────────────────────────────────────


def test_valid_independence_pair():  # type: ignore[no-untyped-def]
    """TC-U-15：T1(c1=F,c2=F,c3=T,c4=T,D=F), T2(c1=T,c2=F,c3=T,c4=T,D=T)。

    c1(OR 耦合夥伴 c2=F) 翻轉，決策翻轉 → 有效獨立對。
    loss 從 8 降至 6。
    """
    cond_set = _make_simple_cond_set()
    # 條件：c1=(a or b) and c and d → k=4: D1.c1=a, D1.c2=b, D1.c3=c, D1.c4=d
    cids = [c.cond_id for c in cond_set.conditions]
    c1, c2, c3, c4 = cids[0], cids[1], cids[2], cids[3]

    # T1: c1=F, c2=F, c3=T, c4=T, decision=F
    # T2: c1=T, c2=F, c3=T, c4=T, decision=T
    log = _make_log([
        {"test_id": "T1", "cond_id": c1, "value": False, "decision": False},
        {"test_id": "T1", "cond_id": c2, "value": False, "decision": False},
        {"test_id": "T1", "cond_id": c3, "value": True,  "decision": False},
        {"test_id": "T1", "cond_id": c4, "value": True,  "decision": False},
        {"test_id": "T2", "cond_id": c1, "value": True,  "decision": True},
        {"test_id": "T2", "cond_id": c2, "value": False, "decision": True},
        {"test_id": "T2", "cond_id": c3, "value": True,  "decision": True},
        {"test_id": "T2", "cond_id": c4, "value": True,  "decision": True},
    ])

    engine = MCDCCoverageEngine()
    matrix = engine.build_matrix(cond_set, log)

    assert (c1, "F2T") in matrix._covered, f"(c1, F2T) 應在 _covered 中"
    assert matrix.compute_loss() == 6, f"預期 loss=6，得到 {matrix.compute_loss()}"


# ─────────────────────────────────────────────
# TC-U-16：遮罩條件不被誤計為獨立對
# ─────────────────────────────────────────────


def test_masking_not_counted():  # type: ignore[no-untyped-def]
    """TC-U-16：c2=True（OR 夥伴）遮罩 c1，不應計為有效獨立對。

    T1(c1=F,c2=T,c3=T,c4=T,D=T), T2(c1=T,c2=T,c3=T,c4=T,D=T)
    c2=True 在兩筆均為 T，且 c1/c2 為 OR 耦合 → OR 夥伴未為 False → 遮罩。
    """
    cond_set = _make_simple_cond_set()
    cids = [c.cond_id for c in cond_set.conditions]
    c1, c2, c3, c4 = cids[0], cids[1], cids[2], cids[3]

    log = _make_log([
        {"test_id": "T1", "cond_id": c1, "value": False, "decision": True},
        {"test_id": "T1", "cond_id": c2, "value": True,  "decision": True},
        {"test_id": "T1", "cond_id": c3, "value": True,  "decision": True},
        {"test_id": "T1", "cond_id": c4, "value": True,  "decision": True},
        {"test_id": "T2", "cond_id": c1, "value": True,  "decision": True},
        {"test_id": "T2", "cond_id": c2, "value": True,  "decision": True},
        {"test_id": "T2", "cond_id": c3, "value": True,  "decision": True},
        {"test_id": "T2", "cond_id": c4, "value": True,  "decision": True},
    ])

    engine = MCDCCoverageEngine()
    matrix = engine.build_matrix(cond_set, log)

    assert (c1, "F2T") not in matrix._covered, (
        f"c2=True（OR 遮罩）時，(c1, F2T) 不應在 _covered 中"
    )


# ─────────────────────────────────────────────
# TC-U-17：增量更新 L(X) 遞減
# ─────────────────────────────────────────────


def test_update_returns_true():  # type: ignore[no-untyped-def]
    """TC-U-17：新增有效貢獻的測試案例後，update() 回傳 True，L(X) 下降。"""
    cond_set = _make_simple_cond_set()
    cids = [c.cond_id for c in cond_set.conditions]
    c1, c2, c3, c4 = cids[0], cids[1], cids[2], cids[3]

    # 初始：T1 單筆，無法構成獨立對
    log = _make_log([
        {"test_id": "T1", "cond_id": c1, "value": False, "decision": False},
        {"test_id": "T1", "cond_id": c2, "value": False, "decision": False},
        {"test_id": "T1", "cond_id": c3, "value": True,  "decision": False},
        {"test_id": "T1", "cond_id": c4, "value": True,  "decision": False},
    ])

    engine = MCDCCoverageEngine()
    matrix = engine.build_matrix(cond_set, log)
    loss_before = matrix.compute_loss()
    assert loss_before == 8

    # 加入 T2（與 T1 構成有效獨立對）
    for entry in [
        {"test_id": "T2", "cond_id": c1, "value": True,  "decision": True},
        {"test_id": "T2", "cond_id": c2, "value": False, "decision": True},
        {"test_id": "T2", "cond_id": c3, "value": True,  "decision": True},
        {"test_id": "T2", "cond_id": c4, "value": True,  "decision": True},
    ]:
        log.append(ProbeRecord(**entry))  # type: ignore[arg-type]

    improved = engine.update(matrix, log, "T2")
    assert improved is True, "update() 應回傳 True"
    assert matrix.compute_loss() < loss_before, "L(X) 應下降"


# ─────────────────────────────────────────────
# TC-U-18：100% 覆蓋後 L(X)=0
# ─────────────────────────────────────────────


def test_full_coverage_loss_zero():  # type: ignore[no-untyped-def]
    """TC-U-18：構造滿足 k=4 所有 8 個獨立對的完整測試集。

    邏輯：(a or b) and c and d
    MC/DC 需要每個條件各一個 F2T 和 T2F 獨立對。
    """
    cond_set = _make_simple_cond_set()
    cids = [c.cond_id for c in cond_set.conditions]
    c1, c2, c3, c4 = cids[0], cids[1], cids[2], cids[3]

    # 構造覆蓋所有條件翻轉的測試集
    # 邏輯：(a or b) and c and d
    # c1(a) 獨立對：c2=F，c3=T，c4=T，a 從 F→T 使決策 F→T
    # c2(b) 獨立對：c1=F，c3=T，c4=T，b 從 F→T 使決策 F→T
    # c3(c) 獨立對：(a=T or b=F)=T，c4=T，c 從 F→T 使決策 F→T
    # c4(d) 獨立對：(a=T or b=F)=T，c3=T，d 從 F→T 使決策 F→T
    entries = [
        # c1 F2T pair: T1(c1=F,c2=F,c3=T,c4=T,D=F) vs T2(c1=T,c2=F,c3=T,c4=T,D=T)
        {"test_id": "T1", "cond_id": c1, "value": False, "decision": False},
        {"test_id": "T1", "cond_id": c2, "value": False, "decision": False},
        {"test_id": "T1", "cond_id": c3, "value": True,  "decision": False},
        {"test_id": "T1", "cond_id": c4, "value": True,  "decision": False},
        {"test_id": "T2", "cond_id": c1, "value": True,  "decision": True},
        {"test_id": "T2", "cond_id": c2, "value": False, "decision": True},
        {"test_id": "T2", "cond_id": c3, "value": True,  "decision": True},
        {"test_id": "T2", "cond_id": c4, "value": True,  "decision": True},
        # c2 F2T pair: T3(c1=F,c2=F,c3=T,c4=T,D=F) vs T4(c1=F,c2=T,c3=T,c4=T,D=T)
        {"test_id": "T3", "cond_id": c1, "value": False, "decision": False},
        {"test_id": "T3", "cond_id": c2, "value": False, "decision": False},
        {"test_id": "T3", "cond_id": c3, "value": True,  "decision": False},
        {"test_id": "T3", "cond_id": c4, "value": True,  "decision": False},
        {"test_id": "T4", "cond_id": c1, "value": False, "decision": True},
        {"test_id": "T4", "cond_id": c2, "value": True,  "decision": True},
        {"test_id": "T4", "cond_id": c3, "value": True,  "decision": True},
        {"test_id": "T4", "cond_id": c4, "value": True,  "decision": True},
        # c3 F2T pair: T5(c1=T,c2=T,c3=F,c4=T,D=F) vs T6(c1=T,c2=T,c3=T,c4=T,D=T)
        # c1,c2,c4 全為 AND 耦合，需兩個測試中都為 True
        {"test_id": "T5", "cond_id": c1, "value": True,  "decision": False},
        {"test_id": "T5", "cond_id": c2, "value": True,  "decision": False},
        {"test_id": "T5", "cond_id": c3, "value": False, "decision": False},
        {"test_id": "T5", "cond_id": c4, "value": True,  "decision": False},
        {"test_id": "T6", "cond_id": c1, "value": True,  "decision": True},
        {"test_id": "T6", "cond_id": c2, "value": True,  "decision": True},
        {"test_id": "T6", "cond_id": c3, "value": True,  "decision": True},
        {"test_id": "T6", "cond_id": c4, "value": True,  "decision": True},
        # c4 F2T pair: T7(c1=T,c2=T,c3=T,c4=F,D=F) vs T8(c1=T,c2=T,c3=T,c4=T,D=T)
        # c1,c2,c3 全為 AND 耦合，需兩個測試中都為 True
        {"test_id": "T7", "cond_id": c1, "value": True,  "decision": False},
        {"test_id": "T7", "cond_id": c2, "value": True,  "decision": False},
        {"test_id": "T7", "cond_id": c3, "value": True,  "decision": False},
        {"test_id": "T7", "cond_id": c4, "value": False, "decision": False},
        {"test_id": "T8", "cond_id": c1, "value": True,  "decision": True},
        {"test_id": "T8", "cond_id": c2, "value": True,  "decision": True},
        {"test_id": "T8", "cond_id": c3, "value": True,  "decision": True},
        {"test_id": "T8", "cond_id": c4, "value": True,  "decision": True},
    ]

    log = _make_log(entries)
    engine = MCDCCoverageEngine()
    matrix = engine.build_matrix(cond_set, log)

    assert matrix.compute_loss() == 0, (
        f"預期 loss=0，得到 {matrix.compute_loss()}。\n"
        f"_covered: {matrix._covered}"
    )
    assert matrix.coverage_ratio == 1.0, (
        f"預期 coverage_ratio=1.0，得到 {matrix.coverage_ratio}"
    )
