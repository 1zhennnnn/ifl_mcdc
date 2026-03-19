"""
Layer 2 GapAnalyzer 單元測試。

TC-U-24: 缺口清單正確識別（k=4，覆蓋 2 個 → 缺口 6 個）
TC-U-25: 難度排序正確（升序）
TC-U-26: L(X)=0 時回傳空清單
TC-U-27: 難度估計公式驗證
"""
from __future__ import annotations

from ifl_mcdc.layer1.ast_parser import ASTParser
from ifl_mcdc.layer1.coverage_engine import MCDCCoverageEngine
from ifl_mcdc.layer2.gap_analyzer import GapAnalyzer
from ifl_mcdc.models.coverage_matrix import MCDCMatrix
from ifl_mcdc.models.decision_node import AtomicCondition, ConditionSet


def _make_cond_set(k: int, expr_str: str = "(c1 or c2) and c3 and c4") -> ConditionSet:
    """建立 k=4 的 ConditionSet，耦合關係來自 ast_parser。"""
    parser = ASTParser()
    nodes = parser.parse_source(f"if {expr_str}: pass\n")
    assert len(nodes) >= 1
    return nodes[0].condition_set


def _make_matrix(cond_set: ConditionSet) -> MCDCMatrix:
    return MCDCMatrix(condition_set=cond_set)


# ─────────────────────────────────────────────
# TC-U-24：缺口清單正確識別
# ─────────────────────────────────────────────


def test_gap_list_count():  # type: ignore[no-untyped-def]
    """TC-U-24：k=4，覆蓋 2 個翻轉對 → 缺口 6 個。"""
    cond_set = _make_cond_set(4)
    matrix = _make_matrix(cond_set)

    # 標記 2 個翻轉對已覆蓋
    cond_ids = [c.cond_id for c in cond_set.conditions]
    matrix.mark_covered(cond_ids[0], "F2T")
    matrix.mark_covered(cond_ids[0], "T2F")

    analyzer = GapAnalyzer()
    gaps = analyzer.analyze(matrix)

    # k=4 共 8 個翻轉對，覆蓋 2 個 → 缺口 6 個
    assert len(gaps) == 6, f"應有 6 個缺口，實際 {len(gaps)}"


# ─────────────────────────────────────────────
# TC-U-25：難度排序正確（升序）
# ─────────────────────────────────────────────


def test_difficulty_ordering():  # type: ignore[no-untyped-def]
    """TC-U-25：GapAnalyzer 回傳缺口應按 estimated_difficulty 升序排列。"""
    cond_set = _make_cond_set(4)
    matrix = _make_matrix(cond_set)

    analyzer = GapAnalyzer()
    gaps = analyzer.analyze(matrix)

    difficulties = [g.estimated_difficulty for g in gaps]
    assert difficulties == sorted(difficulties), (
        f"缺口應按難度升序排列，實際 {difficulties}"
    )


# ─────────────────────────────────────────────
# TC-U-26：L(X)=0 時回傳空清單
# ─────────────────────────────────────────────


def test_empty_when_full_coverage():  # type: ignore[no-untyped-def]
    """TC-U-26：所有翻轉對已覆蓋時，analyze 回傳空列表。"""
    cond_set = _make_cond_set(4)
    matrix = _make_matrix(cond_set)

    # 覆蓋全部 2*k 個翻轉對
    for cond in cond_set.conditions:
        matrix.mark_covered(cond.cond_id, "F2T")
        matrix.mark_covered(cond.cond_id, "T2F")

    analyzer = GapAnalyzer()
    gaps = analyzer.analyze(matrix)

    assert gaps == [], f"100% 覆蓋時應回傳空列表，實際 {gaps}"


# ─────────────────────────────────────────────
# TC-U-27：難度估計公式驗證
# ─────────────────────────────────────────────


def test_difficulty_formula():  # type: ignore[no-untyped-def]
    """TC-U-27：難度 = 耦合邊數量 / (k-1)，驗證公式正確性。

    對於 (c1 or c2) and c3 and c4：
    - c1 耦合：c2(OR), c3(AND), c4(AND) → 3 個耦合邊 → difficulty = 3/(4-1) = 1.0
    - c3 耦合：c1(AND), c2(AND), c4(AND) → 3 個耦合邊 → difficulty = 3/(4-1) = 1.0
    """
    cond_set = _make_cond_set(4)
    matrix = _make_matrix(cond_set)

    analyzer = GapAnalyzer()

    # 驗證每個條件的難度公式
    for cond in cond_set.conditions:
        coupled = cond_set.get_coupled(cond.cond_id)
        expected = len(coupled) / (cond_set.k - 1)
        actual = analyzer._estimate_difficulty(cond_set, cond.cond_id)
        assert abs(actual - expected) < 1e-9, (
            f"cond {cond.cond_id}: 難度應為 {expected:.4f}，實際 {actual:.4f}"
        )


def test_difficulty_formula_k1():  # type: ignore[no-untyped-def]
    """TC-U-27 補充：k=1 時難度為 0.0（避免除以零）。"""
    parser = ASTParser()
    nodes = parser.parse_source("if a: pass\n")
    cond_set = nodes[0].condition_set
    assert cond_set.k == 1

    analyzer = GapAnalyzer()
    difficulty = analyzer._estimate_difficulty(cond_set, cond_set.conditions[0].cond_id)
    assert difficulty == 0.0, f"k=1 時難度應為 0.0，實際 {difficulty}"
