"""
models/ 單元測試。

TC-U-01: AtomicCondition.evaluate 基本測試
TC-U-02: ConditionSet.get_coupled OR 型耦合測試
TC-U-03: MCDCMatrix.compute_loss 初始值測試
TC-U-04: MCDCMatrix.coverage_ratio 計算測試
TC-U-05: ProbeLog 執行緒安全測試
TC-U-06: BoundSpec.to_prompt_str 輸出格式測試
TC-U-07: ValidationResult.to_corrective_prompt 格式測試
"""
from __future__ import annotations

import threading

import pytest

from ifl_mcdc.models.coverage_matrix import MCDCMatrix
from ifl_mcdc.models.decision_node import AtomicCondition, ConditionSet
from ifl_mcdc.models.probe_record import ProbeLog, ProbeRecord
from ifl_mcdc.models.smt_models import BoundSpec
from ifl_mcdc.models.validation import ValidationResult, Violation


# ─── TC-U-01 ───────────────────────────────────────────────────────────────


def test_atomic_condition_evaluate_basic() -> None:
    """TC-U-01: evaluate 對 age>=65 的 True / False 路徑。"""
    cond = AtomicCondition(
        cond_id="D1.c1",
        expression="age >= 65",
        var_names=["age"],
    )
    assert cond.evaluate({"age": 70}) is True
    assert cond.evaluate({"age": 60}) is False


# ─── TC-U-02 ───────────────────────────────────────────────────────────────


def test_condition_set_get_coupled_or() -> None:
    """TC-U-02: k=2，matrix[0][1]='OR' 時 get_coupled('D1.c1') 含 ('D1.c2','OR')。"""
    c1 = AtomicCondition(cond_id="D1.c1", expression="age >= 65", var_names=["age"])
    c2 = AtomicCondition(
        cond_id="D1.c2", expression="age >= 18 and high_risk", var_names=["age", "high_risk"]
    )
    cset = ConditionSet(
        decision_id="D1",
        conditions=[c1, c2],
        coupling_matrix=[
            [None, "OR"],
            ["OR", None],
        ],
    )
    coupled = cset.get_coupled("D1.c1")
    cond_ids_and_types = [(c.cond_id, t) for c, t in coupled]
    assert ("D1.c2", "OR") in cond_ids_and_types


def test_condition_set_get_coupled_unknown_raises() -> None:
    """get_coupled 傳入不存在的 cond_id 應拋出 KeyError。"""
    c1 = AtomicCondition(cond_id="D1.c1", expression="x > 0", var_names=["x"])
    cset = ConditionSet(
        decision_id="D1",
        conditions=[c1],
        coupling_matrix=[[None]],
    )
    with pytest.raises(KeyError):
        cset.get_coupled("D1.c99")


# ─── TC-U-03 ───────────────────────────────────────────────────────────────


def _make_k4_matrix() -> MCDCMatrix:
    """建立 k=4 的 MCDCMatrix 輔助函式。"""
    conditions = [
        AtomicCondition(cond_id=f"D1.c{i}", expression=f"x{i}", var_names=[f"x{i}"])
        for i in range(1, 5)
    ]
    matrix: list[list[str | None]] = [
        [None, "OR", "AND", "AND"],
        ["OR", None, "AND", "AND"],
        ["AND", "AND", None, "AND"],
        ["AND", "AND", "AND", None],
    ]
    cset = ConditionSet(decision_id="D1", conditions=conditions, coupling_matrix=matrix)
    return MCDCMatrix(condition_set=cset)


def test_mcdc_matrix_compute_loss_initial() -> None:
    """TC-U-03: 空 _covered，k=4 → compute_loss() == 8。"""
    mat = _make_k4_matrix()
    assert mat.compute_loss() == 8


# ─── TC-U-04 ───────────────────────────────────────────────────────────────


def test_mcdc_matrix_coverage_ratio() -> None:
    """TC-U-04: _covered 含 2 個翻轉對，k=4 → coverage_ratio == 0.25。"""
    mat = _make_k4_matrix()
    mat.mark_covered("D1.c1", "F2T")
    mat.mark_covered("D1.c2", "T2F")
    assert mat.coverage_ratio == pytest.approx(0.25)


def test_mcdc_matrix_mark_covered_invalid_direction() -> None:
    """mark_covered 傳入非法 flip_direction 應拋出 ValueError。"""
    mat = _make_k4_matrix()
    with pytest.raises(ValueError, match="flip_direction"):
        mat.mark_covered("D1.c1", "INVALID")


def test_mcdc_matrix_get_gap_list_full() -> None:
    """初始時 get_gap_list 應回傳 2*k=8 個 GapEntry。"""
    mat = _make_k4_matrix()
    gaps = mat.get_gap_list()
    assert len(gaps) == 8


def test_mcdc_matrix_coverage_ratio_k0() -> None:
    """k=0 時 coverage_ratio 應回傳 1.0。"""
    cset = ConditionSet(decision_id="D0", conditions=[], coupling_matrix=[])
    mat = MCDCMatrix(condition_set=cset)
    assert mat.coverage_ratio == 1.0


# ─── TC-U-05 ───────────────────────────────────────────────────────────────


def test_probe_log_thread_safety() -> None:
    """TC-U-05: 10 個執行緒各 append 100 筆 → len(records) == 1000。"""
    log = ProbeLog()

    def worker(tid: int) -> None:
        for i in range(100):
            log.append(
                ProbeRecord(
                    test_id=f"T{tid}",
                    cond_id="D1.c1",
                    value=True,
                    decision=True,
                )
            )

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(log.records) == 1000


# ─── TC-U-06 ───────────────────────────────────────────────────────────────


def test_bound_spec_to_prompt_str() -> None:
    """TC-U-06: interval=(18,64)，medical_unit='years' → 輸出含 '18' 和 '64'。"""
    spec = BoundSpec(
        var_name="age",
        var_type="int",
        interval=(18.0, 64.0),
        valid_set=None,
        medical_unit="years",
    )
    text = spec.to_prompt_str()
    assert "18" in text
    assert "64" in text
    assert "years" in text


# ─── TC-U-07 ───────────────────────────────────────────────────────────────


def test_validation_result_to_corrective_prompt() -> None:
    """TC-U-07: 含 2 個 Violation → 輸出字串包含兩個欄位名。"""
    result = ValidationResult(
        passed=False,
        violations=[
            Violation(field="age", description="年齡超出範圍", actual_value="-5"),
            Violation(field="egg_allergy", description="必須為布林值", actual_value="none"),
        ],
    )
    prompt = result.to_corrective_prompt()
    assert "age" in prompt
    assert "egg_allergy" in prompt
    assert prompt.count("\n") == 1  # 2 行以 1 個換行分隔
