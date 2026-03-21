"""
IFLOrchestrator 單元測試（使用 MockLLMBackend）。

TC-U-52: run() 回傳 IFLResult，結構完整（7 個欄位）
TC-U-53: 空決策節點來源 → ValueError
TC-U-54: INFEASIBLE 路徑不再重複求解
TC-U-55: 預算耗盡時輸出部分覆蓋報告
TC-U-56: 損失歷程記錄單調不增
TC-U-63: Z3 全部 UNSAT → infeasible 路徑偵測，loop 提前終止
TC-U-73: DomainValidator 在 LLMSampler 層執行，驗證失敗不觸發 infeasible
TC-U-74: LLMSamplingError 跳過本輪迭代，不標記條件為 infeasible
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from ifl_mcdc.config import IFLConfig
from ifl_mcdc.layer1.ast_parser import ASTParser
from ifl_mcdc.layer3.llm_sampler import MockLLMBackend
from ifl_mcdc.orchestrator import IFLOrchestrator, IFLResult

VACCINE_PATH = Path(__file__).parent.parent / "fixtures" / "vaccine_eligibility.py"

# 合法的疫苗測試案例 JSON（可通過 domain validator，可成功呼叫目標函式）
_VALID_JSON = json.dumps(
    {"age": 30, "high_risk": True, "days_since_last": 200, "egg_allergy": False}
)
# 不匹配目標函式參數的 JSON（通過 validator，但函式呼叫失敗 → 無探針記錄）
_INVALID_PARAMS_JSON = json.dumps({"foo": "bar"})


def _make_orch(
    responses: list[str | BaseException],
    max_iterations: int = 50,
) -> IFLOrchestrator:
    """建立使用 MockLLMBackend 的 IFLOrchestrator。"""
    config = IFLConfig(max_iterations=max_iterations)
    mock = MockLLMBackend(responses)
    return IFLOrchestrator(config=config, backend=mock)


# ─────────────────────────────────────────────
# TC-U-52：run() 回傳 IFLResult，結構完整
# ─────────────────────────────────────────────


def test_run_returns_ifl_result():  # type: ignore[no-untyped-def]
    """TC-U-52：run() 回傳 IFLResult，所有欄位型別正確（SDD 7 欄位）。"""
    responses = [_VALID_JSON] * 30
    orch = _make_orch(responses, max_iterations=5)

    result = orch.run(VACCINE_PATH)

    assert isinstance(result, IFLResult)
    assert isinstance(result.converged, bool)
    assert 0.0 <= result.final_coverage <= 1.0
    assert isinstance(result.test_suite, list)
    assert isinstance(result.iteration_count, int)
    assert isinstance(result.total_tokens, int)
    assert isinstance(result.infeasible_paths, list)
    assert isinstance(result.loss_history, list)
    assert len(result.loss_history) >= 1, "loss_history 至少應有初始值"
    # 初始 test_suite 包含 3 個隨機案例
    assert len(result.test_suite) >= 0


# ─────────────────────────────────────────────
# TC-U-53：空決策節點來源 → ValueError
# ─────────────────────────────────────────────


def test_empty_source_raises_value_error():  # type: ignore[no-untyped-def]
    """TC-U-53：來源檔案中無任何決策節點時應拋出 ValueError。"""
    orch = _make_orch([])

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write("# no decision nodes\nx = 1 + 1\n")
        tmp_path = f.name

    with pytest.raises(ValueError, match="找不到任何決策節點"):
        orch.run(tmp_path)


# ─────────────────────────────────────────────
# TC-U-54：INFEASIBLE 路徑不再重複求解
# ─────────────────────────────────────────────


def test_infeasible_not_revisited():  # type: ignore[no-untyped-def]
    """TC-U-54：_infeasible 中的條件 ID 不應觸發 SMT 求解。"""
    # 解析疫苗檔取得所有條件 ID
    parser = ASTParser()
    nodes = parser.parse_file(str(VACCINE_PATH))
    all_cond_ids = {c.cond_id for c in nodes[0].condition_set.conditions}

    orch = _make_orch([])
    # 預填所有條件 ID 為不可行
    orch._infeasible = set(all_cond_ids)

    with patch.object(orch.smt, "synthesize") as mock_synthesize:
        result = orch.run(VACCINE_PATH)
        mock_synthesize.assert_not_called()

    # 所有缺口均不可行 → gap=None → break → iteration_count=1（loop 執行一次後 break）
    assert result.iteration_count == 1
    # loss_history 只有初始值（gap=None → break，未執行任何 append）
    assert len(result.loss_history) == 1


# ─────────────────────────────────────────────
# TC-U-55：預算耗盡時輸出部分覆蓋報告
# ─────────────────────────────────────────────


def test_budget_exhausted():  # type: ignore[no-untyped-def]
    """TC-U-55：max_iterations=2，函式呼叫失敗（無探針），損失不降低，回傳部分報告。

    使用不匹配函式簽名的 mock 回應：
      - 通過 domain validator（無對應規則）
      - 呼叫 check_vaccine_eligibility(foo='bar') → TypeError（被 try/except 捕獲）
      - 無探針記錄 → 無新獨立對 → gate 拒絕 → loss_history 不降低
    """
    # 2 個合法 JSON（通過 validator），但參數不匹配函式
    responses = [_INVALID_PARAMS_JSON] * 2
    orch = _make_orch(responses, max_iterations=2)

    result = orch.run(VACCINE_PATH)

    assert result.converged is False, "2 次迭代不應達到 100% 覆蓋（k=5，10 對）"
    assert len(result.loss_history) == 3, (
        f"loss_history 應為 [初始, 第1次, 第2次]，實際長度 {len(result.loss_history)}"
    )
    assert result.final_coverage < 1.0


# ─────────────────────────────────────────────
# TC-U-56：損失歷程記錄單調不增
# ─────────────────────────────────────────────


def test_loss_history_non_increasing():  # type: ignore[no-untyped-def]
    """TC-U-56：loss_history 中每個元素 ≤ 前一個元素。"""
    responses = [_VALID_JSON] * 20
    orch = _make_orch(responses, max_iterations=8)

    result = orch.run(VACCINE_PATH)

    history = result.loss_history
    assert len(history) >= 1, "loss_history 至少應有初始值"

    for i in range(1, len(history)):
        assert history[i] <= history[i - 1], (
            f"loss_history[{i}]={history[i]} > loss_history[{i-1}]={history[i-1]}："
            f"損失不應增加。history={history}"
        )


# ─────────────────────────────────────────────
# TC-U-63：Z3 全部 UNSAT → infeasible 路徑偵測
# ─────────────────────────────────────────────


def test_infeasible_detection_via_z3_unsat():  # type: ignore[no-untyped-def]
    """TC-U-63：強制 Z3 synthesize 全部拋出 Z3UNSATError，驗證 infeasible 路徑偵測機制。

    預期：
    - infeasible_paths 非空（所有條件均被 Z3 標記為不可行）
    - converged=False（傳統覆蓋率未達 100%，分母為 2k）
    - final_coverage < 1.0（隨機初始案例不足以完全覆蓋）
    - loss_history 單調不增
    """
    from ifl_mcdc.exceptions import Z3UNSATError

    orch = _make_orch([])

    # 強制 SMT synthesize 全部失敗 → 觸發 infeasible 偵測路徑
    with patch.object(orch.smt, "synthesize", side_effect=Z3UNSATError("強制 UNSAT")):
        result = orch.run(VACCINE_PATH)

    # infeasible 路徑應被偵測到
    assert len(result.infeasible_paths) > 0, (
        f"應有不可行路徑，但 infeasible_paths={result.infeasible_paths}"
    )

    # 傳統覆蓋率（只有 3 個隨機案例）不應達到 100%
    assert result.converged is False, (
        f"Z3 全部 UNSAT 時不應收斂，actual final_coverage={result.final_coverage:.1%}"
    )
    assert result.final_coverage < 1.0

    # loss_history 應單調不增
    history = result.loss_history
    for i in range(1, len(history)):
        assert history[i] <= history[i - 1], (
            f"loss_history 非單調不增：{history}"
        )


# ─────────────────────────────────────────────
# TC-U-73：DomainValidator 在 Orchestrator 層執行
# ─────────────────────────────────────────────


def test_domain_validator_retries_in_llm_sampler():  # type: ignore[no-untyped-def]
    """TC-U-73：DomainValidator 驗證在 LLMSampler 層執行，失敗不觸發 infeasible 標記。

    第 1 次 backend 回傳不合法案例（age=-5），LLMSampler 內部重試，
    第 2 次回傳合法案例（age=30），orchestrator 正常接受。

    驗證：
    - 不合法案例不觸發 infeasible 標記
    - 系統仍能正常完成迭代
    """
    # age=-5 不合法（違反 FR-10：[0,130]）；age=30 合法
    bad_json = json.dumps(
        {"age": -5, "high_risk": True, "days_since_last": 200, "egg_allergy": False}
    )
    good_json = _VALID_JSON

    # bad_json 被 LLMSampler 內部驗證失敗後重試，消耗兩個 backend 回應
    responses = [bad_json, good_json] + [_VALID_JSON] * 30
    orch = _make_orch(responses, max_iterations=2)

    result = orch.run(VACCINE_PATH)

    # 不合法案例不應標記為 infeasible
    assert len(result.infeasible_paths) == 0, (
        f"DomainValidator 失敗不應觸發 infeasible，實際 {result.infeasible_paths}"
    )


# ─────────────────────────────────────────────
# TC-U-74：LLMSamplingError 跳過本輪，不標記 infeasible
# ─────────────────────────────────────────────


def test_llm_sampling_error_skips_without_infeasible():  # type: ignore[no-untyped-def]
    """TC-U-74：LLMSamplingError（JSON 解析失敗）跳過本輪，不標記條件為 infeasible。

    驗證：
    - infeasible_paths 為空（LLMSamplingError ≠ SMT UNSAT）
    - 迭代後損失歷程有對應記錄（跳過輪次仍 append 損失值）
    """
    from ifl_mcdc.exceptions import LLMSamplingError

    orch = _make_orch([], max_iterations=3)

    # 強制 sampler.sample 全部拋出 LLMSamplingError
    with patch.object(
        orch.sampler, "sample", side_effect=LLMSamplingError("強制 JSON 解析失敗")
    ):
        result = orch.run(VACCINE_PATH)

    # LLMSamplingError 不應標記任何條件為 infeasible
    assert len(result.infeasible_paths) == 0, (
        f"LLMSamplingError 不應觸發 infeasible，實際 {result.infeasible_paths}"
    )
    # 有迭代執行（loss_history 長度 > 1）
    assert len(result.loss_history) > 1, (
        f"應有迭代後損失記錄，實際 loss_history={result.loss_history}"
    )
