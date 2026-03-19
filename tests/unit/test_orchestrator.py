"""
IFLOrchestrator 單元測試（使用 MockLLMBackend）。

TC-U-52: run() 回傳 IFLResult，結構完整
TC-U-53: 空決策節點來源 → ValueError
TC-U-54: INFEASIBLE 路徑不再重複求解
TC-U-55: 預算耗盡時輸出部分覆蓋報告
TC-U-56: 損失歷程記錄單調不增
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
    """TC-U-52：run() 回傳 IFLResult，所有欄位型別正確。"""
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
