"""
疫苗邏輯端對端系統測試。

TC-S-04: IFLOrchestrator 完整執行，回傳 IFLResult 結構完整

執行方式：
  RUN_E2E=1 pytest tests/integration/test_vaccine_e2e.py -v
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from ifl_mcdc.config import IFLConfig
from ifl_mcdc.orchestrator import IFLOrchestrator, IFLResult

VACCINE_PATH = Path(__file__).parent.parent / "fixtures" / "vaccine_eligibility.py"


@pytest.mark.skipif(
    not os.environ.get("RUN_E2E"),
    reason="端對端測試，需設定 RUN_E2E=1 才執行（需 LLM API 金鑰）",
)
def test_vaccine_e2e_full_run() -> None:
    """TC-S-04：完整執行 IFLOrchestrator，回傳結構完整的 IFLResult。

    驗證：
    - 回傳 IFLResult，所有欄位型別正確
    - final_coverage ∈ [0.0, 1.0]
    - loss_history 至少有初始值，且單調不增
    - test_suite 為 list（可空）
    """
    config = IFLConfig(max_iterations=50)
    orch = IFLOrchestrator(config=config)

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

    # 損失歷程單調不增
    history = result.loss_history
    for i in range(1, len(history)):
        assert history[i] <= history[i - 1], (
            f"loss_history[{i}]={history[i]} > loss_history[{i-1}]={history[i-1]}：損失不應增加"
        )
