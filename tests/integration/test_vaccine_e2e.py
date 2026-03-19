"""
疫苗邏輯端對端系統測試。

TC-S-04: IFLOrchestrator 完整執行，回傳 IFLResult 結構完整

執行方式：
  RUN_E2E=1 pytest tests/integration/test_vaccine_e2e.py -v -s
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from ifl_mcdc.config import IFLConfig
from ifl_mcdc.layer3.domain_validator import DomainValidator
from ifl_mcdc.orchestrator import IFLOrchestrator, IFLResult

VACCINE_PATH = Path(__file__).parent.parent / "fixtures" / "vaccine_eligibility.py"


@pytest.mark.skipif(
    not os.environ.get("RUN_E2E"),
    reason="端對端測試，需設定 RUN_E2E=1 才執行（需 LLM API 金鑰）",
)
def test_vaccine_e2e_full_run() -> None:
    """TC-S-04：完整執行 IFLOrchestrator，驗證 100% MC/DC 收斂。

    驗證：
    - 回傳 IFLResult，所有欄位型別正確
    - final_coverage == 1.0（100% MC/DC）
    - converged == True
    - iteration_count <= 50
    - loss_history 單調不增，最終為 0
    - test_suite >= 8 個案例（k=5，最少需要 10 個翻轉對）
    - 所有測試案例通過 DomainValidator
    - infeasible_paths 為空
    """
    config = IFLConfig(max_iterations=50)
    orch = IFLOrchestrator(config=config)

    result = orch.run(VACCINE_PATH)

    # ── 列印論文用數據 ──────────────────────────────────
    print("\n" + "=" * 60)
    print("  IFL MC/DC E2E 實驗結果")
    print("=" * 60)
    print(f"  收斂：         {result.converged}")
    print(f"  最終覆蓋率：   {result.final_coverage:.1%}")
    print(f"  迭代次數：     {result.iteration_count}")
    print(f"  消耗 Token：   {result.total_tokens}")
    print(f"  測試案例數：   {len(result.test_suite)}")
    print(f"  不可行路徑：   {result.infeasible_paths}")
    print(f"  Loss 歷程：    {result.loss_history}")
    print("=" * 60)
    for i, tc in enumerate(result.test_suite):
        clean = {k: v for k, v in tc.items() if not k.startswith("__")}
        print(f"  T{i+1:02d}: {clean}")
    print("=" * 60 + "\n")

    # ── 結構正確性 ──────────────────────────────────────
    assert isinstance(result, IFLResult)
    assert isinstance(result.converged, bool)
    assert 0.0 <= result.final_coverage <= 1.0
    assert isinstance(result.test_suite, list)
    assert isinstance(result.iteration_count, int)
    assert isinstance(result.total_tokens, int)
    assert isinstance(result.infeasible_paths, list)
    assert isinstance(result.loss_history, list)
    assert len(result.loss_history) >= 1

    # ── 收斂驗證（論文核心斷言）───────────────────────
    assert result.converged, (
        f"系統未收斂，最終覆蓋率：{result.final_coverage:.1%}"
    )
    assert result.final_coverage == 1.0, (
        f"覆蓋率未達 100%：{result.final_coverage:.1%}"
    )
    assert result.iteration_count <= 50, (
        f"迭代次數超限：{result.iteration_count}"
    )
    assert result.loss_history[-1] == 0, (
        f"最終 loss 非零：{result.loss_history[-1]}"
    )
    assert len(result.infeasible_paths) == 0, (
        f"存在不可行路徑：{result.infeasible_paths}"
    )

    # ── 測試案例數量 ────────────────────────────────────
    assert len(result.test_suite) >= 8, (
        f"測試案例不足：{len(result.test_suite)}"
    )

    # ── Loss 歷程單調不增 ────────────────────────────────
    history = result.loss_history
    for i in range(1, len(history)):
        assert history[i] <= history[i - 1], (
            f"loss_history[{i}]={history[i]} > [{i-1}]={history[i-1]}：損失不應增加"
        )

    # ── 所有測試案例通過領域驗證 ─────────────────────────
    validator = DomainValidator()
    for tc in result.test_suite:
        clean = {k: v for k, v in tc.items() if not k.startswith("__")}
        vr = validator.validate(json.dumps(clean))
        assert vr.passed, (
            f"非法測試案例：{clean}，違規：{vr.violations}"
        )