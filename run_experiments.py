"""
IFL MC/DC 自動實驗腳本
跑 5 次 E2E 並生成論文用報告

執行方式：
  $env:IFL_LLM_API_KEY = "你的金鑰"
  python run_experiments.py
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from ifl_mcdc.config import IFLConfig
from ifl_mcdc.layer3.domain_validator import DomainValidator
from ifl_mcdc.orchestrator import IFLOrchestrator

VACCINE_PATH = Path(__file__).parent / "tests" / "fixtures" / "vaccine_eligibility.py"
REPORT_PATH = Path(__file__).parent / "experiment_report.json"
N_RUNS = 5


def run_once(run_id: int) -> dict:
    print(f"\n{'='*60}")
    print(f"  第 {run_id} 次實驗開始")
    print(f"{'='*60}")

    config = IFLConfig(max_iterations=50)
    orch = IFLOrchestrator(config=config)

    start = time.time()
    result = orch.run(VACCINE_PATH)
    elapsed = time.time() - start

    validator = DomainValidator()
    from collections import defaultdict
    source_stats: dict = defaultdict(lambda: {"total": 0, "valid": 0})
    all_valid = True
    for tc in result.test_suite:
        source = tc.get("__source", "unknown")
        clean = {k: v for k, v in tc.items() if not k.startswith("__")}
        vr = validator.validate(json.dumps(clean))
        source_stats[source]["total"] += 1
        if vr.passed:
            source_stats[source]["valid"] += 1
        else:
            all_valid = False

    data = {
        "run_id":          run_id,
        "converged":       result.converged,
        "final_coverage":  result.final_coverage,
        "iteration_count": result.iteration_count,
        "total_tokens":    result.total_tokens,
        "test_suite_size": len(result.test_suite),
        "infeasible_paths": result.infeasible_paths,
        "loss_history":    result.loss_history,
        "elapsed_seconds": round(elapsed, 2),
        "all_cases_valid": all_valid,
        "test_suite":      [
            {k: v for k, v in tc.items() if not k.startswith("__")}
            for tc in result.test_suite
        ],
    }
    for source, stats in source_stats.items():
        total = stats["total"]
        valid = stats["valid"]
        rate = valid / total if total > 0 else 0.0
        data[f"valid_rate_{source}"] = round(rate, 3)
        data[f"count_{source}"] = total

    print(f"  收斂：       {result.converged}")
    print(f"  最終覆蓋率： {result.final_coverage:.1%}")
    print(f"  迭代次數：   {result.iteration_count}")
    print(f"  Token 消耗： {result.total_tokens}")
    print(f"  測試案例數： {len(result.test_suite)}")
    print(f"  執行時間：   {elapsed:.1f}s")
    print(f"  Loss 歷程：  {result.loss_history}")

    return data


def main() -> None:
    if not os.environ.get("IFL_LLM_API_KEY"):
        print("❌ 請先設定 IFL_LLM_API_KEY")
        return

    results = []
    for i in range(1, N_RUNS + 1):
        try:
            data = run_once(i)
            results.append(data)
        except Exception as e:
            print(f"❌ 第 {i} 次實驗失敗：{e}")
            results.append({"run_id": i, "error": str(e)})

    # 統計摘要
    successful = [r for r in results if r.get("converged")]
    avg_iter   = sum(r["iteration_count"] for r in successful) / len(successful) if successful else 0
    avg_tokens = sum(r["total_tokens"]    for r in successful) / len(successful) if successful else 0
    avg_time   = sum(r["elapsed_seconds"] for r in successful) / len(successful) if successful else 0

    def _avg_rate(key: str) -> float:
        vals = [r[key] for r in successful if key in r]
        return sum(vals) / len(vals) if vals else 0.0

    summary = {
        "total_runs":            N_RUNS,
        "converged_runs":        len(successful),
        "convergence_rate":      f"{len(successful)/N_RUNS:.1%}",
        "avg_iterations":        round(avg_iter, 1),
        "avg_tokens":            round(avg_tokens, 1),
        "avg_elapsed_sec":       round(avg_time, 1),
        "avg_valid_rate_random": round(_avg_rate("valid_rate_random"), 3),
        "avg_valid_rate_smt":    round(_avg_rate("valid_rate_smt"), 3),
        "avg_valid_rate_llm":    round(_avg_rate("valid_rate_llm"), 3),
    }

    report = {"summary": summary, "runs": results}

    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\n{'='*60}")
    print("  實驗完成！摘要")
    print(f"{'='*60}")
    print(f"  收斂率：          {summary['convergence_rate']}")
    print(f"  平均迭代數：      {summary['avg_iterations']}")
    print(f"  平均 Token：      {summary['avg_tokens']}")
    print(f"  平均時間：        {summary['avg_elapsed_sec']}s")
    print(f"  隨機案例有效率：  {summary.get('avg_valid_rate_random', 0):.1%}")
    print(f"  SMT 案例有效率：  {summary.get('avg_valid_rate_smt', 0):.1%}")
    print(f"  LLM 案例有效率：  {summary.get('avg_valid_rate_llm', 0):.1%}")
    print(f"\n  報告已儲存至：{REPORT_PATH}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
