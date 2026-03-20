"""
複雜度對照實驗腳本
測試 k=5, k=8, k=9, k=10 在三種模式下的表現

"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from ifl_mcdc.config import IFLConfig
from ifl_mcdc.layer3.llm_sampler import MockLLMBackend
from ifl_mcdc.orchestrator import IFLOrchestrator

FIXTURES = {
    "k5_vaccine": {
        "path": Path("tests/fixtures/vaccine_eligibility.py"),
        "func": "check_vaccine_eligibility",
        "domain_types": {
            "age": "int",
            "high_risk": "bool",
            "days_since_last": "int",
            "egg_allergy": "bool",
        },
    },
    "k8_drug": {
        "path": Path("tests/fixtures/complex_medical_logic.py"),
        "func": "check_drug_interaction",
        "domain_types": {
            "age": "int",
            "renal_failure": "bool",
            "liver_disease": "bool",
            "taking_warfarin": "bool",
            "taking_aspirin": "bool",
            "systolic_bp": "int",
            "heart_rate": "int",
            "is_pregnant": "bool",
        },
    },
    "k9_surgery": {
        "path": Path("tests/fixtures/surgery_risk.py"),
        "func": "check_surgery_risk",
        "domain_types": {
            "age": "int",
            "obese": "bool",
            "has_diabetes": "bool",
            "has_hypertension": "bool",
            "is_smoker": "bool",
            "low_hemoglobin": "bool",
            "low_platelets": "bool",
            "cardiac_history": "bool",
            "has_copd": "bool",
        },
    },
    "k10_icu": {
        "path": Path("tests/fixtures/icu_admission.py"),
        "func": "check_icu_admission",
        "domain_types": {
            "age": "int",
            "low_bp": "bool",
            "high_heart_rate": "bool",
            "high_resp_rate": "bool",
            "high_temp": "bool",
            "low_gcs": "bool",
            "low_oxygen": "bool",
            "low_urine": "bool",
            "high_creatinine": "bool",
            "sepsis": "bool",
        },
    },
}

N_RUNS = 3  # 每組跑 3 次取平均


def run_experiment(
    name: str,
    fixture: dict,
    mode: str,
) -> dict:
    """執行單次實驗"""
    config = IFLConfig(
        max_iterations=50,
        func_name=fixture["func"],
        domain_types=fixture["domain_types"],
    )

    if mode == "random":
        config = IFLConfig(
            max_iterations=0,
            func_name=fixture["func"],
            domain_types=fixture["domain_types"],
        )
        backend = MockLLMBackend(responses=[])
        orch = IFLOrchestrator(config=config, backend=backend)
    elif mode == "smt_only":
        config = IFLConfig(
            max_iterations=50,
            func_name=fixture["func"],
            domain_types=fixture["domain_types"],
            llm_retry_delay=0.0,
        )
        backend = MockLLMBackend(responses=[Exception("mock")] * 200)
        orch = IFLOrchestrator(config=config, backend=backend)
    else:  # full
        orch = IFLOrchestrator(config=config)

    start = time.time()
    result = orch.run(fixture["path"])
    elapsed = time.time() - start

    return {
        "name": name,
        "mode": mode,
        "converged": result.converged,
        "final_coverage": result.final_coverage,
        "iteration_count": result.iteration_count,
        "total_tokens": result.total_tokens,
        "elapsed_seconds": round(elapsed, 2),
        "loss_history": result.loss_history,
        "infeasible_paths": result.infeasible_paths,
    }


def main() -> None:
    if not os.environ.get("IFL_LLM_API_KEY") and True:
        print("⚠️  未設定 IFL_LLM_API_KEY，完整模式（SMT+LLM）將無法執行")
        print("   只執行 模式A 和 模式B\n")

    has_api = bool(os.environ.get("IFL_LLM_API_KEY"))
    modes = ["random", "smt_only"] + (["full"] if has_api else [])

    all_results = []

    for fixture_name, fixture in FIXTURES.items():
        print(f"\n{'='*60}")
        print(f"  測試標靶：{fixture_name}  函式：{fixture['func']}")
        print(f"{'='*60}")

        for mode in modes:
            mode_label = {"random": "模式A 純隨機", "smt_only": "模式B 純SMT", "full": "模式C SMT+LLM"}[mode]
            runs = []

            for run_i in range(N_RUNS):
                try:
                    result = run_experiment(fixture_name, fixture, mode)
                    runs.append(result)
                    print(f"  {mode_label} 第{run_i+1}次: 收斂={result['converged']}, "
                          f"覆蓋率={result['final_coverage']:.1%}, "
                          f"迭代={result['iteration_count']}, "
                          f"時間={result['elapsed_seconds']}s")
                except Exception as e:
                    print(f"  {mode_label} 第{run_i+1}次: 失敗 - {e}")
                    runs.append({"error": str(e), "converged": False})

            # 計算平均
            successful = [r for r in runs if r.get("converged")]
            summary = {
                "fixture": fixture_name,
                "mode": mode,
                "n_runs": N_RUNS,
                "converged_runs": len(successful),
                "convergence_rate": len(successful) / N_RUNS,
                "avg_coverage": sum(r["final_coverage"] for r in successful) / len(successful) if successful else 0,
                "avg_iterations": sum(r["iteration_count"] for r in successful) / len(successful) if successful else 0,
                "avg_elapsed": sum(r["elapsed_seconds"] for r in successful) / len(successful) if successful else 0,
                "avg_tokens": sum(r["total_tokens"] for r in successful) / len(successful) if successful else 0,
                "runs": runs,
            }
            all_results.append(summary)

    # 儲存結果
    report_path = Path("complexity_experiment_report.json")
    report_path.write_text(
        json.dumps({"results": all_results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # 列印摘要表格
    print(f"\n{'='*80}")
    print("  複雜度實驗摘要")
    print(f"{'='*80}")
    print(f"  {'標靶':<15} {'模式':<15} {'收斂率':<8} {'平均覆蓋率':<10} {'平均迭代':<8} {'平均時間':<8}")
    print(f"  {'-'*70}")
    for r in all_results:
        print(f"  {r['fixture']:<15} {r['mode']:<15} "
              f"{r['convergence_rate']:.0%}     "
              f"{r['avg_coverage']:.1%}      "
              f"{r['avg_iterations']:.1f}      "
              f"{r['avg_elapsed']:.1f}s")

    print(f"\n  報告已儲存至：{report_path.absolute()}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
