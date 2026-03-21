"""
run_experiments.py

三組對照實驗（依企畫書第四階段實驗設計）：
  對照組A：傳統純隨機測試（max_iterations=0）
  對照組B：純 LLM 生成（無 SMT 導引，直接給 LLM 疫苗描述，生成 10 個案例）
  實驗組：本系統 IFL（SMT 導引 + LLM 生成，max_iterations=50）

三項評估指標：
  ① MC/DC 覆蓋率（目標 100%）
  ② 迭代次數 N_iteration
  ③ 有效測試生成比 = Valid Cases / Total Generated Cases

執行方式：
  $env:IFL_LLM_API_KEY = "你的金鑰"
  $env:IFL_LLM_PROVIDER = "openai"
  $env:IFL_LLM_MODEL = "gpt-4o"
  python run_experiments.py 2>&1 | Tee-Object -FilePath experiment_results.txt
"""
from __future__ import annotations

import json
import statistics
import sys
import time
import types
import uuid
from pathlib import Path

import ifl_mcdc.layer1.probe_injector as pi
from ifl_mcdc.config import IFLConfig
from ifl_mcdc.layer1.ast_parser import ASTParser
from ifl_mcdc.layer1.coverage_engine import MCDCCoverageEngine
from ifl_mcdc.layer1.probe_injector import ProbeInjector
from ifl_mcdc.layer3.domain_validator import DEFAULT_MEDICAL_RULES, DomainValidator
from ifl_mcdc.layer3.llm_sampler import LLMSampler
from ifl_mcdc.models.probe_record import ProbeLog
from ifl_mcdc.orchestrator import IFLOrchestrator

VACCINE_PATH = "tests/fixtures/vaccine_eligibility.py"
RUNS = 5
validator = DomainValidator(DEFAULT_MEDICAL_RULES)

PURE_LLM_PROMPT = """
你是一個軟體測試工程師。以下是一個疫苗施打資格篩選函式：

def check_vaccine_eligibility(age, high_risk, days_since_last, egg_allergy):
    if ((age >= 65 or (age >= 18 and high_risk))
        and (days_since_last > 180)
        and not egg_allergy):
        return True
    return False

請生成一個測試案例，以 JSON 格式輸出，
鍵名必須與函式參數名稱完全一致：
{"age": ..., "high_risk": ..., "days_since_last": ..., "egg_allergy": ...}
只輸出 JSON，不要任何說明文字。
"""


# ──────────────────────────────────────────────────────────
# 工具函式
# ──────────────────────────────────────────────────────────


def count_valid(test_suite: list[dict]) -> tuple[int, int]:
    """統計 test_suite 中通過 DomainValidator 的案例數。回傳 (valid, total)。"""
    total = len(test_suite)
    valid = sum(
        1 for tc in test_suite
        if validator.validate(
            json.dumps({k: v for k, v in tc.items() if not k.startswith("__")})
        ).passed
    )
    return valid, total


def summarize(results: list[dict], label: str) -> None:
    """印出一組實驗的 5 次平均摘要。"""
    cov_list   = [r["coverage"]    for r in results]
    iter_list  = [r["iterations"]  for r in results]
    ratio_list = [r["valid_ratio"] for r in results]
    tok_list   = [r["tokens"]      for r in results]
    conv_n     = sum(1 for r in results if r["converged"])

    print(f"\n  收斂率：      {conv_n}/{RUNS} = {conv_n/RUNS:.1%}")
    print(f"  平均覆蓋率：  {statistics.mean(cov_list):.1%}"
          f"  (std={statistics.stdev(cov_list):.1%})" if len(cov_list) > 1 else "")
    print(f"  平均迭代次數：{statistics.mean(iter_list):.1f}")
    print(f"  有效生成比：  {statistics.mean(ratio_list):.1%}")
    print(f"  平均 Token：  {statistics.mean(tok_list):.0f}")


# ──────────────────────────────────────────────────────────
# 對照組A：純隨機測試
# ──────────────────────────────────────────────────────────

print("\n" + "=" * 50)
print("對照組A：純隨機測試")
print("=" * 50)

results_A: list[dict] = []
for run in range(RUNS):
    config = IFLConfig(max_iterations=0)
    orch = IFLOrchestrator(config=config)
    t0 = time.time()
    result = orch.run(VACCINE_PATH)
    elapsed = time.time() - t0

    valid, total = count_valid(result.test_suite)
    rec = {
        "converged":   result.converged,
        "coverage":    result.final_coverage,
        "iterations":  result.iteration_count,
        "tokens":      result.total_tokens,
        "time":        elapsed,
        "total_cases": total,
        "valid_cases": valid,
        "valid_ratio": valid / total if total > 0 else 0.0,
    }
    results_A.append(rec)
    print(f"Run {run+1}: 收斂={result.converged}, "
          f"覆蓋率={result.final_coverage:.1%}, "
          f"迭代={result.iteration_count}, "
          f"有效比={valid}/{total}={valid/total:.1%}")

summarize(results_A, "對照組A")

# ──────────────────────────────────────────────────────────
# 對照組B：純 LLM 生成（無 SMT 導引）
# 直接給 LLM 疫苗邏輯描述，生成 10 個案例後計算 MC/DC 覆蓋率
# ──────────────────────────────────────────────────────────

print("\n" + "=" * 50)
print("對照組B：純 LLM 生成（無 SMT 導引）")
print("=" * 50)

# 預先解析並注入探針（各 run 共用同一份儀表板原始碼）
_parser = ASTParser()
_nodes = _parser.parse_file(VACCINE_PATH)
_src = Path(VACCINE_PATH).read_text(encoding="utf-8")
_injected_src = ProbeInjector(_nodes).inject(_src)

results_B: list[dict] = []
config_b = IFLConfig()
backend_b = config_b.llm_backend
sampler_b = LLMSampler(backend_b, retry_delay=0.5)

for run in range(RUNS):
    t0 = time.time()
    total_generated = 0
    valid_count = 0
    cases: list[dict] = []

    # 生成 10 個案例（模擬 10 次無導引迭代）
    for _ in range(10):
        total_generated += 1
        try:
            case = sampler_b.sample(PURE_LLM_PROMPT)
            vr = validator.validate(json.dumps(case))
            if vr.passed:
                valid_count += 1
                cases.append(case)
        except Exception:
            pass

    # 動態載入儀表板模組並執行案例，計算 MC/DC 覆蓋率
    mod_name = f"_pure_llm_{uuid.uuid4().hex[:8]}"
    mod = types.ModuleType(mod_name)
    exec(compile(_injected_src, mod_name, "exec"), mod.__dict__)  # noqa: S102
    sys.modules[mod_name] = mod

    log = ProbeLog()
    pi._GLOBAL_LOG = log
    mod.__dict__["_ifl_probe"] = pi._ifl_probe
    mod.__dict__["_ifl_record_decision"] = pi._ifl_record_decision

    for case in cases:
        tid = f"T{uuid.uuid4().hex[:8]}"
        setattr(pi._CURRENT_TEST_ID, "value", tid)
        try:
            mod.check_vaccine_eligibility(**case)
        except Exception:
            pass

    matrix = MCDCCoverageEngine().build_matrix(_nodes[0].condition_set, log)
    elapsed = time.time() - t0

    rec = {
        "converged":   matrix.compute_loss() == 0,
        "coverage":    matrix.coverage_ratio,
        "iterations":  total_generated,
        "tokens":      sum(e.get("est_tokens", 0) for e in sampler_b.token_log),
        "time":        elapsed,
        "total_cases": total_generated,
        "valid_cases": valid_count,
        "valid_ratio": valid_count / total_generated if total_generated > 0 else 0.0,
    }
    results_B.append(rec)
    print(f"Run {run+1}: 收斂={matrix.compute_loss()==0}, "
          f"覆蓋率={matrix.coverage_ratio:.1%}, "
          f"有效比={valid_count}/{total_generated}={valid_count/total_generated:.1%}")

summarize(results_B, "對照組B")

# ──────────────────────────────────────────────────────────
# 實驗組：本系統 IFL（SMT 導引 + LLM 生成）
# ──────────────────────────────────────────────────────────

print("\n" + "=" * 50)
print("實驗組：本系統 IFL（SMT + LLM）")
print("=" * 50)

results_C: list[dict] = []
for run in range(RUNS):
    config = IFLConfig(max_iterations=50)
    orch = IFLOrchestrator(config=config)
    t0 = time.time()
    result = orch.run(VACCINE_PATH)
    elapsed = time.time() - t0

    # 有效比：只計算 llm 來源案例（均已通過 DomainValidator）
    llm_cases = [tc for tc in result.test_suite if tc.get("__source") == "llm"]
    total_llm = len(llm_cases)
    # LLM 案例在加入 test_suite 前已通過 assertion，此處驗證為雙重確認
    valid_llm = sum(
        1 for tc in llm_cases
        if validator.validate(
            json.dumps({k: v for k, v in tc.items() if not k.startswith("__")})
        ).passed
    )

    rec = {
        "converged":   result.converged,
        "coverage":    result.final_coverage,
        "iterations":  result.iteration_count,
        "tokens":      result.total_tokens,
        "time":        elapsed,
        "total_cases": len(result.test_suite),
        "valid_cases": valid_llm,
        # 有效比以 LLM 生成案例為分子，總 LLM 迭代次數（=iteration_count）為分母
        "valid_ratio": valid_llm / result.iteration_count if result.iteration_count > 0 else 0.0,
    }
    results_C.append(rec)
    print(f"Run {run+1}: 收斂={result.converged}, "
          f"覆蓋率={result.final_coverage:.1%}, "
          f"迭代={result.iteration_count}, "
          f"有效比={valid_llm}/{result.iteration_count}={rec['valid_ratio']:.1%}")

summarize(results_C, "實驗組")

# ──────────────────────────────────────────────────────────
# 三組對照彙整報告
# ──────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("三組對照實驗結果彙整（{} 次平均）".format(RUNS))
print("=" * 60)


def avg(lst: list[dict], key: str) -> float:
    return statistics.mean(r[key] for r in lst)


def conv_rate(lst: list[dict]) -> float:
    return sum(1 for r in lst if r["converged"]) / len(lst)


print(f"\n{'指標':<20} {'對照A 純隨機':>14} {'對照B 純LLM':>14} {'實驗組 IFL':>14}")
print("-" * 65)
print(f"{'收斂率':<20} {conv_rate(results_A):>13.1%} "
      f"{conv_rate(results_B):>13.1%} {conv_rate(results_C):>13.1%}")
print(f"{'平均覆蓋率':<20} {avg(results_A, 'coverage'):>13.1%} "
      f"{avg(results_B, 'coverage'):>13.1%} {avg(results_C, 'coverage'):>13.1%}")
print(f"{'平均迭代次數':<20} {avg(results_A, 'iterations'):>13.1f} "
      f"{avg(results_B, 'iterations'):>13.1f} {avg(results_C, 'iterations'):>13.1f}")
print(f"{'有效測試生成比':<20} {avg(results_A, 'valid_ratio'):>13.1%} "
      f"{avg(results_B, 'valid_ratio'):>13.1%} {avg(results_C, 'valid_ratio'):>13.1%}")
print(f"{'平均 Token 消耗':<20} {avg(results_A, 'tokens'):>13.0f} "
      f"{avg(results_B, 'tokens'):>13.0f} {avg(results_C, 'tokens'):>13.0f}")
