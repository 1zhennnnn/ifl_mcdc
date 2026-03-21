"""
run_validation.py

企畫書第四階段三組對照實驗驗收腳本。

三個評估指標（來自企畫書）：
  ① MC/DC 覆蓋率 = 100%
  ② 迭代次數 N_iteration
  ③ 有效測試生成比 = Valid Cases / Total Generated Cases

執行方式：
  $env:IFL_LLM_API_KEY = "你的金鑰"
  $env:IFL_LLM_PROVIDER = "openai"   # 或 "anthropic"
  $env:IFL_LLM_MODEL    = "gpt-4o"
  python run_validation.py 2>&1 | Tee-Object -FilePath validation_results.txt
"""
from __future__ import annotations

import json
import os
import statistics
import types
import uuid
from pathlib import Path

# 優先從 .env 載入環境變數（若 pydantic-settings 未預載）
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env", override=False)
except ImportError:
    pass

from ifl_mcdc.config import IFLConfig
from ifl_mcdc.exceptions import LLMSamplingError
from ifl_mcdc.layer1 import probe_injector as pi
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

# ── API 金鑰檢查 ──────────────────────────────────────────
HAS_API_KEY = bool(os.environ.get("IFL_LLM_API_KEY", "").strip())
if not HAS_API_KEY:
    print("=" * 55)
    print("  警告：未設定 IFL_LLM_API_KEY")
    print("  對照組B（純LLM）與實驗組（IFL）將略過。")
    print("  僅執行對照組A（純隨機）。")
    print()
    print("  設定金鑰後重新執行：")
    print("    $env:IFL_LLM_API_KEY = '你的金鑰'")
    print("    $env:IFL_LLM_PROVIDER = 'openai'")
    print("    $env:IFL_LLM_MODEL    = 'gpt-4o'")
    print("=" * 55)

import random as _rand

def _make_pure_llm_prompt() -> str:
    """每次呼叫產生含隨機情境提示的多樣化 Prompt，避免 LLM 固定輸出。"""
    scenarios = [
        "年輕高風險成人（18-40 歲）",
        "中年低風險族群（40-64 歲）",
        "老年族群（65 歲以上）",
        "未達施打年齡（未滿 18 歲）",
        "有蛋類過敏史的患者",
        "距上次接種不足 6 個月者",
        "高風險職業工作者（18-50 歲）",
        "距上次接種超過 2 年的老年人",
    ]
    scenario = _rand.choice(scenarios)
    age_hint = _rand.randint(0, 130)
    days_hint = _rand.randint(0, 3650)
    return f"""你是一個軟體測試工程師，負責測試疫苗施打資格篩選系統。

函式定義：
def check_vaccine_eligibility(age, high_risk, days_since_last, egg_allergy):
    if ((age >= 65 or (age >= 18 and high_risk))
        and (days_since_last > 180) and not egg_allergy):
        return True
    return False

請為以下情境生成一個真實的測試案例：
情境：{scenario}
參考年齡範圍：約 {age_hint} 歲
參考接種間隔：約 {days_hint} 天

要求：
- age: 0~130 之間的整數
- high_risk: true 或 false（是否高風險族群）
- days_since_last: 0~3650 之間的整數（距上次接種天數）
- egg_allergy: true 或 false（是否對蛋過敏）

只輸出 JSON 物件，不要任何說明：
{{"age": int, "high_risk": bool, "days_since_last": int, "egg_allergy": bool}}
"""


def _print_cases(cases: list[dict]) -> None:
    """印出案例明細，每筆一行，含來源標記與驗證結果。"""
    if not cases:
        print("    （無案例）")
        return
    for idx, c in enumerate(cases, 1):
        src    = c.get("__source", "?")
        fields = {k: v for k, v in c.items() if not k.startswith("__")}
        if not fields:
            print(f"    T{idx:02d} [{src:<7}] [ERR]  （LLM 呼叫失敗）")
            continue
        vr   = validator.validate(json.dumps(fields))
        mark = "OK" if vr.passed else "NG"
        vals = ", ".join(f"{k}={v}" for k, v in fields.items())
        print(f"    T{idx:02d} [{src:<7}] [{mark}]  {vals}")


def run_group(name: str, get_result_fn) -> list[dict]:
    """執行單組實驗 RUNS 次，印出每輪結果與案例明細，回傳結果列表。"""
    print(f"\n{'='*50}")
    print(name)
    print(f"{'='*50}")
    results = []
    for i in range(RUNS):
        r = get_result_fn()
        total = r["total"]
        valid = r["valid"]
        ratio = valid / total if total > 0 else 0.0
        results.append({**r, "valid_ratio": ratio})
        infeasible = r.get("infeasible_paths", [])
        loss_hist  = r.get("loss_history", [])
        print(
            f"Run {i+1}: 收斂={r['converged']}, "
            f"覆蓋率={r['coverage']:.1%}, "
            f"迭代={r['iterations']}, "
            f"有效比={valid}/{total}={ratio:.1%}"
        )
        if infeasible:
            print(f"    [infeasible] {infeasible}")
        if loss_hist:
            print(f"    [loss]       {loss_hist}")
        _print_cases(r.get("cases", []))
    return results


# ──────────────────────────────────────────────────────────
# 對照組A：純隨機（max_iterations=0）
# ──────────────────────────────────────────────────────────

def run_random() -> dict:
    config = IFLConfig(max_iterations=0)
    result = IFLOrchestrator(config=config).run(VACCINE_PATH)
    # 保留 __source 供 _print_cases 使用
    raw_cases = list(result.test_suite)
    clean = [{k: v for k, v in tc.items() if not k.startswith("__")} for tc in raw_cases]
    valid = sum(1 for c in clean if validator.validate(json.dumps(c)).passed)
    return {
        "converged":  result.converged,
        "coverage":   result.final_coverage,
        "iterations": result.iteration_count,
        "tokens":     result.total_tokens,
        "total":      len(clean),
        "valid":      valid,
        "cases":      raw_cases,
    }


# ──────────────────────────────────────────────────────────
# 對照組B：純 LLM（無 SMT 導引）
# 直接給 LLM 函式描述，生成 10 個案例後計算 MC/DC 覆蓋率
# ──────────────────────────────────────────────────────────

# 預先解析並注入探針（各 run 共用同一份儀表板原始碼）
_nodes_b = ASTParser().parse_file(VACCINE_PATH)
_src_b   = Path(VACCINE_PATH).read_text(encoding="utf-8")
_inst_b  = ProbeInjector(_nodes_b).inject(_src_b)

def run_pure_llm() -> dict:
    config = IFLConfig()
    sampler = LLMSampler(config.llm_backend, DomainValidator(DEFAULT_MEDICAL_RULES))

    # 動態載入儀表板模組
    mod_name = f"_pure_llm_{uuid.uuid4().hex[:8]}"
    mod = types.ModuleType(mod_name)
    exec(compile(_inst_b, mod_name, "exec"), mod.__dict__)  # noqa: S102

    log = ProbeLog()
    pi._GLOBAL_LOG = log
    mod.__dict__["_ifl_probe"]           = pi._ifl_probe
    mod.__dict__["_ifl_record_decision"] = pi._ifl_record_decision

    total, valid_count = 0, 0
    collected: list[dict] = []
    for _ in range(10):
        total += 1
        try:
            case, vr = sampler.sample(_make_pure_llm_prompt())
            status = "llm" if vr.passed else "llm-ng"
            collected.append({**case, "__source": status})
            if vr.passed:
                valid_count += 1
                tid = f"T{uuid.uuid4().hex[:8]}"
                setattr(pi._CURRENT_TEST_ID, "value", tid)
                try:
                    mod.check_vaccine_eligibility(**case)
                except Exception:
                    pass
        except LLMSamplingError:
            collected.append({"__source": "llm-err"})

    matrix = MCDCCoverageEngine().build_matrix(_nodes_b[0].condition_set, log)
    tokens = sum(e.get("est_tokens", 0) for e in sampler.token_log)
    return {
        "converged":  matrix.compute_loss() == 0,
        "coverage":   matrix.coverage_ratio,
        "iterations": total,
        "tokens":     tokens,
        "total":      total,
        "valid":      valid_count,
        "cases":      collected,
    }


# ──────────────────────────────────────────────────────────
# 實驗組：本系統 IFL（SMT 導引 + LLM 生成）
# ──────────────────────────────────────────────────────────

def run_ifl() -> dict:
    config = IFLConfig(max_iterations=15)
    result = IFLOrchestrator(config=config).run(VACCINE_PATH)

    # 有效比：llm 來源案例（True 側 LLM 生成，已被 AcceptanceGate 接受）
    llm_cases = [
        {k: v for k, v in tc.items() if not k.startswith("__")}
        for tc in result.test_suite
        if tc.get("__source") == "llm"
    ]
    valid = sum(1 for c in llm_cases if validator.validate(json.dumps(c)).passed)
    total = result.iteration_count  # 每次迭代呼叫一次 LLM（True 側）

    return {
        "converged":        result.converged,
        "coverage":         result.final_coverage,
        "iterations":       result.iteration_count,
        "tokens":           result.total_tokens,
        "total":            total if total > 0 else len(llm_cases),
        "valid":            valid,
        "cases":            list(result.test_suite),   # 含 __source / __test_id
        "infeasible_paths": result.infeasible_paths,   # 診斷用
        "loss_history":     result.loss_history,       # 診斷用
    }


# ──────────────────────────────────────────────────────────
# 執行三組
# ──────────────────────────────────────────────────────────

A = run_group("對照組A：純隨機", run_random)

if HAS_API_KEY:
    B = run_group("對照組B：純 LLM（無 SMT 導引）", run_pure_llm)
    C = run_group("實驗組：本系統 IFL", run_ifl)
else:
    B = None
    C = None


# ──────────────────────────────────────────────────────────
# 彙整結果
# ──────────────────────────────────────────────────────────

def avg(lst: list[dict], key: str) -> float:
    return statistics.mean(r[key] for r in lst)

def conv(lst: list[dict]) -> float:
    return sum(1 for r in lst if r["converged"]) / len(lst)


def _fmt_pct(lst, key):
    return f"{avg(lst, key):>13.1%}" if lst is not None else f"{'（略過）':>13}"

def _fmt_f(lst, key):
    return f"{avg(lst, key):>13.1f}" if lst is not None else f"{'（略過）':>13}"

def _fmt_tok(lst, key):
    return f"{avg(lst, key):>13.0f}" if lst is not None else f"{'（略過）':>13}"

def _fmt_conv(lst):
    return f"{conv(lst):>13.1%}" if lst is not None else f"{'（略過）':>13}"

print("\n" + "=" * 65)
print("企畫書評估指標彙整（{} 次平均）".format(RUNS))
print("=" * 65)
print(f"{'指標':<22} {'對照A 純隨機':>14} {'對照B 純LLM':>14} {'實驗組 IFL':>14}")
print("-" * 65)
print(f"{'① MC/DC 覆蓋率':<22} {_fmt_pct(A,'coverage')} {_fmt_pct(B,'coverage')} {_fmt_pct(C,'coverage')}")
print(f"{'① 收斂率':<22} {_fmt_conv(A)} {_fmt_conv(B)} {_fmt_conv(C)}")
print(f"{'② 迭代次數':<22} {_fmt_f(A,'iterations')} {_fmt_f(B,'iterations')} {_fmt_f(C,'iterations')}")
print(f"{'③ 有效測試生成比':<22} {_fmt_pct(A,'valid_ratio')} {_fmt_pct(B,'valid_ratio')} {_fmt_pct(C,'valid_ratio')}")
print(f"{'Token 消耗':<22} {_fmt_tok(A,'tokens')} {_fmt_tok(B,'tokens')} {_fmt_tok(C,'tokens')}")


# ──────────────────────────────────────────────────────────
# 企畫書驗收條件確認
# ──────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("企畫書驗收條件（實驗組 IFL）")
print("=" * 65)

if C is None:
    print("（實驗組略過：未設定 IFL_LLM_API_KEY）")
else:
    ifl_conv  = conv(C)
    ifl_cov   = avg(C, "coverage")
    ifl_valid = avg(C, "valid_ratio")
    ifl_iter  = avg(C, "iterations")

    print(f"① MC/DC 覆蓋率 = 100%：{'PASS' if ifl_cov >= 1.0 else 'FAIL'} ({ifl_cov:.1%})")
    print(f"② 迭代次數 <= 50：{'PASS' if ifl_iter <= 50 else 'FAIL'} ({ifl_iter:.1f} 次)")
    print(f"③ 有效生成比 >= 85%：{'PASS' if ifl_valid >= 0.85 else 'FAIL'} ({ifl_valid:.1%})")
    print(f"   收斂率：{ifl_conv:.1%} ({sum(1 for r in C if r['converged'])}/{RUNS})")
