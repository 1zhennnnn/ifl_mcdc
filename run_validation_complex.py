"""
run_validation_complex.py

複雜 Fixture 三組對照實驗驗收腳本（k=6 / 9 / 10）。

三個測試 Fixture（遞增複雜度）：
  ① 貸款審核  loan_approval   k=6   無共用變數，兩個 OR 子群
  ② 手術風險  surgery_risk    k=9   多層 AND/OR 嵌套，多 bool 欄位
  ③ ICU 入住  icu_admission   k=10  多個 OR 子群，最高複雜度

三個評估指標（同企畫書）：
  ① MC/DC 覆蓋率
  ② 迭代次數 N_iteration
  ③ 有效測試生成比 = Valid Cases / Total Generated Cases

執行方式：
  $env:IFL_LLM_API_KEY = "你的金鑰"
  $env:IFL_LLM_PROVIDER = "openai"   # 或 "anthropic"
  $env:IFL_LLM_MODEL    = "gpt-4o"
  python run_validation_complex.py 2>&1 | Tee-Object -FilePath validation_complex_results.txt
"""
from __future__ import annotations

import json
import os
import random as _rand
import statistics
import types
import uuid
from dataclasses import dataclass, field
from pathlib import Path

# 優先從 .env 載入環境變數
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
from ifl_mcdc.layer3.domain_validator import DomainValidator
from ifl_mcdc.layer3.llm_sampler import LLMSampler
from ifl_mcdc.models.probe_record import ProbeLog
from ifl_mcdc.models.validation import DomainRule
from ifl_mcdc.orchestrator import IFLOrchestrator

# ── 全域設定 ────────────────────────────────────────────────────
RUNS         = 3          # 每組執行次數（3 次求平均，降低 API 費用）
FIXTURES_DIR = Path("tests/fixtures")

HAS_API_KEY = bool(os.environ.get("IFL_LLM_API_KEY", "").strip())
if not HAS_API_KEY:
    print("=" * 60)
    print("  警告：未設定 IFL_LLM_API_KEY")
    print("  對照組B（純LLM）與實驗組（IFL）將略過。")
    print("  僅執行對照組A（純隨機）。")
    print()
    print("  設定金鑰後重新執行：")
    print("    $env:IFL_LLM_API_KEY = '你的金鑰'")
    print("    $env:IFL_LLM_PROVIDER = 'openai'")
    print("    $env:IFL_LLM_MODEL    = 'gpt-4o'")
    print("=" * 60)


# ── Fixture 規格定義 ───────────────────────────────────────────


@dataclass
class FixtureSpec:
    """封裝單一 Fixture 的全部設定與驗證規則。"""

    label: str                           # 顯示名稱
    path: str                            # Fixture 檔案路徑
    expected_k: int                      # 預期原子條件數
    func_name: str                       # 被測函式名稱
    func_sig: str                        # 函式簽名（供 Prompt 使用）
    domain_ctx: str                      # 領域情境描述
    domain_types: dict[str, str]         # 變數型別映射
    domain_bounds: dict[str, list[int]]  # 整數變數邊界
    domain_rules: list[DomainRule]       # 驗收用驗證規則
    max_ifl_iters: int                   # IFL 最大迭代次數
    llm_samples: int                     # 純 LLM 每輪樣本數
    scenarios: list[str]                 # 隨機情境清單（純 LLM 提示多樣化用）
    func_def: str                        # 函式定義文字（供純 LLM Prompt）
    prompt_fields_json: str              # JSON 格式說明（供純 LLM Prompt）


# ══════════════════════════════════════════════════════════════
#  Fixture ①：貸款審核 loan_approval（k=6）
# ══════════════════════════════════════════════════════════════

LOAN_RULES: list[DomainRule] = [
    DomainRule(
        field="credit_score",
        description="信用分數必須為 300～850 之間的整數",
        validator=lambda v: isinstance(v, int) and not isinstance(v, bool) and 300 <= v <= 850,
    ),
    DomainRule(
        field="annual_income",
        description="年收入必須為非負整數（單位：元）",
        validator=lambda v: isinstance(v, int) and not isinstance(v, bool) and v >= 0,
    ),
    DomainRule(
        field="loan_amount",
        description="貸款金額必須為正整數（單位：元）",
        validator=lambda v: isinstance(v, int) and not isinstance(v, bool) and v > 0,
    ),
    DomainRule(
        field="employed",
        description="受僱狀態必須為布林值",
        validator=lambda v: isinstance(v, bool),
    ),
    DomainRule(
        field="has_collateral",
        description="抵押品標記必須為布林值",
        validator=lambda v: isinstance(v, bool),
    ),
    DomainRule(
        field="bankruptcy_history",
        description="破產紀錄標記必須為布林值",
        validator=lambda v: isinstance(v, bool),
    ),
]

LOAN_SPEC = FixtureSpec(
    label="貸款審核 loan_approval（k=6）",
    path=str(FIXTURES_DIR / "loan_approval.py"),
    expected_k=6,
    func_name="check_loan_approval",
    func_sig="check_loan_approval(credit_score, annual_income, loan_amount, employed, has_collateral, bankruptcy_history)",
    domain_ctx="個人信用貸款審核系統",
    domain_types={
        "credit_score":      "int",
        "annual_income":     "int",
        "loan_amount":       "int",
        "employed":          "bool",
        "has_collateral":    "bool",
        "bankruptcy_history":"bool",
    },
    domain_bounds={
        "credit_score":  [300, 850],
        "annual_income": [0, 2_000_000],
        "loan_amount":   [10_000, 10_000_000],
    },
    domain_rules=LOAN_RULES,
    max_ifl_iters=25,
    llm_samples=14,   # 略多於 2*k=12，給純 LLM 公平機會
    scenarios=[
        "信用分數優良（≥700）且年收入充足的穩定受僱者",
        "信用分數不佳（<700）但年收入極高補足條件",
        "大額貸款超過 50 萬，需要抵押品擔保",
        "小額貸款且有抵押品，無破產紀錄",
        "自雇者（employed=False）貸款被拒案例",
        "有破產紀錄（bankruptcy_history=True）申請人",
        "信用分數剛達門檻（700），無抵押但貸款金額合理",
        "中等收入（4-5 萬）加信用良好的標準案例",
    ],
    func_def="""\
def check_loan_approval(credit_score, annual_income, loan_amount, employed, has_collateral, bankruptcy_history):
    if (
        (credit_score >= 700 or annual_income >= 50000)
        and (loan_amount <= 500000 or has_collateral)
        and employed
        and not bankruptcy_history
    ):
        return True
    return False""",
    prompt_fields_json='{"credit_score": int, "annual_income": int, "loan_amount": int, "employed": bool, "has_collateral": bool, "bankruptcy_history": bool}',
)


# ══════════════════════════════════════════════════════════════
#  Fixture ②：手術風險 surgery_risk（k=9）
# ══════════════════════════════════════════════════════════════

_BOOL_RULE = lambda field, desc: DomainRule(  # noqa: E731
    field=field,
    description=f"{desc}必須為布林值",
    validator=lambda v: isinstance(v, bool),
)

SURGERY_RULES: list[DomainRule] = [
    DomainRule(
        field="age",
        description="年齡必須為 0～120 之間的整數",
        validator=lambda v: isinstance(v, int) and not isinstance(v, bool) and 0 <= v <= 120,
    ),
    _BOOL_RULE("obese", "肥胖標記"),
    _BOOL_RULE("has_diabetes", "糖尿病標記"),
    _BOOL_RULE("has_hypertension", "高血壓標記"),
    _BOOL_RULE("is_smoker", "吸菸標記"),
    _BOOL_RULE("low_hemoglobin", "低血紅素標記"),
    _BOOL_RULE("low_platelets", "低血小板標記"),
    _BOOL_RULE("cardiac_history", "心臟病史標記"),
    _BOOL_RULE("has_copd", "慢阻肺標記"),
]

SURGERY_SPEC = FixtureSpec(
    label="手術風險 surgery_risk（k=9）",
    path=str(FIXTURES_DIR / "surgery_risk.py"),
    expected_k=9,
    func_name="check_surgery_risk",
    func_sig="check_surgery_risk(age, obese, has_diabetes, has_hypertension, is_smoker, low_hemoglobin, low_platelets, cardiac_history, has_copd)",
    domain_ctx="術前手術風險評估系統",
    domain_types={
        "age":             "int",
        "obese":           "bool",
        "has_diabetes":    "bool",
        "has_hypertension":"bool",
        "is_smoker":       "bool",
        "low_hemoglobin":  "bool",
        "low_platelets":   "bool",
        "cardiac_history": "bool",
        "has_copd":        "bool",
    },
    domain_bounds={"age": [0, 120]},
    domain_rules=SURGERY_RULES,
    max_ifl_iters=45,
    llm_samples=20,   # 略多於 2*k=18
    scenarios=[
        "高齡（≥70）且肥胖，多重代謝症候群",
        "中年糖尿病合併高血壓患者",
        "長期吸菸合併慢性阻塞性肺病（COPD）",
        "有心臟病史且低血小板，凝血風險高",
        "老年非肥胖但有心臟病史與吸菸史",
        "低血紅素合併糖尿病的中年患者",
        "年輕（<70）無肥胖但多重合併症",
        "完全健康的低風險年輕手術候選者",
        "單純高血壓、無其他危險因子的患者",
        "高齡肥胖、貧血、慢阻肺三重高風險",
    ],
    func_def="""\
def check_surgery_risk(age, obese, has_diabetes, has_hypertension, is_smoker, low_hemoglobin, low_platelets, cardiac_history, has_copd):
    if (
        (age >= 70 or obese)
        and (has_diabetes and has_hypertension)
        and (is_smoker or cardiac_history or has_copd)
        and (low_hemoglobin or low_platelets)
    ):
        return True
    return False""",
    prompt_fields_json='{"age": int, "obese": bool, "has_diabetes": bool, "has_hypertension": bool, "is_smoker": bool, "low_hemoglobin": bool, "low_platelets": bool, "cardiac_history": bool, "has_copd": bool}',
)


# ══════════════════════════════════════════════════════════════
#  Fixture ③：ICU 入住 icu_admission（k=10）
# ══════════════════════════════════════════════════════════════

ICU_RULES: list[DomainRule] = [
    DomainRule(
        field="age",
        description="年齡必須為 0～130 之間的整數",
        validator=lambda v: isinstance(v, int) and not isinstance(v, bool) and 0 <= v <= 130,
    ),
    _BOOL_RULE("low_bp",          "低血壓標記"),
    _BOOL_RULE("high_heart_rate", "心跳過速標記"),
    _BOOL_RULE("high_resp_rate",  "呼吸過速標記"),
    _BOOL_RULE("high_temp",       "高燒標記"),
    _BOOL_RULE("low_gcs",         "低格拉斯哥昏迷指數標記"),
    _BOOL_RULE("low_oxygen",      "低血氧標記"),
    _BOOL_RULE("low_urine",       "少尿標記"),
    _BOOL_RULE("high_creatinine", "高肌酸酐標記"),
    _BOOL_RULE("sepsis",          "敗血症標記"),
]

ICU_SPEC = FixtureSpec(
    label="ICU 入住 icu_admission（k=10）",
    path=str(FIXTURES_DIR / "icu_admission.py"),
    expected_k=10,
    func_name="check_icu_admission",
    func_sig="check_icu_admission(age, low_bp, high_heart_rate, high_resp_rate, high_temp, low_gcs, low_oxygen, low_urine, high_creatinine, sepsis)",
    domain_ctx="ICU 急重症入住評估系統",
    domain_types={
        "age":             "int",
        "low_bp":          "bool",
        "high_heart_rate": "bool",
        "high_resp_rate":  "bool",
        "high_temp":       "bool",
        "low_gcs":         "bool",
        "low_oxygen":      "bool",
        "low_urine":       "bool",
        "high_creatinine": "bool",
        "sepsis":          "bool",
    },
    domain_bounds={"age": [0, 130]},
    domain_rules=ICU_RULES,
    max_ifl_iters=55,
    llm_samples=22,   # 略多於 2*k=20
    scenarios=[
        "低血壓合併心跳過速的失血性休克患者",
        "高燒合併呼吸過速的肺炎重症",
        "意識障礙（低 GCS）合併低血氧的腦中風",
        "敗血症合併多器官功能衰竭（少尿+高肌酸酐）",
        "老年患者（≥65）低血壓與低血氧合併",
        "呼吸衰竭（高呼吸速率+低血氧）需插管",
        "急性腎損傷（少尿+高肌酸酐）的中年患者",
        "心跳過速合併高燒的感染性休克",
        "意識清醒但低血氧的 COVID-19 重症",
        "多重生命徵象異常的複雜重症",
        "成年未滿標準（age<18）—不符合 ICU 成人資格",
        "所有生命徵象正常—不需入住 ICU 的低風險案例",
    ],
    func_def="""\
def check_icu_admission(age, low_bp, high_heart_rate, high_resp_rate, high_temp, low_gcs, low_oxygen, low_urine, high_creatinine, sepsis):
    if (
        age >= 18
        and (low_bp or high_heart_rate)
        and (high_resp_rate or high_temp)
        and (low_gcs or low_oxygen)
        and (low_urine or high_creatinine or sepsis)
    ):
        return True
    return False""",
    prompt_fields_json='{"age": int, "low_bp": bool, "high_heart_rate": bool, "high_resp_rate": bool, "high_temp": bool, "low_gcs": bool, "low_oxygen": bool, "low_urine": bool, "high_creatinine": bool, "sepsis": bool}',
)

ALL_SPECS: list[FixtureSpec] = [LOAN_SPEC, SURGERY_SPEC, ICU_SPEC]


# ── 共用工具 ───────────────────────────────────────────────────


def _make_validator(spec: FixtureSpec) -> DomainValidator:
    return DomainValidator(spec.domain_rules)


def _make_ifl_config(spec: FixtureSpec) -> IFLConfig:
    """建立 Fixture 對應的 IFLConfig（覆蓋 func_name、domain_types 等欄位）。"""
    return IFLConfig(
        max_iterations=spec.max_ifl_iters,
        func_name=spec.func_name,
        func_signature=spec.func_sig,
        domain_context=spec.domain_ctx,
        domain_types=spec.domain_types,
        domain_bounds=spec.domain_bounds,
    )


def _make_pure_llm_prompt(spec: FixtureSpec) -> str:
    """產生含隨機情境提示的多樣化純 LLM Prompt，避免固定輸出。"""
    scenario  = _rand.choice(spec.scenarios)
    # 隨機提示值：對每個 int 欄位抽一個提示值
    hints: list[str] = []
    for var, vtype in spec.domain_types.items():
        if vtype == "int":
            lo, hi = spec.domain_bounds.get(var, [0, 1000])
            hints.append(f"參考 {var}：約 {_rand.randint(lo, hi)}")
    hint_str = "；".join(hints) if hints else ""

    return f"""你是一個軟體測試工程師，負責測試{spec.domain_ctx}。

函式定義：
{spec.func_def}

請為以下情境生成一個真實的測試案例：
情境：{scenario}
{hint_str}

要求（欄位型別說明）：
{spec.prompt_fields_json}
（int 欄位請輸出整數，bool 欄位請輸出 true 或 false）

只輸出 JSON 物件，不要任何說明：
{spec.prompt_fields_json}
"""


def _print_cases(cases: list[dict], validator: DomainValidator) -> None:
    """印出案例明細，每筆一行，含來源標記與驗證結果。"""
    if not cases:
        print("    （無案例）")
        return
    for idx, c in enumerate(cases, 1):
        src    = c.get("__source", "?")
        fields = {k: v for k, v in c.items() if not k.startswith("__")}
        if not fields:
            print(f"    T{idx:02d} [{src:<10}] [ERR]  （LLM 呼叫失敗）")
            continue
        vr   = validator.validate(json.dumps(fields))
        mark = "OK" if vr.passed else "NG"
        vals = ", ".join(f"{k}={v}" for k, v in fields.items())
        print(f"    T{idx:02d} [{src:<10}] [{mark}]  {vals}")


# ── 對照組A：純隨機 ─────────────────────────────────────────────


def run_random(spec: FixtureSpec) -> dict:
    """對照組A：max_iterations=0，純隨機初始測試案例。"""
    config    = _make_ifl_config(spec)
    config    = IFLConfig(
        max_iterations=0,
        func_name=spec.func_name,
        func_signature=spec.func_sig,
        domain_context=spec.domain_ctx,
        domain_types=spec.domain_types,
        domain_bounds=spec.domain_bounds,
    )
    validator = _make_validator(spec)
    result    = IFLOrchestrator(config=config).run(spec.path)

    raw_cases = list(result.test_suite)
    clean     = [{k: v for k, v in tc.items() if not k.startswith("__")} for tc in raw_cases]
    valid     = sum(1 for c in clean if validator.validate(json.dumps(c)).passed)
    return {
        "converged":  result.converged,
        "coverage":   result.final_coverage,
        "iterations": result.iteration_count,
        "tokens":     result.total_tokens,
        "total":      len(clean),
        "valid":      valid,
        "valid_ratio": valid / len(clean) if clean else 0.0,
        "cases":      raw_cases,
        "infeasible_paths": [],
        "loss_history":     [],
    }


# ── 對照組B：純 LLM ─────────────────────────────────────────────


def run_pure_llm(spec: FixtureSpec) -> dict:
    """對照組B：直接給 LLM 函式描述，生成 N 個案例後計算 MC/DC 覆蓋率。"""
    nodes = ASTParser().parse_file(spec.path)
    src   = Path(spec.path).read_text(encoding="utf-8")
    inst  = ProbeInjector(nodes).inject(src)

    config    = _make_ifl_config(spec)
    validator = _make_validator(spec)
    sampler   = LLMSampler(config.llm_backend, validator)

    mod_name = f"_pure_llm_{uuid.uuid4().hex[:8]}"
    mod = types.ModuleType(mod_name)
    exec(compile(inst, mod_name, "exec"), mod.__dict__)  # noqa: S102

    log = ProbeLog()
    pi._GLOBAL_LOG = log
    mod.__dict__["_ifl_probe"]           = pi._ifl_probe
    mod.__dict__["_ifl_record_decision"] = pi._ifl_record_decision

    total, valid_count = 0, 0
    collected: list[dict] = []

    for _ in range(spec.llm_samples):
        total += 1
        try:
            case, vr = sampler.sample(_make_pure_llm_prompt(spec))
            status = "llm" if vr.passed else "llm-ng"
            collected.append({**case, "__source": status})
            if vr.passed:
                valid_count += 1
                tid = f"T{uuid.uuid4().hex[:8]}"
                setattr(pi._CURRENT_TEST_ID, "value", tid)
                try:
                    getattr(mod, spec.func_name)(**case)
                except Exception:
                    pass
        except LLMSamplingError:
            collected.append({"__source": "llm-err"})

    engine = MCDCCoverageEngine()
    matrix = engine.build_matrix(nodes[0].condition_set, log)
    tokens = sum(e.get("est_tokens", 0) for e in sampler.token_log)

    return {
        "converged":   matrix.compute_loss() == 0,
        "coverage":    matrix.coverage_ratio,
        "iterations":  total,
        "tokens":      tokens,
        "total":       total,
        "valid":       valid_count,
        "valid_ratio": valid_count / total if total > 0 else 0.0,
        "cases":       collected,
        "infeasible_paths": [],
        "loss_history":     [],
    }


# ── 實驗組：IFL ─────────────────────────────────────────────────


def run_ifl(spec: FixtureSpec) -> dict:
    """實驗組：SMT 導引 + LLM 生成（IFLOrchestrator）。"""
    config    = _make_ifl_config(spec)
    validator = _make_validator(spec)
    result    = IFLOrchestrator(config=config).run(spec.path)

    llm_cases = [
        {k: v for k, v in tc.items() if not k.startswith("__")}
        for tc in result.test_suite
        if tc.get("__source") == "llm"
    ]
    valid = sum(1 for c in llm_cases if validator.validate(json.dumps(c)).passed)
    total = result.iteration_count

    return {
        "converged":        result.converged,
        "coverage":         result.final_coverage,
        "iterations":       result.iteration_count,
        "tokens":           result.total_tokens,
        "total":            total if total > 0 else len(llm_cases),
        "valid":            valid,
        "valid_ratio":      valid / total if total > 0 else 0.0,
        "cases":            list(result.test_suite),
        "infeasible_paths": result.infeasible_paths,
        "loss_history":     result.loss_history,
    }


# ── 通用實驗組執行器 ────────────────────────────────────────────


def run_group(
    name: str,
    spec: FixtureSpec,
    get_result_fn,
) -> list[dict]:
    """執行單組實驗 RUNS 次，印出每輪結果與案例明細，回傳結果列表。"""
    print(f"\n{'='*55}")
    print(name)
    print(f"{'='*55}")
    validator = _make_validator(spec)
    results   = []

    for i in range(RUNS):
        r = get_result_fn()
        ratio     = r["valid_ratio"]
        infeasible = r.get("infeasible_paths", [])
        loss_hist  = r.get("loss_history", [])
        results.append(r)

        print(
            f"Run {i+1}: 收斂={r['converged']}, "
            f"覆蓋率={r['coverage']:.1%}, "
            f"迭代={r['iterations']}, "
            f"有效比={r['valid']}/{r['total']}={ratio:.1%}"
        )
        if infeasible:
            print(f"    [infeasible] {infeasible}")
        if loss_hist:
            print(f"    [loss]       {loss_hist}")
        _print_cases(r.get("cases", []), validator)

    return results


# ── 格式化工具 ──────────────────────────────────────────────────


def _avg(lst: list[dict], key: str) -> float:
    return statistics.mean(r[key] for r in lst)


def _conv(lst: list[dict]) -> float:
    return sum(1 for r in lst if r["converged"]) / len(lst)


def _fmt_pct(lst, key):
    return f"{_avg(lst, key):>12.1%}" if lst is not None else f"{'（略過）':>12}"


def _fmt_f(lst, key):
    return f"{_avg(lst, key):>12.1f}" if lst is not None else f"{'（略過）':>12}"


def _fmt_tok(lst, key):
    return f"{_avg(lst, key):>12.0f}" if lst is not None else f"{'（略過）':>12}"


def _fmt_conv(lst):
    return f"{_conv(lst):>12.1%}" if lst is not None else f"{'（略過）':>12}"


# ══════════════════════════════════════════════════════════════
#  主執行流程
# ══════════════════════════════════════════════════════════════

fixture_results: dict[str, dict[str, list[dict] | None]] = {}

for spec in ALL_SPECS:
    print(f"\n\n{'#'*60}")
    print(f"#  Fixture：{spec.label}")
    print(f"#  預期 k={spec.expected_k}，IFL 最大迭代={spec.max_ifl_iters}")
    print(f"{'#'*60}")

    A = run_group(f"對照組A：純隨機  [{spec.label}]", spec, lambda s=spec: run_random(s))

    if HAS_API_KEY:
        B = run_group(f"對照組B：純LLM   [{spec.label}]", spec, lambda s=spec: run_pure_llm(s))
        C = run_group(f"實驗組：IFL      [{spec.label}]", spec, lambda s=spec: run_ifl(s))
    else:
        B = None
        C = None

    fixture_results[spec.label] = {"A": A, "B": B, "C": C}

    # ── 單一 Fixture 彙整表 ──
    print(f"\n{'='*65}")
    print(f"【{spec.label}】指標彙整（{RUNS} 次平均）")
    print(f"{'='*65}")
    print(f"{'指標':<22} {'對照A 純隨機':>13} {'對照B 純LLM':>13} {'實驗組 IFL':>13}")
    print("-" * 65)
    print(f"{'① MC/DC 覆蓋率':<22} {_fmt_pct(A,'coverage')} {_fmt_pct(B,'coverage')} {_fmt_pct(C,'coverage')}")
    print(f"{'① 收斂率':<22} {_fmt_conv(A)} {_fmt_conv(B)} {_fmt_conv(C)}")
    print(f"{'② 迭代次數':<22} {_fmt_f(A,'iterations')} {_fmt_f(B,'iterations')} {_fmt_f(C,'iterations')}")
    print(f"{'③ 有效生成比':<22} {_fmt_pct(A,'valid_ratio')} {_fmt_pct(B,'valid_ratio')} {_fmt_pct(C,'valid_ratio')}")
    print(f"{'Token 消耗':<22} {_fmt_tok(A,'tokens')} {_fmt_tok(B,'tokens')} {_fmt_tok(C,'tokens')}")

    # ── 單一 Fixture 驗收條件 ──
    print(f"\n{'='*65}")
    print(f"【{spec.label}】驗收條件（實驗組 IFL）")
    print(f"{'='*65}")
    if C is None:
        print("（實驗組略過：未設定 IFL_LLM_API_KEY）")
    else:
        ifl_conv  = _conv(C)
        ifl_cov   = _avg(C, "coverage")
        ifl_valid = _avg(C, "valid_ratio")
        ifl_iter  = _avg(C, "iterations")
        n_pass    = sum(1 for r in C if r["converged"])

        print(f"① MC/DC 覆蓋率 = 100%：{'PASS' if ifl_cov >= 1.0 else 'FAIL'} ({ifl_cov:.1%})")
        print(f"② 迭代次數 <= {spec.max_ifl_iters}：{'PASS' if ifl_iter <= spec.max_ifl_iters else 'FAIL'} ({ifl_iter:.1f} 次)")
        print(f"③ 有效生成比 >= 85%：{'PASS' if ifl_valid >= 0.85 else 'FAIL'} ({ifl_valid:.1%})")
        print(f"   收斂率：{ifl_conv:.1%} ({n_pass}/{RUNS})")


# ══════════════════════════════════════════════════════════════
#  跨 Fixture 比較總表（僅 IFL 組）
# ══════════════════════════════════════════════════════════════

if HAS_API_KEY:
    print(f"\n\n{'#'*70}")
    print(f"#  跨 Fixture 比較總表（實驗組 IFL，{RUNS} 次平均）")
    print(f"{'#'*70}")
    print(f"{'Fixture':<28} {'k':>4} {'覆蓋率':>9} {'收斂率':>9} {'迭代數':>8} {'有效比':>9} {'Tokens':>9}")
    print("-" * 70)

    for spec in ALL_SPECS:
        C = fixture_results[spec.label]["C"]
        if C is None:
            continue
        k   = spec.expected_k
        cov = _avg(C, "coverage")
        con = _conv(C)
        itr = _avg(C, "iterations")
        vld = _avg(C, "valid_ratio")
        tok = _avg(C, "tokens")
        label_short = spec.label.split("（")[0]
        print(
            f"  {label_short:<26} {k:>4} {cov:>8.1%} {con:>8.1%}"
            f" {itr:>8.1f} {vld:>8.1%} {tok:>9.0f}"
        )

    print("-" * 70)
    print("（高 k 值時覆蓋率可能因 infeasible 路徑而低於 100%，詳見上方各輪 [infeasible] 紀錄）")
