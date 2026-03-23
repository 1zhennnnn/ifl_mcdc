"""
validation_fixtures.py

共用 Fixture 規格定義，供 run_validation_complex.py 與 run_ifl_diversity.py 使用。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ifl_mcdc.models.validation import DomainRule

FIXTURES_DIR = Path("tests/fixtures")


@dataclass
class FixtureSpec:
    """封裝單一 Fixture 的全部設定與驗證規則。"""

    label: str
    path: str
    expected_k: int
    func_name: str
    func_sig: str
    domain_ctx: str
    domain_types: dict[str, str]
    domain_bounds: dict[str, list[int]]
    domain_rules: list[DomainRule]
    max_ifl_iters: int
    llm_samples: int
    scenarios: list[str]
    func_def: str
    prompt_fields_json: str
    fixture_name: str = ""   # 對應 clinical_profiles.json 的 key


# ── 共用 bool 規則產生器 ─────────────────────────────────────────

def _bool_rule(field: str, desc: str) -> DomainRule:
    return DomainRule(
        field=field,
        description=f"{desc}必須為布林值",
        validator=lambda v: isinstance(v, bool),
    )


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
    _bool_rule("employed",          "受僱狀態"),
    _bool_rule("has_collateral",    "抵押品標記"),
    _bool_rule("bankruptcy_history","破產紀錄標記"),
]

LOAN_SPEC = FixtureSpec(
    label="貸款審核 loan_approval（k=6）",
    path=str(FIXTURES_DIR / "loan_approval.py"),
    expected_k=6,
    func_name="check_loan_approval",
    func_sig="check_loan_approval(credit_score, annual_income, loan_amount, employed, has_collateral, bankruptcy_history)",
    domain_ctx="個人信用貸款審核系統",
    domain_types={
        "credit_score":       "int",
        "annual_income":      "int",
        "loan_amount":        "int",
        "employed":           "bool",
        "has_collateral":     "bool",
        "bankruptcy_history": "bool",
    },
    domain_bounds={
        "credit_score":  [300, 850],
        "annual_income": [0, 2_000_000],
        "loan_amount":   [10_000, 10_000_000],
    },
    domain_rules=LOAN_RULES,
    max_ifl_iters=25,
    llm_samples=14,
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
    fixture_name="loan_approval",
)


# ══════════════════════════════════════════════════════════════
#  Fixture ②：手術風險 surgery_risk（k=9）
# ══════════════════════════════════════════════════════════════

SURGERY_RULES: list[DomainRule] = [
    DomainRule(
        field="age",
        description="年齡必須為 0～120 之間的整數",
        validator=lambda v: isinstance(v, int) and not isinstance(v, bool) and 0 <= v <= 120,
    ),
    _bool_rule("obese",            "肥胖標記"),
    _bool_rule("has_diabetes",     "糖尿病標記"),
    _bool_rule("has_hypertension", "高血壓標記"),
    _bool_rule("is_smoker",        "吸菸標記"),
    _bool_rule("low_hemoglobin",   "低血紅素標記"),
    _bool_rule("low_platelets",    "低血小板標記"),
    _bool_rule("cardiac_history",  "心臟病史標記"),
    _bool_rule("has_copd",         "慢阻肺標記"),
]

SURGERY_SPEC = FixtureSpec(
    label="手術風險 surgery_risk（k=9）",
    path=str(FIXTURES_DIR / "surgery_risk.py"),
    expected_k=9,
    func_name="check_surgery_risk",
    func_sig="check_surgery_risk(age, obese, has_diabetes, has_hypertension, is_smoker, low_hemoglobin, low_platelets, cardiac_history, has_copd)",
    domain_ctx="術前手術風險評估系統",
    domain_types={
        "age":              "int",
        "obese":            "bool",
        "has_diabetes":     "bool",
        "has_hypertension": "bool",
        "is_smoker":        "bool",
        "low_hemoglobin":   "bool",
        "low_platelets":    "bool",
        "cardiac_history":  "bool",
        "has_copd":         "bool",
    },
    domain_bounds={"age": [0, 120]},
    domain_rules=SURGERY_RULES,
    max_ifl_iters=45,
    llm_samples=20,
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
    fixture_name="surgery_risk",
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
    _bool_rule("low_bp",           "低血壓標記"),
    _bool_rule("high_heart_rate",  "心跳過速標記"),
    _bool_rule("high_resp_rate",   "呼吸過速標記"),
    _bool_rule("high_temp",        "高燒標記"),
    _bool_rule("low_gcs",          "低格拉斯哥昏迷指數標記"),
    _bool_rule("low_oxygen",       "低血氧標記"),
    _bool_rule("low_urine",        "少尿標記"),
    _bool_rule("high_creatinine",  "高肌酸酐標記"),
    _bool_rule("sepsis",           "敗血症標記"),
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
    llm_samples=22,
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
        "成年未滿標準（age<18）不符合 ICU 成人資格",
        "所有生命徵象正常不需入住 ICU 的低風險案例",
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
    fixture_name="icu_admission",
)

ALL_SPECS: list[FixtureSpec] = [LOAN_SPEC, SURGERY_SPEC, ICU_SPEC]
