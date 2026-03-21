"""
Pairwise Feasibility 與 C2 Constraint 單元測試。

驗證 session 中加入的兩項修正的正確性：
  (D) _build_phi_gap 的配對可行性約束（pairwise feasibility）
  (C2) synthesize_complement 的共用目標變數非目標條件值保持約束

TC-U-80: Pairwise 強制 True 側模型選擇使補集可行的值（b=False）
TC-U-81: Pairwise 使完全被包含條件（subsumed condition）正確回傳 UNSAT
TC-U-82: synthesize_complement——T 側 b=True 時補集 UNSAT 回傳 None
TC-U-83: synthesize_complement——T 側 b=False 時補集 SAT 回傳有效字典
TC-U-84: Pairwise 透過決策表達式傳播約束，限制非目標變數（weight<=70）
TC-U-85: k=9 手術風險——SMTConstraintSynthesizer 可求解至少一個缺口
"""
from __future__ import annotations

from pathlib import Path

import pytest

from ifl_mcdc.layer1.ast_parser import ASTParser
from ifl_mcdc.layer2.smt_synthesizer import SMTConstraintSynthesizer
from ifl_mcdc.models.coverage_matrix import GapEntry

SURGERY_PATH = Path(__file__).parent.parent / "fixtures" / "surgery_risk.py"


def _parse_expr(expr: str):  # type: ignore[no-untyped-def]
    """解析單行 if 語句，回傳第一個 DecisionNode。"""
    parser = ASTParser()
    nodes = parser.parse_source(f"if {expr}: pass\n")
    assert len(nodes) >= 1
    return nodes[0]


def _find_cond(dn, expression: str):  # type: ignore[no-untyped-def]
    """從 DecisionNode 中找到特定表達式的原子條件。"""
    return next(c for c in dn.condition_set.conditions if c.expression == expression)


# ─────────────────────────────────────────────
# TC-U-80：Pairwise 強制 True 側選擇使補集可行的值
# ─────────────────────────────────────────────


def test_pairwise_forces_complement_friendly_model() -> None:
    """TC-U-80：共用變數 a 的 c1(a>=10) T2F 缺口，pairwise feasibility 強制 b=False。

    表達式：(a >= 10 or (a >= 5 and b)) and c
      c1: a>=10（目標，使用 a）
      c2: a>=5（使用 a，共用目標變數）
      c3: b
      c4: c

    分析：
      若 b=True 在 True 側：c2=True（a>=5 AND True），
        補集中 c2 必須保持 True → comp_a>=5，
        decision_comp = OR(comp_a>=10, comp_a>=5 AND True) AND c = True ≠ False → UNSAT。
      若 b=False 在 True 側：c2=False，
        補集中 comp_a 可在 [5,9] → decision_comp = OR(False, False) AND c = False ✓。

    斷言：synthesize 成功（SAT），且 b 的 BoundSpec valid_set={False}。
    """
    dn = _parse_expr("(a >= 10 or (a >= 5 and b)) and c")
    cond_c1 = _find_cond(dn, "a >= 10")

    gap = GapEntry(
        condition_id=cond_c1.cond_id,
        flip_direction="T2F",
        missing_pair_type="unique_cause",
        estimated_difficulty=0.5,
    )

    domain_types = {"a": "int", "b": "bool", "c": "bool"}
    synthesizer = SMTConstraintSynthesizer()
    result = synthesizer.synthesize(dn, gap, domain_types)

    assert result.satisfiable is True, "c1(a>=10) T2F 應可求解（SAT）"
    assert result.bound_specs is not None

    b_spec = next((s for s in result.bound_specs if s.var_name == "b"), None)
    assert b_spec is not None, "bound_specs 應包含 b"
    assert b_spec.valid_set == frozenset({False}), (
        f"Pairwise feasibility 應強制 b=False，使補集可行。"
        f"實際 valid_set={b_spec.valid_set}"
    )


# ─────────────────────────────────────────────
# TC-U-81：完全被包含條件 → Pairwise 使整體 UNSAT
# ─────────────────────────────────────────────


def test_pairwise_subsumed_condition_unsat() -> None:
    """TC-U-81：c1(a>=10) 完全被 c2(a>=5) 包含，pairwise 使 c1 T2F → UNSAT。

    表達式：(a >= 10 or a >= 5) and b
      c1: a>=10（目標，使用 a）
      c2: a>=5（使用 a，c1 的嚴格超集）

    分析：
      a>=10 時 a>=5 必然為 True（被包含）。
      補集需 comp_a<10，且 c2_comp=(comp_a>=5)==(a>=5)=True → comp_a>=5。
      decision_comp = OR(comp_a>=10, comp_a>=5) AND b = True AND b。
      NOT(decision_comp) → NOT(b)，但 True 側需 b=True。
      矛盾 → UNSAT。

    斷言：synthesize 回傳 satisfiable=False（正確識別為不可行缺口）。
    """
    dn = _parse_expr("(a >= 10 or a >= 5) and b")
    cond_c1 = _find_cond(dn, "a >= 10")

    gap = GapEntry(
        condition_id=cond_c1.cond_id,
        flip_direction="T2F",
        missing_pair_type="unique_cause",
        estimated_difficulty=0.5,
    )

    domain_types = {"a": "int", "b": "bool"}
    result = SMTConstraintSynthesizer().synthesize(dn, gap, domain_types)

    assert result.satisfiable is False, (
        "c1(a>=10) 被 c2(a>=5) 完全包含，T2F MC/DC 配對不可行，應回傳 UNSAT"
    )
    assert result.core is not None, "UNSAT 時應有 unsat core"


# ─────────────────────────────────────────────
# TC-U-82：synthesize_complement——T 側 b=True 時 UNSAT → None
# ─────────────────────────────────────────────


def test_complement_returns_none_when_b_true() -> None:
    """TC-U-82：T 側具有 b=True 的具體值，synthesize_complement 應回傳 None。

    表達式：(a >= 10 or (a >= 5 and b)) and c
    T 側具體值：{a: 12, b: True, c: True}

    分析（C2 constraint）：
      c2 = a>=5 在 T 側：12>=5=True → 補集約束 a_comp>=5。
      NOT(f_expr)：NOT(OR(a>=10, a>=5 AND True) AND True) = NOT(a>=5) = a<5。
      a>=5 AND a<5 → UNSAT → 回傳 None。
    """
    dn = _parse_expr("(a >= 10 or (a >= 5 and b)) and c")
    cond_c1 = _find_cond(dn, "a >= 10")

    gap = GapEntry(
        condition_id=cond_c1.cond_id,
        flip_direction="T2F",
        missing_pair_type="unique_cause",
        estimated_difficulty=0.5,
    )

    domain_types = {"a": "int", "b": "bool", "c": "bool"}
    # T 側：b=True，會讓補集中 a_comp>=5 和 a_comp<5 同時成立 → UNSAT
    true_side = {"a": 12, "b": True, "c": True}

    synthesizer = SMTConstraintSynthesizer()
    complement = synthesizer.synthesize_complement(dn, gap, domain_types, true_side)

    assert complement is None, (
        f"T 側 b=True 時補集應 UNSAT 回傳 None，實際 {complement}"
    )


# ─────────────────────────────────────────────
# TC-U-83：synthesize_complement——T 側 b=False 時 SAT → 有效字典
# ─────────────────────────────────────────────


def test_complement_returns_dict_when_b_false() -> None:
    """TC-U-83：T 側具有 b=False 的具體值，synthesize_complement 應回傳有效補集。

    表達式：(a >= 10 or (a >= 5 and b)) and c
    T 側具體值：{a: 12, b: False, c: True}

    分析（C2 constraint）：
      c2 = a>=5 在 T 側：12>=5=True → 補集約束 comp_a>=5。
      NOT(f_expr)：NOT(OR(comp_a>=10, comp_a>=5 AND False) AND True) = comp_a<10。
      comp_a in [5,9] → SAT，decision=False ✓。

    斷言：
      回傳 dict（非 None），且 a（目標變數）< 10（c1=False），b=False、c=True 保持不變。
    """
    dn = _parse_expr("(a >= 10 or (a >= 5 and b)) and c")
    cond_c1 = _find_cond(dn, "a >= 10")

    gap = GapEntry(
        condition_id=cond_c1.cond_id,
        flip_direction="T2F",
        missing_pair_type="unique_cause",
        estimated_difficulty=0.5,
    )

    domain_types = {"a": "int", "b": "bool", "c": "bool"}
    true_side = {"a": 12, "b": False, "c": True}

    synthesizer = SMTConstraintSynthesizer()
    complement = synthesizer.synthesize_complement(dn, gap, domain_types, true_side)

    assert complement is not None, (
        "T 側 b=False 時補集應 SAT，synthesize_complement 不應回傳 None"
    )
    assert isinstance(complement, dict), f"應回傳 dict，得到 {type(complement)}"

    # 目標變數 a 在補集中必須 < 10（c1=False）
    a_comp = complement.get("a")
    assert a_comp is not None, "補集應包含 a 的值"
    assert int(a_comp) < 10, (  # type: ignore[arg-type]
        f"補集 a={a_comp} 應 < 10（c1=a>=10 必須為 False）"
    )

    # 非目標變數 b、c 應與 True 側相同
    assert complement.get("b") is False, (
        f"補集 b 應為 False（與 True 側相同），實際 {complement.get('b')}"
    )
    assert complement.get("c") is True, (
        f"補集 c 應為 True（與 True 側相同），實際 {complement.get('c')}"
    )


# ─────────────────────────────────────────────
# TC-U-84：Pairwise 透過決策表達式傳播，限制非目標變數
# ─────────────────────────────────────────────


def test_pairwise_constrains_nontarget_variable_via_decision() -> None:
    """TC-U-84：c1(age>=65) T2F 缺口，pairwise 透過 f_expr_comp 強制 weight<=70。

    表達式：(age >= 65 or (age >= 18 and weight > 70)) and healthy
      c1: age>=65（目標）
      c2: age>=18（共用 age）
      c3: weight>70（非目標，但影響決策）
      c4: healthy

    分析：
      C2 constraint：c2 在 T 側 True（age>=65→age>=18=True）→ comp_age>=18。
      NOT(decision_comp)：NOT(OR(comp_age>=65, comp_age>=18 AND weight>70) AND healthy)
        = NOT(OR(False, comp_age>=18 AND weight>70)) （comp_age<65, comp_age>=18）
        = NOT(weight>70)
        = weight<=70。
      因此 T 側的 weight（共用原始 Z3 var）被強制 <=70。

    斷言：synthesize 成功，weight 的 BoundSpec interval[0] <= 70。
    """
    dn = _parse_expr("(age >= 65 or (age >= 18 and weight > 70)) and healthy")
    cond_c1 = _find_cond(dn, "age >= 65")

    gap = GapEntry(
        condition_id=cond_c1.cond_id,
        flip_direction="T2F",
        missing_pair_type="unique_cause",
        estimated_difficulty=0.5,
    )

    domain_types = {"age": "int", "weight": "int", "healthy": "bool"}
    synthesizer = SMTConstraintSynthesizer()
    result = synthesizer.synthesize(dn, gap, domain_types)

    assert result.satisfiable is True, (
        "c1(age>=65) T2F 應有可行解（pairwise 選擇 weight<=70）"
    )
    assert result.bound_specs is not None

    weight_spec = next((s for s in result.bound_specs if s.var_name == "weight"), None)
    assert weight_spec is not None, "bound_specs 應包含 weight"
    assert weight_spec.interval is not None, "weight 應有 interval"

    lo, _ = weight_spec.interval
    assert lo <= 70, (
        f"Pairwise 應強制 weight 的 Z3 model 值 <=70（使 c3=weight>70=False，"
        f"讓補集 decision=False 可行）。"
        f"實際 interval 起始={lo}"
    )


# ─────────────────────────────────────────────
# TC-U-85：k=9 手術風險——SMT 可求解至少一個缺口
# ─────────────────────────────────────────────


def test_surgery_risk_k9_smt_sat() -> None:
    """TC-U-85：解析 surgery_risk.py（k=9），SMT 能對第一個缺口（obese F2T）求解。

    surgery_risk 的 OR 子群：(age>=70 or obese)
      cond_obese = "obese"（純 bool，無共用變數）
    布林型 F2T：obese=False → True，對應 NOT_OR_partner(age>=70=False)。

    斷言：synthesize 成功（SAT），solve_time < 10 秒。
    """
    parser = ASTParser()
    nodes = parser.parse_file(str(SURGERY_PATH))
    assert len(nodes) >= 1, f"surgery_risk.py 應有至少 1 個決策節點，實際 {len(nodes)}"
    dn = nodes[0]

    # 選取 obese 條件（純 bool，F2T 最容易）
    cond_obese = next(
        (c for c in dn.condition_set.conditions if c.expression == "obese"),
        None,
    )
    assert cond_obese is not None, (
        f"surgery_risk 應有 obese 條件，實際條件：{[c.expression for c in dn.condition_set.conditions]}"
    )

    gap = GapEntry(
        condition_id=cond_obese.cond_id,
        flip_direction="F2T",
        missing_pair_type="unique_cause",
        estimated_difficulty=0.5,
    )

    domain_types = {
        "age": "int",
        "obese": "bool",
        "has_diabetes": "bool",
        "has_hypertension": "bool",
        "is_smoker": "bool",
        "low_hemoglobin": "bool",
        "low_platelets": "bool",
        "cardiac_history": "bool",
        "has_copd": "bool",
    }

    synthesizer = SMTConstraintSynthesizer()
    result = synthesizer.synthesize(dn, gap, domain_types)

    assert result.satisfiable is True, (
        f"surgery_risk obese F2T 應可求解（SAT），實際 satisfiable={result.satisfiable}"
    )
    assert result.solve_time < 10.0, (
        f"k=9 SMT 應在 10 秒內完成，實際 {result.solve_time:.3f} 秒"
    )
    assert result.bound_specs is not None
    assert len(result.bound_specs) == len(domain_types), (
        f"BoundSpec 數量應等於變數數（{len(domain_types)}），實際 {len(result.bound_specs)}"
    )
