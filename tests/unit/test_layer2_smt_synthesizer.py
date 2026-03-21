"""
Layer 2 SMTConstraintSynthesizer / ASTToZ3Converter / BoundExtractor 單元測試。

TC-U-28: 疫苗邏輯 c2 F2T 缺口 SAT 求解（age 在 [18,64]）
TC-U-29: 互斥條件 UNSAT 識別
TC-U-30: 10 秒超時保護
TC-U-32: ASTToZ3Converter——所有算子類型
TC-U-33: 多重比較連鎖（a < b < c）
TC-U-34: BoundExtractor——整數型邊界萃取
TC-U-35: BoundExtractor——布林型合法集合萃取
TC-U-36: BoundSpec 不可為空區間
"""
from __future__ import annotations

import pytest
import z3

from ifl_mcdc.exceptions import Z3TimeoutError
from ifl_mcdc.layer1.ast_parser import ASTParser
from ifl_mcdc.layer2.bound_extractor import BoundExtractor
from ifl_mcdc.layer2.smt_synthesizer import ASTToZ3Converter, SMTConstraintSynthesizer
from ifl_mcdc.models.coverage_matrix import GapEntry


def _parse_expr(expr: str):  # type: ignore[no-untyped-def]
    """解析單行 if 語句，回傳第一個 DecisionNode。"""
    parser = ASTParser()
    nodes = parser.parse_source(f"if {expr}: pass\n")
    assert len(nodes) >= 1
    return nodes[0]


# ─────────────────────────────────────────────
# TC-U-28：疫苗邏輯 c2 F2T 缺口 SAT 求解
# ─────────────────────────────────────────────


def test_vaccine_c2_f2t_sat():  # type: ignore[no-untyped-def]
    """TC-U-28：(age>=65 or age>=18) and not egg_allergy，c2(age>=18) F2T → SAT。

    SMT 應找到 age 在 [18,64] 的解（使 age>=18 為 True，age>=65 為 False）。
    """
    dn = _parse_expr("(age >= 65 or age >= 18) and not egg_allergy")
    synthesizer = SMTConstraintSynthesizer()

    # 找 age>=18 條件
    cond_age18 = next(
        c for c in dn.condition_set.conditions if c.expression == "age >= 18"
    )

    gap = GapEntry(
        condition_id=cond_age18.cond_id,
        flip_direction="F2T",
        missing_pair_type="unique_cause",
        estimated_difficulty=0.5,
    )

    domain_types = {"age": "int", "egg_allergy": "bool"}
    result = synthesizer.synthesize(dn, gap, domain_types)

    assert result.satisfiable is True, "應可求解（SAT）"
    assert result.model is not None
    # age>=18=True（target）AND decision=True — 不再施加 OR 夥伴耦合約束
    assert result.bound_specs is not None
    age_spec = next(
        (s for s in result.bound_specs if s.var_name == "age"), None
    )
    assert age_spec is not None, "bound_specs 應包含 age"
    assert age_spec.interval is not None, "age 應有 interval"
    lo, hi = age_spec.interval
    assert lo < hi, "interval 應為合法區間（lo < hi）"
    # Z3 model 中 age >= 18（目標條件=True），故 midpoint >= 8（val-10 = lo）
    model_age = (lo + hi) / 2
    assert model_age >= 8, (
        f"Z3 model age={model_age:.0f} 應 >= 8（age>=18=True，interval midpoint=val），"
        f"實際 interval=({lo}, {hi})"
    )


# ─────────────────────────────────────────────
# TC-U-29：互斥條件 UNSAT 識別
# ─────────────────────────────────────────────


def test_unsat_mutually_exclusive():  # type: ignore[no-untyped-def]
    """TC-U-29：x > 10 and x < 5 → 恆 False，gap F2T → UNSAT。"""
    dn = _parse_expr("x > 10 and x < 5")
    synthesizer = SMTConstraintSynthesizer()

    # 任意選一個條件的 F2T 缺口
    cond = dn.condition_set.conditions[0]
    gap = GapEntry(
        condition_id=cond.cond_id,
        flip_direction="F2T",
        missing_pair_type="unique_cause",
        estimated_difficulty=0.5,
    )

    domain_types = {"x": "int"}
    result = synthesizer.synthesize(dn, gap, domain_types)

    assert result.satisfiable is False, "互斥條件應 UNSAT"
    assert result.core is not None, "UNSAT 時應有 unsat core"


# ─────────────────────────────────────────────
# TC-U-30：10 秒超時保護
# ─────────────────────────────────────────────


def test_timeout_protection():  # type: ignore[no-untyped-def]
    """TC-U-30：SMTConstraintSynthesizer.TIMEOUT_MS = 10000（10 秒）。

    此測試驗證超時常數已定義（不實際等待超時）。
    """
    assert SMTConstraintSynthesizer.TIMEOUT_MS == 10_000, (
        f"TIMEOUT_MS 應為 10000，實際 {SMTConstraintSynthesizer.TIMEOUT_MS}"
    )


# ─────────────────────────────────────────────
# TC-U-32：ASTToZ3Converter——所有算子類型
# ─────────────────────────────────────────────


def test_ast_to_z3_all_operators():  # type: ignore[no-untyped-def]
    """TC-U-32：ASTToZ3Converter 正確處理 and/or/not/>/</>=/<=/==/!= 算子。"""
    a = z3.Int("a")
    b = z3.Int("b")
    x = z3.Bool("x")
    z3_vars: dict[str, object] = {"a": a, "b": b, "x": x}

    import ast

    converter = ASTToZ3Converter(z3_vars)

    # and
    expr = converter._visit(ast.parse("a > 0 and b < 10", mode="eval").body)
    s = z3.Solver()
    s.add(expr)  # type: ignore[arg-type]
    assert s.check() == z3.sat

    # or
    expr2 = converter._visit(ast.parse("a > 100 or b < 10", mode="eval").body)
    s2 = z3.Solver()
    s2.add(expr2)  # type: ignore[arg-type]
    assert s2.check() == z3.sat

    # not
    expr3 = converter._visit(ast.parse("not x", mode="eval").body)
    s3 = z3.Solver()
    s3.add(expr3)  # type: ignore[arg-type]
    assert s3.check() == z3.sat

    # >= and <=
    expr4 = converter._visit(ast.parse("a >= 5 and b <= 20", mode="eval").body)
    s4 = z3.Solver()
    s4.add(expr4)  # type: ignore[arg-type]
    assert s4.check() == z3.sat

    # == and !=
    expr5 = converter._visit(ast.parse("a == 7 and b != 7", mode="eval").body)
    s5 = z3.Solver()
    s5.add(expr5)  # type: ignore[arg-type]
    assert s5.check() == z3.sat

    # Constant True/False
    expr6 = converter._visit(ast.parse("True", mode="eval").body)
    s6 = z3.Solver()
    s6.add(expr6)  # type: ignore[arg-type]
    assert s6.check() == z3.sat


# ─────────────────────────────────────────────
# TC-U-33：多重比較連鎖（a < b < c）
# ─────────────────────────────────────────────


def test_chained_comparison():  # type: ignore[no-untyped-def]
    """TC-U-33：ASTToZ3Converter 正確處理連鎖比較 a < b < c。

    Python AST 中 a < b < c 是一個 Compare 節點（兩個 ops）。
    應等同於 a < b AND b < c。
    """
    a = z3.Int("a")
    b = z3.Int("b")
    c = z3.Int("c")
    z3_vars: dict[str, object] = {"a": a, "b": b, "c": c}

    import ast

    converter = ASTToZ3Converter(z3_vars)
    expr = converter._visit(ast.parse("a < b < c", mode="eval").body)

    # 驗證 SAT：a=1, b=2, c=3
    s = z3.Solver()
    s.add(expr)  # type: ignore[arg-type]
    assert s.check() == z3.sat

    # 驗證反例 UNSAT：若加上 c <= a，應 UNSAT（a<b<c 且 c<=a 矛盾）
    s2 = z3.Solver()
    s2.add(expr)  # type: ignore[arg-type]
    s2.add(c <= a)  # type: ignore[operator]
    assert s2.check() == z3.unsat, "a<b<c 且 c<=a 應 UNSAT"


# ─────────────────────────────────────────────
# TC-U-34：BoundExtractor——整數型邊界萃取
# ─────────────────────────────────────────────


def test_bound_extractor_int():  # type: ignore[no-untyped-def]
    """TC-U-34：BoundExtractor 對整數型變數萃取 interval=(val, val+10)。"""
    x = z3.Int("x")
    s = z3.Solver()
    s.add(x == 42)
    assert s.check() == z3.sat
    model = s.model()

    extractor = BoundExtractor()
    specs = extractor.extract(model, {"x": x}, {"x": "int"})

    assert len(specs) == 1
    spec = specs[0]
    assert spec.var_name == "x"
    assert spec.var_type == "int"
    assert spec.interval == (42.0, 52.0), f"期望 (42.0, 52.0)，實際 {spec.interval}"
    assert spec.valid_set is None


# ─────────────────────────────────────────────
# TC-U-35：BoundExtractor——布林型合法集合萃取
# ─────────────────────────────────────────────


def test_bound_extractor_bool_true():  # type: ignore[no-untyped-def]
    """TC-U-35：BoundExtractor 對 bool=True 的變數萃取 valid_set={True}。"""
    b = z3.Bool("b")
    s = z3.Solver()
    s.add(b == z3.BoolVal(True))
    assert s.check() == z3.sat
    model = s.model()

    extractor = BoundExtractor()
    specs = extractor.extract(model, {"b": b}, {"b": "bool"})

    assert len(specs) == 1
    spec = specs[0]
    assert spec.var_name == "b"
    assert spec.var_type == "bool"
    assert spec.valid_set == frozenset({True}), f"期望 frozenset({{True}})，實際 {spec.valid_set}"
    assert spec.interval is None


def test_bound_extractor_bool_false():  # type: ignore[no-untyped-def]
    """TC-U-35 補充：BoundExtractor 對 bool=False 的變數萃取 valid_set={False}。"""
    b = z3.Bool("b")
    s = z3.Solver()
    s.add(b == z3.BoolVal(False))
    assert s.check() == z3.sat
    model = s.model()

    extractor = BoundExtractor()
    specs = extractor.extract(model, {"b": b}, {"b": "bool"})

    spec = specs[0]
    assert spec.valid_set == frozenset({False}), f"期望 frozenset({{False}})，實際 {spec.valid_set}"


# ─────────────────────────────────────────────
# TC-U-36：BoundSpec 不可為空區間
# ─────────────────────────────────────────────


def test_bound_spec_nonempty_interval():  # type: ignore[no-untyped-def]
    """TC-U-36：整數型 BoundSpec 的 interval 不可為空（lo < hi）。"""
    x = z3.Int("x")
    s = z3.Solver()
    s.add(x == 0)
    assert s.check() == z3.sat
    model = s.model()

    extractor = BoundExtractor()
    specs = extractor.extract(model, {"x": x}, {"x": "int"})

    spec = specs[0]
    assert spec.interval is not None
    lo, hi = spec.interval
    assert lo < hi, f"interval 不可為空或反向：({lo}, {hi})"


# ─────────────────────────────────────────────
# TC-U-37：BoundExtractor——bool 型 model_val=None（未約束布林）
# ─────────────────────────────────────────────


def test_bound_extractor_bool_unconstrained():  # type: ignore[no-untyped-def]
    """TC-U-37：未約束的 Bool 變數（model[var]=None）→ valid_set={True,False}，確保 LLM 知道須輸出 bool。"""
    b = z3.Bool("b_unconstrained")
    # 建立一個不含 b 的 SAT model（b 未出現在約束中，model[b] 為 None）
    x = z3.Int("x_dummy")
    s = z3.Solver()
    s.add(x == 1)
    assert s.check() == z3.sat
    model = s.model()

    extractor = BoundExtractor()
    specs = extractor.extract(model, {"b": b}, {"b": "bool"})

    assert len(specs) == 1
    spec = specs[0]
    assert spec.var_name == "b"
    assert spec.var_type == "bool"
    assert spec.valid_set == frozenset({True, False}), (
        f"未約束布林應 valid_set={{True, False}}，實際 {spec.valid_set}"
    )
    assert spec.interval is None


# ─────────────────────────────────────────────
# TC-U-38：BoundExtractor——int 型 model_val=None（未約束整數）
# ─────────────────────────────────────────────


def test_bound_extractor_int_unconstrained():  # type: ignore[no-untyped-def]
    """TC-U-38：未約束的 Int 變數（model[var]=None）→ interval=None, valid_set=None。"""
    y = z3.Int("y_unconstrained")
    # 建立一個不含 y 的 SAT model
    x = z3.Int("x_dummy2")
    s = z3.Solver()
    s.add(x == 5)
    assert s.check() == z3.sat
    model = s.model()

    extractor = BoundExtractor()
    specs = extractor.extract(model, {"y": y}, {"y": "int"})

    assert len(specs) == 1
    spec = specs[0]
    assert spec.var_name == "y"
    assert spec.var_type == "int"
    assert spec.interval is None, f"未約束整數應 interval=None，實際 {spec.interval}"
    assert spec.valid_set is None


# ─────────────────────────────────────────────
# TC-U-39：BoundExtractor——int 型使用 domain_bounds
# ─────────────────────────────────────────────


def test_bound_extractor_int_with_domain_bounds():  # type: ignore[no-untyped-def]
    """TC-U-39：domain_bounds 提供時，int 型使用 [model_val, model_val+10] clamped 到 [lo, hi]。"""
    x = z3.Int("age")
    s = z3.Solver()
    s.add(x == 70)
    assert s.check() == z3.sat
    model = s.model()

    extractor = BoundExtractor()
    specs = extractor.extract(
        model,
        {"age": x},
        {"age": "int"},
        domain_bounds={"age": [0, 130]},
    )

    assert len(specs) == 1
    spec = specs[0]
    # model_val=70, interval=[70, 80], clamped to [0,130] → [70, 80]
    assert spec.interval == (70.0, 80.0), (
        f"domain_bounds 提供時應使用 [model_val, model_val+10] clamped，實際 {spec.interval}"
    )


# ─────────────────────────────────────────────
# TC-U-40：BoundExtractor——float 型正常值
# ─────────────────────────────────────────────


def test_bound_extractor_float_normal():  # type: ignore[no-untyped-def]
    """TC-U-40：float 型變數萃取 interval=(val-10, val+10)。"""
    r = z3.Real("r")
    s = z3.Solver()
    s.add(r == z3.RealVal("3.5"))
    assert s.check() == z3.sat
    model = s.model()

    extractor = BoundExtractor()
    specs = extractor.extract(model, {"r": r}, {"r": "float"})

    assert len(specs) == 1
    spec = specs[0]
    assert spec.var_name == "r"
    assert spec.var_type == "float"
    assert spec.interval is not None
    lo, hi = spec.interval
    assert lo < hi, f"float interval 應為合法區間：({lo}, {hi})"
    # midpoint 應接近 3.5
    assert abs((lo + hi) / 2 - 3.5) < 0.1, (
        f"float interval 中點應接近 3.5，實際 ({lo}, {hi})"
    )


# ─────────────────────────────────────────────
# TC-U-41b：BoundExtractor——float 型 model_val=None（未約束）
# ─────────────────────────────────────────────


def test_bound_extractor_float_unconstrained():  # type: ignore[no-untyped-def]
    """TC-U-41b：未約束的 Real 變數（model[var]=None）→ interval=None。"""
    r = z3.Real("r_unconstrained")
    x = z3.Int("x_dummy3")
    s = z3.Solver()
    s.add(x == 7)
    assert s.check() == z3.sat
    model = s.model()

    extractor = BoundExtractor()
    specs = extractor.extract(model, {"r": r}, {"r": "float"})

    assert len(specs) == 1
    spec = specs[0]
    assert spec.var_type == "float"
    assert spec.interval is None, f"未約束 float 應 interval=None，實際 {spec.interval}"
    assert spec.valid_set is None
