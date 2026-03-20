"""
SMT 約束合成器：將 GapEntry 轉化為 Z3 可求解的 SMT 公式。

包含兩個類別：
  ASTToZ3Converter     — 將 Python AST 節點轉換為 Z3 表達式
  SMTConstraintSynthesizer — 建構 Φ_gap 並求解，輸出 SMTResult

TC-U-28: 疫苗邏輯 c2 F2T 缺口 SAT 求解
TC-U-29: 互斥條件 UNSAT 識別
TC-U-30: 10 秒超時保護
TC-U-32: ASTToZ3Converter——所有算子類型
TC-U-33: 多重比較連鎖（a < b < c）
"""
from __future__ import annotations

import ast
import time

import z3
from z3 import Bool, BoolVal, Int, IntVal, Real, RealVal, Solver, sat

from ifl_mcdc.exceptions import Z3TimeoutError
from ifl_mcdc.models.coverage_matrix import GapEntry
from ifl_mcdc.models.decision_node import AtomicCondition, DecisionNode
from ifl_mcdc.models.smt_models import BoundSpec, SMTResult


class ASTToZ3Converter:
    """將 Python AST 節點轉換為 Z3 表達式。

    支援：BoolOp(and/or)、UnaryOp(not)、Compare(>,<,>=,<=,==,!=)、
          Name、Constant（bool/int/float）。
    """

    def __init__(self, z3_vars: dict[str, object]) -> None:
        self.z3_vars = z3_vars

    @staticmethod
    def _make_bool(name: str) -> object:
        """建立具名 Z3 Bool 符號變數。"""
        return Bool(name)

    def convert(self, decision_node: DecisionNode) -> object:
        """轉換整個 DecisionNode 的布林表達式為 Z3 公式。"""
        tree = ast.parse(decision_node.expression_str, mode="eval")
        return self._visit(tree.body)

    def convert_cond(self, cond: AtomicCondition) -> object:
        """轉換單一原子條件的表達式為 Z3 公式（含 negated 旗標）。"""
        tree = ast.parse(cond.expression, mode="eval")
        expr = self._visit(tree.body)
        if cond.negated:
            return z3.Not(expr)
        return expr

    def _visit(self, node: ast.expr) -> object:
        """遞迴走訪 AST 節點，轉換為對應 Z3 表達式。"""
        if isinstance(node, ast.BoolOp):
            operands = [self._visit(v) for v in node.values]
            if isinstance(node.op, ast.And):
                return z3.And(*operands)
            else:
                return z3.Or(*operands)

        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return z3.Not(self._visit(node.operand))

        elif isinstance(node, ast.Compare):
            left = self._visit(node.left)
            parts: list[object] = []
            prev = left
            for op, comp in zip(node.ops, node.comparators):
                right = self._visit(comp)
                if isinstance(op, ast.Gt):
                    parts.append(prev > right)   # type: ignore[operator]
                elif isinstance(op, ast.Lt):
                    parts.append(prev < right)   # type: ignore[operator]
                elif isinstance(op, ast.GtE):
                    parts.append(prev >= right)  # type: ignore[operator]
                elif isinstance(op, ast.LtE):
                    parts.append(prev <= right)  # type: ignore[operator]
                elif isinstance(op, ast.Eq):
                    parts.append(prev == right)
                elif isinstance(op, ast.NotEq):
                    parts.append(prev != right)
                else:
                    raise NotImplementedError(
                        f"Unsupported comparison operator: {type(op).__name__}"
                    )
                prev = right
            return z3.And(*parts) if len(parts) > 1 else parts[0]

        elif isinstance(node, ast.Name):
            if node.id in self.z3_vars:
                return self.z3_vars[node.id]
            if node.id == "True":
                return BoolVal(True)
            if node.id == "False":
                return BoolVal(False)
            raise KeyError(f"Unknown variable: {node.id!r}")

        elif isinstance(node, ast.Constant):
            if isinstance(node.value, bool):
                return BoolVal(node.value)
            if isinstance(node.value, int):
                return IntVal(node.value)
            if isinstance(node.value, float):
                return RealVal(node.value)
            raise TypeError(
                f"Unsupported constant type: {type(node.value).__name__}"
            )

        else:
            raise NotImplementedError(
                f"Unsupported AST node: {type(node).__name__}"
            )


class SMTConstraintSynthesizer:
    """將缺口轉化為 SMT 公式並求解，輸出 SMTResult。

    SAT 時：提供可行輸入向量（model）。
    UNSAT 時：提供不可滿足性核心（core）。
    超時時：拋出 Z3TimeoutError。
    """

    TIMEOUT_MS: int = 10_000  # 10 秒

    def synthesize(
        self,
        decision_node: DecisionNode,
        gap: GapEntry,
        domain_types: dict[str, str],
        domain_bounds: dict[str, list[int]] | None = None,
    ) -> SMTResult:
        """主方法：合成 Φ_gap 並求解。

        Args:
            decision_node: 決策節點。
            gap: 待填補的缺口（condition_id + flip_direction）。
            domain_types: var_name → "int" | "bool" | "float" 的映射。
            domain_bounds: var_name → [min, max]，用於約束 Z3 整數變數範圍。

        Returns:
            SMTResult：SAT 含 model，UNSAT 含 core。

        Raises:
            Z3TimeoutError: 求解超過 TIMEOUT_MS。
        """
        t0 = time.time()

        # 步驟 1：建立 Z3 變數
        z3_vars = self._create_z3_vars(decision_node, domain_types)

        # 步驟 2：將 DecisionNode 轉為 Z3 公式
        f_expr = ASTToZ3Converter(z3_vars).convert(decision_node)

        # 步驟 3：建構 Φ_gap
        phi_gap = self._build_phi_gap(decision_node, gap, z3_vars, f_expr, domain_types, domain_bounds)

        # 步驟 4：求解
        s = Solver()
        s.set("timeout", self.TIMEOUT_MS)
        s.add(phi_gap)

        try:
            result = s.check()
        except z3.Z3Exception as exc:
            raise Z3TimeoutError(
                f"Z3 求解超時（>{self.TIMEOUT_MS}ms）：{exc}"
            ) from exc

        solve_time = time.time() - t0

        # 超時：z3 回傳 unknown
        if result == z3.unknown:
            raise Z3TimeoutError(
                f"Z3 求解超時（>{self.TIMEOUT_MS}ms），結果 unknown"
            )

        if result == sat:
            model = s.model()
            from ifl_mcdc.layer2.bound_extractor import BoundExtractor
            bound_specs = BoundExtractor().extract(model, z3_vars, domain_types, domain_bounds)
            # model_completion=True で domain bounds 制約済み変数も具体値を取得
            concrete: dict[str, object] = {
                var_name: model.eval(z3_var, model_completion=True)
                for var_name, z3_var in z3_vars.items()
            }

            # 求解互補測試（D=False，目標條件=False，非目標變數固定）
            target_cond_comp = next(
                c for c in decision_node.condition_set.conditions
                if c.cond_id == gap.condition_id
            )
            target_z3_comp = ASTToZ3Converter(z3_vars).convert_cond(target_cond_comp)
            target_var_names = set(target_cond_comp.var_names)

            s_comp = Solver()
            s_comp.set("timeout", 2000)
            s_comp.add(f_expr == BoolVal(False))
            s_comp.add(target_z3_comp == BoolVal(False))
            # 非目標變數：固定為主解（域約束由主解已保證）
            # 目標變數：不加域約束，允許超出邊界以實現條件翻轉
            for var_name, z3_var in z3_vars.items():
                if var_name not in target_var_names:
                    model_val = model[z3_var]
                    if model_val is not None:
                        s_comp.add(z3_var == model_val)

            comp_concrete: dict[str, object] | None = None
            try:
                if s_comp.check() == sat:
                    comp_model = s_comp.model()
                    # model_completion=True 確保被 equality 約束的變數也回傳具體值
                    comp_concrete = {
                        var_name: comp_model.eval(z3_var, model_completion=True)
                        for var_name, z3_var in z3_vars.items()
                    }
            except z3.Z3Exception:
                pass

            return SMTResult(True, concrete, bound_specs, None, solve_time, comp_concrete)
        else:
            # UNSAT：取 unsat core
            s2 = Solver()
            s2.set("unsat_core", True)
            tracked: list[str] = []
            for i, clause in enumerate(phi_gap):
                p = Bool(f"_p{i}")
                s2.assert_and_track(clause, p)
                tracked.append(str(p))
            s2.check()
            core = [str(c) for c in s2.unsat_core()]
            return SMTResult(False, None, None, core, solve_time)

    def _create_z3_vars(
        self,
        dn: DecisionNode,
        domain_types: dict[str, str],
    ) -> dict[str, object]:
        """建立所有出現在條件中的 Z3 變數。"""
        z3_vars: dict[str, object] = {}
        for cond in dn.condition_set.conditions:
            for var_name in cond.var_names:
                if var_name in z3_vars:
                    continue
                t = domain_types.get(var_name, "int")
                if t == "bool":
                    z3_vars[var_name] = Bool(var_name)
                elif t == "float":
                    z3_vars[var_name] = Real(var_name)
                else:
                    z3_vars[var_name] = Int(var_name)
        return z3_vars

    def _build_phi_gap(
        self,
        dn: DecisionNode,
        gap: GapEntry,
        z3_vars: dict[str, object],
        f_expr: object,
        domain_types: dict[str, str],
        domain_bounds: dict[str, list[int]] | None = None,
    ) -> list[object]:
        phi: list[object] = []
        converter = ASTToZ3Converter(z3_vars)

        target_cond = next(
            c for c in dn.condition_set.conditions
            if c.cond_id == gap.condition_id
        )

        # (A) 決策結果為 True（找一個有效的覆蓋案例）
        phi.append(f_expr == BoolVal(True))

        # (B) 目標條件的有效值
        # F2T：目標條件為 True（這個案例中目標條件成立）
        # T2F：目標條件為 False（需要另一個案例，此處找 True 的配對）
        # 實際上我們只需要找「目標條件為 True 且決策為 True」的案例
        # 配對案例（目標條件 False）由 coverage_engine 從既有案例配對
        target_z3 = converter.convert_cond(target_cond)
        if gap.flip_direction == "F2T":
            # 找目標條件為 True 的案例
            phi.append(target_z3 == BoolVal(True))
        else:
            # T2F：找目標條件為 True 的案例（作為配對的 True 端）
            phi.append(target_z3 == BoolVal(True))

        # (C) 耦合夥伴約束：只對不共用變數的夥伴施加約束
        target_vars = set(target_cond.var_names)
        for other, coupling_type in dn.condition_set.get_coupled(gap.condition_id):
            # 若夥伴與目標共用變數，跳過（避免矛盾）
            if set(other.var_names) & target_vars:
                continue
            other_z3 = converter.convert_cond(other)
            if coupling_type == "OR":
                phi.append(other_z3 == BoolVal(False))
            else:
                phi.append(other_z3 == BoolVal(True))

        # (D) int 變數域約束：優先使用 domain_bounds，否則僅約束 >= 0
        for var_name, z3_var in z3_vars.items():
            if domain_types.get(var_name, "int") == "int":
                if domain_bounds and var_name in domain_bounds:
                    lo, hi = domain_bounds[var_name][0], domain_bounds[var_name][1]
                    phi.append(z3_var >= lo)  # type: ignore[operator]
                    phi.append(z3_var <= hi)  # type: ignore[operator]
                else:
                    phi.append(z3_var >= 0)  # type: ignore[operator]

        return phi