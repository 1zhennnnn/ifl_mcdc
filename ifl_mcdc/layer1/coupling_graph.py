"""
耦合圖建構器：從 BoolOp AST 結構建立 k×k 耦合鄰接矩陣。

TC-U-05: OR 強耦合正確識別
TC-U-06: NOT 否定條件的 negated 標記（不影響耦合類型）
TC-U-07: 四條件完整耦合矩陣驗證
TC-U-08: 耦合矩陣 JSON 序列化
"""
from __future__ import annotations

import ast

from ifl_mcdc.models.decision_node import AtomicCondition


class CouplingGraphBuilder:
    """從 BoolOp AST 結構建構 k×k 耦合鄰接矩陣。

    matrix[i][j] = "OR"  → 條件 i 和 j 共享 or 算子
    matrix[i][j] = "AND" → 條件 i 和 j 共享 and 算子
    matrix[i][j] = None  → 無直接耦合

    若同一對已存在耦合類型，優先保留 OR（遮罩效應更強）。
    """

    def build(
        self,
        decision_id: str,
        root_expr: ast.expr,
        conditions: list[AtomicCondition],
    ) -> list[list[str | None]]:
        """建構耦合矩陣。

        Args:
            decision_id: 決策節點 ID（僅供除錯）。
            root_expr: 決策節點的測試表達式 AST。
            conditions: 已提取的原子條件列表（順序與 cond_id 一致）。

        Returns:
            k×k 對稱矩陣，值為 "OR" / "AND" / None。
        """
        k = len(conditions)
        matrix: list[list[str | None]] = [[None] * k for _ in range(k)]

        # 建立 id(ast_node) → 條件索引 的映射
        node_to_idx: dict[int, int] = {
            id(c.ast_node): i for i, c in enumerate(conditions) if c.ast_node is not None
        }

        def _set_coupling(a: int, b: int, op_type: str) -> None:
            """設定耦合，OR 優先保留。"""
            if a == b:
                return
            existing = matrix[a][b]
            if existing == "OR":
                return  # OR 優先，不覆蓋
            matrix[a][b] = op_type
            matrix[b][a] = op_type

        def _collect_direct_leaves(expr: ast.expr) -> list[int]:
            """收集表達式的直接原子葉子索引（穿透 UnaryOp(Not) 和巢狀 BoolOp）。"""
            indices: list[int] = []

            def _walk(e: ast.expr) -> None:
                if isinstance(e, ast.BoolOp):
                    # 巢狀 BoolOp：遞迴收集其葉子（視為此層的葉子）
                    for v in e.values:
                        _walk(v)
                elif isinstance(e, ast.UnaryOp) and isinstance(e.op, ast.Not):
                    _walk(e.operand)
                else:
                    idx = node_to_idx.get(id(e))
                    if idx is not None:
                        indices.append(idx)

            _walk(expr)
            return indices

        def _fill_group(bool_op: ast.BoolOp, op_type: str) -> None:
            """收集此 BoolOp 直接子節點的葉子，兩兩設定耦合。"""
            leaf_indices: list[int] = []
            for value in bool_op.values:
                leaf_indices.extend(_collect_direct_leaves(value))
            for a in leaf_indices:
                for b in leaf_indices:
                    if a != b:
                        _set_coupling(a, b, op_type)

        def _traverse(expr: ast.expr) -> None:
            if isinstance(expr, ast.BoolOp):
                op_type = "OR" if isinstance(expr.op, ast.Or) else "AND"
                _fill_group(expr, op_type)
                for v in expr.values:
                    _traverse(v)  # 遞迴處理巢狀 BoolOp
            elif isinstance(expr, ast.UnaryOp):
                _traverse(expr.operand)

        _traverse(root_expr)
        return matrix
