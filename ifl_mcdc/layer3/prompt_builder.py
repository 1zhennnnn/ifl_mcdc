"""
提示詞建構器：產生四段式 Gap-Guided Prompt。

TC-U-37: bound_specs 全部出現在 §3
TC-U-38: 超過 MAX_TOKENS 時截斷 §1 的 source_context
TC-U-39: §3 和 §4 完整保留（截斷後）
TC-U-40: F2T/T2F 方向描述正確
"""
from __future__ import annotations

import json

from ifl_mcdc.models.coverage_matrix import GapEntry
from ifl_mcdc.models.decision_node import DecisionNode
from ifl_mcdc.models.smt_models import BoundSpec


class PromptConstructor:
    """建構四段式 Gap-Guided Prompt。

    §1 醫療情境 → §2 目標缺口 → §3 精確數值約束 → §4 輸出格式
    """

    MAX_TOKENS: int = 2048

    def build(
        self,
        decision_node: DecisionNode,
        gap: GapEntry,
        bound_specs: list[BoundSpec],
        func_signature: str,
        domain_context: str = "",
    ) -> str:
        """建構四段式提示詞。

        截斷策略：若超過 MAX_TOKENS（以字元數估算），從 §1 的 source_context
        開始截斷，§3 和 §4 必須完整保留。

        Args:
            decision_node: 決策節點（含 source_context）。
            gap: 目標缺口。
            bound_specs: 變數邊界規格列表。
            func_signature: 函式簽名字串。
            domain_context: 選填的情境說明。

        Returns:
            四段式提示詞字串。
        """
        # 找出 gap.condition_id 對應的條件表達式
        cond_expression = ""
        for cond in decision_node.condition_set.conditions:
            if cond.cond_id == gap.condition_id:
                cond_expression = cond.expression
                break

        direction_str = "True" if gap.flip_direction == "F2T" else "False"

        # §3 固定部分（語意引導版）
        lines: list[str] = ["【臨床情境引導】", "請根據以下邏輯約束，生成一個符合真實醫療情境的測試案例：", ""]
        for bs in bound_specs:
            if bs.interval is not None:
                lo, hi = bs.interval
                lo_str = str(int(lo)) if lo == int(lo) else str(lo)
                hi_str = str(int(hi)) if hi == int(hi) else str(hi)
                unit = f"（單位：{bs.medical_unit}）" if bs.medical_unit else ""
                lines.append(
                    f"- {bs.var_name}：必須在 {lo_str} 到 {hi_str} 之間{unit}，"
                    "請選擇一個臨床上合理的具體數值"
                )
            elif bs.valid_set is not None:
                vals = sorted(bs.valid_set, key=str)
                if len(vals) == 1:
                    lines.append(f"- {bs.var_name}：必須為 {vals[0]}")
                else:
                    lines.append(f"- {bs.var_name}：必須為 {vals} 之一")
        lines += [
            "",
            "【語意要求】",
            f"請想像一個真實的{domain_context}場景，確保：",
            "1. 數值符合真實臨床範圍（例如年齡不應為 0 或極端值）",
            "2. 各參數之間有合理的臨床關聯性",
            "3. 整體案例代表一個真實可能存在的病患情境",
            "",
            "【臨床語意要求】",
            "生成的測試案例必須符合真實醫療情境：",
            "- 年齡請選擇臨床上常見的患者年齡（建議 18~85 歲之間）",
            "- 各參數之間應有合理的臨床關聯性",
            "- 避免使用極端值（如 age=0, age=130, days=9999）",
            "- 想像這是一位真實存在的病患",
        ]
        sec3 = "\n".join(lines)

        # §4 固定部分
        example_json = json.dumps(
            {bs.var_name: "..." for bs in bound_specs}, ensure_ascii=False
        )
        sec4 = (
            "【輸出格式】\n"
            "請僅輸出一個合法的 JSON 物件，鍵名與函式參數完全一致。\n"
            "禁止 markdown、禁止說明文字、禁止 ```json 標記。\n"
            f"{example_json}"
        )

        # §2 固定部分
        sec2 = (
            f"【目標缺口】\n"
            f"條件 {gap.condition_id}（{cond_expression}）需要值為\n"
            f"{direction_str}，且整體函式輸出因此改變。"
        )

        # §1 可截斷部分
        source_context = decision_node.source_context

        def _build_full(ctx: str) -> str:
            sec1 = (
                f"【醫療情境】\n"
                f"函式：{func_signature}\n"
                f"情境：{domain_context}\n"
                f"原始碼：\n"
                f"{ctx}"
            )
            return "\n\n".join([sec1, sec2, sec3, sec4])

        full = _build_full(source_context)

        # 截斷：若超過 MAX_TOKENS，逐步縮短 source_context
        if len(full) > self.MAX_TOKENS:
            # 計算固定部分長度（不含 §1 source_context）
            sec1_header = (
                f"【醫療情境】\n"
                f"函式：{func_signature}\n"
                f"情境：{domain_context}\n"
                f"原始碼：\n"
            )
            fixed_len = len(sec1_header) + len("\n\n") * 3 + len(sec2) + len(sec3) + len(sec4)
            budget = self.MAX_TOKENS - fixed_len

            if budget > 0:
                source_context = source_context[:budget]
            else:
                source_context = ""

            full = _build_full(source_context)

        return full
