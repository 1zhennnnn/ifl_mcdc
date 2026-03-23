"""
提示詞建構器：產生四段式 Gap-Guided Prompt。

TC-U-37: bound_specs 全部出現在 §3
TC-U-38: 超過 MAX_TOKENS 時截斷 §1 的 source_context
TC-U-39: §3 和 §4 完整保留（截斷後）
TC-U-40: F2T/T2F 方向描述正確
"""
from __future__ import annotations

import json
import random as _random
from typing import Any

from ifl_mcdc.models.coverage_matrix import GapEntry
from ifl_mcdc.models.decision_node import DecisionNode
from ifl_mcdc.models.smt_models import BoundSpec

_ZONE_LABELS = ["邊界區（靠近臨界點）", "中間區（可行空間中段）", "極端區（可行空間遠端）"]


class PromptConstructor:
    """建構四段式 Gap-Guided Prompt。

    §1 醫療情境 → §2 目標缺口 → §3 精確數值約束 → §4 輸出格式
    """

    MAX_TOKENS: int = 2048

    def __init__(self) -> None:
        self._call_count: int = 0  # 用於系統性循環子區間（邊界→中間→極端→…）

    def build(
        self,
        decision_node: DecisionNode,
        gap: GapEntry,
        bound_specs: list[BoundSpec],
        func_signature: str,
        domain_context: str = "",
        clinical_profile: dict[str, Any] | None = None,
        scenario_hint: str = "",
        domain_types: dict[str, str] | None = None,
        domain_bounds: dict[str, list[int]] | None = None,
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
        # 更新呼叫計數（用於系統性循環子區間）
        self._call_count += 1

        # 找出 gap.condition_id 對應的條件表達式
        cond_expression = ""
        for cond in decision_node.condition_set.conditions:
            if cond.cond_id == gap.condition_id:
                cond_expression = cond.expression
                break

        direction_str = "True" if gap.flip_direction == "F2T" else "False"

        # §diversity（多樣性要求，通用版）
        sec_diversity_lines = [
            "【多樣性要求】",
            f"請生成一個在「{domain_context}」情境上自然多樣的測試案例。",
            "- 數值型參數請在符合約束的範圍內自由選擇，避免每次使用相同數字",
            "- 布林型參數的組合應反映真實的個體差異",
            "- 整體數值組合應代表真實存在的不同個體",
        ]
        if scenario_hint:
            sec_diversity_lines.append(f"- 此次請以「{scenario_hint}」為情境背景")
        sec_diversity = "\n".join(sec_diversity_lines)

        # §3 固定部分（語意引導版）
        # 子區間選擇：用呼叫計數循環（邊界→中間→極端→…），確保系統性覆蓋
        zone_idx = self._call_count % 3

        lines: list[str] = ["【約束條件】", "請根據以下約束，生成一個符合情境的具體案例：", ""]
        for bs in bound_specs:
            if bs.interval is not None:
                if bs.sub_intervals is not None:
                    lo, hi = bs.sub_intervals[zone_idx % len(bs.sub_intervals)]
                    label = _ZONE_LABELS[zone_idx % len(_ZONE_LABELS)]
                    lo_str = str(int(lo)) if lo == int(lo) else str(lo)
                    hi_str = str(int(hi)) if hi == int(hi) else str(hi)
                    unit = f"（單位：{bs.medical_unit}）" if bs.medical_unit else ""
                    lines.append(
                        f"- {bs.var_name}：建議在{label} {lo_str} 到 {hi_str} 之間{unit}"
                    )
                else:
                    lo, hi = bs.interval
                    lo_str = str(int(lo)) if lo == int(lo) else str(lo)
                    hi_str = str(int(hi)) if hi == int(hi) else str(hi)
                    unit = f"（單位：{bs.medical_unit}）" if bs.medical_unit else ""
                    lines.append(f"- {bs.var_name}：必須在 {lo_str} 到 {hi_str} 之間{unit}")
            elif bs.valid_set is not None:
                vals = sorted(bs.valid_set, key=str)
                if len(vals) == 1:
                    lines.append(f"- {bs.var_name}：必須為 {vals[0]}")
                else:
                    lines.append(f"- {bs.var_name}：必須為 {vals} 之一")
        lines += [
            "",
            "【情境語意要求】",
            f"請想像一個真實的「{domain_context}」場景，確保：",
            "1. 所有數值符合真實可能的範圍，避免極端或不合理的組合",
            "2. 各參數之間有合理的邏輯關聯性",
            "3. 整體案例代表一個真實存在的個體",
        ]
        sec3 = "\n".join(lines)

        # §4 固定部分（含型別與值域提醒）
        example_json = json.dumps(
            {bs.var_name: "..." for bs in bound_specs}, ensure_ascii=False
        )
        type_hint_lines: list[str] = []
        if domain_types:
            int_fields = [k for k, v in domain_types.items() if v == "int"]
            bool_fields = [k for k, v in domain_types.items() if v == "bool"]
            if int_fields:
                int_descs = []
                for k in int_fields:
                    if domain_bounds and k in domain_bounds:
                        lo, hi = domain_bounds[k][0], domain_bounds[k][1]
                        int_descs.append(f"{k}（{lo}~{hi}）")
                    else:
                        int_descs.append(k)
                type_hint_lines.append("整數欄位：" + "、".join(int_descs))
            if bool_fields:
                type_hint_lines.append(
                    "布林欄位（必須用 true/false，不得使用 1/0 或字串）："
                    + "、".join(bool_fields)
                )
        type_hint_str = ("\n" + "\n".join(type_hint_lines)) if type_hint_lines else ""
        sec4 = (
            "【輸出格式】\n"
            "請僅輸出一個合法的 JSON 物件，鍵名與函式參數完全一致。\n"
            "禁止 markdown、禁止說明文字、禁止 ```json 標記。\n"
            f"{example_json}"
            f"{type_hint_str}"
        )

        # §2 固定部分
        sec2 = (
            f"【目標缺口】\n"
            f"條件 {gap.condition_id}（{cond_expression}）需要值為\n"
            f"{direction_str}，且整體函式輸出因此改變。"
        )

        # §1 可截斷部分
        source_context = decision_node.source_context

        # §clinical（臨床比例參考段落，選填）
        sec_clinical: str | None = None
        if clinical_profile is not None:
            sec_clinical = _build_clinical_section(clinical_profile)

        def _build_full(ctx: str) -> str:
            sec1 = (
                f"【醫療情境】\n"
                f"函式：{func_signature}\n"
                f"情境：{domain_context}\n"
                f"原始碼：\n"
                f"{ctx}"
            )
            parts = [sec1, sec2, sec_diversity]
            if sec_clinical:
                parts.append(sec_clinical)
            parts += [sec3, sec4]
            return "\n\n".join(parts)

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
            clinical_len = (len("\n\n") + len(sec_clinical)) if sec_clinical else 0
            fixed_len = len(sec1_header) + len("\n\n") * 4 + len(sec2) + len(sec_diversity) + clinical_len + len(sec3) + len(sec4)
            budget = self.MAX_TOKENS - fixed_len

            if budget > 0:
                source_context = source_context[:budget]
            else:
                source_context = ""

            full = _build_full(source_context)

        return full


def _build_clinical_section(profile: dict[str, Any]) -> str:
    """將臨床比例資料轉化為自然語言的 Prompt 段落。"""
    lines: list[str] = ["【臨床流行病學參考資料】", "以下為真實臨床盛行率，僅供參考，不是硬性約束；邊界測試案例仍然可以生成。", ""]

    population = profile.get("population", "")
    if population:
        lines.append(f"族群背景：{population}")
        lines.append("")

    variables = profile.get("variables", {})
    if variables:
        lines.append("各變數盛行率：")
        for var_name, info in variables.items():
            if isinstance(info, dict):
                description = info.get("description", "")
                if description:
                    lines.append(f"- {var_name}：{description}")
            elif isinstance(info, str):
                lines.append(f"- {var_name}：{info}")

    comorbidities = profile.get("comorbidities", "")
    if comorbidities:
        lines.append("")
        lines.append(f"共病說明：{comorbidities}")

    lines.append("")
    lines.append("請盡量讓生成的案例反映上述分布，但優先遵守邏輯約束。")
    return "\n".join(lines)
