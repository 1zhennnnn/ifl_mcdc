"""
MC/DC 覆蓋率引擎：從 ProbeLog 建立並增量更新 MCDCMatrix。

獨立對的三個判斷條件必須同時成立：
  1. 條件翻轉：cᵢ 在 Tⱼ 中為 False，在 Tₖ 中為 True（或反向）
  2. 決策翻轉：D(Tⱼ) ≠ D(Tₖ)
  3. 其他條件無干擾（根據耦合矩陣判定）

TC-U-14: 損失函數初始值計算
TC-U-15: 有效獨立對被正確識別
TC-U-16: 遮罩條件不被誤計為獨立對
TC-U-17: 增量更新 L(X) 遞減
TC-U-18: 100% 覆蓋後 L(X)=0
"""
from __future__ import annotations

from ifl_mcdc.models.coverage_matrix import MCDCMatrix
from ifl_mcdc.models.decision_node import ConditionSet
from ifl_mcdc.models.probe_record import ProbeLog, ProbeRecord


class MCDCCoverageEngine:
    """從 ProbeLog 建立並增量更新 MC/DC 覆蓋率矩陣。"""

    def build_matrix(
        self,
        cond_set: ConditionSet,
        log: ProbeLog,
    ) -> MCDCMatrix:
        """從 ProbeLog 建立初始 MCDCMatrix。

        Args:
            cond_set: 決策節點的條件集合。
            log: 探針觀測日誌。

        Returns:
            初始化的 MCDCMatrix。
        """
        matrix = MCDCMatrix(condition_set=cond_set)
        test_ids = list(dict.fromkeys(r.test_id for r in log.records))
        for test_id in test_ids:
            self._update_one(matrix, log, test_id)
        return matrix

    def update(
        self,
        matrix: MCDCMatrix,
        log: ProbeLog,
        new_test_id: str,
    ) -> bool:
        """增量更新：比較 new_test_id 與現有所有 test_id。

        Args:
            matrix: 現有覆蓋率矩陣（原地修改）。
            log: 探針觀測日誌（含 new_test_id 的記錄）。
            new_test_id: 新加入的測試案例 ID。

        Returns:
            True 若 compute_loss() 因此次更新而降低。
        """
        loss_before = matrix.compute_loss()
        self._update_one(matrix, log, new_test_id)
        return matrix.compute_loss() < loss_before

    def _update_one(
        self,
        matrix: MCDCMatrix,
        log: ProbeLog,
        test_id: str,
    ) -> None:
        """將 test_id 與已存在的所有其他測試比較，更新矩陣。"""
        new_records = log.get_by_test(test_id)
        existing_tests = list(
            dict.fromkeys(
                r.test_id for r in log.records if r.test_id != test_id
            )
        )
        for existing_id in existing_tests:
            existing_records = log.get_by_test(existing_id)
            self._check_pair(matrix, new_records, existing_records)

    def _check_pair(
        self,
        matrix: MCDCMatrix,
        recs_a: list[ProbeRecord],
        recs_b: list[ProbeRecord],
    ) -> None:
        """檢查測試對 (A, B) 是否構成任一條件的有效獨立對。

        條件：
          1. 某條件翻轉
          2. 決策翻轉
          3. _others_ok() 通過（耦合規則）

        若構成，呼叫 matrix.mark_covered() 標記兩個方向。
        """
        map_a: dict[str, ProbeRecord] = {r.cond_id: r for r in recs_a}
        map_b: dict[str, ProbeRecord] = {r.cond_id: r for r in recs_b}

        if not map_a or not map_b:
            return

        # 取決策結果（以第一筆記錄的 decision 欄位為準）
        dec_a = next(iter(map_a.values())).decision
        dec_b = next(iter(map_b.values())).decision

        if dec_a == dec_b:
            return  # 決策未翻轉，此對無法貢獻任何獨立對

        for cond in matrix.condition_set.conditions:
            rec_a = map_a.get(cond.cond_id)
            rec_b = map_b.get(cond.cond_id)

            if rec_a is None or rec_b is None:
                continue

            if rec_a.value == rec_b.value:
                continue  # 此條件未翻轉

            # 條件翻轉 + 決策翻轉 → 潛在獨立對，再檢查其他條件
            if self._others_ok(matrix, cond.cond_id, map_a, map_b):
                flip = "F2T" if (not rec_a.value and rec_b.value) else "T2F"
                matrix.mark_covered(cond.cond_id, flip)
                other_flip = "T2F" if flip == "F2T" else "F2T"
                matrix.mark_covered(cond.cond_id, other_flip)

    def _others_ok(
        self,
        matrix: MCDCMatrix,
        target_id: str,
        map_a: dict[str, ProbeRecord],
        map_b: dict[str, ProbeRecord],
    ) -> bool:
        """檢查除 target 以外的其他條件是否允許此獨立對成立。

        根據耦合類型：
        - OR 耦合的夥伴：兩個測試中都必須為 False（消除遮罩）
        - AND 耦合的夥伴：兩個測試中都必須為 True（確保外層 and 不遮蔽）
        - 無耦合（None）：兩個測試中值相同即可

        Args:
            matrix: 覆蓋率矩陣（含耦合資訊）。
            target_id: 正在檢驗的目標條件 ID。
            map_a: 測試 A 的條件 ID → ProbeRecord 映射。
            map_b: 測試 B 的條件 ID → ProbeRecord 映射。

        Returns:
            True 若所有其他條件均符合耦合規則。
        """
        coupled = matrix.condition_set.get_coupled(target_id)
        for other_cond, coupling_type in coupled:
            rec_a = map_a.get(other_cond.cond_id)
            rec_b = map_b.get(other_cond.cond_id)

            if rec_a is None or rec_b is None:
                continue

            if coupling_type == "OR":
                # 兩者都必須為 False，才能確保 target 的翻轉不被 OR 夥伴遮罩
                if rec_a.value or rec_b.value:
                    return False
            elif coupling_type == "AND":
                # 兩者都必須為 True，才能確保 target 的翻轉不被 AND 短路
                if not rec_a.value or not rec_b.value:
                    return False
            else:
                # 無耦合：兩個測試中值相同即可
                if rec_a.value != rec_b.value:
                    return False

        return True
