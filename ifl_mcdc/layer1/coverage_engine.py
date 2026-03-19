"""
MC/DC 覆蓋率引擎：從 ProbeLog 建立並增量更新 MCDCMatrix。
"""
from __future__ import annotations

from ifl_mcdc.models.coverage_matrix import MCDCMatrix
from ifl_mcdc.models.decision_node import ConditionSet
from ifl_mcdc.models.probe_record import ProbeLog, ProbeRecord


class MCDCCoverageEngine:

    def build_matrix(
        self,
        cond_set: ConditionSet,
        log: ProbeLog,
    ) -> MCDCMatrix:
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
        loss_before = matrix.compute_loss()
        self._update_one(matrix, log, new_test_id)
        return matrix.compute_loss() < loss_before

    def _update_one(
        self,
        matrix: MCDCMatrix,
        log: ProbeLog,
        test_id: str,
    ) -> None:
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
        map_a: dict[str, ProbeRecord] = {r.cond_id: r for r in recs_a}
        map_b: dict[str, ProbeRecord] = {r.cond_id: r for r in recs_b}

        if not map_a or not map_b:
            return

        dec_a = next(iter(map_a.values())).decision
        dec_b = next(iter(map_b.values())).decision

        if dec_a == dec_b:
            return

        for cond in matrix.condition_set.conditions:
            rec_a = map_a.get(cond.cond_id)
            rec_b = map_b.get(cond.cond_id)

            if rec_a is None or rec_b is None:
                continue

            if rec_a.value == rec_b.value:
                continue

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
        """MC/DC 唯一因 variant：所有非目標條件的探針值在兩個測試中必須相同。

        原始探針值相同即保證有效值（含 negated 轉換）相同，
        確保決策翻轉唯一由目標條件引起。
        """
        for cond in matrix.condition_set.conditions:
            if cond.cond_id == target_id:
                continue

            rec_a = map_a.get(cond.cond_id)
            rec_b = map_b.get(cond.cond_id)

            if rec_a is None or rec_b is None:
                continue

            if rec_a.value != rec_b.value:
                return False

        return True