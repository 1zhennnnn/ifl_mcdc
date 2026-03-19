"""
Layer 3 AcceptanceGate 單元測試。

TC-U-52: L(X) 降低時接受（回傳 True）
TC-U-53: L(X) 不變時拒絕（回傳 False）
"""
from __future__ import annotations

from ifl_mcdc.layer1.ast_parser import ASTParser
from ifl_mcdc.layer1.coverage_engine import MCDCCoverageEngine
from ifl_mcdc.layer3.acceptance_gate import AcceptanceGate
from ifl_mcdc.models.probe_record import ProbeLog, ProbeRecord


def _make_log_and_matrix():  # type: ignore[no-untyped-def]
    """建立 (c1 or c2) and c3 and c4 的初始空矩陣和空日誌。"""
    parser = ASTParser()
    nodes = parser.parse_source("if (c1 or c2) and c3 and c4: pass\n")
    dn = nodes[0]
    cond_set = dn.condition_set
    engine = MCDCCoverageEngine()
    log = ProbeLog()
    matrix = engine.build_matrix(cond_set, log)
    return matrix, log, cond_set, engine


# ─────────────────────────────────────────────
# TC-U-52：L(X) 降低時接受
# ─────────────────────────────────────────────


def test_accept_when_loss_decreases():  # type: ignore[no-untyped-def]
    """TC-U-52：新測試案例提供有效獨立對 → L(X) 降低 → evaluate 回傳 True。

    設計：
      c1=F, c2=F, c3=T, c4=T → 決策 False（F）
      c1=T, c2=F, c3=T, c4=T → 決策 True（T）
    兩者差異：c1 翻轉，c2=F（OR partner），c3=c4=T（AND partners）→ 構成 c1 的有效獨立對
    """
    matrix, log, cond_set, engine = _make_log_and_matrix()
    gate = AcceptanceGate(engine=engine)

    # 取出條件 ID
    ids = {c.expression: c.cond_id for c in cond_set.conditions}
    c1_id = ids["c1"]
    c2_id = ids["c2"]
    c3_id = ids["c3"]
    c4_id = ids["c4"]

    # 測試案例 T1：c1=F, c2=F, c3=T, c4=T → decision=False
    t1 = "T1"
    for cid, val in [(c1_id, False), (c2_id, False), (c3_id, True), (c4_id, True)]:
        log.append(ProbeRecord(test_id=t1, cond_id=cid, value=val, decision=False))

    # 先用 T1 建立基準
    engine.update(matrix, log, t1)
    loss_before = matrix.compute_loss()

    # 測試案例 T2：c1=T, c2=F, c3=T, c4=T → decision=True
    t2 = "T2"
    for cid, val in [(c1_id, True), (c2_id, False), (c3_id, True), (c4_id, True)]:
        log.append(ProbeRecord(test_id=t2, cond_id=cid, value=val, decision=True))

    accepted = gate.evaluate(matrix, log, t2)

    assert accepted is True, f"應接受（L(X) 應降低）。更新前 loss={loss_before}，更新後 {matrix.compute_loss()}"


# ─────────────────────────────────────────────
# TC-U-53：L(X) 不變時拒絕
# ─────────────────────────────────────────────


def test_reject_when_loss_unchanged():  # type: ignore[no-untyped-def]
    """TC-U-53：新測試案例不構成任何獨立對 → L(X) 不變 → evaluate 回傳 False。

    設計：
      T1 和 T2 的 c1 都相同（都是 False），不構成任何翻轉對。
    """
    matrix, log, cond_set, engine = _make_log_and_matrix()
    gate = AcceptanceGate(engine=engine)

    ids = {c.expression: c.cond_id for c in cond_set.conditions}
    c1_id = ids["c1"]
    c2_id = ids["c2"]
    c3_id = ids["c3"]
    c4_id = ids["c4"]

    # T1：c1=F, c2=F, c3=T, c4=T → decision=False
    t1 = "T1"
    for cid, val in [(c1_id, False), (c2_id, False), (c3_id, True), (c4_id, True)]:
        log.append(ProbeRecord(test_id=t1, cond_id=cid, value=val, decision=False))
    engine.update(matrix, log, t1)

    # T2：c1=F, c2=F, c3=T, c4=T → decision=False（與 T1 完全相同）
    t2 = "T2"
    for cid, val in [(c1_id, False), (c2_id, False), (c3_id, True), (c4_id, True)]:
        log.append(ProbeRecord(test_id=t2, cond_id=cid, value=val, decision=False))

    accepted = gate.evaluate(matrix, log, t2)

    assert accepted is False, "相同測試案例不應降低 L(X)，應拒絕"
