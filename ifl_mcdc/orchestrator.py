"""
IFL 主控迴圈：協調三層模組，執行完整的 IFL 迭代回饋迴圈。

參考 SDD 第 6 章。
"""
from __future__ import annotations

import random
import sys
import types
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from ifl_mcdc.config import IFLConfig
from ifl_mcdc.exceptions import LLMSamplingError, Z3TimeoutError, Z3UNSATError
from ifl_mcdc.layer1.ast_parser import ASTParser
from ifl_mcdc.layer1.coverage_engine import MCDCCoverageEngine
from ifl_mcdc.layer1.probe_injector import ProbeInjector
from ifl_mcdc.layer2.gap_analyzer import GapAnalyzer
from ifl_mcdc.layer2.smt_synthesizer import SMTConstraintSynthesizer
from ifl_mcdc.layer3.acceptance_gate import AcceptanceGate
from ifl_mcdc.layer3.llm_sampler import LLMBackend, LLMSampler
from ifl_mcdc.layer3.prompt_builder import PromptConstructor
from ifl_mcdc.models.decision_node import DecisionNode
from ifl_mcdc.models.probe_record import ProbeLog


@dataclass
class IFLResult:
    """IFL 主迴圈的執行結果。"""

    converged: bool                       # True = 傳統覆蓋率 100%（compute_loss == 0）
    final_coverage: float                 # 傳統覆蓋率（已覆蓋 / 2k）
    test_suite: list[dict[str, object]]   # 通過 AcceptanceGate 的案例
    iteration_count: int
    total_tokens: int                     # LLM API 消耗的估算 token 數
    infeasible_paths: list[str]           # 被 Z3 標記為不可行的條件 ID
    loss_history: list[int]              # 每次迭代後的損失值（2k - covered）
    all_generated_cases: list[dict[str, object]] = field(default_factory=list)
    # Gate 不論通過與否，所有已生成的案例（含隨機初始、LLM True 側、Z3 補集 False 側）
    # 供多樣性分析使用；不影響覆蓋率計算與 test_suite
    failure_log: list[str] = field(default_factory=list)
    # 每次 LLMSamplingError 或 SMT UNSAT/Timeout 的原因字串，供失敗率統計使用


class IFLOrchestrator:
    """IFL 主控迴圈：協調 Layer1/2/3，驅動迭代直至收斂或預算耗盡。"""

    def __init__(
        self,
        config: IFLConfig,
        backend: LLMBackend | None = None,
    ) -> None:
        self.config = config
        self.parser = ASTParser()
        self.engine = MCDCCoverageEngine()
        self.analyzer = GapAnalyzer()
        self.smt = SMTConstraintSynthesizer(domain_bounds=config.domain_bounds)
        self.prompt = PromptConstructor()
        actual_backend = backend if backend is not None else config.llm_backend
        self.sampler = LLMSampler(actual_backend, config.domain_validator, config.llm_retry_delay)
        self.gate = AcceptanceGate(self.engine)
        self._infeasible: set[str] = set()
        self._comp_failures: dict[str, int] = {}  # 補集連續失敗計數

    def run(self, source_path: str | Path) -> IFLResult:
        """執行 IFL 主流程。

        步驟：
          1. Layer 1 解析原始碼
          2. 探針注入 & 動態載入
          3. 執行 3 個隨機初始測試案例
          4. 建立初始 MCDCMatrix
          5. IFL 迭代迴圈（直至收斂或預算耗盡）
          6. 組裝並回傳 IFLResult

        Args:
            source_path: 目標 Python 原始碼路徑。

        Returns:
            IFLResult 含覆蓋率、測試套件、損失歷程等資訊。

        Raises:
            ValueError: 原始碼中找不到任何決策節點。
        """
        # ── 步驟 1：解析 ──
        decision_nodes = self.parser.parse_file(str(source_path))
        if not decision_nodes:
            raise ValueError(f"在 {source_path} 中找不到任何決策節點")

        dn = decision_nodes[0]

        # ── 步驟 2：探針注入 ──
        source = Path(source_path).read_text(encoding="utf-8")
        injector = ProbeInjector(decision_nodes)
        instrumented_source = injector.inject(source)
        module_name = f"_ifl_inst_{Path(str(source_path)).stem}"
        instrumented_module = self._load_from_string(instrumented_source, module_name)

        log = ProbeLog()
        self._inject_probes(instrumented_module, log)

        # ── 步驟 3：初始隨機測試 ──
        test_suite: list[dict[str, object]] = []
        all_generated: list[dict[str, object]] = []
        failure_log: list[str] = []
        for _ in range(3):
            test_case = self._generate_random_test(
                dn, self.config.domain_types, self.config.domain_bounds
            )
            test_id = self._run_test(instrumented_module, test_case, log)
            entry = {**test_case, "__test_id": test_id, "__source": "random"}
            test_suite.append(entry)
            all_generated.append(entry)

        # ── 步驟 4：建立初始矩陣 ──
        matrix = self.engine.build_matrix(dn.condition_set, log)
        loss_history: list[int] = [matrix.compute_loss()]

        # ── 步驟 5：IFL 迭代迴圈 ──
        iteration = 0

        while matrix.compute_loss() > 0 and iteration < self.config.max_iterations:
            iteration += 1

            # 取難度最低的非不可行缺口
            gaps = self.analyzer.analyze(matrix)
            gap = next(
                (g for g in gaps if g.condition_id not in self._infeasible),
                None,
            )
            if gap is None:
                break  # 所有剩餘缺口均不可行

            # 步驟 2：Z3 合成約束 Φ_gap，得到可行解空間 Ω（bound_specs）
            try:
                smt_result = self.smt.synthesize(
                    dn, gap, self.config.domain_types
                )
            except (Z3TimeoutError, Z3UNSATError) as _smt_exc:
                self._infeasible.add(gap.condition_id)
                failure_log.append(f"SMT_FAIL:{gap.condition_id}:{type(_smt_exc).__name__}")
                loss_history.append(matrix.compute_loss())
                continue

            if not smt_result.satisfiable:
                self._infeasible.add(gap.condition_id)
                failure_log.append(f"SMT_UNSAT:{gap.condition_id}")
                loss_history.append(matrix.compute_loss())
                continue

            # 步驟 3：將 Ω 轉化為自然語言提示，呼叫 LLM 產生測試案例
            # LLMSampler 內部處理 JSON 解析失敗與 DomainValidator 驗證（最多重試 MAX_RETRIES 次）
            scenarios = self.config.scenarios
            scenario_hint = scenarios[iteration % len(scenarios)] if scenarios else ""
            p_prompt = self.prompt.build(
                dn,
                gap,
                smt_result.bound_specs or [],
                self.config.func_signature,
                self.config.domain_context,
                clinical_profile=self.config.clinical_profile,
                scenario_hint=scenario_hint,
                domain_types=self.config.domain_types,
                domain_bounds=self.config.domain_bounds,
            )
            # 步驟 4：LLM 生成 True 側 + Z3 直接合成 False 側（保證 MC/DC 配對有效）
            # 先執行兩側測試（均進入 log），再做 AcceptanceGate 驗證。
            # 這樣 True 側 gate 評估時 False 側已在 log 中，_others_ok 能匹配成對。
            try:
                new_case, _ = self.sampler.sample(p_prompt)

                # Z3 合成 False 側：固定非目標條件的變數值 → 保證 _others_ok 成立
                comp_test = self.smt.synthesize_complement(
                    dn, gap, self.config.domain_types, new_case
                )

                if comp_test is None:
                    # 補集不可行（目標條件在當前 True 側被遮罩，或域內不可偽）
                    # 追蹤連續失敗次數；超過閾值則標記為不可行
                    cnt = self._comp_failures.get(gap.condition_id, 0) + 1
                    self._comp_failures[gap.condition_id] = cnt
                    if cnt >= 3:
                        self._infeasible.add(gap.condition_id)
                    # 仍執行 True 側（加入 log 備用，但不強制配對）
                    true_id = self._run_test(instrumented_module, new_case, log)
                    llm_entry = {**new_case, "__test_id": true_id, "__source": "llm"}
                    all_generated.append(llm_entry)
                    if self.gate.evaluate(matrix, log, true_id):
                        test_suite.append(llm_entry)
                else:
                    # 補集可行：重設失敗計數，執行兩側後 Gate 驗證
                    self._comp_failures[gap.condition_id] = 0
                    true_id = self._run_test(instrumented_module, new_case, log)
                    false_id = self._run_test(instrumented_module, comp_test, log)
                    llm_entry = {**new_case, "__test_id": true_id, "__source": "llm"}
                    comp_entry = {**comp_test, "__test_id": false_id, "__source": "smt_comp"}
                    # 不論 Gate 結果，all_generated 均記錄（供多樣性分析用）
                    all_generated.append(llm_entry)
                    all_generated.append(comp_entry)

                    # Gate 驗證（False 側已在 log，可配對）
                    if self.gate.evaluate(matrix, log, true_id):
                        test_suite.append(llm_entry)
                    if self.gate.evaluate(matrix, log, false_id):
                        test_suite.append(comp_entry)
            except LLMSamplingError as _llm_exc:
                failure_log.append(f"LLM_FAIL:{str(_llm_exc)[:80]}")

            loss_history.append(matrix.compute_loss())

        # ── 步驟 6：組裝結果 ──
        total_tokens = sum(
            cast(int, e.get("est_tokens", 0))
            for e in self.sampler.token_log
        )

        return IFLResult(
            converged=matrix.compute_loss() == 0,
            final_coverage=matrix.coverage_ratio,
            test_suite=test_suite,
            iteration_count=iteration,
            total_tokens=total_tokens,
            infeasible_paths=list(self._infeasible),
            loss_history=loss_history,
            all_generated_cases=all_generated,
            failure_log=failure_log,
        )

    def _run_test(
        self,
        module: types.ModuleType,
        test_case: dict[str, object],
        log: ProbeLog,
    ) -> str:
        """執行單一測試案例，記錄探針，回傳 test_id。"""
        import ifl_mcdc.layer1.probe_injector as pi

        test_id = f"T{uuid.uuid4().hex[:8]}"
        setattr(pi._CURRENT_TEST_ID, "value", test_id)
        try:
            getattr(module, self.config.func_name)(**test_case)
        except Exception as _exc:  # noqa: BLE001  # 使用者函式可能拋出任意例外
            pass  # 測試失敗不影響探針記錄（已在 if 前記錄）
        return test_id

    @staticmethod
    def _load_from_string(source: str, module_name: str) -> types.ModuleType:
        """動態編譯並載入儀表板化模組。"""
        mod = types.ModuleType(module_name)
        exec(compile(source, module_name, "exec"), mod.__dict__)  # noqa: S102
        sys.modules[module_name] = mod
        return mod

    @staticmethod
    def _inject_probes(module: types.ModuleType, log: ProbeLog) -> None:
        """將探針函式注入儀表板化模組，並指向給定的 ProbeLog。"""
        import ifl_mcdc.layer1.probe_injector as pi

        pi._GLOBAL_LOG = log
        setattr(module, "_ifl_probe", pi._ifl_probe)
        setattr(module, "_ifl_record_decision", pi._ifl_record_decision)

    @staticmethod
    def _generate_random_test(
        dn: DecisionNode,
        domain_types: dict[str, str],
        domain_bounds: dict[str, list[int]] | None = None,
    ) -> dict[str, object]:
        """為決策節點的所有變數生成隨機值。

        int → randint(lo, hi)，邊界來自 domain_bounds；若無設定則 [0, 130]。
        bool → choice([True, False])
        """
        test: dict[str, object] = {}
        seen: set[str] = set()
        for cond in dn.condition_set.conditions:
            for var_name in cond.var_names:
                if var_name in seen:
                    continue
                seen.add(var_name)
                var_type = domain_types.get(var_name, "int")
                if var_type == "bool":
                    test[var_name] = random.choice([True, False])
                else:
                    bounds = (domain_bounds or {}).get(var_name, [0, 130])
                    test[var_name] = random.randint(bounds[0], bounds[1])
        return test

