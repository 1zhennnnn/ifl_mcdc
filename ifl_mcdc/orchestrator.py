"""
IFL 主控迴圈：協調三層模組，執行完整的 IFL 迭代回饋迴圈。

參考 SDD 第 6 章。
"""
from __future__ import annotations

import random
import sys
import types
import uuid
from dataclasses import dataclass
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
    test_suite: list[dict[str, object]]   # 所有已接受的測試案例
    iteration_count: int
    total_tokens: int                     # LLM API 消耗的估算 token 數
    infeasible_paths: list[str]           # 被 Z3 標記為不可行的條件 ID
    loss_history: list[int]               # 每次迭代後的損失值（2k - covered）


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
        self.smt = SMTConstraintSynthesizer()
        self.prompt = PromptConstructor()
        actual_backend = backend if backend is not None else config.llm_backend
        self.sampler = LLMSampler(actual_backend, config.domain_validator, config.llm_retry_delay)
        self.gate = AcceptanceGate(self.engine)
        self._infeasible: set[str] = set()

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
        for _ in range(3):
            test_case = self._generate_random_test(dn, self.config.domain_types)
            test_id = self._run_test(instrumented_module, test_case, log)
            test_suite.append({**test_case, "__test_id": test_id, "__source": "random"})

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
                    dn, gap, self.config.domain_types, self.config.domain_bounds
                )
            except (Z3TimeoutError, Z3UNSATError):
                self._infeasible.add(gap.condition_id)
                loss_history.append(matrix.compute_loss())
                continue

            if not smt_result.satisfiable:
                self._infeasible.add(gap.condition_id)
                loss_history.append(matrix.compute_loss())
                continue

            # 步驟 3：將 Ω 轉化為自然語言提示（PromptConstructor）
            p_prompt = self.prompt.build(
                dn,
                gap,
                smt_result.bound_specs or [],
                self.config.func_signature,
                self.config.domain_context,
            )

            # 步驟 4：LLM 在 Ω ∩ Valid(x) 中採樣
            # SDD §7：LLMSamplingError → 跳過此缺口本輪，下輪重試
            try:
                new_case, _ = self.sampler.sample(p_prompt)
            except LLMSamplingError:
                loss_history.append(matrix.compute_loss())
                continue

            # 步驟 5：AcceptanceGate 驗證 L(X) 是否下降
            test_id = self._run_test(instrumented_module, new_case, log)
            accepted = self.gate.evaluate(matrix, log, test_id)

            if accepted:
                test_suite.append(
                    {**new_case, "__test_id": test_id, "__source": "llm"}
                )

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
        setattr(pi._IFL_TEST_ID, "value", test_id)
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

        pi._IFL_GLOBAL_LOG = log
        setattr(module, "_ifl_probe", pi._ifl_probe)
        setattr(module, "_ifl_record_decision", pi._ifl_record_decision)

    @staticmethod
    def _generate_random_test(
        dn: DecisionNode,
        domain_types: dict[str, str],
    ) -> dict[str, object]:
        """為決策節點的所有變數生成隨機值。

        int → randint(0, 130)（符合 DomainValidator 年齡規則 0～130）
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
                    test[var_name] = random.randint(0, 130)
        return test
