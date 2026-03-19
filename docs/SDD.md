**軟體設計文件**

*Software Design Document (SDD)*

**基於 IFL 迭代回饋迴圈之 MC/DC 優化機制系統**

*--- 開發人員完整實作指南 ---*

  ---------------- -----------------------------------
  **文件類型**     軟體設計文件（SDD）

  **對應 SRS**     SRS_IFL_MCDC_System v1.0

  **版本**         1.0（初稿）

  **撰寫日期**     2026 年 3 月 18 日

  **目標讀者**     後端開發工程師、DevOps、技術主管

  **保密等級**     **機密 --- 禁止對外傳閱**
  ---------------- -----------------------------------

**0　如何閱讀本文件**

> **⚠ 警告** 本文件假設讀者已通讀 SRS
> v1.0。請勿在未理解需求的情況下直接跳至實作章節，否則你寫出來的東西會很奇怪。

本 SDD 將 SRS 中的每一條功能需求（FR-01 ～
FR-12）轉化為可直接動手寫程式碼的設計規格，包含類別定義、方法簽名、資料結構、演算法虛擬碼、錯誤處理策略，以及每個模組的單元測試驗收標準。

文件結構如下：

- 第 1 章：整體技術架構選型與目錄結構

- 第 2 章：共用資料模型（Data Model）--- 所有模組都用的類別

- 第 3 章：Layer 1 靜態分析層實作設計（FR-01 ～ FR-04）

- 第 4 章：Layer 2 SMT 推理層實作設計（FR-05 ～ FR-08）

- 第 5 章：Layer 3 LLM 協同層實作設計（FR-09 ～ FR-12）

- 第 6 章：系統整合與 IFL 主控迴圈

- 第 7 章：錯誤處理全策略

- 第 8 章：設定管理與環境佈署

- 第 9 章：單元測試與整合測試規格

- 附錄 A：完整 API 介面規格

- 附錄 B：資料庫 Schema（探針日誌持久化）

**1　整體技術架構選型**

**1.1　技術棧決策**

本系統以 Python 3.11+ 為主要實作語言，理由如下：Python ast 模組原生支援
AST 解析；z3-solver PyPI 套件提供完整 Z3 Python 綁定；主流 LLM
API（OpenAI、Anthropic、Together.ai）均提供 Python SDK。

  ---------------- ----------------------- ----------------------------------------------------
  **元件**         **選用技術**            **選用理由**

  AST 解析         ast（Python stdlib）    零外部依賴，原生支援 Python 3.11 所有語法

  SMT 求解器       z3-solver==4.13.0       業界標準 SMT 求解器，支援 Python API，UNSAT
                                           可輸出不可滿足性核心（Unsatisfiable Core）

  LLM 介接         openai / anthropic /    抽象為統一介面，透過設定切換模型，不鎖定單一供應商
                   together                

  設定管理         pydantic-settings +     型別安全的設定讀取，支援環境變數覆蓋，適合
                   .env                    Docker/K8s 部署

  日誌持久化       SQLite（開發）/         探針日誌需要跨會話查詢，關聯式資料庫利於 JOIN
                   PostgreSQL（正式）      運算計算獨立對

  測試框架         pytest + pytest-cov     業界標準，支援 fixture、parametrize、覆蓋率報告

  型別系統         mypy（strict mode）     強制型別標注，在 CI 階段擋住 90% 的 None 相關 bug
  ---------------- ----------------------- ----------------------------------------------------

**1.2　專案目錄結構**

> **⚠ 警告**
> 請完全按照以下目錄結構建立專案。不要自行發明目錄名稱，不要把所有東西都丟進
> utils.py。
>
> ifl_mcdc/
>
> ├── \_\_init\_\_.py
>
> ├── config.py \# 全域設定（Pydantic Settings）
>
> ├── models/ \# 共用資料模型（第 2 章）
>
> │ ├── \_\_init\_\_.py
>
> │ ├── decision_node.py \# DecisionNode, AtomicCondition, ConditionSet
>
> │ ├── probe_record.py \# ProbeRecord, ProbeLog
>
> │ ├── coverage_matrix.py \# MCDCMatrix, GapEntry, GapList
>
> │ ├── smt_models.py \# BoundSpec, SMTResult, Φgap
>
> │ └── validation.py \# ValidationResult, DomainRule
>
> ├── layer1/ \# 靜態分析層（第 3 章）
>
> │ ├── \_\_init\_\_.py
>
> │ ├── ast_parser.py \# FR-01: ASTParser
>
> │ ├── coupling_graph.py \# FR-02: CouplingGraphBuilder
>
> │ ├── probe_injector.py \# FR-03: ProbeInjector
>
> │ └── coverage_engine.py \# FR-04: MCDCCoverageEngine
>
> ├── layer2/ \# SMT 推理層（第 4 章）
>
> │ ├── \_\_init\_\_.py
>
> │ ├── boolean_derivative.py \# FR-05: BooleanDerivativeEngine
>
> │ ├── gap_analyzer.py \# FR-06: GapAnalyzer
>
> │ ├── smt_synthesizer.py \# FR-07: SMTConstraintSynthesizer
>
> │ └── bound_extractor.py \# FR-08: BoundExtractor
>
> ├── layer3/ \# LLM 協同層（第 5 章）
>
> │ ├── \_\_init\_\_.py
>
> │ ├── prompt_builder.py \# FR-09: PromptConstructor
>
> │ ├── domain_validator.py \# FR-10: DomainValidator
>
> │ ├── llm_sampler.py \# FR-11: LLMSampler
>
> │ └── acceptance_gate.py \# FR-12: AcceptanceGate
>
> ├── orchestrator.py \# IFL 主控迴圈（第 6 章）
>
> ├── db/ \# 資料庫（附錄 B）
>
> │ ├── \_\_init\_\_.py
>
> │ ├── schema.sql
>
> │ └── repository.py
>
> └── tests/ \# 測試（第 9 章）
>
> ├── unit/
>
> ├── integration/
>
> └── fixtures/

**1.3　系統元件關係圖**

![](media/b33085557ef6ea09341b9419f9721114c4e6fda7.png){width="5.833333333333333in"
height="3.5416666666666665in"}

*圖 1-1　元件圖（Component Diagram）--- 三層模組依賴關係*

![](media/0374ad32555547650d3c23a6383b972f6a6e17d4.png){width="5.833333333333333in"
height="3.5416666666666665in"}

*圖 1-2　類別圖（Class Diagram）--- 核心資料模型*

![](media/af49640cceaf3279f078f0b243867936cdbf91e8.png){width="5.833333333333333in"
height="3.2291666666666665in"}

*圖 1-3　使用案例圖（Use Case Diagram）*

**2　共用資料模型（models/）**

本章定義所有模組共用的資料類別。這些是系統的語言------每個開發人員必須熟記這些類別，因為它們像積木一樣被所有層次拼在一起。

**2.1　models/decision_node.py**

**2.1.1　AtomicCondition**

代表一個不可再分割的布林條件，例如 Age \>= 65 或 HighRisk。

> from dataclasses import dataclass, field
>
> from typing import Optional
>
> \@dataclass
>
> class AtomicCondition:
>
> cond_id: str \# 格式：\"D{decision_idx}.c{cond_idx}\"，例如 \"D1.c2\"
>
> expression: str \# 原始 Python 表達式字串，例如 \"Age \>= 65\"
>
> var_names: list\[str\] \# 涉及的變數名稱，例如 \[\"Age\"\]
>
> negated: bool = False \# True 表示此條件在父表達式中帶有 not
>
> ast_node: Optional\[object\] = field(default=None, repr=False) \# 原始
> AST 節點，用於 probe 注入
>
> def evaluate(self, bindings: dict) -\> bool:
>
> \"\"\"在給定的變數綁定下求值此條件。警告：使用
> eval()，只可在受信任的測試環境使用。\"\"\"
>
> return bool(eval(self.expression, {}, bindings))
>
> **💡 說明** cond_id 格式必須嚴格遵守 \"D{n}.c{m}\"。n 從 1
> 開始計數決策節點，m 從 1 開始計數同一決策節點內的條件。

**2.1.2　ConditionSet**

> \@dataclass
>
> class ConditionSet:
>
> decision_id: str \# 對應 DecisionNode.node_id
>
> conditions: list\[AtomicCondition\]
>
> coupling_matrix: list\[list\[str\]\] \# k×k，值為 \"OR\" / \"AND\" /
> None
>
> k: int \# 條件總數，= len(conditions)
>
> def get_coupled(self, cond_id: str) -\> list\[tuple\[AtomicCondition,
> str\]\]:
>
> \"\"\"回傳與 cond_id 耦合的所有條件及其耦合類型。
>
> Returns: \[(condition, coupling_type), \...\] coupling_type 為 \"OR\"
> 或 \"AND\"
>
> \"\"\"
>
> idx = next(i for i, c in enumerate(self.conditions) if c.cond_id ==
> cond_id)
>
> result = \[\]
>
> for j, c in enumerate(self.conditions):
>
> if j != idx and self.coupling_matrix\[idx\]\[j\] is not None:
>
> result.append((c, self.coupling_matrix\[idx\]\[j\]))
>
> return result

**2.1.3　DecisionNode**

> \@dataclass
>
> class DecisionNode:
>
> node_id: str \# 格式：\"D{n}\"，例如 \"D1\"
>
> node_type: str \# \"If\" \| \"While\" \| \"Assert\" \| \"IfExp\"
>
> line_no: int \# 在原始碼中的行號（1-indexed）
>
> expression_str: str \# 完整的布林表達式字串
>
> condition_set: ConditionSet \# 分解後的條件集合
>
> source_context: str = \"\" \# 前後各 2 行的原始碼片段，供除錯用

**2.2　models/probe_record.py**

**2.2.1　ProbeRecord**

> \@dataclass
>
> class ProbeRecord:
>
> test_id: str \# 測試案例的唯一 ID，例如 \"T001\"
>
> cond_id: str \# 對應 AtomicCondition.cond_id
>
> value: bool \# 此條件在本次測試中的實際真值
>
> decision: bool \# 對應決策節點 D(x) 的最終輸出（整體 if 的結果）
>
> timestamp: float \# time.time() 的 Unix timestamp

**2.2.2　ProbeLog**

> \@dataclass
>
> class ProbeLog:
>
> records: list\[ProbeRecord\] = field(default_factory=list)
>
> \_lock: threading.Lock = field(default_factory=threading.Lock,
> repr=False)
>
> def append(self, record: ProbeRecord) -\> None:
>
> \"\"\"執行緒安全的新增。probe() 函式從多執行緒環境呼叫此方法。\"\"\"
>
> with self.\_lock:
>
> self.records.append(record)
>
> def get_by_test(self, test_id: str) -\> list\[ProbeRecord\]:
>
> return \[r for r in self.records if r.test_id == test_id\]
>
> def get_by_cond(self, cond_id: str) -\> list\[ProbeRecord\]:
>
> return \[r for r in self.records if r.cond_id == cond_id\]
>
> **⚠ 警告** ProbeLog 會被 probe()
> 函式從被測程式碼的執行執行緒呼叫。\_lock 是必要的，不能刪掉。

**2.3　models/coverage_matrix.py**

> \@dataclass
>
> class GapEntry:
>
> condition_id: str \# 例如 \"D1.c2\"
>
> flip_direction: str \# \"F2T\"（False→True）或 \"T2F\"（True→False）
>
> missing_pair_type: str \# \"unique_cause\" 或 \"masking\"
>
> estimated_difficulty: float \# 0.0（容易）到
> 1.0（困難），由耦合邊數量計算
>
> \@dataclass
>
> class MCDCMatrix:
>
> condition_set: ConditionSet
>
> \_matrix: dict\[tuple\[str,str\], set\[str\]\] \# (cond_id, flip_dir)
> -\> {test_ids}
>
> \_covered: set\[tuple\[str,str\]\] \# 已確認有有效獨立對的翻轉對
>
> \@property
>
> def k(self) -\> int:
>
> return self.condition_set.k
>
> \@property
>
> def coverage_ratio(self) -\> float:
>
> return len(self.\_covered) / (2 \* self.k) if self.k \> 0 else 1.0
>
> def compute_loss(self) -\> int:
>
> \"\"\"L(X)：未覆蓋的翻轉對數量。L=0 代表 100% MC/DC。\"\"\"
>
> total_pairs = {(c.cond_id, d)
>
> for c in self.condition_set.conditions
>
> for d in (\"F2T\", \"T2F\")}
>
> return len(total_pairs - self.\_covered)
>
> def get_gap_list(self) -\> list\[GapEntry\]:
>
> \"\"\"回傳所有缺口，按 estimated_difficulty 升序排列。\"\"\"
>
> \# 實作見 第 3.4 節
>
> \...

**2.4　models/smt_models.py**

> \@dataclass
>
> class BoundSpec:
>
> var_name: str
>
> var_type: str \# \"int\" \| \"bool\" \| \"str\" \| \"float\"
>
> interval: tuple \| None \# 數值型：(min, max)；None 表示無限制
>
> valid_set: set \| None \# 類別型：允許的值集合；None 表示無限制
>
> medical_unit: str = \"\" \# 例如 \"days\", \"years\"，供 LLM 提示使用
>
> \@dataclass
>
> class SMTResult:
>
> satisfiable: bool
>
> model: dict\[str, object\] \| None \# SAT 時有值，為變數名 -\> 具體值
>
> bound_specs: list\[BoundSpec\] \| None \# SAT 時有值
>
> core: list\[str\] \| None \# UNSAT 時有值，不可滿足性核心
>
> solve_time: float = 0.0 \# 求解耗時（秒）

**3　Layer 1：靜態分析層（layer1/）**

![](media/2fcb30e4c98e2ab1e087e136ae90477258ebc624.png){width="4.583333333333333in"
height="6.041666666666667in"}

*圖 3-1　活動圖（Activity Diagram）--- Layer 1 到 Layer 3 完整執行流程*

**3.1　ast_parser.py（FR-01）**

**3.1.1　類別設計**

> import ast
>
> from pathlib import Path
>
> from ifl_mcdc.models.decision_node import DecisionNode, ConditionSet,
> AtomicCondition
>
> class ASTParser(ast.NodeVisitor):
>
> \"\"\"
>
> 走訪 Python AST，識別所有決策節點並輸出 DecisionNode 清單。
>
> 繼承 ast.NodeVisitor ------ 每遇到目標節點類型就觸發對應的 visit_X
> 方法。
>
> \"\"\"
>
> DECISION_NODE_TYPES = (\"If\", \"While\", \"Assert\", \"IfExp\")
>
> def \_\_init\_\_(self):
>
> self.\_decision_nodes: list\[DecisionNode\] = \[\]
>
> self.\_node_counter: int = 0
>
> self.\_source_lines: list\[str\] = \[\]
>
> def parse_file(self, filepath: str \| Path) -\> list\[DecisionNode\]:
>
> \"\"\"
>
> 主要入口。呼叫方式：
>
> parser = ASTParser()
>
> nodes = parser.parse_file(\"vaccine_eligibility.py\")
>
> \"\"\"
>
> source = Path(filepath).read_text(encoding=\"utf-8\")
>
> self.\_source_lines = source.splitlines()
>
> tree = ast.parse(source) \# 若語法錯誤拋 SyntaxError，呼叫方處理
>
> self.visit(tree) \# 觸發所有 visit_X
>
> return self.\_decision_nodes
>
> def visit_If(self, node: ast.If) -\> None:
>
> self.\_register_decision(node, \"If\", node.test)
>
> self.generic_visit(node) \# 繼續走訪子節點（巢狀 if）
>
> def visit_While(self, node: ast.While) -\> None:
>
> self.\_register_decision(node, \"While\", node.test)
>
> self.generic_visit(node)
>
> def visit_Assert(self, node: ast.Assert) -\> None:
>
> if node.test:
>
> self.\_register_decision(node, \"Assert\", node.test)
>
> self.generic_visit(node)
>
> def visit_IfExp(self, node: ast.IfExp) -\> None:
>
> self.\_register_decision(node, \"IfExp\", node.test)
>
> self.generic_visit(node)
>
> def \_register_decision(self, node: ast.AST, ntype: str, test_expr:
> ast.expr) -\> None:
>
> self.\_node_counter += 1
>
> node_id = f\"D{self.\_node_counter}\"
>
> line_no = getattr(node, \"lineno\", 0)
>
> expr_str = ast.unparse(test_expr) \# Python 3.9+ 內建，把 AST
> 反解回字串
>
> condition_set = self.\_decompose_conditions(node_id, test_expr)
>
> context = self.\_get_context(line_no)
>
> self.\_decision_nodes.append(
>
> DecisionNode(node_id, ntype, line_no, expr_str, condition_set,
> context)
>
> )
>
> def \_decompose_conditions(self, decision_id: str, node: ast.expr) -\>
> ConditionSet:
>
> \"\"\"
>
> 遞迴分解布林表達式，提取原子條件。
>
> 規則：
>
> \- BoolOp (and/or) → 遞迴進入 values 列表
>
> \- UnaryOp (not) → 遞迴進入 operand，標記 negated=True
>
> \- Compare / Name / Call → 終止遞迴，此為一個原子條件
>
> \"\"\"
>
> conditions: list\[AtomicCondition\] = \[\]
>
> cond_counter = \[0\]
>
> def recurse(expr: ast.expr, negated: bool = False):
>
> if isinstance(expr, ast.BoolOp):
>
> for value in expr.values:
>
> recurse(value, negated)
>
> elif isinstance(expr, ast.UnaryOp) and isinstance(expr.op, ast.Not):
>
> recurse(expr.operand, not negated)
>
> else:
>
> cond_counter\[0\] += 1
>
> cond_id = f\"{decision_id}.c{cond_counter\[0\]}\"
>
> expr_str = ast.unparse(expr)
>
> var_names = \[n.id for n in ast.walk(expr) if isinstance(n,
> ast.Name)\]
>
> conditions.append(
>
> AtomicCondition(cond_id, expr_str, var_names, negated, expr)
>
> )
>
> recurse(node)
>
> coupling = CouplingGraphBuilder().build(decision_id, node, conditions)
>
> return ConditionSet(decision_id, conditions, coupling,
> len(conditions))
>
> def \_get_context(self, line_no: int, radius: int = 2) -\> str:
>
> start = max(0, line_no - radius - 1)
>
> end = min(len(self.\_source_lines), line_no + radius)
>
> return \"\\n\".join(self.\_source_lines\[start:end\])
>
> **🔑 重要** visit_If 必須呼叫 self.generic_visit(node) 以處理巢狀
> if。若忘記加，內層的 if 節點將被完全跳過，FR-01 驗收測試會失敗。

**3.2　coupling_graph.py（FR-02）**

**3.2.1　耦合規則說明**

耦合圖決定了 SMT 求解器在為條件 cᵢ 合成 Φ_gap
時，需要額外固定哪些其他條件的值。規則如下：

- 同一個 or BoolOp 的 values 列表內的條件 → OR 耦合 → 在求解 cᵢ
  時需將其他條件設為 False

- 同一個 and BoolOp 的 values 列表內的條件 → AND 耦合 → 在求解 cᵢ
  時需將其他條件設為 True

- 跨 BoolOp 層次的條件（祖先/後裔關係）→ 弱 AND 耦合

**3.2.2　完整實作**

> import ast
>
> from ifl_mcdc.models.decision_node import AtomicCondition
>
> class CouplingGraphBuilder:
>
> \"\"\"
>
> 從 BoolOp AST 結構建構 k×k 耦合鄰接矩陣。
>
> matrix\[i\]\[j\] = \"OR\" → 條件 i 和 j 共享 or 算子，相互 OR 耦合
>
> matrix\[i\]\[j\] = \"AND\" → 條件 i 和 j 共享 and 算子，相互 AND 耦合
>
> matrix\[i\]\[j\] = None → 無直接耦合
>
> \"\"\"
>
> def build(
>
> self,
>
> decision_id: str,
>
> root_expr: ast.expr,
>
> conditions: list\[AtomicCondition\],
>
> ) -\> list\[list\[str \| None\]\]:
>
> k = len(conditions)
>
> matrix: list\[list\[str \| None\]\] = \[\[None\]\*k for \_ in
> range(k)\]
>
> \# 建立 AST 節點 → 條件索引的映射
>
> node_to_idx: dict\[int, int\] = {
>
> id(c.ast_node): i for i, c in enumerate(conditions)
>
> }
>
> def fill_group(bool_op: ast.BoolOp, op_type: str):
>
> \"\"\"
>
> 遞迴找出此 BoolOp 的所有直接原子葉子，兩兩設定耦合。
>
> op_type: \"OR\"（ast.Or）或 \"AND\"（ast.And）
>
> \"\"\"
>
> leaf_indices: list\[int\] = \[\]
>
> def collect_leaves(expr: ast.expr):
>
> if isinstance(expr, ast.BoolOp):
>
> for v in expr.values:
>
> collect_leaves(v)
>
> elif isinstance(expr, ast.UnaryOp) and isinstance(expr.op, ast.Not):
>
> collect_leaves(expr.operand)
>
> else:
>
> idx = node_to_idx.get(id(expr))
>
> if idx is not None:
>
> leaf_indices.append(idx)
>
> \# 只收集此 BoolOp 直接子節點的葉子
>
> for value in bool_op.values:
>
> collect_leaves(value)
>
> \# 兩兩設定耦合（對稱矩陣）
>
> for a in leaf_indices:
>
> for b in leaf_indices:
>
> if a != b:
>
> matrix\[a\]\[b\] = op_type
>
> matrix\[b\]\[a\] = op_type
>
> def traverse(expr: ast.expr):
>
> if isinstance(expr, ast.BoolOp):
>
> op_type = \"OR\" if isinstance(expr.op, ast.Or) else \"AND\"
>
> fill_group(expr, op_type)
>
> for v in expr.values:
>
> traverse(v) \# 遞迴處理巢狀 BoolOp
>
> elif isinstance(expr, ast.UnaryOp):
>
> traverse(expr.operand)
>
> traverse(root_expr)
>
> return matrix
>
> **✅ 實作提示** 如果兩個條件既存在 OR 耦合又存在 AND
> 耦合（罕見，但多層巢狀時可能發生），優先保留 OR 耦合，因為 OR
> 耦合的遮罩效應更強，消除成本更高。

**3.3　probe_injector.py（FR-03）**

**3.3.1　核心設計------短路求值繞過**

> **⚠ 警告** 這是整個系統最容易犯錯的地方。Python 的 or / and
> 是短路求值：(True or X) 中 X
> 永遠不會被求值。必須在複合表達式組合前先將每個原子條件求值並存入暫存變數，否則
> MC/DC 矩陣將出現假性空白。

改寫前（問題程式碼）：

> if (age \>= 65 or high_risk) and days \> 180 and not egg_allergy:
>
> \# age=70, high_risk=True 時，high_risk 根本不會被求值！

改寫後（探針注入後的等效程式碼，概念示意）：

> \# 步驟 1：先無條件求值所有原子條件
>
> \_D1_c1 = \_probe(\"D1.c1\", age \>= 65) \# age \>= 65
>
> \_D1_c2 = \_probe(\"D1.c2\", high_risk) \# HighRisk
>
> \_D1_c3 = \_probe(\"D1.c3\", days \> 180) \# DaysSinceLast \> 180
>
> \_D1_c4 = \_probe(\"D1.c4\", egg_allergy) \# EggAllergy（未加 not）
>
> \# 步驟 2：用暫存變數重組原始表達式，語意完全等價
>
> \_D1_decision = (\_D1_c1 or \_D1_c2) and \_D1_c3 and not \_D1_c4
>
> \_probe_decision(\"D1\", \_D1_decision) \# 記錄決策結果
>
> if \_D1_decision:
>
> approve()

**3.3.2　ProbeInjector 類別實作**

> import ast, textwrap
>
> from ifl_mcdc.models.decision_node import DecisionNode
>
> class ProbeInjector(ast.NodeTransformer):
>
> \"\"\"
>
> 繼承 ast.NodeTransformer，對決策節點進行 AST 重寫。
>
> NodeTransformer 的 visit_X 方法回傳的值會替換原本的節點。
>
> \"\"\"
>
> def \_\_init\_\_(self, decision_nodes: list\[DecisionNode\]):
>
> \# 建立 line_no → DecisionNode 的快速查找
>
> self.\_node_map: dict\[int, DecisionNode\] = {
>
> dn.line_no: dn for dn in decision_nodes
>
> }
>
> self.\_injected_count = 0
>
> def inject(self, source: str) -\> str:
>
> \"\"\"主入口。回傳注入探針後的 Python 原始碼字串。\"\"\"
>
> tree = ast.parse(source)
>
> new_tree = self.visit(tree)
>
> ast.fix_missing_locations(new_tree) \# 補全行號等 metadata
>
> return ast.unparse(new_tree)
>
> def visit_If(self, node: ast.If) -\> ast.AST:
>
> dn = self.\_node_map.get(node.lineno)
>
> if dn is None:
>
> return self.generic_visit(node) \# 不在決策清單中，不改寫
>
> \# 生成：\_D1_c1 = \_probe(\"D1.c1\", age \>= 65) × k 行
>
> assignments = self.\_build_assignments(dn)
>
> \# 生成：\_D1_decision = (\_D1_c1 or \_D1_c2) and \_D1_c3 and not
> \_D1_c4
>
> decision_assign = self.\_build_decision_assign(dn)
>
> \# 生成：\_probe_decision(\"D1\", \_D1_decision)
>
> record_call = self.\_build_record_call(dn)
>
> \# 把原始 if 的 test 換成暫存變數
>
> new_test = ast.Name(id=f\"\_{dn.node_id}\_decision\", ctx=ast.Load())
>
> node.test = new_test
>
> \# 遞迴處理 body 內的巢狀 if
>
> self.generic_visit(node)
>
> \# 在 if 語句前插入所有賦值語句
>
> return \[\*assignments, decision_assign, record_call, node\]
>
> def \_build_assignments(self, dn: DecisionNode) -\>
> list\[ast.Assign\]:
>
> result = \[\]
>
> for cond in dn.condition_set.conditions:
>
> var_name = f\"\_{dn.node_id}\_{cond.cond_id.split(\".\")\[1\]}\" \#
> e.g. \_D1_c1
>
> \# \_D1_c1 = \_probe(\"D1.c1\", age \>= 65)
>
> assign = ast.parse(
>
> f\"{var_name} = \_ifl_probe(\\\"{cond.cond_id}\\\",
> {cond.expression})\"
>
> ).body\[0\]
>
> result.append(assign)
>
> return result
>
> def \_build_decision_assign(self, dn: DecisionNode) -\> ast.Assign:
>
> \# 用暫存變數名稱替換原始表達式中的每個條件
>
> rebuilt_expr = dn.expression_str
>
> for cond in dn.condition_set.conditions:
>
> var_name = f\"\_{dn.node_id}\_{cond.cond_id.split(\".\")\[1\]}\"
>
> rebuilt_expr = rebuilt_expr.replace(cond.expression, var_name, 1)
>
> return ast.parse(f\"\_{dn.node_id}\_decision =
> {rebuilt_expr}\").body\[0\]
>
> def \_build_record_call(self, dn: DecisionNode) -\> ast.Expr:
>
> return ast.parse(
>
> f\'\_ifl_record_decision(\"{dn.node_id}\", \_{dn.node_id}\_decision)\'
>
> ).body\[0\]

**3.3.3　probe() 函式（注入到被測模組的全域函式）**

> \# 此函式會被動態注入到被測模組的命名空間中
>
> \# 不應手動 import，ProbeInjector 會自動處理
>
> import time, threading
>
> from ifl_mcdc.models.probe_record import ProbeRecord, ProbeLog
>
> \_GLOBAL_LOG: ProbeLog = ProbeLog()
>
> \_CURRENT_TEST_ID: threading.local = threading.local()
>
> def \_ifl_probe(cond_id: str, value: bool) -\> bool:
>
> \"\"\"
>
> 記錄條件真值後原樣回傳，確保語意零干擾。
>
> decision 欄位暫填 None，由 \_ifl_record_decision 回填。
>
> \"\"\"
>
> record = ProbeRecord(
>
> test_id=getattr(\_CURRENT_TEST_ID, \"value\", \"UNKNOWN\"),
>
> cond_id=cond_id,
>
> value=bool(value),
>
> decision=False, \# 暫填，稍後由 \_ifl_record_decision 更新
>
> timestamp=time.time()
>
> )
>
> \_GLOBAL_LOG.append(record)
>
> return value \# 必須原樣回傳！
>
> def \_ifl_record_decision(decision_id: str, result: bool) -\> None:
>
> \"\"\"回填最後 k 筆屬於此決策節點的記錄的 decision 欄位。\"\"\"
>
> test_id = getattr(\_CURRENT_TEST_ID, \"value\", \"UNKNOWN\")
>
> \# 找出此測試的最後 k 筆屬於此 decision_id 的記錄
>
> prefix = decision_id + \".c\"
>
> to_update = \[r for r in reversed(\_GLOBAL_LOG.records)
>
> if r.test_id == test_id and r.cond_id.startswith(prefix)\]
>
> for r in to_update:
>
> r.decision = result

**3.4　coverage_engine.py（FR-04）**

**3.4.1　MCDCMatrix 的建構與更新邏輯**

MC/DC 覆蓋率矩陣的核心在於判斷每對測試案例 (Tⱼ, Tₖ) 是否構成某個條件 cᵢ
的有效獨立對。獨立對的三個判斷條件必須同時成立：

1.  條件翻轉：cᵢ 在 Tⱼ 中為 False，在 Tₖ 中為 True（或反向）

2.  決策翻轉：D(Tⱼ) ≠ D(Tₖ)

3.  其他條件無干擾：所有 cⱼ（j≠i）在 Tⱼ 和 Tₖ
    中的值，或者相同，或者根據耦合矩陣判定不影響決策結果

> from ifl_mcdc.models.coverage_matrix import MCDCMatrix, GapEntry
>
> from ifl_mcdc.models.probe_record import ProbeLog
>
> from ifl_mcdc.models.decision_node import ConditionSet
>
> class MCDCCoverageEngine:
>
> def build_matrix(self, cond_set: ConditionSet, log: ProbeLog) -\>
> MCDCMatrix:
>
> \"\"\"從 ProbeLog 建立初始 MCDCMatrix。\"\"\"
>
> matrix = MCDCMatrix(condition_set=cond_set, \_matrix={},
> \_covered=set())
>
> test_ids = list(dict.fromkeys(r.test_id for r in log.records))
>
> for test_id in test_ids:
>
> self.\_update_one(matrix, log, test_id)
>
> return matrix
>
> def update(self, matrix: MCDCMatrix, log: ProbeLog, new_test_id: str)
> -\> bool:
>
> \"\"\"
>
> 增量更新。O(k \* m) 複雜度。
>
> Returns: True 若 L(X) 因此次更新而降低（即新增了至少一個有效獨立對）
>
> \"\"\"
>
> loss_before = matrix.compute_loss()
>
> self.\_update_one(matrix, log, new_test_id)
>
> return matrix.compute_loss() \< loss_before
>
> def \_update_one(self, matrix: MCDCMatrix, log: ProbeLog, test_id:
> str):
>
> new_records = log.get_by_test(test_id)
>
> existing_tests = list(dict.fromkeys(
>
> r.test_id for r in log.records if r.test_id != test_id
>
> ))
>
> for existing_id in existing_tests:
>
> existing_records = log.get_by_test(existing_id)
>
> self.\_check_pair(matrix, new_records, existing_records)
>
> def \_check_pair(self, matrix, recs_a: list, recs_b: list):
>
> \"\"\"
>
> 檢查測試對 (A, B) 是否構成任一條件的有效獨立對。
>
> \"\"\"
>
> map_a = {r.cond_id: r for r in recs_a}
>
> map_b = {r.cond_id: r for r in recs_b}
>
> if not map_a or not map_b:
>
> return
>
> \# 取任意一筆確認決策結果是否翻轉
>
> dec_a = next(iter(map_a.values())).decision
>
> dec_b = next(iter(map_b.values())).decision
>
> if dec_a == dec_b:
>
> return \# 決策未翻轉，此對無法貢獻任何獨立對
>
> for cond in matrix.condition_set.conditions:
>
> rec_a = map_a.get(cond.cond_id)
>
> rec_b = map_b.get(cond.cond_id)
>
> if rec_a is None or rec_b is None:
>
> continue
>
> if rec_a.value == rec_b.value:
>
> continue \# 此條件未翻轉
>
> \# 條件翻轉 + 決策翻轉 → 潛在獨立對，再檢查其他條件
>
> if self.\_others_ok(matrix, cond.cond_id, map_a, map_b):
>
> flip = \"F2T\" if (not rec_a.value and rec_b.value) else \"T2F\"
>
> matrix.\_covered.add((cond.cond_id, flip))
>
> matrix.\_covered.add((cond.cond_id,
>
> \"T2F\" if flip == \"F2T\" else \"F2T\"))
>
> def \_others_ok(self, matrix, target_id, map_a, map_b) -\> bool:
>
> \"\"\"
>
> 檢查除 target 以外的其他條件是否允許此獨立對成立。
>
> 根據耦合類型：
>
> \- OR 耦合的夥伴：兩個測試中都必須為 False（消除遮罩）
>
> \- AND 耦合的夥伴：兩個測試中都必須為 True（確保外層 and 不遮蔽）
>
> \- 無耦合：兩個測試中值相同即可
>
> \"\"\"
>
> coupled = matrix.condition_set.get_coupled(target_id)
>
> for (other_cond, coupling_type) in coupled:
>
> rec_a = map_a.get(other_cond.cond_id)
>
> rec_b = map_b.get(other_cond.cond_id)
>
> if rec_a is None or rec_b is None:
>
> continue
>
> if coupling_type == \"OR\":
>
> \# 兩者都必須為 False，才能確保 target 的翻轉不被 OR 夥伴遮罩
>
> if rec_a.value or rec_b.value:
>
> return False
>
> elif coupling_type == \"AND\":
>
> \# 兩者都必須為 True，才能確保 target 的翻轉不被 AND 短路
>
> if not rec_a.value or not rec_b.value:
>
> return False
>
> return True

**4　Layer 2：SMT 推理層（layer2/）**

![](media/c514f60c6b2ffd02effb07fde87c6bf6aea392a4.png){width="5.833333333333333in"
height="4.0625in"}

*圖 4-1　循序圖（Sequence Diagram）--- UC-03 核心迭代流程，重點顯示
Layer 2 與 Z3 的互動*

**4.1　boolean_derivative.py（FR-05）**

**4.1.1　布林導數的數學定義**

對布林函數 f(x) 中的變數 xᵢ，其布林導數定義為：

> ∂f/∂xᵢ = f(\..., xᵢ=True, \...) XOR f(\..., xᵢ=False, \...)

若 ∂f/∂xᵢ = 0（即兩個 f 值相同），則 xᵢ 在當前輸入下被遮罩，改變 xᵢ
無法影響決策結果。

**4.1.2　實作**

> from z3 import Bool, BoolVal, Solver, sat, unsat, And, Or, Not, Xor,
> is_true
>
> from ifl_mcdc.models.decision_node import DecisionNode,
> AtomicCondition
>
> from ifl_mcdc.models.smt_models import MaskingReport
>
> class BooleanDerivativeEngine:
>
> \"\"\"
>
> 使用 Z3 精確計算布林導數。
>
> 不使用採樣或近似------結果必須是數學精確的。
>
> \"\"\"
>
> def compute(
>
> self,
>
> decision_node: DecisionNode,
>
> target_cond: AtomicCondition,
>
> ) -\> MaskingReport:
>
> \"\"\"
>
> 計算 ∂f/∂target_cond 是否為 0。
>
> 若為 0，找出造成遮罩的其他條件。
>
> \"\"\"
>
> \# 建立 Z3 符號變數
>
> z3_vars = {
>
> c.cond_id: Bool(c.cond_id)
>
> for c in decision_node.condition_set.conditions
>
> }
>
> \# 建立 Z3 布林表達式（從耦合矩陣重建 f 的 Z3 形式）
>
> f_expr = self.\_build_z3_expr(decision_node, z3_vars)
>
> target_var = z3_vars\[target_cond.cond_id\]
>
> \# f_T = f(\..., target=True, \...)
>
> f_T = self.\_substitute(f_expr, target_var, BoolVal(True))
>
> \# f_F = f(\..., target=False, \...)
>
> f_F = self.\_substitute(f_expr, target_var, BoolVal(False))
>
> \# 布林導數 = f_T XOR f_F
>
> \# 若 ∃ 某個賦值使得 f_T XOR f_F = True → 導數非恆為 0 → 不被遮罩
>
> s = Solver()
>
> s.add(Xor(f_T, f_F))
>
> result = s.check()
>
> if result == sat:
>
> \# 導數可為 1，表示存在使 target 能獨立影響決策的輸入
>
> return MaskingReport(
>
> condition_id=target_cond.cond_id,
>
> is_masked=False,
>
> masking_cause=\[\],
>
> derivative_value=1
>
> )
>
> else:
>
> \# 導數恆為 0，target 永遠被遮罩
>
> masking = self.\_find_masking_cause(
>
> decision_node, target_cond, z3_vars, f_T, f_F
>
> )
>
> return MaskingReport(
>
> condition_id=target_cond.cond_id,
>
> is_masked=True,
>
> masking_cause=masking,
>
> derivative_value=0
>
> )
>
> def \_find_masking_cause(self, dn, target, z3_vars, f_T, f_F) -\>
> list\[str\]:
>
> \"\"\"
>
> 暴力搜尋：逐一嘗試固定每個耦合條件，看哪個固定後使導數可以為 1。
>
> 固定後使導數為 1 的條件，即為遮罩原因。
>
> \"\"\"
>
> coupled = dn.condition_set.get_coupled(target.cond_id)
>
> causes = \[\]
>
> for (other_cond, coupling_type) in coupled:
>
> fix_value = BoolVal(False) if coupling_type == \"OR\" else
> BoolVal(True)
>
> other_var = z3_vars\[other_cond.cond_id\]
>
> f_T_fixed = self.\_substitute(f_T, other_var, fix_value)
>
> f_F_fixed = self.\_substitute(f_F, other_var, fix_value)
>
> s2 = Solver()
>
> s2.add(Xor(f_T_fixed, f_F_fixed))
>
> if s2.check() == sat:
>
> causes.append(other_cond.cond_id)
>
> return causes
>
> def \_build_z3_expr(self, dn: DecisionNode, z3_vars: dict):
>
> \"\"\"
>
> 根據耦合矩陣重建 Z3 布林表達式。
>
> 注意：這裡用簡化的重建（只考慮最外層結構），複雜情況需完整 AST→Z3
> 轉換器。
>
> \"\"\"
>
> \# 此處應實作完整的 ast→z3 轉換器，此為虛擬碼示意
>
> \# 實際實作見 smt_synthesizer.py 的 ASTToZ3Converter 類別
>
> from ifl_mcdc.layer2.smt_synthesizer import ASTToZ3Converter
>
> return ASTToZ3Converter(z3_vars).convert(dn)
>
> \@staticmethod
>
> def \_substitute(expr, var, value):
>
> \"\"\"將 Z3 表達式中的 var 替換為 value。使用 z3 的 substitute
> 函式。\"\"\"
>
> from z3 import substitute
>
> return substitute(expr, (var, value))

**4.2　gap_analyzer.py（FR-06）**

> from ifl_mcdc.models.coverage_matrix import MCDCMatrix, GapEntry
>
> from ifl_mcdc.models.decision_node import ConditionSet
>
> class GapAnalyzer:
>
> def analyze(self, matrix: MCDCMatrix) -\> list\[GapEntry\]:
>
> \"\"\"
>
> 從 MCDCMatrix 提取所有缺口（未覆蓋的翻轉對），
>
> 按 estimated_difficulty 升序排列（難度低的先處理）。
>
> \"\"\"
>
> gaps = \[\]
>
> for cond in matrix.condition_set.conditions:
>
> for flip in (\"F2T\", \"T2F\"):
>
> if (cond.cond_id, flip) not in matrix.\_covered:
>
> difficulty = self.\_estimate_difficulty(
>
> matrix.condition_set, cond.cond_id
>
> )
>
> gaps.append(GapEntry(
>
> condition_id=cond.cond_id,
>
> flip_direction=flip,
>
> missing_pair_type=\"unique_cause\",
>
> estimated_difficulty=difficulty
>
> ))
>
> return sorted(gaps, key=lambda g: g.estimated_difficulty)
>
> def \_estimate_difficulty(self, cond_set: ConditionSet, cond_id: str)
> -\> float:
>
> \"\"\"
>
> 難度估計公式：耦合邊數量 / (k-1)
>
> 耦合邊越多，需要同時約束的條件越多，SMT 求解難度越高。
>
> \"\"\"
>
> coupled = cond_set.get_coupled(cond_id)
>
> if cond_set.k \<= 1:
>
> return 0.0
>
> return len(coupled) / (cond_set.k - 1)

**4.3　smt_synthesizer.py（FR-07 --- 核心模組）**

**4.3.1　整體設計**

這是 Layer 2 最複雜的模組。它的任務是把一個缺口（GapEntry）轉化為 Z3
可求解的 SMT 公式，並從求解結果萃取出具體的可行解空間 Ω。

> **🔑 重要** Z3 必須在 10 秒內完成求解（k ≤ 8）。若超時，應設定
> set_param(\"timeout\", 10000) 並捕捉 z3.Z3Exception。超時視同
> UNSAT，標記 infeasible。
>
> import z3
>
> from z3 import Bool, Int, Real, BoolVal, IntVal, Solver, sat, unsat,
> Optimize
>
> from ifl_mcdc.models.coverage_matrix import GapEntry
>
> from ifl_mcdc.models.decision_node import DecisionNode
>
> from ifl_mcdc.models.smt_models import SMTResult, BoundSpec
>
> class SMTConstraintSynthesizer:
>
> \"\"\"
>
> 將缺口轉化為 SMT 公式並求解，輸出 SMTResult。
>
> SAT 時：提供可行輸入向量（model）。
>
> UNSAT 時：提供不可滿足性核心（core），證明此路徑不可達。
>
> \"\"\"
>
> TIMEOUT_MS = 10_000 \# 10 秒
>
> def synthesize(
>
> self,
>
> decision_node: DecisionNode,
>
> gap: GapEntry,
>
> domain_types: dict\[str, str\], \# var_name -\> \"int\" \| \"bool\" \|
> \"float\"
>
> ) -\> SMTResult:
>
> \"\"\"
>
> 主方法。流程：
>
> 1\. 建立 Z3 變數（依 domain_types 選擇 Int / Bool / Real）
>
> 2\. 將 DecisionNode 的布林表達式轉為 Z3 公式
>
> 3\. 加入 Φ_gap 約束（確保 target 條件翻轉、決策翻轉、遮罩消除）
>
> 4\. 求解，回傳 SMTResult
>
> \"\"\"
>
> import time
>
> t0 = time.time()
>
> \# 步驟 1：建立 Z3 變數
>
> z3_vars = self.\_create_z3_vars(decision_node, domain_types)
>
> \# 步驟 2：將 DecisionNode 轉為 Z3 公式
>
> f_expr = ASTToZ3Converter(z3_vars).convert(decision_node)
>
> \# 步驟 3：建構 Φ_gap
>
> phi_gap = self.\_build_phi_gap(
>
> decision_node, gap, z3_vars, f_expr
>
> )
>
> \# 步驟 4：求解
>
> s = Solver()
>
> z3.set_param(\"timeout\", self.TIMEOUT_MS)
>
> s.add(phi_gap)
>
> result = s.check()
>
> solve_time = time.time() - t0
>
> if result == sat:
>
> model = s.model()
>
> bound_specs = BoundExtractor().extract(model, z3_vars, domain_types)
>
> concrete = {str(v): model\[v\] for v in model}
>
> return SMTResult(True, concrete, bound_specs, None, solve_time)
>
> else:
>
> \# 使用 UNSAT core 找出不可滿足的最小約束集
>
> s2 = Solver()
>
> s2.set(\"unsat_core\", True)
>
> tracked = \[\]
>
> for i, clause in enumerate(phi_gap):
>
> p = Bool(f\"p{i}\")
>
> s2.assert_and_track(clause, p)
>
> tracked.append(str(p))
>
> s2.check()
>
> core = \[str(c) for c in s2.unsat_core()\]
>
> return SMTResult(False, None, None, core, solve_time)
>
> def \_create_z3_vars(self, dn, domain_types):
>
> z3_vars = {}
>
> for cond in dn.condition_set.conditions:
>
> for var_name in cond.var_names:
>
> if var_name in z3_vars:
>
> continue
>
> t = domain_types.get(var_name, \"int\")
>
> z3_vars\[var_name\] = Int(var_name) if t == \"int\" else \\
>
> Bool(var_name) if t == \"bool\" else \\
>
> Real(var_name)
>
> return z3_vars
>
> def \_build_phi_gap(self, dn, gap, z3_vars, f_expr) -\> list:
>
> \"\"\"
>
> Φ_gap 由以下幾部分組成：
>
> \(A\) 決策節點輸出為 True（假設要覆蓋 F2T 方向）
>
> \(B\) 目標條件翻轉所需的具體值
>
> \(C\) 耦合夥伴的固定值（消除遮罩）
>
> \(D\) 領域合法性約束（age \> 0 等，從 config 讀取）
>
> \"\"\"
>
> phi = \[\]
>
> target_cond = next(
>
> c for c in dn.condition_set.conditions
>
> if c.cond_id == gap.condition_id
>
> )
>
> \# (A) 決策必須為 True（方向可配置）
>
> phi.append(f_expr == BoolVal(True))
>
> \# (B) 目標條件的方向
>
> target_z3 = ASTToZ3Converter(z3_vars).convert_cond(target_cond)
>
> if gap.flip_direction == \"F2T\":
>
> phi.append(target_z3 == BoolVal(True))
>
> else:
>
> phi.append(target_z3 == BoolVal(False))
>
> \# (C) 耦合夥伴固定值（消除遮罩）
>
> for (other, coupling_type) in
> dn.condition_set.get_coupled(gap.condition_id):
>
> other_z3 = ASTToZ3Converter(z3_vars).convert_cond(other)
>
> fix_val = BoolVal(False) if coupling_type == \"OR\" else BoolVal(True)
>
> phi.append(other_z3 == fix_val)
>
> return phi

**4.4　ASTToZ3Converter（smt_synthesizer.py 內部類別）**

> **⚠ 警告** 這個類別是整個 SMT 層的基礎。任何 AST
> 節點類型的遺漏都會導致 Z3 無法正確建構
> f_expr，進而產生錯誤的求解結果。必須覆蓋所有可能出現在醫療邏輯中的 AST
> 節點類型。
>
> class ASTToZ3Converter:
>
> \"\"\"
>
> 將 Python AST 節點轉換為 Z3 表達式。
>
> 支援：BoolOp(and/or), UnaryOp(not), Compare(\>,\<,\>=,\<=,==,!=),
> Name, Constant
>
> \"\"\"
>
> def \_\_init\_\_(self, z3_vars: dict):
>
> self.z3_vars = z3_vars
>
> def convert(self, decision_node: DecisionNode):
>
> \"\"\"轉換整個 DecisionNode 的表達式。\"\"\"
>
> import ast
>
> tree = ast.parse(decision_node.expression_str, mode=\"eval\")
>
> return self.\_visit(tree.body)
>
> def convert_cond(self, cond: AtomicCondition):
>
> \"\"\"轉換單一原子條件。\"\"\"
>
> import ast
>
> tree = ast.parse(cond.expression, mode=\"eval\")
>
> return self.\_visit(tree.body)
>
> def \_visit(self, node):
>
> import ast
>
> if isinstance(node, ast.BoolOp):
>
> operands = \[self.\_visit(v) for v in node.values\]
>
> if isinstance(node.op, ast.And):
>
> return z3.And(\*operands)
>
> else:
>
> return z3.Or(\*operands)
>
> elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
>
> return z3.Not(self.\_visit(node.operand))
>
> elif isinstance(node, ast.Compare):
>
> left = self.\_visit(node.left)
>
> \# Python Compare 可能有多個比較子（a \< b \< c），逐一處理
>
> parts = \[\]
>
> prev = left
>
> for op, comp in zip(node.ops, node.comparators):
>
> right = self.\_visit(comp)
>
> if isinstance(op, ast.Gt): parts.append(prev \> right)
>
> elif isinstance(op, ast.Lt): parts.append(prev \< right)
>
> elif isinstance(op, ast.GtE):parts.append(prev \>= right)
>
> elif isinstance(op, ast.LtE):parts.append(prev \<= right)
>
> elif isinstance(op, ast.Eq): parts.append(prev == right)
>
> elif isinstance(op, ast.NotEq):parts.append(prev != right)
>
> else: raise NotImplementedError(f\"Unsupported op: {op}\")
>
> prev = right
>
> return z3.And(\*parts) if len(parts) \> 1 else parts\[0\]
>
> elif isinstance(node, ast.Name):
>
> if node.id in self.z3_vars:
>
> return self.z3_vars\[node.id\]
>
> \# 處理 True/False 字面量
>
> if node.id == \"True\": return BoolVal(True)
>
> if node.id == \"False\": return BoolVal(False)
>
> raise KeyError(f\"Unknown variable: {node.id}\")
>
> elif isinstance(node, ast.Constant):
>
> if isinstance(node.value, bool): return BoolVal(node.value)
>
> if isinstance(node.value, int): return IntVal(node.value)
>
> raise TypeError(f\"Unsupported constant type: {type(node.value)}\")
>
> else:
>
> raise NotImplementedError(f\"Unsupported AST node:
> {type(node).\_\_name\_\_}\")

**5　Layer 3：LLM 協同層（layer3/）**

**5.1　prompt_builder.py（FR-09）**

**5.1.1　Gap-Guided Prompt 模板規格**

提示模板分為四個強制段落，每個段落都有明確的資訊對應來源：

  ------------- ----------------------------- ------------------------------------------
  **段落**      **來源**                      **說明**

  §1 醫療情境   DecisionNode.source_context   告訴 LLM
                                              這是什麼系統，確保生成資料有醫療語意

  §2 目標缺口   GapEntry.condition_id +       精確告知 LLM 需要驗證哪個條件的哪個方向
                flip_direction                

  §3 數值約束   BoundSpec 清單                直接從 Z3
                                              模型提取，每個變數的精確範圍或合法值集合

  §4 輸出規範   函式簽名 + JSON schema        要求 LLM 只輸出純 JSON，禁止說明文字
  ------------- ----------------------------- ------------------------------------------

**5.1.2　PromptConstructor 實作**

> import json
>
> from ifl_mcdc.models.coverage_matrix import GapEntry
>
> from ifl_mcdc.models.smt_models import BoundSpec
>
> from ifl_mcdc.models.decision_node import DecisionNode
>
> class PromptConstructor:
>
> MAX_TOKENS = 2048 \# 超過此長度需截斷，從 §1 開始截
>
> def build(
>
> self,
>
> decision_node: DecisionNode,
>
> gap: GapEntry,
>
> bound_specs: list\[BoundSpec\],
>
> func_signature: str, \# 例如 \"check_vaccine_eligibility(age,
> high_risk, days, egg_allergy)\"
>
> domain_context: str = \"\",
>
> ) -\> str:
>
> \"\"\"
>
> 產生缺口導向提示字串。
>
> 每個段落之間以空行分隔，確保 LLM 能清楚識別各段。
>
> \"\"\"
>
> flip_desc = \"True（需要讓此條件為 True）\" if gap.flip_direction ==
> \"F2T\" \\
>
> else \"False（需要讓此條件為 False）\"
>
> \# §1 情境
>
> sec1 = f\"\"\"【醫療情境】
>
> 你正在為以下醫療邏輯函式生成測試資料：
>
> 函式：{func_signature}
>
> 情境：{domain_context}
>
> 原始碼片段：
>
> {decision_node.source_context}\"\"\"
>
> \# §2 目標
>
> cond_expr = next(
>
> c.expression for c in decision_node.condition_set.conditions
>
> if c.cond_id == gap.condition_id
>
> )
>
> sec2 = f\"\"\"
>
> 【目標缺口】
>
> 當前測試集缺少條件 {gap.condition_id}（{cond_expr}）的獨立性驗證。
>
> 需要讓此條件的值為 {flip_desc}，且整體函式輸出因此改變。\"\"\"
>
> \# §3 數值約束
>
> constraint_lines = \[\]
>
> for bs in bound_specs:
>
> if bs.interval:
>
> lo, hi = bs.interval
>
> unit = f\"（單位：{bs.medical_unit}）\" if bs.medical_unit else \"\"
>
> constraint_lines.append(
>
> f\" - {bs.var_name}: {bs.var_type}，範圍 \[{lo}, {hi}\]{unit}\"
>
> )
>
> elif bs.valid_set:
>
> constraint_lines.append(
>
> f\" - {bs.var_name}: 必須為以下值之一 {sorted(bs.valid_set)}\"
>
> )
>
> sec3 = \"\\n【精確數值約束】（必須嚴格遵守，不可自行修改）\\n\" +
> \"\\n\".join(constraint_lines)
>
> \# §4 輸出規範
>
> sec4 = f\"\"\"
>
> 【輸出格式】
>
> 請僅輸出一個合法的 JSON 物件，鍵名必須與函式參數名稱完全一致。
>
> 禁止輸出任何說明文字、markdown 格式、程式碼區塊標記（\`\`\`）。
>
> 範例輸出格式（僅供格式參考，值必須符合上述約束）：
>
> {json.dumps({bs.var_name: \"\...\" for bs in bound_specs},
> ensure_ascii=False)}\"\"\"
>
> full_prompt = sec1 + sec2 + sec3 + sec4
>
> return self.\_truncate_if_needed(full_prompt)
>
> def \_truncate_if_needed(self, prompt: str) -\> str:
>
> \# 粗估 1 token ≈ 3 中文字 ≈ 4 英文字母
>
> estimated_tokens = len(prompt) / 3
>
> if estimated_tokens \<= self.MAX_TOKENS:
>
> return prompt
>
> \# 從 §1（情境）開始截斷，§3（約束）和 §4（格式）必須完整保留
>
> cut_chars = int((estimated_tokens - self.MAX_TOKENS) \* 3)
>
> return prompt\[cut_chars:\]

**5.2　domain_validator.py（FR-10）**

**5.2.1　DomainRule 資料結構**

> from dataclasses import dataclass
>
> from typing import Callable, Any
>
> \@dataclass
>
> class DomainRule:
>
> field: str \# 欄位名稱
>
> description: str \# 人類可讀的規則說明
>
> validator: Callable\[\[Any\], bool\] \# 回傳 True 表示合法
>
> \# 醫療預設規則集（可透過 config 擴充）
>
> DEFAULT_MEDICAL_RULES: list\[DomainRule\] = \[
>
> DomainRule(\"age\", \"年齡必須在 0～130 之間\",
>
> lambda v: isinstance(v, int) and 0 \<= v \<= 130),
>
> DomainRule(\"days_since_last\", \"距上次接種天數必須為非負整數\",
>
> lambda v: isinstance(v, int) and v \>= 0),
>
> DomainRule(\"high_risk\", \"高風險標記必須為布林值\",
>
> lambda v: isinstance(v, bool)),
>
> DomainRule(\"egg_allergy\", \"過敏標記必須為布林值\",
>
> lambda v: isinstance(v, bool)),
>
> \]

**5.2.2　DomainValidator 實作**

> import json
>
> from ifl_mcdc.models.validation import ValidationResult, Violation
>
> class DomainValidator:
>
> def \_\_init\_\_(self, rules: list\[DomainRule\] \| None = None):
>
> self.rules = rules or DEFAULT_MEDICAL_RULES
>
> def validate(self, test_case_json: str) -\> ValidationResult:
>
> \"\"\"
>
> 驗證 LLM 生成的測試案例 JSON 字串。
>
> 1\. 先嘗試 JSON 解析
>
> 2\. 逐一套用 DomainRule
>
> 3\. 回傳 ValidationResult
>
> \"\"\"
>
> try:
>
> data = json.loads(test_case_json)
>
> except json.JSONDecodeError as e:
>
> return ValidationResult(
>
> passed=False,
>
> violations=\[Violation(\"\_\_json\_\_\", \"無效的 JSON 格式\",
> str(e))\]
>
> )
>
> violations = \[\]
>
> for rule in self.rules:
>
> if rule.field not in data:
>
> continue \# 欄位不存在不視為違規（可能是選填欄位）
>
> try:
>
> ok = rule.validator(data\[rule.field\])
>
> except Exception:
>
> ok = False
>
> if not ok:
>
> violations.append(
>
> Violation(rule.field, rule.description, str(data\[rule.field\]))
>
> )
>
> return ValidationResult(passed=len(violations) == 0,
> violations=violations)

**5.3　llm_sampler.py（FR-11）**

**5.3.1　抽象介面設計（Strategy Pattern）**

不同 LLM 供應商的 API 細節各異，使用 Strategy Pattern
封裝差異，讓呼叫方不感知底層 API。

> from abc import ABC, abstractmethod
>
> import json, time
>
> class LLMBackend(ABC):
>
> \@abstractmethod
>
> def complete(self, prompt: str, max_tokens: int = 512) -\> str:
>
> \"\"\"回傳模型的完整文字回應。\"\"\"
>
> \...
>
> class OpenAIBackend(LLMBackend):
>
> def \_\_init\_\_(self, model: str = \"gpt-4o\", api_key: str = \"\"):
>
> from openai import OpenAI
>
> self.client = OpenAI(api_key=api_key)
>
> self.model = model
>
> def complete(self, prompt: str, max_tokens: int = 512) -\> str:
>
> resp = self.client.chat.completions.create(
>
> model=self.model,
>
> messages=\[{\"role\": \"user\", \"content\": prompt}\],
>
> max_tokens=max_tokens,
>
> temperature=0.3, \# 低溫度確保輸出穩定性
>
> )
>
> return resp.choices\[0\].message.content
>
> class AnthropicBackend(LLMBackend):
>
> def \_\_init\_\_(self, model: str = \"claude-sonnet-4-6\", api_key:
> str = \"\"):
>
> import anthropic
>
> self.client = anthropic.Anthropic(api_key=api_key)
>
> self.model = model
>
> def complete(self, prompt: str, max_tokens: int = 512) -\> str:
>
> msg = self.client.messages.create(
>
> model=self.model,
>
> max_tokens=max_tokens,
>
> messages=\[{\"role\": \"user\", \"content\": prompt}\],
>
> )
>
> return msg.content\[0\].text

**5.3.2　LLMSampler 主類別（含重試與 Token 記帳）**

> class LLMSampler:
>
> MAX_RETRIES = 3
>
> RETRY_DELAY = 2.0 \# 秒
>
> def \_\_init\_\_(self, backend: LLMBackend, validator:
> DomainValidator):
>
> self.backend = backend
>
> self.validator = validator
>
> self.token_log: list\[dict\] = \[\] \# 記帳用
>
> def sample(self, prompt: str) -\> tuple\[dict, ValidationResult\]:
>
> \"\"\"
>
> 呼叫 LLM 並嘗試解析 JSON。
>
> 最多重試 MAX_RETRIES 次。
>
> 回傳 (parsed_dict, validation_result)。
>
> 若所有重試失敗，拋出 LLMSamplingError。
>
> \"\"\"
>
> last_error = None
>
> for attempt in range(self.MAX_RETRIES + 1):
>
> if attempt \> 0:
>
> \# 第 2 次起：加入上一次的失敗原因作為修正提示
>
> prompt = self.\_build_retry_prompt(prompt, last_error)
>
> time.sleep(self.RETRY_DELAY \* attempt) \# 指數退避
>
> t0 = time.time()
>
> try:
>
> raw = self.backend.complete(prompt)
>
> except Exception as e:
>
> last_error = f\"API 呼叫失敗：{e}\"
>
> continue
>
> \# 記錄 token 消耗（粗估）
>
> self.token_log.append({
>
> \"attempt\": attempt,
>
> \"elapsed\": time.time() - t0,
>
> \"est_tokens\": len(raw) // 4,
>
> })
>
> \# 嘗試解析 JSON
>
> parsed, parse_error = self.\_parse_json(raw)
>
> if parse_error:
>
> last_error = f\"JSON
> 解析失敗：{parse_error}\\n原始回應：{raw\[:200\]}\"
>
> continue
>
> \# 領域驗證
>
> val_result = self.validator.validate(json.dumps(parsed))
>
> if not val_result.passed:
>
> last_error = f\"領域驗證失敗：{val_result.violations}\"
>
> continue
>
> return parsed, val_result
>
> raise LLMSamplingError(f\"重試 {self.MAX_RETRIES}
> 次後仍失敗，最後錯誤：{last_error}\")
>
> \@staticmethod
>
> def \_parse_json(raw: str) -\> tuple\[dict \| None, str \| None\]:
>
> \"\"\"
>
> 嘗試從 LLM 回應中提取 JSON。
>
> 處理 LLM 常見的不規範輸出，例如把 JSON 包在 \`\`\`json \... \`\`\`
> 裡。
>
> \"\"\"
>
> import re
>
> \# 移除 markdown code block 標記
>
> cleaned = re.sub(r\"\`\`\`(?:json)?\|\`\`\`\", \"\", raw).strip()
>
> try:
>
> return json.loads(cleaned), None
>
> except json.JSONDecodeError as e:
>
> \# 嘗試找到第一個 { 到最後一個 } 之間的內容
>
> match = re.search(r\"\\{.\*\\}\", cleaned, re.DOTALL)
>
> if match:
>
> try:
>
> return json.loads(match.group(0)), None
>
> except json.JSONDecodeError:
>
> pass
>
> return None, str(e)
>
> \@staticmethod
>
> def \_build_retry_prompt(original: str, error: str) -\> str:
>
> return f\"\"\"上一次你的回應有以下問題，請重新生成：
>
> {error}
>
> 請僅輸出合法的 JSON 物件，不要有任何其他文字。
>
> 原始需求：
>
> {original}\"\"\"

**5.4　acceptance_gate.py（FR-12）**

> from ifl_mcdc.models.coverage_matrix import MCDCMatrix
>
> from ifl_mcdc.layer1.coverage_engine import MCDCCoverageEngine
>
> from ifl_mcdc.models.probe_record import ProbeLog
>
> class AcceptanceGate:
>
> \"\"\"
>
> 接受門控：決定新生成的測試案例是否有貢獻，並更新矩陣。
>
> \"\"\"
>
> def \_\_init\_\_(self, engine: MCDCCoverageEngine):
>
> self.engine = engine
>
> def evaluate(
>
> self,
>
> matrix: MCDCMatrix,
>
> log: ProbeLog,
>
> new_test_id: str,
>
> ) -\> bool:
>
> \"\"\"
>
> 執行 x_new，更新矩陣，回傳 L(X) 是否降低。
>
> True = 接受（L 下降）
>
> False = 拒絕（L 不變，觸發回饋修正）
>
> \"\"\"
>
> return self.engine.update(matrix, log, new_test_id)

**6　系統整合：IFL 主控迴圈（orchestrator.py）**

這是系統的神經中樞，協調所有三個層次的運作，實作完整的 IFL
迭代回饋迴圈。

**6.1　IFLOrchestrator 完整實作**

> from dataclasses import dataclass
>
> from pathlib import Path
>
> from ifl_mcdc.config import IFLConfig
>
> from ifl_mcdc.layer1.ast_parser import ASTParser
>
> from ifl_mcdc.layer1.probe_injector import ProbeInjector
>
> from ifl_mcdc.layer1.coverage_engine import MCDCCoverageEngine
>
> from ifl_mcdc.layer2.gap_analyzer import GapAnalyzer
>
> from ifl_mcdc.layer2.smt_synthesizer import SMTConstraintSynthesizer
>
> from ifl_mcdc.layer3.prompt_builder import PromptConstructor
>
> from ifl_mcdc.layer3.llm_sampler import LLMSampler
>
> from ifl_mcdc.layer3.acceptance_gate import AcceptanceGate
>
> from ifl_mcdc.models.probe_record import ProbeLog, \_CURRENT_TEST_ID
>
> import importlib.util, sys, uuid, time
>
> \@dataclass
>
> class IFLResult:
>
> converged: bool \# True = 100% MC/DC 達成
>
> final_coverage: float \# 0.0 \~ 1.0
>
> test_suite: list\[dict\] \# 所有生成的測試案例
>
> iteration_count: int
>
> total_tokens: int \# LLM API 消耗的估算 token 數
>
> infeasible_paths: list\[str\] \# 被標記為 UNSAT 的缺口 ID
>
> loss_history: list\[int\] \# 每次迭代後的 L(X) 值
>
> class IFLOrchestrator:
>
> def \_\_init\_\_(self, config: IFLConfig):
>
> self.config = config
>
> self.parser = ASTParser()
>
> self.engine = MCDCCoverageEngine()
>
> self.analyzer = GapAnalyzer()
>
> self.smt = SMTConstraintSynthesizer()
>
> self.prompt = PromptConstructor()
>
> self.sampler = LLMSampler(config.llm_backend, config.domain_validator)
>
> self.gate = AcceptanceGate(self.engine)
>
> self.\_infeasible: set\[str\] = set()
>
> def run(self, source_path: str) -\> IFLResult:
>
> \"\"\"
>
> IFL 主流程：
>
> 1\. 解析原始碼
>
> 2\. 注入探針
>
> 3\. 執行初始測試（零樣本 LLM 生成）
>
> 4\. IFL 迭代迴圈
>
> \"\"\"
>
> \# ── 步驟 1：解析 ──
>
> decision_nodes = self.parser.parse_file(source_path)
>
> if not decision_nodes:
>
> raise ValueError(f\"在 {source_path} 中找不到任何決策節點\")
>
> \# ── 步驟 2：探針注入 ──
>
> source = Path(source_path).read_text()
>
> injector = ProbeInjector(decision_nodes)
>
> instrumented_source = injector.inject(source)
>
> instrumented_module = self.\_load_from_string(
>
> instrumented_source,
>
> module_name=f\"\_ifl_inst\_{Path(source_path).stem}\"
>
> )
>
> \# ── 步驟 3：建立探針日誌 & 初始測試 ──
>
> log = ProbeLog()
>
> self.\_inject_probes(instrumented_module, log)
>
> test_suite: list\[dict\] = \[\]
>
> \# 生成 3 個初始零樣本測試案例
>
> for \_ in range(3):
>
> tc = self.\_run_zero_shot(decision_nodes\[0\])
>
> if tc:
>
> test_id = self.\_run_test(instrumented_module, tc, log)
>
> test_suite.append({\*\*tc, \"\_\_test_id\": test_id})
>
> \# ── 步驟 4：建立初始矩陣 ──
>
> dn = decision_nodes\[0\] \# 目前針對第一個決策節點
>
> matrix = self.engine.build_matrix(dn.condition_set, log)
>
> loss_history = \[matrix.compute_loss()\]
>
> \# ── 步驟 5：IFL 迭代迴圈 ──
>
> iteration = 0
>
> while matrix.compute_loss() \> 0 and iteration \<
> self.config.max_iterations:
>
> iteration += 1
>
> \# 取難度最低的缺口
>
> gaps = self.analyzer.analyze(matrix)
>
> gap = next(
>
> (g for g in gaps if g.condition_id not in self.\_infeasible),
>
> None
>
> )
>
> if gap is None:
>
> break \# 所有剩餘缺口均不可行
>
> \# Z3 合成約束
>
> smt_result = self.smt.synthesize(dn, gap, self.config.domain_types)
>
> if not smt_result.satisfiable:
>
> \# 不可行路徑，永久標記
>
> self.\_infeasible.add(gap.condition_id)
>
> continue
>
> \# 建構提示並採樣
>
> p_prompt = self.prompt.build(
>
> dn, gap, smt_result.bound_specs,
>
> self.config.func_signature,
>
> self.config.domain_context
>
> )
>
> try:
>
> new_case, \_ = self.sampler.sample(p_prompt)
>
> except LLMSamplingError:
>
> continue \# LLM 採樣失敗，跳過此缺口本輪
>
> \# 執行新測試案例
>
> test_id = self.\_run_test(instrumented_module, new_case, log)
>
> \# 接受門控
>
> accepted = self.gate.evaluate(matrix, log, test_id)
>
> if accepted:
>
> test_suite.append({\*\*new_case, \"\_\_test_id\": test_id})
>
> loss_history.append(matrix.compute_loss())
>
> total_tokens = sum(e\[\"est_tokens\"\] for e in
> self.sampler.token_log)
>
> return IFLResult(
>
> converged=matrix.compute_loss() == 0,
>
> final_coverage=matrix.coverage_ratio,
>
> test_suite=test_suite,
>
> iteration_count=iteration,
>
> total_tokens=total_tokens,
>
> infeasible_paths=list(self.\_infeasible),
>
> loss_history=loss_history,
>
> )
>
> def \_run_test(self, module, test_case: dict, log: ProbeLog) -\> str:
>
> \"\"\"執行單一測試案例，回傳 test_id。\"\"\"
>
> test_id = f\"T{str(uuid.uuid4())\[:8\]}\"
>
> \_CURRENT_TEST_ID.value = test_id
>
> try:
>
> getattr(module, self.config.func_name)(\*\*test_case)
>
> except Exception:
>
> pass \# 測試案例導致異常不影響探針記錄
>
> return test_id
>
> \@staticmethod
>
> def \_load_from_string(source: str, module_name: str):
>
> \"\"\"動態載入儀表板化模組。\"\"\"
>
> import types
>
> mod = types.ModuleType(module_name)
>
> exec(compile(source, module_name, \"exec\"), mod.\_\_dict\_\_)
>
> sys.modules\[module_name\] = mod
>
> return mod
>
> \@staticmethod
>
> def \_inject_probes(module, log: ProbeLog):
>
> \"\"\"把 probe 函式注入到儀表板化模組的命名空間。\"\"\"
>
> from ifl_mcdc.layer1 import probe_injector as pi
>
> pi.\_GLOBAL_LOG = log
>
> module.\_\_dict\_\_\[\"\_ifl_probe\"\] = pi.\_ifl_probe
>
> module.\_\_dict\_\_\[\"\_ifl_record_decision\"\] =
> pi.\_ifl_record_decision

**7　錯誤處理全策略**

> **⚠ 警告** 以下所有例外類型必須全部實作，不允許使用裸的 except
> Exception
> 來吞掉錯誤。每一個錯誤都必須被記錄，且錯誤訊息必須足夠詳細讓開發者在 5
> 分鐘內找到根因。

  -------------------------- ---------------------- -------------------------------------- -----------------------
  **例外類別**               **觸發時機**           **處理策略**                           **日誌格式**

  SyntaxError                ast.parse() 失敗       拋出給呼叫方，附原始碼行號             ERROR \[AST\] line={n},
                                                                                           file={f}, msg={e}

  CouplingBuildError         耦合圖建構異常         回傳空矩陣，標記 WARNING               WARN \[COUPLING\]
                                                                                           decision_id={d},
                                                                                           cond={c}

  ProbeInjectionError        AST 重寫失敗           停止執行，不使用未完整的儀表板化模組   ERROR \[PROBE\]
                                                                                           node={n}, reason={r}

  Z3TimeoutError             Z3 超過 10 秒          標記此缺口為 INFEASIBLE（保守策略）    WARN \[SMT\] gap={g},
                                                                                           timeout=10s

  Z3UNSATError               Z3 回傳 UNSAT          記錄 UNSAT core，永久標記 infeasible   INFO \[SMT\] gap={g},
                                                                                           core=\[{c}\]

  LLMSamplingError           重試 3 次後仍失敗      跳過此缺口本輪，下輪重試               ERROR \[LLM\] gap={g},
                                                                                           last_err={e}

  DomainValidationError      測試資料違反領域規則   回饋違規報告給 LLM，重試               WARN \[VALID\]
                                                                                           field={f}, rule={r},
                                                                                           val={v}

  IterationBudgetExhausted   達到 max_iterations    輸出部分覆蓋報告，不拋例外             WARN \[IFL\]
                                                                                           budget={n},
                                                                                           coverage={c:.1%}
  -------------------------- ---------------------- -------------------------------------- -----------------------

**8　設定管理與環境部署（config.py）**

所有可調整的參數必須集中在 config.py，不允許在任何其他檔案中出現
hardcoded 的 API key、模型名稱或數值門檻值。

> from pydantic_settings import BaseSettings
>
> from pydantic import Field
>
> from ifl_mcdc.layer3.llm_sampler import OpenAIBackend,
> AnthropicBackend, LLMBackend
>
> from ifl_mcdc.layer3.domain_validator import DomainValidator,
> DEFAULT_MEDICAL_RULES
>
> class IFLConfig(BaseSettings):
>
> \# ── LLM ──
>
> llm_provider: str = Field(\"openai\", env=\"IFL_LLM_PROVIDER\")
>
> llm_model: str = Field(\"gpt-4o\", env=\"IFL_LLM_MODEL\")
>
> llm_api_key: str = Field(\"\", env=\"IFL_LLM_API_KEY\")
>
> llm_temperature: float = Field(0.3, env=\"IFL_LLM_TEMPERATURE\")
>
> \# ── SMT ──
>
> smt_timeout_ms: int = Field(10_000, env=\"IFL_SMT_TIMEOUT_MS\")
>
> \# ── IFL 迭代控制 ──
>
> max_iterations: int = Field(50, env=\"IFL_MAX_ITERATIONS\")
>
> min_coverage: float = Field(1.0, env=\"IFL_MIN_COVERAGE\")
>
> \# ── 目標模組 ──
>
> func_name: str = Field(\"check_vaccine_eligibility\",
> env=\"IFL_FUNC_NAME\")
>
> func_signature: str = Field(\"\", env=\"IFL_FUNC_SIGNATURE\")
>
> domain_context: str = Field(\"流感疫苗施打資格篩選系統\",
> env=\"IFL_DOMAIN_CONTEXT\")
>
> \# ── 領域型別定義（變數名 → Z3 型別）──
>
> domain_types: dict\[str, str\] = {
>
> \"age\": \"int\", \"high_risk\": \"bool\",
>
> \"days_since_last\": \"int\", \"egg_allergy\": \"bool\"
>
> }
>
> \@property
>
> def llm_backend(self) -\> LLMBackend:
>
> if self.llm_provider == \"openai\":
>
> return OpenAIBackend(self.llm_model, self.llm_api_key)
>
> elif self.llm_provider == \"anthropic\":
>
> return AnthropicBackend(self.llm_model, self.llm_api_key)
>
> raise ValueError(f\"不支援的 LLM 供應商：{self.llm_provider}\")
>
> \@property
>
> def domain_validator(self) -\> DomainValidator:
>
> return DomainValidator(DEFAULT_MEDICAL_RULES)

**8.1　.env 範本**

> \# .env.example（請勿提交到 Git）
>
> IFL_LLM_PROVIDER=openai
>
> IFL_LLM_MODEL=gpt-4o
>
> IFL_LLM_API_KEY=sk-\...
>
> IFL_SMT_TIMEOUT_MS=10000
>
> IFL_MAX_ITERATIONS=50
>
> IFL_FUNC_NAME=check_vaccine_eligibility
>
> IFL_DOMAIN_CONTEXT=流感疫苗施打資格篩選系統
>
> **⚠ 警告** .env 檔案必須加入 .gitignore。API key
> 不得出現在任何版本控制歷史記錄中。一旦洩漏，立即撤銷並重新生成。

**9　單元測試與整合測試規格**

以下列出每個模組必須通過的關鍵測試案例。使用 pytest，所有測試必須在 CI
中 100% 通過才允許合併 PR。

**9.1　Layer 1 單元測試**

  --------------- --------------------------------------- ------------------------------------------------------
  **模組**        **測試 ID**                             **測試說明**

  ASTParser       test_parse_basic_if                     最簡單的 if (x \> 0) 應識別 1 個 DecisionNode，1
                                                          個條件

  ASTParser       test_parse_nested_if                    巢狀 if 應識別 2 個 DecisionNode（外層 + 內層）

  ASTParser       test_parse_vaccine_logic                疫苗篩選邏輯應識別 1 個 DecisionNode，4 個原子條件

  CouplingGraph   test_or_coupling                        (A or B) and C 中，A 和 B 應為 OR 耦合，A 和 C 應為
                                                          AND 耦合

  CouplingGraph   test_negation_coupling                  not A 中，A 應正確識別且 negated=True

  ProbeInjector   test_probe_fires_despite_shortcircuit   (True or X) 中，即使短路，X 的探針仍必須觸發

  ProbeInjector   test_instrumented_semantics_unchanged   100
                                                          筆隨機輸入在原始模組和儀表板化模組的輸出必須完全相同

  MCDCMatrix      test_matrix_loss_zero_when_covered      手動構造滿足 100% MC/DC 的測試集，compute_loss()
                                                          必須回傳 0

  MCDCMatrix      test_incremental_update_o_k             逐一加入測試案例，每次更新耗時不超過 O(k\*m) 的 5 倍
  --------------- --------------------------------------- ------------------------------------------------------

**9.2　Layer 2 單元測試**

  ---------------------- ------------------------------- ------------------------------
  **模組**               **測試 ID**                     **測試說明**

  BoolDerivativeEngine   test_masking_detected           A and False → A 的導數為
                                                         0（被遮罩），必須正確識別

  BoolDerivativeEngine   test_not_masked                 (A or False) and True → A
                                                         的導數為 1，不被遮罩

  SMTSynthesizer         test_sat_returns_valid_model    疫苗邏輯的 c₂ F2T 缺口，Z3
                                                         必須在 10 秒內回傳 SAT，且 age
                                                         在 \[18,64\]

  SMTSynthesizer         test_unsat_mutually_exclusive   pregnant=True and sex=Male
                                                         的邏輯，Z3 必須回傳 UNSAT

  ASTToZ3Converter       test_compare_nodes              age \>= 65 轉為 z3 Int
                                                         表達式，Z3 可正確求解
  ---------------------- ------------------------------- ------------------------------

**9.3　Layer 3 單元測試**

  ------------------- -------------------------------------- ------------------------------
  **模組**            **測試 ID**                            **測試說明**

  PromptConstructor   test_prompt_contains_all_constraints   輸出提示必須包含所有 BoundSpec
                                                             的數值範圍

  PromptConstructor   test_prompt_under_2048_tokens          生成提示的估算 token 數必須 ≤
                                                             2048

  DomainValidator     test_reject_negative_age               age=-5 必須被拒絕，violations
                                                             非空

  DomainValidator     test_accept_edge_age                   age=0 和 age=130 必須通過驗證

  LLMSampler          test_parse_markdown_wrapped_json       LLM 回應包在 \`\`\`json \`\`\`
                                                             中，仍可正確解析

  LLMSampler          test_retry_on_parse_failure            Mock LLM 前 2 次回傳無效
                                                             JSON，第 3
                                                             次成功，最終應回傳正確結果
  ------------------- -------------------------------------- ------------------------------

**9.4　整合測試**

> \# tests/integration/test_vaccine_e2e.py
>
> def test_full_ifl_vaccine_logic():
>
> \"\"\"
>
> 端對端測試：從疫苗篩選原始碼出發，
>
> 驗證 IFL 系統能在 50 次迭代內達到 100% MC/DC。
>
> 此測試需要 LLM API key，預設在 CI 中 skip（除非設定 RUN_E2E=1）
>
> \"\"\"
>
> import pytest
>
> if not os.environ.get(\"RUN_E2E\"):
>
> pytest.skip(\"E2E tests skipped unless RUN_E2E=1\")
>
> config = IFLConfig(func_name=\"check_vaccine_eligibility\")
>
> orchestrator = IFLOrchestrator(config)
>
> result = orchestrator.run(\"tests/fixtures/vaccine_eligibility.py\")
>
> assert result.converged,
> f\"未收斂，最終覆蓋率：{result.final_coverage:.1%}\"
>
> assert result.final_coverage == 1.0
>
> assert result.iteration_count \<= 50
>
> assert len(result.infeasible_paths) == 0, \"疫苗邏輯不應有不可行路徑\"

**附錄 A　公開 API 介面規格（Python 型別標注版）**

以下是所有模組的完整公開介面，按呼叫順序排列。開發人員實作時的方法簽名必須完全一致，否則整合測試將失敗。

  ------------------------------------- --------------------------- ---------------------------
  **類別.方法**                         **簽名**                    **回傳類型**

  ASTParser.parse_file                  (filepath: str \| Path) -\> list\[DecisionNode\]
                                        list\[DecisionNode\]        

  CouplingGraphBuilder.build            (decision_id, root_expr,    list\[list\[str\|None\]\]
                                        conditions) -\>             
                                        list\[list\[str\|None\]\]   

  ProbeInjector.inject                  (source: str) -\> str       str（儀表板化原始碼）

  MCDCCoverageEngine.build_matrix       (cond_set, log) -\>         MCDCMatrix
                                        MCDCMatrix                  

  MCDCCoverageEngine.update             (matrix, log, new_test_id)  bool（L 是否下降）
                                        -\> bool                    

  BooleanDerivativeEngine.compute       (decision_node,             MaskingReport
                                        target_cond) -\>            
                                        MaskingReport               

  GapAnalyzer.analyze                   (matrix: MCDCMatrix) -\>    list\[GapEntry\]
                                        list\[GapEntry\]            

  SMTConstraintSynthesizer.synthesize   (decision_node, gap,        SMTResult
                                        domain_types) -\> SMTResult 

  BoundExtractor.extract                (z3_model, z3_vars,         list\[BoundSpec\]
                                        domain_types) -\>           
                                        list\[BoundSpec\]           

  PromptConstructor.build               (decision_node, gap,        str（提示字串）
                                        bound_specs, func_sig, ctx) 
                                        -\> str                     

  DomainValidator.validate              (test_case_json: str) -\>   ValidationResult
                                        ValidationResult            

  LLMSampler.sample                     (prompt: str) -\>           tuple\[dict,
                                        tuple\[dict,                ValidationResult\]
                                        ValidationResult\]          

  AcceptanceGate.evaluate               (matrix, log, new_test_id)  bool（是否接受）
                                        -\> bool                    

  IFLOrchestrator.run                   (source_path: str) -\>      IFLResult
                                        IFLResult                   
  ------------------------------------- --------------------------- ---------------------------

**附錄 B　資料庫 Schema（探針日誌持久化）**

開發環境使用 SQLite，正式環境切換為 PostgreSQL（同一 Schema）。

> \-- db/schema.sql
>
> CREATE TABLE IF NOT EXISTS probe_records (
>
> id INTEGER PRIMARY KEY AUTOINCREMENT,
>
> session_id TEXT NOT NULL, \-- IFLOrchestrator.run() 的執行 UUID
>
> test_id TEXT NOT NULL, \-- 例如 \"T3a9f12b8\"
>
> cond_id TEXT NOT NULL, \-- 例如 \"D1.c2\"
>
> value INTEGER NOT NULL, \-- 0 或 1
>
> decision INTEGER NOT NULL, \-- 0 或 1
>
> timestamp REAL NOT NULL, \-- Unix timestamp
>
> created_at TEXT DEFAULT (datetime(\"now\"))
>
> );
>
> CREATE INDEX idx_probe_session_test ON probe_records(session_id,
> test_id);
>
> CREATE INDEX idx_probe_cond ON probe_records(cond_id);
>
> CREATE TABLE IF NOT EXISTS ifl_sessions (
>
> session_id TEXT PRIMARY KEY,
>
> source_path TEXT NOT NULL,
>
> converged INTEGER NOT NULL DEFAULT 0,
>
> final_coverage REAL,
>
> iteration_count INTEGER,
>
> total_tokens INTEGER,
>
> started_at TEXT DEFAULT (datetime(\"now\")),
>
> finished_at TEXT
>
> );
>
> CREATE TABLE IF NOT EXISTS infeasible_paths (
>
> id INTEGER PRIMARY KEY AUTOINCREMENT,
>
> session_id TEXT NOT NULL,
>
> condition_id TEXT NOT NULL, \-- 例如 \"D1.c2\"
>
> unsat_core TEXT, \-- JSON array of UNSAT core clauses
>
> recorded_at TEXT DEFAULT (datetime(\"now\")),
>
> FOREIGN KEY (session_id) REFERENCES ifl_sessions(session_id)
>
> );

*--- 文件結束 ---*
