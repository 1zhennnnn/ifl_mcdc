"""
run_ifl_diversity.py

IFL 測試集多樣性驗證腳本（僅實驗組）。

對每個 Fixture 執行 N_RUNS 次 IFL，彙整全部生成案例，從 7 個面向
驗證測試集是否具備足夠多樣性（無偏向、無聚集、無重複）：

  D1  唯一性率     unique / total（重複案例比例）
  D2  布林平衡度   每個 bool 欄位的 True 比例（理想 ≈ 50%）
  D3  整數分布     min / max / mean / std / domain 覆蓋率
  D4  輸出平衡度   函式回傳 True 的比例（MC/DC 需 True/False 並存）
  D5  成對多樣性   所有測試對的平均標準化 Hamming 距離
  D6  條件激活率   每個原子條件的 True/False 觸發次數
  D7  邊界偏向     整數欄位中「恰等於 Z3 最小邊界」的比例（過高代表 Z3 聚集）

偏向判斷閾值：
  D1 唯一性率    ≥ 0.70  PASS（不重複率達 70% 以上）
  D2 布林平衡    True% ∈ [15%, 85%]  PASS（不偏向單一 bool 值）
  D3 整數覆蓋    domain 覆蓋率 ≥ 20%  PASS（整數不集中於一小段）
  D4 輸出平衡    True% ∈ [25%, 75%]  PASS（MC/DC 自然造成混合）
  D5 成對多樣性  ≥ 0.25  PASS（測試案例夠分散）
  D6 條件激活    每個條件都出現 True 且 False  PASS
  D7 邊界偏向    ≤ 70%   PASS（不超過 7 成案例使用邊界值）

執行方式：
  $env:IFL_LLM_API_KEY = "你的金鑰"
  $env:IFL_LLM_PROVIDER = "openai"
  $env:IFL_LLM_MODEL    = "gpt-4o"
  python run_ifl_diversity.py 2>&1 | Tee-Object -FilePath diversity_results.txt
"""
from __future__ import annotations

import importlib.util
import itertools
import json
import math
import os
import statistics
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env", override=False)
except ImportError:
    pass

from ifl_mcdc.config import IFLConfig
from ifl_mcdc.layer1.ast_parser import ASTParser
from ifl_mcdc.orchestrator import IFLOrchestrator
from validation_fixtures import ALL_SPECS, FixtureSpec

# ── 全域設定 ────────────────────────────────────────────────────
N_RUNS = 5          # 每個 Fixture 執行 IFL 的次數（較多以取得足夠樣本）

HAS_API_KEY = bool(os.environ.get("IFL_LLM_API_KEY", "").strip())
if not HAS_API_KEY:
    print("=" * 58)
    print("  錯誤：未設定 IFL_LLM_API_KEY，無法執行 IFL。")
    print("  設定後重新執行：")
    print("    $env:IFL_LLM_API_KEY = '你的金鑰'")
    print("=" * 58)
    raise SystemExit(1)

# ── 偏向判斷閾值 ────────────────────────────────────────────────
THRESH_UNIQUENESS   = 0.70   # D1
THRESH_BOOL_LO      = 0.15   # D2 lower
THRESH_BOOL_HI      = 0.85   # D2 upper
THRESH_INT_COVERAGE = 0.20   # D3
THRESH_OUTPUT_LO    = 0.25   # D4 lower
THRESH_OUTPUT_HI    = 0.75   # D4 upper
THRESH_PAIRWISE     = 0.25   # D5
THRESH_BOUNDARY     = 0.70   # D7


# ══════════════════════════════════════════════════════════════
#  Step 1：執行 IFL 並蒐集測試案例
# ══════════════════════════════════════════════════════════════


def _make_ifl_config(spec: FixtureSpec) -> IFLConfig:
    return IFLConfig(
        max_iterations=spec.max_ifl_iters,
        func_name=spec.func_name,
        func_signature=spec.func_sig,
        domain_context=spec.domain_ctx,
        domain_types=spec.domain_types,
        domain_bounds=spec.domain_bounds,
    )


def collect_ifl_cases(spec: FixtureSpec) -> tuple[list[dict], list[bool], list[int]]:
    """執行 N_RUNS 次 IFL，回傳 (所有案例, 覆蓋率列表, 迭代數列表)。

    案例 dict 已去除 __source / __test_id，僅保留欄位值。
    """
    config   = _make_ifl_config(spec)
    all_cases: list[dict] = []
    coverages: list[bool] = []
    iters:     list[int]  = []

    for run_idx in range(1, N_RUNS + 1):
        print(f"  Run {run_idx}/{N_RUNS} ...", end=" ", flush=True)
        result = IFLOrchestrator(config=config).run(spec.path)
        coverages.append(result.converged)
        iters.append(result.iteration_count)

        for tc in result.test_suite:
            clean = {k: v for k, v in tc.items() if not k.startswith("__")}
            if clean:
                all_cases.append(clean)
        print(
            f"收斂={result.converged}, "
            f"coverage={result.final_coverage:.1%}, "
            f"cases+={len(result.test_suite)}"
        )

    return all_cases, coverages, iters


# ══════════════════════════════════════════════════════════════
#  Step 2：載入 Fixture 函式，計算函式輸出
# ══════════════════════════════════════════════════════════════


def _load_fixture_fn(spec: FixtureSpec):  # type: ignore[return]
    """用 importlib 載入 Fixture 模組，回傳被測函式。"""
    mod_spec = importlib.util.spec_from_file_location("_fixture_mod", spec.path)
    mod = importlib.util.module_from_spec(mod_spec)  # type: ignore[arg-type]
    mod_spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return getattr(mod, spec.func_name)


def compute_outputs(cases: list[dict], fn) -> list[bool]:  # type: ignore[return]
    """對每個測試案例執行 Fixture 函式，回傳 True/False 輸出列表。"""
    outputs = []
    for c in cases:
        try:
            outputs.append(bool(fn(**c)))
        except Exception:
            outputs.append(False)
    return outputs


# ══════════════════════════════════════════════════════════════
#  Step 3：條件激活率（D6）
# ══════════════════════════════════════════════════════════════


def compute_condition_activation(
    cases: list[dict],
    spec: FixtureSpec,
) -> dict[str, dict[str, int]]:
    """用 eval 對每個案例評估每個原子條件，統計 True/False 次數。

    回傳 {condition_expression: {"true": n, "false": m}}
    """
    parser = ASTParser()
    nodes  = parser.parse_file(spec.path)
    dn     = nodes[0]

    activation: dict[str, dict[str, int]] = {}
    for cond in dn.condition_set.conditions:
        activation[cond.expression] = {"true": 0, "false": 0}

    for case in cases:
        for cond in dn.condition_set.conditions:
            try:
                raw = bool(eval(  # noqa: S307
                    cond.expression,
                    {"__builtins__": {}},
                    dict(case),
                ))
                val = (not raw) if cond.negated else raw
            except Exception:
                continue
            key = "true" if val else "false"
            activation[cond.expression][key] += 1

    return activation


# ══════════════════════════════════════════════════════════════
#  Step 4：多樣性指標計算
# ══════════════════════════════════════════════════════════════


def d1_uniqueness(cases: list[dict]) -> dict:
    """D1：唯一性率 = 不重複案例 / 總案例數。"""
    total  = len(cases)
    frozen = [json.dumps(c, sort_keys=True) for c in cases]
    unique = len(set(frozen))
    dups   = total - unique
    rate   = unique / total if total > 0 else 0.0
    return {
        "total":   total,
        "unique":  unique,
        "dups":    dups,
        "rate":    rate,
        "pass":    rate >= THRESH_UNIQUENESS,
    }


def d2_bool_balance(cases: list[dict], domain_types: dict[str, str]) -> dict:
    """D2：每個 bool 欄位的 True 比例。"""
    bool_fields = [f for f, t in domain_types.items() if t == "bool"]
    result: dict[str, dict] = {}
    for f in bool_fields:
        vals   = [c[f] for c in cases if f in c and isinstance(c[f], bool)]
        if not vals:
            continue
        true_n = sum(1 for v in vals if v)
        true_p = true_n / len(vals)
        result[f] = {
            "n":      len(vals),
            "true_n": true_n,
            "true_p": true_p,
            "pass":   THRESH_BOOL_LO <= true_p <= THRESH_BOOL_HI,
        }
    return result


def d3_int_distribution(
    cases: list[dict],
    domain_types: dict[str, str],
    domain_bounds: dict[str, list[int]],
) -> dict:
    """D3：每個 int 欄位的分布統計（min/max/mean/std/coverage）。"""
    int_fields = [f for f, t in domain_types.items() if t == "int"]
    result: dict[str, dict] = {}
    for f in int_fields:
        vals = [int(c[f]) for c in cases if f in c and not isinstance(c[f], bool)]
        if not vals:
            continue
        lo, hi       = domain_bounds.get(f, [0, max(vals) + 1])
        domain_range = hi - lo if hi > lo else 1
        val_range    = max(vals) - min(vals)
        coverage     = val_range / domain_range
        result[f] = {
            "n":        len(vals),
            "min":      min(vals),
            "max":      max(vals),
            "mean":     statistics.mean(vals),
            "std":      statistics.stdev(vals) if len(vals) > 1 else 0.0,
            "coverage": coverage,
            "pass":     coverage >= THRESH_INT_COVERAGE,
        }
    return result


def d4_output_balance(outputs: list[bool]) -> dict:
    """D4：函式輸出中 True 的比例。"""
    if not outputs:
        return {"true_p": 0.0, "pass": False}
    true_p = sum(outputs) / len(outputs)
    return {
        "total":  len(outputs),
        "true_n": sum(outputs),
        "true_p": true_p,
        "pass":   THRESH_OUTPUT_LO <= true_p <= THRESH_OUTPUT_HI,
    }


def d5_pairwise_diversity(
    cases: list[dict],
    domain_types: dict[str, str],
    domain_bounds: dict[str, list[int]],
) -> dict:
    """D5：所有測試對的平均標準化距離。

    bool 欄位：Hamming（0/1）
    int 欄位：|a-b| / domain_range
    最終距離：各欄位的加權平均（每欄位等權）
    """
    if len(cases) < 2:
        return {"avg_dist": 0.0, "n_pairs": 0, "pass": False}

    fields = list(domain_types.keys())

    def _dist(a: dict, b: dict) -> float:
        scores = []
        for f in fields:
            if f not in a or f not in b:
                continue
            t = domain_types[f]
            if t == "bool":
                scores.append(0.0 if a[f] == b[f] else 1.0)
            else:  # int / float
                lo, hi = domain_bounds.get(f, [0, 1])
                rng = hi - lo if hi > lo else 1
                scores.append(abs(int(a[f]) - int(b[f])) / rng)
        return statistics.mean(scores) if scores else 0.0

    # 抽樣：案例超過 80 時隨機取 80 個，避免 O(n²) 過慢
    sample = cases if len(cases) <= 80 else _reservoir_sample(cases, 80)
    pairs  = list(itertools.combinations(range(len(sample)), 2))
    dists  = [_dist(sample[i], sample[j]) for i, j in pairs]
    avg    = statistics.mean(dists) if dists else 0.0
    return {
        "avg_dist": avg,
        "min_dist": min(dists) if dists else 0.0,
        "n_pairs":  len(dists),
        "pass":     avg >= THRESH_PAIRWISE,
    }


def _reservoir_sample(lst: list, k: int) -> list:
    import random
    return random.sample(lst, k)


def d6_condition_activation(activation: dict[str, dict[str, int]]) -> dict:
    """D6：每個原子條件是否同時出現 True 與 False（MC/DC 基本要求）。"""
    result: dict[str, dict] = {}
    all_pass = True
    for expr, counts in activation.items():
        t, f   = counts["true"], counts["false"]
        ok     = t > 0 and f > 0
        all_pass = all_pass and ok
        result[expr] = {
            "true_n":  t,
            "false_n": f,
            "true_p":  t / (t + f) if (t + f) > 0 else 0.0,
            "pass":    ok,
        }
    return {"conditions": result, "all_pass": all_pass}


def d7_boundary_bias(
    cases: list[dict],
    domain_types: dict[str, str],
    domain_bounds: dict[str, list[int]],
    spec: FixtureSpec,
) -> dict:
    """D7：整數欄位中「恰等於某個 MC/DC 邊界臨界值」的比例。

    從 fixture 的條件表達式解析出所有整數比較閾值（如 age>=65 → 65, 64），
    統計案例值是否恰好落在這些邊界（±1 範圍內）。
    比例過高代表 Z3 總是回傳最小解，測試集缺乏多樣性。
    """
    # 解析條件中的整數閾值
    parser   = ASTParser()
    nodes    = parser.parse_file(spec.path)
    dn       = nodes[0]

    # 從條件表達式中提取閾值（只處理 int 欄位）
    thresholds: dict[str, set[int]] = {}
    import ast as _ast
    for cond in dn.condition_set.conditions:
        try:
            tree = _ast.parse(cond.expression, mode="eval")
            body = tree.body
            if isinstance(body, _ast.Compare):
                # 取比較右側的常數值
                for comp in body.comparators:
                    if isinstance(comp, _ast.Constant) and isinstance(comp.value, int):
                        # 找出涉及哪個 int 欄位
                        if isinstance(body.left, _ast.Name):
                            var = body.left.id
                            if domain_types.get(var) == "int":
                                thresholds.setdefault(var, set())
                                thresholds[var].add(comp.value)
                                thresholds[var].add(comp.value - 1)  # 邊界 -1
        except Exception:
            continue

    if not thresholds:
        return {"boundary_ratio": 0.0, "pass": True, "detail": {}}

    int_fields = [f for f, t in domain_types.items() if t == "int"]
    detail: dict[str, dict] = {}
    total_vals = 0
    boundary_vals = 0

    for f in int_fields:
        if f not in thresholds:
            continue
        vals = [int(c[f]) for c in cases if f in c and not isinstance(c[f], bool)]
        if not vals:
            continue
        bdry_set = thresholds[f]
        on_boundary = sum(1 for v in vals if v in bdry_set)
        total_vals    += len(vals)
        boundary_vals += on_boundary
        detail[f] = {
            "n":            len(vals),
            "boundary_n":   on_boundary,
            "boundary_p":   on_boundary / len(vals),
            "thresholds":   sorted(bdry_set),
        }

    ratio = boundary_vals / total_vals if total_vals > 0 else 0.0
    return {
        "boundary_ratio": ratio,
        "boundary_n":     boundary_vals,
        "total_n":        total_vals,
        "pass":           ratio <= THRESH_BOUNDARY,
        "detail":         detail,
    }


# ══════════════════════════════════════════════════════════════
#  Step 5：報告輸出
# ══════════════════════════════════════════════════════════════

_PASS = "PASS"
_FAIL = "FAIL"
_W    = 58  # 報告寬度


def _mark(ok: bool) -> str:
    return f"[{_PASS}]" if ok else f"[{_FAIL}]"


def print_diversity_report(
    spec: FixtureSpec,
    cases: list[dict],
    coverages: list[bool],
    iters: list[int],
    d1: dict,
    d2: dict,
    d3: dict,
    d4: dict,
    d5: dict,
    d6: dict,
    d7: dict,
) -> None:
    W = _W
    print(f"\n{'='*W}")
    print(f"  多樣性報告：{spec.label}")
    print(f"  N_RUNS={N_RUNS}，總案例={d1['total']}，"
          f"收斂率={sum(coverages)/len(coverages):.0%}，"
          f"平均迭代={statistics.mean(iters):.1f}")
    print(f"{'='*W}")

    # ── D1 唯一性 ──────────────────────────────────────────────
    print(f"\n  D1 唯一性率        {_mark(d1['pass'])}")
    print(f"     total={d1['total']}  unique={d1['unique']}  dups={d1['dups']}")
    print(f"     唯一率={d1['rate']:.1%}  (閾值 >= {THRESH_UNIQUENESS:.0%})")

    # ── D2 布林平衡 ────────────────────────────────────────────
    all_d2_pass = all(v["pass"] for v in d2.values())
    print(f"\n  D2 布林欄位平衡    {_mark(all_d2_pass)}")
    print(f"     {'欄位':<22} {'True%':>7}  {'n':>5}  {'判定':>6}")
    print(f"     {'-'*44}")
    for field, info in d2.items():
        bias = "偏高" if info["true_p"] > THRESH_BOOL_HI else ("偏低" if info["true_p"] < THRESH_BOOL_LO else "正常")
        mark = _mark(info["pass"])
        print(f"     {field:<22} {info['true_p']:>6.1%}  {info['n']:>5}  {mark} {bias}")

    # ── D3 整數分布 ────────────────────────────────────────────
    all_d3_pass = all(v["pass"] for v in d3.values()) if d3 else True
    print(f"\n  D3 整數欄位分布    {_mark(all_d3_pass)}")
    if d3:
        print(f"     {'欄位':<18} {'min':>8} {'max':>8} {'mean':>8} {'std':>8} {'覆蓋%':>7}  {'判定':>6}")
        print(f"     {'-'*64}")
        for field, info in d3.items():
            lo, hi = spec.domain_bounds.get(field, [0, 0])
            print(
                f"     {field:<18} {info['min']:>8} {info['max']:>8}"
                f" {info['mean']:>8.1f} {info['std']:>8.1f}"
                f" {info['coverage']:>6.1%}  {_mark(info['pass'])}"
            )
            print(f"     {'':18} domain=[{lo}, {hi}]")

    # ── D4 輸出平衡 ────────────────────────────────────────────
    print(f"\n  D4 函式輸出平衡    {_mark(d4['pass'])}")
    print(f"     True={d4['true_n']}/{d4['total']}={d4['true_p']:.1%}  "
          f"(閾值 [{THRESH_OUTPUT_LO:.0%}, {THRESH_OUTPUT_HI:.0%}])")

    # ── D5 成對多樣性 ──────────────────────────────────────────
    print(f"\n  D5 成對多樣性分數  {_mark(d5['pass'])}")
    print(f"     avg_dist={d5['avg_dist']:.4f}  min_dist={d5['min_dist']:.4f}"
          f"  pairs={d5['n_pairs']}")
    print(f"     (閾值 >= {THRESH_PAIRWISE:.2f}；0=完全相同，1=完全相異)")

    # ── D6 條件激活 ────────────────────────────────────────────
    print(f"\n  D6 條件激活覆蓋    {_mark(d6['all_pass'])}")
    print(f"     {'條件表達式':<36} {'True':>6} {'False':>6} {'True%':>7}  {'判定':>6}")
    print(f"     {'-'*62}")
    for expr, info in d6["conditions"].items():
        short = (expr[:33] + "...") if len(expr) > 36 else expr
        print(
            f"     {short:<36} {info['true_n']:>6} {info['false_n']:>6}"
            f" {info['true_p']:>6.1%}  {_mark(info['pass'])}"
        )

    # ── D7 邊界偏向 ────────────────────────────────────────────
    print(f"\n  D7 整數邊界偏向    {_mark(d7['pass'])}")
    if d7.get("detail"):
        print(f"     整體邊界率={d7['boundary_ratio']:.1%}  "
              f"boundary_n={d7['boundary_n']}/{d7['total_n']}"
              f"  (閾值 <= {THRESH_BOUNDARY:.0%})")
        for f, info in d7["detail"].items():
            print(f"     {f}: boundary_n={info['boundary_n']}/{info['n']} "
                  f"={info['boundary_p']:.1%}  "
                  f"閾值={info['thresholds']}")
    else:
        print("     無整數比較條件（或無法解析）")

    # ── 綜合判定 ───────────────────────────────────────────────
    all_pass = (
        d1["pass"]
        and all_d2_pass
        and all_d3_pass
        and d4["pass"]
        and d5["pass"]
        and d6["all_pass"]
        and d7["pass"]
    )
    print(f"\n  {'='*W}")
    verdict = _PASS if all_pass else _FAIL
    print(f"  綜合多樣性判定：[{verdict}]")
    print(f"  {'='*W}")


# ══════════════════════════════════════════════════════════════
#  主執行流程
# ══════════════════════════════════════════════════════════════

summary_rows: list[dict] = []

for spec in ALL_SPECS:
    print(f"\n\n{'#'*60}")
    print(f"#  Fixture：{spec.label}")
    print(f"#  執行 {N_RUNS} 次 IFL（max_ifl_iters={spec.max_ifl_iters}）")
    print(f"{'#'*60}")

    # Step 1：蒐集案例
    cases, coverages, iters = collect_ifl_cases(spec)
    print(f"\n  >> 總案例數={len(cases)}，收斂={sum(coverages)}/{N_RUNS}")

    # Step 2：載入函式，計算輸出
    fn      = _load_fixture_fn(spec)
    outputs = compute_outputs(cases, fn)

    # Step 3：條件激活
    activation = compute_condition_activation(cases, spec)

    # Step 4：計算各指標
    d1 = d1_uniqueness(cases)
    d2 = d2_bool_balance(cases, spec.domain_types)
    d3 = d3_int_distribution(cases, spec.domain_types, spec.domain_bounds)
    d4 = d4_output_balance(outputs)
    d5 = d5_pairwise_diversity(cases, spec.domain_types, spec.domain_bounds)
    d6 = d6_condition_activation(activation)
    d7 = d7_boundary_bias(cases, spec.domain_types, spec.domain_bounds, spec)

    # Step 5：報告
    print_diversity_report(spec, cases, coverages, iters, d1, d2, d3, d4, d5, d6, d7)

    all_pass = (
        d1["pass"]
        and all(v["pass"] for v in d2.values())
        and (all(v["pass"] for v in d3.values()) if d3 else True)
        and d4["pass"]
        and d5["pass"]
        and d6["all_pass"]
        and d7["pass"]
    )
    summary_rows.append({
        "label":        spec.label,
        "k":            spec.expected_k,
        "n_cases":      d1["total"],
        "convergence":  f"{sum(coverages)}/{N_RUNS}",
        "uniqueness":   d1["rate"],
        "pairwise":     d5["avg_dist"],
        "output_true":  d4["true_p"],
        "boundary":     d7["boundary_ratio"],
        "all_pass":     all_pass,
    })


# ── 總覽表 ─────────────────────────────────────────────────────
print(f"\n\n{'#'*70}")
print(f"#  跨 Fixture 多樣性總覽（N_RUNS={N_RUNS}）")
print(f"{'#'*70}")
print(f"  {'Fixture':<28} {'k':>3} {'案例':>5} {'收斂':>5}  "
      f"{'唯一%':>6}  {'成對D':>6}  {'輸出T%':>7}  {'邊界%':>6}  {'判定':>6}")
print(f"  {'-'*75}")
for row in summary_rows:
    short = row["label"].split("（")[0]
    print(
        f"  {short:<28} {row['k']:>3} {row['n_cases']:>5} {row['convergence']:>5}  "
        f"{row['uniqueness']:>5.1%}  {row['pairwise']:>6.4f}  "
        f"{row['output_true']:>6.1%}  {row['boundary']:>5.1%}  "
        f"{_mark(row['all_pass'])}"
    )
