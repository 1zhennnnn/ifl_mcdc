"""
run_ifl_diversity.py

IFL 測試集多樣性驗證腳本（D1~D8 全新指標體系）。

對每個 Fixture 執行 N_RUNS 次 IFL，彙整全部生成案例，
以 8 個面向驗證測試集是否具備足夠多樣性：

  D1  唯一性率     unique/total ≥ 0.70
  D2  函式輸出平衡 True% ∈ [25%, 75%]
  D3  條件激活覆蓋 每個條件 True% ∈ [5%, 95%]
  D4  整數邊界偏向 boundary_ratio ≤ 0.70
  D5  Shannon Entropy H_n ≥ 0.72（Shannon 1948）
  D6  Bin Coverage  bins_covered/n_bins ≥ 0.80（Sturges' Rule）
  D7  Bootstrap KS  p > 0.05（Kolmogorov-Smirnov）
  D8  Wasserstein   W ≤ 0.15（Wasserstein 1969）

執行方式：
  $env:IFL_LLM_API_KEY = "你的金鑰"
  python run_ifl_diversity.py
  python run_ifl_diversity.py --compare            # 含隨機基線對比
  python run_ifl_diversity.py --n-bootstrap 500   # 調整 D7 bootstrap 次數
  python run_ifl_diversity.py 2>&1 | Tee-Object -FilePath diversity_results.txt
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import statistics
from datetime import datetime
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env", override=False)
except ImportError:
    pass

from ifl_mcdc.config import IFLConfig
from ifl_mcdc.orchestrator import IFLOrchestrator
from validation_fixtures import ALL_SPECS, FixtureSpec
from diversity_reporter import (
    analyze_structural_bias,
    compute_condition_activation,
    compute_d1, compute_d2, compute_d3, compute_d4,
    compute_d5, compute_d6, compute_d7, compute_d8,
    print_report,
    THRESH_D1_UNIQUENESS, THRESH_D2_OUTPUT_LO, THRESH_D2_OUTPUT_HI,
    THRESH_D4_BOUNDARY, THRESH_D5_ENTROPY, THRESH_D6_BIN_COV,
    THRESH_D7_KS_P, THRESH_D8_WASS,
)
from statistical_validator import (
    generate_random_baseline,
    compare_before_after,
    print_comparison_report,
)

# ── 全域設定 ────────────────────────────────────────────────────
N_RUNS = 5   # 每個 Fixture 執行 IFL 的次數

HAS_API_KEY = bool(os.environ.get("IFL_LLM_API_KEY", "").strip())
if not HAS_API_KEY:
    print("=" * 58)
    print("  錯誤：未設定 IFL_LLM_API_KEY，無法執行 IFL。")
    print("  設定後重新執行：")
    print("    $env:IFL_LLM_API_KEY = '你的金鑰'")
    print("=" * 58)
    raise SystemExit(1)


# ── argparse ────────────────────────────────────────────────────
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="IFL 多樣性驗證腳本（D1~D8）")
    parser.add_argument(
        "--compare", action="store_true",
        help="在每個 Fixture 結束後輸出純隨機基線對比（使用 Wilcoxon + Holm-Bonferroni）",
    )
    parser.add_argument(
        "--n-bootstrap", type=int, default=1000,
        metavar="N",
        help="D7 Bootstrap KS 的重抽樣次數（預設 1000）",
    )
    return parser.parse_args()


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
        fixture_name=spec.fixture_name,
        scenarios=spec.scenarios,
    )


def collect_ifl_cases(
    spec: FixtureSpec,
) -> tuple[list[dict], list[bool], list[int], list[dict]]:
    """執行 N_RUNS 次 IFL，回傳 (所有案例, 收斂列表, 迭代數列表, 含元資料詳細列表)。

    案例 dict 已去除 __source / __test_id，僅保留欄位值。
    詳細案例另含 _run, _source, _fixture 元資料。
    """
    config     = _make_ifl_config(spec)
    all_cases: list[dict]  = []
    detailed:  list[dict]  = []
    coverages: list[bool]  = []
    iters:     list[int]   = []

    for run_idx in range(1, N_RUNS + 1):
        print(f"  Run {run_idx}/{N_RUNS} ...", end=" ", flush=True)
        result = IFLOrchestrator(config=config).run(spec.path)
        coverages.append(result.converged)
        iters.append(result.iteration_count)

        for tc in result.all_generated_cases:
            source = tc.get("__source", "")
            clean  = {k: v for k, v in tc.items() if not k.startswith("__")}
            if clean:
                all_cases.append(clean)
                detailed.append({
                    "_fixture": spec.label,
                    "_run":     run_idx,
                    "_source":  source,
                    **clean,
                })
        print(
            f"收斂={result.converged}, "
            f"coverage={result.final_coverage:.1%}, "
            f"cases+={len(result.test_suite)}"
        )

    return all_cases, coverages, iters, detailed


# ══════════════════════════════════════════════════════════════
#  Step 2：載入 Fixture 函式，計算函式輸出
# ══════════════════════════════════════════════════════════════

def _load_fixture_fn(spec: FixtureSpec):  # type: ignore[return]
    """用 importlib 載入 Fixture 模組，回傳被測函式。"""
    mod_spec = importlib.util.spec_from_file_location("_fixture_mod", spec.path)
    mod      = importlib.util.module_from_spec(mod_spec)  # type: ignore[arg-type]
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
#  Step 3：--compare 隨機基線對比
# ══════════════════════════════════════════════════════════════

def _compute_metric_scores(
    cases: list[dict],
    outputs: list[bool],
    d1: dict, d2: dict, d3: dict, d4: dict,
    d5: dict, d6: dict, d7: dict, d8: dict,
) -> list[float]:
    """從 D1~D8 結果中提取數值分數列表（用於統計比較）。"""
    scores = [
        d1.get("rate",          0.0),           # D1 唯一率
        d2.get("true_p",        0.0),           # D2 輸出 True%（理想 0.5）
        _d3_score(d3),                           # D3 條件激活分數
        1.0 - d4.get("boundary_ratio", 1.0),    # D4 邊界偏向（越低越好，取反）
        d5.get("avg_H_n",       0.0),           # D5 Entropy
        d6.get("avg_coverage",  0.0),           # D6 Bin Coverage
        d7.get("fields", {}) and _d7_mean_p(d7) or 1.0,   # D7 KS p-value 平均（越高越好）
        1.0 - d8.get("max_W",   1.0),           # D8 Wasserstein（越低越好，取反）
    ]
    return scores


def _d3_score(d3: dict) -> float:
    """D3：計算所有條件 True% 與 0.5 的平均接近程度（越接近 0.5 越好）。"""
    conds = d3.get("conditions", {})
    if not conds:
        return 0.0
    scores = [1.0 - abs(v["true_p"] - 0.5) * 2 for v in conds.values()]
    return statistics.mean(scores)


def _d7_mean_p(d7: dict) -> float:
    """D7：計算所有欄位 p_value 的平均值。"""
    fields = d7.get("fields", {})
    if not fields:
        return 1.0
    return statistics.mean(v["p_value"] for v in fields.values())


METRIC_NAMES = ["D1-唯一率", "D2-輸出平衡", "D3-條件激活", "D4-邊界偏向",
                "D5-Entropy", "D6-BinCov",  "D7-KS_p",    "D8-Wasserstein"]


def _compare_with_baseline(
    spec: FixtureSpec,
    ifl_cases: list[dict],
    ifl_outputs: list[bool],
    d1: dict, d2: dict, d3: dict, d4: dict,
    d5: dict, d6: dict, d7: dict, d8: dict,
    n_bootstrap: int,
) -> None:
    """生成隨機基線並與 IFL 對比，輸出 Wilcoxon 報告。"""
    print(f"\n  [隨機基線對比] 生成純隨機案例...")
    baseline_cases = generate_random_baseline(spec, n_runs=N_RUNS, cases_per_run=30)
    base_clean     = [{k: v for k, v in c.items() if not k.startswith("_")}
                      for c in baseline_cases]

    fn           = _load_fixture_fn(spec)
    base_outputs = compute_outputs(base_clean, fn)
    base_activ   = compute_condition_activation(base_clean, spec)

    bd1 = compute_d1(base_clean)
    bd2 = compute_d2(base_outputs)
    bd3 = compute_d3(base_activ)
    bd4 = compute_d4(base_clean, spec.domain_types, spec.domain_bounds, spec)
    bd5 = compute_d5(base_clean, spec.domain_types, spec.domain_bounds)
    bd6 = compute_d6(base_clean, spec.domain_types, spec.domain_bounds)
    bd7 = compute_d7(base_clean, spec.domain_types, spec.domain_bounds, n_bootstrap)
    bd8 = compute_d8(base_clean, spec.domain_types, spec.domain_bounds)

    before_scores = _compute_metric_scores(
        base_clean, base_outputs, bd1, bd2, bd3, bd4, bd5, bd6, bd7, bd8
    )
    after_scores = _compute_metric_scores(
        ifl_cases, ifl_outputs, d1, d2, d3, d4, d5, d6, d7, d8
    )

    comp = compare_before_after(before_scores, after_scores, METRIC_NAMES)
    print_comparison_report(
        comp,
        title=f"IFL vs 隨機基線（{spec.label}）",
    )


# ══════════════════════════════════════════════════════════════
#  主執行流程
# ══════════════════════════════════════════════════════════════

def main() -> None:
    args               = _parse_args()
    n_bootstrap        = args.n_bootstrap
    do_compare         = args.compare
    summary_rows:      list[dict] = []
    all_detailed_cases:list[dict] = []
    results_dir        = Path("results")
    results_dir.mkdir(exist_ok=True)

    for spec in ALL_SPECS:
        print(f"\n\n{'#'*62}")
        print(f"#  Fixture：{spec.label}")
        print(f"#  執行 {N_RUNS} 次 IFL（max_ifl_iters={spec.max_ifl_iters}）")
        print(f"{'#'*62}")

        # Step 1：蒐集案例
        cases, coverages, iters, detailed = collect_ifl_cases(spec)
        all_detailed_cases.extend(detailed)
        print(f"\n  >> 總案例數={len(cases)}，收斂={sum(coverages)}/{N_RUNS}")

        # Step 2：函式輸出
        fn      = _load_fixture_fn(spec)
        outputs = compute_outputs(cases, fn)

        # Step 3：條件激活
        activation = compute_condition_activation(cases, spec)

        # Step 4：D1~D8
        d1 = compute_d1(cases)
        d2 = compute_d2(outputs)
        d3 = compute_d3(activation)
        d4 = compute_d4(cases, spec.domain_types, spec.domain_bounds, spec)
        d5 = compute_d5(cases, spec.domain_types, spec.domain_bounds)
        d6 = compute_d6(cases, spec.domain_types, spec.domain_bounds)
        d7 = compute_d7(cases, spec.domain_types, spec.domain_bounds, n_bootstrap)
        d8 = compute_d8(cases, spec.domain_types, spec.domain_bounds)

        # 結構性偏向分析
        bias = analyze_structural_bias(spec)

        # Step 5：報告
        print_report(
            spec=spec, n_runs=N_RUNS, cases=cases, outputs=outputs,
            coverages=coverages, iters=iters,
            d1=d1, d2=d2, d3=d3, d4=d4,
            d5=d5, d6=d6, d7=d7, d8=d8,
            bias=bias,
        )

        # Step 6：--compare 基線對比
        if do_compare:
            _compare_with_baseline(
                spec, cases, outputs, d1, d2, d3, d4, d5, d6, d7, d8, n_bootstrap
            )

        # 綜合判定
        all_pass = (
            d1["pass"]
            and d2["pass"]
            and d3["all_pass"]
            and d4["pass"]
            and d5.get("pass", True)
            and d6.get("pass", True)
            and d7.get("pass", True)
            and d8.get("pass", True)
        )
        summary_rows.append({
            "label":        spec.label,
            "k":            spec.expected_k,
            "n_cases":      d1["total"],
            "convergence":  f"{sum(coverages)}/{N_RUNS}",
            "D1_uniqueness": round(d1["rate"], 3),
            "D2_output_tp":  round(d2["true_p"], 3),
            "D3_cond_pass":  d3["all_pass"],
            "D4_boundary":   round(d4.get("boundary_ratio", 0.0), 3),
            "D5_entropy":    round(d5.get("avg_H_n", 0.0), 3),
            "D6_bincov":     round(d6.get("avg_coverage", 0.0), 3),
            "D7_ks_pass":    d7.get("pass", True),
            "D8_wass":       round(d8.get("max_W", 0.0), 3),
            "all_pass":      all_pass,
            # 詳細資料供 JSON 輸出
            "d1": d1, "d2": d2, "d3": d3, "d4": d4,
            "d5": d5, "d6": d6, "d7": d7, "d8": d8,
            "bias": bias,
        })

        # Step 7：JSON 輸出（per-fixture）
        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        fix_slug = spec.label.split("（")[0].replace(" ", "_")
        json_path = results_dir / f"diversity_report_{fix_slug}_{ts}.json"
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "fixture":   spec.label,
                    "n_runs":    N_RUNS,
                    "n_cases":   len(cases),
                    "convergence": f"{sum(coverages)}/{N_RUNS}",
                    "metrics":   {
                        "d1": d1, "d2": d2, "d3": d3, "d4": d4,
                        "d5": d5, "d6": d6, "d7": d7, "d8": d8,
                    },
                    "structural_bias": bias,
                    "all_pass":  all_pass,
                },
                f,
                ensure_ascii=False,
                indent=2,
                default=str,
            )
        print(f"\n  [JSON] 已匯出 -> {json_path}")

    # ── 跨 Fixture 總覽表 ────────────────────────────────────────
    W = 105
    print(f"\n\n{'#'*W}")
    print(f"#  跨 Fixture 多樣性總覽（N_RUNS={N_RUNS}，n_bootstrap={n_bootstrap}）")
    print(f"{'#'*W}")
    print(
        f"  {'Fixture':<28} {'k':>3} {'案例':>5} {'收斂':>5}  "
        f"{'D1唯一%':>7}  {'D2輸出T%':>8}  {'D3激活':>6}  "
        f"{'D4邊界%':>7}  {'D5 H_n':>7}  {'D6覆蓋%':>7}  "
        f"{'D7 KS':>6}  {'D8 W':>6}  {'判定':>6}"
    )
    print(f"  {'-'*W}")
    for row in summary_rows:
        short = row["label"].split("（")[0]
        d3ok  = "[PASS]" if row["D3_cond_pass"] else "[FAIL]"
        d7ok  = "[PASS]" if row["D7_ks_pass"]   else "[FAIL]"
        mark  = "[PASS]" if row["all_pass"]      else "[FAIL]"
        print(
            f"  {short:<28} {row['k']:>3} {row['n_cases']:>5} {row['convergence']:>5}  "
            f"{row['D1_uniqueness']:>6.1%}  {row['D2_output_tp']:>7.1%}  "
            f"{d3ok:>6}  {row['D4_boundary']:>6.1%}  "
            f"{row['D5_entropy']:>7.4f}  {row['D6_bincov']:>6.1%}  "
            f"{d7ok:>6}  {row['D8_wass']:>6.4f}  {mark}"
        )

    # ── 匯出全部生成案例 JSON ─────────────────────────────────────
    all_json_path = Path("ifl_diversity_cases.json")
    with all_json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "meta": {
                    "n_runs":       N_RUNS,
                    "total_cases":  len(all_detailed_cases),
                    "fixtures":     [s.label for s in ALL_SPECS],
                    "timestamp":    datetime.now().isoformat(),
                },
                "cases": all_detailed_cases,
            },
            f,
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    print(f"\n[JSON] 已匯出 {len(all_detailed_cases)} 個案例 -> {all_json_path.resolve()}")


if __name__ == "__main__":
    main()
