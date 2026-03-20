"""
LLM 語意合理性驗證腳本

目的：大量測試 LLM 語意合理性，收集足夠樣本來統計有效率。

流程：
1. 解析疫苗邏輯，取得所有缺口（k=5，共 10 個缺口）
2. 對每個缺口，讓 LLM 生成 10 個案例
3. 每個案例用 DomainValidator 驗證
4. 統計整體有效率和每個缺口的有效率
"""
from pathlib import Path
from ifl_mcdc.config import IFLConfig
from ifl_mcdc.layer1.ast_parser import ASTParser
from ifl_mcdc.layer2.gap_analyzer import GapAnalyzer
from ifl_mcdc.layer2.smt_synthesizer import SMTConstraintSynthesizer
from ifl_mcdc.layer3.prompt_builder import PromptConstructor
from ifl_mcdc.layer3.llm_sampler import LLMSampler
from ifl_mcdc.layer3.domain_validator import DomainValidator
from ifl_mcdc.models.coverage_matrix import MCDCMatrix
import json, os

N_SAMPLES_PER_GAP = 10  # 每個缺口生成幾個案例

def main():
    config = IFLConfig()
    parser = ASTParser()
    nodes = parser.parse_file('tests/fixtures/vaccine_eligibility.py')
    dn = nodes[0]

    # 建立空矩陣取得所有缺口
    matrix = MCDCMatrix(condition_set=dn.condition_set)
    analyzer = GapAnalyzer()
    gaps = analyzer.analyze(matrix)

    smt = SMTConstraintSynthesizer()
    prompt_builder = PromptConstructor()
    sampler = LLMSampler(config.llm_backend, config.domain_validator)
    validator = DomainValidator()

    results = []
    print(f"共 {len(gaps)} 個缺口，每個生成 {N_SAMPLES_PER_GAP} 個案例")
    print("=" * 60)

    for gap in gaps:
        print(f"\n缺口：{gap.condition_id} {gap.flip_direction}")

        # SMT 求解取得約束
        try:
            smt_result = smt.synthesize(dn, gap, config.domain_types)
            if not smt_result.satisfiable:
                print(f"  SMT UNSAT，跳過")
                continue
        except Exception as e:
            print(f"  SMT 失敗：{e}")
            continue

        # 建立 prompt
        p = prompt_builder.build(
            dn, gap, smt_result.bound_specs or [],
            config.func_signature, config.domain_context
        )

        # 生成 N_SAMPLES_PER_GAP 個案例
        gap_results = []
        for i in range(N_SAMPLES_PER_GAP):
            try:
                case, vr = sampler.sample(p)
                valid = vr.passed
                gap_results.append({
                    "case": case,
                    "valid": valid,
                    "violations": [v.description for v in vr.violations]
                })
                mark = "[OK]" if valid else "[NG]"
                print(f"  [{i+1:2d}] {mark} {case}")
                if not valid:
                    print(f"       違規：{vr.violations[0].description}")
            except Exception as e:
                print(f"  [{i+1:2d}] [NG] 失敗：{e}")
                gap_results.append({"case": None, "valid": False, "error": str(e)})

        valid_count = sum(1 for r in gap_results if r["valid"])
        rate = valid_count / len(gap_results) * 100
        print(f"  有效率：{valid_count}/{len(gap_results)} = {rate:.0f}%")

        results.append({
            "gap": f"{gap.condition_id}_{gap.flip_direction}",
            "valid_count": valid_count,
            "total": len(gap_results),
            "valid_rate": rate,
            "samples": gap_results
        })

    # 總結
    total_valid = sum(r["valid_count"] for r in results)
    total_all = sum(r["total"] for r in results)
    overall_rate = total_valid / total_all * 100 if total_all > 0 else 0

    print("\n" + "=" * 60)
    print("  LLM 語意有效性驗證摘要")
    print("=" * 60)
    print(f"  總生成案例數：{total_all}")
    print(f"  有效案例數：  {total_valid}")
    print(f"  整體有效率：  {overall_rate:.1f}%")
    print("=" * 60)

    # 儲存報告
    report = {"overall_valid_rate": overall_rate, "gaps": results}
    Path("llm_semantics_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"  報告已儲存至：llm_semantics_report.json")


if __name__ == "__main__":
    main()
