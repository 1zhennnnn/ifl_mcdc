"""Re-run random + smt_only experiments."""
from __future__ import annotations
import json, time
from pathlib import Path
from ifl_mcdc.config import IFLConfig
from ifl_mcdc.layer3.llm_sampler import MockLLMBackend
from ifl_mcdc.orchestrator import IFLOrchestrator

FIXTURES = {
    "k5_vaccine": {
        "path": Path("tests/fixtures/vaccine_eligibility.py"),
        "func": "check_vaccine_eligibility",
        "domain_types": {"age":"int","high_risk":"bool","days_since_last":"int","egg_allergy":"bool"},
    },
    "k8_drug": {
        "path": Path("tests/fixtures/complex_medical_logic.py"),
        "func": "check_drug_interaction",
        "domain_types": {"age":"int","renal_failure":"bool","liver_disease":"bool",
                         "taking_warfarin":"bool","taking_aspirin":"bool",
                         "systolic_bp":"int","heart_rate":"int","is_pregnant":"bool"},
    },
    "k9_surgery": {
        "path": Path("tests/fixtures/surgery_risk.py"),
        "func": "check_surgery_risk",
        "domain_types": {"age":"int","obese":"bool","has_diabetes":"bool",
                         "has_hypertension":"bool","is_smoker":"bool","low_hemoglobin":"bool",
                         "low_platelets":"bool","cardiac_history":"bool","has_copd":"bool"},
    },
    "k10_icu": {
        "path": Path("tests/fixtures/icu_admission.py"),
        "func": "check_icu_admission",
        "domain_types": {"age":"int","low_bp":"bool","high_heart_rate":"bool",
                         "high_resp_rate":"bool","high_temp":"bool","low_gcs":"bool",
                         "low_oxygen":"bool","low_urine":"bool",
                         "high_creatinine":"bool","sepsis":"bool"},
    },
}
N = 3
new_runs: dict[str, dict[str, list[dict]]] = {}

for name, fix in FIXTURES.items():
    new_runs[name] = {}
    for mode in ("random", "smt_only"):
        runs = []
        for i in range(N):
            if mode == "random":
                cfg = IFLConfig(max_iterations=0, func_name=fix["func"],
                                domain_types=fix["domain_types"])
                orch = IFLOrchestrator(cfg, MockLLMBackend([]))
            else:
                cfg = IFLConfig(max_iterations=50, func_name=fix["func"],
                                domain_types=fix["domain_types"])
                orch = IFLOrchestrator(cfg, MockLLMBackend([]))
            t0 = time.time()
            r = orch.run(fix["path"])
            el = round(time.time() - t0, 3)
            runs.append({
                "name": name, "mode": mode,
                "converged": r.converged,
                "final_coverage": r.final_coverage,
                "iteration_count": r.iteration_count,
                "total_tokens": r.total_tokens,
                "elapsed_seconds": el,
                "loss_history": r.loss_history,
                "infeasible_paths": r.infeasible_paths,
            })
            print(f"  {name}/{mode} #{i+1}: conv={r.converged} "
                  f"cov={r.final_coverage:.0%} iters={r.iteration_count} t={el:.3f}s")
        new_runs[name][mode] = runs

Path("_new_runs.json").write_text(
    json.dumps(new_runs, ensure_ascii=False, indent=2), encoding="utf-8"
)
print("done -> _new_runs.json")
