"""
邊界萃取器：從 Z3 模型提取 BoundSpec 列表。

TC-U-34: BoundExtractor——整數型邊界萃取
TC-U-35: BoundExtractor——布林型合法集合萃取
TC-U-36: BoundSpec 不可為空區間
"""
from __future__ import annotations

import z3

from ifl_mcdc.models.smt_models import BoundSpec


class BoundExtractor:
    """從 Z3 SAT model 萃取每個變數的 BoundSpec。

    int  型：interval = (model_val, model_val + 10)（從 model_val 起向上 10，確保 LLM 選到有效值）
    bool 型：valid_set = frozenset({bool(model_val)})
    Z3 未約束變數（model[var] 為 None）：interval = None
    """

    def extract(
        self,
        z3_model: z3.ModelRef,
        z3_vars: dict[str, object],
        domain_types: dict[str, str],
        domain_bounds: dict[str, list[int]] | None = None,
    ) -> list[BoundSpec]:
        """從 Z3 model 萃取 BoundSpec 列表。

        Args:
            z3_model: Z3 SAT model（s.model()）。
            z3_vars: var_name → Z3 變數的映射。
            domain_types: var_name → "int" | "bool" | "float" 的映射。
            domain_bounds: var_name → [min, max]。若有設定，int 型 interval 的
                           下界 clamp 到 lo，上界 clamp 到 hi。

        Returns:
            每個變數一個 BoundSpec 的列表。
        """
        specs: list[BoundSpec] = []
        for var_name, z3_var in z3_vars.items():
            var_type = domain_types.get(var_name, "int")
            model_val = z3_model[z3_var]

            if var_type == "bool":
                if model_val is None:
                    # 未約束的 bool 變數：告知 LLM 必須是 bool 值（True 或 False）
                    specs.append(
                        BoundSpec(
                            var_name=var_name,
                            var_type="bool",
                            interval=None,
                            valid_set=frozenset({True, False}),
                        )
                    )
                else:
                    bool_val = bool(z3.is_true(model_val))
                    specs.append(
                        BoundSpec(
                            var_name=var_name,
                            var_type="bool",
                            interval=None,
                            valid_set=frozenset({bool_val}),
                        )
                    )

            elif var_type == "int":
                if model_val is None:
                    specs.append(
                        BoundSpec(
                            var_name=var_name,
                            var_type="int",
                            interval=None,
                            valid_set=None,
                        )
                    )
                else:
                    try:
                        int_val = int(str(model_val))
                    except (ValueError, TypeError):
                        int_val = 0
                    if domain_bounds and var_name in domain_bounds:
                        lo, hi = domain_bounds[var_name][0], domain_bounds[var_name][1]
                        interval: tuple[float, float] = (
                            float(max(lo, int_val)),
                            float(min(hi, int_val + 10)),
                        )
                    else:
                        interval = (float(int_val), float(int_val + 10))
                    specs.append(
                        BoundSpec(
                            var_name=var_name,
                            var_type="int",
                            interval=interval,
                            valid_set=None,
                        )
                    )

            else:  # float / real
                if model_val is None:
                    specs.append(
                        BoundSpec(
                            var_name=var_name,
                            var_type="float",
                            interval=None,
                            valid_set=None,
                        )
                    )
                else:
                    try:
                        float_val = float(z3.RealVal(str(model_val)).as_decimal(10).rstrip("?"))
                    except (ValueError, ArithmeticError):
                        float_val = 0.0
                    specs.append(
                        BoundSpec(
                            var_name=var_name,
                            var_type="float",
                            interval=(float_val - 10.0, float_val + 10.0),
                            valid_set=None,
                        )
                    )

        return specs
