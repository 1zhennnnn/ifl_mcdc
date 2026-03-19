"""
系統全域設定：所有可調整參數集中管理。

參考 SDD 第 8 章。
"""
from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings

from ifl_mcdc.layer3.domain_validator import DEFAULT_MEDICAL_RULES, DomainValidator
from ifl_mcdc.layer3.llm_sampler import AnthropicBackend, LLMBackend, OpenAIBackend


class IFLConfig(BaseSettings):
    """IFL 系統全域設定，可透過環境變數或 .env 檔覆蓋。"""

    model_config = {"env_prefix": "IFL_", "env_file": ".env", "extra": "ignore"}

    # ── LLM ──
    llm_provider: str = Field(default="openai")
    llm_model: str = Field(default="gpt-4o")
    llm_api_key: str = Field(default="")
    llm_temperature: float = Field(default=0.3)

    # ── SMT ──
    smt_timeout_ms: int = Field(default=10_000)

    # ── IFL 迭代控制 ──
    max_iterations: int = Field(default=50)
    min_coverage: float = Field(default=1.0)

    # ── 目標模組 ──
    func_name: str = Field(default="check_vaccine_eligibility")
    func_signature: str = Field(
        default="check_vaccine_eligibility(age, high_risk, days_since_last, egg_allergy)"
    )
    domain_context: str = Field(default="流感疫苗施打資格篩選系統")

    # ── 領域型別定義（變數名 → Z3 型別）──
    domain_types: dict[str, str] = Field(
        default={
            "age": "int",
            "high_risk": "bool",
            "days_since_last": "int",
            "egg_allergy": "bool",
        }
    )

    @property
    def llm_backend(self) -> LLMBackend:
        """根據 llm_provider 建立對應的 LLM 後端。"""
        if self.llm_provider == "openai":
            return OpenAIBackend(self.llm_model, self.llm_api_key, self.llm_temperature)
        if self.llm_provider == "anthropic":
            return AnthropicBackend(self.llm_model, self.llm_api_key)
        raise ValueError(f"不支援的 LLM 供應商：{self.llm_provider!r}")

    @property
    def domain_validator(self) -> DomainValidator:
        """建立帶預設醫療規則的驗證器。"""
        return DomainValidator(DEFAULT_MEDICAL_RULES)
