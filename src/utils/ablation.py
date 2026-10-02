"""Ablation flags for FedProtoContinual.

Every component that the ablation study turns on/off is controlled by a
run-config key (see [tool.flwr.app.config] in pyproject.toml), so a whole
experiment ladder can be launched by changing flags only:

    use-local-adapter      personalization branch (adapter_local + alpha gate)
    enable-expansion       width/depth expansion of the local adapter
    enable-incorporation   candidacy -> vote -> incorporate protocol
    enable-kd              global-knowledge distillation term (L_kd)
    fl-algorithm           "fedavg" | "fedprox" (server aggregation + client prox term)
    proximal-mu            FedProx proximal weight (only used when fl-algorithm=fedprox)

Dependencies are validated explicitly (RuntimeError-style, never a silent
fallback): expansion and incorporation both act on the local adapter, so they
cannot be enabled while use-local-adapter is false.
"""

from dataclasses import asdict, dataclass
from typing import Any, Mapping

VALID_FL_ALGORITHMS = ("fedavg", "fedprox")

_TRUE = {"true", "1", "yes", "on"}
_FALSE = {"false", "0", "no", "off"}


def as_bool(value: Any, key: str) -> bool:
    if isinstance(value, bool):
        return value
    token = str(value).strip().lower()
    if token in _TRUE:
        return True
    if token in _FALSE:
        return False
    raise ValueError(
        f"Invalid boolean for run-config key '{key}': {value!r}. " "Use true/false."
    )


@dataclass(frozen=True)
class AblationFlags:
    use_local_adapter: bool
    enable_expansion: bool
    enable_incorporation: bool
    enable_kd: bool
    fl_algorithm: str
    proximal_mu: float  # effective value: 0.0 unless fl_algorithm == "fedprox"

    def as_dict(self) -> dict:
        return asdict(self)

    def describe(self) -> str:
        parts = [
            f"fl-algorithm={self.fl_algorithm}"
            + (f"(mu={self.proximal_mu})" if self.fl_algorithm == "fedprox" else ""),
            f"local-adapter={'on' if self.use_local_adapter else 'off'}",
            f"kd={'on' if self.enable_kd else 'off'}",
            f"expansion={'on' if self.enable_expansion else 'off'}",
            f"incorporation={'on' if self.enable_incorporation else 'off'}",
        ]
        return " | ".join(parts)


def resolve_ablation_flags(run_config: Mapping[str, Any]) -> AblationFlags:
    use_local = as_bool(run_config.get("use-local-adapter", True), "use-local-adapter")
    expansion = as_bool(run_config.get("enable-expansion", True), "enable-expansion")
    incorporation = as_bool(
        run_config.get("enable-incorporation", True), "enable-incorporation"
    )
    kd = as_bool(run_config.get("enable-kd", True), "enable-kd")

    algorithm = str(run_config.get("fl-algorithm", "fedavg")).strip().lower()
    if algorithm not in VALID_FL_ALGORITHMS:
        raise ValueError(
            f"Unknown fl-algorithm '{algorithm}'. Expected one of {VALID_FL_ALGORITHMS}."
        )

    if not use_local and expansion:
        raise ValueError(
            "enable-expansion=true requires use-local-adapter=true: the expansion "
            "acts on the local adapter, which is not used when it is disabled."
        )
    if not use_local and incorporation:
        raise ValueError(
            "enable-incorporation=true requires use-local-adapter=true: candidates "
            "are promoted from the local adapter, which is not used when it is disabled."
        )

    mu = float(run_config.get("proximal-mu", 0.0))
    if algorithm == "fedprox":
        if mu <= 0.0:
            raise ValueError(
                "fl-algorithm=fedprox requires proximal-mu > 0 "
                f"(got {mu}); with mu=0 FedProx is just FedAvg."
            )
    else:
        mu = 0.0

    return AblationFlags(
        use_local_adapter=use_local,
        enable_expansion=expansion,
        enable_incorporation=incorporation,
        enable_kd=kd,
        fl_algorithm=algorithm,
        proximal_mu=mu,
    )
