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
    ce-local-classes       cross-entropy restricted to the classes present in the client's
                           local data (inference still scores every seen prototype)
    proto-prepass          before local training, build provisional prototypes (local only)
                           for classes that have no server prototype yet
    proto-postpass         after local training, recompute the prototype statistics sent
                           to the server with the trained model in eval mode
    split-optimizers       two optimizers/objectives: shared params (extractor, global and
                           incorporated adapters) on L_g = CE + lambda-proto-shared*L_proto
                           (+ prox, + KD if kd-on has "shared") evaluated on the shared
                           embedding; local params (local adapter, alpha gate, classifier
                           scale) on L_l = CE + lambda-proto-local*L_proto (+ KD if kd-on
                           has "local") evaluated on the personalised embedding, with the
                           shared embedding detached at the local branch input
    kd-on                  where KD acts under split-optimizers: local | shared | both
                           (ignored without split-optimizers: KD is then on the fused h)
    kd-scale-mode          logit scale of the KL distillation (kd-mode=kl): const (teacher
                           and student use kd-logit-scale) | local (both use the detached
                           learned classifier scale) | hybrid (teacher kd-logit-scale,
                           student local scale). kd-logit-scale=1.0 is the previous behaviour;
                           ~20 matches the classifier's nominal scale.

Learning-rate multipliers (always available, default 1.0; applied on top of the round lr):
lr-scale-extractor, lr-scale-adapter-global (global + incorporated + shadow adapters) and
lr-scale-local (local adapter + alpha gate + classifier scale). Model init knobs:
near-identity-eps-local (local adapter weight std, default 1e-3) and alpha-init (alpha-gate
logit, default 0.0 -> alpha 0.25).

Dependencies are validated explicitly (RuntimeError-style, never a silent
fallback): expansion and incorporation both act on the local adapter, so they
cannot be enabled while use-local-adapter is false.
"""

from dataclasses import asdict, dataclass
from typing import Any, Mapping

VALID_FL_ALGORITHMS = ("fedavg", "fedprox")
VALID_KD_ON = ("local", "shared", "both")
VALID_KD_SCALE_MODES = ("const", "local", "hybrid")

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
    ce_local_classes: bool = False
    proto_prepass: bool = False
    proto_postpass: bool = False
    split_optimizers: bool = False
    kd_on: str = "local"
    kd_scale_mode: str = "const"

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
        # Only listed when enabled, so the default ladder output stays unchanged.
        if self.ce_local_classes:
            parts.append("ce-local-classes=on")
        if self.proto_prepass:
            parts.append("proto-prepass=on")
        if self.proto_postpass:
            parts.append("proto-postpass=on")
        if self.kd_scale_mode != "const":
            parts.append(f"kd-scale-mode={self.kd_scale_mode}")
        if self.split_optimizers:
            note = "" if self.use_local_adapter else ",inactive:no-local-adapter"
            parts.append(f"split-optimizers=on(kd-on={self.kd_on}{note})")
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

    kd_on = str(run_config.get("kd-on", "local")).strip().lower()
    if kd_on not in VALID_KD_ON:
        raise ValueError(f"Unknown kd-on '{kd_on}'. Expected one of {VALID_KD_ON}.")

    kd_scale_mode = str(run_config.get("kd-scale-mode", "const")).strip().lower()
    if kd_scale_mode not in VALID_KD_SCALE_MODES:
        raise ValueError(
            f"Unknown kd-scale-mode '{kd_scale_mode}'. Expected one of {VALID_KD_SCALE_MODES}."
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
        ce_local_classes=as_bool(
            run_config.get("ce-local-classes", False), "ce-local-classes"
        ),
        proto_prepass=as_bool(run_config.get("proto-prepass", False), "proto-prepass"),
        proto_postpass=as_bool(
            run_config.get("proto-postpass", False), "proto-postpass"
        ),
        split_optimizers=as_bool(
            run_config.get("split-optimizers", False), "split-optimizers"
        ),
        kd_on=kd_on,
        kd_scale_mode=kd_scale_mode,
    )
