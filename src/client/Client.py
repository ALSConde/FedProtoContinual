from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional
import torch
from core.Strategy import Strategy


@dataclass
class TrainResult:
    state_dict: Dict[str, torch.Tensor]
    metrics: Dict[str, float]


class Client:
    def __init__(
        self,
        strategy: Strategy,
        train_config: Optional[Dict[str, Any]] = None,
        on_send: Optional[Callable[[TrainResult], None]] = None,
        on_request_global_model: Optional[Callable[[], Dict[str, torch.Tensor]]] = None,
    ) -> None:
        self.strategy = strategy
        self.train_config = train_config or {}
        self.on_send = on_send
        self.on_request_global_model = on_request_global_model

    def execute_round(
        self,
        server_round: int,
        global_state_dict: Optional[Dict[str, torch.Tensor]] = None,
    ) -> TrainResult:
        state_dict = global_state_dict

        if state_dict is None and self.on_request_global_model is not None:
            state_dict = self.on_request_global_model()

        if state_dict is not None:
            self.strategy.model.load_state_dict(state_dict)

        lr = self.train_config.get("lr")
        if lr is not None:
            for group in self.strategy.optimizer.param_groups:
                group["lr"] = float(lr)

        self.strategy.start()
        self.strategy.evaluation()

        result = TrainResult(
            state_dict=self.strategy.model.state_dict(),
            metrics={
                "round": float(server_round),
                "train_loss": float(self.strategy.total_loss),
                "train_acc": float(self.strategy.total_correct),
                "eval_loss": float(self.strategy.eval_loss),
                "eval_acc": float(self.strategy.eval_acc),
            },
        )

        if self.on_send is not None:
            self.on_send(result)

        return result
