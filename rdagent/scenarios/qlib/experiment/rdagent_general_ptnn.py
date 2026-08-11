from typing import Any

import torch
import torch.nn.functional as F
from qlib.contrib.model.pytorch_general_nn import GeneralPTNN as QlibGeneralPTNN
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau


class _NoOpScheduler:
    def step(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, _state_dict: dict[str, Any]) -> None:
        return None


class GeneralPTNN(QlibGeneralPTNN):
    """QLib GeneralPTNN with a small, explicitly bounded training interface."""

    def __init__(
        self,
        n_epochs=200,
        lr=0.001,
        metric="",
        batch_size=2000,
        early_stop=20,
        loss="mse",
        weight_decay=0.0,
        optimizer="adam",
        optimizer_momentum=0.0,
        gradient_clip_mode="value",
        gradient_clip_threshold=3.0,
        huber_delta=1.0,
        scheduler="plateau",
        scheduler_factor=0.5,
        scheduler_patience=5,
        scheduler_min_lr=1e-6,
        scheduler_threshold=1e-5,
        n_jobs=10,
        GPU=0,
        seed=None,
        pt_model_uri="qlib.contrib.model.pytorch_gru_ts.GRUModel",
        pt_model_kwargs=None,
    ):
        optimizer_name = str(optimizer).lower()
        if optimizer_name == "gd":
            optimizer_name = "sgd"
        if optimizer_name not in {"adam", "adamw", "sgd"}:
            raise ValueError(f"Unsupported optimizer: {optimizer}")
        momentum = float(optimizer_momentum)
        if not 0.0 <= momentum < 1.0:
            raise ValueError("optimizer_momentum must be in [0, 1)")
        if optimizer_name != "sgd" and momentum != 0.0:
            raise ValueError("optimizer_momentum is only valid for SGD")

        loss_name = str(loss).lower()
        if loss_name not in {"mse", "mae", "huber"}:
            raise ValueError(f"Unsupported loss: {loss}")
        if float(huber_delta) <= 0.0:
            raise ValueError("huber_delta must be positive")

        clip_mode = str(gradient_clip_mode).lower()
        if clip_mode not in {"none", "value", "norm"}:
            raise ValueError(f"Unsupported gradient_clip_mode: {gradient_clip_mode}")
        clip_threshold = float(gradient_clip_threshold)
        if clip_threshold < 0.0 or (clip_mode != "none" and clip_threshold == 0.0):
            raise ValueError("gradient_clip_threshold must be non-negative for none and positive otherwise")

        scheduler_name = str(scheduler).lower()
        if scheduler_name not in {"none", "plateau"}:
            raise ValueError(f"Unsupported scheduler: {scheduler}")
        if not 0.0 < float(scheduler_factor) < 1.0:
            raise ValueError("scheduler_factor must be in (0, 1)")
        if int(scheduler_patience) < 0 or float(scheduler_min_lr) < 0.0 or float(scheduler_threshold) < 0.0:
            raise ValueError("scheduler patience, min_lr, and threshold must be non-negative")

        if pt_model_kwargs is None:
            pt_model_kwargs = {
                "d_feat": 6,
                "hidden_size": 64,
                "num_layers": 2,
                "dropout": 0.0,
            }

        base_optimizer = "gd" if optimizer_name == "sgd" else "adam"
        super().__init__(
            n_epochs=n_epochs,
            lr=lr,
            metric=metric,
            batch_size=batch_size,
            early_stop=early_stop,
            loss=loss_name,
            weight_decay=weight_decay,
            optimizer=base_optimizer,
            n_jobs=n_jobs,
            GPU=GPU,
            seed=seed,
            pt_model_uri=pt_model_uri,
            pt_model_kwargs=pt_model_kwargs,
        )

        self.optimizer = optimizer_name
        self.optimizer_momentum = momentum
        self.huber_delta = float(huber_delta)
        self.gradient_clip_mode = clip_mode
        self.gradient_clip_threshold = clip_threshold
        self.scheduler = scheduler_name

        if optimizer_name == "adamw":
            self.train_optimizer = optim.AdamW(
                self.dnn_model.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        elif optimizer_name == "sgd":
            self.train_optimizer = optim.SGD(
                self.dnn_model.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
                momentum=momentum,
            )

        if scheduler_name == "plateau":
            self.lr_scheduler = ReduceLROnPlateau(
                self.train_optimizer,
                mode="min",
                factor=float(scheduler_factor),
                patience=int(scheduler_patience),
                min_lr=float(scheduler_min_lr),
                threshold=float(scheduler_threshold),
            )
        else:
            self.lr_scheduler = _NoOpScheduler()
        self.logger.info(
            "RD-Agent effective training settings:\n"
            "optimizer : %s\nloss : %s\ngradient_clip : %s(%s)\nscheduler : %s",
            self.optimizer,
            self.loss,
            self.gradient_clip_mode,
            self.gradient_clip_threshold,
            self.scheduler,
        )

    @staticmethod
    def _align_weight(weight: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        while weight.dim() < target.dim():
            weight = weight.unsqueeze(-1)
        return weight

    def loss_fn(self, pred, label, weight=None):
        if self.loss == "mse":
            return super().loss_fn(pred, label, weight)

        mask = torch.isfinite(label)
        target = label[mask].view(-1, 1)
        prediction = pred[mask]
        if weight is None:
            selected_weight = torch.ones_like(target)
        else:
            selected_weight = self._align_weight(weight[mask], target)
        if self.loss == "mae":
            element_loss = F.l1_loss(prediction, target, reduction="none")
        elif self.loss == "huber":
            element_loss = F.huber_loss(prediction, target, reduction="none", delta=self.huber_delta)
        else:
            raise ValueError(f"Unsupported loss: {self.loss}")
        return torch.mean(selected_weight * element_loss)

    def _clip_gradients(self) -> None:
        if self.gradient_clip_mode == "value":
            torch.nn.utils.clip_grad_value_(self.dnn_model.parameters(), self.gradient_clip_threshold)
        elif self.gradient_clip_mode == "norm":
            torch.nn.utils.clip_grad_norm_(self.dnn_model.parameters(), self.gradient_clip_threshold)

    def train_epoch(self, data_loader):
        self.dnn_model.train()
        for data, weight in data_loader:
            feature, label = self._get_fl(data)
            pred = self.dnn_model(feature.float())
            loss = self.loss_fn(pred, label, weight.to(self.device))
            self.train_optimizer.zero_grad()
            loss.backward()
            self._clip_gradients()
            self.train_optimizer.step()
