import copy
import math
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from qlib.contrib.model.pytorch_general_nn import GeneralPTNN as QlibGeneralPTNN
from qlib.data.dataset.handler import DataHandlerLP
from qlib.data.dataset.weight import Reweighter
from qlib.model.utils import ConcatDataset
from qlib.utils import get_or_create_path
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Sampler


class _NoOpScheduler:
    def step(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, _state_dict: dict[str, Any]) -> None:
        return None


class _DateBatchSampler(Sampler[list[int]]):
    def __init__(self, index: pd.Index, *, shuffle: bool) -> None:
        if not isinstance(index, pd.MultiIndex):
            raise ValueError("date batching requires a MultiIndex with a datetime level")
        if "datetime" in index.names:
            dates = index.get_level_values("datetime")
        else:
            datetime_levels = [
                level
                for level in range(index.nlevels)
                if pd.api.types.is_datetime64_any_dtype(index.get_level_values(level).dtype)
            ]
            if len(datetime_levels) != 1:
                raise ValueError("date batching requires exactly one datetime index level")
            dates = index.get_level_values(datetime_levels[0])

        grouped: dict[Any, list[int]] = {}
        for position, date in enumerate(dates):
            grouped.setdefault(date, []).append(position)
        self._batches = list(grouped.values())
        self.shuffle = bool(shuffle)

    def __iter__(self):
        order = np.arange(len(self._batches))
        if self.shuffle:
            np.random.shuffle(order)
        for position in order:
            yield self._batches[int(position)]

    def __len__(self) -> int:
        return len(self._batches)

    def ordered_positions(self) -> list[int]:
        return [position for batch in self._batches for position in batch]


class _OrdinalScoreModel(nn.Module):
    def __init__(self, base_model: nn.Module, num_bins: int) -> None:
        super().__init__()
        self.base_model = base_model
        self.num_bins = int(num_bins)
        step = 2.0 / max(1, self.num_bins - 1)
        inverse_softplus_step = math.log(math.expm1(step))
        raw = torch.full((self.num_bins - 1,), inverse_softplus_step)
        raw[0] = -1.0
        self.raw_cutpoints = nn.Parameter(raw)

    def location(self, features: torch.Tensor) -> torch.Tensor:
        return self.base_model(features)

    def cutpoints(self) -> torch.Tensor:
        if self.raw_cutpoints.numel() == 1:
            return self.raw_cutpoints
        first = self.raw_cutpoints[:1]
        increments = F.softplus(self.raw_cutpoints[1:])
        return torch.cat([first, first + torch.cumsum(increments, dim=0)])

    def ordinal_logits(self, location: torch.Tensor) -> torch.Tensor:
        return location - self.cutpoints().view(1, -1)

    def score_from_location(self, location: torch.Tensor) -> torch.Tensor:
        probabilities_above = torch.sigmoid(self.ordinal_logits(location))
        return probabilities_above.sum(dim=1, keepdim=True) / float(self.num_bins - 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.score_from_location(self.location(features))


class GeneralPTNN(QlibGeneralPTNN):
    """QLib GeneralPTNN with a bounded, configurable training interface."""

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
        loss_temperature=1.0,
        tail_fraction=0.2,
        tail_top_weight=2.0,
        tail_bottom_weight=1.0,
        ordinal_num_bins=5,
        sam_enabled=False,
        sam_rho=0.05,
        sam_adaptive=False,
        batch_mode="sample",
        train_shuffle=True,
        train_drop_last=True,
        checkpoint_metric="loss",
        checkpoint_topk=20,
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
        if loss_name not in {"mse", "mae", "huber", "pairwise", "listnet", "tail_listnet", "ordinal"}:
            raise ValueError(f"Unsupported loss: {loss}")
        if float(huber_delta) <= 0.0:
            raise ValueError("huber_delta must be positive")
        if float(loss_temperature) <= 0.0:
            raise ValueError("loss_temperature must be positive")
        if not 0.0 < float(tail_fraction) <= 0.5:
            raise ValueError("tail_fraction must be in (0, 0.5]")
        if float(tail_top_weight) < 0.0 or float(tail_bottom_weight) < 0.0:
            raise ValueError("tail weights must be non-negative")
        if loss_name == "tail_listnet" and float(tail_top_weight) == 0.0 and float(tail_bottom_weight) == 0.0:
            raise ValueError("tail_listnet requires a positive tail weight")
        if int(ordinal_num_bins) < 2:
            raise ValueError("ordinal_num_bins must be at least 2")

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

        batch_mode_name = str(batch_mode).lower()
        if batch_mode_name not in {"sample", "date"}:
            raise ValueError(f"Unsupported batch_mode: {batch_mode}")
        if batch_mode_name == "date" and bool(train_drop_last):
            raise ValueError("train_drop_last must be false for date batches")
        if loss_name in {"pairwise", "listnet", "tail_listnet"} and batch_mode_name != "date":
            raise ValueError("ranking losses require date batches")
        checkpoint_metric_name = str(checkpoint_metric).lower()
        if checkpoint_metric_name not in {"loss", "ic", "rank_ic", "icir", "topk_precision"}:
            raise ValueError(f"Unsupported checkpoint_metric: {checkpoint_metric}")
        if checkpoint_metric_name != "loss" and batch_mode_name != "date":
            raise ValueError("cross-sectional checkpoint metrics require date batches")
        if int(checkpoint_topk) < 1:
            raise ValueError("checkpoint_topk must be positive")

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
        self.loss_temperature = float(loss_temperature)
        self.tail_fraction = float(tail_fraction)
        self.tail_top_weight = float(tail_top_weight)
        self.tail_bottom_weight = float(tail_bottom_weight)
        self.ordinal_num_bins = int(ordinal_num_bins)
        self.ordinal_boundaries: torch.Tensor | None = None
        self.feature_names: tuple[str, ...] | None = None
        self.sam_enabled = bool(sam_enabled)
        self.sam_rho = float(sam_rho)
        self.sam_adaptive = bool(sam_adaptive)
        if self.sam_enabled and self.sam_rho <= 0.0:
            raise ValueError("sam_rho must be positive")
        self.batch_mode = batch_mode_name
        self.train_shuffle = bool(train_shuffle)
        self.train_drop_last = bool(train_drop_last)
        self.checkpoint_metric = checkpoint_metric_name
        self.checkpoint_topk = int(checkpoint_topk)
        self.gradient_clip_mode = clip_mode
        self.gradient_clip_threshold = clip_threshold
        self.scheduler = scheduler_name

        if loss_name == "ordinal":
            self.dnn_model = _OrdinalScoreModel(self.dnn_model, self.ordinal_num_bins).to(self.device)
        self.train_optimizer = self._build_optimizer()

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
            "optimizer : %s\nloss : %s\nsam : %s\nbatch_mode : %s\ncheckpoint : %s\n"
            "gradient_clip : %s(%s)\nscheduler : %s",
            self.optimizer,
            self.loss,
            self.sam_enabled,
            self.batch_mode,
            self.checkpoint_metric,
            self.gradient_clip_mode,
            self.gradient_clip_threshold,
            self.scheduler,
        )

    def _build_optimizer(self):
        kwargs = {"lr": self.lr, "weight_decay": self.weight_decay}
        if self.optimizer == "adam":
            return optim.Adam(self.dnn_model.parameters(), **kwargs)
        if self.optimizer == "adamw":
            return optim.AdamW(self.dnn_model.parameters(), **kwargs)
        return optim.SGD(self.dnn_model.parameters(), momentum=self.optimizer_momentum, **kwargs)

    @staticmethod
    def _finite_supervision(
        pred: torch.Tensor,
        label: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prediction = pred.reshape(-1)
        target = label.reshape(-1)
        if prediction.numel() != target.numel():
            raise ValueError("prediction and label sizes do not match")
        finite_label = torch.isfinite(target)
        if not torch.any(finite_label):
            raise ValueError("training batch has no finite labels")
        prediction = prediction[finite_label]
        if not torch.all(torch.isfinite(prediction)):
            raise FloatingPointError("non-finite prediction for a finite label")
        return prediction, target[finite_label].to(prediction), finite_label

    @staticmethod
    def _finite_weight(
        weight: torch.Tensor,
        finite_label: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        flat_weight = weight.reshape(-1)
        if flat_weight.numel() != finite_label.numel():
            raise ValueError("sample weight and label sizes do not match")
        selected_weight = flat_weight[finite_label].to(target)
        if not torch.all(torch.isfinite(selected_weight)):
            raise FloatingPointError("non-finite sample weight for a finite label")
        return selected_weight

    @staticmethod
    def _require_finite_loss(loss: torch.Tensor) -> torch.Tensor:
        if not torch.all(torch.isfinite(loss)):
            raise FloatingPointError("non-finite training loss")
        return loss

    def _pointwise_loss(self, pred: torch.Tensor, label: torch.Tensor, weight=None) -> torch.Tensor:
        prediction, target, finite_label = self._finite_supervision(pred, label)
        prediction = prediction.reshape(-1, 1)
        target = target.reshape(-1, 1)
        if weight is None:
            selected_weight = torch.ones_like(target)
        else:
            selected_weight = self._finite_weight(weight, finite_label, target).reshape(-1, 1)
        if self.loss == "mse":
            element_loss = F.mse_loss(prediction, target, reduction="none")
        elif self.loss == "mae":
            element_loss = F.l1_loss(prediction, target, reduction="none")
        else:
            element_loss = F.huber_loss(prediction, target, reduction="none", delta=self.huber_delta)
        return self._require_finite_loss(torch.mean(selected_weight * element_loss))

    def _ranking_inputs(self, pred: torch.Tensor, label: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        prediction, target, _ = self._finite_supervision(pred, label)
        if prediction.numel() < 2:
            raise ValueError("ranking loss requires at least two finite stocks in each date batch")
        return prediction, target

    def _listnet_loss(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target_probability = torch.softmax(target / self.loss_temperature, dim=0)
        prediction_log_probability = torch.log_softmax(prediction / self.loss_temperature, dim=0)
        return -(target_probability * prediction_log_probability).sum()

    def _ranking_loss(self, pred: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        prediction, target = self._ranking_inputs(pred, label)
        if self.loss == "pairwise":
            row, column = torch.triu_indices(target.numel(), target.numel(), offset=1, device=target.device)
            label_difference = target[row] - target[column]
            informative = label_difference != 0
            if not torch.any(informative):
                return self._require_finite_loss(prediction.sum() * 0.0)
            direction = torch.sign(label_difference[informative])
            score_difference = prediction[row[informative]] - prediction[column[informative]]
            loss = F.softplus(-direction * score_difference / self.loss_temperature).mean()
            return self._require_finite_loss(loss)
        base_loss = self._listnet_loss(prediction, target)
        if self.loss == "listnet":
            return self._require_finite_loss(base_loss)

        tail_size = min(target.numel(), max(2, int(math.ceil(target.numel() * self.tail_fraction))))
        ordered = torch.argsort(target)
        bottom = ordered[:tail_size]
        top = ordered[-tail_size:]
        weighted_losses = [base_loss]
        weights = [1.0]
        if self.tail_top_weight > 0.0:
            weighted_losses.append(self._listnet_loss(prediction[top], target[top]) * self.tail_top_weight)
            weights.append(self.tail_top_weight)
        if self.tail_bottom_weight > 0.0:
            weighted_losses.append(
                self._listnet_loss(-prediction[bottom], -target[bottom]) * self.tail_bottom_weight
            )
            weights.append(self.tail_bottom_weight)
        return self._require_finite_loss(sum(weighted_losses) / sum(weights))

    def _ordinal_loss(self, location: torch.Tensor, label: torch.Tensor, weight=None) -> torch.Tensor:
        if self.ordinal_boundaries is None:
            raise ValueError("ordinal training boundaries have not been initialized from the training fold")
        selected_location, target, finite_label = self._finite_supervision(location, label)
        selected_location = selected_location.reshape(-1, 1)
        boundaries = self.ordinal_boundaries.to(target.device)
        classes = torch.bucketize(target, boundaries)
        thresholds = torch.arange(self.ordinal_num_bins - 1, device=target.device)
        cumulative_target = (classes.unsqueeze(1) > thresholds.unsqueeze(0)).float()
        logits = self.dnn_model.ordinal_logits(selected_location)
        element_loss = F.binary_cross_entropy_with_logits(logits, cumulative_target, reduction="none").mean(dim=1)
        if weight is None:
            return self._require_finite_loss(element_loss.mean())
        selected_weight = self._finite_weight(weight, finite_label, element_loss)
        return self._require_finite_loss((element_loss * selected_weight).mean())

    def loss_fn(self, pred, label, weight=None):
        if self.loss in {"mse", "mae", "huber"}:
            return self._pointwise_loss(pred, label, weight)
        if self.loss in {"pairwise", "listnet", "tail_listnet"}:
            return self._ranking_loss(pred, label)
        return self._ordinal_loss(pred, label, weight)

    def _forward_for_loss(self, feature: torch.Tensor) -> torch.Tensor:
        if self.loss == "ordinal":
            return self.dnn_model.location(feature.float())
        return self.dnn_model(feature.float())

    def _clip_gradients(self) -> None:
        if self.gradient_clip_mode == "value":
            torch.nn.utils.clip_grad_value_(self.dnn_model.parameters(), self.gradient_clip_threshold)
        elif self.gradient_clip_mode == "norm":
            torch.nn.utils.clip_grad_norm_(self.dnn_model.parameters(), self.gradient_clip_threshold)

    def _sam_perturb(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        gradient_norms = []
        for parameter in self.dnn_model.parameters():
            if parameter.grad is None:
                continue
            parameter_scale = parameter.detach().abs() if self.sam_adaptive else 1.0
            gradient_norms.append((parameter_scale * parameter.grad).norm(p=2))
        if not gradient_norms:
            return []
        gradient_norm = torch.norm(torch.stack(gradient_norms), p=2)
        scale = self.sam_rho / (gradient_norm + 1e-12)
        perturbations = []
        with torch.no_grad():
            for parameter in self.dnn_model.parameters():
                if parameter.grad is None:
                    continue
                multiplier = parameter.detach().pow(2) if self.sam_adaptive else 1.0
                perturbation = multiplier * parameter.grad * scale.to(parameter)
                parameter.add_(perturbation)
                perturbations.append((parameter, perturbation))
        return perturbations

    @staticmethod
    def _disable_batch_norm_running_stats(model: nn.Module) -> list[tuple[nn.Module, bool]]:
        previous = []
        for module in model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                previous.append((module, module.track_running_stats))
                module.track_running_stats = False
        return previous

    def train_epoch(self, data_loader):
        self.dnn_model.train()
        for data, weight in data_loader:
            feature, label = self._get_fl(data)
            pred = self._forward_for_loss(feature)
            loss = self.loss_fn(pred, label, weight.to(self.device))
            self.train_optimizer.zero_grad()
            loss.backward()
            if not self.sam_enabled:
                self._clip_gradients()
                self.train_optimizer.step()
                continue

            perturbations = self._sam_perturb()
            self.train_optimizer.zero_grad()
            batch_norm_state = self._disable_batch_norm_running_stats(self.dnn_model)
            try:
                second_pred = self._forward_for_loss(feature)
                second_loss = self.loss_fn(second_pred, label, weight.to(self.device))
                second_loss.backward()
            finally:
                with torch.no_grad():
                    for parameter, perturbation in perturbations:
                        parameter.sub_(perturbation)
                for module, track_running_stats in batch_norm_state:
                    module.track_running_stats = track_running_stats
            self._clip_gradients()
            self.train_optimizer.step()

    def test_epoch(self, data_loader):
        if self.loss != "ordinal":
            return super().test_epoch(data_loader)
        self.dnn_model.eval()
        losses = []
        for data, weight in data_loader:
            feature, label = self._get_fl(data)
            with torch.no_grad():
                location = self._forward_for_loss(feature)
                losses.append(self.loss_fn(location, label, weight.to(self.device)).item())
        mean_loss = float(np.mean(losses))
        return mean_loss, mean_loss

    @staticmethod
    def _prepared_index(data: Any) -> pd.Index:
        if isinstance(data, pd.DataFrame):
            return data.index
        if hasattr(data, "get_index"):
            return data.get_index()
        raise ValueError("prepared QLib data does not expose an index for date batching")

    def build_data_loader(self, data: Any, weights: Any, *, train: bool):
        if isinstance(data, pd.DataFrame):
            index = data.index
            values = data.values
        else:
            if hasattr(data, "config"):
                data.config(fillna_type="ffill+bfill")
            index = self._prepared_index(data)
            values = data
        dataset = ConcatDataset(values, weights)
        if self.batch_mode == "date":
            sampler = _DateBatchSampler(index, shuffle=self.train_shuffle if train else False)
            loader = DataLoader(
                dataset,
                batch_sampler=sampler,
                num_workers=self.n_jobs,
            )
            if not train:
                index = index.take(sampler.ordered_positions())
        else:
            loader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=self.train_shuffle if train else False,
                num_workers=self.n_jobs,
                drop_last=self.train_drop_last if train else False,
            )
        return loader, index

    def _extract_training_labels(self, data: Any) -> np.ndarray:
        if isinstance(data, pd.DataFrame):
            if isinstance(data.columns, pd.MultiIndex) and "label" in data.columns.get_level_values(0):
                values = data["label"].to_numpy().reshape(-1)
            else:
                values = data.iloc[:, -1].to_numpy().reshape(-1)
            return values[np.isfinite(values)]
        if hasattr(data, "config"):
            data.config(fillna_type="ffill+bfill")
        labels = []
        for batch in DataLoader(data, batch_size=self.batch_size, num_workers=self.n_jobs):
            _feature, label = self._get_fl(batch)
            labels.append(label.detach().cpu().numpy().reshape(-1))
        values = np.concatenate(labels)
        return values[np.isfinite(values)]

    def prepare_training_data(self, train_data: Any) -> None:
        if self.loss != "ordinal":
            return
        labels = self._extract_training_labels(train_data)
        if labels.size < self.ordinal_num_bins:
            raise ValueError("ordinal loss requires at least num_bins finite training labels")
        quantiles = np.arange(1, self.ordinal_num_bins) / self.ordinal_num_bins
        boundaries = np.quantile(labels, quantiles)
        if np.unique(boundaries).size != boundaries.size:
            raise ValueError("ordinal training-fold quantile boundaries are not unique")
        self.ordinal_boundaries = torch.as_tensor(boundaries, dtype=torch.float32, device=self.device)

    @staticmethod
    def _correlation(prediction: np.ndarray, target: np.ndarray, *, rank: bool) -> float:
        if rank:
            prediction = pd.Series(prediction).rank(method="average").to_numpy()
            target = pd.Series(target).rank(method="average").to_numpy()
        if prediction.size < 2 or np.std(prediction) == 0.0 or np.std(target) == 0.0:
            return float("nan")
        return float(np.corrcoef(prediction, target)[0, 1])

    def _evaluate_loader(self, data_loader) -> tuple[float, float]:
        self.dnn_model.eval()
        losses = []
        loss_weights = []
        daily_values = []
        for data, weight in data_loader:
            feature, label = self._get_fl(data)
            with torch.no_grad():
                pred = self._forward_for_loss(feature)
                batch_loss = self.loss_fn(pred, label, weight.to(self.device)).item()
                score = self.dnn_model.score_from_location(pred) if self.loss == "ordinal" else pred
                prediction = score.detach().cpu().numpy().reshape(-1)
            target = label.detach().cpu().numpy().reshape(-1)
            if prediction.size != target.size:
                raise ValueError("prediction and label sizes do not match during evaluation")
            finite_target = np.isfinite(target)
            prediction = prediction[finite_target]
            target = target[finite_target]
            if not np.isfinite(prediction).all():
                raise FloatingPointError("non-finite prediction for a finite label during evaluation")
            losses.append(batch_loss)
            loss_weights.append(1.0 if self.batch_mode == "date" else float(target.size))
            if self.checkpoint_metric == "ic":
                daily_values.append(self._correlation(prediction, target, rank=False))
            elif self.checkpoint_metric == "rank_ic":
                daily_values.append(self._correlation(prediction, target, rank=True))
            elif self.checkpoint_metric == "icir":
                daily_values.append(self._correlation(prediction, target, rank=False))
            elif self.checkpoint_metric == "topk_precision":
                count = min(self.checkpoint_topk, prediction.size)
                if count == 0:
                    daily_values.append(float("nan"))
                    continue
                predicted_top = set(np.argpartition(prediction, -count)[-count:])
                target_top = set(np.argpartition(target, -count)[-count:])
                daily_values.append(len(predicted_top & target_top) / count)

        if not losses:
            raise ValueError("evaluation data loader produced no batches")
        mean_loss = float(np.average(losses, weights=loss_weights))
        if self.checkpoint_metric == "loss":
            return mean_loss, mean_loss
        finite_values = np.asarray(daily_values, dtype=float)
        finite_values = finite_values[np.isfinite(finite_values)]
        if finite_values.size == 0:
            return mean_loss, float("-inf")
        if self.checkpoint_metric == "icir":
            std = float(np.std(finite_values, ddof=1)) if finite_values.size > 1 else 0.0
            return mean_loss, float(np.mean(finite_values) / std) if std > 0.0 else float("-inf")
        return mean_loss, float(np.mean(finite_values))

    def _checkpoint_label(self) -> str:
        if self.checkpoint_metric == "topk_precision":
            return f"topk_precision@{self.checkpoint_topk}"
        return self.checkpoint_metric

    def _checkpoint_direction(self) -> str:
        return "minimize" if self.checkpoint_metric == "loss" else "maximize"

    @staticmethod
    def _format_training_value(value: float) -> str:
        return f"{value:.6f}" if np.isfinite(value) else "unavailable"

    def _schema_model(self) -> nn.Module:
        if isinstance(self.dnn_model, _OrdinalScoreModel):
            return self.dnn_model.base_model
        return self.dnn_model

    @staticmethod
    def _normalise_feature_names(columns: Any) -> tuple[str, ...]:
        names = []
        for column in columns:
            if isinstance(column, tuple):
                if not column:
                    raise ValueError("feature schema contains an empty column name")
                column = column[-1]
            names.append(str(column))
        return tuple(names)

    @classmethod
    def _feature_names_from_frame(cls, data: Any) -> tuple[str, ...] | None:
        if not isinstance(data, pd.DataFrame) or not isinstance(data.columns, pd.MultiIndex):
            return None
        if "feature" not in data.columns.get_level_values(0):
            return None
        return cls._normalise_feature_names(data["feature"].columns)

    @classmethod
    def _feature_names_from_dataset(
        cls,
        dataset: Any,
        *,
        data_key: str,
        prepared_data: Any = None,
    ) -> tuple[str, ...] | None:
        handler = getattr(dataset, "handler", None)
        get_cols = getattr(handler, "get_cols", None)
        if callable(get_cols):
            return cls._normalise_feature_names(get_cols(col_set="feature", data_key=data_key))
        return cls._feature_names_from_frame(prepared_data)

    @staticmethod
    def _required_feature_names(model: nn.Module) -> tuple[str, ...]:
        required = getattr(model, "required_feature_names", ())
        if isinstance(required, str):
            required = (required,)
        elif not isinstance(required, (list, tuple)):
            raise ValueError("required_feature_names must be a string, list, or tuple")
        names = tuple(str(name) for name in required)
        if len(set(names)) != len(names):
            raise ValueError("required_feature_names contains duplicates")
        return names

    def _bind_feature_schema(
        self,
        dataset: Any,
        *,
        data_key: str,
        prepared_data: Any = None,
    ) -> None:
        model = self._schema_model()
        required_names = self._required_feature_names(model)
        feature_names = self._feature_names_from_dataset(
            dataset,
            data_key=data_key,
            prepared_data=prepared_data,
        )
        if feature_names is None:
            if required_names:
                raise ValueError("dataset does not expose feature names required by the model")
            return
        if len(set(feature_names)) != len(feature_names):
            raise ValueError("dataset feature names must be unique")

        expected_features = self.pt_model_kwargs.get("num_features")
        if expected_features is not None and len(feature_names) != int(expected_features):
            raise ValueError(
                f"dataset exposes {len(feature_names)} features but model expects {int(expected_features)}"
            )
        if self.feature_names is not None and feature_names != self.feature_names:
            raise ValueError("feature schema changed between model training and inference")

        missing_names = [name for name in required_names if name not in feature_names]
        if missing_names:
            raise ValueError(f"model requires unavailable features: {', '.join(missing_names)}")

        model.feature_names = feature_names
        model.feature_index = {name: position for position, name in enumerate(feature_names)}
        self.feature_names = feature_names

    def fit(self, dataset, evals_result=None, save_path=None, reweighter=None):
        if evals_result is None:
            evals_result = {}
        checkpoint_label = self._checkpoint_label()
        checkpoint_direction = self._checkpoint_direction()
        scheduler_monitor = "valid_loss" if self.scheduler == "plateau" else "none"
        self.logger.info(
            "RD-Agent training context (authoritative): optimizer=%s; loss=%s; batch_mode=%s; "
            "checkpoint=%s; direction=%s; scheduler=%s; scheduler_monitor=%s; epochs=%d; "
            "early_stop_patience=%d",
            self.optimizer,
            self.loss,
            self.batch_mode,
            checkpoint_label,
            checkpoint_direction,
            self.scheduler,
            scheduler_monitor,
            self.n_epochs,
            self.early_stop,
        )
        train_data = dataset.prepare("train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        valid_data = dataset.prepare("valid", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        if len(train_data) == 0 or len(valid_data) == 0:
            raise ValueError("Empty data from dataset, please check your dataset config.")
        self._bind_feature_schema(dataset, data_key=DataHandlerLP.DK_L, prepared_data=train_data)
        if reweighter is None:
            train_weights = np.ones(len(train_data))
            valid_weights = np.ones(len(valid_data))
        elif isinstance(reweighter, Reweighter):
            train_weights = reweighter.reweight(train_data)
            valid_weights = reweighter.reweight(valid_data)
        else:
            raise ValueError("Unsupported reweighter type.")
        self.prepare_training_data(train_data)
        train_loader, _ = self.build_data_loader(train_data, train_weights, train=True)
        valid_loader, _ = self.build_data_loader(valid_data, valid_weights, train=False)

        save_path = get_or_create_path(save_path)
        evals_result["train"] = []
        evals_result["valid"] = []
        best_score = float("inf") if self.checkpoint_metric == "loss" else float("-inf")
        best_epoch = None
        best_param = None
        stop_steps = 0
        self.fitted = False
        self.logger.info("training...")
        for step in range(self.n_epochs):
            self.train_epoch(train_loader)
            train_loss, train_score = self._evaluate_loader(train_loader)
            valid_loss, valid_score = self._evaluate_loader(valid_loader)
            learning_rate = float(self.train_optimizer.param_groups[0]["lr"])
            if self.checkpoint_metric == "loss":
                self.logger.info(
                    "Epoch%d: train_loss=%s; valid_loss=%s; lr=%.6g",
                    step,
                    self._format_training_value(train_loss),
                    self._format_training_value(valid_loss),
                    learning_rate,
                )
            else:
                self.logger.info(
                    "Epoch%d: train_loss=%s; valid_loss=%s; train_%s=%s; valid_%s=%s; lr=%.6g",
                    step,
                    self._format_training_value(train_loss),
                    self._format_training_value(valid_loss),
                    checkpoint_label,
                    self._format_training_value(train_score),
                    checkpoint_label,
                    self._format_training_value(valid_score),
                    learning_rate,
                )
            evals_result["train"].append(train_score)
            evals_result["valid"].append(valid_score)
            self.lr_scheduler.step(valid_loss)

            improved = np.isfinite(valid_score) and (
                valid_score < best_score if self.checkpoint_metric == "loss" else valid_score > best_score
            )
            if improved:
                best_score = valid_score
                best_epoch = step
                best_param = copy.deepcopy(self.dnn_model.state_dict())
                stop_steps = 0
            else:
                stop_steps += 1
                if stop_steps >= self.early_stop:
                    self.logger.info(
                        "early stop: checkpoint=%s; patience=%d; epoch=%d",
                        checkpoint_label,
                        self.early_stop,
                        step,
                    )
                    break
        if best_param is None or best_epoch is None:
            raise ValueError(f"No finite validation {self.checkpoint_metric} checkpoint was produced")
        self.logger.info(
            "best checkpoint: metric=%s; direction=%s; value=%.6f; epoch=%d",
            checkpoint_label,
            checkpoint_direction,
            best_score,
            best_epoch,
        )
        self.dnn_model.load_state_dict(best_param)
        torch.save(best_param, save_path)
        self.fitted = True
        if self.use_gpu:
            torch.cuda.empty_cache()

    def predict(self, dataset, batch_size=None, n_jobs=None):
        prepared_data = None
        if not callable(getattr(getattr(dataset, "handler", None), "get_cols", None)):
            prepared_data = dataset.prepare("test", col_set=["feature", "label"], data_key=DataHandlerLP.DK_I)
        self._bind_feature_schema(dataset, data_key=DataHandlerLP.DK_I, prepared_data=prepared_data)
        return super().predict(dataset, batch_size=batch_size, n_jobs=n_jobs)
