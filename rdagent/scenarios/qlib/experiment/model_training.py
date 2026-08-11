import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any


DEFAULT_TRAINING_HYPERPARAMETERS = {
    "n_epochs": 100,
    "lr": 2e-4,
    "early_stop": 10,
    "batch_size": 256,
    "weight_decay": 1e-4,
    "optimizer": {"name": "adam", "momentum": 0.0},
    "loss": {"name": "mse", "huber_delta": 1.0},
    "gradient_clip": {"mode": "value", "threshold": 3.0},
    "scheduler": {
        "name": "plateau",
        "factor": 0.5,
        "patience": 5,
        "min_lr": 1e-6,
        "threshold": 1e-5,
    },
}

_CORE_KEYS = {"n_epochs", "lr", "early_stop", "batch_size", "weight_decay"}
_STRUCTURED_KEYS = {"optimizer", "loss", "gradient_clip", "scheduler", "time_series_lookback"}
_COMPATIBILITY_KEYS = {
    "optimizer_momentum",
    "huber_delta",
    "gradient_clip_norm",
    "gradient_clip_value",
    "scheduler_factor",
    "scheduler_patience",
    "scheduler_min_lr",
    "scheduler_threshold",
    "lookback",
    "step_len",
}


def normalize_model_type(model_type: str) -> str:
    value = str(model_type).strip().lower().replace("_", "").replace("-", "")
    if value == "tabular":
        return "Tabular"
    if value in {"timeseries", "ts"}:
        return "TimeSeries"
    raise ValueError("model_type must be Tabular or TimeSeries")


def _as_mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"training_hyperparameters.{field} must be a mapping")
    return dict(value)


def _check_keys(value: dict[str, Any], allowed: set[str], field: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"Unsupported training_hyperparameters.{field} fields: {', '.join(unknown)}")


def _as_float(value: Any, field: str, *, minimum: float | None = None, strict_minimum: bool = False) -> float:
    if isinstance(value, bool):
        raise ValueError(f"training_hyperparameters.{field} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"training_hyperparameters.{field} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"training_hyperparameters.{field} must be finite")
    if minimum is not None and (result <= minimum if strict_minimum else result < minimum):
        comparator = ">" if strict_minimum else ">="
        raise ValueError(f"training_hyperparameters.{field} must be {comparator} {minimum}")
    return result


def _as_int(value: Any, field: str, *, minimum: int) -> int:
    numeric = _as_float(value, field)
    if not numeric.is_integer():
        raise ValueError(f"training_hyperparameters.{field} must be an integer")
    result = int(numeric)
    if result < minimum:
        raise ValueError(f"training_hyperparameters.{field} must be >= {minimum}")
    return result


def _canonical_name(value: Any, field: str, aliases: dict[str, str]) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"training_hyperparameters.{field} must be a non-empty string")
    token = value.strip().split(".")[-1].lower().replace("-", "").replace("_", "")
    try:
        return aliases[token]
    except KeyError as exc:
        supported = ", ".join(sorted(set(aliases.values())))
        raise ValueError(f"Unsupported training_hyperparameters.{field}={value!r}; supported: {supported}") from exc


def _normalise_optimizer(raw: dict[str, Any]) -> dict[str, Any]:
    optimizer_is_explicit = "optimizer" in raw
    value = raw.get("optimizer", DEFAULT_TRAINING_HYPERPARAMETERS["optimizer"])
    config = {"name": value} if isinstance(value, str) else _as_mapping(value, "optimizer")
    _check_keys(config, {"name", "momentum"}, "optimizer")
    if "optimizer_momentum" in raw:
        if optimizer_is_explicit and "momentum" in config:
            raise ValueError("Specify optimizer.momentum or optimizer_momentum, not both")
        config["momentum"] = raw["optimizer_momentum"]
    name = _canonical_name(
        config.get("name", "adam"),
        "optimizer.name",
        {"adam": "adam", "adamw": "adamw", "sgd": "sgd", "gd": "sgd"},
    )
    momentum = _as_float(config.get("momentum", 0.0), "optimizer.momentum", minimum=0.0)
    if momentum >= 1.0:
        raise ValueError("training_hyperparameters.optimizer.momentum must be < 1.0")
    if name != "sgd" and momentum != 0.0:
        raise ValueError("training_hyperparameters.optimizer.momentum is only valid for SGD")
    return {"name": name, "momentum": momentum}


def _normalise_loss(raw: dict[str, Any]) -> dict[str, Any]:
    loss_is_explicit = "loss" in raw
    value = raw.get("loss", DEFAULT_TRAINING_HYPERPARAMETERS["loss"])
    config = {"name": value} if isinstance(value, str) else _as_mapping(value, "loss")
    _check_keys(config, {"name", "huber_delta"}, "loss")
    if "huber_delta" in raw:
        if loss_is_explicit and "huber_delta" in config:
            raise ValueError("Specify loss.huber_delta or huber_delta, not both")
        config["huber_delta"] = raw["huber_delta"]
    name = _canonical_name(
        config.get("name", "mse"),
        "loss.name",
        {
            "mse": "mse",
            "mseloss": "mse",
            "meansquarederror": "mse",
            "mae": "mae",
            "l1": "mae",
            "l1loss": "mae",
            "meanabsoluteerror": "mae",
            "huber": "huber",
            "huberloss": "huber",
        },
    )
    delta = _as_float(config.get("huber_delta", 1.0), "loss.huber_delta", minimum=0.0, strict_minimum=True)
    return {"name": name, "huber_delta": delta}


def _normalise_gradient_clip(raw: dict[str, Any]) -> dict[str, Any]:
    aliases_present = [key for key in ("gradient_clip_norm", "gradient_clip_value") if key in raw]
    if "gradient_clip" in raw and aliases_present:
        raise ValueError("Specify gradient_clip or one legacy gradient_clip_* field, not both")
    if len(aliases_present) > 1:
        raise ValueError("Specify only one of gradient_clip_norm and gradient_clip_value")
    if aliases_present:
        alias = aliases_present[0]
        config: dict[str, Any] = {
            "mode": "norm" if alias.endswith("norm") else "value",
            "threshold": raw[alias],
        }
    else:
        value = raw.get("gradient_clip", DEFAULT_TRAINING_HYPERPARAMETERS["gradient_clip"])
        config = {"mode": value} if isinstance(value, str) else _as_mapping(value, "gradient_clip")
    _check_keys(config, {"mode", "threshold"}, "gradient_clip")
    mode = _canonical_name(
        config.get("mode", "value"),
        "gradient_clip.mode",
        {
            "none": "none",
            "off": "none",
            "disabled": "none",
            "value": "value",
            "clipgradvalue": "value",
            "norm": "norm",
            "clipgradnorm": "norm",
        },
    )
    threshold = _as_float(
        config.get("threshold", 0.0 if mode == "none" else 3.0),
        "gradient_clip.threshold",
        minimum=0.0,
        strict_minimum=mode != "none",
    )
    return {"mode": mode, "threshold": threshold}


def _normalise_scheduler(raw: dict[str, Any]) -> dict[str, Any]:
    scheduler_is_explicit = "scheduler" in raw
    value = raw.get("scheduler", DEFAULT_TRAINING_HYPERPARAMETERS["scheduler"])
    config = {"name": value} if isinstance(value, str) else _as_mapping(value, "scheduler")
    _check_keys(config, {"name", "factor", "patience", "min_lr", "threshold"}, "scheduler")
    for source, target in {
        "scheduler_factor": "factor",
        "scheduler_patience": "patience",
        "scheduler_min_lr": "min_lr",
        "scheduler_threshold": "threshold",
    }.items():
        if source not in raw:
            continue
        if scheduler_is_explicit and target in config:
            raise ValueError(f"Specify scheduler.{target} or {source}, not both")
        config[target] = raw[source]
    name = _canonical_name(
        config.get("name", "plateau"),
        "scheduler.name",
        {
            "plateau": "plateau",
            "reducelronplateau": "plateau",
            "none": "none",
            "off": "none",
            "disabled": "none",
            "constant": "none",
        },
    )
    factor = _as_float(config.get("factor", 0.5), "scheduler.factor", minimum=0.0, strict_minimum=True)
    if factor >= 1.0:
        raise ValueError("training_hyperparameters.scheduler.factor must be < 1.0")
    return {
        "name": name,
        "factor": factor,
        "patience": _as_int(config.get("patience", 5), "scheduler.patience", minimum=0),
        "min_lr": _as_float(config.get("min_lr", 1e-6), "scheduler.min_lr", minimum=0.0),
        "threshold": _as_float(config.get("threshold", 1e-5), "scheduler.threshold", minimum=0.0),
    }


def normalize_training_hyperparameters(
    value: Mapping[str, Any] | None,
    model_type: str,
) -> dict[str, Any]:
    raw = dict(value or {})
    unknown = sorted(set(raw) - _CORE_KEYS - _STRUCTURED_KEYS - _COMPATIBILITY_KEYS)
    if unknown:
        raise ValueError(f"Unsupported training_hyperparameters fields: {', '.join(unknown)}")
    normalized: dict[str, Any] = {
        "n_epochs": _as_int(raw.get("n_epochs", 100), "n_epochs", minimum=1),
        "lr": _as_float(raw.get("lr", 2e-4), "lr", minimum=0.0, strict_minimum=True),
        "early_stop": _as_int(raw.get("early_stop", 10), "early_stop", minimum=1),
        "batch_size": _as_int(raw.get("batch_size", 256), "batch_size", minimum=1),
        "weight_decay": _as_float(raw.get("weight_decay", 1e-4), "weight_decay", minimum=0.0),
        "optimizer": _normalise_optimizer(raw),
        "loss": _normalise_loss(raw),
        "gradient_clip": _normalise_gradient_clip(raw),
        "scheduler": _normalise_scheduler(raw),
    }
    normalized_model_type = normalize_model_type(model_type)
    lookback_fields = [key for key in ("time_series_lookback", "lookback", "step_len") if key in raw]
    if len(lookback_fields) > 1:
        raise ValueError("Specify only one of time_series_lookback, lookback, and step_len")
    if normalized_model_type == "TimeSeries":
        lookback_value = raw.get(lookback_fields[0], 20) if lookback_fields else 20
        normalized["time_series_lookback"] = _as_int(lookback_value, "time_series_lookback", minimum=1)
    else:
        if lookback_fields:
            raise ValueError("time_series_lookback is only valid for TimeSeries models")
    return normalized


def build_model_run_env(
    training_hyperparameters: Mapping[str, Any] | None,
    model_type: str,
    *,
    num_features: int | str | None = None,
) -> dict[str, str]:
    config = normalize_training_hyperparameters(training_hyperparameters, model_type)
    optimizer = config["optimizer"]
    loss = config["loss"]
    clip = config["gradient_clip"]
    scheduler = config["scheduler"]
    env = {
        "PYTHONPATH": "./",
        "n_epochs": str(config["n_epochs"]),
        "lr": str(config["lr"]),
        "early_stop": str(config["early_stop"]),
        "batch_size": str(config["batch_size"]),
        "weight_decay": str(config["weight_decay"]),
        "optimizer": optimizer["name"],
        "optimizer_momentum": str(optimizer["momentum"]),
        "loss": loss["name"],
        "huber_delta": str(loss["huber_delta"]),
        "gradient_clip_mode": clip["mode"],
        "gradient_clip_threshold": str(clip["threshold"]),
        "scheduler": scheduler["name"],
        "scheduler_factor": str(scheduler["factor"]),
        "scheduler_patience": str(scheduler["patience"]),
        "scheduler_min_lr": str(scheduler["min_lr"]),
        "scheduler_threshold": str(scheduler["threshold"]),
    }
    if normalize_model_type(model_type) == "TimeSeries":
        lookback = str(config["time_series_lookback"])
        env.update({"dataset_cls": "TSDatasetH", "step_len": lookback, "num_timesteps": lookback})
    else:
        env["dataset_cls"] = "DatasetH"
    if num_features is not None:
        env["num_features"] = str(num_features)
    return env


def inject_model_training_adapter(workspace: Any) -> None:
    source_path = Path(__file__).with_name("rdagent_general_ptnn.py")
    workspace.inject_files(**{source_path.name: source_path.read_text()})
