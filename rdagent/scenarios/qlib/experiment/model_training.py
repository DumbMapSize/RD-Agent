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
    "sam": {"enabled": False, "rho": 0.05, "adaptive": False},
    "data_loader": {"batch_mode": "sample", "shuffle": True, "drop_last": True},
    "checkpoint": {"metric": "loss", "topk": 20},
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
_STRUCTURED_KEYS = {
    "optimizer",
    "loss",
    "sam",
    "data_loader",
    "checkpoint",
    "gradient_clip",
    "scheduler",
    "time_series_lookback",
}
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


def _as_bool(value: Any, field: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1"}:
            return True
        if normalized in {"false", "no", "0"}:
            return False
    raise ValueError(f"training_hyperparameters.{field} must be boolean")


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
    _check_keys(
        config,
        {
            "name",
            "huber_delta",
            "temperature",
            "tail_fraction",
            "top_weight",
            "bottom_weight",
            "num_bins",
        },
        "loss",
    )
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
            "pairwise": "pairwise",
            "ranknet": "pairwise",
            "listnet": "listnet",
            "taillistnet": "tail_listnet",
            "tailawarelistnet": "tail_listnet",
            "ordinal": "ordinal",
            "cumulativeordinal": "ordinal",
        },
    )
    conditional_fields = {
        "temperature": {"pairwise", "listnet", "tail_listnet"},
        "tail_fraction": {"tail_listnet"},
        "top_weight": {"tail_listnet"},
        "bottom_weight": {"tail_listnet"},
        "num_bins": {"ordinal"},
    }
    for field, allowed_losses in conditional_fields.items():
        if field in config and name not in allowed_losses:
            allowed = ", ".join(sorted(allowed_losses))
            raise ValueError(
                f"training_hyperparameters.loss.{field} is only valid for {allowed} loss"
            )
    delta = _as_float(config.get("huber_delta", 1.0), "loss.huber_delta", minimum=0.0, strict_minimum=True)
    normalized = {"name": name, "huber_delta": delta}
    if name in {"pairwise", "listnet", "tail_listnet"}:
        normalized["temperature"] = _as_float(
            config.get("temperature", 1.0),
            "loss.temperature",
            minimum=0.0,
            strict_minimum=True,
        )
    if name == "tail_listnet":
        tail_fraction = _as_float(
            config.get("tail_fraction", 0.2),
            "loss.tail_fraction",
            minimum=0.0,
            strict_minimum=True,
        )
        if tail_fraction > 0.5:
            raise ValueError("training_hyperparameters.loss.tail_fraction must be <= 0.5")
        normalized.update(
            {
                "tail_fraction": tail_fraction,
                "top_weight": _as_float(config.get("top_weight", 2.0), "loss.top_weight", minimum=0.0),
                "bottom_weight": _as_float(
                    config.get("bottom_weight", 1.0),
                    "loss.bottom_weight",
                    minimum=0.0,
                ),
            }
        )
        if normalized["top_weight"] == 0.0 and normalized["bottom_weight"] == 0.0:
            raise ValueError("tail_listnet requires a positive top_weight or bottom_weight")
    if name == "ordinal":
        normalized["num_bins"] = _as_int(config.get("num_bins", 5), "loss.num_bins", minimum=2)
    return normalized


def _normalise_sam(raw: dict[str, Any]) -> dict[str, Any]:
    value = raw.get("sam", DEFAULT_TRAINING_HYPERPARAMETERS["sam"])
    config = _as_mapping(value, "sam")
    _check_keys(config, {"enabled", "rho", "adaptive"}, "sam")
    enabled = _as_bool(config.get("enabled", False), "sam.enabled")
    defaults = DEFAULT_TRAINING_HYPERPARAMETERS["sam"]
    if not enabled:
        return dict(defaults)
    return {
        "enabled": True,
        "rho": _as_float(config.get("rho", defaults["rho"]), "sam.rho", minimum=0.0, strict_minimum=True),
        "adaptive": _as_bool(config.get("adaptive", defaults["adaptive"]), "sam.adaptive"),
    }


def _normalise_data_loader(raw: dict[str, Any]) -> dict[str, Any]:
    value = raw.get("data_loader", DEFAULT_TRAINING_HYPERPARAMETERS["data_loader"])
    config = _as_mapping(value, "data_loader")
    _check_keys(config, {"batch_mode", "shuffle", "drop_last"}, "data_loader")
    batch_mode = _canonical_name(
        config.get("batch_mode", "sample"),
        "data_loader.batch_mode",
        {"sample": "sample", "row": "sample", "date": "date", "day": "date"},
    )
    drop_last = _as_bool(config.get("drop_last", batch_mode == "sample"), "data_loader.drop_last")
    if batch_mode == "date" and drop_last:
        raise ValueError("training_hyperparameters.data_loader.drop_last must be false for date batches")
    return {
        "batch_mode": batch_mode,
        "shuffle": _as_bool(config.get("shuffle", True), "data_loader.shuffle"),
        "drop_last": drop_last,
    }


def _normalise_checkpoint(raw: dict[str, Any]) -> dict[str, Any]:
    value = raw.get("checkpoint", DEFAULT_TRAINING_HYPERPARAMETERS["checkpoint"])
    config = _as_mapping(value, "checkpoint")
    _check_keys(config, {"metric", "topk"}, "checkpoint")
    metric = _canonical_name(
        config.get("metric", "loss"),
        "checkpoint.metric",
        {
            "loss": "loss",
            "ic": "ic",
            "rankic": "rank_ic",
            "icir": "icir",
            "topkprecision": "topk_precision",
        },
    )
    return {
        "metric": metric,
        "topk": _as_int(config.get("topk", 20), "checkpoint.topk", minimum=1),
    }


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
    if name == "none":
        defaults = DEFAULT_TRAINING_HYPERPARAMETERS["scheduler"]
        return {
            "name": "none",
            "factor": defaults["factor"],
            "patience": defaults["patience"],
            "min_lr": defaults["min_lr"],
            "threshold": defaults["threshold"],
        }
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
        "sam": _normalise_sam(raw),
        "data_loader": _normalise_data_loader(raw),
        "checkpoint": _normalise_checkpoint(raw),
        "gradient_clip": _normalise_gradient_clip(raw),
        "scheduler": _normalise_scheduler(raw),
    }
    ranking_losses = {"pairwise", "listnet", "tail_listnet"}
    if normalized["loss"]["name"] in ranking_losses and normalized["data_loader"]["batch_mode"] != "date":
        raise ValueError("ranking losses require training_hyperparameters.data_loader.batch_mode='date'")
    if normalized["checkpoint"]["metric"] != "loss" and normalized["data_loader"]["batch_mode"] != "date":
        raise ValueError("cross-sectional checkpoint metrics require date batches")
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


def _require_generated_fields(config: Mapping[str, Any], required: set[str], field: str = "") -> None:
    missing = sorted(required - set(config))
    if not missing:
        return
    path = f"training_hyperparameters.{field}" if field else "training_hyperparameters"
    raise ValueError(f"Generated {path} must explicitly provide: {', '.join(missing)}")


def normalize_generated_training_hyperparameters(
    value: Mapping[str, Any] | None,
    model_type: str,
) -> dict[str, Any]:
    """Validate an LLM-generated training contract before applying runtime defaults."""
    if not isinstance(value, Mapping):
        raise ValueError("Generated training_hyperparameters must be a mapping")
    raw = dict(value)
    required_root = _CORE_KEYS | (_STRUCTURED_KEYS - {"time_series_lookback"})
    _require_generated_fields(raw, required_root)

    nested_fields = {
        "optimizer": {"name"},
        "loss": {"name"},
        "sam": {"enabled"},
        "data_loader": {"batch_mode", "shuffle", "drop_last"},
        "checkpoint": {"metric"},
        "gradient_clip": {"mode"},
        "scheduler": {"name"},
    }
    nested = {}
    for field, required in nested_fields.items():
        nested[field] = _as_mapping(raw[field], field)
        _require_generated_fields(nested[field], required, field)

    normalized = normalize_training_hyperparameters(raw, model_type)
    if normalized["optimizer"]["name"] == "sgd":
        _require_generated_fields(nested["optimizer"], {"momentum"}, "optimizer")

    loss_name = normalized["loss"]["name"]
    loss_fields = {
        "huber": {"huber_delta"},
        "pairwise": {"temperature"},
        "listnet": {"temperature"},
        "tail_listnet": {"temperature", "tail_fraction", "top_weight", "bottom_weight"},
        "ordinal": {"num_bins"},
    }
    _require_generated_fields(nested["loss"], loss_fields.get(loss_name, set()), "loss")

    if normalized["sam"]["enabled"]:
        _require_generated_fields(nested["sam"], {"rho", "adaptive"}, "sam")
    if normalized["checkpoint"]["metric"] == "topk_precision":
        _require_generated_fields(nested["checkpoint"], {"topk"}, "checkpoint")
    if normalized["gradient_clip"]["mode"] != "none":
        _require_generated_fields(nested["gradient_clip"], {"threshold"}, "gradient_clip")
    if normalized["scheduler"]["name"] == "plateau":
        _require_generated_fields(
            nested["scheduler"],
            {"factor", "patience", "min_lr", "threshold"},
            "scheduler",
        )
    if normalize_model_type(model_type) == "TimeSeries":
        _require_generated_fields(raw, {"time_series_lookback"})
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
    sam = config["sam"]
    data_loader = config["data_loader"]
    checkpoint = config["checkpoint"]
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
        "loss_temperature": str(loss.get("temperature", 1.0)),
        "tail_fraction": str(loss.get("tail_fraction", 0.2)),
        "tail_top_weight": str(loss.get("top_weight", 2.0)),
        "tail_bottom_weight": str(loss.get("bottom_weight", 1.0)),
        "ordinal_num_bins": str(loss.get("num_bins", 5)),
        "sam_enabled": str(sam["enabled"]).lower(),
        "sam_rho": str(sam["rho"]),
        "sam_adaptive": str(sam["adaptive"]).lower(),
        "batch_mode": data_loader["batch_mode"],
        "train_shuffle": str(data_loader["shuffle"]).lower(),
        "train_drop_last": str(data_loader["drop_last"]).lower(),
        "checkpoint_metric": checkpoint["metric"],
        "checkpoint_topk": str(checkpoint["topk"]),
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
