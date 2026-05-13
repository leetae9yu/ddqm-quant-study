"""Validated experiment configuration grammar for EQR autonomous runs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import yaml

from .features.feature_registry import available_feature_families
from .metrics import REQUIRED_METRIC_KEYS
from .models.registry import available_models


class ConfigValidationError(ValueError):
    """Raised when an experiment config violates the safe grammar."""


REPO_ROOT = Path(__file__).resolve().parents[3]
APPROVED_PATH_ROOTS = {
    "data": REPO_ROOT / "data",
    "configs": REPO_ROOT / "configs",
    "experiments": REPO_ROOT / "experiments",
    "reports": REPO_ROOT / "reports",
    "site": REPO_ROOT / "site",
}

SHELL_COMMAND_PATTERN = re.compile(r"(;|&&|\|\||`|\$\(|\b(?:bash|sh|zsh|curl|wget|python|python3|pip|conda|rm|mv|cp|chmod|chown|sudo)\b)")
WRDS_LOGIN_PATTERN = re.compile(r"(wrds\s*\.\s*connection|create_pgpass|pgpass|wrds_username|wrds_password|wrds\s+login)", re.IGNORECASE)
PATH_LIKE_KEYS = {
    "data_dir",
    "template",
    "output_dir",
    "manifest_path",
    "model_dir",
    "report_dir",
}


@dataclass(frozen=True)
class DataConfig:
    start_date: date
    end_date: date
    data_dir: str


@dataclass(frozen=True)
class UniverseFilters:
    share_codes: tuple[int, ...]
    exchange_codes: tuple[int, ...]
    min_market_cap: float | None = None
    exclude_financials: bool = False


@dataclass(frozen=True)
class PanelConfig:
    universe: UniverseFilters
    forward_horizons: tuple[int, ...]
    frequency: str = "monthly"


@dataclass(frozen=True)
class PITAvailabilityRules:
    compustat_lag_days: int
    ibes_lag_days: int
    macro_release_lag_days: int
    forbid_future_leakage: bool = True


@dataclass(frozen=True)
class FeaturesConfig:
    families: dict[str, bool]
    pit_availability: PITAvailabilityRules


@dataclass(frozen=True)
class ModelConfig:
    name: str
    target_column: str
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    search_space: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExplicitDateRange:
    start: date
    end: date


@dataclass(frozen=True)
class SplitConfig:
    train_fraction: float | None = None
    validation_fraction: float | None = None
    holdout_fraction: float | None = None
    train: ExplicitDateRange | None = None
    validation: ExplicitDateRange | None = None
    holdout: ExplicitDateRange | None = None


@dataclass(frozen=True)
class BudgetConfig:
    max_trials: int
    max_runtime_minutes: int
    retry_limit: int


@dataclass(frozen=True)
class PromotionConfig:
    required_metrics: tuple[str, ...]
    metric_thresholds: dict[str, float]


@dataclass(frozen=True)
class ReportConfig:
    template: str
    output_formats: tuple[str, ...]


@dataclass(frozen=True)
class RetentionPolicy:
    keep_last: int
    max_age_days: int


@dataclass(frozen=True)
class ArtifactsConfig:
    output_dir: str
    retention_policy: RetentionPolicy


@dataclass(frozen=True)
class ExperimentConfig:
    data: DataConfig
    panel: PanelConfig
    features: FeaturesConfig
    model: ModelConfig
    splits: SplitConfig
    budget: BudgetConfig
    promotion: PromotionConfig
    report: ReportConfig
    artifacts: ArtifactsConfig

    def normalized_dict(self) -> dict[str, Any]:
        return _normalize_for_json(asdict(self))

    def stable_hash(self) -> str:
        payload = json.dumps(self.normalized_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if raw is None:
        raise ConfigValidationError("Config is empty")
    if not isinstance(raw, Mapping):
        raise ConfigValidationError("Config root must be a mapping")
    return parse_experiment_config(raw)


def parse_experiment_config(raw: Mapping[str, Any]) -> ExperimentConfig:
    _validate_supported_yaml_tree(raw)
    _scan_for_unsafe_values(raw)
    _require_keys(raw, {"data", "panel", "features", "model", "splits", "budget", "promotion", "report", "artifacts"}, "root")

    data = _parse_data(_mapping(raw["data"], "data"))
    panel = _parse_panel(_mapping(raw["panel"], "panel"))
    features = _parse_features(_mapping(raw["features"], "features"))
    model = _parse_model(_mapping(raw["model"], "model"))
    splits = _parse_splits(_mapping(raw["splits"], "splits"), data)
    budget = _parse_budget(_mapping(raw["budget"], "budget"))
    promotion = _parse_promotion(_mapping(raw["promotion"], "promotion"))
    report = _parse_report(_mapping(raw["report"], "report"))
    artifacts = _parse_artifacts(_mapping(raw["artifacts"], "artifacts"))
    return ExperimentConfig(data, panel, features, model, splits, budget, promotion, report, artifacts)


def _parse_data(raw: Mapping[str, Any]) -> DataConfig:
    _require_keys(raw, {"start_date", "end_date", "data_dir"}, "data")
    start = _parse_date(raw["start_date"], "data.start_date")
    end = _parse_date(raw["end_date"], "data.end_date")
    if start >= end:
        raise ConfigValidationError("data.start_date must be before data.end_date")
    data_dir = _safe_path_string(raw["data_dir"], "data.data_dir", {"data"})
    return DataConfig(start_date=start, end_date=end, data_dir=data_dir)


def _parse_panel(raw: Mapping[str, Any]) -> PanelConfig:
    _require_keys(raw, {"universe", "forward_horizons", "frequency"}, "panel")
    universe_raw = _mapping(raw["universe"], "panel.universe")
    _require_keys(universe_raw, {"share_codes", "exchange_codes", "min_market_cap", "exclude_financials"}, "panel.universe")
    share_codes = _int_tuple(universe_raw["share_codes"], "panel.universe.share_codes", min_value=1)
    exchange_codes = _int_tuple(universe_raw["exchange_codes"], "panel.universe.exchange_codes", min_value=1)
    min_market_cap = universe_raw["min_market_cap"]
    if min_market_cap is not None and (not isinstance(min_market_cap, int | float) or min_market_cap < 0):
        raise ConfigValidationError("panel.universe.min_market_cap must be a non-negative number or null")
    exclude_financials = _bool(universe_raw["exclude_financials"], "panel.universe.exclude_financials")
    horizons = _int_tuple(raw["forward_horizons"], "panel.forward_horizons", min_value=1)
    frequency = _string(raw["frequency"], "panel.frequency")
    if frequency != "monthly":
        raise ConfigValidationError("panel.frequency must be 'monthly'")
    return PanelConfig(UniverseFilters(share_codes, exchange_codes, float(min_market_cap) if min_market_cap is not None else None, exclude_financials), horizons, frequency)


def _parse_features(raw: Mapping[str, Any]) -> FeaturesConfig:
    _require_keys(raw, {"families", "pit_availability"}, "features")
    families_raw = _mapping(raw["families"], "features.families")
    valid_families = set(available_feature_families())
    unknown = sorted(set(families_raw).difference(valid_families))
    if unknown:
        raise ConfigValidationError(f"Unknown feature families: {', '.join(unknown)}")
    families = {name: _bool(families_raw.get(name, True), f"features.families.{name}") for name in sorted(valid_families)}
    if not any(families.values()):
        raise ConfigValidationError("At least one feature family must be enabled")

    pit_raw = _mapping(raw["pit_availability"], "features.pit_availability")
    _require_keys(pit_raw, {"compustat_lag_days", "ibes_lag_days", "macro_release_lag_days", "forbid_future_leakage"}, "features.pit_availability")
    return FeaturesConfig(
        families=families,
        pit_availability=PITAvailabilityRules(
            compustat_lag_days=_non_negative_int(pit_raw["compustat_lag_days"], "features.pit_availability.compustat_lag_days"),
            ibes_lag_days=_non_negative_int(pit_raw["ibes_lag_days"], "features.pit_availability.ibes_lag_days"),
            macro_release_lag_days=_non_negative_int(pit_raw["macro_release_lag_days"], "features.pit_availability.macro_release_lag_days"),
            forbid_future_leakage=_bool(pit_raw["forbid_future_leakage"], "features.pit_availability.forbid_future_leakage"),
        ),
    )


def _parse_model(raw: Mapping[str, Any]) -> ModelConfig:
    _require_keys(raw, {"name", "target_column", "hyperparameters", "search_space"}, "model")
    name = _string(raw["name"], "model.name")
    if name not in available_models():
        raise ConfigValidationError(f"Unknown model '{name}'. Available models: {', '.join(available_models())}")
    target_column = _string(raw["target_column"], "model.target_column")
    hyperparameters = dict(_mapping(raw["hyperparameters"], "model.hyperparameters"))
    search_space = dict(_mapping(raw["search_space"], "model.search_space"))
    return ModelConfig(name=name, target_column=target_column, hyperparameters=hyperparameters, search_space=search_space)


def _parse_splits(raw: Mapping[str, Any], data: DataConfig) -> SplitConfig:
    fraction_keys = {"train_fraction", "validation_fraction", "holdout_fraction"}
    date_keys = {"train", "validation", "holdout"}
    present_fraction_keys = fraction_keys.intersection(raw)
    present_date_keys = date_keys.intersection(raw)
    extra = set(raw).difference(fraction_keys.union(date_keys))
    if extra:
        raise ConfigValidationError(f"Unexpected keys in splits: {', '.join(sorted(extra))}")
    if present_fraction_keys and present_date_keys:
        raise ConfigValidationError("splits must use either fractions or explicit date ranges, not both")
    if present_fraction_keys != fraction_keys and present_date_keys != date_keys:
        raise ConfigValidationError("splits must define train/validation/holdout fractions or date ranges")
    if present_fraction_keys:
        train = _fraction(raw["train_fraction"], "splits.train_fraction")
        validation = _fraction(raw["validation_fraction"], "splits.validation_fraction")
        holdout = _fraction(raw["holdout_fraction"], "splits.holdout_fraction")
        if abs((train + validation + holdout) - 1.0) > 1e-9:
            raise ConfigValidationError("split fractions must sum to 1.0")
        return SplitConfig(train_fraction=train, validation_fraction=validation, holdout_fraction=holdout)

    train_range = _parse_explicit_range(raw["train"], "splits.train")
    validation_range = _parse_explicit_range(raw["validation"], "splits.validation")
    holdout_range = _parse_explicit_range(raw["holdout"], "splits.holdout")
    for name, range_ in (("train", train_range), ("validation", validation_range), ("holdout", holdout_range)):
        if range_.start < data.start_date or range_.end > data.end_date:
            raise ConfigValidationError(f"splits.{name} must fall within data date range")
    if not (train_range.end < validation_range.start <= validation_range.end < holdout_range.start <= holdout_range.end):
        raise ConfigValidationError("explicit split date ranges must be ordered and non-overlapping")
    return SplitConfig(train=train_range, validation=validation_range, holdout=holdout_range)


def _parse_budget(raw: Mapping[str, Any]) -> BudgetConfig:
    _require_keys(raw, {"max_trials", "max_runtime_minutes", "retry_limit"}, "budget")
    max_trials = _positive_int(raw["max_trials"], "budget.max_trials")
    max_runtime = _positive_int(raw["max_runtime_minutes"], "budget.max_runtime_minutes")
    retry_limit = _non_negative_int(raw["retry_limit"], "budget.retry_limit")
    return BudgetConfig(max_trials=max_trials, max_runtime_minutes=max_runtime, retry_limit=retry_limit)


def _parse_promotion(raw: Mapping[str, Any]) -> PromotionConfig:
    _require_keys(raw, {"required_metrics", "metric_thresholds"}, "promotion")
    required_metrics = tuple(_string(item, "promotion.required_metrics[]") for item in _sequence(raw["required_metrics"], "promotion.required_metrics"))
    thresholds_raw = _mapping(raw["metric_thresholds"], "promotion.metric_thresholds")
    thresholds = {str(key): _number(value, f"promotion.metric_thresholds.{key}") for key, value in thresholds_raw.items()}
    valid_metrics = set(REQUIRED_METRIC_KEYS)
    unknown_required = sorted(set(required_metrics).difference(valid_metrics))
    if unknown_required:
        raise ConfigValidationError(f"Unknown required promotion metrics: {', '.join(unknown_required)}")
    unknown_thresholds = sorted(set(thresholds).difference(valid_metrics))
    if unknown_thresholds:
        raise ConfigValidationError(f"Unknown promotion threshold metrics: {', '.join(unknown_thresholds)}")
    missing = sorted(set(required_metrics).difference(thresholds))
    if missing:
        raise ConfigValidationError(f"promotion.metric_thresholds missing required metrics: {', '.join(missing)}")
    return PromotionConfig(required_metrics=required_metrics, metric_thresholds=thresholds)


def _parse_report(raw: Mapping[str, Any]) -> ReportConfig:
    _require_keys(raw, {"template", "output_formats"}, "report")
    template = _safe_path_string(raw["template"], "report.template", {"configs"})
    formats = tuple(_string(item, "report.output_formats[]") for item in _sequence(raw["output_formats"], "report.output_formats"))
    allowed = {"html", "json", "markdown", "md"}
    unknown = sorted(set(formats).difference(allowed))
    if unknown:
        raise ConfigValidationError(f"Unsupported report output formats: {', '.join(unknown)}")
    return ReportConfig(template=template, output_formats=formats)


def _parse_artifacts(raw: Mapping[str, Any]) -> ArtifactsConfig:
    _require_keys(raw, {"output_dir", "retention_policy"}, "artifacts")
    output_dir = _safe_path_string(raw["output_dir"], "artifacts.output_dir", {"experiments", "reports", "site"})
    policy_raw = _mapping(raw["retention_policy"], "artifacts.retention_policy")
    _require_keys(policy_raw, {"keep_last", "max_age_days"}, "artifacts.retention_policy")
    return ArtifactsConfig(
        output_dir=output_dir,
        retention_policy=RetentionPolicy(
            keep_last=_positive_int(policy_raw["keep_last"], "artifacts.retention_policy.keep_last"),
            max_age_days=_positive_int(policy_raw["max_age_days"], "artifacts.retention_policy.max_age_days"),
        ),
    )


def _parse_explicit_range(raw: Any, path: str) -> ExplicitDateRange:
    mapping = _mapping(raw, path)
    _require_keys(mapping, {"start", "end"}, path)
    start = _parse_date(mapping["start"], f"{path}.start")
    end = _parse_date(mapping["end"], f"{path}.end")
    if start >= end:
        raise ConfigValidationError(f"{path}.start must be before {path}.end")
    return ExplicitDateRange(start=start, end=end)


def _validate_supported_yaml_tree(value: Any, path: str = "root", seen: set[int] | None = None) -> None:
    seen = seen or set()
    if isinstance(value, Mapping):
        object_id = id(value)
        if object_id in seen:
            raise ConfigValidationError(f"Recursive YAML aliases are not allowed at {path}")
        seen.add(object_id)
        for key, item in value.items():
            if not isinstance(key, str):
                raise ConfigValidationError(f"Config keys must be strings at {path}")
            _validate_supported_yaml_tree(item, f"{path}.{key}", seen)
        seen.remove(object_id)
        return
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        object_id = id(value)
        if object_id in seen:
            raise ConfigValidationError(f"Recursive YAML aliases are not allowed at {path}")
        seen.add(object_id)
        for index, item in enumerate(value):
            _validate_supported_yaml_tree(item, f"{path}[{index}]", seen)
        seen.remove(object_id)
        return
    if isinstance(value, str | int | float | bool | date | datetime) or value is None:
        return
    raise ConfigValidationError(f"Unsupported YAML value type at {path}: {type(value).__name__}")


def _scan_for_unsafe_values(value: Any, path: str = "root") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key)
            _scan_string(key_text, f"{path}.{key_text}")
            _scan_for_unsafe_values(item, f"{path}.{key_text}")
    elif isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        for index, item in enumerate(value):
            _scan_for_unsafe_values(item, f"{path}[{index}]")
    elif isinstance(value, str):
        _scan_string(value, path)


def _scan_string(value: str, path: str) -> None:
    if ".." in Path(value).parts or value in {".", ".."} or "../" in value or "..\\" in value:
        raise ConfigValidationError(f"Path traversal is not allowed at {path}")
    if SHELL_COMMAND_PATTERN.search(value):
        raise ConfigValidationError(f"Shell command patterns are not allowed at {path}")
    if WRDS_LOGIN_PATTERN.search(value):
        raise ConfigValidationError(f"WRDS login patterns are not allowed at {path}")


def _safe_path_string(value: Any, path: str, allowed_roots: set[str]) -> str:
    text = _string(value, path)
    candidate = Path(text)
    if candidate.is_absolute():
        resolved = candidate.resolve(strict=False)
    else:
        resolved = (REPO_ROOT / candidate).resolve(strict=False)
    roots = [APPROVED_PATH_ROOTS[name].resolve(strict=False) for name in allowed_roots]
    if not any(resolved == root or root in resolved.parents for root in roots):
        raise ConfigValidationError(f"{path} must stay under approved roots: {', '.join(sorted(allowed_roots))}")
    return text


def _require_keys(raw: Mapping[str, Any], expected: set[str], path: str) -> None:
    actual = set(raw)
    missing = sorted(expected.difference(actual))
    extra = sorted(actual.difference(expected))
    if missing:
        raise ConfigValidationError(f"Missing keys in {path}: {', '.join(missing)}")
    if extra:
        raise ConfigValidationError(f"Unexpected keys in {path}: {', '.join(extra)}")


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigValidationError(f"{path} must be a mapping")
    return value


def _sequence(value: Any, path: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        raise ConfigValidationError(f"{path} must be a sequence")
    if not value:
        raise ConfigValidationError(f"{path} must not be empty")
    return value


def _string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigValidationError(f"{path} must be a non-empty string")
    return value


def _bool(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigValidationError(f"{path} must be a boolean")
    return value


def _parse_date(value: Any, path: str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if not isinstance(value, str):
        raise ConfigValidationError(f"{path} must be an ISO date string")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ConfigValidationError(f"{path} must be an ISO date string") from exc


def _int_tuple(value: Any, path: str, *, min_value: int) -> tuple[int, ...]:
    items = _sequence(value, path)
    parsed = tuple(_positive_int(item, f"{path}[]") for item in items)
    if any(item < min_value for item in parsed):
        raise ConfigValidationError(f"{path} values must be >= {min_value}")
    return parsed


def _positive_int(value: Any, path: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ConfigValidationError(f"{path} must be a positive integer")
    return value


def _non_negative_int(value: Any, path: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ConfigValidationError(f"{path} must be a non-negative integer")
    return value


def _fraction(value: Any, path: str) -> float:
    number = _number(value, path)
    if number <= 0 or number >= 1:
        raise ConfigValidationError(f"{path} must be between 0 and 1")
    return number


def _number(value: Any, path: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ConfigValidationError(f"{path} must be numeric")
    return float(value)


def _normalize_for_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _normalize_for_json(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_normalize_for_json(item) for item in value]
    if isinstance(value, date):
        return value.isoformat()
    return value
