import tomllib
from pathlib import Path

SCHEMA = {
    "paths": {"data": str, "runs": str, "fonts": str},
    "data": {"funsd_val_pages": (int, 0, 100), "cache_max_side": (int, 320, 8000), "word_cache_height": (int, 32, 128), "workers": (int, 1, 64)},
    "synth": {"pages": (int, 1, 10**6), "workers": (int, 1, 64), "seed": (int, 0, 2**31)},
    "crnn": {"channels": (list, 7, 7), "hidden": (int, 16, 2048), "max_width": (int, 64, 2048)},
    "detector": {
        "channels": (list, 4, 4), "blocks": (list, 4, 4), "fpn": (int, 8, 1024), "k": (float, 1.0, 200.0),
        "shrink": (float, 0.1, 0.9), "thresh_min": (float, 0.0, 1.0), "thresh_max": (float, 0.0, 1.0),
    },
    "train_reader": {
        "batch_size": (int, 1, 4096), "steps": (int, 1, 10**8), "lr": (float, 1e-7, 1.0), "weight_decay": (float, 0.0, 1.0),
        "warmup": (int, 0, 10**7), "synthetic_only_steps": (int, 0, 10**8), "real_fraction": (float, 0.0, 1.0),
        "workers": (int, 0, 64), "threads": (int, 1, 256), "eval_every": (int, 1, 10**8), "save_every": (int, 1, 10**8),
        "log_every": (int, 1, 10**7), "seed": (int, 0, 2**31),
    },
    "train_detector": {
        "batch_size": (int, 1, 1024), "crop": (int, 128, 2048), "steps": (int, 1, 10**8), "lr": (float, 1e-7, 1.0),
        "weight_decay": (float, 0.0, 1.0), "warmup": (int, 0, 10**7), "synthetic_only_steps": (int, 0, 10**8),
        "real_fraction": (float, 0.0, 1.0), "workers": (int, 0, 64), "threads": (int, 1, 256), "eval_every": (int, 1, 10**8),
        "save_every": (int, 1, 10**8), "log_every": (int, 1, 10**7), "min_scale": (float, 0.1, 4.0), "max_scale": (float, 0.1, 4.0), "seed": (int, 0, 2**31),
    },
    "detect": {"max_side": (int, 256, 8192), "threshold": (float, 0.0, 1.0), "box_threshold": (float, 0.0, 1.0), "unclip": (float, 0.5, 4.0), "min_size": (float, 0.0, 100.0)},
    "pipeline": {"beam_width": (int, 0, 64), "find_page": bool, "dewarp": bool, "min_dewarp_lines": (int, 2, 1000), "min_dewarp_bend": (float, 0.0, 10.0)},
    "eval": {"workers": (int, 1, 64)},
}


class ConfigError(ValueError):
    pass


def _check(section, key, value, rule):
    where = f"[{section}] {key}"
    if rule is str or rule is bool:
        if not isinstance(value, rule):
            raise ConfigError(f"{where} must be {rule.__name__}")
        return
    kind = rule[0]
    if kind is list:
        if not isinstance(value, list) or not rule[1] <= len(value) <= rule[2]:
            raise ConfigError(f"{where} must be a list of length {rule[1]}")
        if not all(isinstance(v, int) and v > 0 for v in value):
            raise ConfigError(f"{where} must hold positive integers")
        return
    if kind is float and isinstance(value, int) and not isinstance(value, bool):
        value = float(value)
    if not isinstance(value, kind) or isinstance(value, bool):
        raise ConfigError(f"{where} must be {kind.__name__}")
    if not rule[1] <= value <= rule[2]:
        raise ConfigError(f"{where} must be between {rule[1]} and {rule[2]}")


def validate(config):
    for section, rules in SCHEMA.items():
        if section not in config:
            raise ConfigError(f"missing section [{section}]")
        for key, rule in rules.items():
            if key not in config[section]:
                raise ConfigError(f"missing [{section}] {key}")
            _check(section, key, config[section][key], rule)
        unknown = set(config[section]) - set(rules)
        if unknown:
            raise ConfigError(f"unknown keys in [{section}]: {sorted(unknown)}")
    for section in ("train_reader", "train_detector"):
        for key in ("lr", "weight_decay", "real_fraction", "min_scale", "max_scale"):
            if key in config[section]:
                config[section][key] = float(config[section][key])
    if config["detector"]["thresh_min"] >= config["detector"]["thresh_max"]:
        raise ConfigError("[detector] thresh_min must be below thresh_max")
    if config["train_detector"]["min_scale"] > config["train_detector"]["max_scale"]:
        raise ConfigError("[train_detector] min_scale must not exceed max_scale")
    if config["detector"]["fpn"] % 4:
        raise ConfigError("[detector] fpn must be divisible by 4")
    return config


def load(path="config.toml"):
    with open(path, "rb") as handle:
        config = tomllib.load(handle)
    config = validate(config)
    config["root"] = str(Path(path).resolve().parent)
    return config


def path(config, key):
    value = Path(config["paths"][key])
    return value if value.is_absolute() else Path(config["root"]) / value
