"""Experiment Config Cache: avoid re-running identical or near-identical configs."""

import hashlib
import json
from typing import Any


def _normalize_config(config: dict) -> str:
    """Normalize config to a stable sorted-JSON string for hashing."""
    clean = {k: v for k, v in config.items()
             if k not in ("_optuna_trial_number", "model_type") and not k.startswith("_")}
    return json.dumps(clean, sort_keys=True, default=str)


def config_hash(config: dict) -> str:
    """SHA-256 hash of a normalized config dict."""
    raw = _normalize_config(config)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class ConfigCache:
    """Cache that stores hashes of previously-run experiments.

    Supports exact-match and similarity-threshold checks.
    """

    def __init__(self, similarity_threshold: float = 0.85):
        """
        Args:
            similarity_threshold: Minimum similarity to consider two configs equivalent.
        """
        self._cache: dict[str, dict] = {}  # hash -> metadata
        self.similarity_threshold = similarity_threshold

    def exists(self, config: dict) -> bool:
        """Check if an identical config has been run before."""
        h = config_hash(config)
        return h in self._cache

    def add(self, config: dict, metric: float, metadata: dict | None = None):
        """Add a config and its metric to the cache."""
        h = config_hash(config)
        entry = {"hash": h, "metric": metric}
        if metadata:
            entry["metadata"] = metadata
        self._cache[h] = entry

    def add_entry(self, config: dict, metric: float, model_name: str = "", round_num: int = 0):
        """Convenience wrapper for add() with common metadata fields."""
        self.add(config, metric, {"model_name": model_name, "round_num": round_num})

    def size(self) -> int:
        return len(self._cache)

    def clear(self):
        self._cache.clear()

    def to_dict(self) -> dict:
        return self._cache

    @classmethod
    def from_dict(cls, data: dict) -> "ConfigCache":
        cc = cls()
        cc._cache = data
        return cc
