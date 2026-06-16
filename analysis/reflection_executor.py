"""Reflection Executor: Apply reflection results to search space."""

import copy
import random
from typing import Any

from llm.schemas import ReflectionResult
from search.topk_pool import TopKPool


class ReflectionExecutor:
    """Maps ReflectionAgent output to concrete search-space modifications.

    Supported actions:
    - reduce_learning_rate: halve lr
    - increase_regularization: increase dropout + weight_decay
    - try_new_architecture: ban best model, explore untried
    - fine_tune_best: sample near best config
    """

    def __init__(self, task_type: str):
        self.task_type = task_type
        self._banned_models: set[str] = set()
        self._action_history: list[str] = []
        self._last_search_space_mod: dict[str, Any] = {}
        self._topk_pool: TopKPool | None = None

    def set_topk_pool(self, pool: TopKPool | None):
        self._topk_pool = pool

    def apply(
        self,
        reflection: ReflectionResult | None,
        tried_models: list[str],
        candidate_configs: list[dict],
    ) -> dict[str, Any]:
        if reflection is None:
            return self._last_search_space_mod

        action = reflection.next_action
        self._action_history.append(action)

        if action == "reduce_learning_rate":
            self._last_search_space_mod = self._reduce_lr(candidate_configs)
        elif action == "increase_regularization":
            self._last_search_space_mod = self._increase_reg(candidate_configs)
        elif action == "try_new_architecture":
            self._last_search_space_mod = self._try_new(tried_models)
        elif action == "fine_tune_best":
            self._last_search_space_mod = self._fine_tune(candidate_configs)

        return self._last_search_space_mod

    def get_action_history(self) -> list[str]:
        return self._action_history.copy()

    def reset_bans(self):
        self._banned_models.clear()

    def _reduce_lr(self, candidate_configs: list[dict]) -> dict[str, Any]:
        mod = {"action": "reduce_learning_rate", "modifications": []}
        for cfg in candidate_configs:
            if "lr" in cfg:
                old_lr = cfg["lr"]
                cfg["lr"] = old_lr * 0.5
                mod["modifications"].append(f"lr: {old_lr:.6f} -> {cfg['lr']:.6f}")
        return mod

    def _increase_reg(self, candidate_configs: list[dict]) -> dict[str, Any]:
        mod = {"action": "increase_regularization", "modifications": []}
        for cfg in candidate_configs:
            if "dropout" in cfg:
                old_dropout = cfg["dropout"]
                cfg["dropout"] = min(old_dropout + 0.1, 0.8)
                mod["modifications"].append(f"dropout: {old_dropout} -> {cfg['dropout']}")
            if "weight_decay" in cfg:
                old_wd = cfg["weight_decay"]
                cfg["weight_decay"] = old_wd * 2.0
                mod["modifications"].append(f"weight_decay: {old_wd:.6f} -> {cfg['weight_decay']:.6f}")
        return mod

    def _try_new(self, tried_models: list[str]) -> dict[str, Any]:
        if self.task_type == "classification":
            all_models = {"GCN", "GAT", "GraphSAGE", "LightGBM", "MLP", "GCNII"}
        else:
            all_models = {"Popularity", "ItemCF", "BPR_MF", "LightGCN", "SASRec"}

        untried = list(all_models - set(tried_models) - self._banned_models)
        if untried:
            preferred = random.choice(untried)
            self._banned_models.add(preferred)
            return {"action": "try_new_architecture", "preferred_model": preferred, "untried": untried}
        else:
            self.reset_bans()
            return {"action": "try_new_architecture", "preferred_model": None, "note": "all_tried"}

    def _fine_tune(self, candidate_configs: list[dict]) -> dict[str, Any]:
        mod = {"action": "fine_tune_best", "modifications": []}
        if self._topk_pool and len(self._topk_pool) > 0:
            best_config = self._topk_pool.get_best()
            if best_config:
                for cfg in candidate_configs:
                    for key in best_config:
                        if key in cfg and key not in ("model_type", "_optuna_trial_number"):
                            old_val = cfg[key]
                            if isinstance(old_val, (int, float)):
                                factor = 0.8 + 0.4 * random.random()
                                cfg[key] = old_val * factor
                                mod["modifications"].append(f"{key}: {old_val} -> {cfg[key]:.4f}")
        return mod
