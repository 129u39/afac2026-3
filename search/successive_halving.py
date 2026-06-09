"""Successive Halving：渐进式淘汰搜索策略。"""

import time
from typing import Callable, Any
from dataclasses import dataclass


@dataclass
class HalvingResult:
    """单阶段结果。"""
    config: dict
    metric: float
    train_time: float
    status: str = "success"


class SuccessiveHalving:
    """Successive Halving 搜索策略。

    阶段1: epochs=5, 测试 N configs, 保留 Top-K
    阶段2: epochs=20, 保留 Top-K
    阶段3: epochs=100, 保留 Top-K
    最终: epochs=300, 评估最终模型

    目标：用最少的计算资源筛选出最有潜力的配置。
    """

    def __init__(
        self,
        run_fn: Callable[[dict], Any],
        phases: list[dict] | None = None,
    ):
        """
        Args:
            run_fn: 执行实验的函数，接受 config dict，返回 ExperimentResult
            phases: 阶段配置列表，None 使用默认配置
        """
        self.run_fn = run_fn
        self.phases = phases or [
            {"name": "筛选", "epochs": 5, "top_k": 32},
            {"name": "粗选", "epochs": 20, "top_k": 8},
            {"name": "精选", "epochs": 100, "top_k": 3},
            {"name": "最终", "epochs": 300, "top_k": 1},
        ]

    def run(self, initial_configs: list[dict]) -> list[HalvingResult]:
        """执行 Successive Halving 搜索。

        Args:
            initial_configs: 初始配置列表

        返回:
            最终阶段的结果列表
        """
        configs = initial_configs
        results = []

        for phase_idx, phase in enumerate(self.phases):
            phase_name = phase["name"]
            epochs = phase["epochs"]
            top_k = phase["top_k"]

            print(f"\n[SuccessiveHalving] 阶段 {phase_idx+1}: {phase_name}")
            print(f"  配置数: {len(configs)}, epochs={epochs}, 保留 Top-{top_k}")

            # 修改 epochs
            phase_configs = []
            for cfg in configs:
                cfg_copy = cfg.copy()
                cfg_copy["epochs"] = epochs
                if "patience" in cfg_copy:
                    cfg_copy["patience"] = min(cfg_copy["patience"], epochs // 3)
                phase_configs.append(cfg_copy)

            # 执行实验
            phase_results = self._run_phase(phase_configs)

            # 选择 Top-K
            phase_results.sort(key=lambda r: r.metric, reverse=True)
            results = phase_results[:top_k]

            # 打印结果
            print(f"  完成: {len(phase_results)} 个配置")
            for i, r in enumerate(results[:5]):
                print(f"    #{i+1}: {r.config.get('model_type', '?')} metric={r.metric:.4f} time={r.train_time:.1f}s")

            # 准备下一阶段的配置
            configs = [r.config for r in results]

            if len(configs) <= 1:
                break

        return results

    def _run_phase(self, configs: list[dict]) -> list[HalvingResult]:
        """执行一个阶段的实验。"""
        results = []
        for i, config in enumerate(configs):
            start_time = time.time()
            try:
                result = self.run_fn(config)
                train_time = time.time() - start_time

                # 从 ExperimentResult 中提取 metric
                if hasattr(result, 'metric'):
                    metric = result.metric
                elif hasattr(result, 'metrics'):
                    # 尝试从 metrics 中获取
                    metrics = result.metrics
                    metric = metrics.get('val_accuracy', metrics.get('ndcg@k', 0.0))
                else:
                    metric = 0.0

                results.append(HalvingResult(
                    config=config,
                    metric=metric,
                    train_time=train_time,
                    status="success",
                ))
            except Exception as e:
                results.append(HalvingResult(
                    config=config,
                    metric=0.0,
                    train_time=time.time() - start_time,
                    status="failed",
                ))

        return results
