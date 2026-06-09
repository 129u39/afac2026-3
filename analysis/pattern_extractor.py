"""模式提取器：从实验历史中提取可操作的规律。"""

from dataclasses import dataclass
from memory import ExperimentMemory, ExperimentRecord


@dataclass
class ExperimentPattern:
    """实验模式。"""
    pattern_type: str       # "config_relation", "failure_reason", "success_factor"
    description: str        # 人类可读描述
    confidence: float       # 置信度
    action: str             # 建议动作
    evidence: list[str]     # 支持证据


class PatternExtractor:
    """从实验历史中提取模式。

    不是简单的趋势判断，而是理解：
    - 什么配置组合总是有效
    - 什么情况下会失败
    - 为什么某些参数影响大
    """

    def extract(self, memory: ExperimentMemory, task_type: str) -> list[ExperimentPattern]:
        """提取所有模式。"""
        patterns = []
        patterns.extend(self._extract_success_patterns(memory, task_type))
        patterns.extend(self._extract_failure_patterns(memory, task_type))
        patterns.extend(self._extract_config_relations(memory, task_type))
        return patterns

    def _extract_success_patterns(self, memory: ExperimentMemory, task_type: str) -> list[ExperimentPattern]:
        """提取成功模式：什么配置组合总是有效。"""
        patterns = []
        metric_key = "val_accuracy" if task_type == "classification" else "ndcg@k"

        # 找出 Top-5 实验
        top_exps = memory.all_sorted_by_metric(metric_key)[:5]
        if len(top_exps) < 3:
            return patterns

        # 分析共同特征
        if task_type == "classification":
            # 检查模型类型
            model_counts = {}
            for exp in top_exps:
                m = exp.config.get("model_type", "?")
                model_counts[m] = model_counts.get(m, 0) + 1

            dominant_model = max(model_counts, key=model_counts.get)
            if model_counts[dominant_model] >= 3:
                patterns.append(ExperimentPattern(
                    pattern_type="success_factor",
                    description=f"{dominant_model} 在 Top-5 中出现 {model_counts[dominant_model]} 次",
                    confidence=0.8,
                    action=f"focus_on_{dominant_model.lower()}",
                    evidence=[f"Round {e.round_num}: {e.metrics.get(metric_key, 0):.4f}" for e in top_exps[:3]],
                ))

            # 检查层数
            layer_counts = {}
            for exp in top_exps:
                n = exp.config.get("num_layers", 2)
                layer_counts[n] = layer_counts.get(n, 0) + 1

            dominant_layers = max(layer_counts, key=layer_counts.get)
            if layer_counts[dominant_layers] >= 3:
                patterns.append(ExperimentPattern(
                    pattern_type="success_factor",
                    description=f"num_layers={dominant_layers} 在 Top-5 中表现最好",
                    confidence=0.7,
                    action=f"use_{dominant_layers}_layers",
                    evidence=[f"Round {e.round_num}: layers={e.config.get('num_layers')}, metric={e.metrics.get(metric_key, 0):.4f}" for e in top_exps[:3]],
                ))

            # 检查 dropout
            low_dropout = [e for e in top_exps if e.config.get("dropout", 0.5) <= 0.1]
            if len(low_dropout) >= 3:
                patterns.append(ExperimentPattern(
                    pattern_type="success_factor",
                    description=f"低 dropout (<=0.1) 在 Top-5 中出现 {len(low_dropout)} 次",
                    confidence=0.75,
                    action="use_low_dropout",
                    evidence=[f"Round {e.round_num}: dropout={e.config.get('dropout')}, metric={e.metrics.get(metric_key, 0):.4f}" for e in low_dropout],
                ))

        return patterns

    def _extract_failure_patterns(self, memory: ExperimentMemory, task_type: str) -> list[ExperimentPattern]:
        """提取失败模式：什么情况下会失败。"""
        patterns = []
        metric_key = "val_accuracy" if task_type == "classification" else "ndcg@k"

        # 找出 Bottom-5 实验
        bottom_exps = memory.all_sorted_by_metric(metric_key)[-5:]
        if len(bottom_exps) < 3:
            return patterns

        # 检查高 dropout
        high_dropout = [e for e in bottom_exps if e.config.get("dropout", 0) >= 0.4]
        if len(high_dropout) >= 3:
            patterns.append(ExperimentPattern(
                pattern_type="failure_reason",
                description=f"高 dropout (>=0.4) 导致性能下降",
                confidence=0.7,
                action="avoid_high_dropout",
                evidence=[f"Round {e.round_num}: dropout={e.config.get('dropout')}, metric={e.metrics.get(metric_key, 0):.4f}" for e in high_dropout],
            ))

        # 检查高学习率
        high_lr = [e for e in bottom_exps if e.config.get("lr", 0) >= 0.008]
        if len(high_lr) >= 3:
            patterns.append(ExperimentPattern(
                pattern_type="failure_reason",
                description=f"高学习率 (>=0.008) 导致训练不稳定",
                confidence=0.65,
                action="avoid_high_lr",
                evidence=[f"Round {e.round_num}: lr={e.config.get('lr')}, metric={e.metrics.get(metric_key, 0):.4f}" for e in high_lr],
            ))

        return patterns

    def _extract_config_relations(self, memory: ExperimentMemory, task_type: str) -> list[ExperimentPattern]:
        """提取配置关系：参数之间的相互作用。"""
        patterns = []
        metric_key = "val_accuracy" if task_type == "classification" else "ndcg@k"

        if len(memory.records) < 10:
            return patterns

        # 分析 hidden_dim 和 num_layers 的组合
        if task_type == "classification":
            best_config = memory.get_best(task_type)
            if best_config:
                best_hidden = best_config.config.get("hidden_dim", 64)
                best_layers = best_config.config.get("num_layers", 2)
                best_metric = best_config.metrics.get(metric_key, 0)

                # 检查是否有更好的组合未被尝试
                untested = []
                for hidden in [128, 256, 512]:
                    for layers in [3, 4]:
                        tested = any(
                            e.config.get("hidden_dim") == hidden and e.config.get("num_layers") == layers
                            for e in memory.records
                        )
                        if not tested:
                            untested.append((hidden, layers))

                if untested:
                    patterns.append(ExperimentPattern(
                        pattern_type="config_relation",
                        description=f"最佳配置 hidden={best_hidden}, layers={best_layers}，但 {len(untested)} 种组合未测试",
                        confidence=0.6,
                        action="explore_untested_combinations",
                        evidence=[f"未测试: hidden={h}, layers={l}" for h, l in untested[:3]],
                    ))

        return patterns

    def format_for_prompt(self, patterns: list[ExperimentPattern]) -> str:
        """格式化模式为提示词。"""
        if not patterns:
            return "暂未发现明显模式。"

        lines = []
        for p in patterns:
            lines.append(f"[{p.pattern_type}] {p.description} (置信度: {p.confidence:.2f})")
            lines.append(f"  建议动作: {p.action}")
            if p.evidence:
                lines.append(f"  证据: {p.evidence[0]}")
        return "\n".join(lines)
