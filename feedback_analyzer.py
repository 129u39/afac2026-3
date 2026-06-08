"""反馈分析器 V2：从历史实验中提取规律，识别趋势与风险，生成策略建议。"""

import math
from memory import ExperimentMemory


class FeedbackAnalyzer:
    """分析实验历史，生成优化建议。

    V2 增强:
    - 分析维度: metric_delta, training_time, improvement_rate, variance
    - 风险识别: overfitting, unstable, plateau
    - 四种状态: improving / plateau / overfitting / unstable
    """

    def analyze(self, memory: ExperimentMemory, task_type: str) -> dict:
        """分析实验历史，返回分析结果和建议。

        Args:
            memory: 实验记忆
            task_type: "classification" 或 "recommendation"

        返回:
            {
                "trend": "improving" | "plateau" | "overfitting" | "unstable" | "declining",
                "risk": "none" | "overfitting" | "unstable" | "plateau",
                "suggestions": list[str],
                "best_config": dict | None,
                "tried_models": list[str],
                "metric_key": str,
                "metric_delta": float,
                "improvement_rate": float,
                "variance": float,
                "avg_training_time": float,
                "recommendation": str,
            }
        """
        metric_key = "val_accuracy" if task_type == "classification" else "ndcg@k"
        records = memory.all_sorted_by_metric(metric_key)

        if not records:
            return {
                "trend": "no_data",
                "risk": "none",
                "suggestions": ["运行基线模型获取初始指标"],
                "best_config": None,
                "tried_models": [],
                "metric_key": metric_key,
                "metric_delta": 0.0,
                "improvement_rate": 0.0,
                "variance": 0.0,
                "avg_training_time": 0.0,
                "recommendation": "start_with_baseline",
            }

        tried_models = list({r.model_type for r in memory.records})
        best = records[0]
        recent = memory.recent(5)

        # ── 计算分析维度 ──────────────────────────────

        # metric_delta: 最近一轮与最佳的差距
        recent_metric = recent[-1].metrics.get(metric_key, 0)
        best_metric = best.metrics.get(metric_key, 0)
        metric_delta = recent_metric - best_metric

        # improvement_rate: 最近 5 轮中有提升的比例
        improvement_count = 0
        for i in range(1, len(recent)):
            prev = recent[i - 1].metrics.get(metric_key, 0)
            curr = recent[i].metrics.get(metric_key, 0)
            if curr > prev + 1e-6:
                improvement_count += 1
        improvement_rate = improvement_count / max(len(recent) - 1, 1)

        # variance: 最近 5 轮指标的方差
        recent_metrics = [r.metrics.get(metric_key, 0) for r in recent]
        variance = self._variance(recent_metrics)

        # avg_training_time: 平均训练时间
        training_times = [r.elapsed_seconds for r in recent]
        avg_training_time = sum(training_times) / len(training_times) if training_times else 0

        # ── 识别趋势与风险 ────────────────────────────

        trend = self._classify_trend(recent_metrics, improvement_rate, variance)
        risk = self._classify_risk(trend, variance, recent_metrics)

        # ── 生成建议 ──────────────────────────────────

        suggestions = self._generate_suggestions(
            memory, task_type, trend, risk, tried_models, best, metric_key, variance
        )

        recommendation = self._generate_recommendation(trend, risk)

        return {
            "trend": trend,
            "risk": risk,
            "suggestions": suggestions,
            "best_config": best.config if best else None,
            "tried_models": tried_models,
            "metric_key": metric_key,
            "metric_delta": metric_delta,
            "improvement_rate": improvement_rate,
            "variance": variance,
            "avg_training_time": avg_training_time,
            "recommendation": recommendation,
        }

    def _classify_trend(
        self, recent_metrics: list[float], improvement_rate: float, variance: float
    ) -> str:
        """分类趋势。

        识别:
        - improving: 改善率 > 0.5 且方差低
        - plateau: 改善率低但方差也低
        - overfitting: 指标先升后降
        - unstable: 方差高
        - declining: 持续下降
        """
        if len(recent_metrics) < 2:
            return "insufficient_data"

        # 检查是否过拟合（先升后降）
        if len(recent_metrics) >= 3:
            peak_idx = recent_metrics.index(max(recent_metrics))
            if 0 < peak_idx < len(recent_metrics) - 1:
                # 最高点在中间，后面下降
                after_peak = recent_metrics[peak_idx + 1:]
                if all(m < recent_metrics[peak_idx] for m in after_peak):
                    return "overfitting"

        # 检查是否不稳定
        if variance > 0.001 and len(recent_metrics) >= 3:
            return "unstable"

        # 检查是否在提升
        if improvement_rate >= 0.5:
            return "improving"

        # 检查是否在下降
        if len(recent_metrics) >= 2 and recent_metrics[-1] < recent_metrics[0] - 1e-6:
            return "declining"

        # 默认：平台期
        return "plateau"

    def _classify_risk(
        self, trend: str, variance: float, recent_metrics: list[float]
    ) -> str:
        """分类风险等级。"""
        if trend == "overfitting":
            return "overfitting"
        if trend == "unstable":
            return "unstable"
        if trend == "plateau" and len(recent_metrics) >= 5:
            return "plateau"
        return "none"

    def _generate_recommendation(self, trend: str, risk: str) -> str:
        """生成高层推荐策略。"""
        if trend == "improving":
            return "continue_current_direction"
        if trend == "overfitting":
            return "increase_regularization"
        if trend == "unstable":
            return "reduce_learning_rate"
        if trend == "plateau":
            return "try_new_architecture"
        if trend == "declining":
            return "revert_to_best"
        return "start_with_baseline"

    def _variance(self, values: list[float]) -> float:
        """计算方差。"""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        return sum((v - mean) ** 2 for v in values) / len(values)

    def _generate_suggestions(
        self,
        memory: ExperimentMemory,
        task_type: str,
        trend: str,
        risk: str,
        tried_models: list[str],
        best,
        metric_key: str,
        variance: float,
    ) -> list[str]:
        """基于分析结果生成策略建议。"""
        suggestions = []

        if task_type == "classification":
            all_models = {"GCN", "GAT", "GraphSAGE"}
            untried = all_models - set(tried_models)

            if trend == "improving":
                suggestions.append("趋势向好，继续在当前模型上微调超参")
                if best.config.get("hidden_dim", 64) < 256:
                    suggestions.append("尝试增加隐藏维度至256")
                if best.config.get("num_layers", 2) < 3:
                    suggestions.append("尝试增加层数至3")

            elif trend == "overfitting":
                suggestions.append("检测到过拟合，增加正则化")
                suggestions.append("增大 dropout 或 weight_decay")
                suggestions.append("减少模型容量（降低 hidden_dim）")

            elif trend == "unstable":
                suggestions.append("训练不稳定，降低学习率")
                suggestions.append("增加 batch_size（如有）")
                suggestions.append("尝试更保守的超参")

            elif trend == "plateau":
                if untried:
                    suggestions.append(f"尝试新模型架构: {', '.join(untried)}")
                suggestions.append("调整学习率（尝试更小或更大的值）")
                suggestions.append("调整 dropout / weight_decay 正则化")

            elif trend == "declining":
                suggestions.append("指标下降，回退到最佳配置")
                if untried:
                    suggestions.append(f"尝试新模型架构: {', '.join(untried)}")

            elif trend == "no_data":
                suggestions.append("先运行 GCN 基线（2层, 64维, lr=0.01）")

        elif task_type == "recommendation":
            all_models = {"Popularity", "ItemCF", "BPR_MF", "SASRec", "LightGCN"}
            untried = all_models - set(tried_models)

            if trend == "improving":
                suggestions.append("趋势向好，继续微调")
                if "SASRec" not in tried_models:
                    suggestions.append("尝试 SASRec 捕捉序列模式")
                if "LightGCN" not in tried_models:
                    suggestions.append("尝试 LightGCN 图卷积")

            elif trend == "overfitting":
                suggestions.append("检测到过拟合，增加正则化")
                suggestions.append("增大 weight_decay")
                suggestions.append("减少 embedding_dim")

            elif trend == "unstable":
                suggestions.append("训练不稳定，降低学习率")
                suggestions.append("增大 batch_size")

            elif trend == "plateau":
                if untried:
                    suggestions.append(f"尝试新模型: {', '.join(untried)}")
                suggestions.append("调整 embedding 维度")
                suggestions.append("尝试融合多个模型的预测")

            elif trend == "declining":
                suggestions.append("指标下降，回退到最佳配置")
                if untried:
                    suggestions.append(f"尝试新模型: {', '.join(untried)}")

            elif trend == "no_data":
                suggestions.append("先运行 Popularity 基线获取初始分数")

        return suggestions
