"""反馈分析器：从历史实验中提取规律，生成策略建议。"""

from memory import ExperimentMemory


class FeedbackAnalyzer:
    """分析实验历史，生成优化建议。"""

    def analyze(self, memory: ExperimentMemory, task_type: str) -> dict:
        """分析实验历史，返回分析结果和建议。

        Args:
            memory: 实验记忆
            task_type: "classification" 或 "recommendation"

        返回:
            {
                "trend": "improving" | "plateau" | "declining",
                "suggestions": list[str],
                "best_config": dict | None,
                "tried_models": list[str],
                "metric_key": str,
            }
        """
        metric_key = "val_accuracy" if task_type == "classification" else "ndcg@k"
        records = memory.all_sorted_by_metric(metric_key)

        if not records:
            return {
                "trend": "no_data",
                "suggestions": ["运行基线模型获取初始指标"],
                "best_config": None,
                "tried_models": [],
                "metric_key": metric_key,
            }

        tried_models = list({r.model_type for r in memory.records})
        best = records[0]
        recent = memory.recent(3)

        # 分析趋势
        if len(recent) >= 2:
            recent_metrics = [r.metrics.get(metric_key, 0) for r in recent]
            if recent_metrics[-1] > recent_metrics[0]:
                trend = "improving"
            elif recent_metrics[-1] < recent_metrics[-2]:
                trend = "declining"
            else:
                trend = "plateau"
        else:
            trend = "insufficient_data"

        # 生成建议
        suggestions = self._generate_suggestions(
            memory, task_type, trend, tried_models, best, metric_key
        )

        return {
            "trend": trend,
            "suggestions": suggestions,
            "best_config": best.config if best else None,
            "tried_models": tried_models,
            "metric_key": metric_key,
        }

    def _generate_suggestions(
        self,
        memory: ExperimentMemory,
        task_type: str,
        trend: str,
        tried_models: list[str],
        best,
        metric_key: str,
    ) -> list[str]:
        """基于分析结果生成策略建议。"""
        suggestions = []

        if task_type == "classification":
            all_models = {"GCN", "GAT", "GraphSAGE"}
            untried = all_models - set(tried_models)

            if trend == "improving":
                suggestions.append("趋势向好，继续在当前模型上微调超参")
                # 检查是否可以增加容量
                if best.config.get("hidden_dim", 64) < 256:
                    suggestions.append("尝试增加隐藏维度至256")
                if best.config.get("num_layers", 2) < 3:
                    suggestions.append("尝试增加层数至3")

            elif trend == "plateau" or trend == "declining":
                if untried:
                    suggestions.append(f"尝试新模型架构: {', '.join(untried)}")
                suggestions.append("调整学习率（尝试更小或更大的值）")
                suggestions.append("调整 dropout / weight_decay 正则化")
                # 检查是否过拟合
                val_accs = [r.metrics.get("val_accuracy", 0) for r in memory.recent(5)]
                if len(val_accs) >= 3 and max(val_accs) - min(val_accs) > 0.02:
                    suggestions.append("验证指标波动较大，建议增加正则化")

            elif trend == "no_data":
                suggestions.append("先运行 GCN 基线（2层, 64维, lr=0.01）")

        elif task_type == "recommendation":
            all_models = {"Popularity", "ItemCF", "BPR_MF", "SASRec"}
            untried = all_models - set(tried_models)

            if trend == "improving":
                suggestions.append("趋势向好，继续微调")
                if "SASRec" not in tried_models:
                    suggestions.append("尝试 SASRec 捕捉序列模式")

            elif trend == "plateau" or trend == "declining":
                if untried:
                    suggestions.append(f"尝试新模型: {', '.join(untried)}")
                suggestions.append("调整 embedding 维度")
                suggestions.append("尝试融合多个模型的预测")

            elif trend == "no_data":
                suggestions.append("先运行 Popularity 基线获取初始分数")

        return suggestions
