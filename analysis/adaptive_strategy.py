"""自适应策略：根据数据特征自动调整搜索策略。"""

from dataclasses import dataclass


@dataclass
class DataProfile:
    """数据特征画像。"""
    task_type: str
    num_nodes: int           # 节点数/用户数
    num_features: int        # 特征维度/物品数
    num_classes: int         # 类别数
    feature_sparsity: float  # 特征稀疏度
    class_imbalance: float   # 类别不平衡度（最大类/最小类）
    avg_degree: float        # 平均度（分类任务）
    label_entropy: float     # 标签熵（分类任务）


class AdaptiveStrategy:
    """自适应策略：根据数据特征自动调整 Agent 行为。

    核心思想：不同数据特征需要不同的搜索策略。
    - 稀疏数据 → 更强正则化
    - 类别不平衡 → 类别加权
    - 图稀疏 → 更多层卷积
    """

    def __init__(self):
        self.recommendations: list[str] = []

    def analyze(self, data: dict, task_type: str) -> DataProfile:
        """分析数据特征。"""
        import numpy as np

        if task_type == "classification":
            return self._analyze_cls(data)
        else:
            return self._analyze_rec(data)

    def _analyze_cls(self, data: dict) -> DataProfile:
        """分析分类数据。"""
        import numpy as np

        features = data["features"].toarray()
        labels = data["labels"]
        train_idx = data["train_idx"]
        adj = data["adj_csr"]

        # 特征稀疏度
        sparsity = (features == 0).sum() / features.size

        # 类别不平衡
        unique, counts = np.unique(labels[train_idx], return_counts=True)
        imbalance = counts.max() / counts.min()

        # 平均度
        avg_degree = adj.nnz / adj.shape[0]

        # 标签熵
        probs = counts / counts.sum()
        entropy = -np.sum(probs * np.log2(probs + 1e-10))

        return DataProfile(
            task_type="classification",
            num_nodes=adj.shape[0],
            num_features=features.shape[1],
            num_classes=int(labels.max()) + 1,
            feature_sparsity=sparsity,
            class_imbalance=imbalance,
            avg_degree=avg_degree,
            label_entropy=entropy,
        )

    def _analyze_rec(self, data: dict) -> DataProfile:
        """分析推荐数据。"""
        import numpy as np

        train_df = data["train_df"]
        num_items = data["num_items"]

        # 用户数
        num_users = train_df["uid"].nunique()

        # 平均序列长度
        avg_seq_len = train_df["item_seq_dedup"].apply(lambda x: len(str(x).split(","))).mean()

        # 物品流行度分布
        item_counts = train_df["target_iid"].value_counts()
        imbalance = item_counts.iloc[0] / item_counts.iloc[-1] if len(item_counts) > 1 else 1.0

        return DataProfile(
            task_type="recommendation",
            num_nodes=num_users,
            num_features=num_items,
            num_classes=0,
            feature_sparsity=1.0 - (avg_seq_len / num_items),
            class_imbalance=imbalance,
            avg_degree=avg_seq_len,
            label_entropy=0.0,
        )

    def get_search_strategy(self, profile: DataProfile) -> dict:
        """根据数据特征生成搜索策略。"""
        strategy = {
            "focus_model": None,
            "dropout_range": [0.0, 0.5],
            "lr_range": [1e-3, 1e-2],
            "hidden_dim_range": [64, 256],
            "num_layers_range": [2, 3],
            "weight_decay_range": [0.0, 5e-4],
            "reasoning": [],
        }

        if profile.task_type == "classification":
            # 稀疏特征 → 更低 dropout
            if profile.feature_sparsity > 0.8:
                strategy["dropout_range"] = [0.0, 0.2]
                strategy["reasoning"].append(f"特征稀疏度 {profile.feature_sparsity:.1%}，建议低 dropout")

            # 类别不平衡 → 更强正则化
            if profile.class_imbalance > 10:
                strategy["weight_decay_range"] = [1e-4, 1e-3]
                strategy["reasoning"].append(f"类别不平衡度 {profile.class_imbalance:.1f}，建议强正则化")

            # 图稀疏 → 更多层
            if profile.avg_degree < 3:
                strategy["num_layers_range"] = [3, 4]
                strategy["reasoning"].append(f"平均度 {profile.avg_degree:.1f}，建议 3-4 层卷积")

            # GraphSAGE 对稀疏图效果好
            strategy["focus_model"] = "GraphSAGE"
            strategy["reasoning"].append("GraphSAGE 对稀疏图效果最好")

        elif profile.task_type == "recommendation":
            # 用户少 → 简单模型
            if profile.num_nodes < 10000:
                strategy["focus_model"] = "BPR_MF"
                strategy["reasoning"].append(f"用户数 {profile.num_nodes}，建议 BPR_MF")
            else:
                strategy["focus_model"] = "ItemCF"
                strategy["reasoning"].append(f"用户数 {profile.num_nodes}，建议 ItemCF")

        return strategy

    def format_for_prompt(self, profile: DataProfile, strategy: dict) -> str:
        """格式化为提示词。"""
        lines = [
            f"## 数据画像",
            f"- 任务类型: {profile.task_type}",
            f"- 节点/用户数: {profile.num_nodes}",
            f"- 特征/物品数: {profile.num_features}",
            f"- 特征稀疏度: {profile.feature_sparsity:.1%}",
            f"- 类别不平衡度: {profile.class_imbalance:.1f}",
        ]

        if profile.task_type == "classification":
            lines.extend([
                f"- 类别数: {profile.num_classes}",
                f"- 平均度: {profile.avg_degree:.1f}",
                f"- 标签熵: {profile.label_entropy:.2f}",
            ])

        lines.extend([
            "",
            f"## 推荐策略",
            f"- 重点模型: {strategy['focus_model']}",
            f"- Dropout 范围: {strategy['dropout_range']}",
            f"- 学习率范围: {strategy['lr_range']}",
            f"- 隐藏维度范围: {strategy['hidden_dim_range']}",
            f"- 层数范围: {strategy['num_layers_range']}",
            "",
            f"## 推理依据",
        ])

        for r in strategy["reasoning"]:
            lines.append(f"- {r}")

        return "\n".join(lines)
