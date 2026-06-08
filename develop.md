# 技术文档 V0 — AFAC2026 自动化实验 Agent

## 1. 模型

### 分类（GNN）

| 架构 | 层类型 | 特点 |
|------|--------|------|
| GCN | `GCNConv` | 谱图卷积，简单高效 |
| GAT | `GATConv` | 多头注意力（8头） |
| GraphSAGE | `SAGEConv` | 邻居采样聚合 |

结构：`Input(767) → [GNN + BN + ReLU + Dropout] × N → Linear(10)`

训练：Adam，交叉熵，早停 patience=30，验证 80:20 分层划分。

### 推荐

| 模型 | 原理 | 训练时间 |
|------|------|----------|
| Popularity | 全局热门 | <1s |
| ItemCF | 物品共现余弦相似度 | ~38s |
| BPR-MF | 矩阵分解 + BPR 损失 | ~4s |
| SASRec | 自注意力序列（2层2头，max_len=50） | ~74s |

冷启动：空序列用户降级到 Popularity。

---

## 2. 超参搜索空间

### 分类

| 参数 | 范围 |
|------|------|
| model_type | GCN, GAT, GraphSAGE |
| hidden_dim | 64, 128, 256 |
| num_layers | 2, 3 |
| dropout | 0.0, 0.3, 0.5 |
| lr | 0.001, 0.005, 0.01 |
| weight_decay | 0.0, 5e-4 |
| epochs / patience | 200 / 30 |

### 推荐

| 参数 | 范围 |
|------|------|
| model_type | Popularity, ItemCF, BPR_MF, SASRec |
| embedding_dim | 32, 64, 128 |
| lr | 0.001, 0.005, 0.01 |
| epochs | 50, 100 |
| batch_size | 256, 512 |
| weight_decay | 0.0, 1e-5 |

---

## 3. 全局参数

| 参数 | 值 | 说明 |
|------|----|------|
| SEED | 42 | 随机种子 |
| VAL_RATIO | 0.2 | 验证比例 |
| TOTAL_BUDGET | 7200s | 总预算 |
| SAFETY_MARGIN | 300s | 安全余量 |

---

## 4. 数据集

| | A1 分类 | A2 推荐 |
|---|---------|---------|
| 节点/用户 | 13,752 | 50,000 |
| 特征/物品 | 767维 | 2,156件 |
| 训练 | 11,001 | 40,000 |
| 测试 | 2,751 | 10,000 |
| 类别/特征 | 10类 | 8+4维 |

---

## 5. Agent 策略

- **初始**：分类→GCN，推荐→Popularity
- **探索**：60% 尝试新架构，30% 微调最佳配置，10% 随机
- **停止**：预算耗尽 或 连续5轮无提升

---

## 6. V0 基准（3分钟快速测试）

### 分类

| 轮次 | 模型 | dim | layers | dropout | lr | Acc |
|------|------|-----|--------|---------|-----|-----|
| 0 | GCN | 64 | 2 | 0.5 | 0.01 | 0.220 |
| 3 | GraphSAGE | 64 | 2 | 0.0 | 0.001 | 0.227 |
| 5 | GraphSAGE | 64 | 3 | 0.0 | 0.001 | **0.252** |

**结论**：GraphSAGE > GCN > GAT；3层 > 2层；低 dropout 更优。

### 推荐

| 模型 | 训练时间 |
|------|----------|
| Popularity | <1s |
| BPR-MF | ~4s |
| ItemCF | ~38s |
| SASRec | ~74s |

---

## 7. 待改进

1. 推荐评估重新训练模型，开销 2×
2. ItemCF 共现矩阵构建慢（~40s）
3. SASRec 少量 epoch 预测趋于均匀
4. 用户/物品特征利用不充分
5. 可尝试集成融合、贝叶斯优化、Qwen embedding
