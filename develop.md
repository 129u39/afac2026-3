# 技术文档 — AFAC2026 自动化实验 Agent

## 1. 系统模块详解

### 1.1 数据加载 (`data_loader.py`)

**分类任务**：从 `.npz` 文件加载 CSR 格式的邻接矩阵和节点特征矩阵，转换为 PyG `Data` 对象。

- 邻接矩阵：`scipy.sparse.csr_matrix` → COO → `edge_index` + `edge_weight`
- 特征矩阵：CSR → 稠密 `torch.Tensor`（13752 × 767，约 40MB）
- 标签：训练节点真实标签，测试节点为 -1
- Mask：`train_mask` / `test_mask` 布尔向量

**推荐任务**：从 CSV 文件加载用户交互序列、用户特征、物品特征。

- 序列三种表示：`item_seq_raw`（含重复）、`item_seq_dedup`（去重）、`item_seq_counts`（频次）
- 冷启动处理：测试集中部分用户序列为空，降级到流行度推荐

### 1.2 模型层 (`models/`)

#### 1.2.1 GNN 分类器 (`gnn_classifier.py`)

统一接口 `GNNClassifier`，通过 `model_type` 参数切换架构：

| 架构 | 层类型 | 特点 |
|------|--------|------|
| GCN | `GCNConv` | 谱图卷积，简单高效 |
| GAT | `GATConv` | 多头注意力，8头中间层 + 单头输出层 |
| GraphSAGE | `SAGEConv` | 邻居采样聚合，适合大图 |

**网络结构**：
```
Input(767) → Linear → [GNN Layer + BatchNorm + ReLU + Dropout] × N → Linear(10)
```

**训练细节**：
- 优化器：Adam
- 损失函数：交叉熵（仅在训练节点上计算）
- 早停：patience 轮内验证集无提升则停止
- 验证划分：从 train_idx 中按 80:20 分层采样

#### 1.2.2 推荐模型 (`recommender.py`)

| 模型 | 原理 | 训练成本 | 适用场景 |
|------|------|----------|----------|
| Popularity | 全局热门物品排序 | 零 | 冷启动基线 |
| ItemCF | 物品共现余弦相似度 | 中（~40s） | 利用协同信号 |
| BPR-MF | 矩阵分解 + BPR 损失 | 低（~4s） | 用户/物品特征融合 |
| SASRec | 自注意力序列模型 | 高（~74s） | 捕捉序列模式 |

**BPR-MF 细节**：
- 用户/物品 Embedding 维度：32/64/128
- 训练样本：(user, pos_item, neg_item) 三元组，负采样 1:1
- 损失：`-log(sigmoid(pos_score - neg_score))`

**SASRec 细节**：
- 序列最大长度：50（截断取最近50个交互）
- Transformer Encoder：2层，2头，FFN维度 = 4 × embedding_dim
- 因果 Mask：只看前面的物品
- Padding：左侧填充 0

**冷启动处理**：
- 空序列用户 → 降级到 Popularity
- BPR-MF：未知用户使用 user_idx=0（全局平均向量）
- SASRec：空序列全为 padding，输出近似均匀分布

### 1.3 评估模块 (`evaluate.py`)

**分类评估**：
- 从 `train_idx` 中按 80:20 分层划分 train/val
- 计算验证集 Accuracy

**推荐评估**：
- 从 `train_df` 中按 80:20 划分 train/val
- 在子集上重新 fit 模型（保证评估公平）
- 计算 NDCG@10 和 Hit@10

**注意**：推荐评估会重新训练模型，因此每轮实验实际包含一次训练 + 一次验证训练。

### 1.4 实验记忆 (`memory.py`)

`ExperimentMemory` 存储所有实验记录：

```python
@dataclass
class ExperimentRecord:
    round_num: int           # 实验轮次
    model_type: str          # 模型类型
    config: dict             # 完整配置
    metrics: dict            # 评估指标
    elapsed_seconds: float   # 耗时
    notes: str               # 备注
```

支持查询：
- `best_classification()` / `best_recommendation()` — 最佳实验
- `by_model_type(type)` — 按模型筛选
- `recent(n)` — 最近 n 条
- `all_sorted_by_metric(key)` — 按指标排序

### 1.5 反馈分析器 (`feedback_analyzer.py`)

分析历史实验，输出：
- **趋势**：`improving` / `plateau` / `declining` / `insufficient_data`
- **建议**：基于规则的策略建议列表

**建议生成规则**（分类任务）：
- 趋势提升 → 继续微调，尝试增加容量
- 趋势停滞/下降 → 尝试未试过的架构，调整正则化
- 波动大 → 增加 dropout / weight_decay

**建议生成规则**（推荐任务）：
- 趋势提升 → 继续微调，尝试 SASRec
- 趋势停滞/下降 → 尝试更复杂模型，调 embedding 维度

### 1.6 策略规划器 (`planner.py`)

**初始策略**：
- 分类：GCN (2层, 64维, lr=0.01)
- 推荐：Popularity

**探索策略**：
1. 60% 概率尝试未试过的模型架构
2. 30% 概率在最佳配置上微调（随机修改一个超参）
3. 10% 概率随机探索

**微调操作**：
- 随机修改一个超参：hidden_dim / dropout / lr / weight_decay / num_layers
- 新值从搜索空间中随机选取

**停止条件**：
- 预算耗尽（剩余 < 平均轮耗时 或 < 60s）
- 连续 5 轮无提升

### 1.7 预算管理器 (`budget_manager.py`)

- 总预算：7200s（2小时）
- 安全余量：300s（预留生成提交）
- 有效预算：6900s
- 初始预估轮耗时：120s
- 动态更新：根据已完成轮次的平均耗时调整

### 1.8 过程日志 (`trajectory_logger.py`)

JSON 格式记录每轮实验，B榜要求的字段：
- `round` — 实验轮次
- `config` — 当前实验配置
- `metrics` — 评估指标
- `feedback` — 反馈信息（趋势 + 建议）
- `strategy` — 下一轮优化策略
- `elapsed_seconds` — 本轮耗时

---

## 2. 超参搜索空间

### 2.1 分类任务

| 参数 | 搜索范围 | 说明 |
|------|----------|------|
| `model_type` | GCN, GAT, GraphSAGE | 模型架构 |
| `hidden_dim` | 64, 128, 256 | 隐藏层维度 |
| `num_layers` | 2, 3 | GNN 层数 |
| `dropout` | 0.0, 0.3, 0.5 | Dropout 比率 |
| `lr` | 0.001, 0.005, 0.01 | 学习率 |
| `weight_decay` | 0.0, 5e-4 | L2 正则化 |
| `epochs` | 200 | 最大训练轮次 |
| `patience` | 30 | 早停轮次 |

### 2.2 推荐任务

| 参数 | 搜索范围 | 说明 |
|------|----------|------|
| `model_type` | Popularity, ItemCF, BPR_MF, SASRec | 模型类型 |
| `embedding_dim` | 32, 64, 128 | Embedding 维度 |
| `lr` | 0.001, 0.005, 0.01 | 学习率 |
| `epochs` | 50, 100 | 训练轮次 |
| `batch_size` | 256, 512 | 批大小 |
| `weight_decay` | 0.0, 1e-5 | L2 正则化 |

---

## 3. 全局配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `SEED` | 42 | 随机种子 |
| `VAL_RATIO` | 0.2 | 验证集比例 |
| `TOTAL_BUDGET_SECONDS` | 7200 | 总预算（秒） |
| `SAFETY_MARGIN_SECONDS` | 300 | 安全余量（秒） |

---

## 4. 数据集参数

### 4.1 A1 产品分类

| 属性 | 值 |
|------|-----|
| 节点数 | 13,752 |
| 边数 | 27,995 |
| 特征维度 | 767 |
| 类别数 | 10 |
| 训练节点 | 11,001 |
| 测试节点 | 2,751 |
| 特征稀疏度 | ~99.7% |

### 4.2 A2 产品推荐

| 属性 | 值 |
|------|-----|
| 用户数 | 50,000 |
| 物品数 | 2,156 |
| 训练集 | 40,000 |
| 测试集 | 10,000 |
| 用户特征 | 8 维（u_cat_01~08） |
| 物品特征 | 4 维（i_cat_01~03, i_bucket_01） |

---

## 5. 性能基准

基于 3 分钟预算的快速测试结果：

### 分类任务（7 轮实验）

| 轮次 | 模型 | 隐藏维度 | 层数 | Dropout | LR | Val Acc |
|------|------|----------|------|---------|-----|---------|
| 0 | GCN | 64 | 2 | 0.5 | 0.01 | 0.2204 |
| 1 | GCN | 256 | 2 | 0.5 | 0.01 | 0.2140 |
| 2 | GAT | 64 | 2 | 0.5 | 0.01 | 0.2117 |
| 3 | GraphSAGE | 64 | 2 | 0.0 | 0.001 | 0.2267 |
| 4 | GraphSAGE | 64 | 2 | 0.0 | 0.001 | 0.2303 |
| 5 | GraphSAGE | 64 | 3 | 0.0 | 0.001 | **0.2517** |
| 6 | GraphSAGE | 64 | 3 | 0.5 | 0.001 | 0.2254 |

**观察**：GraphSAGE 在此数据集上表现最好，3层优于2层，无 Dropout 优于有 Dropout。

### 推荐任务（模型对比）

| 模型 | 训练时间 | 说明 |
|------|----------|------|
| Popularity | <1s | 全局热门基线 |
| ItemCF | ~38s | 物品协同过滤 |
| BPR-MF | ~4s | 矩阵分解 |
| SASRec | ~74s | 自注意力序列模型 |

---

## 6. 已知限制与改进方向

### 当前限制

1. **评估开销**：推荐任务的 `evaluate_recommendation` 会重新训练模型，增加约 2× 时间
2. **ItemCF 训练慢**：构建共现矩阵需 ~40s，限制了快速迭代
3. **SASRec 预测质量**：少量 epoch 训练时预测趋于均匀分布
4. **特征利用不足**：推荐任务中用户/物品特征未被充分使用（仅 BPR-MF 隐式利用）

### 改进方向

1. **集成学习**：融合多个模型的预测分数
2. **更高效的搜索**：贝叶斯优化替代随机搜索
3. **特征工程**：为 GNN 添加更多结构特征（度中心性、社区检测等）
4. **预训练 Embedding**：使用 Qwen text-embedding-v4 生成物品语义向量
5. **自适应预算分配**：根据模型训练速度动态调整每轮预算
