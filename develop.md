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

---

# 技术文档 V1 — Bandit-Guided + Optuna-Driven Agent

## 目标

将 V0 的规则式 Agent 升级为：

```text
Rule-Based Agent (V0)
    ↓
Bandit-Guided Agent (V1)
    ↓
Optuna-Driven Agent (V1)
    ↓
Memory-Augmented Agent (V1)
```

架构：**Bandit 选模型 → Optuna 调超参 → Reflection 评估决策**

---

## Milestone 1 — 增强实验记忆

**修改**: `memory.py`

- `ExperimentRecord` 增加 `exp_id`, `task`, `timestamp`, `status` 字段
- `ExperimentMemory` 增加:
  - `save(path)` / `load(path)` — JSON 持久化
  - `get_best(task)` — 返回指定任务最佳记录
  - `get_last_k(k)` — 返回最近 k 条
  - `get_by_model(model)` — 按模型筛选

---

## Milestone 2 — 增强预算管理

**修改**: `budget_manager.py`

- `should_continue()` — 综合判断（时间 + 无提升轮数）
- `can_run(config)` — 预估能否跑完一轮
- `remaining_budget()` — 结构化预算信息
- `record_improvement(improved)` — 追踪无提升轮数
- 停止条件: `no_improvement_rounds >= 5` 或 `remaining_time < 10min`

---

## Milestone 3 — UCB Bandit 规划器

**新建**: `planner/bandit.py`, `planner/bandit_planner.py`

- `UCBBandit`:
  - `select_arm()` — UCB 公式选臂
  - `update(arm, reward)` — 更新统计
  - UCB = `mean_reward + c * sqrt(log(t) / n)`
- 分类臂: `["GCN", "GraphSAGE", "GAT"]`
- 推荐臂: `["Popularity", "ItemCF", "BPR_MF", "SASRec"]`
- `BanditPlanner` 替代原 `Planner`

---

## Milestone 4 — Optuna 超参搜索

**新建**: `search/search_space.py`, `search/objective.py`, `search/optuna_planner.py`

- 分类搜索空间: `model_type`, `hidden_dim`, `num_layers`, `dropout`, `lr`, `weight_decay`
- 推荐搜索空间: `embedding_dim`, `lr`, `batch_size`, `l2`
- `OptunaPlanner.next_config()` — 从 Study 采样
- `OptunaPlanner.update_result()` — 更新 Study

---

## Milestone 5 — LightGCN

**新建**: `models/recommendation/lightgcn.py`

- `fit()` / `predict()` / `recommend_topk()`
- 超参: `embedding_dim`, `num_layers`, `lr`, `weight_decay`
- 注册到 `RecommenderSystem`

---

## Milestone 6 — Feedback Analyzer V2

**修改**: `feedback_analyzer.py`

- 分析: `metric_delta`, `training_time`, `improvement_rate`, `variance`
- 输出: `{trend, risk, recommendation}`
- 识别: `improving` / `plateau` / `overfitting` / `unstable`

---

## Milestone 7 — 检索记忆

**新建**: `memory/vector_store.py`, `memory/retriever.py`

- `ConfigEncoder` — 配置编码为向量
- `Retriever.top_k_similar(config, k=5)` — 返回最相似历史实验

---

## Milestone 8 — 反思 Agent

**新建**: `analysis/reflection.py`

- 输入: `recent_experiments`, `feedback`
- 输出: `{observation, reasoning, next_action, confidence}`
- 参与 `planner.next_config()` 决策

---

## Milestone 9 — 轨迹合规

**修改**: `trajectory_logger.py`

- 日志结构: `{round, config, metric, feedback, decision, runtime}`
- 保存: `output/trajectory_classification.json`, `output/trajectory_recommendation.json`
- 每轮完整可复现

---

## V1 最终架构

```text
ExperimentMemory (M1)
        ↓
BanditPlanner (M3) — 选模型
        ↓
OptunaPlanner (M4) — 调超参
        ↓
ExperimentRunner — 训练+评估
        ↓
FeedbackAnalyzer V2 (M6) — 趋势分析
        ↓
ReflectionAgent (M8) — 反思决策
        ↓
TrajectoryLogger V2 (M9) — 日志记录
        ↓
Retriever (M7) — 检索相似实验
```

---

## 依赖新增

```bash
pip install optuna
```

---

## 运行

```bash
python run_classification.py
python run_recommendation.py
```

自动完成: 数据加载 → 实验选择 → 训练 → 验证 → 反馈分析 → 策略更新 → 日志记录 → 提交文件生成
