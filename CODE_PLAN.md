# AFAC2026-3 Classification Recovery Code Plan

## Version

v2.0

## Current Status

当前验证精度：

```text
0.50 ~ 0.55
```

当前系统：

```text
GraphSAGE
GCN
GAT
GCNII(4层)

Bandit
Optuna
Reflection
KnowledgeBase
```

问题：

当前 Agent 搜索框架复杂度远高于模型能力。

模型搜索空间没有覆盖真正高收益区域。

---

# Phase 0 数据诊断（最高优先级）

## 新增

```text
analysis/data_diagnostics.py
```

### 输出

```python
num_nodes
num_edges
avg_degree
density
num_classes
train_size
test_size
```

### 类别分布

```python
np.bincount(labels[train_idx])
```

输出：

```text
[DATA]
Class Distribution:
Class0=...
Class1=...
...
```

---

## 连通性分析

新增：

```python
from scipy.sparse.csgraph import connected_components
```

输出：

```text
[GRAPH]
Connected Components: 127

Largest Component Ratio: 0.91
```

---

## 孤立节点统计

输出：

```text
[GRAPH]
Isolated Nodes: 532
Ratio: 3.8%
```

---

## 日志格式

```python
logger.info(
    f"[DATA] nodes={n_nodes} "
    f"edges={n_edges} "
    f"avg_degree={avg_degree:.2f}"
)
```

---

# Phase 1 固定验证集

## 当前问题

train_gnn()

切一次。

evaluate()

再切一次。

导致：

```text
训练目标 ≠ 搜索目标
```

---

## 修改

新增：

```python
splitter.py
```

### 初始化时生成

```python
train_idx
val_idx
```

固定保存。

```python
split_cache.pkl
```

---

## 所有模型共用

```python
train_mask
val_mask
```

---

## 输出

```text
[SPLIT]
train=8421
val=2105

seed=42
```

---

# Phase 2 MLP Benchmark

新增：

```text
models/mlp_baseline.py
```

结构：

```python
767
 ->
512
 ->
256
 ->
10
```

---

## 输出日志

```text
[MLP]
epoch=20
val_acc=0.6342
```

---

## 目的

验证：

```text
图结构是否真正有用
```

---

# Phase 3 LightGBM Benchmark

新增：

```text
models/lgb_baseline.py
```

输入：

```python
features.toarray()
```

直接训练。

---

## 输出

```text
[LGB]
num_features=767

val_acc=0.71
```

---

## 判断

如果：

```text
LGB > GNN
```

说明：

```text
图结构质量不足
```

搜索策略要重构。

---

# Phase 4 真正启用 Feature Selection

## 当前问题

FeatureSelector结果未进入PyG。

---

## 修改

classification_to_pyg()

支持：

```python
selected_features
```

参数。

---

## 新逻辑

```python
pyg_data.x =
selected_features
```

而不是：

```python
features.toarray()
```

---

## 日志

```text
[FEATURE]
original_dim=767

selected_dim=128
```

---

# Phase 5 深层GCNII

## 当前

```python
layers=4
```

---

## 新搜索空间

```python
layers

8
16
32
64
```

---

## 新参数

```python
alpha

0.1
0.2
0.3
```

```python
theta

0.5
1.0
1.5
```

---

## 日志

```text
[GCNII]
layers=32
alpha=0.2
theta=1.0

best_val_acc=0.78
```

---

# Phase 6 图结构增强

新增：

```text
features/graph_features.py
```

---

## Degree

```python
degree
log_degree
```

---

## PageRank

```python
pagerank
```

---

## KCore

```python
kcore
```

---

## 拼接

```python
X =
[
feature
degree
pagerank
]
```

---

## 日志

```text
[GRAPH_FEATURE]

degree=True
pagerank=True
kcore=True

new_dim=771
```

---

# Phase 7 Label Propagation

新增：

```text
models/label_propagation.py
```

---

## 目的

验证图同质性。

---

## 输出

```text
[LP]

val_acc=0.76
```

---

## 判断

如果：

```text
LP > GCN
```

说明：

```text
图标签传播能力极强
```

---

# Phase 8 Node2Vec

新增：

```text
models/node2vec_features.py
```

---

## 生成

```python
64维

128维
```

Embedding。

---

## 拼接

```python
X =
[
raw_feature
node2vec
]
```

---

## 日志

```text
[NODE2VEC]

dim=128

walk_length=20

num_walks=10
```

---

# Phase 9 搜索框架降级

当前：

```text
Bandit
Reflection
Planner
```

过重。

---

## 修改

前100轮实验：

```python
RuleBasedSearch
```

仅使用：

```python
Optuna
```

---

## 满足条件

```python
10轮无提升
```

再调用：

```python
Reflection
```

---

## 日志

```text
[SEARCH]

planner=rule

reason=early_stage
```

---

# Phase 10 排行榜输出

新增：

```text
leaderboard.py
```

实时记录：

```python
model
feature
val_acc
runtime
```

---

## 输出

```text
================================================

Rank1

GCNII-32
Feature=128
Acc=0.8012

------------------------------------------------

Rank2

LGB
Acc=0.7864

------------------------------------------------

Rank3

GraphSAGE
Acc=0.7421

================================================
```

---

# Expected Result

## 第一阶段

```text
0.50
 ->
0.65
```

修复验证体系。

---

## 第二阶段

```text
0.65
 ->
0.75
```

Feature Selection + GCNII。

---

## 第三阶段

```text
0.75
 ->
0.80+
```

Node2Vec + Deep GCNII。

---

# Success Criterion

达到：

```text
Val Accuracy >= 0.80
```

且：

```text
连续5次实验波动 < 1%
```

说明搜索空间和评估体系已经稳定。
