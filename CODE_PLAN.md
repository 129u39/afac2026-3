# AFAC2026-A1 Classification Improvement Plan

## Current Status

### Dataset

* Nodes: 13,752
* Features: 767
* Classes: 10
* Train Nodes: 11,001
* Test Nodes: 2,751

### Graph Statistics

* Average Degree: 2.04
* Isolated Node Ratio: 31.0%
* Feature Sparsity: 82.2%
* Class Imbalance: 17.7×

### Current Best Result

| Model             | Validation Accuracy |
| ----------------- | ------------------- |
| Hybrid            | 0.6243              |
| Enhanced LightGBM | 0.5043              |
| LGB Ensemble      | 0.4993              |
| Raw LightGBM      | 0.3857              |

---

# Goal

## Short-term

Validation Accuracy ≥ 0.70

## Mid-term

Validation Accuracy ≥ 0.75

## Final Target

Validation Accuracy ≥ 0.80

---

# Phase 1: Benchmark Framework

## Objective

建立可信基线，明确性能来源。

## Tasks

新增目录：

```text
models/
├── mlp.py
├── graphsage.py
├── gcnii.py
├── label_propagation.py
```

统一接口：

```python
fit()
predict()
evaluate()
```

新增：

```text
benchmark_runner.py
```

统一输出：

```text
================================
Benchmark Results
================================

MLP                0.xxxx
LightGBM           0.xxxx
GraphSAGE          0.xxxx
GCNII              0.xxxx
LabelPropagation   0.xxxx

================================
```

## Deliverable

获得可靠基线排行榜。

---

# Phase 2: Stable Validation Split

## Problem

当前训练与评估存在不同划分。

会导致：

* Optuna误导
* Bandit误导
* 结果不可复现

## Tasks

新增：

```text
utils/fixed_split.py
```

固定：

```python
seed = 42
```

保存：

```text
cache/split.pkl
```

内容：

```python
train_idx
val_idx
```

所有模型共用。

## Log

```text
[SPLIT]

train=8800
val=2201
seed=42
```

---

# Phase 3: Long-tail Learning

## Problem

类别不平衡：

```text
17.7×
```

## Tasks

新增：

```text
losses/
├── weighted_ce.py
├── focal_loss.py
```

### Weighted Cross Entropy

```python
weight[c] =
N / (num_classes * count[c])
```

### Focal Loss

搜索：

```python
gamma ∈ {1,2,3}
```

## Logs

```text
[CLASS_WEIGHT]

min=0.12
max=2.15
```

```text
[LOSS]

type=focal
gamma=2
```

## Metrics

同时记录：

```python
accuracy
macro_f1
balanced_accuracy
```

## Deliverable

Accuracy > 0.68

---

# Phase 4: GraphSAGE Main Track

## Reason

当前图：

```text
avg_degree=2.04
isolated=31%
```

GraphSAGE通常比GCN更稳。

## Search Space

```python
hidden_dim:
64
128
256

layers:
2
3
4

dropout:
0.3
0.5
0.7

lr:
1e-2
5e-3
1e-3
```

## Log

```text
[SAGE]

hidden=256
layers=3
dropout=0.5

acc=0.712
```

## Deliverable

Accuracy > 0.70

---

# Phase 5: Deep GCNII

## Problem

当前GCNII仅4层。

没有发挥GCNII优势。

## Search Space

```python
layers:
8
16
32

alpha:
0.1
0.2
0.3

theta:
0.5
1.0
1.5
```

## Log

```text
[GCNII]

layers=16
alpha=0.2
theta=1.0

acc=0.756
```

## Deliverable

超过GraphSAGE。

---

# Phase 6: Label Propagation

## Objective

验证图结构同质性。

## Tasks

新增：

```text
models/label_propagation.py
```

## Log

```text
[LP]

acc=0.xxxx
```

## Decision Rule

### LP > 0.75

说明：

图结构极强。

后续路线：

```text
LP + GCNII
```

### LP < 0.60

说明：

特征主导。

后续路线：

```text
MLP + LightGBM
```

---

# Phase 7: Isolated Node Modeling

## Problem

31%节点无邻居。

GNN无法聚合信息。

## Tasks

新增：

```text
features/node_stats.py
```

构造：

```python
degree
is_isolated
```

拼接：

```python
X =
[
raw_feature,
degree,
is_isolated
]
```

## Log

```text
[ISOLATED]

count=4263
ratio=31%
```

## Deliverable

Accuracy +1~3%

---

# Phase 8: Feature Selection Integration

## Problem

Feature Selection可能仅用于LightGBM。

GNN未使用。

## Tasks

修改：

```python
classification_to_pyg()
```

支持：

```python
selected_features
```

## Search Space

```python
64
128
256
```

## Log

```text
[FEATURE]

original_dim=767
selected_dim=128
```

## Deliverable

训练加速且精度不下降。

---

# Phase 9: Node2Vec Features

## Condition

仅当：

```text
GraphSAGE > LightGBM
```

时执行。

## Tasks

新增：

```text
features/node2vec_feature.py
```

参数：

```python
embedding_dim:
64
128
```

拼接：

```python
raw_feature + node2vec
```

## Log

```text
[NODE2VEC]

dim=128
```

## Deliverable

Accuracy > 0.78

---

# Phase 10: Hybrid Expert Model

## Objective

区别处理：

* 孤立节点
* 非孤立节点

## Architecture

```text
Shared Encoder
      |
-----------------
|               |
MLP         GraphSAGE
|               |
-----------------
      |
Classifier
```

## Routing

```python
degree == 0

→ MLP Branch

degree > 0

→ GNN Branch
```

## Log

```text
[EXPERT]

isolated=4263
graph_nodes=9489
```

---

# Leaderboard System

新增：

```text
output/leaderboard.csv
```

字段：

```python
timestamp
model
feature_dim
loss
accuracy
macro_f1
runtime
```

实时输出：

```text
================================

Rank1
GCNII16 + Focal
Acc=0.791

--------------------------------

Rank2
GraphSAGE
Acc=0.774

--------------------------------

Rank3
Hybrid
Acc=0.624

================================
```

---

# Execution Order

严格按照以下顺序执行：

1. Benchmark Framework
2. Fixed Validation Split
3. Weighted CE
4. Focal Loss
5. GraphSAGE
6. Deep GCNII
7. Label Propagation
8. Isolated Node Features
9. Feature Selection Integration
10. Node2Vec
11. Hybrid Expert

---

# Pause Agent Development

暂时停止投入：

* Bandit
* Reflection
* Planner
* Knowledge Base
* LLM Search

原因：

当前瓶颈是：

```text
Graph Modeling
+
Class Imbalance
+
Isolated Nodes
```

而非搜索策略。

优先提升模型上限，再引入自动搜索框架。
