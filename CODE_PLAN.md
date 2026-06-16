# AFAC2026-3 Code Optimization Plan

Version: v1.0
Target Repository: afac2026-3
Priority: High
Expected Gain: +3% ~ +10% leaderboard score

---

# 1. Goals

当前系统已经具备：

* UCB Bandit
* Optuna Search
* Qwen Reflection
* Knowledge Base
* TopK Pool
* Ensemble Builder

但存在以下问题：

1. Bandit奖励失真
2. Reflection无法影响搜索空间
3. KnowledgeBase只写不读
4. 实验重复率较高
5. LLM调用成本过高
6. Ensemble尚未充分利用

本计划目标：

```text
提升实验效率
提升探索质量
减少重复实验
增加知识迁移能力
提升最终集成性能
```

---

# 2. Phase-1 修复 Bandit Reward

## 当前问题

agent.py

```python
self._best_metric = result.metric

bandit.update_compute_aware(
    metric=result.metric,
    best_metric=self._best_metric
)
```

bandit.py

```python
reward = max(
    0,
    metric - best_metric
) / runtime
```

多数情况下：

```text
metric == best_metric
reward = 0
```

导致Bandit无法学习。

---

## 修改方案

新增：

```python
prev_best_metric
```

流程：

```python
prev_best = self._best_metric

result = runner.run(...)

improvement = max(
    0,
    result.metric - prev_best
)

reward = improvement / runtime
```

---

## 进一步增强

使用归一化奖励：

```python
reward =
0.7 * improvement_ratio
+
0.3 * speed_score
```

其中：

```python
improvement_ratio =
(metric - prev_best)
/ max(prev_best,1e-6)

speed_score =
1/runtime
```

---

# 3. Phase-2 Reflection → Search Space

## 当前问题

Reflection输出：

```python
next_action
confidence
```

但不会影响实验。

---

## 新增 Reflection Executor

新增：

```python
analysis/reflection_executor.py
```

```python
class ReflectionExecutor:
```

负责修改搜索空间。

---

## 动作映射

### reduce_learning_rate

```python
search_space["lr"] *= 0.5
```

---

### increase_regularization

```python
dropout += 0.1

weight_decay *= 2
```

---

### try_new_architecture

禁止连续尝试当前最佳模型：

```python
ban_model(best_model)
```

优先探索未尝试模型。

---

### fine_tune_best

围绕最佳配置微调：

```python
TopKPool.sample_near_best()
```

---

# 4. Phase-3 Experiment Cache

## 问题

大量重复实验：

```python
lr=0.001

lr=0.0011
```

价值极低。

---

## 新增

```python
memory/config_cache.py
```

---

## 配置哈希

```python
config_hash = sha256(
    normalized_config
)
```

---

## 执行前检查

```python
if cache.exists(config_hash):
    skip()
```

---

## 相似配置过滤

由：

```python
threshold=0.99
```

改为：

```python
threshold=0.90
```

推荐：

```python
0.85
```

---

# 5. Phase-4 Knowledge Retrieval

## 当前问题

KnowledgeBase：

```python
add()
```

大量调用

```python
get_relevant_knowledge()
```

几乎未使用。

---

## 改造

Candidate生成前：

```python
knowledge =
knowledge_base.get_relevant_knowledge(
    task_type,
    model_name
)
```

---

## Warm Start

若存在历史最优：

```python
best_config
```

则：

```python
candidate_pool.append(
    perturb(best_config)
)
```

---

## Prompt注入

QwenPlanner增加：

```python
knowledge_context
```

作为额外上下文。

---

# 6. Phase-5 Meta Learning

## 新增代理模型

```python
search/meta_surrogate.py
```

训练：

```python
config
    ->
metric
```

映射。

---

## 输入

```python
model_type

hidden_dim

lr

dropout

weight_decay
```

---

## 输出

```python
predicted_metric
```

---

## 用途

真实实验前：

```python
predict()
```

筛除最差50%
配置。

---

## 目标

```text
减少真实训练次数
提升预算利用率
```

---

# 7. Phase-6 TopK Ensemble

## 当前状态

仅保存：

```python
TopK Models
```

---

## 新增

```python
Ensemble Search
```

---

## 分类任务

自动尝试：

```python
GCN + GraphSAGE

GCN + APPNP

GraphSAGE + APPNP

Top3 Voting
```

---

## 权重优化

根据验证集：

```python
weight_i
=
metric_i
/
sum(metric)
```

---

## 进一步优化

使用：

```python
Logistic Stacking
```

融合预测。

---

# 8. Phase-7 Adaptive LLM

## 当前问题

每轮调用：

```python
Reflection

Planner
```

成本高。

---

## 新策略

只有满足：

```python
连续3轮无提升
```

或者：

```python
budget < 20%
```

才调用Qwen。

---

## 普通情况

使用：

```python
Rule Planner
```

---

## 预期

减少：

```text
70%~90%
LLM调用
```

---

# 9. Phase-8 Knowledge Structure Upgrade

## 当前结构

```python
KnowledgeEntry
```

仅保存：

```python
best_config
```

---

## 升级

```python
KnowledgeEntry
```

新增：

```python
top_successes

top_failures

patterns

confidence
```

---

## 示例

```json
{
  "model":"GraphSAGE",
  "success_patterns":[
    "hidden_dim=256"
  ],
  "failure_patterns":[
    "dropout>0.7"
  ]
}
```

---

# 10. Priority

## P0（立即）

### 1

Bandit Reward修复

预计：

```text
+2% ~ +5%
```

---

### 2

Reflection Executor

预计：

```text
+1% ~ +3%
```

---

### 3

Experiment Cache

预计：

```text
+20%~40%
实验效率提升
```

---

## P1（本周）

### 4

Knowledge Retrieval

### 5

Adaptive LLM

---

## P2（冲榜）

### 6

Meta Learning

### 7

Advanced Ensemble

---

# Final Architecture

```text
Bandit
    ↓
Knowledge Retrieval
    ↓
Reflection Executor
    ↓
Candidate Generator
    ↓
Meta Predictor
    ↓
Experiment Runner
    ↓
TopK Pool
    ↓
Ensemble
    ↓
Knowledge Update
```

Expected Outcome:

```text
更少实验
更高命中率
更强泛化
更高Leaderboard分数
```
