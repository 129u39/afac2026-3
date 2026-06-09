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

---

# 技术文档 V2 — LLM-Agent（Qwen 驱动）

## 目标

将 V1 的规则式反思和规划升级为 LLM 驱动的决策闭环：

```text
Rule-Based Agent (V0)
    ↓
Bandit-Guided Agent (V1)
    ↓
LLM-Agent (V2) ← 当前
```

核心升级：**Qwen LLM 参与反思和最终决策**

---

## 架构

### 三层决策

```text
BanditPlanner → 选择模型家族（UCB）
    ↓
OptunaPlanner → 生成 Top-K 候选配置（TPE）
    ↓
QwenPlanner → 最终决策（LLM 选择最优候选）
```

### 反馈闭环

```text
实验 → 观察 → 反思(Qwen) → 规划(Qwen) → 实验
```

---

## 目录结构

```text
afac2026-3/

agent.py                    # 主控 Agent V2

llm/
├── client.py               # QwenClient (DashScope API)
├── schemas.py              # Pydantic 结构化输出
├── parser.py               # JSON 提取 + 校验
└── prompts.py              # 提示词模板

planner/
├── bandit.py               # UCB 多臂老虎机
├── bandit_planner.py       # Bandit 规划器
├── optuna_planner.py       # Optuna 超参搜索
└── qwen_planner.py         # Qwen 最终决策器

memory/
├── __init__.py             # ExperimentRecord + ExperimentMemory
├── vector_store.py         # 向量存储
├── retriever.py            # 实验检索
└── knowledge_base.py       # 跨任务知识库

analysis/
├── feedback.py             # 反馈分析器
└── reflection.py           # 反思 Agent (规则 + Qwen)

runner/
└── experiment_runner.py    # 统一实验运行器

budget/
└── budget_manager.py       # 预算管理

logger/
└── trajectory_logger.py    # 轨迹日志

models/
├── gnn_classifier.py       # GCN/GAT/GraphSAGE
├── recommender.py          # Popularity/ItemCF/BPR-MF/SASRec
└── recommendation/
    └── lightgcn.py         # LightGCN
```

---

## 1. LLM 集成

### llm/client.py

```python
class QwenClient:
    BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    def chat(system_prompt, user_prompt, model="qwen-plus") → str
```

- 使用 DashScope OpenAI 兼容接口
- 环境变量: `DASHSCOPE_API_KEY`
- 无 API Key 时自动 fallback 到规则模式

---

### llm/schemas.py

Pydantic 结构化输出：

```python
class ReflectionResult:
    observation: str     # 观察
    reasoning: str       # 推理
    next_action: str     # 行动建议
    confidence: float    # 置信度

class PlannerDecision:
    selected_config: dict  # 选中的配置
    reason: str            # 选择原因
    confidence: float      # 置信度

class StopDecision:
    should_stop: bool
    reason: str

class ResearchResult:
    findings: list[str]
    suggestions: list[str]
    risk_assessment: str
```

---

### llm/parser.py

```python
parse_structured(response_text, schema_class) → Pydantic | None
```

提取策略：
1. 直接解析整个响应
2. 提取 ` ```json ... ``` ` 代码块
3. 提取 `{ ... }` JSON 对象

---

### llm/prompts.py

提示词模板：

| 模板 | 用途 |
|------|------|
| `SYSTEM_PROMPT` | 系统角色设定 |
| `REFLECTION_PROMPT` | 反思生成 |
| `PLANNER_PROMPT` | 配置选择 |
| `STOP_PROMPT` | 停止判断 |
| `KNOWLEDGE_PROMPT` | 知识提取 |

---

## 2. Qwen Planner

### planner/qwen_planner.py

```python
class QwenPlanner:
    def select(candidate_configs, reflection, budget, history) → PlannerDecision
```

职责：
- 从 Bandit + Optuna 的候选配置中做最终选择
- **不直接生成参数**，而是在候选中选择最合适的
- 结合反思结果和预算状态做决策

---

## 3. 反思 Agent V2

### analysis/reflection.py

双模式：

| 模式 | 条件 | 行为 |
|------|------|------|
| Qwen | API Key 可用 | 调用 LLM 生成深度反思 |
| 规则 | API Key 不可用 | 基于规则的快速反思（fallback） |

```python
class ReflectionAgent:
    def reflect(memory, feedback, task_type, similar_experiments, budget_state) → ReflectionResult
```

---

## 4. 知识库

### memory/knowledge_base.py

跨任务经验存储：

```python
class KnowledgeBase:
    def add(entry)              # 添加知识
    def get_relevant_knowledge(task_type, model)  # 检索相关知识
    def format_for_prompt(task_type)  # 格式化为提示词
```

示例：
```json
{
  "GraphSAGE": {
    "classification": {"best_metric": 0.2285, "insights": ["3层优于2层"]},
    "recommendation": {"best_metric": 0.13, "insights": ["可迁移到图推荐"]}
  }
}
```

---

## 5. 实验运行器

### runner/experiment_runner.py

```python
class ExperimentRunner:
    def run(config) → ExperimentResult

class ExperimentResult:
    metric: float
    train_time: float
    model_name: str
    config: dict
    metrics: dict
    model: Any
    status: str
```

---

## 6. 主循环

### agent/main_agent.py

```python
while budget.should_continue():
    # 1. Bandit 选择模型
    model = bandit.select()

    # 2. Optuna 生成候选配置
    candidates = optuna.next_configs(model)

    # 3. Qwen 反思
    reflection = reflection_agent.reflect(memory, feedback)

    # 4. Qwen 最终决策
    config = qwen_planner.select(candidates, reflection)

    # 5. 执行实验
    result = runner.run(config)

    # 6. 分析反馈
    feedback = analyzer.analyze(result)

    # 7. 更新所有组件
    memory.add(result)
    bandit.update(...)
    optuna.update(...)
    knowledge_base.add(...)
    logger.log(...)
```

---

## 7. 依赖

```bash
uv pip install dashscope openai pydantic optuna
```

---

## 8. 运行

```bash
# 设置 API Key（可选，无则使用规则模式）
export DASHSCOPE_API_KEY=sk-xxx

# 运行
python run_all.py
```

---

## 9. V1 → V2 变更清单

| 文件 | 变更 |
|------|------|
| `agent.py` | 重写：集成 QwenPlanner + KnowledgeBase |
| `analysis/reflection.py` | 升级：双模式（Qwen + 规则） |
| `requirements.txt` | 新增：dashscope, openai, pydantic |

| 文件 | 新增 |
|------|------|
| `llm/client.py` | QwenClient |
| `llm/schemas.py` | Pydantic 模型 |
| `llm/parser.py` | 输出解析器 |
| `llm/prompts.py` | 提示词模板 |
| `planner/qwen_planner.py` | Qwen 规划器 |
| `memory/knowledge_base.py` | 知识库 |
| `runner/experiment_runner.py` | 统一运行器 |

---

## 10. V2 实验结果

### 分类任务

| 突破 | Accuracy | 模型 |
|------|----------|------|
| 1st | 0.2154 | GCN |
| 2nd | 0.2217 | GraphSAGE |
| 3rd | **0.2285** | GraphSAGE (高正则) |

### 推荐任务

| 突破 | NDCG@k | 模型 |
|------|--------|------|
| 1st | 0.1247 | Popularity |
| 2nd | **0.1328** | ItemCF |

### 综合得分

```
最终得分 = 0.5 × 0.2285 + 0.5 × 0.1328 = 0.1807
```

---

## 11. 已知限制与后续优化

| 问题 | 状态 | 计划 |
|------|------|------|
| SASRec 训练慢 (~74s) | 已跳过 | 优化 DataLoader |
| LightGCN 稀疏矩阵慢 | 已修复 coalesce | 需要更大规模测试 |
| 用户/物品特征未使用 | 未实现 | P2: 特征融合 |
| 模型集成融合 | V3 已实现 | Top-K Blending |
| Qwen API 调用延迟 | 已有 fallback | 优化 prompt 长度 |

---

# 技术文档 V3 — Compute-Aware AutoML Agent

## 目标

从"Agent驱动实验"升级为"Agent驱动大规模搜索"：

```text
Agent-Driven Experiment (V2)
    ↓
Compute-Aware AutoML (V3) ← 当前
```

核心目标：**充分利用120分钟预算，最大化Leaderboard分数**

---

## V3 架构

```text
                ┌─────────────┐
                │ Qwen Planner│
                └──────┬──────┘
                       │
                       ▼
          ┌─────────────────────────┐
          │ Search Space Generator  │
          └──────────┬──────────────┘
                     │
                     ▼
              ┌────────────┐
              │ Optuna TPE │
              └─────┬──────┘
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
 Successive Halving       Random Explore
        │                       │
        └───────────┬───────────┘
                    ▼
             TopK Candidate Pool
                    │
                    ▼
             Qwen Reflection
                    │
                    ▼
             Search Space Update
                    │
                    ▼
               Final Ensemble
```

---

## V3 模块 1 — TopK Candidate Pool

### search/topk_pool.py

```python
class TopKPool:
    def add(config, metric, model_name, train_time)
    def get_top_k(k=5) → list[dict]
    def get_focus_configs(n=10, focus_ratio=0.8) → list[dict]
```

- 维护 Top20 配置（按 metric 排序）
- 80% 搜索围绕 TopK，20% 随机探索
- 持久化到 `output/top_pool_{task}.json`

---

## V3 模块 2 — Successive Halving

### search/successive_halving.py

```python
class SuccessiveHalving:
    def run(initial_configs) → list[HalvingResult]
```

阶段式淘汰：

| 阶段 | Epochs | 配置数 | 保留 |
|------|--------|--------|------|
| 筛选 | 5 | 128 | Top 32 |
| 粗选 | 20 | 32 | Top 8 |
| 精选 | 100 | 8 | Top 3 |
| 最终 | 300 | 3 | Top 1 |

收益：**GPU利用率最大化**

---

## V3 模块 3 — Compute-Aware Reward

### planner/bandit.py

```python
def update_compute_aware(arm, metric, best_metric, runtime):
    reward = max(0, metric - best_metric) / runtime
```

| 实验 | 提升 | 耗时 | Reward |
|------|------|------|--------|
| A | +0.01 | 2s | 0.005 |
| B | +0.015 | 60s | 0.00025 |

**优先选择快速提升的配置**

---

## V3 模块 4 — GraphSAGE Local Search

### search/optuna_planner.py

```python
# V3: 80% 概率采样 GraphSAGE
if random.random() < 0.8:
    model_type = "GraphSAGE"
```

GraphSAGE 专用搜索空间：

| 参数 | 范围 |
|------|------|
| hidden_dim | 128, 256, 512 |
| num_layers | 2, 3, 4 |
| dropout | 0.0, 0.05, 0.1, 0.2 |
| lr | 5e-4 ~ 3e-3 |

---

## V3 模块 5 — Ensemble Builder

### models/ensemble.py

```python
class EnsembleBuilder:
    def add_model(model, config, metric)
    def build(weights="metric")
    def predict_cls(data) → np.ndarray
    def predict_rec(uid, seq, top_k) → list[str]
```

集成策略：

| 任务 | 方法 |
|------|------|
| 分类 | softmax 概率加权平均 |
| 推荐 | 分数向量加权平均 |

权重策略：
- `equal`: 等权重
- `metric`: 按指标加权
- `softmax`: softmax 归一化

---

## V3 主循环

### agent.py

```python
while budget.should_continue():
    # 1. Bandit 选择模型（Compute-Aware Reward）
    model = bandit.select()

    # 2. 生成候选配置（TopK Pool 优先）
    candidates = generate_candidates_v3(model)

    # 3. Qwen 反思
    reflection = reflection_agent.reflect(...)

    # 4. Qwen 最终决策
    config = qwen_planner.select(candidates, reflection)

    # 5. 执行实验
    result = runner.run(config)

    # 6. 更新所有组件
    memory.add(result)
    topk_pool.add(result)
    bandit.update_compute_aware(result)
    ensemble.add_model(result.model)
    logger.log(...)

# 实验结束后
ensemble.build()
generate_submission()  # 使用集成模型
```

---

## V3 依赖

```bash
uv pip install dashscope openai pydantic optuna
```

---

## V3 运行

```bash
# 设置 API Key（可选）
export DASHSCOPE_API_KEY=sk-xxx

# 运行
python run_all.py
```

---

## V3 新增文件

| 文件 | 功能 |
|------|------|
| `search/topk_pool.py` | Top-K 配置池 |
| `search/successive_halving.py` | 渐进式淘汰搜索 |
| `models/ensemble.py` | 模型集成构建器 |

## V3 修改文件

| 文件 | 变更 |
|------|------|
| `agent.py` | 集成 TopK Pool + Ensemble + Compute-Aware Reward |
| `planner/bandit.py` | 新增 `update_compute_aware()` |
| `search/optuna_planner.py` | GraphSAGE 优先采样 |

---

## V3 预期收益

| 改进 | 预期收益 |
|------|----------|
| TopK Pool | 搜索效率提升 3-5x |
| Compute-Aware Reward | 快速筛选高价值配置 |
| GraphSAGE Focus | 集中资源在最优模型 |
| Ensemble | +1%~5% 最终分数 |
| Successive Halving | GPU 利用率最大化 |

---

# 技术文档 V4 — 模型族扩展 + 特征融合

## 目标

从单一模型搜索扩展到多模型族搜索：

```text
V3: GraphSAGE Focus
    ↓
V4: GraphSAGE + APPNP + GCNII + MLP + Feature Fusion ← 当前
```

核心升级：
1. 分类：新增 APPNP（稀疏图）、GCNII（深层GNN）、MLP（基线验证）
2. 推荐：新增特征融合模型，利用 user/item 特征
3. 训练：AMP 混合精度加速

---

## V4 新增模型

### 分类模型

| 模型 | 特点 | 适用场景 |
|------|------|----------|
| **GraphSAGE** | 邻居采样聚合 | 通用，稀疏图 |
| **APPNP** | MLP + PPR 传播 | 稀疏图，弱连接图 |
| **GCNII** | 初始残差 + 恒等映射 | 深层 GNN，解决过平滑 |
| **MLP** | 纯 MLP，不使用图 | 基线验证 |

### 推荐模型

| 模型 | 特点 | 适用场景 |
|------|------|----------|
| **LightGCN** | 图卷积 | 通用 |
| **FeatureFusion** | 图嵌入 + 特征嵌入 | 有用户/物品特征 |
| **Reranker** | 重排序优化 NDCG | 召回后优化 |

---

## V4 分类搜索空间

| 模型 | 参数 | 范围 |
|------|------|------|
| GraphSAGE | hidden_dim | 128, 256, 512 |
| | num_layers | 2, 3, 4 |
| | dropout | 0.0 ~ 0.2 |
| APPNP | hidden_dim | 64, 128, 256 |
| | K (传播步数) | 5 ~ 15 |
| | alpha (重启概率) | 0.05 ~ 0.2 |
| GCNII | hidden_dim | 64, 128, 256 |
| | num_layers | 4 ~ 16 |
| | alpha | 0.05 ~ 0.2 |
| | theta | 0.3 ~ 0.7 |
| MLP | hidden_dim | 64, 128, 256 |
| | num_layers | 2, 4 |

---

## V4 推荐特征融合

### FeatureFusionModel

```python
final_emb = graph_emb + α * feature_emb
```

- `graph_emb`: LightGCN 学习的图嵌入
- `feature_emb`: UserEncoder/ItemEncoder 学习的特征嵌入
- `α`: 融合权重（默认 0.3）

### 特征编码器

```python
class UserEncoder:
    def forward(user_idx, user_features) → user_embedding

class ItemEncoder:
    def forward(item_idx, item_features) → item_embedding
```

---

## V4 训练优化

### AMP 混合精度

```python
from models.utils import enable_amp, enable_tf32

# 启用 TF32（Ampere+ GPU）
enable_tf32()

# 启用 AMP
scaler = enable_amp()
```

### 自动 Batch Size

```python
from models.utils import get_optimal_batch_size

batch_size = get_optimal_batch_size(model, device)
# 根据 GPU 显存自动选择：16GB→1024, 8GB→512, 4GB→256
```

---

## V4 新增文件

| 文件 | 功能 |
|------|------|
| `models/appnp.py` | APPNP 模型（稀疏图） |
| `models/gcnii.py` | GCNII 模型（深层 GNN） |
| `models/mlp_baseline.py` | MLP 基线模型 |
| `models/recommendation/feature_fusion.py` | 特征融合推荐模型 |
| `models/recommendation/rerank.py` | 重排序优化器 |

## V4 修改文件

| 文件 | 变更 |
|------|------|
| `models/utils.py` | 新增 AMP、TF32、自动 Batch Size |
| `planner/bandit_planner.py` | 新增 APPNP、GCNII、MLP 臂 |
| `search/optuna_planner.py` | 新增模型搜索空间 |
| `runner/experiment_runner.py` | 支持新模型训练 |

---

## V4 运行

```bash
python run_all.py
```

输出示例：
```
[Agent] 数据画像:
  特征稀疏度: 82.2%
  类别不平衡度: 17.9
  平均度: 2.0

Round 0: APPNP (适合稀疏图)
  Config: {'model_type': 'APPNP', 'hidden_dim': 128, 'K': 10, 'alpha': 0.1}
  metric: 0.2350

Round 1: GraphSAGE
  Config: {'model_type': 'GraphSAGE', 'hidden_dim': 256, 'num_layers': 3}
  metric: 0.2412
```

---

## V4 预期收益

| 改进 | 预期收益 |
|------|----------|
| APPNP | 稀疏图 +2%~5% |
| GCNII | 深层 GNN +1%~3% |
| MLP 基线 | 验证图结构有效性 |
| 特征融合 | 推荐 +3%~8% |
| AMP 加速 | 训练速度 +30%~50% |
| 跨任务知识迁移 | 已有框架 | P3: 完善迁移策略 |
