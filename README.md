# AFAC2026 - 稀疏反馈下的自动化实验 Agent

AFAC2026 比赛方案：面向有限预算、稀疏反馈、禁止并行约束下的自动化实验智能体系统。

## 项目概述

本项目实现了一个能够在多轮交互中持续优化实验决策的 Agent 系统，面向两个金融场景图学习子任务：

- **A1 产品分类**：图节点分类（GCN/GAT/GraphSAGE），评测指标为 Accuracy
- **A2 产品推荐**：序列推荐（Popularity/ItemCF/BPR-MF/SASRec），评测指标为 NDCG@10

最终得分 = 0.5 × Accuracy + 0.5 × NDCG@10

## 系统架构

```
┌─────────────────────────────────────────────┐
│              Agent 主控 (agent.py)           │
│  ┌─────────┐ ┌──────────┐ ┌──────────────┐  │
│  │ 实验记忆 │ │ 策略规划  │ │ 预算管理器   │  │
│  │ Memory   │ │ Planner  │ │ BudgetMgr    │  │
│  └────┬────┘ └─────┬────┘ └──────┬───────┘  │
│       │            │             │           │
│  ┌────▼────────────▼─────────────▼────────┐  │
│  │         实验执行器 (ExperimentRunner)    │  │
│  │  ┌───────────┐     ┌───────────────┐   │  │
│  │  │ 分类模型池 │     │ 推荐模型池     │   │  │
│  │  │ GCN/GAT/  │     │ BPR/SASRec/   │   │  │
│  │  │ GraphSAGE │     │ Popularity    │   │  │
│  │  └───────────┘     └───────────────┘   │  │
│  └────────────────────────────────────────┘  │
│                      │                       │
│  ┌───────────────────▼───────────────────┐  │
│  │         反馈分析器 (FeedbackAnalyzer)   │  │
│  │  内部验证 + 趋势分析 → 策略建议         │  │
│  └───────────────────────────────────────┘  │
└─────────────────────────────────────────────┘
```

## 文件结构

```
afac2026-3/
├── config.py                # 全局配置（超参搜索空间、预算、路径）
├── data_loader.py           # 数据加载与预处理
├── models/
│   ├── gnn_classifier.py    # 图神经网络分类器（GCN/GAT/GraphSAGE）
│   ├── recommender.py       # 推荐模型（Popularity/ItemCF/BPR-MF/SASRec）
│   └── utils.py             # 模型工具函数
├── evaluate.py              # 本地验证评估
├── memory.py                # 实验记忆模块
├── feedback_analyzer.py     # 反馈分析与策略建议
├── planner.py               # 策略规划器
├── budget_manager.py        # 预算管理器
├── trajectory_logger.py     # 过程日志记录器
├── agent.py                 # Agent 主控核心
├── submit.py                # 提交文件生成与校验
├── run_classification.py    # 分类任务运行脚本
├── run_recommendation.py    # 推荐任务运行脚本
├── run_all.py               # 一键运行两个任务
└── requirements.txt         # 依赖清单
```

## 快速开始

### 环境要求

- Python 3.10+
- PyTorch 2.0+
- torch_geometric 2.3+

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行

```bash
# 一键运行两个任务（各用2小时预算）
python run_all.py

# 或分别运行
python run_classification.py   # 生成 output/A1.csv
python run_recommendation.py   # 生成 output/A2.csv
```

### 输出文件

- `output/A1.csv` — 产品分类提交文件
- `output/A2.csv` — 产品推荐提交文件
- `output/trajectory_classification.json` — 分类实验日志
- `output/trajectory_recommendation.json` — 推荐实验日志

## Agent 工作流程

1. **加载数据** → 读取 .npz（分类）/ CSV（推荐）
2. **规划配置** → 基于历史实验和反馈分析，选择下一个模型/超参
3. **执行实验** → 训练模型 + 内部验证集评估
4. **分析反馈** → 判断趋势（提升/停滞/下降）→ 生成优化建议
5. **记录日志** → 写入 trajectory JSON（B榜要求）
6. **循环** → 直到预算耗尽或连续多轮无提升
7. **生成提交** → 输出 A1.csv / A2.csv

## 实验策略

### 分类任务

| 阶段 | 模型 | 策略 |
|------|------|------|
| 基线 | GCN (2层, 64维) | 快速验证数据加载正确 |
| 探索 | GAT / GraphSAGE | 尝试不同架构 |
| 微调 | 最佳架构 | 调学习率、层数、dropout |

### 推荐任务

| 阶段 | 模型 | 策略 |
|------|------|------|
| 基线 | Popularity | 零成本全局热门 |
| 探索 | ItemCF / BPR-MF | 协同过滤 / 矩阵分解 |
| 序列 | SASRec | 自注意力序列模型 |
| 微调 | 最佳模型 | 调 embedding 维度、学习率 |

## 技术细节

详见 [develop.md](develop.md)。
