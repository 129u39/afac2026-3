"""全局配置：超参搜索空间、预算、路径。"""

import os

# ── 路径 ──────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

CLS_DATA_DIR = os.path.join(DATA_DIR, "A分类", "A分类")
CLS_NPZ = os.path.join(CLS_DATA_DIR, "A1.npz")

REC_DATA_DIR = os.path.join(DATA_DIR, "A推荐", "A推荐")
REC_TRAIN = os.path.join(REC_DATA_DIR, "train.csv")
REC_TEST = os.path.join(REC_DATA_DIR, "test.csv")
REC_USER = os.path.join(REC_DATA_DIR, "user.csv")
REC_ITEM = os.path.join(REC_DATA_DIR, "item.csv")

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 预算 ──────────────────────────────────────────────
TOTAL_BUDGET_SECONDS = 7200  # 2小时
SAFETY_MARGIN_SECONDS = 300  # 预留5分钟生成提交

# ── 分类超参搜索空间 ─────────────────────────────────
CLS_SEARCH_SPACE = {
    "model_type": ["GCN", "GAT", "GraphSAGE"],
    "hidden_dim": [64, 128, 256],
    "num_layers": [2, 3],
    "dropout": [0.0, 0.3, 0.5],
    "lr": [0.001, 0.005, 0.01],
    "weight_decay": [0.0, 5e-4],
    "epochs": [200],
    "patience": [30],
}

# ── 推荐超参搜索空间 ─────────────────────────────────
REC_SEARCH_SPACE = {
    "model_type": ["Popularity", "ItemCF", "BPR_MF", "SASRec"],
    "embedding_dim": [32, 64, 128],
    "lr": [0.001, 0.005, 0.01],
    "epochs": [50, 100],
    "batch_size": [256, 512],
    "weight_decay": [0.0, 1e-5],
}

# ── 随机种子 ─────────────────────────────────────────
SEED = 42

# ── 验证比例 ─────────────────────────────────────────
VAL_RATIO = 0.2
