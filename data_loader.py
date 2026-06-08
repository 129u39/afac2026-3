"""数据加载与预处理。"""

import os

import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix
from torch_geometric.data import Data


def load_classification(path: str) -> dict:
    """加载分类任务 .npz 数据。

    返回:
        adj_csr: scipy.sparse.csr_matrix  邻接矩阵
        features: scipy.sparse.csr_matrix  节点特征
        labels: np.ndarray  标签 (测试节点为 -1)
        train_idx: np.ndarray  训练节点索引
        test_idx: np.ndarray  测试节点索引
        num_classes: int  类别数
    """
    d = np.load(path)
    adj_csr = csr_matrix(
        (d["adj_data"], d["adj_indices"], d["adj_indptr"]),
        shape=tuple(d["adj_shape"]),
    )
    features = csr_matrix(
        (d["attr_data"], d["attr_indices"], d["attr_indptr"]),
        shape=tuple(d["attr_shape"]),
    )
    labels = d["labels"]
    train_idx = d["train_idx"]
    test_idx = d["test_idx"]
    num_classes = int(labels.max()) + 1  # 0~9 → 10

    return {
        "adj_csr": adj_csr,
        "features": features,
        "labels": labels,
        "train_idx": train_idx,
        "test_idx": test_idx,
        "num_classes": num_classes,
        "num_nodes": adj_csr.shape[0],
        "num_features": features.shape[1],
    }


def classification_to_pyg(data: dict, device: torch.device = torch.device("cpu")) -> Data:
    """将分类数据转换为 PyG Data 对象。"""
    adj = data["adj_csr"]
    # 转为 COO 格式获取边索引
    adj_coo = adj.tocoo()
    edge_index = torch.tensor(np.vstack([adj_coo.row, adj_coo.col]), dtype=torch.long)
    edge_weight = torch.tensor(adj_coo.data, dtype=torch.float32)

    # 稠密化特征矩阵 (13752 x 767 不算太大)
    feat_dense = torch.tensor(data["features"].toarray(), dtype=torch.float32)

    labels = torch.tensor(data["labels"], dtype=torch.long)
    train_mask = torch.zeros(data["num_nodes"], dtype=torch.bool)
    train_mask[data["train_idx"]] = True
    test_mask = torch.zeros(data["num_nodes"], dtype=torch.bool)
    test_mask[data["test_idx"]] = True

    pyg_data = Data(
        x=feat_dense,
        edge_index=edge_index,
        edge_weight=edge_weight,
        y=labels,
        train_mask=train_mask,
        test_mask=test_mask,
        num_classes=data["num_classes"],
    )
    return pyg_data.to(device)


def load_recommendation(path_dir: str) -> dict:
    """加载推荐任务数据。

    返回:
        train_df, test_df, user_df, item_df: DataFrame
        all_iid: 所有候选物品ID列表
    """
    train_df = pd.read_csv(os.path.join(path_dir, "train.csv"))
    test_df = pd.read_csv(os.path.join(path_dir, "test.csv"))
    user_df = pd.read_csv(os.path.join(path_dir, "user.csv"))
    item_df = pd.read_csv(os.path.join(path_dir, "item.csv"))

    all_iid = item_df["iid"].tolist()

    return {
        "train_df": train_df,
        "test_df": test_df,
        "user_df": user_df,
        "item_df": item_df,
        "all_iid": all_iid,
        "num_items": len(all_iid),
    }


def parse_seq_dedup(seq_str: str) -> list[str]:
    """解析去重后的序列字符串。"""
    if pd.isna(seq_str) or str(seq_str).strip() == "":
        return []
    return [s.strip() for s in str(seq_str).split(",") if s.strip()]


def parse_seq_counts(seq_str: str) -> dict[str, int]:
    """解析物品频次字符串，返回 {iid: count}。"""
    if pd.isna(seq_str) or str(seq_str).strip() == "":
        return {}
    result = {}
    for item in str(seq_str).split(","):
        item = item.strip()
        if ":" in item:
            iid, cnt = item.split(":")
            result[iid.strip()] = int(cnt)
    return result


