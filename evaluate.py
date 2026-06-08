"""本地验证评估函数。"""

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from data_loader import parse_seq_dedup


def evaluate_classification(model, data, train_idx, labels, val_ratio: float = 0.2) -> dict:
    """从训练节点中划分验证集，评估分类准确率。

    返回:
        {"val_accuracy": float, "val_size": int}
    """
    # 划分 train / val
    train_sub, val_sub = train_test_split(
        train_idx, test_size=val_ratio, random_state=42, stratify=labels[train_idx]
    )

    # 创建 mask
    val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    val_mask[val_sub] = True

    model.eval()
    with torch.no_grad():
        out = model(data)
        pred = out.argmax(dim=1)
        val_pred = pred[val_mask].cpu().numpy()
        val_true = labels[val_sub]

    acc = accuracy_score(val_true, val_pred)
    return {"val_accuracy": acc, "val_size": len(val_sub)}


def evaluate_recommendation(
    model,
    rec_data: dict,
    val_ratio: float = 0.2,
    k: int = 10,
) -> dict:
    """从训练集中划分验证集，评估推荐 NDCG@K。

    返回:
        {"ndcg@k": float, "hit@k": float, "val_size": int}
    """
    train_df = rec_data["train_df"]
    train_sub, val_sub = train_test_split(
        train_df, test_size=val_ratio, random_state=42
    )

    # 在子集上重新 fit
    from models.recommender import RecommenderSystem

    sub_data = {**rec_data, "train_df": train_sub}

    eval_model = RecommenderSystem(model_type=model.model_type, **model.kwargs)
    eval_model.fit(sub_data)

    # 评估
    ndcg_list = []
    hit_list = []

    for _, row in val_sub.iterrows():
        uid = row["uid"]
        target = row["target_iid"]
        seq = parse_seq_dedup(row["item_seq_dedup"])

        pred_list = eval_model.predict(uid, seq, top_k=k)

        # NDCG@K
        if target in pred_list:
            rank = pred_list.index(target) + 1
            ndcg = 1.0 / np.log2(rank + 1)
            ndcg_list.append(ndcg)
            hit_list.append(1.0)
        else:
            ndcg_list.append(0.0)
            hit_list.append(0.0)

    return {
        "ndcg@k": np.mean(ndcg_list) if ndcg_list else 0.0,
        "hit@k": np.mean(hit_list) if hit_list else 0.0,
        "val_size": len(val_sub),
    }
