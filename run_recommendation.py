"""运行产品推荐任务 — 序列优先策略。"""

import os
import time
import numpy as np
import pandas as pd
from collections import Counter, defaultdict

import config
from data_loader import load_recommendation, parse_seq_dedup, parse_seq_counts
from submit import validate_A2


def evaluate_model(rec_data, max_samples=1000):
    """评估推荐模型（使用全量训练集）。"""
    from sklearn.model_selection import train_test_split

    train_df = rec_data["train_df"]
    train_sub, val_sub = train_test_split(train_df, test_size=0.2, random_state=42)

    if len(val_sub) > max_samples:
        val_sub = val_sub.sample(n=max_samples, random_state=42)

    # 使用全量训练集计算共现和流行度
    counter = Counter()
    for _, row in train_df.iterrows():
        counts = parse_seq_counts(row["item_seq_counts"])
        for iid, cnt in counts.items():
            counter[iid] += 1
        if pd.notna(row["target_iid"]):
            counter[row["target_iid"]] += 1

    total = sum(counter.values())
    item_scores = {iid: counter.get(iid, 0) / total for iid in rec_data["all_iid"]}

    cooccur = defaultdict(Counter)
    for _, row in train_df.iterrows():
        seq = parse_seq_dedup(str(row["item_seq_dedup"]))
        target = row["target_iid"]
        if pd.notna(target):
            for iid in seq:
                if iid != target:
                    cooccur[target][iid] += 1
                    cooccur[iid][target] += 1

    # 评估
    ndcg_list = []
    for _, row in val_sub.iterrows():
        seq = parse_seq_dedup(str(row["item_seq_dedup"]))
        target = row["target_iid"]
        seq_set = set(seq)

        # 计算分数
        scores = {}
        for iid in rec_data["all_iid"]:
            if iid in seq_set:
                continue

            score = 0.0

            # 共现分数
            if iid in cooccur:
                cooccur_score = 0.0
                for seq_iid in seq:
                    if seq_iid in cooccur[iid]:
                        cooccur_score += cooccur[iid][seq_iid]
                cooccur_score = cooccur_score / max(len(seq), 1)
                score += 0.7 * min(cooccur_score, 1.0)

            # 流行度
            score += 0.3 * item_scores.get(iid, 0.0)

            scores[iid] = score

        ranked = sorted(scores.items(), key=lambda x: -x[1])
        pred = [iid for iid, _ in ranked[:10]]

        if target in pred:
            rank = pred.index(target) + 1
            ndcg_list.append(1.0 / np.log2(rank + 1))
        else:
            ndcg_list.append(0.0)

    return np.mean(ndcg_list) if ndcg_list else 0.0


def main():
    print("=" * 60)
    print("AFAC2026 - 产品推荐任务 (A2)")
    print("=" * 60)

    total_start = time.time()

    # 加载数据
    print("\n[1] 加载数据...")
    rec_data = load_recommendation(config.REC_DATA_DIR)
    print(f"  训练集: {len(rec_data['train_df'])} 行")
    print(f"  测试集: {len(rec_data['test_df'])} 行")
    print(f"  用户数: {len(rec_data['user_df'])}")
    print(f"  物品数: {rec_data['num_items']}")

    # 评估
    print("\n[2] 评估模型...")
    ndcg = evaluate_model(rec_data)
    print(f"  NDCG@10: {ndcg:.4f}")

    # 使用全量训练集构建模型
    print("\n[3] 构建最终模型...")
    train_df = rec_data["train_df"]

    # 计算物品流行度
    counter = Counter()
    for _, row in train_df.iterrows():
        counts = parse_seq_counts(row["item_seq_counts"])
        for iid, cnt in counts.items():
            counter[iid] += 1
        if pd.notna(row["target_iid"]):
            counter[row["target_iid"]] += 1

    total = sum(counter.values())
    item_scores = {iid: counter.get(iid, 0) / total for iid in rec_data["all_iid"]}

    # 计算序列共现
    cooccur = defaultdict(Counter)
    for _, row in train_df.iterrows():
        seq = parse_seq_dedup(str(row["item_seq_dedup"]))
        target = row["target_iid"]
        if pd.notna(target):
            for iid in seq:
                if iid != target:
                    cooccur[target][iid] += 1
                    cooccur[iid][target] += 1

    # 生成提交文件
    print("\n[4] 生成 A2.csv...")
    test_df = rec_data["test_df"]
    results = []

    for _, row in test_df.iterrows():
        uid = row["uid"]
        seq = parse_seq_dedup(str(row["item_seq_dedup"]))
        seq_set = set(seq)

        # 计算分数
        scores = {}
        for iid in rec_data["all_iid"]:
            if iid in seq_set:
                continue

            score = 0.0

            # 共现分数
            if iid in cooccur:
                cooccur_score = 0.0
                for seq_iid in seq:
                    if seq_iid in cooccur[iid]:
                        cooccur_score += cooccur[iid][seq_iid]
                cooccur_score = cooccur_score / max(len(seq), 1)
                score += 0.7 * min(cooccur_score, 1.0)

            # 流行度
            score += 0.3 * item_scores.get(iid, 0.0)

            scores[iid] = score

        ranked = sorted(scores.items(), key=lambda x: -x[1])
        pred = [iid for iid, _ in ranked[:10]]
        results.append({"uid": uid, "prediction": ",".join(pred)})

    output_path = os.path.join(config.OUTPUT_DIR, "A2.csv")
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    print(f"  已保存: {output_path} ({len(df)} 行)")

    # 校验
    validate_A2(output_path, len(test_df), rec_data["all_iid"])

    # 总结
    total_time = time.time() - total_start
    print("\n" + "=" * 60)
    print("最终总结")
    print("=" * 60)
    print(f"  推荐 NDCG@10: {ndcg:.4f}")
    print(f"  总耗时: {total_time:.1f}s")
    print(f"  提交文件: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
