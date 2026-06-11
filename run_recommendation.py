"""运行产品推荐任务 — 交叉验证 + 最优策略。"""

import os
import time
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import KFold

import config
from data_loader import load_recommendation, parse_seq_dedup
from submit import validate_A2


def evaluate_strategy(train_df):
    """5-fold 交叉验证评估策略。"""
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    ndcg_scores = []

    for train_idx, val_idx in kf.split(train_df):
        train_sub = train_df.iloc[train_idx]
        val_sub = train_df.iloc[val_idx]

        # 计算目标物品（仅训练子集）
        sub_target_counts = Counter(train_sub['target_iid'].tolist())
        sub_sorted = sorted(sub_target_counts.items(), key=lambda x: -x[1])
        sub_top = [iid for iid, _ in sub_sorted]
        sub_target_items = set(train_sub['target_iid'].unique())

        ndcg_list = []
        for _, row in val_sub.iterrows():
            seq = parse_seq_dedup(str(row['item_seq_dedup']))
            target = row['target_iid']
            seq_targets = [iid for iid in seq if iid in sub_target_items]

            pred = []
            for iid in seq_targets:
                if iid not in pred:
                    pred.append(iid)
            for iid in sub_top:
                if iid not in pred:
                    pred.append(iid)
                if len(pred) >= 10:
                    break

            if target in pred:
                rank = pred.index(target) + 1
                ndcg_list.append(1.0 / np.log2(rank + 1))
            else:
                ndcg_list.append(0.0)

        ndcg_scores.append(np.mean(ndcg_list))

    return np.mean(ndcg_scores), np.std(ndcg_scores)


def main():
    print("=" * 60)
    print("AFAC2026 - 产品推荐任务 (A2)")
    print("=" * 60)

    total_start = time.time()

    # 加载数据
    print("\n[1] 加载数据...")
    rec_data = load_recommendation(config.REC_DATA_DIR)
    train_df = rec_data["train_df"]
    test_df = rec_data["test_df"]
    print(f"  训练集: {len(train_df)} 行")
    print(f"  测试集: {len(test_df)} 行")

    # 计算目标物品流行度
    print("\n[2] 计算目标物品流行度...")
    target_counts = Counter(train_df["target_iid"].tolist())
    sorted_targets = sorted(target_counts.items(), key=lambda x: -x[1])
    top_target_items = [iid for iid, _ in sorted_targets]
    target_items = set(train_df["target_iid"].unique())

    print(f"  目标物品种类: {len(target_items)}")
    print(f"  Top-10 覆盖: {sum(target_counts.get(iid, 0) for iid in top_target_items[:10])/len(train_df)*100:.1f}%")

    # 交叉验证评估
    print("\n[3] 交叉验证评估...")
    ndcg_mean, ndcg_std = evaluate_strategy(train_df)
    print(f"  NDCG@10: {ndcg_mean:.4f} ± {ndcg_std:.4f}")

    # 生成提交
    print("\n[4] 生成 A2.csv...")
    results = []

    for _, row in test_df.iterrows():
        uid = row["uid"]
        seq = parse_seq_dedup(str(row["item_seq_dedup"]))
        seq_targets = [iid for iid in seq if iid in target_items]

        pred = []
        for iid in seq_targets:
            if iid not in pred:
                pred.append(iid)
        for iid in top_target_items:
            if iid not in pred:
                pred.append(iid)
            if len(pred) >= 10:
                break

        while len(pred) < 10:
            pred.append(top_target_items[len(pred) % len(top_target_items)])

        results.append({"uid": uid, "prediction": ",".join(pred[:10])})

    df = pd.DataFrame(results)
    output_path = os.path.join(config.OUTPUT_DIR, "A2.csv")
    df.to_csv(output_path, index=False)
    print(f"  已保存: {output_path} ({len(df)} 行)")

    # 校验
    validate_A2(output_path, len(test_df), rec_data["all_iid"])

    # 总结
    total_time = time.time() - total_start
    print("\n" + "=" * 60)
    print("最终总结")
    print("=" * 60)
    print(f"  策略: 序列中目标物品 + 目标物品流行度")
    print(f"  NDCG@10: {ndcg_mean:.4f} ± {ndcg_std:.4f}")
    print(f"  总耗时: {total_time:.1f}s")
    print(f"  提交文件: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
