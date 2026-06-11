"""运行产品推荐任务 — 序列优先 + 共现优化。"""

import os
import time
import numpy as np
import pandas as pd
from collections import Counter, defaultdict

import config
from data_loader import load_recommendation, parse_seq_dedup, parse_seq_counts
from submit import validate_A2


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

    # 构建共现矩阵（全量训练集）
    print("\n[2] 构建共现矩阵...")
    cooccur = defaultdict(Counter)
    item_target_count = Counter()

    for _, row in train_df.iterrows():
        seq = parse_seq_dedup(str(row["item_seq_dedup"]))
        target = row["target_iid"]
        if pd.notna(target):
            item_target_count[target] += 1
            for iid in seq:
                if iid != target:
                    cooccur[target][iid] += 1
                    cooccur[iid][target] += 1

    total_targets = sum(item_target_count.values())
    item_pop = {iid: item_target_count.get(iid, 0) / total_targets for iid in rec_data["all_iid"]}

    print(f"  共现物品数: {len(cooccur)}")
    print(f"  目标物品数: {len(item_target_count)}")

    # 生成提交
    print("\n[3] 生成 A2.csv...")
    results = []

    for _, row in test_df.iterrows():
        uid = row["uid"]
        seq = parse_seq_dedup(str(row["item_seq_dedup"]))
        seq_set = set(seq)

        # 计算每个候选物品的分数
        scores = {}

        for iid in rec_data["all_iid"]:
            if iid in seq_set:
                continue

            score = 0.0

            # 共现分数（权重 0.8）
            if iid in cooccur:
                cooccur_score = 0.0
                for seq_iid in seq:
                    if seq_iid in cooccur[iid]:
                        cooccur_score += cooccur[iid][seq_iid]
                cooccur_score = cooccur_score / max(len(seq), 1)
                score += 0.8 * min(cooccur_score, 1.0)

            # 流行度（权重 0.2）
            score += 0.2 * item_pop.get(iid, 0.0)

            scores[iid] = score

        # 排序
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        pred = [iid for iid, _ in ranked[:10]]
        results.append({"uid": uid, "prediction": ",".join(pred)})

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
    print(f"  策略: 共现(0.8) + 流行度(0.2)")
    print(f"  总耗时: {total_time:.1f}s")
    print(f"  提交文件: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
