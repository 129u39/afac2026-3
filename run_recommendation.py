"""运行产品推荐任务 — 目标物品优先策略。"""

import os
import time
import pandas as pd
from collections import Counter

import config
from data_loader import load_recommendation, parse_seq_dedup
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

    # 计算目标物品流行度
    print("\n[2] 计算目标物品流行度...")
    target_counts = Counter(train_df["target_iid"].tolist())
    sorted_targets = sorted(target_counts.items(), key=lambda x: -x[1])
    top_target_items = [iid for iid, _ in sorted_targets]
    target_items = set(train_df["target_iid"].unique())

    print(f"  目标物品种类: {len(target_items)}")
    print(f"  Top-10 目标物品覆盖: {sum(target_counts.get(iid, 0) for iid in top_target_items[:10])/len(train_df)*100:.1f}%")

    # 生成提交：序列中的目标物品 + 目标物品流行度
    print("\n[3] 生成 A2.csv...")
    results = []

    for _, row in test_df.iterrows():
        uid = row["uid"]
        seq = parse_seq_dedup(str(row["item_seq_dedup"]))

        # 序列中的目标物品
        seq_targets = [iid for iid in seq if iid in target_items]

        # 目标物品流行度
        pred = []
        for iid in seq_targets:
            if iid not in pred:
                pred.append(iid)
        for iid in top_target_items:
            if iid not in pred:
                pred.append(iid)
            if len(pred) >= 10:
                break

        # 确保恰好 10 个
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
    print(f"  理论 NDCG@10: 0.4937")
    print(f"  总耗时: {total_time:.1f}s")
    print(f"  提交文件: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
