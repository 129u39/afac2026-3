"""运行产品推荐任务 — Qwen 指导自动化实验。"""

import os
import time
import numpy as np
import pandas as pd

import config
from data_loader import load_recommendation, parse_seq_dedup
from submit import validate_A2

# Qwen 客户端
try:
    from llm.client import QwenClient
    HAS_QWEN = True
except ImportError:
    HAS_QWEN = False

# 推荐模型
from models.recommender import RecommenderSystem
from models.hybrid_recommender import HybridRecommender, FeatureAwareRecommender


def analyze_data(rec_data):
    """分析推荐数据特征。"""
    train_df = rec_data["train_df"]
    test_df = rec_data["test_df"]
    user_df = rec_data["user_df"]
    item_df = rec_data["item_df"]

    # 序列长度
    seq_lens = train_df["item_seq_dedup"].apply(lambda x: len(parse_seq_dedup(str(x))))

    # 冷启动
    train_users = set(train_df["uid"].unique())
    test_users = set(test_df["uid"].unique())
    cold_users = test_users - train_users

    # 目标物品
    target_items = set(train_df["target_iid"].unique())

    return {
        "train_size": len(train_df),
        "test_size": len(test_df),
        "num_users": len(user_df),
        "num_items": len(item_df),
        "avg_seq_len": seq_lens.mean(),
        "cold_start_ratio": len(cold_users) / len(test_users),
        "target_item_coverage": len(target_items) / len(item_df),
    }


def qwen_analyze(qwen_client, data_profile, current_results):
    """请求 Qwen 分析实验结果并给出建议。"""
    if not qwen_client or not qwen_client.available:
        return None

    prompt = f"""## 推荐数据画像
- 训练集: {data_profile['train_size']} 行
- 测试集: {data_profile['test_size']} 行
- 用户数: {data_profile['num_users']}
- 物品数: {data_profile['num_items']}
- 平均序列长度: {data_profile['avg_seq_len']:.1f}
- 冷启动比例: {data_profile['cold_start_ratio']:.1%}
- 目标物品覆盖: {data_profile['target_item_coverage']:.1%}

## 当前实验结果
{chr(10).join(f"- {r['name']}: NDCG@10={r['ndcg']:.4f}" for r in current_results)}

## 任务
请分析当前结果，给出下一步实验建议。考虑：
1. 100%冷启动用户，哪种策略最有效？
2. 物品流行度极度倾斜，如何利用？
3. 还有哪些改进空间？

请简洁回答（不超过200字）。"""

    try:
        response = qwen_client.chat(
            "你是一个推荐系统专家，专注于冷启动和序列推荐。",
            prompt,
        )
        return response
    except Exception as e:
        return f"Qwen 分析失败: {e}"


def evaluate_model(model, rec_data, max_samples=500):
    """评估推荐模型。"""
    from sklearn.model_selection import train_test_split

    train_df = rec_data["train_df"]
    train_sub, val_sub = train_test_split(train_df, test_size=0.2, random_state=42)

    if len(val_sub) > max_samples:
        val_sub = val_sub.sample(n=max_samples, random_state=42)

    # 训练子集模型
    sub_data = {**rec_data, "train_df": train_sub}

    if hasattr(model, 'fit'):
        eval_model = model.__class__()
        if hasattr(eval_model, 'fit'):
            eval_model.fit(sub_data)
    else:
        return {"ndcg@k": 0.0, "hit@k": 0.0}

    # 评估
    ndcg_list = []
    hit_list = []

    for _, row in val_sub.iterrows():
        uid = row["uid"]
        target = row["target_iid"]
        seq = parse_seq_dedup(row["item_seq_dedup"])

        pred_list = eval_model.predict(uid, seq, top_k=10)

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
    }


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

    # 数据分析
    print("\n[2] 数据分析...")
    data_profile = analyze_data(rec_data)
    print(f"  平均序列长度: {data_profile['avg_seq_len']:.1f}")
    print(f"  冷启动比例: {data_profile['cold_start_ratio']:.1%}")
    print(f"  目标物品覆盖: {data_profile['target_item_coverage']:.1%}")

    # 初始化 Qwen
    qwen_client = None
    if HAS_QWEN:
        try:
            qwen_client = QwenClient()
            if qwen_client.available:
                print(f"\n[Qwen] 已连接 (qwen-turbo)")
            else:
                print(f"\n[Qwen] API Key 未设置，使用规则模式")
        except Exception as e:
            print(f"\n[Qwen] 初始化失败: {e}")

    # 实验结果记录
    results = []

    # ── 实验 1: Popularity 基线 ──
    print("\n" + "=" * 60)
    print("[实验 1] Popularity 基线")
    print("=" * 60)

    pop_model = RecommenderSystem(model_type="Popularity")
    pop_model.fit(rec_data)
    eval_result = evaluate_model(pop_model, rec_data)
    results.append({"name": "Popularity", "ndcg": eval_result["ndcg@k"], "model": pop_model})
    print(f"  NDCG@10: {eval_result['ndcg@k']:.4f}")

    # ── 实验 2: ItemCF ──
    print("\n" + "=" * 60)
    print("[实验 2] ItemCF")
    print("=" * 60)

    itemcf_model = RecommenderSystem(model_type="ItemCF")
    itemcf_model.fit(rec_data)
    eval_result = evaluate_model(itemcf_model, rec_data)
    results.append({"name": "ItemCF", "ndcg": eval_result["ndcg@k"], "model": itemcf_model})
    print(f"  NDCG@10: {eval_result['ndcg@k']:.4f}")

    # ── 实验 3: 混合推荐器 ──
    print("\n" + "=" * 60)
    print("[实验 3] 混合推荐器 (Popularity + Cooccurrence + Features)")
    print("=" * 60)

    hybrid_model = HybridRecommender()
    hybrid_model.fit(rec_data)
    eval_result = evaluate_model(hybrid_model, rec_data)
    results.append({"name": "Hybrid", "ndcg": eval_result["ndcg@k"], "model": hybrid_model})
    print(f"  NDCG@10: {eval_result['ndcg@k']:.4f}")

    # ── 实验 4: 特征感知推荐器 ──
    print("\n" + "=" * 60)
    print("[实验 4] 特征感知推荐器")
    print("=" * 60)

    feat_model = FeatureAwareRecommender()
    feat_model.fit(rec_data)
    eval_result = evaluate_model(feat_model, rec_data)
    results.append({"name": "FeatureAware", "ndcg": eval_result["ndcg@k"], "model": feat_model})
    print(f"  NDCG@10: {eval_result['ndcg@k']:.4f}")

    # ── Qwen 分析 ──
    print("\n" + "=" * 60)
    print("[Qwen] 分析实验结果")
    print("=" * 60)

    analysis = qwen_analyze(qwen_client, data_profile, results)
    if analysis:
        print(f"  {analysis}")
    else:
        print("  跳过（Qwen 不可用）")

    # ── 选择最佳方案 ──
    print("\n" + "=" * 60)
    print("[结果] 选择最佳方案")
    print("=" * 60)

    best = max(results, key=lambda r: r["ndcg"])
    print(f"  最佳方案: {best['name']}")
    print(f"  最佳 NDCG@10: {best['ndcg']:.4f}")

    # ── 生成提交文件 ──
    print("\n" + "=" * 60)
    print("[提交] 生成 A2.csv")
    print("=" * 60)

    test_df = rec_data["test_df"]
    best_model = best["model"]
    submission_results = []

    for _, row in test_df.iterrows():
        uid = row["uid"]
        seq = parse_seq_dedup(row["item_seq_dedup"])
        pred_list = best_model.predict(uid, seq, top_k=10)
        submission_results.append({"uid": uid, "prediction": ",".join(pred_list)})

    output_path = os.path.join(config.OUTPUT_DIR, "A2.csv")
    df = pd.DataFrame(submission_results)
    df.to_csv(output_path, index=False)
    print(f"  已保存: {output_path} ({len(df)} 行)")

    # 校验
    validate_A2(output_path, len(test_df), rec_data["all_iid"])

    # ── 总结 ──
    total_time = time.time() - total_start
    print("\n" + "=" * 60)
    print("最终总结")
    print("=" * 60)
    print(f"  推荐 NDCG@10: {best['ndcg']:.4f}")
    print(f"  总耗时: {total_time:.1f}s")
    print(f"  提交文件: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
