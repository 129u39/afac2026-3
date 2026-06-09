"""一键运行两个任务：先分类，再推荐。"""

import os
import sys
import time

import config
from agent import Agent
from submit import validate_A1, validate_A2
from data_loader import load_classification, load_recommendation


def main():
    total_start = time.time()

    print("=" * 60)
    print("AFAC2026 - 自动化实验 Agent")
    print("=" * 60)

    # ── 任务1：产品分类 ──
    print("\n" + ">>>" + " 产品分类 " + "<<<")
    cls_agent = Agent(
        task_type="classification",
        data_path=config.CLS_NPZ,
        output_dir=config.OUTPUT_DIR,
    )
    cls_result = cls_agent.run()

    cls_output = os.path.join(config.OUTPUT_DIR, "A1.csv")
    cls_agent.generate_submission(cls_output)

    cls_data = load_classification(config.CLS_NPZ)
    validate_A1(cls_output, len(cls_data["test_idx"]), cls_data["num_classes"])

    print(f"\n分类结果: accuracy={cls_result['best_metric']:.4f}")

    # ── 任务2：产品推荐 ──
    print("\n" + ">>>" + " 产品推荐 " + "<<<")
    rec_agent = Agent(
        task_type="recommendation",
        data_path=config.REC_DATA_DIR,
        output_dir=config.OUTPUT_DIR,
    )
    rec_result = rec_agent.run()

    rec_output = os.path.join(config.OUTPUT_DIR, "A2.csv")
    rec_agent.generate_submission(rec_output)

    rec_data = load_recommendation(config.REC_DATA_DIR)
    validate_A2(rec_output, len(rec_data["test_df"]), rec_data["all_iid"])

    print(f"\n推荐结果: ndcg@10={rec_result['best_metric']:.4f}")

    # ── 汇总 ──
    total_time = time.time() - total_start
    final_score = 0.5 * cls_result["best_metric"] + 0.5 * rec_result["best_metric"]

    print("\n" + "=" * 60)
    print("最终汇总")
    print("=" * 60)
    print(f"  分类 Accuracy:  {cls_result['best_metric']:.4f}")
    print(f"  推荐 NDCG@10:   {rec_result['best_metric']:.4f}")
    print(f"  最终得分:       {final_score:.4f}")
    print(f"  总耗时:         {total_time/60:.1f} 分钟")
    print(f"  提交文件:       {cls_output}, {rec_output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
