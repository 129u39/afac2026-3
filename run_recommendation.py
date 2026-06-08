"""运行产品推荐任务。"""

import os
import sys

import config
from agent import Agent
from submit import validate_A2
from data_loader import load_recommendation


def main():
    print("=" * 60)
    print("AFAC2026 - 产品推荐任务 (A2)")
    print("=" * 60)

    agent = Agent(
        task_type="recommendation",
        data_path=config.REC_DATA_DIR,
        output_dir=config.OUTPUT_DIR,
    )

    # 运行实验循环
    result = agent.run()
    print(f"\n最佳指标: ndcg@k={result['best_metric']:.4f}")
    print(f"最佳配置: {result['best_config']}")
    print(f"总轮次: {result['num_rounds']}")

    # 生成提交文件
    output_path = os.path.join(config.OUTPUT_DIR, "A2.csv")
    agent.generate_submission(output_path)

    # 校验
    rec_data = load_recommendation(config.REC_DATA_DIR)
    validate_A2(output_path, len(rec_data["test_df"]), rec_data["all_iid"])


if __name__ == "__main__":
    main()
