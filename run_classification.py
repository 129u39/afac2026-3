"""运行产品分类任务。"""

import os
import sys

import config
from agent import Agent
from submit import validate_A1


def main():
    print("=" * 60)
    print("AFAC2026 - 产品分类任务 (A1)")
    print("=" * 60)

    agent = Agent(
        task_type="classification",
        data_path=config.CLS_NPZ,
        output_dir=config.OUTPUT_DIR,
    )

    # 运行实验循环
    result = agent.run()
    print(f"\n最佳指标: val_accuracy={result['best_metric']:.4f}")
    print(f"最佳配置: {result['best_config']}")
    print(f"总轮次: {result['num_rounds']}")

    # 生成提交文件
    output_path = os.path.join(config.OUTPUT_DIR, "A1.csv")
    agent.generate_submission(output_path)

    # 校验
    from data_loader import load_classification
    data = load_classification(config.CLS_NPZ)
    validate_A1(output_path, len(data["test_idx"]), data["num_classes"])


if __name__ == "__main__":
    main()
