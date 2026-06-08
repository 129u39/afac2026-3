"""提交文件生成与格式校验。"""

import os
import pandas as pd


def validate_A1(path: str, num_test_nodes: int, num_classes: int) -> bool:
    """校验 A1.csv 提交文件格式。

    Args:
        path: 文件路径
        num_test_nodes: 测试节点数
        num_classes: 类别数

    返回:
        True 如果格式正确
    """
    df = pd.read_csv(path)

    # 检查列名
    if list(df.columns) != ["test_idx", "label"]:
        print(f"[ERROR] 列名应为 ['test_idx', 'label']，实际为 {list(df.columns)}")
        return False

    # 检查行数
    if len(df) != num_test_nodes:
        print(f"[ERROR] 行数应为 {num_test_nodes}，实际为 {len(df)}")
        return False

    # 检查标签范围
    invalid = df[(df["label"] < 0) | (df["label"] >= num_classes)]
    if len(invalid) > 0:
        print(f"[ERROR] 存在非法标签: {invalid['label'].unique()}")
        return False

    print(f"[OK] A1.csv 格式正确: {len(df)} 行, 标签范围 [0, {num_classes-1}]")
    return True


def validate_A2(path: str, num_test_users: int, all_iid: list[str]) -> bool:
    """校验 A2.csv 提交文件格式。

    Args:
        path: 文件路径
        num_test_users: 测试用户数
        all_iid: 所有候选物品ID列表

    返回:
        True 如果格式正确
    """
    df = pd.read_csv(path)

    # 检查列名
    if list(df.columns) != ["uid", "prediction"]:
        print(f"[ERROR] 列名应为 ['uid', 'prediction']，实际为 {list(df.columns)}")
        return False

    # 检查行数
    if len(df) != num_test_users:
        print(f"[ERROR] 行数应为 {num_test_users}，实际为 {len(df)}")
        return False

    # 检查每个用户的推荐列表
    iid_set = set(all_iid)
    errors = 0
    for _, row in df.iterrows():
        preds = str(row["prediction"]).split(",")
        if len(preds) != 10:
            errors += 1
            continue
        for p in preds:
            if p.strip() not in iid_set:
                errors += 1
                break

    if errors > 0:
        print(f"[WARN] {errors} 行存在格式问题（非10个物品或非法ID）")

    print(f"[OK] A2.csv 格式检查完成: {len(df)} 行, {errors} 行有问题")
    return errors == 0


def generate_A1(predictions, test_idx, output_path: str):
    """生成 A1.csv。"""
    df = pd.DataFrame({"test_idx": test_idx, "label": predictions})
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


def generate_A2(results: list[dict], output_path: str):
    """生成 A2.csv。

    Args:
        results: [{"uid": str, "prediction": str}, ...]
    """
    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path
