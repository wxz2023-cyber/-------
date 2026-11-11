
import numpy as np

from pathlib import Path
from collections import defaultdict


BASE_DIR = Path(__file__).resolve().parent

INPUT_PATH = (
    BASE_DIR / "processed_data"
    / "item_semantic_ids_finetuned.npy"
)

OUTPUT_PATH = (
    BASE_DIR / "processed_data"
    / "item_semantic_ids_unique.npy"
)

semantic_ids = np.load(INPUT_PATH)

num_items = len(semantic_ids) - 1

unique_ids = np.full(
    (len(semantic_ids), 4),
    -1,
    dtype=np.int32
)

groups = defaultdict(list)
missing_items = []

for item_id in range(1, len(semantic_ids)):

    code = semantic_ids[item_id]

    if np.any(code < 0):
        missing_items.append(item_id)
        continue

    groups[tuple(code.tolist())].append(item_id)


print("========== 处理重复编码 ==========")

max_group_size = 0

for code, item_ids in groups.items():

    max_group_size = max(
        max_group_size,
        len(item_ids)
    )

    for suffix, item_id in enumerate(item_ids):

        unique_ids[item_id, :3] = code

        unique_ids[item_id, 3] = suffix


print("有效语义编码组数量:", len(groups))
print("最大重复组:", max_group_size)


# 对缺少文本的商品使用特殊编码
# 256不属于原先0到255的码本编号范围

for suffix, item_id in enumerate(missing_items):

    unique_ids[item_id] = [
        256, 256, 256, suffix
    ]


print("缺少文本的商品数:", len(missing_items))


# ==========================================
# 验证编码唯一性
# ==========================================

real_item_codes = unique_ids[1:]

unique_count = len(
    np.unique(real_item_codes, axis=0)
)

print("\n========== 编码检查 ==========")

print("实际商品数量:", num_items)
print("不同完整编码数量:", unique_count)

print(
    "是否全部唯一:",
    unique_count == num_items
)

if unique_count != num_items:
    raise ValueError("仍然存在重复编码")

if np.any(real_item_codes < 0):
    raise ValueError("存在未编码的真实商品")

np.save(
    OUTPUT_PATH,
    unique_ids
)

print("\n前10件商品的最终编码:")

for item_id in range(1, 11):

    print(
        f"商品{item_id}:",
        unique_ids[item_id].tolist()
    )

print("\n最终编码矩阵形状:", unique_ids.shape)

print("第四个编号最大值:", unique_ids[1:, 3].max())

print("保存路径:", OUTPUT_PATH)
