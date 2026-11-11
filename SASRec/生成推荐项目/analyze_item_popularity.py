
import math
import time
import numpy as np
import torch

from pathlib import Path
from collections import defaultdict

from tiger import TIGER
from tiger_data import (
    prepare_tiger_data,
    BOS_ID,
    TOKEN_START,
    CODEBOOK_SIZE
)


BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR / "checkpoints"
    / "tiger_continue_best.pth"
)

SEMANTIC_PATH = (
    BASE_DIR / "processed_data"
    / "item_semantic_ids_unique.npy"
)

BATCH_SIZE = 8
BEAM_SIZE = 20
TOP_K = 10
PRINT_EVERY = 5000

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

print("当前设备:", device)


# ==========================================
# 1. 加载商品Semantic ID
# ==========================================

print("\n========== 加载商品编码 ==========")

semantic_ids = np.load(SEMANTIC_PATH)

num_items = len(semantic_ids) - 1

children = defaultdict(set)
code_to_item = {}

for item_id in range(1, len(semantic_ids)):

    code = semantic_ids[item_id]

    if np.any(code < 0):
        raise ValueError(
            f"商品{item_id}没有有效Semantic ID"
        )

    tokens = tuple(
        TOKEN_START
        + position * CODEBOOK_SIZE
        + int(value)
        for position, value in enumerate(code)
    )

    if tokens in code_to_item:
        raise ValueError("存在重复商品编码")

    code_to_item[tokens] = item_id

    for position in range(4):

        prefix = tokens[:position]

        children[prefix].add(tokens[position])

children = {
    prefix: np.array(
        sorted(values),
        dtype=np.int64
    )
    for prefix, values in children.items()
}

print("商品数量:", num_items)
print("唯一编码数量:", len(code_to_item))


# ==========================================
# 2. 加载TIGER模型
# ==========================================

print("\n========== 加载TIGER模型 ==========")

checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu",
    weights_only=True
)

model = TIGER(
    **checkpoint["model_config"]
).to(device)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("模型Epoch:", checkpoint["epoch"])
print("验证Loss:", checkpoint["val_loss"])


# ==========================================
# 3. 加载训练集和验证集
# ==========================================

print("\n========== 加载数据 ==========")

train_loader, val_loader, test_loader = (
    prepare_tiger_data(
        batch_size=BATCH_SIZE,
        max_history=20
    )
)

train_dataset = train_loader.dataset
val_dataset = val_loader.dataset

print("训练样本:", len(train_dataset))
print("验证样本:", len(val_dataset))


# ==========================================
# 4. 统计训练集商品流行度
# ==========================================

print("\n========== 统计商品流行度 ==========")

item_popularity = np.zeros(
    num_items + 1,
    dtype=np.int64
)

item_to_id = train_dataset.item_to_id

# 仅根据训练集的目标商品统计流行度
# 不使用验证集和测试集信息

for record in train_dataset.records:

    asin = str(record["parent_asin"])

    item_id = int(item_to_id[asin])

    item_popularity[item_id] += 1

print("训练目标总数:", item_popularity.sum())

print(
    "在训练目标中出现过的商品数:",
    np.count_nonzero(item_popularity[1:])
)

print(
    "在训练目标中从未出现的商品数:",
    np.sum(item_popularity[1:] == 0)
)

print(
    "单件商品最高出现次数:",
    item_popularity.max()
)


# ==========================================
# 5. 根据商品流行度分组
# ==========================================

print("\n========== 商品流行度分组 ==========")

# 按训练目标频率从高到低排序
# 相同频率时按商品ID排序

sorted_items = sorted(
    range(1, num_items + 1),
    key=lambda item_id: (
        -item_popularity[item_id],
        item_id
    )
)

# 热门：排名前10%的商品
# 中等：排名10%到50%的商品
# 长尾：排名后50%的商品

head_end = int(num_items * 0.10)
middle_end = int(num_items * 0.50)

item_groups = np.empty(
    num_items + 1,
    dtype=object
)

item_groups[0] = "padding"

for rank, item_id in enumerate(sorted_items):

    if rank < head_end:
        group_name = "热门商品"

    elif rank < middle_end:
        group_name = "中等热度商品"

    else:
        group_name = "长尾商品"

    item_groups[item_id] = group_name


group_names = [
    "热门商品",
    "中等热度商品",
    "长尾商品"
]

groups = {}

for name in group_names:

    groups[name] = {
        "count": 0,
        "hits": 0,
        "ndcg": 0.0,
        "zero_frequency": 0
    }


for name in group_names:

    item_ids = [
        item_id
        for item_id in range(1, num_items + 1)
        if item_groups[item_id] == name
    ]

    frequencies = item_popularity[item_ids]

    print(f"\n{name}")

    print("商品数量:", len(item_ids))

    print("最高训练频率:", frequencies.max())

    print("最低训练频率:", frequencies.min())

    print(
        "平均训练频率:",
        round(float(frequencies.mean()), 2)
    )


# ==========================================
# 6. Beam Search
# ==========================================

@torch.inference_mode()
def beam_search_batch(encoder_input):

    memory, src_padding_mask = model.encode(
        encoder_input
    )

    batch_size = encoder_input.shape[0]

    beams = [
        [((), 0.0)]
        for _ in range(batch_size)
    ]

    for step in range(4):

        prefixes = []
        scores = []
        owner_ids = []

        for user_idx, user_beams in enumerate(beams):

            for prefix, score in user_beams:

                prefixes.append(prefix)
                scores.append(score)
                owner_ids.append(user_idx)

        decoder_input = torch.tensor(
            [
                [BOS_ID] + list(prefix)
                for prefix in prefixes
            ],
            dtype=torch.long,
            device=device
        )

        owner_tensor = torch.tensor(
            owner_ids,
            dtype=torch.long,
            device=device
        )

        selected_memory = memory.index_select(
            0,
            owner_tensor
        )

        selected_mask = src_padding_mask.index_select(
            0,
            owner_tensor
        )

        logits = model.decode(
            decoder_input,
            selected_memory,
            selected_mask
        )

        log_probs = torch.log_softmax(
            logits[:, -1, :],
            dim=-1
        ).float().cpu().numpy()

        candidates = [
            []
            for _ in range(batch_size)
        ]

        for row_idx, prefix in enumerate(prefixes):

            user_idx = owner_ids[row_idx]

            allowed = children.get(prefix)

            if allowed is None:
                continue

            token_scores = log_probs[
                row_idx,
                allowed
            ]

            num_keep = min(
                BEAM_SIZE,
                len(allowed)
            )

            top_indices = np.argsort(
                token_scores
            )[-num_keep:][::-1]

            for local_idx in top_indices:

                token = int(allowed[local_idx])

                new_prefix = prefix + (token,)

                new_score = (
                    scores[row_idx]
                    + float(token_scores[local_idx])
                )

                candidates[user_idx].append(
                    (new_prefix, new_score)
                )

        new_beams = []

        for user_candidates in candidates:

            user_candidates.sort(
                key=lambda x: x[1],
                reverse=True
            )

            new_beams.append(
                user_candidates[:BEAM_SIZE]
            )

        beams = new_beams

    recommendations = []

    for user_beams in beams:

        item_ids = []

        for prefix, score in user_beams:

            item_id = code_to_item.get(prefix)

            if item_id is not None:
                item_ids.append(item_id)

            if len(item_ids) >= TOP_K:
                break

        recommendations.append(item_ids)

    return recommendations


# ==========================================
# 7. 分组评估
# ==========================================

print("\n========== 开始热门与长尾分析 ==========")

total_samples = 0
total_hits = 0
total_ndcg = 0.0

start_time = time.time()
last_print = 0

# 统计推荐列表覆盖了多少种商品

recommended_item_ids = set()

recommended_group_counts = {
    name: 0
    for name in group_names
}

for batch in val_loader:

    encoder_input = batch[
        "encoder_input"
    ].to(device)

    targets = batch[
        "target_item_id"
    ].cpu().numpy()

    predictions = beam_search_batch(
        encoder_input
    )

    for predicted_items, target in zip(
        predictions,
        targets
    ):

        target = int(target)

        group_name = item_groups[target]

        groups[group_name]["count"] += 1

        if item_popularity[target] == 0:

            groups[group_name]["zero_frequency"] += 1

        for item_id in predicted_items:

            recommended_item_ids.add(item_id)

            recommended_group = item_groups[item_id]

            recommended_group_counts[
                recommended_group
            ] += 1

        if target in predicted_items:

            rank = predicted_items.index(target) + 1

            gain = 1.0 / math.log2(rank + 1)

            groups[group_name]["hits"] += 1
            groups[group_name]["ndcg"] += gain

            total_hits += 1
            total_ndcg += gain

    total_samples += len(targets)

    if total_samples - last_print >= PRINT_EVERY:

        recall = total_hits / total_samples

        ndcg = total_ndcg / total_samples

        elapsed = (
            time.time() - start_time
        ) / 60

        print(
            f"已分析 {total_samples} 条 | "
            f"Recall@10: {recall:.6f} | "
            f"NDCG@10: {ndcg:.6f} | "
            f"耗时: {elapsed:.2f}分钟",
            flush=True
        )

        last_print = total_samples


# ==========================================
# 8. 整体结果
# ==========================================

print("\n========== 整体验证结果 ==========")

print("验证样本:", total_samples)
print("命中样本:", total_hits)

print(
    "Recall@10:",
    f"{total_hits / total_samples:.6f}"
)

print(
    "NDCG@10:",
    f"{total_ndcg / total_samples:.6f}"
)


# ==========================================
# 9. 各流行度分组结果
# ==========================================

print("\n========== 不同流行度推荐效果 ==========")

for name in group_names:

    result = groups[name]

    count = result["count"]

    if count == 0:
        continue

    recall = result["hits"] / count

    ndcg = result["ndcg"] / count

    print(f"\n{name}")

    print("验证样本数:", count)

    print("命中样本数:", result["hits"])

    print(
        "Recall@10:",
        f"{recall:.6f}"
    )

    print(
        "NDCG@10:",
        f"{ndcg:.6f}"
    )

    print(
        "训练目标零频商品对应的验证样本数:",
        result["zero_frequency"]
    )


# ==========================================
# 10. 推荐商品分布
# ==========================================

print("\n========== 推荐列表流行度分布 ==========")

total_recommendations = sum(
    recommended_group_counts.values()
)

for name in group_names:

    count = recommended_group_counts[name]

    print(
        f"{name}推荐次数:",
        count
    )

    print(
        f"{name}推荐占比:",
        f"{count / total_recommendations:.2%}"
    )


print("\n========== 商品覆盖率 ==========")

print(
    "至少被推荐过一次的不同商品数:",
    len(recommended_item_ids)
)

print(
    "商品覆盖率:",
    f"{len(recommended_item_ids) / num_items:.2%}"
)


# ==========================================
# 11. 数据一致性检查
# ==========================================

group_total = sum(
    groups[name]["count"]
    for name in group_names
)

group_hits = sum(
    groups[name]["hits"]
    for name in group_names
)

if group_total != total_samples:
    raise ValueError("分组样本数不一致")

if group_hits != total_hits:
    raise ValueError("分组命中数不一致")


# ==========================================
# 12. 结束
# ==========================================

elapsed = (
    time.time() - start_time
) / 60

print("\n========== 分析完成 ==========")

print("总耗时:", round(elapsed, 2), "分钟")
print("统计校验:", group_total, group_hits)
