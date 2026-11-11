
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
    / "tiger_popularity_best.pth"
)

SEMANTIC_PATH = (
    BASE_DIR / "processed_data"
    / "item_semantic_ids_unique.npy"
)

BATCH_SIZE = 8
BEAM_SIZE = 20
TOP_K = 10
PRINT_EVERY = 10000

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

print("当前设备:", device)


# ==========================================
# 1. 加载商品编码，建立前缀树
# ==========================================

print("\n========== 加载商品编码 ==========")

semantic_ids = np.load(SEMANTIC_PATH)

num_items = len(semantic_ids) - 1

children = defaultdict(set)
code_to_item = {}

for item_id in range(1, len(semantic_ids)):

    code = semantic_ids[item_id]

    if np.any(code < 0):
        raise ValueError(f"商品{item_id}没有有效编码")

    tokens = tuple(
        TOKEN_START
        + position * CODEBOOK_SIZE
        + int(value)
        for position, value in enumerate(code)
    )

    if tokens in code_to_item:
        raise ValueError("发现重复编码")

    code_to_item[tokens] = item_id

    for position in range(4):
        children[tokens[:position]].add(
            tokens[position]
        )

children = {
    prefix: np.array(sorted(values), dtype=np.int64)
    for prefix, values in children.items()
}

print("商品数量:", num_items)
print("前缀树节点数量:", len(children))


# ==========================================
# 2. 加载Epoch 25模型
# ==========================================

print("\n========== 加载模型 ==========")

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
print("模型普通Val Loss:", checkpoint["val_loss"])


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

print("训练样本:", len(train_dataset))
print("验证样本:", len(val_loader.dataset))


# ==========================================
# 4. 统计商品流行度
# ==========================================

print("\n========== 统计商品流行度 ==========")

item_counts = np.zeros(
    num_items + 1,
    dtype=np.int64
)

item_to_id = train_dataset.item_to_id

for record in train_dataset.records:

    item_id = int(
        item_to_id[str(record["parent_asin"])]
    )

    item_counts[item_id] += 1

sorted_items = sorted(
    range(1, num_items + 1),
    key=lambda x: (-item_counts[x], x)
)

head_end = int(num_items * 0.10)
middle_end = int(num_items * 0.50)

item_groups = np.zeros(
    num_items + 1,
    dtype=np.int64
)

for rank, item_id in enumerate(sorted_items):

    if rank < head_end:
        item_groups[item_id] = 0

    elif rank < middle_end:
        item_groups[item_id] = 1

    else:
        item_groups[item_id] = 2

group_names = [
    "热门商品",
    "中等热度商品",
    "长尾商品"
]

print("流行度分组完成")


# ==========================================
# 5. Beam Search
# ==========================================

@torch.inference_mode()
def beam_search_batch(encoder_input):

    memory, padding_mask = model.encode(
        encoder_input
    )

    batch_size = len(encoder_input)

    beams = [
        [((), 0.0)]
        for _ in range(batch_size)
    ]

    for step in range(4):

        prefixes = []
        scores = []
        owners = []

        for user_idx, user_beams in enumerate(beams):

            for prefix, score in user_beams:

                prefixes.append(prefix)
                scores.append(score)
                owners.append(user_idx)

        decoder_input = torch.tensor(
            [
                [BOS_ID] + list(prefix)
                for prefix in prefixes
            ],
            dtype=torch.long,
            device=device
        )

        owner_tensor = torch.tensor(
            owners,
            dtype=torch.long,
            device=device
        )

        selected_memory = memory.index_select(
            0,
            owner_tensor
        )

        selected_mask = padding_mask.index_select(
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

            user_idx = owners[row_idx]

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

            for idx in top_indices:

                token = int(allowed[idx])

                new_prefix = prefix + (token,)

                new_score = (
                    scores[row_idx]
                    + float(token_scores[idx])
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

        items = []

        for prefix, score in user_beams:

            item_id = code_to_item.get(prefix)

            if item_id is not None:
                items.append(item_id)

            if len(items) >= TOP_K:
                break

        recommendations.append(items)

    return recommendations


# ==========================================
# 6. 初始化统计
# ==========================================

group_samples = np.zeros(3, dtype=np.int64)
group_hits = np.zeros(3, dtype=np.int64)
group_ndcg = np.zeros(3, dtype=np.float64)

recommendation_counts = np.zeros(
    3,
    dtype=np.int64
)

recommended_items = set()

total_samples = 0
total_hits = 0
total_ndcg = 0.0

start_time = time.time()
last_print = 0


# ==========================================
# 7. 评估完整验证集
# ==========================================

print("\n========== 开始评估Epoch 25 ==========")

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

        group = item_groups[target]

        group_samples[group] += 1

        for item_id in predicted_items:

            recommended_items.add(item_id)

            recommendation_counts[
                item_groups[item_id]
            ] += 1

        if target in predicted_items:

            rank = predicted_items.index(target) + 1

            gain = 1.0 / math.log2(rank + 1)

            total_hits += 1
            total_ndcg += gain

            group_hits[group] += 1
            group_ndcg[group] += gain

    total_samples += len(targets)

    if total_samples - last_print >= PRINT_EVERY:

        recall = total_hits / total_samples
        ndcg = total_ndcg / total_samples

        elapsed = (
            time.time() - start_time
        ) / 60

        print(
            f"已测 {total_samples} 条 | "
            f"Recall@10: {recall:.6f} | "
            f"NDCG@10: {ndcg:.6f} | "
            f"耗时: {elapsed:.2f}分钟",
            flush=True
        )

        last_print = total_samples


# ==========================================
# 8. 输出结果
# ==========================================

print("\n========== Epoch 25整体结果 ==========")

print("验证样本:", total_samples)
print("命中数量:", total_hits)

overall_recall = total_hits / total_samples
overall_ndcg = total_ndcg / total_samples

coverage = len(recommended_items) / num_items

print("Recall@10:", f"{overall_recall:.6f}")
print("NDCG@10:", f"{overall_ndcg:.6f}")
print("商品覆盖率:", f"{coverage:.2%}")

print("\n========== 各热度分组 ==========")

total_recommendations = recommendation_counts.sum()

group_recalls = []

for group in range(3):

    count = group_samples[group]

    recall = (
        group_hits[group] / count
        if count > 0 else 0
    )

    ndcg = (
        group_ndcg[group] / count
        if count > 0 else 0
    )

    share = (
        recommendation_counts[group]
        / total_recommendations
        if total_recommendations > 0 else 0
    )

    group_recalls.append(recall)

    print(f"\n{group_names[group]}")
    print("验证样本:", count)
    print("命中数量:", group_hits[group])
    print("Recall@10:", f"{recall:.6f}")
    print("NDCG@10:", f"{ndcg:.6f}")
    print("推荐位置占比:", f"{share:.2%}")


# ==========================================
# 9. 自动与Epoch 15比较
# ==========================================

print("\n========== Epoch 15 vs Epoch 25 ==========")

baseline = {
    "Recall@10": 0.080000,
    "NDCG@10": 0.041191,
    "热门Recall": 0.138364,
    "中等Recall": 0.014918,
    "长尾Recall": 0.000436,
    "覆盖率": 0.1202
}

current = {
    "Recall@10": overall_recall,
    "NDCG@10": overall_ndcg,
    "热门Recall": group_recalls[0],
    "中等Recall": group_recalls[1],
    "长尾Recall": group_recalls[2],
    "覆盖率": coverage
}

for name in baseline:

    old = baseline[name]
    new = current[name]

    print(
        f"{name}: "
        f"原来={old:.6f} | "
        f"现在={new:.6f} | "
        f"变化={new - old:+.6f}"
    )


# ==========================================
# 10. 数据校验与结束
# ==========================================

if group_samples.sum() != total_samples:
    raise ValueError("分组样本数不一致")

if group_hits.sum() != total_hits:
    raise ValueError("分组命中数不一致")

print("\n========== 实验结束 ==========")

print(
    "总耗时:",
    round((time.time() - start_time) / 60, 2),
    "分钟"
)

print("模型Epoch:", checkpoint["epoch"])
print("全部统计校验通过")
