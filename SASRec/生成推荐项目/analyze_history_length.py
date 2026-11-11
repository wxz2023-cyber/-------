
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
    PAD_ID,
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
        raise ValueError("发现重复商品编码")

    code_to_item[tokens] = item_id

    for position in range(4):

        prefix = tokens[:position]

        children[prefix].add(
            tokens[position]
        )

children = {
    prefix: np.array(
        sorted(values),
        dtype=np.int64
    )
    for prefix, values in children.items()
}

print("商品总数量:", len(code_to_item))
print("前缀树节点数量:", len(children))


# ==========================================
# 2. 加载训练好的TIGER
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
# 3. 加载完整验证集
# ==========================================

print("\n========== 加载验证集 ==========")

train_loader, val_loader, test_loader = (
    prepare_tiger_data(
        batch_size=BATCH_SIZE,
        max_history=20
    )
)

print("验证样本数量:", len(val_loader.dataset))


# ==========================================
# 4. Beam Search
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

                token = int(
                    allowed[local_idx]
                )

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
# 5. 历史长度分组
# ==========================================

groups = {
    "1-2": {
        "count": 0,
        "hits": 0,
        "ndcg": 0.0
    },
    "3-5": {
        "count": 0,
        "hits": 0,
        "ndcg": 0.0
    },
    "6-10": {
        "count": 0,
        "hits": 0,
        "ndcg": 0.0
    },
    "11-20": {
        "count": 0,
        "hits": 0,
        "ndcg": 0.0
    }
}


def get_group(length):

    if length <= 2:
        return "1-2"

    if length <= 5:
        return "3-5"

    if length <= 10:
        return "6-10"

    return "11-20"


# ==========================================
# 6. 评估不同历史长度
# ==========================================

print("\n========== 开始历史长度分析 ==========")

total_samples = 0
total_hits = 0
total_ndcg = 0.0

start_time = time.time()
last_print = 0

first_example = True

for batch in val_loader:

    encoder_input = batch[
        "encoder_input"
    ].to(device)

    targets = batch[
        "target_item_id"
    ].cpu().numpy()

    # 一个商品由4个Semantic Token构成
    history_lengths = (
        (encoder_input != PAD_ID)
        .sum(dim=1)
        .cpu()
        .numpy() // 4
    )

    predictions = beam_search_batch(
        encoder_input
    )

    if first_example:

        print("\n第一条样本:")
        print("历史长度:", int(history_lengths[0]))
        print("真实商品:", int(targets[0]))
        print("Top10推荐:", predictions[0])

        first_example = False

    for predicted_items, target, length in zip(
        predictions,
        targets,
        history_lengths
    ):

        group_name = get_group(
            int(length)
        )

        groups[group_name]["count"] += 1

        target = int(target)

        if target in predicted_items:

            rank = (
                predicted_items.index(target)
                + 1
            )

            gain = (
                1.0 / math.log2(rank + 1)
            )

            total_hits += 1
            total_ndcg += gain

            groups[group_name]["hits"] += 1
            groups[group_name]["ndcg"] += gain

    total_samples += len(targets)

    if total_samples - last_print >= PRINT_EVERY:

        current_recall = (
            total_hits / total_samples
        )

        current_ndcg = (
            total_ndcg / total_samples
        )

        elapsed = (
            time.time() - start_time
        ) / 60

        print(
            f"已分析 {total_samples} 条 | "
            f"Recall@10: {current_recall:.6f} | "
            f"NDCG@10: {current_ndcg:.6f} | "
            f"耗时: {elapsed:.2f}分钟",
            flush=True
        )

        last_print = total_samples


# ==========================================
# 7. 输出整体指标
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
# 8. 输出各历史长度的指标
# ==========================================

print("\n========== 不同历史长度的推荐效果 ==========")

for name, result in groups.items():

    count = result["count"]

    if count == 0:
        continue

    recall = (
        result["hits"] / count
    )

    ndcg = (
        result["ndcg"] / count
    )

    print(f"\n历史长度: {name}")

    print("样本数量:", count)
    print("命中数量:", result["hits"])

    print(
        "Recall@10:",
        f"{recall:.6f}"
    )

    print(
        "NDCG@10:",
        f"{ndcg:.6f}"
    )


# ==========================================
# 9. 检查统计结果
# ==========================================

group_total = sum(
    result["count"]
    for result in groups.values()
)

if group_total != total_samples:
    raise ValueError("分组样本数与总数不一致")

group_hits = sum(
    result["hits"]
    for result in groups.values()
)

if group_hits != total_hits:
    raise ValueError("分组命中数与总命中数不一致")


# ==========================================
# 10. 结束
# ==========================================

elapsed = (
    time.time() - start_time
) / 60

print("\n========== 分析完成 ==========")

print("总耗时:", round(elapsed, 2), "分钟")

print("分组样本数校验:", group_total)
print("分组命中数校验:", group_hits)

print("\n历史长度分析结束")
