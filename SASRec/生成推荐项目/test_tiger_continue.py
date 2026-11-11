
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

SPLIT = "test"
MAX_SAMPLES = None
PRINT_EVERY = 5000

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

print("当前设备:", device)


# ==========================================
# 1. 加载商品编码
# ==========================================

print("\n========== 加载商品编码 ==========")

semantic_ids = np.load(SEMANTIC_PATH)

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
        raise ValueError("发现重复商品编码")

    code_to_item[tokens] = item_id

    for position in range(4):
        prefix = tokens[:position]
        children[prefix].add(tokens[position])

children = {
    prefix: np.array(sorted(values), dtype=np.int64)
    for prefix, values in children.items()
}

print("商品数量:", len(code_to_item))
print("前缀树节点数:", len(children))


# ==========================================
# 2. 加载Epoch 15模型
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

print("加载Epoch:", checkpoint["epoch"])
print("验证Loss:", checkpoint["val_loss"])


# ==========================================
# 3. 加载用户数据
# ==========================================

print("\n========== 加载测试数据 ==========")

train_loader, val_loader, test_loader = prepare_tiger_data(
    batch_size=BATCH_SIZE,
    max_history=20
)

if SPLIT == "validation":
    data_loader = val_loader

elif SPLIT == "test":
    data_loader = test_loader

else:
    raise ValueError("SPLIT只能是validation或test")


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
                row_idx, allowed
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
# 5. 评估Recall@10和NDCG@10
# ==========================================

print("\n========== 开始评估 ==========")

print("数据集:", SPLIT)
print("Beam Size:", BEAM_SIZE)
print("Top K:", TOP_K)

total_samples = 0
total_hits = 0
total_ndcg = 0.0

start_time = time.time()

first_example = True
last_print = 0

for batch in data_loader:

    encoder_input = batch[
        "encoder_input"
    ].to(device)

    targets = batch[
        "target_item_id"
    ].cpu().numpy()

    if MAX_SAMPLES is not None:

        remaining = MAX_SAMPLES - total_samples

        if remaining <= 0:
            break

        encoder_input = encoder_input[:remaining]
        targets = targets[:remaining]

    predictions = beam_search_batch(
        encoder_input
    )

    if first_example:

        print("\n第一条测试样本:")
        print("真实商品:", int(targets[0]))
        print("Top10推荐:", predictions[0])

        first_example = False

    for predicted_items, target in zip(
        predictions,
        targets
    ):

        if int(target) in predicted_items:

            rank = (
                predicted_items.index(int(target))
                + 1
            )

            total_hits += 1

            total_ndcg += (
                1.0 / math.log2(rank + 1)
            )

    total_samples += len(targets)

    if total_samples - last_print >= PRINT_EVERY:

        recall = total_hits / total_samples
        ndcg = total_ndcg / total_samples

        elapsed = (
            time.time() - start_time
        ) / 60

        print(
            f"已测试 {total_samples} 条 | "
            f"Recall@10: {recall:.6f} | "
            f"NDCG@10: {ndcg:.6f} | "
            f"耗时: {elapsed:.2f}分钟",
            flush=True
        )

        last_print = total_samples

    if (
        MAX_SAMPLES is not None
        and total_samples >= MAX_SAMPLES
    ):
        break


# ==========================================
# 6. 输出最终结果
# ==========================================

if total_samples == 0:
    raise ValueError("没有测试样本")

final_recall = total_hits / total_samples
final_ndcg = total_ndcg / total_samples

print("\n========== TIGER Epoch 15评估结果 ==========")

print("数据集:", SPLIT)
print("实际测试样本:", total_samples)
print("命中样本:", total_hits)

print(
    "Recall@10:",
    f"{final_recall:.6f}"
)

print(
    "NDCG@10:",
    f"{final_ndcg:.6f}"
)

print(
    "总耗时:",
    round(
        (time.time() - start_time) / 60,
        2
    ),
    "分钟"
)

print("\n评估完成")
