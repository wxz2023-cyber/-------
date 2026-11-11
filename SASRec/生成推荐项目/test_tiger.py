
import math
import time
import numpy as np
import torch

from pathlib import Path
from collections import defaultdict

from tiger import TIGER
from tiger_data import (
    prepare_tiger_data,
    PAD_ID,
    BOS_ID,
    TOKEN_START,
    CODEBOOK_SIZE
)


# ==========================================
# 1. 配置
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "checkpoints" / "tiger_best.pth"

SEMANTIC_PATH = (
    BASE_DIR / "processed_data"
    / "item_semantic_ids_unique.npy"
)

BATCH_SIZE = 8

BEAM_SIZE = 20
TOP_K = 10

# 先用验证集检查程序
# 确认正常后改成 "test"
SPLIT = "test"

# 首次只测200条
# 完整测试改为 None
MAX_SAMPLES = None

PRINT_EVERY = 100

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

print("当前设备:", device)


# ==========================================
# 2. 加载商品编码，建立前缀树
# ==========================================

print("\n========== 加载商品编码 ==========")

semantic_ids = np.load(SEMANTIC_PATH)

# prefix -> 可以继续生成的Token
children = defaultdict(set)

# 完整Token序列 -> 商品ID
code_to_item = {}

for item_id in range(1, len(semantic_ids)):

    code = semantic_ids[item_id]

    if np.any(code < 0):
        raise ValueError(
            f"商品{item_id}没有有效编码"
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

    for position in range(len(tokens)):

        prefix = tokens[:position]

        next_token = tokens[position]

        children[prefix].add(next_token)

children = {
    prefix: np.array(
        sorted(values),
        dtype=np.int64
    )
    for prefix, values in children.items()
}

print("商品总数:", len(code_to_item))
print("前缀树节点数:", len(children))


# ==========================================
# 3. 加载TIGER模型
# ==========================================

print("\n========== 加载模型 ==========")

checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu",
    weights_only=True
)

config = checkpoint["model_config"]

model = TIGER(**config).to(device)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("模型Epoch:", checkpoint["epoch"])
print("保存时Val Loss:", checkpoint["val_loss"])


# ==========================================
# 4. 加载验证集或测试集
# ==========================================

print("\n========== 加载用户数据 ==========")

train_loader, val_loader, test_loader = (
    prepare_tiger_data(
        batch_size=BATCH_SIZE,
        max_history=20
    )
)

if SPLIT == "validation":
    data_loader = val_loader

elif SPLIT == "test":
    data_loader = test_loader

else:
    raise ValueError("SPLIT只能是validation或test")


# ==========================================
# 5. Beam Search
# ==========================================

@torch.inference_mode()
def beam_search_batch(encoder_input):

    model.eval()

    memory, src_padding_mask = model.encode(
        encoder_input
    )

    batch_size = encoder_input.shape[0]

    # 每个用户维护一组候选序列
    beams = [
        [((), 0.0)]
        for _ in range(batch_size)
    ]

    # 每件商品的编码有4个Token
    for step in range(4):

        prefixes = []
        scores = []
        owner_ids = []

        for user_idx, user_beams in enumerate(beams):

            for prefix, score in user_beams:

                prefixes.append(prefix)
                scores.append(score)
                owner_ids.append(user_idx)

        # Decoder输入：BOS + 已生成的Token
        decoder_rows = [
            [BOS_ID] + list(prefix)
            for prefix in prefixes
        ]

        decoder_input = torch.tensor(
            decoder_rows,
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

        # 所有Beam同时运行Decoder
        logits = model.decode(
            decoder_input,
            selected_memory,
            selected_mask
        )

        next_logits = logits[:, -1, :]

        # 计算模型预测的对数概率
        log_probs = torch.log_softmax(
            next_logits,
            dim=-1
        ).float().cpu().numpy()

        candidates = [
            []
            for _ in range(batch_size)
        ]

        for row_idx, prefix in enumerate(prefixes):

            user_idx = owner_ids[row_idx]

            allowed_tokens = children.get(prefix)

            if allowed_tokens is None:
                continue

            token_scores = log_probs[
                row_idx,
                allowed_tokens
            ]

            # 每个Beam最多保留BEAM_SIZE个扩展
            num_keep = min(
                BEAM_SIZE,
                len(allowed_tokens)
            )

            top_indices = np.argpartition(
                token_scores,
                -num_keep
            )[-num_keep:]

            for local_idx in top_indices:

                token = int(
                    allowed_tokens[local_idx]
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

    # 4个Token已经生成
    # EOS在当前模型中是固定输出，不影响排序

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
# 6. 计算Recall@10和NDCG@10
# ==========================================

def calculate_metrics(recommendations, targets):

    total_hits = 0
    total_ndcg = 0.0

    for predicted_items, target in zip(
        recommendations,
        targets
    ):

        target = int(target)

        if target in predicted_items:

            rank = predicted_items.index(target) + 1

            total_hits += 1

            total_ndcg += (
                1.0 / math.log2(rank + 1)
            )

    count = len(targets)

    recall = total_hits / count
    ndcg = total_ndcg / count

    return recall, ndcg, total_hits


# ==========================================
# 7. 正式评估
# ==========================================

print("\n========== 开始Beam Search评估 ==========")

print("数据集:", SPLIT)
print("Beam Size:", BEAM_SIZE)
print("Top K:", TOP_K)
print("样本限制:", MAX_SAMPLES)

start_time = time.time()

total_samples = 0
total_hits = 0
total_ndcg = 0.0

example_printed = False

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

    recall, ndcg, hits = calculate_metrics(
        predictions,
        targets
    )

    count = len(targets)

    total_hits += hits
    total_ndcg += ndcg * count
    total_samples += count

    if not example_printed:

        print("\n第一条样本:")

        print(
            "真实商品:",
            int(targets[0])
        )

        print(
            "Top10预测:",
            predictions[0]
        )

        example_printed = True

    if (
        total_samples % PRINT_EVERY < count
        or (
            MAX_SAMPLES is not None
            and total_samples >= MAX_SAMPLES
        )
    ):

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
            f"已测 {total_samples} 条 | "
            f"Recall@10: {current_recall:.4f} | "
            f"NDCG@10: {current_ndcg:.4f} | "
            f"耗时: {elapsed:.2f}分钟",
            flush=True
        )

    if (
        MAX_SAMPLES is not None
        and total_samples >= MAX_SAMPLES
    ):
        break


# ==========================================
# 8. 最终结果
# ==========================================

final_recall = total_hits / total_samples
final_ndcg = total_ndcg / total_samples

print("\n========== TIGER评估结果 ==========")

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
    "耗时:",
    round(
        (time.time() - start_time) / 60,
        2
    ),
    "分钟"
)

print("\n评估完成")
