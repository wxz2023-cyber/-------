
import csv
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

OUTPUT_PATH = (
    BASE_DIR / "processed_data"
    / "beam_size_results.csv"
)

BATCH_SIZE = 8
TOP_K = 10

BEAM_SIZES = [10, 20, 50]

PRINT_EVERY = 10000

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
            f"商品{item_id}没有有效编码"
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

print("商品数量:", len(code_to_item))
print("前缀树节点数量:", len(children))


# ==========================================
# 2. 加载TIGER
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
# 3. 加载数据
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

item_popularity = np.zeros(
    num_items + 1,
    dtype=np.int64
)

item_to_id = train_dataset.item_to_id

for record in train_dataset.records:

    asin = str(record["parent_asin"])

    item_id = int(item_to_id[asin])

    item_popularity[item_id] += 1


sorted_items = sorted(
    range(1, num_items + 1),
    key=lambda item_id: (
        -item_popularity[item_id],
        item_id
    )
)

head_end = int(num_items * 0.10)
middle_end = int(num_items * 0.50)

item_group = np.zeros(
    num_items + 1,
    dtype=np.int32
)

for rank, item_id in enumerate(sorted_items):

    if rank < head_end:
        item_group[item_id] = 0

    elif rank < middle_end:
        item_group[item_id] = 1

    else:
        item_group[item_id] = 2


GROUP_NAMES = [
    "热门",
    "中等热度",
    "长尾"
]

print("商品流行度分组完成")


# ==========================================
# 5. Beam Search
# ==========================================

@torch.inference_mode()
def beam_search_batch(encoder_input, beam_size):

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
                beam_size,
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
                user_candidates[:beam_size]
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
# 6. 评估单个Beam Size
# ==========================================

def evaluate_beam_size(beam_size):

    print(
        f"\n========== Beam Size = {beam_size} ==========",
        flush=True
    )

    total_samples = 0
    total_hits = 0
    total_ndcg = 0.0

    group_samples = np.zeros(
        3,
        dtype=np.int64
    )

    group_hits = np.zeros(
        3,
        dtype=np.int64
    )

    group_ndcg = np.zeros(
        3,
        dtype=np.float64
    )

    recommendation_counts = np.zeros(
        3,
        dtype=np.int64
    )

    recommended_items = set()

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

        predictions = beam_search_batch(
            encoder_input,
            beam_size
        )

        if first_example:

            print("第一条真实商品:", int(targets[0]))

            print("第一条Top10:", predictions[0])

            first_example = False

        for predicted_items, target in zip(
            predictions,
            targets
        ):

            target = int(target)

            group = item_group[target]

            group_samples[group] += 1

            for item_id in predicted_items:

                recommended_items.add(item_id)

                recommended_group = item_group[item_id]

                recommendation_counts[
                    recommended_group
                ] += 1

            if target in predicted_items:

                rank = (
                    predicted_items.index(target)
                    + 1
                )

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
                f"Beam {beam_size} | "
                f"已分析 {total_samples} | "
                f"Recall@10: {recall:.6f} | "
                f"NDCG@10: {ndcg:.6f} | "
                f"耗时: {elapsed:.2f}分钟",
                flush=True
            )

            last_print = total_samples

    elapsed = (
        time.time() - start_time
    ) / 60

    total_recommendations = (
        recommendation_counts.sum()
    )

    result = {
        "beam_size": beam_size,
        "samples": total_samples,
        "hits": total_hits,
        "recall": total_hits / total_samples,
        "ndcg": total_ndcg / total_samples,
        "coverage": len(recommended_items) / num_items,
        "unique_items": len(recommended_items),
        "minutes": elapsed
    }

    print("\n整体 Recall@10:", f'{result["recall"]:.6f}')
    print("整体 NDCG@10:", f'{result["ndcg"]:.6f}')
    print("商品覆盖率:", f'{result["coverage"]:.2%}')

    for group in range(3):

        name = GROUP_NAMES[group]

        count = group_samples[group]

        recall = (
            group_hits[group] / count
            if count > 0
            else 0.0
        )

        ndcg = (
            group_ndcg[group] / count
            if count > 0
            else 0.0
        )

        share = (
            recommendation_counts[group]
            / total_recommendations
            if total_recommendations > 0
            else 0.0
        )

        result[f"group_{group}_recall"] = recall
        result[f"group_{group}_ndcg"] = ndcg
        result[f"group_{group}_share"] = share

        print(f"\n{name}:")
        print("验证目标数量:", count)
        print("命中数量:", group_hits[group])
        print("Recall@10:", f"{recall:.6f}")
        print("NDCG@10:", f"{ndcg:.6f}")
        print("推荐位置占比:", f"{share:.2%}")

    print("\n耗时:", round(elapsed, 2), "分钟")

    return result


# ==========================================
# 7. 测试不同Beam Size
# ==========================================

print("\n========== Beam Size消融实验 ==========")

results = []

for beam_size in BEAM_SIZES:

    result = evaluate_beam_size(
        beam_size
    )

    results.append(result)


# ==========================================
# 8. 汇总实验结果
# ==========================================

print("\n========== Beam Size对比 ==========")

for result in results:

    print(
        f"Beam={result['beam_size']} | "
        f"Recall@10={result['recall']:.6f} | "
        f"NDCG@10={result['ndcg']:.6f} | "
        f"Coverage={result['coverage']:.2%} | "
        f"LongTailRecall={result['group_2_recall']:.6f} | "
        f"Time={result['minutes']:.2f}分钟"
    )


# ==========================================
# 9. 保存结果
# ==========================================

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

with open(
    OUTPUT_PATH,
    "w",
    newline="",
    encoding="utf-8-sig"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(results[0].keys())
    )

    writer.writeheader()
    writer.writerows(results)

print("\n结果保存位置:")
print(OUTPUT_PATH)

print("\nBeam Size消融实验完成")
