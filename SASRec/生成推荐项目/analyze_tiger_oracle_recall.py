
import csv
import time
import math
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

MODEL_PATH = BASE_DIR / "checkpoints" / "tiger_popularity_best.pth"

SEMANTIC_PATH = (
    BASE_DIR / "processed_data"
    / "item_semantic_ids_unique.npy"
)

OUTPUT_PATH = (
    BASE_DIR / "processed_data"
    / "tiger_oracle_recall_results.csv"
)

BATCH_SIZE = 8
BEAM_SIZE = 50
PRINT_EVERY = 10000

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

print("当前设备:", device)


# ==========================================
# 1. 构建Semantic ID前缀树
# ==========================================

semantic_ids = np.load(SEMANTIC_PATH)

num_items = len(semantic_ids) - 1

children = defaultdict(set)
code_to_item = {}

for item_id in range(1, num_items + 1):

    code = semantic_ids[item_id]

    if np.any(code < 0):
        raise ValueError(f"商品{item_id}编码无效")

    tokens = tuple(
        TOKEN_START + position * CODEBOOK_SIZE + int(value)
        for position, value in enumerate(code)
    )

    if tokens in code_to_item:
        raise ValueError("发现重复Semantic ID")

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


# ==========================================
# 2. 加载TIGER模型
# ==========================================

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


# ==========================================
# 3. 加载训练集和验证集
# ==========================================

train_loader, val_loader, test_loader = (
    prepare_tiger_data(
        batch_size=BATCH_SIZE,
        max_history=20
    )
)

train_dataset = train_loader.dataset

item_counts = np.zeros(
    num_items + 1,
    dtype=np.int64
)

for record in train_dataset.records:

    asin = str(record["parent_asin"])
    item_id = int(train_dataset.item_to_id[asin])

    item_counts[item_id] += 1


# ==========================================
# 4. 按训练频率划分商品
# ==========================================

sorted_items = sorted(
    range(1, num_items + 1),
    key=lambda item_id: (-item_counts[item_id], item_id)
)

hot_end = int(num_items * 0.10)
middle_end = int(num_items * 0.50)

item_groups = np.zeros(
    num_items + 1,
    dtype=np.int64
)

for rank, item_id in enumerate(sorted_items):

    if rank < hot_end:
        item_groups[item_id] = 0

    elif rank < middle_end:
        item_groups[item_id] = 1

    else:
        item_groups[item_id] = 2

GROUP_NAMES = [
    "热门商品",
    "中等热度商品",
    "长尾商品"
]


# ==========================================
# 5. Beam Search生成50个候选商品
# ==========================================

@torch.inference_mode()
def beam_search_batch(encoder_input):

    memory, padding_mask = model.encode(
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

            keep = min(
                BEAM_SIZE,
                len(allowed)
            )

            best_indices = np.argsort(
                token_scores
            )[-keep:][::-1]

            for idx in best_indices:

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

    results = []

    for user_beams in beams:

        user_results = []

        for prefix, score in user_beams:

            item_id = code_to_item.get(prefix)

            if item_id is not None:
                user_results.append(item_id)

        results.append(user_results)

    return results


# ==========================================
# 6. 初始化统计
# ==========================================

group_samples = np.zeros(3, dtype=np.int64)

group_hits_10 = np.zeros(3, dtype=np.int64)
group_hits_20 = np.zeros(3, dtype=np.int64)
group_hits_50 = np.zeros(3, dtype=np.int64)

group_rank_1_10 = np.zeros(3, dtype=np.int64)
group_rank_11_20 = np.zeros(3, dtype=np.int64)
group_rank_21_50 = np.zeros(3, dtype=np.int64)

group_ndcg_10 = np.zeros(3, dtype=np.float64)

candidate_group_counts = np.zeros(
    3,
    dtype=np.int64
)

candidate_items = set()

candidate_lengths = []

total_samples = 0

start_time = time.time()

last_print = 0


# ==========================================
# 7. 完整验证集评估
# ==========================================

print("\n========== 开始Oracle Recall实验 ==========")

for batch in val_loader:

    encoder_input = batch[
        "encoder_input"
    ].to(device)

    targets = batch[
        "target_item_id"
    ].cpu().numpy()

    candidate_lists = beam_search_batch(
        encoder_input
    )

    for candidates, target in zip(
        candidate_lists,
        targets
    ):

        target = int(target)

        group = item_groups[target]

        group_samples[group] += 1

        candidate_lengths.append(
            len(candidates)
        )

        for item_id in candidates:

            candidate_items.add(item_id)

            candidate_group_counts[
                item_groups[item_id]
            ] += 1

        if target in candidates:

            rank = candidates.index(target) + 1

            if rank <= 10:

                group_hits_10[group] += 1
                group_hits_20[group] += 1
                group_hits_50[group] += 1

                group_rank_1_10[group] += 1

                group_ndcg_10[group] += (
                    1.0 / math.log2(rank + 1)
                )

            elif rank <= 20:

                group_hits_20[group] += 1
                group_hits_50[group] += 1

                group_rank_11_20[group] += 1

            elif rank <= 50:

                group_hits_50[group] += 1

                group_rank_21_50[group] += 1

    total_samples += len(targets)

    if total_samples - last_print >= PRINT_EVERY:

        recall_10 = (
            group_hits_10.sum() / total_samples
        )

        recall_50 = (
            group_hits_50.sum() / total_samples
        )

        elapsed = (
            time.time() - start_time
        ) / 60

        print(
            f"已处理 {total_samples} 条 | "
            f"Recall@10: {recall_10:.6f} | "
            f"Oracle Recall@50: {recall_50:.6f} | "
            f"耗时: {elapsed:.2f}分钟",
            flush=True
        )

        last_print = total_samples


# ==========================================
# 8. 输出整体结果
# ==========================================

total_hits_10 = group_hits_10.sum()
total_hits_20 = group_hits_20.sum()
total_hits_50 = group_hits_50.sum()

recall_10 = total_hits_10 / total_samples
recall_20 = total_hits_20 / total_samples
recall_50 = total_hits_50 / total_samples

print("\n========== 整体结果 ==========")

print("验证样本:", total_samples)

print("Recall@10:", f"{recall_10:.6f}")
print("Recall@20:", f"{recall_20:.6f}")
print("Oracle Recall@50:", f"{recall_50:.6f}")

print(
    "Top 11-50额外命中:",
    total_hits_50 - total_hits_10
)

print(
    "50候选潜在召回提升空间:",
    f"{recall_50 - recall_10:.6f}"
)

if total_hits_50 > 0:

    print(
        "候选命中后已进入Top10的比例:",
        f"{total_hits_10 / total_hits_50:.2%}"
    )

print(
    "平均候选数量:",
    f"{np.mean(candidate_lengths):.2f}"
)

print(
    "最小候选数量:",
    min(candidate_lengths)
)

print(
    "候选商品总体覆盖率:",
    f"{len(candidate_items) / num_items:.2%}"
)


# ==========================================
# 9. 输出各流行度分组
# ==========================================

print("\n========== 分组召回 ==========")

rows = []

for group in range(3):

    count = group_samples[group]

    r10 = group_hits_10[group] / count
    r20 = group_hits_20[group] / count
    r50 = group_hits_50[group] / count

    extra_hits = (
        group_hits_50[group]
        - group_hits_10[group]
    )

    print(f"\n{GROUP_NAMES[group]}")

    print("验证样本:", count)

    print("Top10命中:", group_hits_10[group])

    print("Top11-20命中:", group_rank_11_20[group])

    print("Top21-50命中:", group_rank_21_50[group])

    print("Recall@10:", f"{r10:.6f}")
    print("Recall@20:", f"{r20:.6f}")
    print("Oracle Recall@50:", f"{r50:.6f}")

    print("第11-50名额外命中:", extra_hits)

    if group_hits_50[group] > 0:

        print(
            "候选命中后进入Top10比例:",
            f"{group_hits_10[group] / group_hits_50[group]:.2%}"
        )

    rows.append({
        "group": GROUP_NAMES[group],
        "samples": int(count),
        "hits_at_10": int(group_hits_10[group]),
        "hits_at_20": int(group_hits_20[group]),
        "hits_at_50": int(group_hits_50[group]),
        "recall_at_10": float(r10),
        "recall_at_20": float(r20),
        "oracle_recall_at_50": float(r50),
        "extra_hits_11_to_50": int(extra_hits),
        "ndcg_at_10": float(group_ndcg_10[group] / count)
    })


# ==========================================
# 10. 候选商品流行度分布
# ==========================================

print("\n========== 50候选商品分布 ==========")

total_candidates = candidate_group_counts.sum()

for group in range(3):

    share = (
        candidate_group_counts[group]
        / total_candidates
    )

    print(
        f"{GROUP_NAMES[group]}: {share:.2%}"
    )


# ==========================================
# 11. 保存CSV
# ==========================================

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

rows.insert(
    0,
    {
        "group": "整体",
        "samples": total_samples,
        "hits_at_10": int(total_hits_10),
        "hits_at_20": int(total_hits_20),
        "hits_at_50": int(total_hits_50),
        "recall_at_10": float(recall_10),
        "recall_at_20": float(recall_20),
        "oracle_recall_at_50": float(recall_50),
        "extra_hits_11_to_50": int(
            total_hits_50 - total_hits_10
        ),
        "ndcg_at_10": float(
            group_ndcg_10.sum() / total_samples
        )
    }
)

with open(
    OUTPUT_PATH,
    "w",
    newline="",
    encoding="utf-8-sig"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(rows[0].keys())
    )

    writer.writeheader()
    writer.writerows(rows)


# ==========================================
# 12. 数据校验
# ==========================================

assert group_samples.sum() == total_samples

assert np.all(group_hits_10 <= group_hits_20)

assert np.all(group_hits_20 <= group_hits_50)

assert np.all(
    group_rank_1_10
    + group_rank_11_20
    + group_rank_21_50
    == group_hits_50
)

print("\n========== 实验完成 ==========")

print("结果保存:", OUTPUT_PATH)

print(
    "总耗时:",
    f"{(time.time() - start_time) / 60:.2f}分钟"
)

print("全部统计校验通过")
