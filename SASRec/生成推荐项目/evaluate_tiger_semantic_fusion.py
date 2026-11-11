
import csv
import time
import numpy as np
import torch
import torch.nn.functional as F

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

EMBEDDING_PATH = (
    BASE_DIR / "processed_data"
    / "item_embeddings.npy"
)

SEMANTIC_PATH = (
    BASE_DIR / "processed_data"
    / "item_semantic_ids_unique.npy"
)

OUTPUT_PATH = (
    BASE_DIR / "processed_data"
    / "tiger_semantic_fusion_results.csv"
)

BATCH_SIZE = 8
MAX_HISTORY = 20
BEAM_SIZE = 50
RECALL_K = 50
PRINT_EVERY = 10000

FUSION_CONFIGS = [
    (50, 0),
    (40, 10),
    (30, 20),
    (20, 30),
    (10, 40),
    (0, 50)
]

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

print("设备:", device)

start_time = time.time()


# ==========================================
# 1. 加载商品Semantic ID和Embedding
# ==========================================

semantic_ids = np.load(SEMANTIC_PATH)

embeddings = np.load(
    EMBEDDING_PATH
).astype(np.float32)

if len(semantic_ids) != len(embeddings):
    raise ValueError("Semantic ID与Embedding数量不一致")

num_items = len(semantic_ids) - 1

embeddings = np.nan_to_num(
    embeddings,
    nan=0.0,
    posinf=0.0,
    neginf=0.0
)

valid_embedding = (
    np.linalg.norm(embeddings, axis=1) > 1e-8
)

valid_embedding[0] = False

embedding_tensor = torch.tensor(
    embeddings,
    dtype=torch.float32,
    device=device
)

embedding_tensor = F.normalize(
    embedding_tensor,
    p=2,
    dim=1
)

valid_embedding_tensor = torch.tensor(
    valid_embedding,
    dtype=torch.bool,
    device=device
)

catalog_embeddings = embedding_tensor[1:]

print("商品数量:", num_items)


# ==========================================
# 2. 建立Semantic ID映射和前缀树
# ==========================================

children = defaultdict(set)
code_to_item = {}

for item_id in range(1, num_items + 1):

    code = semantic_ids[item_id]

    if np.any(code < 0):
        raise ValueError(f"商品{item_id}编码无效")

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

        children[tokens[:position]].add(
            tokens[position]
        )

children = {
    prefix: np.array(sorted(values), dtype=np.int64)
    for prefix, values in children.items()
}


# ==========================================
# 3. 加载TIGER模型
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

print("TIGER Epoch:", checkpoint["epoch"])


# ==========================================
# 4. 加载训练集与验证集
# ==========================================

train_loader, val_loader, test_loader = (
    prepare_tiger_data(
        batch_size=BATCH_SIZE,
        max_history=MAX_HISTORY
    )
)

train_dataset = train_loader.dataset

item_counts = np.zeros(
    num_items + 1,
    dtype=np.int64
)

for record in train_dataset.records:

    asin = str(record["parent_asin"])

    item_id = int(
        train_dataset.item_to_id[asin]
    )

    item_counts[item_id] += 1


# ==========================================
# 5. 按训练频率划分商品组
# ==========================================

sorted_items = sorted(
    range(1, num_items + 1),
    key=lambda i: (-item_counts[i], i)
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

GROUP_NAMES = ["热门", "中等", "长尾"]


# ==========================================
# 6. TIGER Beam Search
# ==========================================

@torch.inference_mode()
def tiger_beam_search(encoder_input):

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
            logits[:, -1, :].float(),
            dim=-1
        ).cpu().numpy()

        candidates = [
            []
            for _ in range(batch_size)
        ]

        for row_idx, prefix in enumerate(prefixes):

            user_idx = owners[row_idx]

            allowed = children[prefix]

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

        items = []

        for prefix, score in user_beams:

            item_id = code_to_item.get(prefix)

            if item_id is not None:
                items.append(item_id)

        results.append(items[:RECALL_K])

    return results


# ==========================================
# 7. 将历史Semantic Token还原为商品ID
# ==========================================

def decode_history(encoder_input):

    data = encoder_input.cpu().numpy()

    batch_size = len(data)

    data = data.reshape(
        batch_size,
        MAX_HISTORY,
        4
    )

    history_ids = np.zeros(
        (batch_size, MAX_HISTORY),
        dtype=np.int64
    )

    for i in range(batch_size):

        for j in range(MAX_HISTORY):

            tokens = tuple(
                int(x)
                for x in data[i, j]
            )

            if tokens == (0, 0, 0, 0):
                continue

            item_id = code_to_item.get(tokens)

            if item_id is None:
                raise ValueError(
                    f"历史商品编码不存在: {tokens}"
                )

            history_ids[i, j] = item_id

    return history_ids


# ==========================================
# 8. Last-1语义召回
# ==========================================

@torch.inference_mode()
def semantic_recall(encoder_input):

    history_np = decode_history(
        encoder_input
    )

    history_ids = torch.tensor(
        history_np,
        dtype=torch.long,
        device=device
    )

    batch_size = history_ids.shape[0]

    # 找到每条历史中最近一件有有效文本向量的商品
    valid_history = valid_embedding_tensor[
        history_ids
    ]

    positions = torch.arange(
        MAX_HISTORY,
        device=device
    ).unsqueeze(0).expand(
        batch_size,
        -1
    )

    last_positions = torch.where(
        valid_history,
        positions,
        -1
    ).max(dim=1).values

    has_history = last_positions >= 0

    safe_positions = last_positions.clamp_min(0)

    last_items = history_ids.gather(
        1,
        safe_positions.unsqueeze(1)
    ).squeeze(1)

    user_vectors = embedding_tensor[
        last_items
    ]

    scores = (
        user_vectors
        @ catalog_embeddings.T
    )

    invalid_catalog = (
        ~valid_embedding_tensor[1:]
    )

    scores[:, invalid_catalog] = -1e9

    top_indices = torch.topk(
        scores,
        k=RECALL_K,
        dim=1
    ).indices

    top_items = top_indices + 1

    top_items = torch.where(
        has_history.unsqueeze(1),
        top_items,
        torch.zeros_like(top_items)
    )

    results = []

    for row in top_items.cpu().numpy():

        results.append(
            [
                int(item)
                for item in row
                if item > 0
            ]
        )

    return results


# ==========================================
# 9. 固定50件商品的候选融合
# ==========================================

def fuse_fixed_budget(
    tiger_items,
    semantic_items,
    tiger_quota,
    semantic_quota
):

    selected = []
    seen = set()

    def add(item):

        if item not in seen:
            selected.append(item)
            seen.add(item)

    # 先分配各路预定名额
    for item in tiger_items[:tiger_quota]:
        add(item)

    for item in semantic_items[:semantic_quota]:
        add(item)

    # 如果发生重合，交替使用两路剩余候选补齐
    tiger_index = tiger_quota
    semantic_index = semantic_quota

    while len(selected) < RECALL_K:

        progressed = False

        if tiger_quota > 0 and tiger_index < len(tiger_items):

            add(tiger_items[tiger_index])

            tiger_index += 1

            progressed = True

        if len(selected) >= RECALL_K:
            break

        if semantic_quota > 0 and semantic_index < len(semantic_items):

            add(semantic_items[semantic_index])

            semantic_index += 1

            progressed = True

        if not progressed:
            break

    return selected[:RECALL_K]


# ==========================================
# 10. 统计器
# ==========================================

def new_stats():

    return {
        "samples": 0,
        "hits": 0,
        "group_samples": np.zeros(3, dtype=np.int64),
        "group_hits": np.zeros(3, dtype=np.int64),
        "unique_items": set(),
        "total_candidates": 0
    }


stats = {
    f"TIGER-{a}_Semantic-{b}": new_stats()
    for a, b in FUSION_CONFIGS
}

union_stats = new_stats()

complement = {
    "both": np.zeros(3, dtype=np.int64),
    "tiger_only": np.zeros(3, dtype=np.int64),
    "semantic_only": np.zeros(3, dtype=np.int64),
    "neither": np.zeros(3, dtype=np.int64)
}


def update_stats(result, candidates, target, group):

    result["samples"] += 1
    result["group_samples"][group] += 1

    result["unique_items"].update(candidates)

    result["total_candidates"] += len(candidates)

    if target in candidates:

        result["hits"] += 1
        result["group_hits"][group] += 1


# ==========================================
# 11. 完整验证集评估
# ==========================================

print("\n========== 开始多路召回实验 ==========")

processed = 0
last_print = 0

for batch in val_loader:

    encoder_input = batch[
        "encoder_input"
    ].to(device)

    targets = batch[
        "target_item_id"
    ].cpu().numpy()

    tiger_lists = tiger_beam_search(
        encoder_input
    )

    semantic_lists = semantic_recall(
        batch["encoder_input"]
    )

    for tiger_items, semantic_items, target in zip(
        tiger_lists,
        semantic_lists,
        targets
    ):

        target = int(target)
        group = int(item_groups[target])

        tiger_hit = target in tiger_items
        semantic_hit = target in semantic_items

        if tiger_hit and semantic_hit:
            complement["both"][group] += 1

        elif tiger_hit:
            complement["tiger_only"][group] += 1

        elif semantic_hit:
            complement["semantic_only"][group] += 1

        else:
            complement["neither"][group] += 1

        # 联合候选，最多100件
        union_items = list(
            dict.fromkeys(
                tiger_items + semantic_items
            )
        )

        update_stats(
            union_stats,
            union_items,
            target,
            group
        )

        # 固定50件候选的融合实验
        for tiger_quota, semantic_quota in FUSION_CONFIGS:

            name = (
                f"TIGER-{tiger_quota}"
                f"_Semantic-{semantic_quota}"
            )

            candidates = fuse_fixed_budget(
                tiger_items,
                semantic_items,
                tiger_quota,
                semantic_quota
            )

            update_stats(
                stats[name],
                candidates,
                target,
                group
            )

    processed += len(targets)

    if processed - last_print >= PRINT_EVERY:

        print(
            f"已处理 {processed} 条 | "
            f"耗时 {(time.time() - start_time) / 60:.2f}分钟",
            flush=True
        )

        last_print = processed


# ==========================================
# 12. 输出命中互补性
# ==========================================

print("\n========== 两路召回互补性 ==========")

for group in range(3):

    print(f"\n{GROUP_NAMES[group]}")

    for name, values in complement.items():

        print(
            name,
            ":",
            int(values[group])
        )


print("\n========== 固定50候选对比 ==========")

rows = []


def summarize(name, result):

    n = result["samples"]

    recall = result["hits"] / n

    group_recalls = []

    for group in range(3):

        count = result["group_samples"][group]

        value = (
            result["group_hits"][group] / count
            if count else 0.0
        )

        group_recalls.append(value)

    coverage = (
        len(result["unique_items"])
        / num_items
    )

    average_candidates = (
        result["total_candidates"] / n
    )

    print(f"\n{name}")
    print("Recall:", f"{recall:.6f}")
    print("热门Recall:", f"{group_recalls[0]:.6f}")
    print("中等Recall:", f"{group_recalls[1]:.6f}")
    print("长尾Recall:", f"{group_recalls[2]:.6f}")
    print("商品覆盖率:", f"{coverage:.2%}")
    print("平均候选数:", f"{average_candidates:.2f}")

    rows.append({
        "method": name,
        "recall": recall,
        "hot_recall": group_recalls[0],
        "middle_recall": group_recalls[1],
        "tail_recall": group_recalls[2],
        "coverage": coverage,
        "average_candidates": average_candidates
    })


for name, result in stats.items():

    summarize(name, result)


print("\n========== 最多100件联合候选 ==========")

summarize(
    "TIGER50+Semantic50-Union",
    union_stats
)


# ==========================================
# 13. 保存CSV
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
        fieldnames=list(rows[0].keys())
    )

    writer.writeheader()
    writer.writerows(rows)


# ==========================================
# 14. 校验实验结果
# ==========================================

total_complement = sum(
    values.sum()
    for values in complement.values()
)

assert total_complement == len(val_loader.dataset)

assert stats[
    "TIGER-50_Semantic-0"
]["samples"] == len(val_loader.dataset)

assert stats[
    "TIGER-0_Semantic-50"
]["samples"] == len(val_loader.dataset)

print("\n========== 实验完成 ==========")

print("结果已保存:", OUTPUT_PATH)

print(
    "总耗时:",
    f"{(time.time()-start_time)/60:.2f}分钟"
)
