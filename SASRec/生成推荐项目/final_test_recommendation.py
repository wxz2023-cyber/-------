
import csv
import math
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


# ==========================================
# 1. 实验配置
# ==========================================

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
    / "final_test_results.csv"
)

BATCH_SIZE = 8
MAX_HISTORY = 20

BEAM_SIZE = 50
CANDIDATE_K = 50
FINAL_K = 10

TIGER_QUOTA = 40
SEMANTIC_QUOTA = 10

RRF_K = 60
ALPHAS = [0.8, 0.9]

PRINT_EVERY = 10000

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

start_time = time.time()

print("当前设备:", device)


# ==========================================
# 2. 加载商品向量和Semantic ID
# ==========================================

print("\n========== 加载商品数据 ==========")

semantic_ids = np.load(SEMANTIC_PATH)

embeddings = np.load(
    EMBEDDING_PATH
).astype(np.float32)

if len(semantic_ids) != len(embeddings):
    raise ValueError("商品向量和Semantic ID数量不一致")

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

embedding_tensor = F.normalize(
    torch.tensor(
        embeddings,
        dtype=torch.float32,
        device=device
    ),
    dim=1
)

catalog_embeddings = embedding_tensor[1:]

valid_embedding_tensor = torch.tensor(
    valid_embedding,
    dtype=torch.bool,
    device=device
)

print("商品数量:", num_items)


# ==========================================
# 3. 建立商品映射与Trie
# ==========================================

children = defaultdict(set)

code_to_item = {}

for item_id in range(1, num_items + 1):

    code = semantic_ids[item_id]

    if np.any(code < 0):
        raise ValueError("存在无效Semantic ID")

    tokens = tuple(
        TOKEN_START
        + position * CODEBOOK_SIZE
        + int(value)
        for position, value in enumerate(code)
    )

    if tokens in code_to_item:
        raise ValueError("Semantic ID存在重复")

    code_to_item[tokens] = item_id

    for position in range(4):

        children[
            tokens[:position]
        ].add(tokens[position])

children = {
    prefix: np.array(
        sorted(values),
        dtype=np.int64
    )
    for prefix, values in children.items()
}


# ==========================================
# 4. 加载TIGER模型
# ==========================================

print("\n========== 加载TIGER ==========")

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
# 5. 加载训练集和测试集
# ==========================================

print("\n========== 加载测试数据 ==========")

train_loader, val_loader, test_loader = (
    prepare_tiger_data(
        batch_size=BATCH_SIZE,
        max_history=MAX_HISTORY
    )
)

train_dataset = train_loader.dataset

print("训练样本:", len(train_dataset))
print("测试样本:", len(test_loader.dataset))


# ==========================================
# 6. 用训练集统计商品热度
# ==========================================

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

sorted_items = sorted(
    range(1, num_items + 1),
    key=lambda x: (-item_counts[x], x)
)

hot_end = int(num_items * 0.10)

middle_end = int(num_items * 0.50)

item_groups = np.zeros(
    num_items + 1,
    dtype=np.int64
)

for rank, item_id in enumerate(sorted_items):

    if rank < hot_end:
        group = 0

    elif rank < middle_end:
        group = 1

    else:
        group = 2

    item_groups[item_id] = group

GROUP_NAMES = ["热门", "中等", "长尾"]


# ==========================================
# 7. TIGER Beam Search
# ==========================================

@torch.inference_mode()
def tiger_search(encoder_input):

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

        logits = model.decode(
            decoder_input,
            memory.index_select(0, owner_tensor),
            padding_mask.index_select(0, owner_tensor)
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

        for user_idx in range(batch_size):

            candidates[user_idx].sort(
                key=lambda x: x[1],
                reverse=True
            )

            beams[user_idx] = candidates[
                user_idx
            ][:BEAM_SIZE]

    results = []

    for user_beams in beams:

        items = []

        for prefix, score in user_beams:

            item_id = code_to_item.get(prefix)

            if item_id is not None:
                items.append(item_id)

        results.append(
            items[:CANDIDATE_K]
        )

    return results


# ==========================================
# 8. 还原用户历史商品
# ==========================================

def decode_history(encoder_input):

    data = encoder_input.cpu().numpy()

    data = data.reshape(
        len(data),
        MAX_HISTORY,
        4
    )

    history = np.zeros(
        (len(data), MAX_HISTORY),
        dtype=np.int64
    )

    for i in range(len(data)):

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

            history[i, j] = item_id

    return history


# ==========================================
# 9. Last-1语义召回
# ==========================================

@torch.inference_mode()
def semantic_search(encoder_input):

    history = decode_history(encoder_input)

    history_ids = torch.tensor(
        history,
        dtype=torch.long,
        device=device
    )

    valid_history = valid_embedding_tensor[
        history_ids
    ]

    positions = torch.arange(
        MAX_HISTORY,
        device=device
    ).unsqueeze(0).expand(
        len(history_ids),
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

    scores[:, ~valid_embedding_tensor[1:]] = -1e9

    top_items = torch.topk(
        scores,
        k=CANDIDATE_K,
        dim=1
    ).indices + 1

    results = []

    for i, row in enumerate(
        top_items.cpu().numpy()
    ):

        if not bool(has_history[i].item()):
            results.append([])

        else:
            results.append(
                [int(x) for x in row]
            )

    return results


# ==========================================
# 10. 构造40+10联合候选
# ==========================================

def build_candidates(tiger_items, semantic_items):

    selected = []
    seen = set()

    def add(item):

        if item not in seen:
            selected.append(item)
            seen.add(item)

    for item in tiger_items[:TIGER_QUOTA]:
        add(item)

    for item in semantic_items[:SEMANTIC_QUOTA]:
        add(item)

    t = TIGER_QUOTA
    s = SEMANTIC_QUOTA

    while len(selected) < CANDIDATE_K:

        progressed = False

        if t < len(tiger_items):

            add(tiger_items[t])
            t += 1
            progressed = True

        if len(selected) >= CANDIDATE_K:
            break

        if s < len(semantic_items):

            add(semantic_items[s])
            s += 1
            progressed = True

        if not progressed:
            break

    return selected[:CANDIDATE_K]


# ==========================================
# 11. RRF排序
# ==========================================

def rrf_rerank(
    candidates,
    tiger_items,
    semantic_items,
    alpha
):

    tiger_rank = {
        item: rank
        for rank, item in enumerate(
            tiger_items,
            start=1
        )
    }

    semantic_rank = {
        item: rank
        for rank, item in enumerate(
            semantic_items,
            start=1
        )
    }

    scores = []

    for item in candidates:

        score = 0.0

        if item in tiger_rank:

            score += (
                alpha
                / (RRF_K + tiger_rank[item])
            )

        if item in semantic_rank:

            score += (
                (1 - alpha)
                / (RRF_K + semantic_rank[item])
            )

        scores.append(
            (item, score)
        )

    scores.sort(
        key=lambda x: (-x[1], x[0])
    )

    return [
        item
        for item, score in scores[:FINAL_K]
    ]


# ==========================================
# 12. 初始化评估器
# ==========================================

def new_stats():

    return {
        "samples": 0,
        "hits": 0,
        "ndcg": 0.0,
        "group_samples": np.zeros(
            3,
            dtype=np.int64
        ),
        "group_hits": np.zeros(
            3,
            dtype=np.int64
        ),
        "group_ndcg": np.zeros(
            3,
            dtype=np.float64
        ),
        "unique_items": set()
    }


stats = {
    "TIGER-Top10": new_stats(),
    "RRF-0.8": new_stats(),
    "RRF-0.9": new_stats()
}


def update_stats(result, predictions, target):

    group = int(item_groups[target])

    result["samples"] += 1

    result["group_samples"][group] += 1

    result["unique_items"].update(predictions)

    if target in predictions:

        rank = predictions.index(target) + 1

        gain = 1.0 / math.log2(rank + 1)

        result["hits"] += 1
        result["ndcg"] += gain

        result["group_hits"][group] += 1
        result["group_ndcg"][group] += gain


# ==========================================
# 13. 完整测试集评估
# ==========================================

print("\n========== 开始最终测试 ==========")

processed = 0
last_print = 0

for batch in test_loader:

    encoder_input = batch[
        "encoder_input"
    ].to(device)

    targets = batch[
        "target_item_id"
    ].cpu().numpy()

    tiger_lists = tiger_search(
        encoder_input
    )

    semantic_lists = semantic_search(
        batch["encoder_input"]
    )

    for tiger_items, semantic_items, target in zip(
        tiger_lists,
        semantic_lists,
        targets
    ):

        target = int(target)

        update_stats(
            stats["TIGER-Top10"],
            tiger_items[:FINAL_K],
            target
        )

        candidates = build_candidates(
            tiger_items,
            semantic_items
        )

        for alpha in ALPHAS:

            predictions = rrf_rerank(
                candidates,
                tiger_items,
                semantic_items,
                alpha
            )

            update_stats(
                stats[f"RRF-{alpha}"],
                predictions,
                target
            )

    processed += len(targets)

    if processed - last_print >= PRINT_EVERY:

        print(
            f"已测试 {processed} 条 | "
            f"耗时 {(time.time()-start_time)/60:.2f}分钟",
            flush=True
        )

        last_print = processed


# ==========================================
# 14. 汇总各模型指标
# ==========================================

print("\n========== 最终测试结果 ==========")

rows = []

# 之前独立测试得到的SASRec结果
# 仅作为历史基准引用，不是本脚本重新运行

rows.append({
    "model": "SASRec-Historical-Test",
    "recall_at_10": 0.079547,
    "ndcg_at_10": 0.041964,
    "hot_recall": "",
    "middle_recall": "",
    "tail_recall": "",
    "coverage": "",
    "samples": 94762,
    "source": "previous_test"
})

print("\nSASRec（历史测试结果）")
print("Recall@10: 0.079547")
print("NDCG@10: 0.041964")


for name, result in stats.items():

    n = result["samples"]

    recall = result["hits"] / n
    ndcg = result["ndcg"] / n

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

    print(f"\n{name}")

    print("Recall@10:", f"{recall:.6f}")
    print("NDCG@10:", f"{ndcg:.6f}")

    for group, group_name in enumerate(GROUP_NAMES):

        print(
            f"{group_name} Recall@10:",
            f"{group_recalls[group]:.6f}"
        )

    print(
        "商品覆盖率:",
        f"{coverage:.2%}"
    )

    rows.append({
        "model": name,
        "recall_at_10": recall,
        "ndcg_at_10": ndcg,
        "hot_recall": group_recalls[0],
        "middle_recall": group_recalls[1],
        "tail_recall": group_recalls[2],
        "coverage": coverage,
        "samples": n,
        "source": "current_test"
    })


# ==========================================
# 15. 保存CSV
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
# 16. 结果校验
# ==========================================

expected_samples = len(test_loader.dataset)

for name, result in stats.items():

    assert result["samples"] == expected_samples, (
        f"{name}测试样本数量异常"
    )

assert expected_samples == 94762, (
    "当前测试集规模与历史基准不同，"
    "请检查数据划分后再比较"
)

print("\n========== 测试完成 ==========")

print("测试集样本数:", expected_samples)

print("结果保存:", OUTPUT_PATH)

print(
    "总耗时:",
    f"{(time.time()-start_time)/60:.2f}分钟"
)

print("全部数据校验通过")
