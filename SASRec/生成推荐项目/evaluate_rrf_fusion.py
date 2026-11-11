
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
    / "rrf_fusion_results.csv"
)

BATCH_SIZE = 8
MAX_HISTORY = 20

BEAM_SIZE = 50
CANDIDATE_K = 50
FINAL_K = 10

TIGER_QUOTA = 40
SEMANTIC_QUOTA = 10

RRF_K = 60

ALPHAS = [
    0.5,
    0.7,
    0.8,
    0.9,
    0.95
]

PRINT_EVERY = 10000

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

start_time = time.time()

print("当前设备:", device)


# ==========================================
# 1. 加载商品Embedding和Semantic ID
# ==========================================

print("\n========== 加载商品数据 ==========")

semantic_ids = np.load(
    SEMANTIC_PATH
)

embeddings = np.load(
    EMBEDDING_PATH
).astype(np.float32)

if len(semantic_ids) != len(embeddings):
    raise ValueError(
        "Semantic ID和Embedding商品数量不一致"
    )

num_items = len(semantic_ids) - 1

embeddings = np.nan_to_num(
    embeddings,
    nan=0.0,
    posinf=0.0,
    neginf=0.0
)

valid_embedding = (
    np.linalg.norm(
        embeddings,
        axis=1
    ) > 1e-8
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

catalog_embeddings = (
    embedding_tensor[1:]
)

valid_embedding_tensor = torch.tensor(
    valid_embedding,
    dtype=torch.bool,
    device=device
)

print("商品数量:", num_items)

print(
    "有效文本向量商品数:",
    int(valid_embedding[1:].sum())
)


# ==========================================
# 2. 构建Semantic ID前缀树
# ==========================================

print("\n========== 建立Semantic ID映射 ==========")

children = defaultdict(set)
code_to_item = {}

for item_id in range(
    1,
    num_items + 1
):

    code = semantic_ids[item_id]

    if np.any(code < 0):
        raise ValueError(
            f"商品{item_id}的Semantic ID无效"
        )

    tokens = tuple(
        TOKEN_START
        + position * CODEBOOK_SIZE
        + int(value)
        for position, value in enumerate(code)
    )

    if tokens in code_to_item:
        raise ValueError(
            "存在重复Semantic ID"
        )

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

print(
    "商品映射数量:",
    len(code_to_item)
)


# ==========================================
# 3. 加载训练好的TIGER模型
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

print(
    "加载模型Epoch:",
    checkpoint["epoch"]
)


# ==========================================
# 4. 加载训练集、验证集
# ==========================================

print("\n========== 加载数据 ==========")

train_loader, val_loader, test_loader = (
    prepare_tiger_data(
        batch_size=BATCH_SIZE,
        max_history=MAX_HISTORY
    )
)

train_dataset = train_loader.dataset

print(
    "训练样本数量:",
    len(train_dataset)
)

print(
    "验证样本数量:",
    len(val_loader.dataset)
)


# ==========================================
# 5. 统计商品流行度
# ==========================================

item_counts = np.zeros(
    num_items + 1,
    dtype=np.int64
)

for record in train_dataset.records:

    asin = str(
        record["parent_asin"]
    )

    item_id = int(
        train_dataset.item_to_id[asin]
    )

    item_counts[item_id] += 1

sorted_items = sorted(
    range(1, num_items + 1),
    key=lambda i: (
        -item_counts[i],
        i
    )
)

hot_end = int(
    num_items * 0.10
)

middle_end = int(
    num_items * 0.50
)

item_groups = np.zeros(
    num_items + 1,
    dtype=np.int64
)

for rank, item_id in enumerate(
    sorted_items
):

    if rank < hot_end:
        group = 0

    elif rank < middle_end:
        group = 1

    else:
        group = 2

    item_groups[item_id] = group

GROUP_NAMES = [
    "热门",
    "中等",
    "长尾"
]

print("商品流行度分组完成")


# ==========================================
# 6. TIGER Beam Search
# ==========================================

@torch.inference_mode()
def tiger_search(encoder_input):

    memory, padding_mask = model.encode(
        encoder_input
    )

    batch_size = (
        encoder_input.shape[0]
    )

    beams = [
        [((), 0.0)]
        for _ in range(batch_size)
    ]

    for step in range(4):

        prefixes = []
        scores = []
        owners = []

        for user_idx, user_beams in enumerate(
            beams
        ):

            for prefix, score in user_beams:

                prefixes.append(
                    prefix
                )

                scores.append(
                    score
                )

                owners.append(
                    user_idx
                )

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

        selected_memory = (
            memory.index_select(
                0,
                owner_tensor
            )
        )

        selected_mask = (
            padding_mask.index_select(
                0,
                owner_tensor
            )
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

        for row_idx, prefix in enumerate(
            prefixes
        ):

            user_idx = owners[row_idx]

            allowed = children.get(
                prefix
            )

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

                token = int(
                    allowed[idx]
                )

                new_prefix = (
                    prefix + (token,)
                )

                new_score = (
                    scores[row_idx]
                    + float(
                        token_scores[idx]
                    )
                )

                candidates[
                    user_idx
                ].append(
                    (
                        new_prefix,
                        new_score
                    )
                )

        new_beams = []

        for user_candidates in candidates:

            user_candidates.sort(
                key=lambda x: x[1],
                reverse=True
            )

            new_beams.append(
                user_candidates[
                    :BEAM_SIZE
                ]
            )

        beams = new_beams

    results = []

    for user_beams in beams:

        items = []

        for prefix, score in user_beams:

            item_id = (
                code_to_item.get(
                    prefix
                )
            )

            if item_id is not None:
                items.append(
                    item_id
                )

        results.append(
            items[:CANDIDATE_K]
        )

    return results


# ==========================================
# 7. 将历史Semantic ID还原成商品ID
# ==========================================

def decode_history(encoder_input):

    data = (
        encoder_input.cpu()
        .numpy()
    )

    batch_size = len(data)

    data = data.reshape(
        batch_size,
        MAX_HISTORY,
        4
    )

    history_ids = np.zeros(
        (
            batch_size,
            MAX_HISTORY
        ),
        dtype=np.int64
    )

    for i in range(batch_size):

        for j in range(MAX_HISTORY):

            tokens = tuple(
                int(x)
                for x in data[i, j]
            )

            if tokens == (
                0, 0, 0, 0
            ):
                continue

            item_id = (
                code_to_item.get(
                    tokens
                )
            )

            if item_id is None:
                raise ValueError(
                    f"无法识别历史商品编码: {tokens}"
                )

            history_ids[i, j] = (
                item_id
            )

    return history_ids


# ==========================================
# 8. Last-1语义召回
# ==========================================

@torch.inference_mode()
def semantic_search(encoder_input):

    history_np = decode_history(
        encoder_input
    )

    history_ids = torch.tensor(
        history_np,
        dtype=torch.long,
        device=device
    )

    batch_size = (
        history_ids.shape[0]
    )

    valid_history = (
        valid_embedding_tensor[
            history_ids
        ]
    )

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

    has_history = (
        last_positions >= 0
    )

    safe_positions = (
        last_positions.clamp_min(0)
    )

    last_items = (
        history_ids.gather(
            1,
            safe_positions.unsqueeze(1)
        ).squeeze(1)
    )

    user_vectors = (
        embedding_tensor[
            last_items
        ]
    )

    scores = (
        user_vectors
        @ catalog_embeddings.T
    )

    invalid_catalog = (
        ~valid_embedding_tensor[1:]
    )

    scores[:, invalid_catalog] = (
        -1e9
    )

    top_indices = torch.topk(
        scores,
        k=CANDIDATE_K,
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
# 9. 构造40+10联合候选
# ==========================================

def build_candidates(
    tiger_items,
    semantic_items
):

    selected = []
    seen = set()

    def add(item):

        if item not in seen:

            selected.append(
                item
            )

            seen.add(
                item
            )

    for item in tiger_items[
        :TIGER_QUOTA
    ]:

        add(item)

    for item in semantic_items[
        :SEMANTIC_QUOTA
    ]:

        add(item)

    tiger_index = TIGER_QUOTA
    semantic_index = SEMANTIC_QUOTA

    while len(selected) < CANDIDATE_K:

        progressed = False

        if tiger_index < len(tiger_items):

            add(
                tiger_items[
                    tiger_index
                ]
            )

            tiger_index += 1

            progressed = True

        if len(selected) >= CANDIDATE_K:
            break

        if semantic_index < len(
            semantic_items
        ):

            add(
                semantic_items[
                    semantic_index
                ]
            )

            semantic_index += 1

            progressed = True

        if not progressed:
            break

    return selected[:CANDIDATE_K]


# ==========================================
# 10. RRF融合排序
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
                / (
                    RRF_K
                    + tiger_rank[item]
                )
            )

        if item in semantic_rank:

            score += (
                (1 - alpha)
                / (
                    RRF_K
                    + semantic_rank[item]
                )
            )

        scores.append(
            (
                item,
                score
            )
        )

    scores.sort(
        key=lambda x: (
            -x[1],
            x[0]
        )
    )

    return [
        item
        for item, score in scores[
            :FINAL_K
        ]
    ]


# ==========================================
# 11. 初始化评估统计
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
    "TIGER-Top10": new_stats()
}

for alpha in ALPHAS:

    stats[
        f"RRF-{alpha}"
    ] = new_stats()


def update_stats(
    result,
    predictions,
    target
):

    group = (
        item_groups[target]
    )

    result["samples"] += 1

    result["group_samples"][
        group
    ] += 1

    result["unique_items"].update(
        predictions
    )

    if target in predictions:

        rank = (
            predictions.index(
                target
            ) + 1
        )

        gain = (
            1.0
            / math.log2(
                rank + 1
            )
        )

        result["hits"] += 1

        result["ndcg"] += gain

        result["group_hits"][
            group
        ] += 1

        result["group_ndcg"][
            group
        ] += gain


# ==========================================
# 12. 完整验证集评估
# ==========================================

print(
    "\n========== 开始RRF融合实验 =========="
)

processed = 0
last_print = 0

for batch in val_loader:

    encoder_input = (
        batch[
            "encoder_input"
        ].to(device)
    )

    targets = (
        batch[
            "target_item_id"
        ].cpu().numpy()
    )

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

        target = int(
            target
        )

        update_stats(
            stats[
                "TIGER-Top10"
            ],
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
                stats[
                    f"RRF-{alpha}"
                ],
                predictions,
                target
            )

    processed += len(
        targets
    )

    if processed - last_print >= PRINT_EVERY:

        print(
            f"已处理 {processed} 条 | "
            f"耗时 "
            f"{(time.time() - start_time) / 60:.2f}分钟",
            flush=True
        )

        last_print = processed


# ==========================================
# 13. 输出结果
# ==========================================

print(
    "\n========== RRF实验结果 =========="
)

rows = []

for name, result in stats.items():

    n = result["samples"]

    recall = (
        result["hits"] / n
    )

    ndcg = (
        result["ndcg"] / n
    )

    group_recalls = []

    for group in range(3):

        count = (
            result["group_samples"][
                group
            ]
        )

        group_recall = (
            result["group_hits"][
                group
            ] / count
            if count > 0
            else 0.0
        )

        group_recalls.append(
            group_recall
        )

    coverage = (
        len(
            result["unique_items"]
        ) / num_items
    )

    print(f"\n{name}")

    print(
        "Recall@10:",
        f"{recall:.6f}"
    )

    print(
        "NDCG@10:",
        f"{ndcg:.6f}"
    )

    for group, group_name in enumerate(
        GROUP_NAMES
    ):

        print(
            f"{group_name} Recall@10:",
            f"{group_recalls[group]:.6f}"
        )

    print(
        "商品覆盖率:",
        f"{coverage:.2%}"
    )

    rows.append(
        {
            "method": name,
            "recall_at_10": recall,
            "ndcg_at_10": ndcg,
            "hot_recall_at_10": group_recalls[0],
            "middle_recall_at_10": group_recalls[1],
            "tail_recall_at_10": group_recalls[2],
            "coverage": coverage
        }
    )


# ==========================================
# 14. 保存CSV
# ==========================================

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

with open(
    OUTPUT_PATH,
    "w",
    encoding="utf-8-sig",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(
            rows[0].keys()
        )
    )

    writer.writeheader()

    writer.writerows(
        rows
    )


# ==========================================
# 15. 检查结果
# ==========================================

expected_samples = len(
    val_loader.dataset
)

for name, result in stats.items():

    assert (
        result["samples"]
        == expected_samples
    ), f"{name}样本数量异常"


print(
    "\n========== 实验完成 =========="
)

print(
    "结果保存到:",
    OUTPUT_PATH
)

print(
    "总耗时:",
    f"{(time.time() - start_time) / 60:.2f}分钟"
)
