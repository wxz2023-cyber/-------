
import csv
import time
import numpy as np
import torch
import torch.nn.functional as F

from pathlib import Path
from tiger_data import prepare_tiger_data


BASE_DIR = Path(__file__).resolve().parent

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
    / "semantic_recall_results.csv"
)

BATCH_SIZE = 64
TOP_K = 100
MAX_HISTORY = 20
PRINT_EVERY = 10000

METHODS = {
    "Last-1": 1,
    "Mean-5": 5,
    "Mean-10": 10
}

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

print("当前设备:", device)

start_time = time.time()


# ==========================================
# 1. 加载商品Embedding
# ==========================================

print("\n========== 加载商品向量 ==========")

item_embeddings = np.load(
    EMBEDDING_PATH
).astype(np.float32)

semantic_ids = np.load(
    SEMANTIC_PATH
)

if len(item_embeddings) != len(semantic_ids):
    raise ValueError("Embedding与Semantic ID商品数量不一致")

num_items = len(item_embeddings) - 1

print("商品数量:", num_items)
print("向量维度:", item_embeddings.shape[1])

item_embeddings = np.nan_to_num(
    item_embeddings,
    nan=0.0,
    posinf=0.0,
    neginf=0.0
)

valid_embeddings = (
    np.linalg.norm(item_embeddings, axis=1) > 1e-8
)

valid_embeddings[0] = False

embedding_tensor = torch.tensor(
    item_embeddings,
    dtype=torch.float32,
    device=device
)

embedding_tensor = F.normalize(
    embedding_tensor,
    p=2,
    dim=1
)

catalog_embeddings = embedding_tensor[1:]

catalog_valid = torch.tensor(
    valid_embeddings[1:],
    dtype=torch.bool,
    device=device
)

print(
    "具有有效Embedding的商品数:",
    int(valid_embeddings[1:].sum())
)


# ==========================================
# 2. 建立Semantic Token到商品ID的映射
# ==========================================

print("\n========== 建立商品映射 ==========")

TOKEN_START = 3
CODEBOOK_SIZE = 257

token_to_item = {}

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

    if tokens in token_to_item:
        raise ValueError("Semantic Token编码发生重复")

    token_to_item[tokens] = item_id

print("成功建立商品映射:", len(token_to_item))


# ==========================================
# 3. 加载训练集和验证集
# ==========================================

print("\n========== 加载数据 ==========")

train_loader, val_loader, test_loader = (
    prepare_tiger_data(
        batch_size=BATCH_SIZE,
        max_history=MAX_HISTORY
    )
)

train_dataset = train_loader.dataset

print("训练样本:", len(train_dataset))
print("验证样本:", len(val_loader.dataset))


# ==========================================
# 4. 统计训练目标商品频率
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


# ==========================================
# 5. 商品流行度分组
# ==========================================

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
        item_groups[item_id] = 0

    elif rank < middle_end:
        item_groups[item_id] = 1

    else:
        item_groups[item_id] = 2

GROUP_NAMES = [
    "热门",
    "中等",
    "长尾"
]

print("热门商品数:", hot_end)
print("中等商品数:", middle_end - hot_end)
print("长尾商品数:", num_items - middle_end)


# ==========================================
# 6. 将历史Semantic Token还原成商品ID
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

            item_id = token_to_item.get(tokens)

            if item_id is None:
                raise ValueError(
                    f"无法识别历史Semantic Token: {tokens}"
                )

            history_ids[i, j] = item_id

    return history_ids


# ==========================================
# 7. 构造用户向量
# ==========================================

def build_user_embeddings(history_ids, k):

    recent_items = history_ids[:, -k:]

    recent_vectors = embedding_tensor[
        recent_items
    ]

    valid_mask = (
        (recent_items > 0)
        & torch.tensor(
            valid_embeddings,
            dtype=torch.bool,
            device=device
        )[recent_items]
    )

    valid_mask_float = (
        valid_mask.unsqueeze(-1).float()
    )

    sum_vectors = (
        recent_vectors * valid_mask_float
    ).sum(dim=1)

    counts = valid_mask_float.sum(dim=1)

    user_vectors = (
        sum_vectors / counts.clamp_min(1.0)
    )

    user_vectors = F.normalize(
        user_vectors,
        p=2,
        dim=1
    )

    valid_queries = (
        (counts.squeeze(-1) > 0)
        & (torch.linalg.vector_norm(
            user_vectors,
            dim=1
        ) > 1e-8)
    )

    return user_vectors, valid_queries


# ==========================================
# 8. 初始化统计
# ==========================================

stats = {}

for method in METHODS:

    stats[method] = {
        "samples": 0,
        "hits_50": 0,
        "hits_100": 0,
        "group_samples": np.zeros(3, dtype=np.int64),
        "group_hits_50": np.zeros(3, dtype=np.int64),
        "group_hits_100": np.zeros(3, dtype=np.int64),
        "recommended_items": set(),
        "invalid_queries": 0
    }


# ==========================================
# 9. 完整验证集评估
# ==========================================

print("\n========== 开始纯语义召回实验 ==========")

processed = 0
last_print = 0

catalog_mask = ~catalog_valid

for batch in val_loader:

    history_np = decode_history(
        batch["encoder_input"]
    )

    history_ids = torch.tensor(
        history_np,
        dtype=torch.long,
        device=device
    )

    targets = batch[
        "target_item_id"
    ].cpu().numpy()

    target_groups = item_groups[targets]

    for method, k in METHODS.items():

        result = stats[method]

        user_vectors, valid_queries = (
            build_user_embeddings(
                history_ids,
                k
            )
        )

        # 余弦相似度矩阵
        # [Batch, 384] @ [384, 商品数]

        scores = (
            user_vectors
            @ catalog_embeddings.T
        )

        # 排除缺失文本向量的商品
        scores[:, catalog_mask] = -1e9

        top_indices = torch.topk(
            scores,
            k=TOP_K,
            dim=1
        ).indices

        # 索引0对应原商品ID=1
        top_items = top_indices + 1

        # 无有效历史向量时，不生成候选
        top_items = torch.where(
            valid_queries.unsqueeze(1),
            top_items,
            torch.zeros_like(top_items)
        )

        predictions = (
            top_items.cpu().numpy()
        )

        valid_queries_np = (
            valid_queries.cpu().numpy()
        )

        result["invalid_queries"] += int(
            (~valid_queries_np).sum()
        )

        for i, target in enumerate(targets):

            target = int(target)

            group = int(target_groups[i])

            recommendations = predictions[i]

            result["samples"] += 1

            result["group_samples"][group] += 1

            result["recommended_items"].update(
                recommendations[
                    recommendations > 0
                ].tolist()
            )

            if target in recommendations[:50]:

                result["hits_50"] += 1

                result["group_hits_50"][group] += 1

            if target in recommendations[:100]:

                result["hits_100"] += 1

                result["group_hits_100"][group] += 1

    processed += len(targets)

    if processed - last_print >= PRINT_EVERY:

        elapsed = (
            time.time() - start_time
        ) / 60

        print(
            f"已处理 {processed} 条 | "
            f"耗时 {elapsed:.2f}分钟",
            flush=True
        )

        last_print = processed


# ==========================================
# 10. 输出结果
# ==========================================

print("\n========== 实验结果 ==========")

rows = []

for method in METHODS:

    result = stats[method]

    n = result["samples"]

    recall_50 = result["hits_50"] / n

    recall_100 = result["hits_100"] / n

    coverage = (
        len(result["recommended_items"])
        / num_items
    )

    group_recalls_50 = []
    group_recalls_100 = []

    print(f"\n========== {method} ==========")

    print("Recall@50:", f"{recall_50:.6f}")
    print("Recall@100:", f"{recall_100:.6f}")

    print("商品覆盖率@100:", f"{coverage:.2%}")

    print(
        "无有效历史向量样本:",
        result["invalid_queries"]
    )

    for group in range(3):

        count = result["group_samples"][group]

        r50 = (
            result["group_hits_50"][group] / count
            if count else 0.0
        )

        r100 = (
            result["group_hits_100"][group] / count
            if count else 0.0
        )

        group_recalls_50.append(r50)
        group_recalls_100.append(r100)

        print(
            f"{GROUP_NAMES[group]} "
            f"Recall@50={r50:.6f} "
            f"Recall@100={r100:.6f}"
        )

    rows.append({
        "method": method,
        "recall_at_50": recall_50,
        "recall_at_100": recall_100,
        "hot_recall_50": group_recalls_50[0],
        "middle_recall_50": group_recalls_50[1],
        "tail_recall_50": group_recalls_50[2],
        "hot_recall_100": group_recalls_100[0],
        "middle_recall_100": group_recalls_100[1],
        "tail_recall_100": group_recalls_100[2],
        "coverage_at_100": coverage,
        "invalid_queries": result["invalid_queries"]
    })


# ==========================================
# 11. 保存CSV
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
        fieldnames=list(rows[0].keys())
    )

    writer.writeheader()
    writer.writerows(rows)


# ==========================================
# 12. 结束
# ==========================================

print("\n结果保存至:", OUTPUT_PATH)

print(
    "总耗时:",
    f"{(time.time() - start_time) / 60:.2f}分钟"
)

print("实验完成")
