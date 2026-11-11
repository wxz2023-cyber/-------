
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

MODEL_PATH = BASE_DIR / "checkpoints" / "tiger_popularity_best.pth"
SEMANTIC_PATH = BASE_DIR / "processed_data" / "item_semantic_ids_unique.npy"
OUTPUT_PATH = BASE_DIR / "processed_data" / "popularity_reranking_results.csv"

BATCH_SIZE = 8
BEAM_SIZE = 50
TOP_K = 10

LAMBDAS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]
PRINT_EVERY = 10000

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

print("当前设备:", device)


# ==========================================
# 1. 建立Semantic ID前缀树
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
        TOKEN_START + i * CODEBOOK_SIZE + int(v)
        for i, v in enumerate(code)
    )

    if tokens in code_to_item:
        raise ValueError("发现重复Semantic ID")

    code_to_item[tokens] = item_id

    for position in range(4):
        children[tokens[:position]].add(tokens[position])

children = {
    prefix: np.array(sorted(values), dtype=np.int64)
    for prefix, values in children.items()
}

print("商品数量:", len(code_to_item))


# ==========================================
# 2. 加载模型
# ==========================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu",
    weights_only=True
)

model = TIGER(**checkpoint["model_config"]).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("模型Epoch:", checkpoint["epoch"])


# ==========================================
# 3. 加载数据
# ==========================================

train_loader, val_loader, test_loader = prepare_tiger_data(
    batch_size=BATCH_SIZE,
    max_history=20
)

train_dataset = train_loader.dataset

item_counts = np.zeros(num_items + 1, dtype=np.int64)

for record in train_dataset.records:

    asin = str(record["parent_asin"])
    item_id = int(train_dataset.item_to_id[asin])

    item_counts[item_id] += 1

log_popularity = np.log1p(item_counts)


# ==========================================
# 4. 商品热门程度分组
# ==========================================

sorted_items = sorted(
    range(1, num_items + 1),
    key=lambda i: (-item_counts[i], i)
)

head_end = int(num_items * 0.1)
middle_end = int(num_items * 0.5)

item_group = np.zeros(num_items + 1, dtype=np.int32)

for rank, item_id in enumerate(sorted_items):

    if rank < head_end:
        item_group[item_id] = 0

    elif rank < middle_end:
        item_group[item_id] = 1

    else:
        item_group[item_id] = 2

GROUP_NAMES = ["热门", "中等热度", "长尾"]


# ==========================================
# 5. Beam Search返回商品和原始分数
# ==========================================

@torch.inference_mode()
def beam_search_batch(encoder_input):

    memory, padding_mask = model.encode(encoder_input)

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
            0, owner_tensor
        )

        selected_mask = padding_mask.index_select(
            0, owner_tensor
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

            token_scores = log_probs[row_idx, allowed]

            keep = min(BEAM_SIZE, len(allowed))

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
                user_results.append(
                    (item_id, score)
                )

        results.append(user_results)

    return results


# ==========================================
# 6. 流行度校正排序
# ==========================================

def rerank(candidates, lam):

    adjusted = []

    for item_id, score in candidates:

        new_score = (
            score
            - lam * log_popularity[item_id]
        )

        adjusted.append(
            (item_id, new_score)
        )

    adjusted.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return [
        item_id
        for item_id, score in adjusted[:TOP_K]
    ]


# ==========================================
# 7. 初始化各组统计
# ==========================================

statistics = {}

for lam in LAMBDAS:

    statistics[lam] = {
        "samples": 0,
        "hits": 0,
        "ndcg": 0.0,
        "group_samples": np.zeros(3, dtype=np.int64),
        "group_hits": np.zeros(3, dtype=np.int64),
        "recommended_counts": np.zeros(3, dtype=np.int64),
        "recommended_items": set()
    }


# ==========================================
# 8. 完整验证集评估
# ==========================================

print("\n========== 开始重排序实验 ==========")
print("Beam Size:", BEAM_SIZE)
print("候选Lambda:", LAMBDAS)

start_time = time.time()
processed = 0
last_print = 0

for batch in val_loader:

    encoder_input = batch["encoder_input"].to(device)
    targets = batch["target_item_id"].cpu().numpy()

    candidate_lists = beam_search_batch(encoder_input)

    for candidates, target in zip(candidate_lists, targets):

        target = int(target)
        target_group = item_group[target]

        for lam in LAMBDAS:

            result = statistics[lam]

            recommendations = rerank(candidates, lam)

            result["samples"] += 1
            result["group_samples"][target_group] += 1

            for item_id in recommendations:

                result["recommended_items"].add(item_id)

                result["recommended_counts"][
                    item_group[item_id]
                ] += 1

            if target in recommendations:

                rank = recommendations.index(target) + 1

                result["hits"] += 1
                result["group_hits"][target_group] += 1

                result["ndcg"] += (
                    1.0 / math.log2(rank + 1)
                )

    processed += len(targets)

    if processed - last_print >= PRINT_EVERY:

        print(
            f"已处理 {processed} 条 | "
            f"耗时 {(time.time()-start_time)/60:.2f}分钟",
            flush=True
        )

        last_print = processed


# ==========================================
# 9. 输出结果
# ==========================================

print("\n========== 实验结果 ==========")

rows = []

for lam in LAMBDAS:

    result = statistics[lam]

    n = result["samples"]

    recall = result["hits"] / n
    ndcg = result["ndcg"] / n

    coverage = (
        len(result["recommended_items"]) / num_items
    )

    total_recommendations = (
        result["recommended_counts"].sum()
    )

    group_recalls = []

    for group in range(3):

        count = result["group_samples"][group]

        group_recall = (
            result["group_hits"][group] / count
            if count > 0 else 0.0
        )

        group_recalls.append(group_recall)

    hot_share = (
        result["recommended_counts"][0]
        / total_recommendations
    )

    tail_share = (
        result["recommended_counts"][2]
        / total_recommendations
    )

    print(f"\nLambda = {lam}")

    print(f"Recall@10: {recall:.6f}")
    print(f"NDCG@10: {ndcg:.6f}")
    print(f"热门Recall: {group_recalls[0]:.6f}")
    print(f"中等Recall: {group_recalls[1]:.6f}")
    print(f"长尾Recall: {group_recalls[2]:.6f}")
    print(f"商品覆盖率: {coverage:.2%}")
    print(f"热门推荐占比: {hot_share:.2%}")
    print(f"长尾推荐占比: {tail_share:.2%}")

    rows.append({
        "lambda": lam,
        "recall_at_10": recall,
        "ndcg_at_10": ndcg,
        "hot_recall": group_recalls[0],
        "middle_recall": group_recalls[1],
        "tail_recall": group_recalls[2],
        "coverage": coverage,
        "hot_recommendation_share": hot_share,
        "tail_recommendation_share": tail_share
    })


# ==========================================
# 10. 保存CSV
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
# 11. 输出最高Recall的配置
# ==========================================

best = max(
    rows,
    key=lambda x: x["recall_at_10"]
)

print("\n========== 最佳整体Recall配置 ==========")

print("Lambda:", best["lambda"])
print("Recall@10:", f'{best["recall_at_10"]:.6f}')
print("NDCG@10:", f'{best["ndcg_at_10"]:.6f}')
print("长尾Recall:", f'{best["tail_recall"]:.6f}')
print("商品覆盖率:", f'{best["coverage"]:.2%}')

print("\n结果保存至:", OUTPUT_PATH)

print(
    "总耗时:",
    round((time.time() - start_time) / 60, 2),
    "分钟"
)

print("\n实验完成")
