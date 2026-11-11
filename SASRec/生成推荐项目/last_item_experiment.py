
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset

from sasrec import SASRec
from video_games_data import SequenceDataset


torch.manual_seed(42)

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "checkpoints" / "sasrec_15epoch_best.pth"
MAPPING_PATH = BASE_DIR / "processed_data" / "item_to_id.json"

OUTPUT_DIR = BASE_DIR / "diagnosis_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 32
TOP_K = 10

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

print("当前设备:", device)


with open(MAPPING_PATH, "r") as f:
    item_to_id = json.load(f)

checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu",
    weights_only=True
)

if len(item_to_id) != checkpoint["num_items"]:
    raise ValueError("商品映射与模型不一致")

model = SASRec(
    num_items=checkpoint["num_items"],
    embedding_dim=checkpoint["embedding_dim"],
    max_len=checkpoint["max_len"],
    num_heads=checkpoint["num_heads"],
    num_layers=checkpoint["num_layers"],
    dropout=checkpoint["dropout"]
).to(device)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("模型加载成功")
print("最佳 Epoch:", checkpoint["epoch"])


print("\n========== 加载数据 ==========")

ds = load_dataset(
    "yufan/amazon2023-user-interactions",
    "seq_maxlen20_Video_Games",
    token=True
)

train_data = ds["train"]
val_data = ds["validation"]

print("训练样本数:", len(train_data))
print("验证样本数:", len(val_data))


print("\n========== 统计商品转移关系 ==========")

transition_counts = defaultdict(Counter)
popularity_counts = Counter()

for record in train_data:

    target = item_to_id[record["parent_asin"]]

    popularity_counts[target] += 1

    history = record["history"]

    if len(history) == 0:
        continue

    last_item = item_to_id[history[-1]]

    transition_counts[last_item][target] += 1


popular_items = [
    item
    for item, count in popularity_counts.most_common()
]

popular_top10 = popular_items[:TOP_K]


def make_top10(counter):

    recommendations = []
    seen = set()

    for item, count in counter.most_common(TOP_K):

        if item not in seen:
            recommendations.append(item)
            seen.add(item)

    if len(recommendations) < TOP_K:

        for item in popular_items:

            if item not in seen:
                recommendations.append(item)
                seen.add(item)

            if len(recommendations) == TOP_K:
                break

    return recommendations


transition_top10 = {
    item: make_top10(counter)
    for item, counter in transition_counts.items()
}

print("有转移记录的商品数:", len(transition_top10))
print("热门推荐 Top10:", popular_top10)


print("\n========== 创建验证 DataLoader ==========")

val_dataset = SequenceDataset(
    val_data,
    item_to_id,
    max_len=checkpoint["max_len"]
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


EXPERIMENTS = [
    "SASRec_Full",
    "SASRec_LastItem",
    "Transition",
    "Popularity"
]

stats = {
    name: {
        "samples": 0,
        "hits": 0,
        "ndcg_sum": 0.0
    }
    for name in EXPERIMENTS
}

transition_covered = 0

discounts = 1.0 / torch.log2(
    torch.arange(
        1,
        TOP_K + 1,
        device=device,
        dtype=torch.float32
    ) + 1
)


def update_stats(name, top10, y):

    hits = top10.eq(y.unsqueeze(1))

    recall = hits.any(dim=1).float()

    ndcg = (
        hits.float() * discounts
    ).sum(dim=1)

    stats[name]["samples"] += len(y)
    stats[name]["hits"] += int(recall.sum().item())
    stats[name]["ndcg_sum"] += ndcg.sum().item()


print("\n========== 开始实验 ==========")

with torch.no_grad():

    for batch_index, (X, y) in enumerate(val_loader):

        last_items = X[:, -1].tolist()

        X_full = X.to(device)
        y = y.to(device)

        full_scores = model(X_full)

        full_top10 = torch.topk(
            full_scores,
            k=TOP_K,
            dim=1
        ).indices

        update_stats(
            "SASRec_Full",
            full_top10,
            y
        )

        X_last = torch.zeros_like(X_full)

        X_last[:, -1] = X_full[:, -1]

        last_scores = model(X_last)

        last_top10 = torch.topk(
            last_scores,
            k=TOP_K,
            dim=1
        ).indices

        update_stats(
            "SASRec_LastItem",
            last_top10,
            y
        )

        transition_lists = []

        for last_item in last_items:

            if last_item in transition_top10:

                transition_covered += 1

                candidates = transition_top10[last_item]

            else:

                candidates = popular_top10

            transition_lists.append(candidates)

        transition_top = torch.tensor(
            transition_lists,
            dtype=torch.long,
            device=device
        )

        update_stats(
            "Transition",
            transition_top,
            y
        )

        popularity_top = torch.tensor(
            popular_top10,
            dtype=torch.long,
            device=device
        ).unsqueeze(0).expand(
            len(y),
            -1
        )

        update_stats(
            "Popularity",
            popularity_top,
            y
        )

        if (batch_index + 1) % 500 == 0:

            processed = min(
                (batch_index + 1) * BATCH_SIZE,
                len(val_dataset)
            )

            print(
                f"已评估 {processed}/{len(val_dataset)} 条",
                flush=True
            )


print("\n========== 最终实验结果 ==========")

rows = []

for name in EXPERIMENTS:

    item = stats[name]

    recall10 = item["hits"] / item["samples"]
    ndcg10 = item["ndcg_sum"] / item["samples"]

    print(f"\n模型: {name}")
    print("验证样本数:", item["samples"])
    print("命中数量:", item["hits"])
    print(f"Recall@10: {recall10:.6f}")
    print(f"NDCG@10:   {ndcg10:.6f}")

    rows.append({
        "model": name,
        "samples": item["samples"],
        "hits": item["hits"],
        "recall10": recall10,
        "ndcg10": ndcg10
    })


coverage = transition_covered / len(val_dataset)

print("\n商品转移基线覆盖率:", f"{coverage:.2%}")


csv_path = OUTPUT_DIR / "last_item_comparison.csv"

with open(csv_path, "w", newline="") as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "model",
            "samples",
            "hits",
            "recall10",
            "ndcg10"
        ]
    )

    writer.writeheader()
    writer.writerows(rows)


print("\n========== 实验完成 ==========")

print("结果文件:", csv_path)
