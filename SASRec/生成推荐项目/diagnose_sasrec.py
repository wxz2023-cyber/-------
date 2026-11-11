
import csv
import json
import math
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset

from sasrec import SASRec
from video_games_data import SequenceDataset


torch.manual_seed(42)

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR / "checkpoints" / "sasrec_15epoch_best.pth"
)

MAPPING_PATH = (
    BASE_DIR / "processed_data" / "item_to_id.json"
)

OUTPUT_DIR = BASE_DIR / "diagnosis_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 32

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

print("当前设备:", device)

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"没有找到模型文件: {MODEL_PATH}"
    )

if not MAPPING_PATH.exists():
    raise FileNotFoundError(
        f"没有找到商品映射文件: {MAPPING_PATH}"
    )


with open(MAPPING_PATH, "r") as f:
    item_to_id = json.load(f)

checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu",
    weights_only=True
)

if len(item_to_id) != checkpoint["num_items"]:
    raise ValueError("商品映射数量与模型不一致")

model = SASRec(
    num_items=checkpoint["num_items"],
    embedding_dim=checkpoint["embedding_dim"],
    max_len=checkpoint["max_len"],
    num_heads=checkpoint["num_heads"],
    num_layers=checkpoint["num_layers"],
    dropout=checkpoint["dropout"]
).to(device)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("成功加载最佳模型")
print("最佳 Epoch:", checkpoint["epoch"])
print("原验证 NDCG@10:", checkpoint["best_ndcg"])


print("\n开始加载数据...")

ds = load_dataset(
    "yufan/amazon2023-user-interactions",
    "seq_maxlen20_Video_Games",
    token=True
)

train_data = ds["train"]
val_data = ds["validation"]

MAX_LEN = checkpoint["max_len"]


def get_group(length):

    if length == 0:
        return "0"

    if length == 1:
        return "1"

    if length <= 5:
        return "2-5"

    if length <= 10:
        return "6-10"

    return "11-20"


GROUPS = ["0", "1", "2-5", "6-10", "11-20"]


def analyze_distribution(data, split_name):

    group_counts = Counter()
    length_counts = Counter()
    user_ids = set()
    target_items = set()

    for record in data:

        length = min(len(record["history"]), MAX_LEN)

        group_counts[get_group(length)] += 1
        length_counts[length] += 1

        user_ids.add(record["user_id"])
        target_items.add(record["parent_asin"])

    total = len(data)

    print(f"\n========== {split_name} 历史长度分布 ==========")

    print("样本总数:", total)
    print("不同用户数:", len(user_ids))
    print("不同目标商品数:", len(target_items))

    rows = []

    for group in GROUPS:

        count = group_counts[group]
        percentage = count / total * 100

        print(
            f"历史长度 {group:>5} | "
            f"样本: {count:>7} | "
            f"占比: {percentage:.2f}%"
        )

        rows.append({
            "split": split_name,
            "history_group": group,
            "samples": count,
            "percentage": percentage
        })

    print("\n各长度的具体数量:")

    for length in range(MAX_LEN + 1):
        if length_counts[length] > 0:
            print(
                f"长度 {length:>2}: "
                f"{length_counts[length]}"
            )

    return rows, user_ids, target_items


train_rows, train_users, train_items = analyze_distribution(
    train_data,
    "Train"
)

val_rows, val_users, val_items = analyze_distribution(
    val_data,
    "Validation"
)


print("\n========== 数据重叠检查 ==========")

shared_users = train_users & val_users
unseen_targets = val_items - train_items

print("训练集用户数:", len(train_users))
print("验证集用户数:", len(val_users))
print("两者共同用户数:", len(shared_users))

print("训练集目标商品数:", len(train_items))
print("验证集目标商品数:", len(val_items))
print("验证集未在训练目标中出现的商品数:", len(unseen_targets))

unseen_target_samples = sum(
    record["parent_asin"] not in train_items
    for record in val_data
)

print(
    "验证集冷启动目标样本数:",
    unseen_target_samples
)

print(
    "验证集冷启动目标样本占比:",
    f"{unseen_target_samples / len(val_data) * 100:.2f}%"
)


distribution_path = OUTPUT_DIR / "history_distribution.csv"

with open(distribution_path, "w", newline="") as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "split",
            "history_group",
            "samples",
            "percentage"
        ]
    )

    writer.writeheader()
    writer.writerows(train_rows + val_rows)


print("\n========== 开始按历史长度评估 ==========")

val_dataset = SequenceDataset(
    val_data,
    item_to_id,
    max_len=MAX_LEN
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

results = {
    group: {
        "samples": 0,
        "hits": 0,
        "ndcg_sum": 0.0
    }
    for group in GROUPS
}

position_discounts = (
    1.0 / torch.log2(
        torch.arange(
            1,
            11,
            device=device,
            dtype=torch.float32
        ) + 1
    )
)

total_hits = 0
total_ndcg = 0.0
total_evaluated = 0


with torch.no_grad():

    for batch_index, (X, y) in enumerate(val_loader):

        lengths = (X != 0).sum(dim=1)

        if torch.any(lengths == 0):
            raise ValueError(
                "发现空历史样本，需要单独处理"
            )

        X = X.to(device)
        y = y.to(device)

        scores = model(X)

        top10 = torch.topk(
            scores,
            k=10,
            dim=1
        ).indices

        hits = top10.eq(y.unsqueeze(1))

        recall = hits.any(dim=1).float()

        ndcg = (
            hits.float() * position_discounts
        ).sum(dim=1)

        recall_values = recall.cpu().tolist()
        ndcg_values = ndcg.cpu().tolist()
        length_values = lengths.tolist()

        for length, hit, ndcg_score in zip(
            length_values,
            recall_values,
            ndcg_values
        ):

            group = get_group(length)

            results[group]["samples"] += 1
            results[group]["hits"] += int(hit)
            results[group]["ndcg_sum"] += ndcg_score

            total_hits += int(hit)
            total_ndcg += ndcg_score
            total_evaluated += 1

        if (batch_index + 1) % 500 == 0:

            print(
                f"已评估 {total_evaluated}/{len(val_dataset)} 条",
                flush=True
            )


print("\n========== 不同历史长度的推荐效果 ==========")

metric_rows = []

for group in GROUPS:

    count = results[group]["samples"]
    hits = results[group]["hits"]
    ndcg_sum = results[group]["ndcg_sum"]

    if count == 0:
        recall10 = 0.0
        ndcg10 = 0.0
    else:
        recall10 = hits / count
        ndcg10 = ndcg_sum / count

    print(
        f"历史长度 {group:>5} | "
        f"样本: {count:>7} | "
        f"Recall@10: {recall10:.6f} | "
        f"NDCG@10: {ndcg10:.6f}"
    )

    metric_rows.append({
        "history_group": group,
        "samples": count,
        "hits": hits,
        "recall10": recall10,
        "ndcg10": ndcg10
    })


overall_recall = total_hits / total_evaluated
overall_ndcg = total_ndcg / total_evaluated

print("\n========== 整体验证结果 ==========")

print("评估样本数:", total_evaluated)
print(f"Recall@10: {overall_recall:.6f}")
print(f"NDCG@10: {overall_ndcg:.6f}")

print(
    "与原最佳 NDCG 的差值:",
    round(overall_ndcg - checkpoint["best_ndcg"], 8)
)


metrics_path = OUTPUT_DIR / "recall_by_history_length.csv"

with open(metrics_path, "w", newline="") as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "history_group",
            "samples",
            "hits",
            "recall10",
            "ndcg10"
        ]
    )

    writer.writeheader()
    writer.writerows(metric_rows)


print("\n========== 实验完成 ==========")

print("历史分布文件:", distribution_path)
print("分组评估文件:", metrics_path)
