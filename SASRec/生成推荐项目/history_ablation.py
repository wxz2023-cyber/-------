
import csv
import json
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

MAX_LEN = checkpoint["max_len"]

print("成功加载最佳 Epoch:", checkpoint["epoch"])


print("\n========== 加载数据 ==========")

ds = load_dataset(
    "yufan/amazon2023-user-interactions",
    "seq_maxlen20_Video_Games",
    token=True
)

train_data = ds["train"]
val_data = ds["validation"]
test_data = ds["test"]

print("Train:", len(train_data))
print("Validation:", len(val_data))
print("Test:", len(test_data))


print("\n========== 实验A：检查时间划分 ==========")

train_latest = {}

for record in train_data:

    user = record["user_id"]
    timestamp = record["timestamp"]

    if (
        user not in train_latest
        or timestamp > train_latest[user]
    ):
        train_latest[user] = timestamp


val_records = {
    record["user_id"]: record
    for record in val_data
}

test_records = {
    record["user_id"]: record
    for record in test_data
}

valid_order = 0
invalid_order = 0
missing_users = 0
equal_timestamps = 0

for user, val_record in val_records.items():

    if (
        user not in train_latest
        or user not in test_records
    ):
        missing_users += 1
        continue

    train_time = train_latest[user]
    val_time = val_record["timestamp"]
    test_time = test_records[user]["timestamp"]

    if train_time < val_time < test_time:
        valid_order += 1
    elif train_time <= val_time <= test_time:
        equal_timestamps += 1
    else:
        invalid_order += 1


print("严格满足 Train < Val < Test:", valid_order)
print("存在相同时间戳:", equal_timestamps)
print("时间顺序不符合预期:", invalid_order)
print("用户记录不完整:", missing_users)

print(
    "注意：这里只检查目标交互时间顺序，"
    "不能单独证明不存在数据泄漏"
)


print("\n========== 实验B：历史长度截断 ==========")

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

history_limits = [5, 10, 20]

stats = {
    length: {
        "samples": 0,
        "hits": 0,
        "ndcg_sum": 0.0
    }
    for length in history_limits
}

groups = ["2-5", "6-10", "11-20"]

group_stats = {
    length: {
        group: {
            "samples": 0,
            "hits": 0,
            "ndcg_sum": 0.0
        }
        for group in groups
    }
    for length in history_limits
}

discounts = 1.0 / torch.log2(
    torch.arange(
        1,
        TOP_K + 1,
        device=device,
        dtype=torch.float32
    ) + 1
)


def get_group(length):

    if length <= 5:
        return "2-5"

    if length <= 10:
        return "6-10"

    return "11-20"


with torch.no_grad():

    for batch_index, (X, y) in enumerate(val_loader):

        original_lengths = (X != 0).sum(dim=1).tolist()

        X = X.to(device)
        y = y.to(device)

        for history_limit in history_limits:

            X_modified = X.clone()

            if history_limit < MAX_LEN:
                X_modified[:, :-history_limit] = 0

            scores = model(X_modified)

            top10 = torch.topk(
                scores,
                k=TOP_K,
                dim=1
            ).indices

            hits = top10.eq(y.unsqueeze(1))

            recall_values = (
                hits.any(dim=1)
                .int()
                .cpu()
                .tolist()
            )

            ndcg_values = (
                (hits.float() * discounts)
                .sum(dim=1)
                .cpu()
                .tolist()
            )

            for original_length, hit, ndcg in zip(
                original_lengths,
                recall_values,
                ndcg_values
            ):

                group = get_group(original_length)

                stats[history_limit]["samples"] += 1
                stats[history_limit]["hits"] += hit
                stats[history_limit]["ndcg_sum"] += ndcg

                group_stats[history_limit][group]["samples"] += 1
                group_stats[history_limit][group]["hits"] += hit
                group_stats[history_limit][group]["ndcg_sum"] += ndcg

        if (batch_index + 1) % 500 == 0:
            print(
                f"已处理 {(batch_index + 1) * BATCH_SIZE} 条",
                flush=True
            )


print("\n========== 整体实验结果 ==========")

rows = []

for history_limit in history_limits:

    item = stats[history_limit]

    recall10 = item["hits"] / item["samples"]
    ndcg10 = item["ndcg_sum"] / item["samples"]

    print(
        f"最近 {history_limit:>2} 件 | "
        f"Recall@10: {recall10:.6f} | "
        f"NDCG@10: {ndcg10:.6f}"
    )

    rows.append({
        "history_limit": history_limit,
        "group": "overall",
        "samples": item["samples"],
        "recall10": recall10,
        "ndcg10": ndcg10
    })


print("\n========== 不同历史组的实验结果 ==========")

for group in groups:

    print(f"\n原始历史长度：{group}")

    for history_limit in history_limits:

        item = group_stats[history_limit][group]

        if item["samples"] == 0:
            continue

        recall10 = item["hits"] / item["samples"]
        ndcg10 = item["ndcg_sum"] / item["samples"]

        print(
            f"最近 {history_limit:>2} 件 | "
            f"Recall@10: {recall10:.6f} | "
            f"NDCG@10: {ndcg10:.6f}"
        )

        rows.append({
            "history_limit": history_limit,
            "group": group,
            "samples": item["samples"],
            "recall10": recall10,
            "ndcg10": ndcg10
        })


csv_path = OUTPUT_DIR / "history_ablation.csv"

with open(csv_path, "w", newline="") as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "history_limit",
            "group",
            "samples",
            "recall10",
            "ndcg10"
        ]
    )

    writer.writeheader()
    writer.writerows(rows)


print("\n========== 实验完成 ==========")
print("结果文件:", csv_path)
