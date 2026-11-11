
import csv
import json
import math
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

OUTPUT_DIR = BASE_DIR / "test_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 32
TOP_K = 10

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

print("当前设备:", device)


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


print("\n========== 检查数据时间顺序 ==========")

train_latest = {}

for record in train_data:

    user = record["user_id"]
    timestamp = record["timestamp"]

    if (
        user not in train_latest
        or timestamp > train_latest[user]
    ):
        train_latest[user] = timestamp


val_by_user = {
    record["user_id"]: record
    for record in val_data
}

test_by_user = {
    record["user_id"]: record
    for record in test_data
}

strict_order = 0
equal_time = 0
wrong_order = 0
missing_user = 0
prefix_mismatch = 0

for user, test_record in test_by_user.items():

    if user not in val_by_user or user not in train_latest:
        missing_user += 1
        continue

    train_time = train_latest[user]
    val_record = val_by_user[user]

    val_time = val_record["timestamp"]
    test_time = test_record["timestamp"]

    if train_time < val_time < test_time:
        strict_order += 1

    elif train_time <= val_time <= test_time:
        equal_time += 1

    else:
        wrong_order += 1

    history = test_record["history"]

    if (
        len(history) == 0
        or history[-1] != val_record["parent_asin"]
    ):
        prefix_mismatch += 1


print("严格满足 Train < Val < Test:", strict_order)
print("存在相同时间戳:", equal_time)
print("时间顺序异常:", wrong_order)
print("缺少用户记录:", missing_user)
print("测试历史与验证目标衔接异常:", prefix_mismatch)

if wrong_order > 0 or missing_user > 0 or prefix_mismatch > 0:
    print("\n数据检查未通过，暂不运行最终测试。")
    raise SystemExit(1)

if equal_time > 0:
    print(
        "\n警告：存在时间戳相同的交互。"
        "当前只能确认时间没有倒退，"
        "无法仅靠时间戳确认这些交互的先后顺序。"
    )

print("\n初步时间顺序检查通过")


print("\n========== 加载最佳 SASRec ==========")

with open(MAPPING_PATH, "r") as f:
    item_to_id = json.load(f)

checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu",
    weights_only=True
)

if len(item_to_id) != checkpoint["num_items"]:
    raise ValueError("商品映射与训练模型不一致")

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

print("加载成功")
print("最佳训练轮数:", checkpoint["epoch"])
print("最佳验证 NDCG@10:", checkpoint["best_ndcg"])


print("\n========== 构建简单推荐基线 ==========")

popularity = Counter()
transitions = defaultdict(Counter)

for record in train_data:

    target = item_to_id[record["parent_asin"]]
    popularity[target] += 1

    if record["history"]:

        last_item = item_to_id[record["history"][-1]]
        transitions[last_item][target] += 1


popular_items = [
    item
    for item, count in popularity.most_common()
]

popular_top10 = popular_items[:TOP_K]


def get_transition_top10(last_item):

    counter = transitions.get(last_item)

    recommendations = []
    seen = set()

    if counter:

        for item, count in counter.most_common(TOP_K):
            recommendations.append(item)
            seen.add(item)

    for item in popular_items:

        if len(recommendations) >= TOP_K:
            break

        if item not in seen:
            recommendations.append(item)
            seen.add(item)

    return recommendations


transition_top10 = {
    item: get_transition_top10(item)
    for item in transitions
}


print("\n========== 创建测试 DataLoader ==========")

test_dataset = SequenceDataset(
    test_data,
    item_to_id,
    max_len=checkpoint["max_len"]
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


stats = {
    name: {
        "samples": 0,
        "hits": 0,
        "ndcg_sum": 0.0
    }
    for name in ["SASRec", "Transition", "Popularity"]
}

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

    recalls = hits.any(dim=1).float()

    ndcgs = (
        hits.float() * discounts
    ).sum(dim=1)

    stats[name]["samples"] += len(y)
    stats[name]["hits"] += int(recalls.sum().item())
    stats[name]["ndcg_sum"] += ndcgs.sum().item()


print("\n========== 开始独立测试 ==========")

with torch.no_grad():

    for batch_index, (X, y) in enumerate(test_loader):

        last_items = X[:, -1].tolist()

        X = X.to(device)
        y = y.to(device)

        scores = model(X)

        sasrec_top10 = torch.topk(
            scores,
            k=TOP_K,
            dim=1
        ).indices

        update_stats("SASRec", sasrec_top10, y)

        transition_lists = [
            transition_top10.get(item, popular_top10)
            for item in last_items
        ]

        transition_top = torch.tensor(
            transition_lists,
            dtype=torch.long,
            device=device
        )

        update_stats("Transition", transition_top, y)

        popularity_top = torch.tensor(
            popular_top10,
            dtype=torch.long,
            device=device
        ).unsqueeze(0).expand(len(y), -1)

        update_stats("Popularity", popularity_top, y)

        if (batch_index + 1) % 500 == 0:

            processed = min(
                (batch_index + 1) * BATCH_SIZE,
                len(test_dataset)
            )

            print(
                f"已测试 {processed}/{len(test_dataset)} 条",
                flush=True
            )


print("\n========== 最终测试结果 ==========")

rows = []

for name, item in stats.items():

    recall10 = item["hits"] / item["samples"]
    ndcg10 = item["ndcg_sum"] / item["samples"]

    print(f"\n模型: {name}")
    print("测试样本数:", item["samples"])
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


csv_path = OUTPUT_DIR / "sasrec_final_test.csv"

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


print("\n========== 实验结束 ==========")
print("最佳模型 Epoch:", checkpoint["epoch"])
print("最终测试结果已保存:", csv_path)
