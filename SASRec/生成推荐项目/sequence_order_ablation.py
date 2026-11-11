
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

print("加载模型成功")
print("最佳 Epoch:", checkpoint["epoch"])


print("\n加载验证集...")

ds = load_dataset(
    "yufan/amazon2023-user-interactions",
    "seq_maxlen20_Video_Games",
    token=True
)

val_dataset = SequenceDataset(
    ds["validation"],
    item_to_id,
    max_len=checkpoint["max_len"]
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("验证样本数:", len(val_dataset))


def shuffle_history(X, keep_last=False):

    X_new = X.clone()

    for i in range(X.shape[0]):

        valid_indices = torch.nonzero(
            X[i] != 0
        ).flatten()

        if keep_last:
            valid_indices = valid_indices[:-1]

        if len(valid_indices) <= 1:
            continue

        original_items = X[i, valid_indices].clone()

        permutation = torch.randperm(
            len(valid_indices)
        )

        X_new[i, valid_indices] = original_items[permutation]

    return X_new


discounts = 1.0 / torch.log2(
    torch.arange(
        1,
        TOP_K + 1,
        device=device,
        dtype=torch.float32
    ) + 1
)


experiments = {
    "original": {
        "samples": 0,
        "hits": 0,
        "ndcg_sum": 0.0,
        "changed": 0,
        "top1_changed": 0
    },
    "shuffle_all": {
        "samples": 0,
        "hits": 0,
        "ndcg_sum": 0.0,
        "changed": 0,
        "top1_changed": 0
    },
    "keep_last": {
        "samples": 0,
        "hits": 0,
        "ndcg_sum": 0.0,
        "changed": 0,
        "top1_changed": 0
    }
}


@torch.no_grad()
def get_predictions(X, y):

    X = X.to(device)
    y = y.to(device)

    scores = model(X)

    top10 = torch.topk(
        scores,
        k=TOP_K,
        dim=1
    ).indices

    hits = top10.eq(y.unsqueeze(1))

    recall = hits.any(dim=1).int()

    ndcg = (
        hits.float() * discounts
    ).sum(dim=1)

    return top10, recall, ndcg


print("\n========== 开始序列顺序消融实验 ==========")

for batch_index, (X, y) in enumerate(val_loader):

    X_all_shuffled = shuffle_history(
        X,
        keep_last=False
    )

    X_keep_last = shuffle_history(
        X,
        keep_last=True
    )

    original_top10, original_recall, original_ndcg = (
        get_predictions(X, y)
    )

    all_top10, all_recall, all_ndcg = (
        get_predictions(X_all_shuffled, y)
    )

    last_top10, last_recall, last_ndcg = (
        get_predictions(X_keep_last, y)
    )

    batch_results = {
        "original": (
            original_top10,
            original_recall,
            original_ndcg,
            X
        ),
        "shuffle_all": (
            all_top10,
            all_recall,
            all_ndcg,
            X_all_shuffled
        ),
        "keep_last": (
            last_top10,
            last_recall,
            last_ndcg,
            X_keep_last
        )
    }

    for name, (top10, recall, ndcg, X_modified) in batch_results.items():

        item = experiments[name]

        item["samples"] += X.shape[0]
        item["hits"] += recall.sum().item()
        item["ndcg_sum"] += ndcg.sum().item()

        if name != "original":

            item["changed"] += (
                (X_modified != X).any(dim=1)
                .sum()
                .item()
            )

            item["top1_changed"] += (
                top10[:, 0] != original_top10[:, 0]
            ).sum().item()

    if (batch_index + 1) % 500 == 0:

        count = min(
            (batch_index + 1) * BATCH_SIZE,
            len(val_dataset)
        )

        print(
            f"已评估 {count}/{len(val_dataset)} 条",
            flush=True
        )


print("\n========== 实验结果 ==========")

rows = []

for name, item in experiments.items():

    total = item["samples"]

    recall10 = item["hits"] / total
    ndcg10 = item["ndcg_sum"] / total

    change_rate = item["changed"] / total
    top1_change_rate = item["top1_changed"] / total

    print(f"\n实验: {name}")
    print(f"Recall@10: {recall10:.6f}")
    print(f"NDCG@10:   {ndcg10:.6f}")

    if name != "original":
        print(f"输入实际改变比例: {change_rate:.2%}")
        print(f"Top1预测改变比例: {top1_change_rate:.2%}")

    rows.append({
        "experiment": name,
        "samples": total,
        "recall10": recall10,
        "ndcg10": ndcg10,
        "input_change_rate": change_rate,
        "top1_change_rate": top1_change_rate
    })


csv_path = OUTPUT_DIR / "sequence_order_ablation.csv"

with open(csv_path, "w", newline="") as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "experiment",
            "samples",
            "recall10",
            "ndcg10",
            "input_change_rate",
            "top1_change_rate"
        ]
    )

    writer.writeheader()
    writer.writerows(rows)


print("\n========== 与原始模型指标核对 ==========")

print("原最佳 Recall@10: 0.092906")
print("原最佳 NDCG@10:   0.047798")

print(
    "本次原始 NDCG@10:",
    round(
        experiments["original"]["ndcg_sum"]
        / experiments["original"]["samples"],
        6
    )
)

print("\n实验完成")
print("结果文件:", csv_path)
