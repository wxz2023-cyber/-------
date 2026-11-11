
import csv
import time

import torch
import torch.nn.functional as F

from pathlib import Path

from sasrec import SASRec
from video_games_data import prepare_data


torch.manual_seed(42)

BATCH_SIZE = 32
MAX_EPOCH = 15
PATIENCE = 3
LEARNING_RATE = 0.0005

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

print("当前设备:", device)

base_dir = Path(__file__).resolve().parent
checkpoint_dir = base_dir / "checkpoints"

old_model_path = checkpoint_dir / "sasrec_5epoch_best.pth"
new_model_path = checkpoint_dir / "sasrec_15epoch_best.pth"
csv_path = checkpoint_dir / "sasrec_continue_metrics.csv"

if not old_model_path.exists():
    raise FileNotFoundError(
        f"没有找到之前训练的模型: {old_model_path}"
    )

checkpoint = torch.load(
    old_model_path,
    map_location="cpu",
    weights_only=True
)

start_epoch = checkpoint["epoch"]
best_ndcg = checkpoint["best_ndcg"]

print("成功加载第", start_epoch, "轮模型")
print("原最佳 NDCG@10:", best_ndcg)

train_loader, val_loader, item_to_id = prepare_data(
    batch_size=BATCH_SIZE,
    max_len=checkpoint["max_len"],
    train_limit=None,
    val_limit=None
)

if len(item_to_id) != checkpoint["num_items"]:
    raise ValueError("商品数量与原模型不一致")

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

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

optimizer.load_state_dict(
    checkpoint["optimizer_state_dict"]
)

for group in optimizer.param_groups:
    group["lr"] = LEARNING_RATE

print("模型参数恢复成功")
print("优化器状态恢复成功")
print("当前学习率:", optimizer.param_groups[0]["lr"])


def train_one_epoch():

    model.train()

    total_loss = 0.0
    total_samples = 0

    start_time = time.time()

    for batch_index, (X, y) in enumerate(train_loader):

        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        scores = model(X)

        loss = F.cross_entropy(scores, y)

        if not torch.isfinite(loss).item():
            raise ValueError("Loss 出现 NaN 或 Inf")

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        batch_size = X.shape[0]

        total_loss += loss.item() * batch_size
        total_samples += batch_size

        if (batch_index + 1) % 1000 == 0:

            elapsed = (time.time() - start_time) / 60

            print(
                f"Batch {batch_index + 1}/{len(train_loader)} | "
                f"平均 Loss: {total_loss / total_samples:.4f} | "
                f"耗时: {elapsed:.2f} 分钟",
                flush=True
            )

    return total_loss / total_samples


@torch.no_grad()
def evaluate():

    model.eval()

    total_loss = 0.0
    total_recall = 0.0
    total_ndcg = 0.0
    total_samples = 0

    positions = torch.arange(
        1, 11,
        device=device,
        dtype=torch.float32
    )

    discounts = 1.0 / torch.log2(positions + 1)

    for X, y in val_loader:

        X = X.to(device)
        y = y.to(device)

        scores = model(X)

        loss = F.cross_entropy(scores, y)

        top10 = torch.topk(
            scores,
            k=10,
            dim=1
        ).indices

        hits = top10.eq(y.unsqueeze(1))

        recall = hits.any(dim=1).float()

        ndcg = (
            hits.float() * discounts
        ).sum(dim=1)

        batch_size = X.shape[0]

        total_loss += loss.item() * batch_size
        total_recall += recall.sum().item()
        total_ndcg += ndcg.sum().item()
        total_samples += batch_size

    return (
        total_loss / total_samples,
        total_recall / total_samples,
        total_ndcg / total_samples
    )


history = []
no_improvement = 0
best_epoch = start_epoch

start_time = time.time()

print("\n========== 开始继续训练 ==========")


for epoch in range(start_epoch + 1, MAX_EPOCH + 1):

    epoch_start = time.time()

    print(f"\n========== Epoch {epoch}/{MAX_EPOCH} ==========")

    train_loss = train_one_epoch()

    print("开始验证...")

    val_loss, recall10, ndcg10 = evaluate()

    minutes = (time.time() - epoch_start) / 60

    print(f"\nEpoch {epoch} 结果")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Val Loss:   {val_loss:.4f}")
    print(f"Recall@10:  {recall10:.6f}")
    print(f"NDCG@10:    {ndcg10:.6f}")
    print(f"本轮耗时:   {minutes:.2f} 分钟")

    history.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "recall10": recall10,
        "ndcg10": ndcg10,
        "minutes": minutes
    })

    with open(csv_path, "w", newline="") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "train_loss",
                "val_loss",
                "recall10",
                "ndcg10",
                "minutes"
            ]
        )

        writer.writeheader()
        writer.writerows(history)

    if ndcg10 > best_ndcg:

        best_ndcg = ndcg10
        best_epoch = epoch
        no_improvement = 0

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "best_ndcg": best_ndcg,
                "num_items": len(item_to_id),
                "embedding_dim": checkpoint["embedding_dim"],
                "max_len": checkpoint["max_len"],
                "num_heads": checkpoint["num_heads"],
                "num_layers": checkpoint["num_layers"],
                "dropout": checkpoint["dropout"]
            },
            new_model_path
        )

        print("NDCG@10 提高！已保存新的最佳模型")

    else:

        no_improvement += 1

        print(
            f"NDCG@10 未提高: "
            f"{no_improvement}/{PATIENCE}"
        )

        if no_improvement >= PATIENCE:
            print("触发 Early Stopping")
            break


total_minutes = (time.time() - start_time) / 60

print("\n========== 继续训练结束 ==========")

for record in history:

    print(
        f"Epoch {record['epoch']} | "
        f"Train Loss: {record['train_loss']:.4f} | "
        f"Val Loss: {record['val_loss']:.4f} | "
        f"Recall@10: {record['recall10']:.6f} | "
        f"NDCG@10: {record['ndcg10']:.6f}"
    )

print(f"\n最佳 Epoch: {best_epoch}")
print(f"最佳 NDCG@10: {best_ndcg:.6f}")
print(f"总耗时: {total_minutes:.2f} 分钟")

if new_model_path.exists():
    print("新最佳模型:", new_model_path)
else:
    print("新训练没有超过第5轮，继续使用原模型:", old_model_path)

print("训练记录:", csv_path)
