
import csv
import time

import torch
import torch.nn.functional as F

from pathlib import Path

from sasrec import SASRec
from video_games_data import prepare_data


torch.manual_seed(42)

TRAIN_LIMIT = None
VAL_LIMIT = None

BATCH_SIZE = 32
EPOCHS = 5

LEARNING_RATE = 0.001
PATIENCE = 2

EMBEDDING_DIM = 64
MAX_LEN = 20
NUM_HEADS = 2
NUM_LAYERS = 2
DROPOUT = 0.1


device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

print("当前设备:", device)


train_loader, val_loader, item_to_id = prepare_data(
    batch_size=BATCH_SIZE,
    max_len=MAX_LEN,
    train_limit=TRAIN_LIMIT,
    val_limit=VAL_LIMIT
)


model = SASRec(
    num_items=len(item_to_id),
    embedding_dim=EMBEDDING_DIM,
    max_len=MAX_LEN,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT
).to(device)


optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


output_dir = Path(__file__).resolve().parent / "checkpoints"
output_dir.mkdir(parents=True, exist_ok=True)

model_path = output_dir / "sasrec_5epoch_best.pth"
csv_path = output_dir / "sasrec_5epoch_metrics.csv"


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
            raise ValueError("训练 Loss 出现 NaN 或 Inf")

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        batch_size = X.shape[0]

        total_loss += loss.item() * batch_size
        total_samples += batch_size

        if (batch_index + 1) % 500 == 0:

            elapsed = time.time() - start_time

            print(
                f"Batch {batch_index + 1}/{len(train_loader)} | "
                f"平均 Loss: {total_loss / total_samples:.4f} | "
                f"耗时: {elapsed / 60:.2f} 分钟",
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
        1,
        11,
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


best_ndcg = -1.0
best_epoch = 0
no_improvement = 0

history = []

start_time = time.time()

print("\n========== 开始5轮训练 ==========")


for epoch in range(1, EPOCHS + 1):

    epoch_start = time.time()

    print(f"\n========== Epoch {epoch}/{EPOCHS} ==========")

    train_loss = train_one_epoch()

    print("开始验证...")

    val_loss, recall10, ndcg10 = evaluate()

    epoch_minutes = (time.time() - epoch_start) / 60

    print(f"\nEpoch {epoch} 训练结果")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Val Loss:   {val_loss:.4f}")
    print(f"Recall@10:  {recall10:.6f}")
    print(f"NDCG@10:    {ndcg10:.6f}")
    print(f"本轮耗时:   {epoch_minutes:.2f} 分钟")

    history.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "recall10": recall10,
        "ndcg10": ndcg10,
        "minutes": epoch_minutes
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
                "embedding_dim": EMBEDDING_DIM,
                "max_len": MAX_LEN,
                "num_heads": NUM_HEADS,
                "num_layers": NUM_LAYERS,
                "dropout": DROPOUT
            },
            model_path
        )

        print("NDCG@10 提高，最佳模型已保存")

    else:

        no_improvement += 1

        print(
            f"NDCG@10 未改善："
            f"{no_improvement}/{PATIENCE}"
        )

        if no_improvement >= PATIENCE:

            print("\n触发 Early Stopping")
            break


total_minutes = (time.time() - start_time) / 60

print("\n========== 训练结束 ==========")

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
print("模型文件:", model_path)
print("指标文件:", csv_path)
