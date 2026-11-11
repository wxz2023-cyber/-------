
import csv
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from sasrec_position import SASRecPosition
from video_games_data import prepare_data


torch.manual_seed(42)

BATCH_SIZE = 32
EPOCHS = 3
PATIENCE = 2
LEARNING_RATE = 0.001

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

print("当前设备:", device)

train_loader, val_loader, item_to_id = prepare_data(
    batch_size=BATCH_SIZE,
    max_len=20,
    train_limit=None,
    val_limit=None
)

model = SASRecPosition(
    num_items=len(item_to_id),
    embedding_dim=64,
    max_len=20,
    num_heads=2,
    num_layers=2,
    dropout=0.1
).to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

base_dir = Path(__file__).resolve().parent
output_dir = base_dir / "checkpoints"

output_dir.mkdir(parents=True, exist_ok=True)

model_path = output_dir / "sasrec_position_best.pth"
csv_path = output_dir / "sasrec_position_metrics.csv"


def train_one_epoch():

    model.train()

    total_loss = 0.0
    total_positions = 0

    start_time = time.time()

    for batch_index, (X, y) in enumerate(train_loader):

        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        h, sequence, lengths = model.forward_all(X)

        targets = torch.zeros_like(sequence)

        targets[:, :-1] = sequence[:, 1:]

        targets.scatter_(
            1,
            (lengths - 1).unsqueeze(1),
            y.unsqueeze(1)
        )

        valid_mask = sequence != 0

        valid_hidden = h[valid_mask]
        valid_targets = targets[valid_mask]

        scores = (
            valid_hidden
            @ model.item_embedding.weight.T
        )

        scores = scores.masked_fill(
            torch.arange(
                scores.shape[1],
                device=device
            ).unsqueeze(0) == 0,
            float("-inf")
        )

        loss = F.cross_entropy(
            scores,
            valid_targets
        )

        if not torch.isfinite(loss).item():
            raise ValueError("训练 Loss 出现 NaN 或 Inf")

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        count = valid_targets.numel()

        total_loss += loss.item() * count
        total_positions += count

        if (batch_index + 1) % 500 == 0:

            elapsed = (time.time() - start_time) / 60

            print(
                f"Batch {batch_index + 1}/{len(train_loader)} | "
                f"平均 Loss: {total_loss / total_positions:.4f} | "
                f"已用时: {elapsed:.2f} 分钟",
                flush=True
            )

    return total_loss / total_positions


@torch.no_grad()
def evaluate():

    model.eval()

    total_loss = 0.0
    total_hits = 0
    total_ndcg = 0.0
    total_samples = 0

    discounts = 1.0 / torch.log2(
        torch.arange(
            1,
            11,
            device=device,
            dtype=torch.float32
        ) + 1
    )

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

        count = X.shape[0]

        total_loss += loss.item() * count
        total_hits += recall.sum().item()
        total_ndcg += ndcg.sum().item()
        total_samples += count

    return (
        total_loss / total_samples,
        total_hits / total_samples,
        total_ndcg / total_samples
    )


best_ndcg = -1.0
best_epoch = 0
no_improvement = 0

history = []

start_time = time.time()

print("\n========== 开始逐位置训练 ==========")

for epoch in range(1, EPOCHS + 1):

    print(f"\n========== Epoch {epoch}/{EPOCHS} ==========")

    train_loss = train_one_epoch()

    print("开始验证...")

    val_loss, recall10, ndcg10 = evaluate()

    print(f"Train Loss: {train_loss:.4f}")
    print(f"Val Loss:   {val_loss:.4f}")
    print(f"Recall@10:  {recall10:.6f}")
    print(f"NDCG@10:    {ndcg10:.6f}")

    history.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "recall10": recall10,
        "ndcg10": ndcg10
    })

    with open(csv_path, "w", newline="") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "train_loss",
                "val_loss",
                "recall10",
                "ndcg10"
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
                "epoch": epoch,
                "best_ndcg": best_ndcg,
                "num_items": len(item_to_id),
                "embedding_dim": 64,
                "max_len": 20,
                "num_heads": 2,
                "num_layers": 2,
                "dropout": 0.1
            },
            model_path
        )

        print("最佳模型保存成功")

    else:

        no_improvement += 1

        if no_improvement >= PATIENCE:
            print("触发 Early Stopping")
            break


elapsed = (time.time() - start_time) / 60

print("\n========== 实验完成 ==========")

print("最佳 Epoch:", best_epoch)
print("最佳 NDCG@10:", round(best_ndcg, 6))
print("总耗时:", round(elapsed, 2), "分钟")
print("模型文件:", model_path)
print("训练记录:", csv_path)

print("\n原 SASRec 最佳验证结果:")
print("Recall@10: 0.092906")
print("NDCG@10:   0.047798")
