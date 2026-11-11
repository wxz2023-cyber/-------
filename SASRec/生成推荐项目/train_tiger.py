
import time
import random
import numpy as np
import torch

from pathlib import Path
from tiger import TIGER
from tiger_data import prepare_tiger_data


# ==========================================
# 1. 基础配置
# ==========================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

BASE_DIR = Path(__file__).resolve().parent

CHECKPOINT_DIR = BASE_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = (
    CHECKPOINT_DIR / "tiger_best.pth"
)

LAST_MODEL_PATH = (
    CHECKPOINT_DIR / "tiger_last.pth"
)

EPOCHS = 5
BATCH_SIZE = 64

LEARNING_RATE = 0.0005
WEIGHT_DECAY = 0.01
PATIENCE = 2

PRINT_EVERY = 200

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

print("当前设备:", device)


# ==========================================
# 2. 加载数据
# ==========================================

print("\n========== 加载TIGER数据 ==========")

train_loader, val_loader, test_loader = (
    prepare_tiger_data(
        batch_size=BATCH_SIZE,
        max_history=20
    )
)

print("训练Batch数:", len(train_loader))
print("验证Batch数:", len(val_loader))


# ==========================================
# 3. 创建模型
# ==========================================

print("\n========== 创建TIGER模型 ==========")

model = TIGER(
    d_model=128,
    nhead=4,
    num_encoder_layers=2,
    num_decoder_layers=2,
    dim_feedforward=512,
    dropout=0.1,
    max_encoder_len=80,
    max_decoder_len=5
).to(device)

num_params = sum(
    p.numel()
    for p in model.parameters()
)

print("模型参数量:", num_params)


# ==========================================
# 4. 优化器
# ==========================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


# ==========================================
# 5. 数据移动到计算设备
# ==========================================

def move_batch(batch):

    encoder_input = batch[
        "encoder_input"
    ].to(device)

    decoder_input = batch[
        "decoder_input"
    ].to(device)

    labels = batch[
        "labels"
    ].to(device)

    return encoder_input, decoder_input, labels


# ==========================================
# 6. 训练一个Epoch
# ==========================================

def train_one_epoch(epoch):

    model.train()

    total_loss = 0.0
    total_samples = 0

    start = time.time()

    for step, batch in enumerate(
        train_loader,
        start=1
    ):

        encoder_input, decoder_input, labels = (
            move_batch(batch)
        )

        optimizer.zero_grad()

        loss, logits = model.compute_loss(
            encoder_input,
            decoder_input,
            labels
        )

        if not torch.isfinite(loss).item():
            raise ValueError(
                "Loss出现NaN或Inf"
            )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        batch_size = encoder_input.shape[0]

        total_loss += loss.item() * batch_size
        total_samples += batch_size

        if step % PRINT_EVERY == 0:

            avg_loss = total_loss / total_samples

            elapsed = (
                time.time() - start
            ) / 60

            print(
                f"Epoch {epoch} "
                f"Step {step}/{len(train_loader)} "
                f"Loss: {avg_loss:.6f} "
                f"用时: {elapsed:.1f}分钟",
                flush=True
            )

    avg_loss = total_loss / total_samples

    return avg_loss


# ==========================================
# 7. 验证模型
# ==========================================

@torch.no_grad()
def evaluate():

    model.eval()

    total_loss = 0.0
    total_samples = 0

    correct = 0
    total_tokens = 0

    for batch in val_loader:

        encoder_input, decoder_input, labels = (
            move_batch(batch)
        )

        loss, logits = model.compute_loss(
            encoder_input,
            decoder_input,
            labels
        )

        batch_size = encoder_input.shape[0]

        total_loss += loss.item() * batch_size
        total_samples += batch_size

        predictions = torch.argmax(
            logits,
            dim=-1
        )

        # 只统计4个商品编码位置
        # 第5个位置在当前模型中只能输出EOS

        correct += (
            predictions[:, :4] == labels[:, :4]
        ).sum().item()

        total_tokens += batch_size * 4

    avg_loss = total_loss / total_samples

    token_accuracy = (
        correct / total_tokens
    )

    return avg_loss, token_accuracy


# ==========================================
# 8. 保存模型
# ==========================================

def save_checkpoint(path, epoch, train_loss, val_loss):

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "model_config": {
                "d_model": 128,
                "nhead": 4,
                "num_encoder_layers": 2,
                "num_decoder_layers": 2,
                "dim_feedforward": 512,
                "dropout": 0.1,
                "max_encoder_len": 80,
                "max_decoder_len": 5
            }
        },
        path
    )


# ==========================================
# 9. 正式训练
# ==========================================

print("\n========== 开始训练TIGER ==========")

best_val_loss = float("inf")
best_epoch = 0

patience_count = 0

history = []

total_start = time.time()

for epoch in range(1, EPOCHS + 1):

    print(
        f"\n========== Epoch {epoch}/{EPOCHS} ==========",
        flush=True
    )

    train_loss = train_one_epoch(epoch)

    print(
        "\n训练Loss:",
        f"{train_loss:.6f}",
        flush=True
    )

    print(
        "开始验证...",
        flush=True
    )

    val_loss, token_accuracy = evaluate()

    print(
        "验证Loss:",
        f"{val_loss:.6f}"
    )

    print(
        "验证Token准确率:",
        f"{token_accuracy:.2%}"
    )

    history.append(
        {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "token_accuracy": token_accuracy
        }
    )

    save_checkpoint(
        LAST_MODEL_PATH,
        epoch,
        train_loss,
        val_loss
    )

    if val_loss < best_val_loss:

        best_val_loss = val_loss
        best_epoch = epoch

        patience_count = 0

        save_checkpoint(
            BEST_MODEL_PATH,
            epoch,
            train_loss,
            val_loss
        )

        print("已保存新的最佳模型")

    else:

        patience_count += 1

        print(
            "验证Loss未改善，连续次数:",
            patience_count
        )

        if patience_count >= PATIENCE:

            print("\n触发Early Stopping")
            break

    elapsed = (
        time.time() - total_start
    ) / 60

    print(
        "累计训练时间:",
        f"{elapsed:.2f}分钟"
    )


# ==========================================
# 10. 输出训练总结
# ==========================================

print("\n========== 训练总结 ==========")

for item in history:

    print(
        f"Epoch {item['epoch']} | "
        f"Train Loss: {item['train_loss']:.6f} | "
        f"Val Loss: {item['val_loss']:.6f} | "
        f"Token Acc: {item['token_accuracy']:.2%}"
    )

print("\n最佳Epoch:", best_epoch)

print(
    "最佳验证Loss:",
    f"{best_val_loss:.6f}"
)

print(
    "总训练时间:",
    f"{(time.time() - total_start) / 60:.2f}分钟"
)

print("\n最佳模型保存位置:")
print(BEST_MODEL_PATH)

print("\n最后一轮模型保存位置:")
print(LAST_MODEL_PATH)

print("\nTIGER训练完成")
