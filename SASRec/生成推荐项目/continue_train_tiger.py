
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

OLD_MODEL_PATH = CHECKPOINT_DIR / "tiger_best.pth"

NEW_BEST_PATH = CHECKPOINT_DIR / "tiger_continue_best.pth"

NEW_LAST_PATH = CHECKPOINT_DIR / "tiger_continue_last.pth"

TOTAL_EPOCHS = 15

BATCH_SIZE = 64
LEARNING_RATE = 0.0002

PATIENCE = 3
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

print("\n========== 加载数据 ==========")

train_loader, val_loader, test_loader = (
    prepare_tiger_data(
        batch_size=BATCH_SIZE,
        max_history=20
    )
)

print("训练样本:", len(train_loader.dataset))
print("验证样本:", len(val_loader.dataset))


# ==========================================
# 3. 加载之前训练的模型
# ==========================================

print("\n========== 恢复训练 ==========")

checkpoint = torch.load(
    OLD_MODEL_PATH,
    map_location="cpu",
    weights_only=True
)

config = checkpoint["model_config"]

model = TIGER(**config).to(device)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

start_epoch = checkpoint["epoch"] + 1

best_val_loss = checkpoint["val_loss"]

best_epoch = checkpoint["epoch"]

print("之前训练到Epoch:", checkpoint["epoch"])
print("之前最佳Val Loss:", best_val_loss)
print("本次从Epoch开始:", start_epoch)


# ==========================================
# 4. 恢复优化器
# ==========================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=0.01
)

optimizer.load_state_dict(
    checkpoint["optimizer_state_dict"]
)

# 恢复优化器状态后重新设置学习率
for group in optimizer.param_groups:
    group["lr"] = LEARNING_RATE

print("本次学习率:", LEARNING_RATE)


# ==========================================
# 5. 移动Batch到设备
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
            raise ValueError("训练Loss出现NaN或Inf")

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        count = encoder_input.shape[0]

        total_loss += loss.item() * count
        total_samples += count

        if step % PRINT_EVERY == 0:

            avg_loss = total_loss / total_samples

            elapsed = (
                time.time() - start
            ) / 60

            print(
                f"Epoch {epoch} "
                f"Step {step}/{len(train_loader)} "
                f"Loss: {avg_loss:.6f} "
                f"用时: {elapsed:.2f}分钟",
                flush=True
            )

    return total_loss / total_samples


# ==========================================
# 7. 验证模型
# ==========================================

@torch.no_grad()
def evaluate():

    model.eval()

    total_loss = 0.0
    total_samples = 0

    correct_tokens = 0
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

        count = encoder_input.shape[0]

        total_loss += loss.item() * count
        total_samples += count

        predictions = torch.argmax(
            logits[:, :4, :],
            dim=-1
        )

        correct_tokens += (
            predictions == labels[:, :4]
        ).sum().item()

        total_tokens += count * 4

    val_loss = total_loss / total_samples

    token_accuracy = correct_tokens / total_tokens

    return val_loss, token_accuracy


# ==========================================
# 8. 保存检查点
# ==========================================

def save_checkpoint(path, epoch, train_loss, val_loss):

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "model_config": config
        },
        path
    )


# ==========================================
# 9. 正式继续训练
# ==========================================

print("\n========== 开始继续训练 ==========")

history = []

patience_count = 0

total_start = time.time()

for epoch in range(start_epoch, TOTAL_EPOCHS + 1):

    print(
        f"\n========== Epoch {epoch}/{TOTAL_EPOCHS} ==========",
        flush=True
    )

    train_loss = train_one_epoch(epoch)

    print(
        "训练Loss:",
        f"{train_loss:.6f}",
        flush=True
    )

    print("开始验证...", flush=True)

    val_loss, token_accuracy = evaluate()

    print("验证Loss:", f"{val_loss:.6f}")

    print(
        "Token Accuracy:",
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
        NEW_LAST_PATH,
        epoch,
        train_loss,
        val_loss
    )

    if val_loss < best_val_loss:

        best_val_loss = val_loss
        best_epoch = epoch

        patience_count = 0

        save_checkpoint(
            NEW_BEST_PATH,
            epoch,
            train_loss,
            val_loss
        )

        print("已保存新的最佳模型")

    else:

        patience_count += 1

        print(
            "验证Loss未改善:",
            patience_count,
            "/",
            PATIENCE
        )

        if patience_count >= PATIENCE:

            print("\n触发Early Stopping")
            break

    elapsed = (
        time.time() - total_start
    ) / 60

    print(
        "累计用时:",
        f"{elapsed:.2f}分钟"
    )


# ==========================================
# 10. 输出最终结果
# ==========================================

print("\n========== 继续训练总结 ==========")

for record in history:

    print(
        f"Epoch {record['epoch']} | "
        f"Train Loss: {record['train_loss']:.6f} | "
        f"Val Loss: {record['val_loss']:.6f} | "
        f"Token Acc: {record['token_accuracy']:.2%}"
    )

print("\n最佳Epoch:", best_epoch)

print(
    "最佳验证Loss:",
    f"{best_val_loss:.6f}"
)

print(
    "总用时:",
    f"{(time.time() - total_start) / 60:.2f}分钟"
)

if NEW_BEST_PATH.exists():

    print("\n新最佳模型:")
    print(NEW_BEST_PATH)

else:

    print("\n继续训练未超过原来的最佳模型")
    print("请继续使用:", OLD_MODEL_PATH)

print("\n最后一轮模型:")
print(NEW_LAST_PATH)

print("\n继续训练结束")
