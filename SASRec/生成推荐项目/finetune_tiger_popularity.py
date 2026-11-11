
import time
import random
import numpy as np
import torch
import torch.nn.functional as F

from pathlib import Path
from tiger import TIGER
from tiger_data import prepare_tiger_data


SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

BASE_DIR = Path(__file__).resolve().parent

OLD_MODEL_PATH = (
    BASE_DIR / "checkpoints"
    / "tiger_popularity_best.pth"
)

BEST_MODEL_PATH = (
    BASE_DIR / "checkpoints"
    / "tiger_popularity_best.pth"
)

LAST_MODEL_PATH = (
    BASE_DIR / "checkpoints"
    / "tiger_popularity_last.pth"
)

EPOCHS = 5
BATCH_SIZE = 64

LEARNING_RATE = 0.00005
WEIGHT_DECAY = 0.01

ALPHA = 0.35
MIN_WEIGHT = 0.5
MAX_WEIGHT = 3.0

PATIENCE = 2
PRINT_EVERY = 200

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

print("当前设备:", device)


# ==========================================
# 1. 加载数据
# ==========================================

print("\n========== 加载TIGER数据 ==========")

train_loader, val_loader, test_loader = (
    prepare_tiger_data(
        batch_size=BATCH_SIZE,
        max_history=20
    )
)

train_dataset = train_loader.dataset

print("训练样本数:", len(train_loader.dataset))
print("验证样本数:", len(val_loader.dataset))


# ==========================================
# 2. 统计训练集商品频率
# ==========================================

print("\n========== 统计商品频率 ==========")

item_to_id = train_dataset.item_to_id

num_items = len(item_to_id)

item_counts = np.zeros(
    num_items + 1,
    dtype=np.int64
)

for record in train_dataset.records:

    asin = str(record["parent_asin"])

    item_id = int(item_to_id[asin])

    item_counts[item_id] += 1

print("训练目标总次数:", item_counts.sum())

print(
    "出现过的不同商品数:",
    np.count_nonzero(item_counts[1:])
)

print(
    "没有出现过的商品数:",
    np.sum(item_counts[1:] == 0)
)


# ==========================================
# 3. 计算流行度权重
# ==========================================

print("\n========== 创建流行度权重 ==========")

positive_counts = item_counts[
    item_counts > 0
]

reference_frequency = float(
    np.median(positive_counts)
)

print("非零商品频率中位数:", reference_frequency)

# 频率越高，权重通常越低
# 频率越低，权重通常越高

frequencies = np.maximum(
    item_counts.astype(np.float64),
    1.0
)

weights = (
    reference_frequency / frequencies
) ** ALPHA

weights = np.clip(
    weights,
    MIN_WEIGHT,
    MAX_WEIGHT
)

# Padding不参与训练
weights[0] = 0.0

# 保证训练集平均样本权重约为1
train_sample_weights = np.repeat(
    weights,
    item_counts
)

mean_weight = train_sample_weights.mean()

weights[1:] /= mean_weight

item_weights = torch.tensor(
    weights,
    dtype=torch.float32,
    device=device
)

print("权重系数 ALPHA:", ALPHA)

print(
    "有效商品权重最小值:",
    weights[1:].min()
)

print(
    "有效商品权重最大值:",
    weights[1:].max()
)

print(
    "训练样本平均权重:",
    np.sum(weights * item_counts) / item_counts.sum()
)


# ==========================================
# 4. 加载Epoch 15模型
# ==========================================

print("\n========== 加载原TIGER ==========")

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

start_epoch = checkpoint["epoch"]

print("原模型Epoch:", start_epoch)
print("原模型验证Loss:", checkpoint["val_loss"])


# ==========================================
# 5. 创建优化器
# ==========================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

# 这是新的加权训练目标
# 为了减少旧优化器动量对新目标的影响
# 这里重新创建优化器，不恢复旧动量


# ==========================================
# 6. 计算单个Batch的Loss
# ==========================================

def calculate_loss(batch):

    encoder_input = batch[
        "encoder_input"
    ].to(device)

    decoder_input = batch[
        "decoder_input"
    ].to(device)

    labels = batch[
        "labels"
    ].to(device)

    target_item_ids = batch[
        "target_item_id"
    ].to(device)

    logits = model(
        encoder_input,
        decoder_input
    )

    # 只取4个Semantic Token位置
    # 第5个EOS位置是确定的，不能提供分类信息

    token_logits = logits[:, :4, :]

    token_labels = labels[:, :4]

    batch_size = token_logits.shape[0]

    token_losses = F.cross_entropy(
        token_logits.reshape(
            -1,
            token_logits.shape[-1]
        ),
        token_labels.reshape(-1),
        reduction="none"
    )

    token_losses = token_losses.reshape(
        batch_size,
        4
    )

    # 每件商品的4个Token损失取平均
    sample_losses = token_losses.mean(dim=1)

    sample_weights = item_weights[
        target_item_ids
    ]

    # 重加权损失
    weighted_loss = (
        sample_losses * sample_weights
    ).sum() / sample_weights.sum().clamp_min(1e-8)

    # 原模型5个位置的平均Loss
    # 由于EOS位置没有预测难度，这里乘4/5
    # 使未加权Loss与旧实验的数值口径接近

    ordinary_loss = sample_losses.mean() * 0.8

    return weighted_loss, ordinary_loss


# ==========================================
# 7. 训练一个Epoch
# ==========================================

def train_one_epoch(epoch):

    model.train()

    weighted_total = 0.0
    ordinary_total = 0.0
    sample_total = 0

    start_time = time.time()

    for step, batch in enumerate(
        train_loader,
        start=1
    ):

        optimizer.zero_grad()

        weighted_loss, ordinary_loss = (
            calculate_loss(batch)
        )

        if not torch.isfinite(weighted_loss).item():
            raise ValueError("训练Loss出现NaN或Inf")

        weighted_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        count = len(batch["target_item_id"])

        weighted_total += weighted_loss.item() * count
        ordinary_total += ordinary_loss.item() * count

        sample_total += count

        if step % PRINT_EVERY == 0:

            elapsed = (
                time.time() - start_time
            ) / 60

            print(
                f"Epoch {epoch} "
                f"Step {step}/{len(train_loader)} "
                f"Weighted Loss: "
                f"{weighted_total / sample_total:.6f} "
                f"Original Loss: "
                f"{ordinary_total / sample_total:.6f} "
                f"耗时: {elapsed:.2f}分钟",
                flush=True
            )

    return (
        weighted_total / sample_total,
        ordinary_total / sample_total
    )


# ==========================================
# 8. 验证模型
# ==========================================

@torch.inference_mode()
def evaluate():

    model.eval()

    weighted_numerator = 0.0
    weighted_denominator = 0.0

    ordinary_total = 0.0
    sample_total = 0

    correct_tokens = 0
    total_tokens = 0

    for batch in val_loader:

        encoder_input = batch[
            "encoder_input"
        ].to(device)

        decoder_input = batch[
            "decoder_input"
        ].to(device)

        labels = batch[
            "labels"
        ].to(device)

        target_item_ids = batch[
            "target_item_id"
        ].to(device)

        logits = model(
            encoder_input,
            decoder_input
        )

        token_logits = logits[:, :4, :]
        token_labels = labels[:, :4]

        count = token_logits.shape[0]

        token_losses = F.cross_entropy(
            token_logits.reshape(
                -1,
                token_logits.shape[-1]
            ),
            token_labels.reshape(-1),
            reduction="none"
        ).reshape(count, 4)

        sample_losses = token_losses.mean(dim=1)

        sample_weights = item_weights[
            target_item_ids
        ]

        weighted_numerator += (
            sample_losses * sample_weights
        ).sum().item()

        weighted_denominator += (
            sample_weights.sum().item()
        )

        ordinary_total += (
            sample_losses.sum().item() * 0.8
        )

        predictions = torch.argmax(
            token_logits,
            dim=-1
        )

        correct_tokens += (
            predictions == token_labels
        ).sum().item()

        total_tokens += count * 4
        sample_total += count

    weighted_loss = (
        weighted_numerator / weighted_denominator
    )

    ordinary_loss = (
        ordinary_total / sample_total
    )

    token_accuracy = (
        correct_tokens / total_tokens
    )

    return (
        weighted_loss,
        ordinary_loss,
        token_accuracy
    )


# ==========================================
# 9. 保存检查点
# ==========================================

def save_checkpoint(
    path,
    epoch,
    train_loss,
    val_weighted_loss,
    val_ordinary_loss
):

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_config": config,
            "train_loss": train_loss,
            "val_loss": val_ordinary_loss,
            "val_weighted_loss": val_weighted_loss,
            "popularity_alpha": ALPHA
        },
        path
    )


# ==========================================
# 10. 先评估原始模型
# ==========================================

print("\n========== 微调前评估 ==========")

baseline_weighted, baseline_ordinary, baseline_acc = (
    evaluate()
)

print(
    "原模型加权Val Loss:",
    f"{baseline_weighted:.6f}"
)

print(
    "原模型普通Val Loss:",
    f"{baseline_ordinary:.6f}"
)

print(
    "原模型Token Accuracy:",
    f"{baseline_acc:.2%}"
)

best_weighted_loss = baseline_weighted

best_epoch = start_epoch

save_checkpoint(
    BEST_MODEL_PATH,
    start_epoch,
    0.0,
    baseline_weighted,
    baseline_ordinary
)


# ==========================================
# 11. 正式微调
# ==========================================

print("\n========== 开始流行度重加权训练 ==========")

patience_count = 0

history = []

total_start = time.time()

for epoch in range(
    start_epoch + 1,
    start_epoch + EPOCHS + 1
):

    print(
        f"\n========== Epoch {epoch} ==========",
        flush=True
    )

    train_weighted, train_ordinary = (
        train_one_epoch(epoch)
    )

    val_weighted, val_ordinary, val_acc = (
        evaluate()
    )

    print(
        "训练加权Loss:",
        f"{train_weighted:.6f}"
    )

    print(
        "训练普通Loss:",
        f"{train_ordinary:.6f}"
    )

    print(
        "验证加权Loss:",
        f"{val_weighted:.6f}"
    )

    print(
        "验证普通Loss:",
        f"{val_ordinary:.6f}"
    )

    print(
        "Token Accuracy:",
        f"{val_acc:.2%}"
    )

    history.append(
        {
            "epoch": epoch,
            "val_weighted": val_weighted,
            "val_ordinary": val_ordinary,
            "token_accuracy": val_acc
        }
    )

    save_checkpoint(
        LAST_MODEL_PATH,
        epoch,
        train_weighted,
        val_weighted,
        val_ordinary
    )

    if val_weighted < best_weighted_loss:

        best_weighted_loss = val_weighted
        best_epoch = epoch

        patience_count = 0

        save_checkpoint(
            BEST_MODEL_PATH,
            epoch,
            train_weighted,
            val_weighted,
            val_ordinary
        )

        print("已保存新的最佳加权模型")

    else:

        patience_count += 1

        print(
            "加权Val Loss未改善:",
            patience_count,
            "/",
            PATIENCE
        )

        if patience_count >= PATIENCE:
            print("触发Early Stopping")
            break


# ==========================================
# 12. 输出总结
# ==========================================

print("\n========== 训练总结 ==========")

for result in history:

    print(
        f"Epoch {result['epoch']} | "
        f"Weighted Val: {result['val_weighted']:.6f} | "
        f"Ordinary Val: {result['val_ordinary']:.6f} | "
        f"Token Acc: {result['token_accuracy']:.2%}"
    )

print("\n最佳Epoch:", best_epoch)

print(
    "最佳加权Val Loss:",
    f"{best_weighted_loss:.6f}"
)

print(
    "总用时:",
    f"{(time.time() - total_start) / 60:.2f}分钟"
)

print("\n最佳模型:", BEST_MODEL_PATH)
print("最后模型:", LAST_MODEL_PATH)

if best_epoch == start_epoch:
    print("加权微调未优于初始模型")

print("\n流行度重加权实验结束")
