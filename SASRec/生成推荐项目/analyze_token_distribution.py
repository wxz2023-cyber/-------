
import numpy as np
import torch

from pathlib import Path

from tiger import TIGER
from tiger_data import (
    prepare_tiger_data,
    TOKEN_START,
    CODEBOOK_SIZE
)


BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR / "checkpoints"
    / "tiger_continue_best.pth"
)

BATCH_SIZE = 128

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

print("当前设备:", device)


# ==========================================
# 1. 加载模型
# ==========================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu",
    weights_only=True
)

model = TIGER(
    **checkpoint["model_config"]
).to(device)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("模型Epoch:", checkpoint["epoch"])


# ==========================================
# 2. 加载验证集
# ==========================================

train_loader, val_loader, test_loader = (
    prepare_tiger_data(
        batch_size=BATCH_SIZE,
        max_history=20
    )
)


# ==========================================
# 3. 统计标签分布和模型预测
# ==========================================

all_labels = []
all_predictions = []

print("\n========== 收集验证集数据 ==========")

with torch.inference_mode():

    for step, batch in enumerate(val_loader, start=1):

        encoder_input = batch[
            "encoder_input"
        ].to(device)

        decoder_input = batch[
            "decoder_input"
        ].to(device)

        labels = batch[
            "labels"
        ].to(device)

        logits = model(
            encoder_input,
            decoder_input
        )

        predictions = torch.argmax(
            logits[:, :4, :],
            dim=-1
        )

        all_labels.append(
            labels[:, :4].cpu().numpy()
        )

        all_predictions.append(
            predictions.cpu().numpy()
        )

        if step % 200 == 0:
            print(
                f"已处理 {step * BATCH_SIZE} 条",
                flush=True
            )

all_labels = np.concatenate(
    all_labels,
    axis=0
)

all_predictions = np.concatenate(
    all_predictions,
    axis=0
)

total_samples = len(all_labels)

print("实际样本数:", total_samples)


# ==========================================
# 4. 分析每个Token的位置
# ==========================================

print("\n========== Token标签分布 ==========")

for position in range(4):

    offset = (
        TOKEN_START
        + position * CODEBOOK_SIZE
    )

    labels = all_labels[:, position] - offset

    predictions = (
        all_predictions[:, position] - offset
    )

    values, counts = np.unique(
        labels,
        return_counts=True
    )

    order = np.argsort(counts)[::-1]

    most_common = values[order[0]]
    most_common_count = counts[order[0]]

    majority_accuracy = (
        most_common_count / total_samples
    )

    model_accuracy = np.mean(
        predictions == labels
    )

    print(f"\n========== Token {position + 1} ==========")

    print("不同真实编号数量:", len(values))

    print("出现最多的编号:", most_common)

    print(
        "该编号出现比例:",
        f"{majority_accuracy:.4%}"
    )

    print(
        "永远预测最热门编号的准确率:",
        f"{majority_accuracy:.4%}"
    )

    print(
        "TIGER真实准确率:",
        f"{model_accuracy:.4%}"
    )

    print(
        "超过热门编号基准:",
        f"{model_accuracy - majority_accuracy:+.4%}"
    )

    print("出现最多的前5个编号:")

    for index in order[:5]:

        value = values[index]
        count = counts[index]

        print(
            f"编号{value}: "
            f"{count}次，"
            f"占比{count / total_samples:.2%}"
        )


# ==========================================
# 5. 重点分析Token 4
# ==========================================

print("\n========== Token 4专项分析 ==========")

offset = TOKEN_START + 3 * CODEBOOK_SIZE

labels4 = all_labels[:, 3] - offset

predictions4 = all_predictions[:, 3] - offset

zero_mask = labels4 == 0
nonzero_mask = labels4 != 0

print(
    "真实Token 4等于0的比例:",
    f"{zero_mask.mean():.4%}"
)

print(
    "真实Token 4不等于0的比例:",
    f"{nonzero_mask.mean():.4%}"
)

print(
    "模型预测Token 4等于0的比例:",
    f"{(predictions4 == 0).mean():.4%}"
)

if zero_mask.sum() > 0:

    zero_accuracy = np.mean(
        predictions4[zero_mask] == labels4[zero_mask]
    )

    print(
        "真实编号为0时的预测准确率:",
        f"{zero_accuracy:.4%}"
    )

if nonzero_mask.sum() > 0:

    nonzero_accuracy = np.mean(
        predictions4[nonzero_mask]
        == labels4[nonzero_mask]
    )

    print(
        "真实编号非0时的预测准确率:",
        f"{nonzero_accuracy:.4%}"
    )

    print(
        "非0样本数量:",
        int(nonzero_mask.sum())
    )


# ==========================================
# 6. 保存统计结果
# ==========================================

OUTPUT_PATH = (
    BASE_DIR / "processed_data"
    / "tiger_token_diagnostics.npz"
)

np.savez(
    OUTPUT_PATH,
    labels=all_labels,
    predictions=all_predictions
)

print("\n========== 分析完成 ==========")

print("结果保存位置:", OUTPUT_PATH)
