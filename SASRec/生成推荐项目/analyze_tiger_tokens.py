
import numpy as np
import torch

from pathlib import Path

from tiger import TIGER
from tiger_data import prepare_tiger_data


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


print("\n========== 加载模型 ==========")

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


print("\n========== 加载验证集 ==========")

train_loader, val_loader, test_loader = (
    prepare_tiger_data(
        batch_size=BATCH_SIZE,
        max_history=20
    )
)


correct_per_position = np.zeros(4, dtype=np.int64)

total_samples = 0
all_correct = 0

prefix_correct = np.zeros(4, dtype=np.int64)

print("\n========== 开始分析 ==========")

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

        targets = labels[:, :4]

        correct = (
            predictions == targets
        )

        correct_per_position += (
            correct.sum(dim=0).cpu().numpy()
        )

        all_correct += (
            correct.all(dim=1).sum().item()
        )

        for position in range(4):

            prefix_correct[position] += (
                correct[:, :position + 1]
                .all(dim=1)
                .sum()
                .item()
            )

        total_samples += len(encoder_input)

        if step % 200 == 0:

            print(
                f"已经分析 {total_samples} 条",
                flush=True
            )


print("\n========== 逐Token准确率 ==========")

for position in range(4):

    accuracy = (
        correct_per_position[position]
        / total_samples
    )

    print(
        f"Token {position + 1} Accuracy: "
        f"{accuracy:.4%}"
    )


print("\n========== 前缀准确率 ==========")

for position in range(4):

    accuracy = (
        prefix_correct[position]
        / total_samples
    )

    print(
        f"前{position + 1}个Token全部正确: "
        f"{accuracy:.4%}"
    )


print("\n========== 完整编码准确率 ==========")

print(
    "四个Token全部正确:",
    f"{all_correct / total_samples:.4%}"
)

print("验证样本数量:", total_samples)

print("\n分析完成")
