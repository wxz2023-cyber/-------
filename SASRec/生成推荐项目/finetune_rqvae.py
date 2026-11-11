
import time
import numpy as np
import torch

from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

from rqvae import RQVAE


torch.manual_seed(42)
np.random.seed(42)

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "processed_data" / "item_embeddings.npy"

INIT_MODEL_PATH = (
    BASE_DIR / "checkpoints" / "rqvae_kmeans_initialized.pth"
)

BEST_MODEL_PATH = (
    BASE_DIR / "checkpoints" / "rqvae_finetuned_best.pth"
)

BEST_ID_PATH = (
    BASE_DIR / "processed_data" / "item_semantic_ids_finetuned.npy"
)

EPOCHS = 10
BATCH_SIZE = 256

ENCODER_LR = 0.00003
CODEBOOK_LR = 0.0001
DECODER_LR = 0.00003

MIN_UNIQUE_RATIO = 0.90

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

print("当前设备:", device)


# ==========================================
# 1. 加载商品向量
# ==========================================

print("\n========== 加载商品向量 ==========")

all_embeddings = np.load(
    DATA_PATH
).astype(np.float32)

valid_mask = np.linalg.norm(
    all_embeddings,
    axis=1
) > 1e-8

valid_mask[0] = False

valid_item_ids = np.where(valid_mask)[0]

X = all_embeddings[valid_item_ids]

print("商品矩阵形状:", all_embeddings.shape)
print("有效商品数量:", len(X))

dataset = TensorDataset(
    torch.from_numpy(X)
)

train_loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

eval_loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ==========================================
# 2. 加载K-Means初始化模型
# ==========================================

print("\n========== 加载初始化模型 ==========")

model = RQVAE(
    input_dim=384,
    hidden_dim=256,
    latent_dim=64,
    num_codebooks=3,
    codebook_size=256
).to(device)

checkpoint = torch.load(
    INIT_MODEL_PATH,
    map_location="cpu",
    weights_only=True
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

print("加载成功:", INIT_MODEL_PATH)


# ==========================================
# 3. 优化器
# ==========================================

optimizer = torch.optim.Adam(
    [
        {
            "params": model.encoder.parameters(),
            "lr": ENCODER_LR
        },
        {
            "params": model.quantizer.parameters(),
            "lr": CODEBOOK_LR
        },
        {
            "params": model.decoder.parameters(),
            "lr": DECODER_LR
        }
    ]
)


# ==========================================
# 4. 评估整个RQ-VAE
# ==========================================

@torch.no_grad()
def evaluate_model():

    model.eval()

    total_recon = 0.0
    total_quant = 0.0
    total_samples = 0

    code_list = []

    for (batch,) in eval_loader:

        batch = batch.to(device)

        output = model(batch)

        count = len(batch)

        total_recon += (
            output["reconstruction_loss"].item() * count
        )

        total_quant += (
            output["quantization_loss"].item() * count
        )

        total_samples += count

        code_list.append(
            output["semantic_ids"].cpu().numpy()
        )

    codes = np.concatenate(
        code_list,
        axis=0
    ).astype(np.int32)

    unique_codes, counts = np.unique(
        codes,
        axis=0,
        return_counts=True
    )

    layer_usage = []

    for layer in range(3):

        used = len(
            np.unique(codes[:, layer])
        )

        layer_usage.append(used)

    return {
        "recon": total_recon / total_samples,
        "quant": total_quant / total_samples,
        "num_unique": len(unique_codes),
        "unique_ratio": len(unique_codes) / len(codes),
        "max_collision": int(counts.max()),
        "layer_usage": layer_usage,
        "codes": codes
    }


# ==========================================
# 5. 保存检查点和Semantic ID
# ==========================================

def save_model(epoch, metrics):

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "input_dim": 384,
            "hidden_dim": 256,
            "latent_dim": 64,
            "num_codebooks": 3,
            "codebook_size": 256,
            "reconstruction_loss": metrics["recon"],
            "num_unique": metrics["num_unique"]
        },
        BEST_MODEL_PATH
    )

    full_ids = np.full(
        (len(all_embeddings), 3),
        -1,
        dtype=np.int32
    )

    full_ids[valid_item_ids] = metrics["codes"]

    np.save(
        BEST_ID_PATH,
        full_ids
    )


# ==========================================
# 6. 评估K-Means初始化结果
# ==========================================

print("\n========== 微调前评估 ==========")

initial = evaluate_model()

initial_unique = initial["num_unique"]

min_allowed_unique = int(
    np.ceil(initial_unique * MIN_UNIQUE_RATIO)
)

print("量化后重建Loss:", f'{initial["recon"]:.6f}')
print("量化Loss:", f'{initial["quant"]:.6f}')
print("不同Semantic ID:", initial_unique)
print("唯一编码比例:", f'{initial["unique_ratio"]:.2%}')
print("各层码本使用量:", initial["layer_usage"])

print(
    "允许的最低不同编码数量:",
    min_allowed_unique
)

best_recon = initial["recon"]
best_epoch = 0

save_model(0, initial)

print("已保存初始化模型作为基准")


# ==========================================
# 7. 联合微调
# ==========================================

print("\n========== 开始联合微调 ==========")

start_time = time.time()

for epoch in range(1, EPOCHS + 1):

    model.train()

    train_total = 0.0
    train_recon = 0.0
    train_quant = 0.0
    train_samples = 0

    for (batch,) in train_loader:

        batch = batch.to(device)

        optimizer.zero_grad()

        output = model(batch)

        loss = output["loss"]

        if not torch.isfinite(loss).item():
            raise ValueError("训练Loss出现NaN或Inf")

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        count = len(batch)

        train_total += loss.item() * count

        train_recon += (
            output["reconstruction_loss"].item() * count
        )

        train_quant += (
            output["quantization_loss"].item() * count
        )

        train_samples += count

    train_total /= train_samples
    train_recon /= train_samples
    train_quant /= train_samples

    metrics = evaluate_model()

    print(f"\n========== Epoch {epoch}/{EPOCHS} ==========")

    print("Train Total Loss:", f"{train_total:.6f}")
    print("Train Recon Loss:", f"{train_recon:.6f}")
    print("Train Quant Loss:", f"{train_quant:.6f}")

    print("完整重建Loss:", f'{metrics["recon"]:.6f}')
    print("不同Semantic ID数量:", metrics["num_unique"])

    print(
        "唯一编码比例:",
        f'{metrics["unique_ratio"]:.2%}'
    )

    print(
        "各层码本使用量:",
        metrics["layer_usage"]
    )

    print(
        "最大编码组商品数:",
        metrics["max_collision"]
    )

    diversity_ok = (
        metrics["num_unique"] >= min_allowed_unique
        and min(metrics["layer_usage"]) >= 200
    )

    reconstruction_better = (
        metrics["recon"] < best_recon
    )

    if diversity_ok and reconstruction_better:

        best_recon = metrics["recon"]
        best_epoch = epoch

        save_model(epoch, metrics)

        print("已保存新的最佳模型")

    else:

        if not diversity_ok:
            print("编码多样性不足，不保存本轮模型")

        else:
            print("重建Loss未改善，不保存本轮模型")


# ==========================================
# 8. 检查最终保存的模型
# ==========================================

print("\n========== 加载最佳模型 ==========")

best_checkpoint = torch.load(
    BEST_MODEL_PATH,
    map_location="cpu",
    weights_only=True
)

model.load_state_dict(
    best_checkpoint["model_state_dict"]
)

best_metrics = evaluate_model()

print("\n========== 最终结果 ==========")

print("最佳Epoch:", best_epoch)

print(
    "最佳重建Loss:",
    f'{best_metrics["recon"]:.6f}'
)

print(
    "不同Semantic ID数量:",
    best_metrics["num_unique"]
)

print(
    "唯一编码比例:",
    f'{best_metrics["unique_ratio"]:.2%}'
)

print(
    "各层码本使用量:",
    best_metrics["layer_usage"]
)

print(
    "最大编码组商品数:",
    best_metrics["max_collision"]
)

print(
    "总耗时(分钟):",
    round((time.time() - start_time) / 60, 2)
)

print("\n模型文件:", BEST_MODEL_PATH)
print("编码文件:", BEST_ID_PATH)

if best_epoch == 0:
    print("\n本次微调未超过初始化基准，保留了初始化模型")

print("\n训练结束")
