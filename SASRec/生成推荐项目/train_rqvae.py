
import time
import numpy as np
import torch

from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

from rqvae import RQVAE


torch.manual_seed(42)
np.random.seed(42)

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "processed_data"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_PATH = DATA_DIR / "item_embeddings.npy"
MODEL_PATH = CHECKPOINT_DIR / "rqvae_best.pth"
SEMANTIC_ID_PATH = DATA_DIR / "item_semantic_ids.npy"

BATCH_SIZE = 128
EPOCHS = 5
LEARNING_RATE = 0.001

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

print("当前设备:", device)


print("\n========== 加载商品向量 ==========")

all_embeddings = np.load(INPUT_PATH)

print("原始矩阵形状:", all_embeddings.shape)

if all_embeddings.shape[1] != 384:
    raise ValueError("商品向量维度不是384")

valid_mask = np.linalg.norm(
    all_embeddings,
    axis=1
) > 1e-8

valid_mask[0] = False

valid_item_ids = np.where(valid_mask)[0]

valid_embeddings = all_embeddings[
    valid_item_ids
].astype(np.float32)

print("有效商品数:", len(valid_item_ids))

train_tensor = torch.from_numpy(
    valid_embeddings
)

train_dataset = TensorDataset(train_tensor)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)


print("\n========== 创建 RQ-VAE ==========")

model = RQVAE(
    input_dim=384,
    hidden_dim=256,
    latent_dim=64,
    num_codebooks=3,
    codebook_size=256
).to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


print("\n========== 开始训练 ==========")

best_loss = float("inf")
start_time = time.time()

for epoch in range(1, EPOCHS + 1):

    model.train()

    total_loss = 0.0
    total_recon = 0.0
    total_quant = 0.0
    total_samples = 0

    for (X,) in train_loader:

        X = X.to(device)

        optimizer.zero_grad()

        output = model(X)

        loss = output["loss"]

        if not torch.isfinite(loss).item():
            raise ValueError("Loss 出现 NaN 或 Inf")

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        count = X.shape[0]

        total_loss += loss.item() * count
        total_recon += (
            output["reconstruction_loss"].item() * count
        )
        total_quant += (
            output["quantization_loss"].item() * count
        )

        total_samples += count

    avg_loss = total_loss / total_samples
    avg_recon = total_recon / total_samples
    avg_quant = total_quant / total_samples

    print(f"\nEpoch {epoch}/{EPOCHS}")
    print(f"Total Loss: {avg_loss:.6f}")
    print(f"Recon Loss: {avg_recon:.6f}")
    print(f"Quant Loss: {avg_quant:.6f}")

    if avg_loss < best_loss:

        best_loss = avg_loss

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "input_dim": 384,
                "hidden_dim": 256,
                "latent_dim": 64,
                "num_codebooks": 3,
                "codebook_size": 256
            },
            MODEL_PATH
        )

        print("已保存最佳训练检查点")


print("\n========== 生成 Semantic ID ==========")

checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu",
    weights_only=True
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

semantic_ids_all = np.full(
    (len(all_embeddings), 3),
    -1,
    dtype=np.int32
)

all_semantic_ids = []

with torch.no_grad():

    for start in range(0, len(valid_embeddings), BATCH_SIZE):

        end = start + BATCH_SIZE

        batch = torch.from_numpy(
            valid_embeddings[start:end]
        ).to(device)

        ids = model.encode(batch)

        all_semantic_ids.append(
            ids.cpu().numpy()
        )

all_semantic_ids = np.concatenate(
    all_semantic_ids,
    axis=0
)

semantic_ids_all[valid_item_ids] = all_semantic_ids

np.save(
    SEMANTIC_ID_PATH,
    semantic_ids_all
)


print("\n========== 检查编码质量 ==========")

unique_codes = np.unique(
    all_semantic_ids,
    axis=0
)

num_unique = len(unique_codes)

print("有效商品数:", len(valid_item_ids))
print("不同Semantic ID数量:", num_unique)

print(
    "商品编码重复数量:",
    len(valid_item_ids) - num_unique
)

print(
    "唯一编码比例:",
    f"{num_unique / len(valid_item_ids):.2%}"
)

for layer in range(3):

    used_codes = len(
        np.unique(all_semantic_ids[:, layer])
    )

    print(
        f"第{layer + 1}层使用的码本编号数:",
        used_codes,
        "/ 256"
    )

print("\n前5件有效商品的编码:")

for item_id in valid_item_ids[:5]:

    print(
        f"商品{item_id}:",
        semantic_ids_all[item_id].tolist()
    )


elapsed = (time.time() - start_time) / 60

print("\n========== 训练完成 ==========")

print("总耗时:", round(elapsed, 2), "分钟")
print("最佳模型:", MODEL_PATH)
print("Semantic ID文件:", SEMANTIC_ID_PATH)
print("编码矩阵形状:", semantic_ids_all.shape)
