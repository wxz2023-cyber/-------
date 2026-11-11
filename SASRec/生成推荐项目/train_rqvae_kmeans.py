
import numpy as np
import torch
import torch.nn.functional as F

from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader
from sklearn.cluster import MiniBatchKMeans

from rqvae import RQVAE


torch.manual_seed(42)
np.random.seed(42)

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "processed_data" / "item_embeddings.npy"

MODEL_PATH = BASE_DIR / "checkpoints" / "rqvae_kmeans_initialized.pth"

ID_PATH = BASE_DIR / "processed_data" / "item_semantic_ids_kmeans.npy"

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 256
WARMUP_EPOCHS = 20
LEARNING_RATE = 0.001
NUM_CODEBOOKS = 3
CODEBOOK_SIZE = 256

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

print("当前设备:", device)


# ==========================================
# 1. 加载商品向量
# ==========================================

print("\n========== 加载数据 ==========")

all_embeddings = np.load(DATA_PATH).astype(np.float32)

valid_mask = np.linalg.norm(
    all_embeddings,
    axis=1
) > 1e-8

valid_mask[0] = False

valid_item_ids = np.where(valid_mask)[0]

X = all_embeddings[valid_item_ids]

print("商品向量矩阵:", all_embeddings.shape)
print("有效商品数量:", len(X))

dataset = TensorDataset(torch.from_numpy(X))

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)


# ==========================================
# 2. 创建模型
# ==========================================

print("\n========== 创建模型 ==========")

model = RQVAE(
    input_dim=384,
    hidden_dim=256,
    latent_dim=64,
    num_codebooks=3,
    codebook_size=256
).to(device)


# ==========================================
# 3. 预训练 Encoder 和 Decoder
# ==========================================

print("\n========== 预训练自编码器 ==========")

optimizer = torch.optim.Adam(
    list(model.encoder.parameters())
    + list(model.decoder.parameters()),
    lr=LEARNING_RATE
)

for epoch in range(1, WARMUP_EPOCHS + 1):

    model.encoder.train()
    model.decoder.train()

    total_loss = 0
    total_samples = 0

    for (batch,) in loader:

        batch = batch.to(device)

        optimizer.zero_grad()

        z = model.encoder(batch)

        reconstructed = model.decoder(z)

        loss = F.mse_loss(reconstructed, batch)

        loss.backward()

        optimizer.step()

        total_loss += loss.item() * len(batch)
        total_samples += len(batch)

    avg_loss = total_loss / total_samples

    if epoch == 1 or epoch % 5 == 0:
        print(
            f"Warmup Epoch {epoch}/{WARMUP_EPOCHS} "
            f"Loss: {avg_loss:.6f}"
        )


# ==========================================
# 4. 计算全部商品的 Encoder 输出
# ==========================================

print("\n========== 生成64维潜在向量 ==========")

model.encoder.eval()

latent_list = []

with torch.no_grad():

    for start in range(0, len(X), BATCH_SIZE):

        batch = torch.from_numpy(
            X[start:start + BATCH_SIZE]
        ).to(device)

        z = model.encoder(batch)

        latent_list.append(
            z.cpu().numpy()
        )

latent_vectors = np.concatenate(
    latent_list,
    axis=0
).astype(np.float32)

print("Encoder输出形状:", latent_vectors.shape)


# ==========================================
# 5. 三层残差 K-Means
# ==========================================

print("\n========== K-Means初始化码本 ==========")

residual = latent_vectors.copy()

kmeans_codes = []

for layer in range(NUM_CODEBOOKS):

    print(f"\n第{layer + 1}层码本初始化")

    kmeans = MiniBatchKMeans(
        n_clusters=CODEBOOK_SIZE,
        random_state=42 + layer,
        batch_size=2048,
        n_init=3,
        max_iter=200
    )

    kmeans.fit(residual)

    centers = kmeans.cluster_centers_.astype(np.float32)

    # 对全部商品重新寻找最近码本向量
    distances = (
        np.sum(residual ** 2, axis=1, keepdims=True)
        + np.sum(centers ** 2, axis=1)[None, :]
        - 2 * residual @ centers.T
    )

    indices = np.argmin(distances, axis=1)

    # 将K-Means中心写入RQ-VAE码本
    with torch.no_grad():

        model.quantizer.codebooks[layer].weight.copy_(
            torch.from_numpy(centers).to(device)
        )

    # 保存这一层码本编号
    kmeans_codes.append(indices)

    # 计算下一层要量化的残差
    residual = residual - centers[indices]

    print(
        "实际使用编号数:",
        len(np.unique(indices)),
        "/",
        CODEBOOK_SIZE
    )

    print(
        "剩余残差均方:",
        float(np.mean(residual ** 2))
    )


# ==========================================
# 6. 使用真正的RQ-VAE量化器重新编码
# ==========================================

print("\n========== 生成Semantic ID ==========")

model.eval()

all_codes = []

with torch.no_grad():

    for start in range(0, len(X), BATCH_SIZE):

        batch = torch.from_numpy(
            X[start:start + BATCH_SIZE]
        ).to(device)

        codes = model.encode(batch)

        all_codes.append(
            codes.cpu().numpy()
        )

all_codes = np.concatenate(
    all_codes,
    axis=0
).astype(np.int32)


# ==========================================
# 7. 检查码本使用情况
# ==========================================

print("\n========== 编码质量检查 ==========")

unique_codes, counts = np.unique(
    all_codes,
    axis=0,
    return_counts=True
)

num_unique = len(unique_codes)
num_items = len(all_codes)

print("有效商品数:", num_items)
print("不同Semantic ID数量:", num_unique)
print("重复编码商品数量:", num_items - num_unique)
print("唯一编码比例:", f"{num_unique / num_items:.2%}")

print("最大编码组商品数:", counts.max())

for layer in range(NUM_CODEBOOKS):

    values, layer_counts = np.unique(
        all_codes[:, layer],
        return_counts=True
    )

    print(
        f"第{layer + 1}层码本使用数量:",
        len(values),
        "/ 256"
    )

    print(
        f"第{layer + 1}层最大编号使用次数:",
        layer_counts.max()
    )


# ==========================================
# 8. 保存完整商品编码
# ==========================================

semantic_ids_all = np.full(
    (len(all_embeddings), NUM_CODEBOOKS),
    -1,
    dtype=np.int32
)

semantic_ids_all[valid_item_ids] = all_codes

np.save(ID_PATH, semantic_ids_all)

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "stage": "kmeans_initialized",
        "input_dim": 384,
        "hidden_dim": 256,
        "latent_dim": 64,
        "num_codebooks": NUM_CODEBOOKS,
        "codebook_size": CODEBOOK_SIZE
    },
    MODEL_PATH
)


# ==========================================
# 9. 输出前10件商品编码
# ==========================================

print("\n========== 前10件商品 ==========")

for item_id in valid_item_ids[:10]:

    print(
        f"商品{item_id}:",
        semantic_ids_all[item_id].tolist()
    )

print("\n========== 完成 ==========")

print("模型保存位置:", MODEL_PATH)
print("Semantic ID保存位置:", ID_PATH)
print("编码矩阵形状:", semantic_ids_all.shape)
