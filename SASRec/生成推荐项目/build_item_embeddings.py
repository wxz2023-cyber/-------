
import json
import random
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "processed_data"

INPUT_PATH = DATA_DIR / "item_texts.jsonl"
OUTPUT_PATH = DATA_DIR / "item_embeddings.npy"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

BATCH_SIZE = 32
MAX_TEXT_CHARS = 1500

random.seed(42)
torch.manual_seed(42)


device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

print("当前设备:", device)


print("\n========== 读取商品文本 ==========")

items = []

with open(INPUT_PATH, "r", encoding="utf-8") as f:
    for line in f:
        items.append(json.loads(line))

items.sort(key=lambda x: x["item_id"])

print("商品数量:", len(items))


print("\n========== 检查文本质量 ==========")

valid_items = [
    item for item in items
    if item["text"].strip()
]

missing_items = [
    item for item in items
    if not item["text"].strip()
]

title_only = [
    item for item in valid_items
    if item["text"].strip() == item["title"].strip()
]

text_lengths = [
    len(item["text"])
    for item in valid_items
]

print("有文本商品:", len(valid_items))
print("缺失文本商品:", len(missing_items))
print("只有标题的商品:", len(title_only))

print(
    "只有标题的比例:",
    f"{len(title_only) / max(len(items), 1):.2%}"
)

if text_lengths:
    print("平均文本长度:", round(np.mean(text_lengths), 1))
    print("文本长度中位数:", round(np.median(text_lengths), 1))


print("\n========== 随机查看10件商品 ==========")

examples = random.sample(
    valid_items,
    min(10, len(valid_items))
)

for item in examples:

    print("\n商品ID:", item["item_id"])
    print("商品名称:", item["title"])
    print("商品文本:", item["text"][:200])


print("\n========== 加载文本编码器 ==========")

model = SentenceTransformer(
    MODEL_NAME,
    device=device
)

embedding_dim = model.get_sentence_embedding_dimension()

print("模型名称:", MODEL_NAME)
print("向量维度:", embedding_dim)


print("\n========== 生成语义 Embedding ==========")

texts = [
    item["text"][:MAX_TEXT_CHARS]
    for item in valid_items
]

embeddings = model.encode(
    texts,
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
)

print("有效商品向量形状:", embeddings.shape)


print("\n========== 按商品ID保存 ==========")

max_item_id = max(
    item["item_id"]
    for item in items
)

all_embeddings = np.zeros(
    (max_item_id + 1, embedding_dim),
    dtype=np.float32
)

for item, vector in zip(valid_items, embeddings):
    all_embeddings[item["item_id"]] = vector


np.save(
    OUTPUT_PATH,
    all_embeddings
)

print("最终矩阵形状:", all_embeddings.shape)
print("保存成功:", OUTPUT_PATH)


print("\n========== 检查保存结果 ==========")

loaded = np.load(OUTPUT_PATH)

print("读取后的形状:", loaded.shape)

print(
    "Padding向量是否为0:",
    np.all(loaded[0] == 0)
)

if missing_items:
    print(
        "缺失文本商品ID:",
        [item["item_id"] for item in missing_items[:10]]
    )

print("\n程序运行完成")
