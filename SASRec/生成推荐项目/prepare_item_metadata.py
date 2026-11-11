
import json
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download


BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "processed_data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

MAPPING_PATH = DATA_DIR / "item_to_id.json"
OUTPUT_PATH = DATA_DIR / "item_texts.jsonl"

REPO_ID = "McAuley-Lab/Amazon-Reviews-2023"

FILE_NAME = (
    "raw_meta_Video_Games/"
    "full-00000-of-00001.parquet"
)

REVISION = "ecfa1c3f76fff4ffd907b68285f6ac0ffea0b5b5"


def clean_text(value):

    if value is None:
        return ""

    if isinstance(value, (list, tuple)):
        return " ".join(
            clean_text(item)
            for item in value
            if item is not None
        ).strip()

    if isinstance(value, str):
        if value.strip().lower() in ["none", "null", "nan"]:
            return ""
        return value.strip()

    return str(value).strip()


if not MAPPING_PATH.exists():
    raise FileNotFoundError(
        f"没有找到商品映射文件: {MAPPING_PATH}"
    )


with open(MAPPING_PATH, "r", encoding="utf-8") as f:
    item_to_id = json.load(f)

print("需要匹配的商品数量:", len(item_to_id))


print("\n========== 下载商品元数据 ==========")

print("仓库:", REPO_ID)
print("文件:", FILE_NAME)
print("版本:", REVISION)

parquet_path = hf_hub_download(
    repo_id=REPO_ID,
    filename=FILE_NAME,
    repo_type="dataset",
    revision=REVISION
)

print("下载成功")
print("本地文件:", parquet_path)


print("\n========== 读取元数据 ==========")

parquet_file = pq.ParquetFile(parquet_path)

print("文件总行数:", parquet_file.metadata.num_rows)
print("可用字段:", parquet_file.schema.names)

columns = [
    "parent_asin",
    "title",
    "description",
    "features",
    "categories"
]

available_columns = parquet_file.schema.names

columns = [
    column
    for column in columns
    if column in available_columns
]

if "parent_asin" not in columns:
    raise ValueError("元数据中缺少 parent_asin 字段")


matched = {}
total_records = 0

for batch in parquet_file.iter_batches(
    batch_size=2048,
    columns=columns
):

    records = batch.to_pylist()

    for record in records:

        total_records += 1

        asin = record.get("parent_asin")

        if asin not in item_to_id:
            continue

        title = clean_text(record.get("title"))
        description = clean_text(record.get("description"))
        features = clean_text(record.get("features"))
        categories = clean_text(record.get("categories"))

        text = " ".join(
            part
            for part in [
                title,
                categories,
                features,
                description
            ]
            if part
        )

        if asin not in matched:

            matched[asin] = {
                "title": title,
                "text": text
            }

        elif len(text) > len(matched[asin]["text"]):

            matched[asin] = {
                "title": title,
                "text": text
            }

    if total_records % 20480 == 0:

        print(
            f"已读取 {total_records} 条 | "
            f"已匹配 {len(matched)} 件商品",
            flush=True
        )


print("\n元数据读取完成")

print("读取记录数:", total_records)
print("匹配商品数:", len(matched))


print("\n========== 保存商品文本 ==========")

missing_count = 0

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:

    for asin, item_id in sorted(
        item_to_id.items(),
        key=lambda x: x[1]
    ):

        info = matched.get(asin, {})

        title = info.get("title", "")
        text = info.get("text", "")

        has_metadata = bool(text.strip())

        if not has_metadata:
            missing_count += 1

        output = {
            "item_id": item_id,
            "parent_asin": asin,
            "title": title,
            "text": text,
            "has_metadata": has_metadata
        }

        f.write(
            json.dumps(
                output,
                ensure_ascii=False
            ) + "\n"
        )


total_items = len(item_to_id)
valid_items = total_items - missing_count

print("\n========== 最终结果 ==========")

print("商品总数:", total_items)
print("成功匹配文本:", valid_items)
print("缺失文本商品:", missing_count)

print(
    "文本覆盖率:",
    f"{valid_items / total_items * 100:.2f}%"
)

print("文件保存位置:", OUTPUT_PATH)


print("\n========== 查看前3件有文本的商品 ==========")

shown = 0

with open(OUTPUT_PATH, "r", encoding="utf-8") as f:

    for line in f:

        item = json.loads(line)

        if not item["has_metadata"]:
            continue

        print("\n商品ID:", item["item_id"])
        print("ASIN:", item["parent_asin"])
        print("商品名称:", item["title"])
        print("商品文本:", item["text"][:300])

        shown += 1

        if shown == 3:
            break


print("\n程序运行完成")
