
import json
import ast
import numpy as np
import torch

from pathlib import Path
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader


BASE_DIR = Path(__file__).resolve().parent

ITEM_MAP_PATH = BASE_DIR / "processed_data" / "item_to_id.json"

SEMANTIC_ID_PATH = (
    BASE_DIR / "processed_data" / "item_semantic_ids_unique.npy"
)

DATASET_NAME = "yufan/amazon2023-user-interactions"
DATASET_CONFIG = "seq_maxlen20_Video_Games"

MAX_HISTORY = 20
NUM_TOKENS_PER_ITEM = 4

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2

CODEBOOK_SIZE = 257
TOKEN_START = 3

VOCAB_SIZE = (
    TOKEN_START
    + NUM_TOKENS_PER_ITEM * CODEBOOK_SIZE
)


def load_item_mapping():

    with open(ITEM_MAP_PATH, "r", encoding="utf-8") as f:
        item_to_id = json.load(f)

    return item_to_id


def parse_history(history):

    if isinstance(history, str):
        history = ast.literal_eval(history)

    if history is None:
        return []

    return list(history)


class TigerDataset(Dataset):

    def __init__(
        self,
        records,
        item_to_id,
        semantic_ids,
        max_history=20
    ):

        self.records = records
        self.item_to_id = item_to_id
        self.semantic_ids = semantic_ids
        self.max_history = max_history

    def __len__(self):

        return len(self.records)

    def get_item_id(self, item):

        item = str(item)

        if item not in self.item_to_id:
            raise KeyError(
                f"商品 {item} 不存在于 item_to_id.json"
            )

        return int(self.item_to_id[item])

    def encode_item(self, item_id):

        code = self.semantic_ids[item_id]

        if np.any(code < 0):
            raise ValueError(
                f"商品{item_id}没有有效Semantic ID"
            )

        tokens = []

        for position, value in enumerate(code):

            value = int(value)

            if value < 0 or value >= CODEBOOK_SIZE:
                raise ValueError(
                    f"商品{item_id}的第{position + 1}个编号越界"
                )

            token = (
                TOKEN_START
                + position * CODEBOOK_SIZE
                + value
            )

            tokens.append(token)

        return tokens

    def __getitem__(self, index):

        record = self.records[index]

        history = parse_history(
            record["history"]
        )

        history = history[-self.max_history:]

        target_item_id = self.get_item_id(
            record["parent_asin"]
        )

        history_item_ids = [
            self.get_item_id(item)
            for item in history
        ]

        encoder_tokens = []

        for item_id in history_item_ids:

            encoder_tokens.extend(
                self.encode_item(item_id)
            )

        max_encoder_length = (
            self.max_history * NUM_TOKENS_PER_ITEM
        )

        padding_length = (
            max_encoder_length - len(encoder_tokens)
        )

        encoder_tokens = (
            [PAD_ID] * padding_length
            + encoder_tokens
        )

        target_tokens = self.encode_item(
            target_item_id
        )

        decoder_input = (
            [BOS_ID] + target_tokens
        )

        labels = (
            target_tokens + [EOS_ID]
        )

        return {
            "encoder_input": torch.tensor(
                encoder_tokens,
                dtype=torch.long
            ),
            "decoder_input": torch.tensor(
                decoder_input,
                dtype=torch.long
            ),
            "labels": torch.tensor(
                labels,
                dtype=torch.long
            ),
            "target_item_id": torch.tensor(
                target_item_id,
                dtype=torch.long
            )
        }


def prepare_tiger_data(
    batch_size=128,
    max_history=20
):

    print("========== 加载商品映射 ==========")

    item_to_id = load_item_mapping()

    print("商品映射数量:", len(item_to_id))

    print("\n========== 加载Semantic ID ==========")

    semantic_ids = np.load(
        SEMANTIC_ID_PATH
    )

    print("Semantic ID矩阵:", semantic_ids.shape)

    if semantic_ids.shape != (len(item_to_id) + 1, 4):
        raise ValueError(
            "Semantic ID矩阵与商品映射数量不匹配"
        )

    print("\n========== 加载Amazon用户行为 ==========")

    data = load_dataset(
        DATASET_NAME,
        DATASET_CONFIG,
        token=True
    )

    print(data)

    if "validation" in data:
        validation_split = "validation"
    elif "val" in data:
        validation_split = "val"
    else:
        raise ValueError("未找到验证集")

    train_dataset = TigerDataset(
        records=data["train"],
        item_to_id=item_to_id,
        semantic_ids=semantic_ids,
        max_history=max_history
    )

    val_dataset = TigerDataset(
        records=data[validation_split],
        item_to_id=item_to_id,
        semantic_ids=semantic_ids,
        max_history=max_history
    )

    test_dataset = TigerDataset(
        records=data["test"],
        item_to_id=item_to_id,
        semantic_ids=semantic_ids,
        max_history=max_history
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    print("\n========== 数据准备完成 ==========")

    print("训练集:", len(train_dataset))
    print("验证集:", len(val_dataset))
    print("测试集:", len(test_dataset))
    print("TIGER词表大小:", VOCAB_SIZE)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":

    train_loader, val_loader, test_loader = (
        prepare_tiger_data(
            batch_size=4,
            max_history=20
        )
    )

    batch = next(iter(train_loader))

    print("\n========== 检查一个Batch ==========")

    for name, value in batch.items():

        print(
            name,
            "shape:",
            tuple(value.shape)
        )

    print("\n第一条样本的Encoder输入:")
    print(batch["encoder_input"][0].tolist())

    print("\n第一条样本的Decoder输入:")
    print(batch["decoder_input"][0].tolist())

    print("\n第一条样本的训练标签:")
    print(batch["labels"][0].tolist())

    print("\n第一条样本对应的真实商品ID:")
    print(batch["target_item_id"][0].item())

    print("\n数据检查结束")
