
import json

import torch
from pathlib import Path
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader


DATA_DIR = Path(__file__).resolve().parent / "processed_data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


class SequenceDataset(Dataset):

    def __init__(self, data, item_to_id, max_len=20):
        self.data = data
        self.item_to_id = item_to_id
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        record = self.data[index]

        history = record["history"][-self.max_len:]

        x = [
            self.item_to_id[item]
            for item in history
        ]

        x = [0] * (self.max_len - len(x)) + x

        y = self.item_to_id[record["parent_asin"]]

        return (
            torch.tensor(x, dtype=torch.long),
            torch.tensor(y, dtype=torch.long)
        )


def prepare_data(
    batch_size=64,
    max_len=20,
    train_limit=50000,
    val_limit=5000
):

    ds = load_dataset(
        "yufan/amazon2023-user-interactions",
        "seq_maxlen20_Video_Games",
        token=True
    )

    mapping_path = DATA_DIR / "item_to_id.json"

    if mapping_path.exists():

        with open(mapping_path, "r") as f:
            item_to_id = json.load(f)

    else:

        all_items = set()

        for split in ["train", "validation", "test"]:
            for record in ds[split]:
                all_items.add(record["parent_asin"])
                all_items.update(record["history"])

        item_to_id = {
            item: i + 1
            for i, item in enumerate(sorted(all_items))
        }

        with open(mapping_path, "w") as f:
            json.dump(item_to_id, f)

    train_data = ds["train"]
    val_data = ds["validation"]

    if train_limit is not None:
        train_data = train_data.select(
            range(min(train_limit, len(train_data)))
        )

    if val_limit is not None:
        val_data = val_data.select(
            range(min(val_limit, len(val_data)))
        )

    train_dataset = SequenceDataset(
        train_data,
        item_to_id,
        max_len
    )

    val_dataset = SequenceDataset(
        val_data,
        item_to_id,
        max_len
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

    print("商品数量:", len(item_to_id))
    print("训练样本:", len(train_dataset))
    print("验证样本:", len(val_dataset))

    return train_loader, val_loader, item_to_id


if __name__ == "__main__":

    train_loader, val_loader, item_to_id = prepare_data()

    X, y = next(iter(train_loader))

    print("X.shape:", X.shape)
    print("y.shape:", y.shape)

