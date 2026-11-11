
import csv
import time
import numpy as np
import torch

from pathlib import Path
from collections import defaultdict


from tiger import TIGER

from tiger_data import (
    prepare_tiger_data,
    BOS_ID,
    TOKEN_START,
    CODEBOOK_SIZE
)


BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR / "checkpoints"
    / "tiger_popularity_best.pth"
)

SEMANTIC_PATH = (
    BASE_DIR / "processed_data"
    / "item_semantic_ids_unique.npy"
)

OUTPUT_PATH = (
    BASE_DIR / "processed_data"
    / "exact_vs_beam_results.csv"
)

BEAM_SIZE = 50

SAMPLES_PER_GROUP = 10

ITEM_BATCH_SIZE = 256

SEED = 42

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

print("当前设备:", device)

start_time = time.time()


# ==========================================
# 1. 加载Semantic ID
# ==========================================

semantic_ids = np.load(SEMANTIC_PATH)

num_items = len(semantic_ids) - 1

item_ids = np.arange(
    1,
    num_items + 1
)

all_codes = []

children = defaultdict(set)
code_to_item = {}

for item_id in item_ids:

    code = semantic_ids[item_id]

    if np.any(code < 0):
        raise ValueError("存在无效Semantic ID")

    tokens = tuple(
        TOKEN_START
        + position * CODEBOOK_SIZE
        + int(value)
        for position, value in enumerate(code)
    )

    if tokens in code_to_item:
        raise ValueError("Semantic ID不唯一")

    code_to_item[tokens] = int(item_id)

    all_codes.append(tokens)

    for position in range(4):
        children[tokens[:position]].add(
            tokens[position]
        )

children = {
    key: np.array(
        sorted(values),
        dtype=np.int64
    )
    for key, values in children.items()
}

all_codes = np.array(
    all_codes,
    dtype=np.int64
)

# 对每个商品，Decoder输入BOS和前3个Token
all_decoder_inputs = np.concatenate(
    [
        np.full(
            (num_items, 1),
            BOS_ID,
            dtype=np.int64
        ),
        all_codes[:, :3]
    ],
    axis=1
)

print("商品数量:", num_items)
print("Semantic ID形状:", all_codes.shape)


# ==========================================
# 2. 加载模型
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
# 3. 加载数据
# ==========================================

train_loader, val_loader, test_loader = (
    prepare_tiger_data(
        batch_size=8,
        max_history=20
    )
)

train_dataset = train_loader.dataset
val_dataset = val_loader.dataset

item_to_id = train_dataset.item_to_id

item_counts = np.zeros(
    num_items + 1,
    dtype=np.int64
)

for record in train_dataset.records:

    asin = str(record["parent_asin"])

    item_id = int(item_to_id[asin])

    item_counts[item_id] += 1


# ==========================================
# 4. 按训练集频率分组
# ==========================================

sorted_items = sorted(
    range(1, num_items + 1),
    key=lambda i: (-item_counts[i], i)
)

head_end = int(num_items * 0.10)
middle_end = int(num_items * 0.50)

item_groups = np.zeros(
    num_items + 1,
    dtype=np.int64
)

for rank, item_id in enumerate(sorted_items):

    if rank < head_end:
        item_groups[item_id] = 0

    elif rank < middle_end:
        item_groups[item_id] = 1

    else:
        item_groups[item_id] = 2

GROUP_NAMES = [
    "热门",
    "中等",
    "长尾"
]


# ==========================================
# 5. 分层随机抽取验证样本
# ==========================================

rng = np.random.default_rng(SEED)

group_indices = [
    [],
    [],
    []
]

for index, record in enumerate(
    val_dataset.records
):

    asin = str(record["parent_asin"])

    target = int(item_to_id[asin])

    group = item_groups[target]

    group_indices[group].append(index)

selected = []

for group in range(3):

    indices = group_indices[group]

    count = min(
        SAMPLES_PER_GROUP,
        len(indices)
    )

    sampled = rng.choice(
        indices,
        size=count,
        replace=False
    )

    for index in sampled:

        selected.append(
            (
                GROUP_NAMES[group],
                int(index)
            )
        )

print("\n========== 抽样结果 ==========")

for group in range(3):

    print(
        GROUP_NAMES[group],
        "原样本:",
        len(group_indices[group]),
        "抽样:",
        min(
            SAMPLES_PER_GROUP,
            len(group_indices[group])
        )
    )

print("合计抽样:", len(selected))


# ==========================================
# 6. 全商品精确打分
# ==========================================

@torch.inference_mode()
def exact_score_all_items(memory, padding_mask):

    all_scores = []

    for start in range(
        0,
        num_items,
        ITEM_BATCH_SIZE
    ):

        end = min(
            start + ITEM_BATCH_SIZE,
            num_items
        )

        decoder_inputs = torch.tensor(
            all_decoder_inputs[start:end],
            dtype=torch.long,
            device=device
        )

        target_codes = torch.tensor(
            all_codes[start:end],
            dtype=torch.long,
            device=device
        )

        count = end - start

        selected_memory = memory.expand(
            count,
            -1,
            -1
        )

        selected_mask = padding_mask.expand(
            count,
            -1
        )

        logits = model.decode(
            decoder_inputs,
            selected_memory,
            selected_mask
        )

        log_probs = torch.log_softmax(
            logits.float(),
            dim=-1
        )

        token_log_probs = log_probs.gather(
            dim=2,
            index=target_codes.unsqueeze(-1)
        ).squeeze(-1)

        scores = token_log_probs.sum(
            dim=1
        )

        all_scores.append(
            scores.cpu().numpy()
        )

    return np.concatenate(
        all_scores
    )


# ==========================================
# 7. Beam Search
# ==========================================

@torch.inference_mode()
def beam_search_one(memory, padding_mask):

    beams = [
        ((), 0.0)
    ]

    for step in range(4):

        decoder_input = torch.tensor(
            [
                [BOS_ID] + list(prefix)
                for prefix, score in beams
            ],
            dtype=torch.long,
            device=device
        )

        count = len(beams)

        selected_memory = memory.expand(
            count,
            -1,
            -1
        )

        selected_mask = padding_mask.expand(
            count,
            -1
        )

        logits = model.decode(
            decoder_input,
            selected_memory,
            selected_mask
        )

        log_probs = torch.log_softmax(
            logits[:, -1, :].float(),
            dim=-1
        ).cpu().numpy()

        candidates = []

        for row_idx, (prefix, score) in enumerate(beams):

            allowed = children[prefix]

            token_scores = log_probs[
                row_idx,
                allowed
            ]

            keep = min(
                BEAM_SIZE,
                len(allowed)
            )

            best_indices = np.argsort(
                token_scores
            )[-keep:][::-1]

            for idx in best_indices:

                token = int(allowed[idx])

                new_score = (
                    score
                    + float(token_scores[idx])
                )

                candidates.append(
                    (
                        prefix + (token,),
                        new_score
                    )
                )

        candidates.sort(
            key=lambda x: x[1],
            reverse=True
        )

        beams = candidates[:BEAM_SIZE]

    results = []

    for prefix, score in beams:

        item_id = code_to_item.get(prefix)

        if item_id is not None:
            results.append(
                (item_id, score)
            )

    return results


# ==========================================
# 8. 逐个用户执行精确打分与Beam Search
# ==========================================

print("\n========== 开始精确打分实验 ==========")

rows = []

for number, (group_name, index) in enumerate(
    selected,
    start=1
):

    sample = val_dataset[index]

    encoder_input = sample[
        "encoder_input"
    ].unsqueeze(0).to(device)

    target = int(
        sample["target_item_id"]
    )

    with torch.inference_mode():

        memory, padding_mask = model.encode(
            encoder_input
        )

        exact_scores = exact_score_all_items(
            memory,
            padding_mask
        )

        beam_results = beam_search_one(
            memory,
            padding_mask
        )

    exact_order = np.argsort(
        -exact_scores,
        kind="stable"
    )

    exact_top50 = (
        item_ids[exact_order[:50]]
        .tolist()
    )

    beam_top50 = [
        item_id
        for item_id, score in beam_results[:50]
    ]

    exact_rank = int(
        np.where(item_ids[exact_order] == target)[0][0]
    ) + 1

    beam_rank = (
        beam_top50.index(target) + 1
        if target in beam_top50
        else 0
    )

    # 检查Beam输出的分数是否与精确打分一致
    beam_score_errors = []

    for item_id, beam_score in beam_results:

        exact_score = exact_scores[item_id - 1]

        beam_score_errors.append(
            abs(beam_score - exact_score)
        )

    max_score_error = (
        max(beam_score_errors)
        if beam_score_errors
        else 0.0
    )

    overlap = len(
        set(exact_top50)
        & set(beam_top50)
    )

    row = {
        "group": group_name,
        "sample_index": index,
        "target_item_id": target,
        "exact_rank": exact_rank,
        "beam_rank": beam_rank,
        "exact_hit_10": int(
            exact_rank <= 10
        ),
        "exact_hit_50": int(
            exact_rank <= 50
        ),
        "beam_hit_10": int(
            0 < beam_rank <= 10
        ),
        "beam_hit_50": int(
            beam_rank > 0
        ),
        "top50_overlap": overlap,
        "max_score_error": max_score_error
    }

    rows.append(row)

    print(
        f"[{number}/{len(selected)}] "
        f"{group_name} | "
        f"目标={target} | "
        f"精确排名={exact_rank} | "
        f"Beam排名={beam_rank if beam_rank else '未进入50'} | "
        f"候选重合={overlap}/50 | "
        f"分数误差={max_score_error:.6f}",
        flush=True
    )


# ==========================================
# 9. 分组统计
# ==========================================

print("\n========== 对比结果 ==========")

def summarize(group_name, records):

    n = len(records)

    if n == 0:
        return

    exact_10 = sum(
        row["exact_hit_10"]
        for row in records
    )

    exact_50 = sum(
        row["exact_hit_50"]
        for row in records
    )

    beam_10 = sum(
        row["beam_hit_10"]
        for row in records
    )

    beam_50 = sum(
        row["beam_hit_50"]
        for row in records
    )

    missed_by_beam = sum(
        row["exact_hit_50"] == 1
        and row["beam_hit_50"] == 0
        for row in records
    )

    average_overlap = np.mean(
        [
            row["top50_overlap"]
            for row in records
        ]
    )

    print(f"\n{group_name}")

    print("样本数:", n)

    print(
        "精确Recall@10:",
        f"{exact_10 / n:.6f}"
    )

    print(
        "精确Recall@50:",
        f"{exact_50 / n:.6f}"
    )

    print(
        "Beam Recall@10:",
        f"{beam_10 / n:.6f}"
    )

    print(
        "Beam Recall@50:",
        f"{beam_50 / n:.6f}"
    )

    print(
        "精确Top50命中但Beam漏掉:",
        missed_by_beam
    )

    print(
        "平均Top50重合数量:",
        f"{average_overlap:.2f}/50"
    )


summarize("整体", rows)

for group_name in GROUP_NAMES:

    records = [
        row
        for row in rows
        if row["group"] == group_name
    ]

    summarize(group_name, records)


# ==========================================
# 10. 保存结果
# ==========================================

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

with open(
    OUTPUT_PATH,
    "w",
    newline="",
    encoding="utf-8-sig"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(rows[0].keys())
    )

    writer.writeheader()
    writer.writerows(rows)

print("\n结果保存至:", OUTPUT_PATH)

print(
    "总用时:",
    f"{(time.time() - start_time) / 60:.2f}分钟"
)

print("实验完成")
