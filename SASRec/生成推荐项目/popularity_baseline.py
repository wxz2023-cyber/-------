
import math
from collections import Counter
from datasets import load_dataset


print("开始加载数据...")

ds = load_dataset(
    "yufan/amazon2023-user-interactions",
    "seq_maxlen20_Video_Games",
    token=True
)

train_data = ds["train"]
val_data = ds["validation"]

print("训练样本数量:", len(train_data))
print("验证样本数量:", len(val_data))


print("\n开始统计商品热门程度...")

item_counter = Counter()

for record in train_data:
    item = record["parent_asin"]
    item_counter[item] += 1


top10_items = item_counter.most_common(10)

print("\n========== 最热门的10件商品 ==========")

for rank, (item, count) in enumerate(top10_items, start=1):
    print(
        f"排名 {rank:02d} | "
        f"商品: {item} | "
        f"交互次数: {count}"
    )


print("\n开始评估 Popularity Baseline...")

top10_ids = [
    item
    for item, count in top10_items
]

rank_map = {
    item: rank
    for rank, item in enumerate(top10_ids, start=1)
}

total_samples = 0
total_hits = 0
total_ndcg = 0.0

for record in val_data:

    true_item = record["parent_asin"]

    total_samples += 1

    if true_item in rank_map:

        rank = rank_map[true_item]

        total_hits += 1

        total_ndcg += 1.0 / math.log2(rank + 1)


recall10 = total_hits / total_samples
ndcg10 = total_ndcg / total_samples


print("\n========== Popularity Baseline ==========")

print("验证样本数:", total_samples)
print("命中样本数:", total_hits)

print(f"Recall@10: {recall10:.6f}")
print(f"NDCG@10:   {ndcg10:.6f}")


print("\n========== 与 SASRec 对比 ==========")

sasrec_recall = 0.059750
sasrec_ndcg = 0.030521

print(
    f"SASRec Recall@10:     {sasrec_recall:.6f}"
)

print(
    f"Popularity Recall@10: {recall10:.6f}"
)

print(
    f"SASRec NDCG@10:       {sasrec_ndcg:.6f}"
)

print(
    f"Popularity NDCG@10:   {ndcg10:.6f}"
)

print("\nRecall@10 差值:", round(sasrec_recall - recall10, 6))
print("NDCG@10 差值:", round(sasrec_ndcg - ndcg10, 6))

if sasrec_recall > recall10 and sasrec_ndcg > ndcg10:
    print("\nSASRec 在两个指标上均超过 Popularity Baseline")
elif sasrec_recall < recall10 and sasrec_ndcg < ndcg10:
    print("\nPopularity Baseline 在两个指标上均超过 SASRec")
else:
    print("\n两个模型各有优势，需要进一步分析")
