
from huggingface_hub import get_token
from datasets import load_dataset

print("Hugging Face 登录状态：", get_token() is not None)

print("开始加载 Video_Games 数据集")

ds = load_dataset(
    "yufan/amazon2023-user-interactions",
    "seq_maxlen20_Video_Games",
    token=True
)

print("\n数据集信息：")
print(ds)

print("\n训练集第一条记录：")
print(ds["train"][0])

print("\n训练集字段：")
print(ds["train"].column_names)

print("\n数据划分：")
print(list(ds.keys()))
