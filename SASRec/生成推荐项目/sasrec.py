
import torch
import torch.nn as nn


class SASRecBlock(nn.Module):

    def __init__(self, embedding_dim, num_heads, dropout=0.1):
        super().__init__()

        self.norm1 = nn.LayerNorm(embedding_dim)

        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm2 = nn.LayerNorm(embedding_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 4, embedding_dim)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, causal_mask, padding_mask):

        z = self.norm1(x)

        h, _ = self.attention(
            z, z, z,
            attn_mask=causal_mask,
            key_padding_mask=padding_mask,
            need_weights=False
        )

        x = x + self.dropout(h)

        z = self.norm2(x)

        x = x + self.dropout(self.ffn(z))

        x = x.masked_fill(
            padding_mask.unsqueeze(-1),
            0.0
        )

        return x


class SASRec(nn.Module):

    def __init__(
        self,
        num_items,
        embedding_dim=64,
        max_len=20,
        num_heads=2,
        num_layers=2,
        dropout=0.1
    ):
        super().__init__()

        self.max_len = max_len

        self.item_embedding = nn.Embedding(
            num_items + 1,
            embedding_dim,
            padding_idx=0
        )

        self.position_embedding = nn.Embedding(
            max_len,
            embedding_dim
        )

        self.blocks = nn.ModuleList([
            SASRecBlock(
                embedding_dim,
                num_heads,
                dropout
            )
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

        nn.init.normal_(
            self.item_embedding.weight,
            mean=0.0,
            std=0.02
        )

        nn.init.normal_(
            self.position_embedding.weight,
            mean=0.0,
            std=0.02
        )

        with torch.no_grad():
            self.item_embedding.weight[0].zero_()

    def forward(self, X):

        batch_size, seq_len = X.shape

        if seq_len > self.max_len:
            raise ValueError("序列长度超过 max_len")

        lengths = (X != 0).sum(dim=1)

        if torch.any(lengths == 0):
            raise ValueError("不能输入全为0的序列")

        padding_counts = seq_len - lengths

        indices = (
            torch.arange(
                seq_len,
                device=X.device
            ).unsqueeze(0)
            + padding_counts.unsqueeze(1)
        ) % seq_len

        X = X.gather(1, indices)

        padding_mask = X == 0

        positions = torch.arange(
            seq_len,
            device=X.device
        ).unsqueeze(0)

        x = (
            self.item_embedding(X)
            + self.position_embedding(positions)
        )

        x = self.dropout(x)

        x = x.masked_fill(
            padding_mask.unsqueeze(-1),
            0.0
        )

        causal_mask = torch.triu(
            torch.ones(
                seq_len,
                seq_len,
                dtype=torch.bool,
                device=X.device
            ),
            diagonal=1
        )

        for block in self.blocks:
            x = block(
                x,
                causal_mask,
                padding_mask
            )

        x = self.final_norm(x)

        row_indices = torch.arange(
            batch_size,
            device=X.device
        )

        h = x[row_indices, lengths - 1]

        scores = h @ self.item_embedding.weight.T

        scores = scores.masked_fill(
            torch.arange(
                scores.shape[1],
                device=X.device
            ).unsqueeze(0) == 0,
            float("-inf")
        )

        return scores


if __name__ == "__main__":

    torch.manual_seed(42)

    model = SASRec(
        num_items=25612,
        embedding_dim=64,
        max_len=20,
        num_heads=2,
        num_layers=2
    )

    X = torch.tensor([
        [0, 0, 0, 1, 2, 3],
        [0, 0, 4, 5, 6, 7],
        [1, 2, 3, 4, 5, 6]
    ])

    scores = model(X)

    print("输入形状:", X.shape)
    print("输出形状:", scores.shape)
    print("预测商品:", scores.argmax(dim=1))
    print("是否存在 NaN:", torch.isnan(scores).any().item())

