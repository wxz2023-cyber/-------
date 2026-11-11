
import torch

from sasrec import SASRec


class SASRecPosition(SASRec):

    def forward_all(self, X):

        batch_size, seq_len = X.shape

        lengths = (X != 0).sum(dim=1)

        if torch.any(lengths == 0):
            raise ValueError("不能输入空历史序列")

        if seq_len > self.max_len:
            raise ValueError("序列长度超过 max_len")

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

        h = (
            self.item_embedding(X)
            + self.position_embedding(positions)
        )

        h = self.dropout(h)

        h = h.masked_fill(
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
            h = block(
                h,
                causal_mask,
                padding_mask
            )

        h = self.final_norm(h)

        return h, X, lengths


if __name__ == "__main__":

    model = SASRecPosition(
        num_items=25612,
        embedding_dim=64,
        max_len=20,
        num_heads=2,
        num_layers=2
    )

    X = torch.tensor([
        [0, 0, 0, 1, 2, 3],
        [0, 0, 4, 5, 6, 7]
    ])

    h, sequence, lengths = model.forward_all(X)

    print("隐藏向量形状:", h.shape)
    print("整理后的序列:", sequence)
    print("有效长度:", lengths)

    scores = model(X)

    print("最终位置评分形状:", scores.shape)
