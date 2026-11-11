
import torch
import torch.nn as nn
import torch.nn.functional as F

from tiger_data import (
    VOCAB_SIZE,
    PAD_ID,
    BOS_ID,
    EOS_ID,
    TOKEN_START,
    CODEBOOK_SIZE,
    NUM_TOKENS_PER_ITEM
)


class TIGER(nn.Module):

    def __init__(
        self,
        vocab_size=VOCAB_SIZE,
        d_model=128,
        nhead=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=512,
        dropout=0.1,
        max_encoder_len=80,
        max_decoder_len=5
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_encoder_len = max_encoder_len
        self.max_decoder_len = max_decoder_len

        self.token_embedding = nn.Embedding(
            vocab_size,
            d_model,
            padding_idx=PAD_ID
        )

        self.encoder_position_embedding = nn.Embedding(
            max_encoder_len,
            d_model
        )

        self.decoder_position_embedding = nn.Embedding(
            max_decoder_len,
            d_model
        )

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )

        self.output_layer = nn.Linear(
            d_model,
            vocab_size
        )

        self.dropout = nn.Dropout(dropout)

        allowed = torch.zeros(
            max_decoder_len,
            vocab_size,
            dtype=torch.bool
        )

        for position in range(NUM_TOKENS_PER_ITEM):

            start = (
                TOKEN_START
                + position * CODEBOOK_SIZE
            )

            end = start + CODEBOOK_SIZE

            if position == 3:
                end = start + 12

            allowed[position, start:end] = True

        allowed[NUM_TOKENS_PER_ITEM, EOS_ID] = True

        self.register_buffer(
            "allowed_tokens",
            allowed
        )

    def add_position_embedding(self, tokens, embedding):

        batch_size, seq_len = tokens.shape

        positions = torch.arange(
            seq_len,
            device=tokens.device
        ).unsqueeze(0)

        positions = positions.expand(
            batch_size,
            seq_len
        )

        x = self.token_embedding(tokens)

        x = x + embedding(positions)

        return self.dropout(x)

    def encode(self, encoder_input):

        src = self.add_position_embedding(
            encoder_input,
            self.encoder_position_embedding
        )

        src_padding_mask = (
            encoder_input == PAD_ID
        )

        memory = self.transformer.encoder(
            src,
            src_key_padding_mask=src_padding_mask
        )

        return memory, src_padding_mask

    def decode(
        self,
        decoder_input,
        memory,
        src_padding_mask
    ):

        tgt = self.add_position_embedding(
            decoder_input,
            self.decoder_position_embedding
        )

        tgt_len = decoder_input.shape[1]

        causal_mask = torch.triu(
            torch.ones(
                tgt_len,
                tgt_len,
                dtype=torch.bool,
                device=decoder_input.device
            ),
            diagonal=1
        )

        decoder_output = self.transformer.decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=causal_mask,
            memory_key_padding_mask=src_padding_mask
        )

        logits = self.output_layer(
            decoder_output
        )

        allowed = self.allowed_tokens[:tgt_len]

        logits = logits.masked_fill(
            ~allowed.unsqueeze(0),
            -1e4
        )

        return logits

    def forward(self, encoder_input, decoder_input):

        memory, src_padding_mask = self.encode(
            encoder_input
        )

        logits = self.decode(
            decoder_input,
            memory,
            src_padding_mask
        )

        return logits

    def compute_loss(
        self,
        encoder_input,
        decoder_input,
        labels
    ):

        logits = self.forward(
            encoder_input,
            decoder_input
        )

        loss = F.cross_entropy(
            logits.reshape(-1, self.vocab_size),
            labels.reshape(-1)
        )

        return loss, logits


if __name__ == "__main__":

    torch.manual_seed(42)

    model = TIGER()

    batch_size = 4

    encoder_input = torch.zeros(
        batch_size,
        80,
        dtype=torch.long
    )

    for i in range(batch_size):

        encoder_input[i, -4:] = torch.tensor(
            [4, 262, 517, 774]
        )

    decoder_input = torch.tensor(
        [
            [1, 178, 292, 531, 774],
            [1, 178, 292, 531, 774],
            [1, 178, 292, 531, 774],
            [1, 178, 292, 531, 774]
        ],
        dtype=torch.long
    )

    labels = torch.tensor(
        [
            [178, 292, 531, 774, 2],
            [178, 292, 531, 774, 2],
            [178, 292, 531, 774, 2],
            [178, 292, 531, 774, 2]
        ],
        dtype=torch.long
    )

    print("========== 创建TIGER ==========")

    print(
        "模型参数量:",
        sum(p.numel() for p in model.parameters())
    )

    print("\n========== 前向传播 ==========")

    loss, logits = model.compute_loss(
        encoder_input,
        decoder_input,
        labels
    )

    print("Encoder输入:", encoder_input.shape)

    print("Decoder输入:", decoder_input.shape)

    print("输出Logits:", logits.shape)

    print("Loss:", loss.item())

    print("\n========== 反向传播 ==========")

    loss.backward()

    print("反向传播成功")

    print("\n========== 检查预测 ==========")

    predicted_tokens = torch.argmax(
        logits,
        dim=-1
    )

    print(
        "预测Token:",
        predicted_tokens[0].tolist()
    )

    print(
        "真实Token:",
        labels[0].tolist()
    )

    print("\n模型检查结束")
