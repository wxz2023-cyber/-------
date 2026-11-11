
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualQuantizer(nn.Module):

    def __init__(
        self,
        embedding_dim=64,
        num_codebooks=3,
        codebook_size=256,
        commitment_weight=0.25
    ):
        super().__init__()

        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.commitment_weight = commitment_weight

        self.codebooks = nn.ModuleList([
            nn.Embedding(codebook_size, embedding_dim)
            for _ in range(num_codebooks)
        ])

        for codebook in self.codebooks:
            nn.init.normal_(
                codebook.weight,
                mean=0.0,
                std=0.05
            )

    def forward(self, z):

        quantized_total = torch.zeros_like(z)

        all_indices = []
        codebook_losses = []
        commitment_losses = []

        for codebook in self.codebooks:

            residual = z - quantized_total.detach()

            weights = codebook.weight

            distances = (
                residual.pow(2).sum(dim=1, keepdim=True)
                + weights.pow(2).sum(dim=1).unsqueeze(0)
                - 2 * residual @ weights.T
            )

            indices = torch.argmin(distances, dim=1)

            quantized = codebook(indices)

            codebook_loss = F.mse_loss(
                quantized,
                residual.detach()
            )

            commitment_loss = F.mse_loss(
                residual,
                quantized.detach()
            )

            codebook_losses.append(codebook_loss)
            commitment_losses.append(commitment_loss)

            quantized_total = quantized_total + quantized

            all_indices.append(indices)

        quantized_st = z + (quantized_total - z).detach()

        semantic_ids = torch.stack(
            all_indices,
            dim=1
        )

        codebook_loss = torch.stack(codebook_losses).mean()

        commitment_loss = torch.stack(commitment_losses).mean()

        quantization_loss = (
            codebook_loss
            + self.commitment_weight * commitment_loss
        )

        return quantized_st, semantic_ids, quantization_loss


class RQVAE(nn.Module):

    def __init__(
        self,
        input_dim=384,
        hidden_dim=256,
        latent_dim=64,
        num_codebooks=3,
        codebook_size=256
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )

        self.quantizer = ResidualQuantizer(
            embedding_dim=latent_dim,
            num_codebooks=num_codebooks,
            codebook_size=codebook_size
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):

        z = self.encoder(x)

        quantized, semantic_ids, quantization_loss = (
            self.quantizer(z)
        )

        reconstructed = self.decoder(quantized)

        reconstruction_loss = F.mse_loss(
            reconstructed,
            x
        )

        total_loss = reconstruction_loss + quantization_loss

        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "quantization_loss": quantization_loss,
            "reconstructed": reconstructed,
            "semantic_ids": semantic_ids
        }

    @torch.no_grad()
    def encode(self, x):

        z = self.encoder(x)

        _, semantic_ids, _ = self.quantizer(z)

        return semantic_ids


if __name__ == "__main__":

    torch.manual_seed(42)

    model = RQVAE()

    x = torch.randn(4, 384)

    output = model(x)

    print("输入形状:", x.shape)
    print("重建形状:", output["reconstructed"].shape)
    print("Semantic ID形状:", output["semantic_ids"].shape)

    print("\nSemantic IDs:")
    print(output["semantic_ids"])

    print("\n总Loss:", output["loss"].item())

    output["loss"].backward()

    print("反向传播成功")
