from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from semantic_id.algorithms.rq_vae_modules import MLPLayers, ResidualVectorQuantizer


class RQVAEModule(nn.Module):
    """
    Residual Quantization Variational AutoEncoder (RQ-VAE).
    Encodes input vectors into discrete codes using Residual Vector Quantization.

    Reference: reference/rq/models/rqvae.py
    """

    def __init__(
        self,
        in_dim: int = 768,
        num_emb_list: Optional[List[int]] = None,
        e_dim: int = 64,
        layers: Optional[List[int]] = None,
        dropout_prob: float = 0.0,
        bn: bool = False,
        loss_type: str = "mse",
        quant_loss_weight: float = 1.0,
        beta: float = 0.25,
        kmeans_init: bool = False,
        kmeans_iters: int = 100,
        sk_epsilons: Optional[List[float]] = None,
        sk_iters: int = 100,
    ):
        super(RQVAEModule, self).__init__()

        self.in_dim = in_dim
        self.num_emb_list = (
            num_emb_list if num_emb_list is not None else [256, 256, 256, 256]
        )
        self.e_dim = e_dim

        self.layers = layers if layers is not None else [512, 256, 128]
        self.dropout_prob = dropout_prob
        self.bn = bn
        self.loss_type = loss_type
        self.quant_loss_weight = quant_loss_weight
        self.beta = beta
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilons = sk_epsilons
        self.sk_iters = sk_iters

        # Encoder Architecture
        self.encode_layer_dims = [self.in_dim] + self.layers + [self.e_dim]
        self.encoder = MLPLayers(
            layers=self.encode_layer_dims, dropout=self.dropout_prob, bn=self.bn
        )

        # Residual Quantizer
        self.rq = ResidualVectorQuantizer(
            n_e_list=self.num_emb_list,
            e_dim=self.e_dim,
            beta=self.beta,
            kmeans_init=self.kmeans_init,
            kmeans_iters=self.kmeans_iters,
            sk_epsilons=self.sk_epsilons,
            sk_iters=self.sk_iters,
        )

        # Decoder Architecture (Symmetric to Encoder)
        self.decode_layer_dims = self.encode_layer_dims[::-1]
        self.decoder = MLPLayers(
            layers=self.decode_layer_dims, dropout=self.dropout_prob, bn=self.bn
        )

    def forward(
        self, x: torch.Tensor, use_sk: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the VAE.

        Args:
            x: Input tensor (B, D).
            use_sk: Whether to use Sinkhorn algorithm in quantization.

        Returns:
            out: Reconstructed input.
            rq_loss: Quantization loss.
            indices: Discrete codes (B, L).
        """
        x_encoded = self.encoder(x)
        x_q, rq_loss, indices = self.rq(x_encoded, use_sk=use_sk)
        out = self.decoder(x_q)

        return out, rq_loss, indices

    @torch.no_grad()
    def get_indices(self, xs: torch.Tensor, use_sk: bool = False) -> torch.Tensor:
        """
        Encode inputs to discrete indices (inference mode).
        """
        x_e = self.encoder(xs)
        _, _, indices = self.rq(x_e, use_sk=use_sk)
        result: torch.Tensor = indices
        return result

    def compute_loss(
        self, out: torch.Tensor, quant_loss: torch.Tensor, xs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute total loss (Reconstruction + Quantization).
        """
        if self.loss_type == "mse":
            loss_recon = F.mse_loss(out, xs, reduction="mean")
        elif self.loss_type == "l1":
            loss_recon = F.l1_loss(out, xs, reduction="mean")
        else:
            raise ValueError("incompatible loss type")

        loss_total = loss_recon + self.quant_loss_weight * quant_loss

        return loss_total, loss_recon
