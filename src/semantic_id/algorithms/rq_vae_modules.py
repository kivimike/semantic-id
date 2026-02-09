import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_normal_
from typing import List, Union, Optional

from semantic_id.utils.clustering import (
    kmeans_torch,
    sinkhorn_algorithm,
    center_distance_for_constraint,
)


def activation_layer(activation_name="relu", emb_dim=None):
    """
    Factory for activation functions.
    """
    if activation_name is None:
        return None

    name = activation_name.lower() if isinstance(activation_name, str) else ""

    if name == "sigmoid":
        return nn.Sigmoid()
    elif name == "tanh":
        return nn.Tanh()
    elif name == "relu":
        return nn.ReLU()
    elif name == "leakyrelu":
        return nn.LeakyReLU()
    elif name == "none":
        return None
    elif issubclass(activation_name, nn.Module):
        return activation_name()
    else:
        raise NotImplementedError(
            f"activation function {activation_name} is not implemented"
        )


class MLPLayers(nn.Module):
    """
    Multi-Layer Perceptron used in Encoder and Decoder.
    Reference: reference/rq/models/layers.py
    """

    def __init__(
        self,
        layers: List[int],
        dropout: float = 0.0,
        activation: str = "relu",
        bn: bool = False,
    ):
        super(MLPLayers, self).__init__()
        self.layers = layers
        self.dropout = dropout
        self.activation = activation
        self.use_bn = bn

        mlp_modules = []
        for idx, (input_size, output_size) in enumerate(
            zip(self.layers[:-1], self.layers[1:])
        ):
            mlp_modules.append(nn.Dropout(p=self.dropout))
            mlp_modules.append(nn.Linear(input_size, output_size))

            if self.use_bn and idx != (len(self.layers) - 2):
                mlp_modules.append(nn.BatchNorm1d(num_features=output_size))

            activation_func = activation_layer(self.activation, output_size)
            if activation_func is not None and idx != (len(self.layers) - 2):
                mlp_modules.append(activation_func)

        self.mlp_layers = nn.Sequential(*mlp_modules)
        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    def forward(self, input_feature):
        return self.mlp_layers(input_feature)


class VectorQuantizer(nn.Module):
    """
    Vector Quantization Layer.
    Uses K-Means for initialization and Sinkhorn for balanced assignment (optional).
    Reference: reference/rq/models/vq.py
    """

    def __init__(
        self,
        n_e,
        e_dim,
        beta=0.25,
        kmeans_init=False,
        kmeans_iters=10,
        sk_epsilon=0.003,
        sk_iters=100,
    ):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilon = sk_epsilon
        self.sk_iters = sk_iters

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        if not kmeans_init:
            self.initted = True
            self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        else:
            self.initted = False
            self.embedding.weight.data.zero_()

    def get_codebook(self):
        return self.embedding.weight

    def get_codebook_entry(self, indices, shape=None):
        # get quantized latent vectors
        z_q = self.embedding(indices)
        if shape is not None:
            z_q = z_q.view(shape)

        return z_q

    def init_emb(self, data):
        """
        Initialize embeddings using K-Means on the first batch.
        Uses shared clustering utilities.
        """
        centers = kmeans_torch(
            data,
            num_clusters=self.n_e,
            max_iter=self.kmeans_iters,
            # seed not strictly passed in reference but implied or global
        )
        self.embedding.weight.data.copy_(centers)
        self.initted = True

    def forward(self, x, use_sk=True):
        # Flatten input
        latent = x.view(-1, self.e_dim)

        if not self.initted and self.training:
            self.init_emb(latent)

        # Calculate the L2 Norm between latent and Embedded weights
        # d = ||x||^2 + ||e||^2 - 2*x*e^T
        d = (
            torch.sum(latent**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1, keepdim=True).t()
            - 2 * torch.matmul(latent, self.embedding.weight.t())
        )

        if not use_sk or self.sk_epsilon <= 0:
            indices = torch.argmin(d, dim=-1)
        else:
            # Use shared util for center distance constraint
            # Note: d here corresponds to squared L2 distance
            # (which is what usually goes into Sinkhorn/entropy regularized transport cost)

            # The reference implementation calls center_distance_for_constraint on 'd'
            # d is effectively squared distance if latent and weights are unnormalized?
            # Yes, standard L2 squared expansion.
            d_centered = center_distance_for_constraint(d)

            # Sinkhorn requires float64 for numerical stability (exp(-d/eps)
            # with small eps overflows in float32). MPS doesn't support float64,
            # so we move the computation to CPU for that step.
            original_device = d_centered.device
            try:
                d_centered = d_centered.double()
            except TypeError:
                # MPS and some other backends don't support float64 —
                # move to CPU where double precision is available.
                d_centered = d_centered.cpu().double()

            Q = sinkhorn_algorithm(d_centered, self.sk_epsilon, self.sk_iters)
            Q = Q.to(device=original_device, dtype=d.dtype)

            if torch.isnan(Q).any() or torch.isinf(Q).any():
                print(f"Sinkhorn Algorithm returns nan/inf values.")
            indices = torch.argmax(Q, dim=-1)

        x_q = self.embedding(indices).view(x.shape)

        # compute loss for embedding
        commitment_loss = F.mse_loss(x_q.detach(), x)
        codebook_loss = F.mse_loss(x_q, x.detach())
        loss = codebook_loss + self.beta * commitment_loss

        # preserve gradients (straight-through estimator)
        x_q = x + (x_q - x).detach()

        indices = indices.view(x.shape[:-1])

        return x_q, loss, indices


class ResidualVectorQuantizer(nn.Module):
    """
    Residual Vector Quantizer (RVQ).
    Chains multiple VectorQuantizers to quantize residuals.
    Reference: reference/rq/models/rq.py
    """

    def __init__(
        self,
        n_e_list,
        e_dim,
        sk_epsilons,
        beta=0.25,
        kmeans_init=False,
        kmeans_iters=100,
        sk_iters=100,
    ):
        super().__init__()
        self.n_e_list = n_e_list
        self.e_dim = e_dim
        self.num_quantizers = len(n_e_list)
        self.beta = beta
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilons = sk_epsilons
        self.sk_iters = sk_iters

        # Ensure sk_epsilons matches n_e_list length
        if self.sk_epsilons is None:
            self.sk_epsilons = [0.0] * len(n_e_list)

        self.vq_layers = nn.ModuleList(
            [
                VectorQuantizer(
                    n_e,
                    e_dim,
                    beta=self.beta,
                    kmeans_init=self.kmeans_init,
                    kmeans_iters=self.kmeans_iters,
                    sk_epsilon=sk_epsilon,
                    sk_iters=sk_iters,
                )
                for n_e, sk_epsilon in zip(n_e_list, self.sk_epsilons)
            ]
        )

    def get_codebook(self):
        all_codebook = []
        for quantizer in self.vq_layers:
            codebook = quantizer.get_codebook()
            all_codebook.append(codebook)
        return torch.stack(all_codebook)

    def forward(self, x, use_sk=True):
        all_losses = []
        all_indices = []

        x_q = 0
        residual = x
        for quantizer in self.vq_layers:
            x_res, loss, indices = quantizer(residual, use_sk=use_sk)
            residual = residual - x_res
            x_q = x_q + x_res

            all_losses.append(loss)
            all_indices.append(indices)

        # mean_losses = torch.stack(all_losses).mean() # Original
        # Ensure we don't fail if empty, though VQ list shouldn't be empty
        mean_losses = torch.stack(all_losses).mean()
        all_indices = torch.stack(all_indices, dim=-1)

        return x_q, mean_losses, all_indices
