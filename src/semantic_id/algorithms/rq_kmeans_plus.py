import logging
import os
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn


def get_last_linear_layer(module: nn.Module) -> Optional[nn.Linear]:
    """
    Recursively find the last Linear layer in a module.
    """
    seq = getattr(module, "mlp_layers", None)
    if seq is not None and isinstance(seq, nn.Module):
        modules = list(seq.modules())
    else:
        modules = list(module.modules())

    for m in reversed(modules):
        if isinstance(m, nn.Linear):
            return m
    return None


class ResidualEncoderWrapper(nn.Module):
    """
    Wraps an encoder to provide a residual connection: Z = X + MLP(X).
    Used in RQ-KMeans+ to warm-start from RQ-KMeans (where encoder is effectively identity).
    """

    def __init__(self, original_encoder: nn.Module) -> None:
        super().__init__()
        self.mlp = original_encoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result: torch.Tensor = x + self.mlp(x)
        return result


def apply_rqkmeans_plus_strategy(model: Any, codebook_path: str, device: str) -> Any:
    """
    Applies the RQ-KMeans+ strategy to an RQ-VAE model.

    1. Wraps the encoder with a Residual connection.
    2. Zero-initializes the last layer of the encoder (so initially Z = X).
    3. Loads pre-trained RQ-KMeans codebooks into the VQ layers.

    Args:
        model: RQVAEModule instance.
        codebook_path: Path to .npz file containing codebooks.
        device: Device to load weights onto.

    Returns:
        model: Modified model.
    """
    logging.info(
        ">>> [RQ-KMeans+] Strategy: Applying Residual Connection & Warm-start..."
    )

    # 1. Residual Connection
    if hasattr(model, "encoder"):
        # Ensure we don't double wrap if called multiple times (though fit usually called once)
        if not isinstance(model.encoder, ResidualEncoderWrapper):
            model.encoder = ResidualEncoderWrapper(model.encoder)
            model.encoder.to(device)
            logging.info(
                "    [Structure] Encoder wrapped with Residual Connection (Z = X + MLP(X))"
            )
    else:
        logging.error("    [Error] Could not find 'encoder' in model.")
        return model

    # 2. Zero-Initialization
    logging.info("    [Init] Applying Zero-Initialization to Encoder's last layer...")

    # model.encoder is now ResidualEncoderWrapper, so access .mlp
    raw_mlp = model.encoder.mlp
    last_linear = get_last_linear_layer(raw_mlp)

    if last_linear:
        with torch.no_grad():
            last_linear.weight.fill_(0.0)
            if last_linear.bias is not None:
                last_linear.bias.fill_(0.0)
        logging.info(f"    [Init] Zero-init applied to Linear layer: {last_linear}")
    else:
        logging.warning("    [Warning] Could not find last Linear layer to zero-init.")

    # 3. Load Codebooks
    if not os.path.exists(codebook_path):
        # We raise error here as without codebooks RQ-KMeans+ doesn't make sense
        raise FileNotFoundError(f"{codebook_path} not found")

    logging.info(f"    [Weights] Loading codebooks from {codebook_path}")
    npz_data = np.load(codebook_path)

    target_layers = None
    if hasattr(model, "rq") and hasattr(model.rq, "vq_layers"):
        target_layers = model.rq.vq_layers

    if target_layers:
        success_count = 0
        for i, layer in enumerate(target_layers):
            emb_layer = layer.embedding if hasattr(layer, "embedding") else layer

            # Key format from RQ-KMeans save: "codebook_0", "codebook_1", ...
            key = f"codebook_{i}"
            if key in npz_data:
                centroids = npz_data[key]
                with torch.no_grad():
                    emb_layer.weight.data.copy_(torch.from_numpy(centroids).to(device))
                success_count += 1
                logging.info(f"      -> Loaded Codebook Level {i}")

        if success_count == 0:
            logging.warning("      -> No codebooks loaded! Check .npz keys.")
    else:
        logging.error("    [Error] Could not locate VQ layers.")

    return model
