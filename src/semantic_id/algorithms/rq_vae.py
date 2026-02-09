import copy
import json
import os
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Optional, Union, Dict, Literal
from tqdm import tqdm

from semantic_id.core import BaseSemanticEncoder, ArrayLike
from semantic_id.algorithms.rq_vae_module import RQVAEModule
from semantic_id.algorithms.rq_kmeans_plus import apply_rqkmeans_plus_strategy


class RQVAE(BaseSemanticEncoder):
    """
    RQ-VAE Semantic Encoder.

    Uses a Residual Quantization Variational AutoEncoder to learn discrete codes
    that reconstruct the input embeddings. Supports the RQ-KMeans+ initialization strategy.
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
        # Training params
        lr: float = 1e-4,
        weight_decay: float = 0.0,
        batch_size: int = 2048,
        epochs: int = 100,
        lr_scheduler: Optional[
            Literal["cosine", "step", "constant_with_warmup"]
        ] = None,
        warmup_epochs: int = 0,
        lr_step_size: int = 50,
        lr_gamma: float = 0.5,
        device: str = "cpu",
        verbose: Union[bool, int] = False,
    ):
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

        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr_scheduler = lr_scheduler
        self.warmup_epochs = warmup_epochs
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma
        self.device = device
        self.verbose = verbose

        # Initialize model
        self.module = RQVAEModule(
            in_dim=self.in_dim,
            num_emb_list=self.num_emb_list,
            e_dim=self.e_dim,
            layers=self.layers,
            dropout_prob=self.dropout_prob,
            bn=self.bn,
            loss_type=self.loss_type,
            quant_loss_weight=self.quant_loss_weight,
            beta=self.beta,
            kmeans_init=self.kmeans_init,
            kmeans_iters=self.kmeans_iters,
            sk_epsilons=self.sk_epsilons,
            sk_iters=self.sk_iters,
        )

        # Training history (populated during fit)
        self.history_: Dict[str, List[float]] = {}

    def _build_scheduler(self, optimizer: torch.optim.Optimizer, steps_per_epoch: int):
        """Build a learning rate scheduler based on configuration."""
        if self.lr_scheduler is None:
            return None

        total_steps = self.epochs * steps_per_epoch
        warmup_steps = self.warmup_epochs * steps_per_epoch

        if self.lr_scheduler == "cosine":
            # Cosine annealing with optional warmup
            if warmup_steps > 0:

                def lr_lambda(step):
                    if step < warmup_steps:
                        return float(step) / float(max(1, warmup_steps))
                    progress = float(step - warmup_steps) / float(
                        max(1, total_steps - warmup_steps)
                    )
                    return 0.5 * (1.0 + np.cos(np.pi * progress))

                return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
            else:
                return torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=total_steps
                )

        elif self.lr_scheduler == "step":
            step_epochs = self.lr_step_size * steps_per_epoch
            return torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=step_epochs, gamma=self.lr_gamma
            )

        elif self.lr_scheduler == "constant_with_warmup":
            if warmup_steps > 0:

                def lr_lambda(step):
                    if step < warmup_steps:
                        return float(step) / float(max(1, warmup_steps))
                    return 1.0

                return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
            return None

        return None

    def _compute_collision_rate(self, codes: torch.Tensor) -> float:
        """Compute collision rate for a batch of codes."""
        N = codes.shape[0]
        # Convert to numpy for unique check
        codes_np = codes.cpu().numpy().astype(np.int32)
        codes_view = np.ascontiguousarray(codes_np).view(
            np.dtype((np.void, codes_np.dtype.itemsize * codes_np.shape[1]))
        )
        n_unique = len(np.unique(codes_view))
        return 1.0 - (n_unique / N)

    def fit(
        self,
        X: ArrayLike,
        *,
        device: str = None,
        pretrained_codebook_path: Optional[str] = None,
    ) -> "RQVAE":
        """
        Train the RQ-VAE model.

        Args:
            X: Input embeddings (N, D).
            device: Training device (overrides init param if provided).
            pretrained_codebook_path: Path to .npz file from RQKMeans for RQ-KMeans+ initialization.
        """
        target_device = device if device is not None else self.device

        # Convert data
        if isinstance(X, np.ndarray):
            X_tensor = torch.from_numpy(X).float()
        else:
            X_tensor = torch.tensor(X, dtype=torch.float)

        # Move model to device
        self.module.to(target_device)
        self.module.train()

        # Determine logging interval
        log_interval = 10
        if isinstance(self.verbose, int) and not isinstance(self.verbose, bool):
            if self.verbose > 0:
                log_interval = self.verbose

        # Setup monitoring batch if verbose
        X_monitor = None
        if self.verbose:
            monitor_size = min(len(X_tensor), 2048)
            indices = torch.randperm(len(X_tensor))[:monitor_size]
            X_monitor = X_tensor[indices].to(target_device)

        # Apply RQ-KMeans+ Strategy if requested
        if pretrained_codebook_path:
            if self.verbose:
                print(f"Applying RQ-KMeans+ strategy using {pretrained_codebook_path}")
            apply_rqkmeans_plus_strategy(
                self.module, pretrained_codebook_path, target_device
            )

        # Training Setup
        optimizer = torch.optim.AdamW(
            self.module.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        dataset = TensorDataset(X_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # LR Scheduler
        scheduler = self._build_scheduler(optimizer, steps_per_epoch=len(dataloader))

        # Checkpointing state
        best_loss = float("inf")
        best_collision_rate = float("inf")
        best_loss_state = None
        best_collision_state = None

        # Training history
        self.history_ = {
            "loss": [],
            "recon_loss": [],
            "collision_rate": [],
            "lr": [],
        }

        # Training Loop
        epoch_iter = range(self.epochs)
        if self.verbose:
            epoch_iter = tqdm(epoch_iter, desc="RQ-VAE training", unit="epoch")

        for epoch in epoch_iter:
            self.module.train()
            total_loss = 0.0
            recon_loss = 0.0

            for batch in dataloader:
                x_batch = batch[0].to(target_device)

                optimizer.zero_grad()

                # Forward
                out, rq_loss, _ = self.module(x_batch)

                # Loss
                loss, r_loss = self.module.compute_loss(out, rq_loss, x_batch)

                loss.backward()
                optimizer.step()

                if scheduler is not None:
                    scheduler.step()

                total_loss += loss.item()
                recon_loss += r_loss.item()

            avg_loss = total_loss / len(dataloader)
            avg_recon = recon_loss / len(dataloader)
            current_lr = optimizer.param_groups[0]["lr"]

            # Record history
            self.history_["loss"].append(avg_loss)
            self.history_["recon_loss"].append(avg_recon)
            self.history_["lr"].append(current_lr)

            # Compute collision rate on monitor batch for checkpointing
            collision_rate = 0.0
            if X_monitor is not None:
                with torch.no_grad():
                    self.module.eval()
                    codes = self.module.get_indices(X_monitor)
                    collision_rate = self._compute_collision_rate(codes)
                    self.module.train()

            self.history_["collision_rate"].append(collision_rate)

            # Checkpointing: save best by loss
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_loss_state = copy.deepcopy(self.module.state_dict())

            # Checkpointing: save best by collision rate
            if collision_rate < best_collision_rate:
                best_collision_rate = collision_rate
                best_collision_state = copy.deepcopy(self.module.state_dict())

            # Update tqdm postfix
            if self.verbose and hasattr(epoch_iter, "set_postfix"):
                postfix = {"loss": f"{avg_loss:.4f}", "recon": f"{avg_recon:.4f}"}
                if X_monitor is not None:
                    postfix["collision"] = f"{collision_rate:.4f}"
                epoch_iter.set_postfix(postfix)

            if self.verbose and (epoch + 1) % log_interval == 0:
                # Calculate Stability Metrics on Monitor Batch
                metrics_str = ""
                if X_monitor is not None:
                    with torch.no_grad():
                        self.module.eval()
                        codes = self.module.get_indices(X_monitor)
                        self.module.train()

                    level_stats = []
                    for i, n_emb in enumerate(self.num_emb_list):
                        level_codes = codes[:, i]
                        # Utilization
                        n_unique = len(torch.unique(level_codes))
                        util_pct = (n_unique / n_emb) * 100

                        # Perplexity
                        counts = torch.bincount(
                            level_codes.long(), minlength=n_emb
                        ).float()
                        probs = counts / counts.sum()
                        perp = torch.exp(
                            -torch.sum(probs * torch.log(probs + 1e-10))
                        ).item()

                        level_stats.append(f"L{i}: {util_pct:.1f}% (Perp: {perp:.1f})")

                    metrics_str = " - " + " | ".join(level_stats)

                lr_str = f" - LR: {current_lr:.2e}" if self.lr_scheduler else ""
                collision_str = (
                    f" - Collision: {collision_rate:.4f}"
                    if X_monitor is not None
                    else ""
                )
                tqdm.write(
                    f"Epoch {epoch+1}/{self.epochs} - "
                    f"Loss: {avg_loss:.4f} (Recon: {avg_recon:.4f})"
                    f"{collision_str}{lr_str}{metrics_str}"
                )

        # Restore best model by collision rate (preferred), else best by loss
        if best_collision_state is not None:
            self.module.load_state_dict(best_collision_state)
            if self.verbose:
                print(
                    f"Restored best model (collision_rate={best_collision_rate:.4f}, "
                    f"best_loss={best_loss:.4f})"
                )
        elif best_loss_state is not None:
            self.module.load_state_dict(best_loss_state)
            if self.verbose:
                print(f"Restored best model (loss={best_loss:.4f})")

        self.module.eval()
        return self

    def encode(
        self, X: ArrayLike, *, device: str = None, batch_size: Optional[int] = None
    ) -> np.ndarray:
        target_device = device if device is not None else self.device
        bs = batch_size if batch_size is not None else self.batch_size

        if isinstance(X, np.ndarray):
            X_tensor = torch.from_numpy(X).float()
        else:
            X_tensor = torch.tensor(X, dtype=torch.float)

        self.module.to(target_device)
        self.module.eval()

        codes_list = []

        with torch.no_grad():
            dataset = TensorDataset(X_tensor)
            dataloader = DataLoader(dataset, batch_size=bs, shuffle=False)

            batch_iter = dataloader
            if self.verbose and len(dataloader) > 1:
                batch_iter = tqdm(dataloader, desc="RQ-VAE encode", unit="batch")

            for batch in batch_iter:
                x_batch = batch[0].to(target_device)
                indices = self.module.get_indices(x_batch)
                codes_list.append(indices.cpu().numpy().astype(np.int32))

        return np.concatenate(codes_list, axis=0)

    # semantic_id() inherited from BaseSemanticEncoder (supports plain and token formats)

    def decode(self, codes: np.ndarray) -> np.ndarray:
        # RQ-VAE decoding from indices involves:
        # 1. Lookup embeddings for indices
        # 2. Sum them (RQ)
        # 3. Pass through Decoder MLP

        self.module.eval()
        # Ensure module is on same device as we expect or CPU
        # For simple decoding, let's assume CPU or current device
        device = next(self.module.parameters()).device

        codes_tensor = torch.from_numpy(codes).to(device)

        with torch.no_grad():
            # 1 & 2: Reconstruct quantized vector (x_q)
            # We need to access the RQ layer
            x_q = 0
            # Assuming codes shape matches layers
            for i, layer in enumerate(self.module.rq.vq_layers):
                indices = codes_tensor[:, i]
                z_q = layer.get_codebook_entry(indices)
                x_q = x_q + z_q

            # 3. Decoder
            out = self.module.decoder(x_q)

        return out.cpu().numpy()

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)

        # Save Metadata
        metadata = {
            "in_dim": self.in_dim,
            "num_emb_list": self.num_emb_list,
            "e_dim": self.e_dim,
            "layers": self.layers,
            "dropout_prob": self.dropout_prob,
            "bn": self.bn,
            "loss_type": self.loss_type,
            "quant_loss_weight": self.quant_loss_weight,
            "beta": self.beta,
            "kmeans_init": self.kmeans_init,
            "kmeans_iters": self.kmeans_iters,
            "sk_epsilons": self.sk_epsilons,
            "sk_iters": self.sk_iters,
            "type": "RQVAE",
        }

        with open(os.path.join(path, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        # Save Model Weights
        torch.save(self.module.state_dict(), os.path.join(path, "model.pt"))

    @classmethod
    def load(cls, path: str, *, device: str = "cpu") -> "RQVAE":
        with open(os.path.join(path, "metadata.json"), "r") as f:
            metadata = json.load(f)

        if metadata.get("type") != "RQVAE":
            raise ValueError(f"Invalid model type: {metadata.get('type')}")

        # Filter metadata to init params
        # (Simplified: assume all metadata keys match init args)
        init_params = {k: v for k, v in metadata.items() if k != "type"}

        instance = cls(**init_params, device=device)

        # Load Weights
        state_dict = torch.load(os.path.join(path, "model.pt"), map_location=device)
        instance.module.load_state_dict(state_dict)
        instance.module.to(device)

        return instance
