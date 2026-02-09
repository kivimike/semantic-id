"""Tests for RQVAE training features: LR scheduler, checkpointing, history."""

import numpy as np

from semantic_id.algorithms.rq_vae import RQVAE


def _make_data(n=100, d=16):
    return np.random.randn(n, d).astype(np.float32)


def test_rq_vae_training_history():
    """Verify training history is populated after fit."""
    X = _make_data()
    model = RQVAE(
        in_dim=16,
        num_emb_list=[8, 8],
        e_dim=8,
        layers=[16],
        batch_size=32,
        epochs=5,
        device="cpu",
        verbose=True,
    )
    model.fit(X)

    assert "loss" in model.history_
    assert "recon_loss" in model.history_
    assert "collision_rate" in model.history_
    assert "lr" in model.history_
    assert len(model.history_["loss"]) == 5


def test_rq_vae_cosine_scheduler():
    """Verify cosine LR scheduler produces decreasing LR."""
    X = _make_data()
    model = RQVAE(
        in_dim=16,
        num_emb_list=[8, 8],
        e_dim=8,
        layers=[16],
        batch_size=32,
        epochs=10,
        lr=1e-3,
        lr_scheduler="cosine",
        device="cpu",
    )
    model.fit(X)

    lrs = model.history_["lr"]
    # Cosine schedule should generally decrease
    assert lrs[-1] < lrs[0]


def test_rq_vae_step_scheduler():
    """Verify step LR scheduler reduces LR at step boundaries."""
    X = _make_data()
    model = RQVAE(
        in_dim=16,
        num_emb_list=[8, 8],
        e_dim=8,
        layers=[16],
        batch_size=32,
        epochs=10,
        lr=1e-3,
        lr_scheduler="step",
        lr_step_size=3,
        lr_gamma=0.5,
        device="cpu",
    )
    model.fit(X)

    lrs = model.history_["lr"]
    # After step_size epochs worth of steps, LR should have dropped
    assert lrs[-1] < lrs[0]


def test_rq_vae_constant_warmup_scheduler():
    """Verify constant_with_warmup starts low and ramps up."""
    X = _make_data()
    model = RQVAE(
        in_dim=16,
        num_emb_list=[8, 8],
        e_dim=8,
        layers=[16],
        batch_size=32,
        epochs=10,
        lr=1e-3,
        lr_scheduler="constant_with_warmup",
        warmup_epochs=3,
        device="cpu",
    )
    model.fit(X)

    lrs = model.history_["lr"]
    # First epoch LR should be lower than last (warmup)
    assert lrs[0] < lrs[-1]


def test_rq_vae_no_scheduler():
    """Verify that without scheduler, LR stays constant."""
    X = _make_data()
    model = RQVAE(
        in_dim=16,
        num_emb_list=[8, 8],
        e_dim=8,
        layers=[16],
        batch_size=32,
        epochs=5,
        lr=1e-3,
        lr_scheduler=None,
        device="cpu",
    )
    model.fit(X)

    lrs = model.history_["lr"]
    assert all(abs(lr - 1e-3) < 1e-8 for lr in lrs)


def test_rq_vae_checkpointing():
    """Verify that best model is restored after training."""
    X = _make_data(n=50)
    model = RQVAE(
        in_dim=16,
        num_emb_list=[4, 4],
        e_dim=8,
        layers=[16],
        batch_size=25,
        epochs=5,
        lr=1e-3,
        device="cpu",
        verbose=True,
    )
    model.fit(X)

    # After fit, the model should be in eval mode (best checkpoint restored)
    assert not model.module.training

    # The model should still produce valid codes
    codes = model.encode(X)
    assert codes.shape == (50, 2)
    assert codes.dtype == np.int32


def test_rq_vae_weight_decay():
    """Verify weight_decay parameter is accepted."""
    X = _make_data(n=30)
    model = RQVAE(
        in_dim=16,
        num_emb_list=[4, 4],
        e_dim=8,
        layers=[16],
        batch_size=15,
        epochs=2,
        lr=1e-3,
        weight_decay=0.01,
        device="cpu",
    )
    model.fit(X)
    codes = model.encode(X)
    assert codes.shape == (30, 2)
