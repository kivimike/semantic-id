"""End-to-end integration tests."""

import numpy as np
import pytest

from semantic_id import RQVAE, RQKMeans, SemanticIdEngine, evaluate


def test_full_pipeline_rq_kmeans(tmp_path):
    """fit -> encode -> unique_ids -> save -> load -> encode consistency."""
    rng = np.random.RandomState(42)
    X = rng.randn(60, 8).astype(np.float32)

    encoder = RQKMeans(n_levels=2, n_clusters=5, random_state=42)
    engine = SemanticIdEngine(encoder=encoder)
    engine.fit(X)

    ids_before = engine.unique_ids(X, fmt="plain")
    assert len(ids_before) == 60
    assert all(isinstance(i, str) for i in ids_before)

    save_path = str(tmp_path / "engine")
    engine.save(save_path)
    loaded = SemanticIdEngine.load(save_path)

    ids_after = loaded.unique_ids(X, fmt="plain")
    assert ids_before == ids_after


def test_evaluate_with_encoder():
    """evaluate() computes quantization_mse when encoder supports decode."""
    rng = np.random.RandomState(0)
    X = rng.randn(40, 8).astype(np.float32)

    model = RQKMeans(n_levels=2, n_clusters=5, random_state=42)
    model.fit(X)
    codes = model.encode(X)

    results = evaluate(X, codes, encoder=model, k=5)
    assert "quantization_mse" in results
    assert results["quantization_mse"] >= 0
    assert results["n_samples"] == 40


def test_token_format_round_trip():
    """Token-format IDs are generated correctly through the engine."""
    rng = np.random.RandomState(42)
    X = rng.randn(20, 8).astype(np.float32)

    encoder = RQKMeans(n_levels=3, n_clusters=5, random_state=42)
    engine = SemanticIdEngine(encoder=encoder)
    engine.fit(X)

    ids = engine.unique_ids(X, fmt="token")
    for sid in ids:
        assert sid.startswith("<a_")
        assert "<b_" in sid
        assert "<c_" in sid


def test_rqvae_l1_loss():
    """RQVAE trains with l1 loss type."""
    X = np.random.RandomState(0).randn(30, 16).astype(np.float32)
    model = RQVAE(
        in_dim=16,
        num_emb_list=[4, 4],
        e_dim=8,
        layers=[16],
        loss_type="l1",
        epochs=2,
        batch_size=16,
        device="cpu",
    )
    model.fit(X)
    codes = model.encode(X)
    assert codes.shape == (30, 2)


def test_rqvae_cosine_scheduler():
    """RQVAE trains with cosine LR scheduler and warmup."""
    X = np.random.RandomState(0).randn(30, 16).astype(np.float32)
    model = RQVAE(
        in_dim=16,
        num_emb_list=[4, 4],
        e_dim=8,
        layers=[16],
        epochs=3,
        batch_size=16,
        lr_scheduler="cosine",
        warmup_epochs=1,
        device="cpu",
    )
    model.fit(X)
    assert len(model.history_["lr"]) == 3


def test_rqvae_step_scheduler():
    """RQVAE trains with step LR scheduler."""
    X = np.random.RandomState(0).randn(30, 16).astype(np.float32)
    model = RQVAE(
        in_dim=16,
        num_emb_list=[4, 4],
        e_dim=8,
        layers=[16],
        epochs=3,
        batch_size=16,
        lr_scheduler="step",
        lr_step_size=1,
        device="cpu",
    )
    model.fit(X)
    assert len(model.history_["lr"]) == 3


def test_rqvae_constant_warmup_scheduler():
    """RQVAE trains with constant_with_warmup scheduler."""
    X = np.random.RandomState(0).randn(30, 16).astype(np.float32)
    model = RQVAE(
        in_dim=16,
        num_emb_list=[4, 4],
        e_dim=8,
        layers=[16],
        epochs=3,
        batch_size=16,
        lr_scheduler="constant_with_warmup",
        warmup_epochs=1,
        device="cpu",
    )
    model.fit(X)
    assert len(model.history_["lr"]) == 3


def test_rqvae_verbose_logging(capsys):
    """RQVAE verbose mode with int interval."""
    X = np.random.RandomState(0).randn(30, 16).astype(np.float32)
    model = RQVAE(
        in_dim=16,
        num_emb_list=[4, 4],
        e_dim=8,
        layers=[16],
        epochs=4,
        batch_size=16,
        device="cpu",
        verbose=2,
    )
    model.fit(X)
    assert len(model.history_["loss"]) == 4


def test_rqvae_invalid_loss_type():
    """RQVAE raises on invalid loss type during forward."""
    import torch

    from semantic_id.algorithms.rq_vae_module import RQVAEModule

    module = RQVAEModule(
        in_dim=8, num_emb_list=[4], e_dim=4, layers=[8], loss_type="bad"
    )
    x = torch.randn(2, 8)
    out, rq_loss, _ = module(x)
    with pytest.raises(ValueError, match="incompatible"):
        module.compute_loss(out, rq_loss, x)
