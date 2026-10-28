import copy

import numpy as np
import pytest
import torch
from torch import nn

from src.models.deep import build_model, count_params
from src.preprocessing import PerLeadNormalizer
from src.train import predict_records, prepare_signals


@pytest.mark.parametrize("name", ["simple_cnn", "resnet1d"])
@pytest.mark.parametrize("seconds", [2.5, 10])
def test_output_shape(name, seconds):
    model = build_model(name).eval()
    x = torch.randn(3, int(seconds * 100), 12)
    with torch.no_grad():
        assert model(x).shape == (3, 12)


def test_param_counts_in_expected_range():
    assert 1_000_000 <= count_params(build_model("resnet1d")) <= 3_000_000
    assert count_params(build_model("simple_cnn")) < 1_000_000


def test_sliding_window_aggregation_one_row_per_record():
    model = build_model("simple_cnn")
    X = np.random.default_rng(0).normal(size=(5, 1000, 12)).astype(np.float32)
    out = predict_records(model, X, window=250, overlap=0.5, batch_size=8)
    assert out.shape == (5, 12)
    assert np.isfinite(out).all()


def test_max_aggregation_over_windows():
    class FirstSampleModel(nn.Module):           # logits = the window's first sample
        def forward(self, x):
            return x[:, 0, :]
    X = np.zeros((1, 1000, 12), np.float32)
    X[0, 500] = 7.0                               # start of the window at 500 only
    out = predict_records(FirstSampleModel(), X, window=250, overlap=0.5)
    assert np.allclose(out, 7.0)


def test_training_steps_reduce_loss():
    torch.manual_seed(0)
    model = build_model("simple_cnn")
    x = torch.randn(16, 250, 12)
    y = (torch.rand(16, 12) > 0.7).float()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    model.train()
    first = loss_fn(model(x), y).item()
    for _ in range(30):
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
    model.eval()
    assert loss_fn(model(x), y).item() < first


def test_normalizer_uses_train_only(config):
    rng = np.random.default_rng(0)
    train = rng.normal(2.0, 3.0, size=(20, 1000, 12)).astype(np.float32)
    val = rng.normal(-50.0, 0.1, size=(5, 1000, 12)).astype(np.float32)
    cfg = copy.deepcopy(config)
    norm = PerLeadNormalizer().fit(prepare_signals(train, 100, cfg))
    assert np.allclose(norm.mean, train.reshape(-1, 12).mean(axis=0), atol=1e-3)
    assert np.allclose(norm.std, train.reshape(-1, 12).std(axis=0), atol=1e-3)
    v = prepare_signals(val, 100, cfg, norm)
    assert v.mean() < -10                         # val shifted by train statistics, not its own


def test_chunked_normalizer_matches_direct():
    X = np.random.default_rng(1).normal(size=(50, 100, 12)).astype(np.float32) * 5 + 1
    norm = PerLeadNormalizer().fit(X, chunk=7)
    assert np.allclose(norm.mean, X.reshape(-1, 12).mean(axis=0), atol=1e-4)
    assert np.allclose(norm.std, X.reshape(-1, 12).std(axis=0), atol=1e-4)
