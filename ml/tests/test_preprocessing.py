import numpy as np
import pytest

from src.preprocessing import (LEAD_ORDER, PerLeadNormalizer, inference_windows, preprocess,
                               random_crop, reorder_leads, sliding_windows, training_example,
                               window_starts)


def _signal(seconds, fs, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(seconds * fs, 12)).astype(np.float32)


def test_resample_500_to_100():
    x = preprocess(_signal(10, 500), fs=500, target_fs=100)
    assert x.shape == (1000, 12)
    assert x.dtype == np.float32


@pytest.mark.parametrize("seconds", [5, 10, 30, 60])
def test_resample_lengths(seconds):
    assert preprocess(_signal(seconds, 500), 500, 100).shape == (seconds * 100, 12)


def test_lead_reordering_case_insensitive():
    sig = np.tile(np.arange(12, dtype=float), (50, 1))       # column j holds value j
    names = [n.lower() for n in LEAD_ORDER]
    perm = np.random.default_rng(1).permutation(12)
    shuffled, shuffled_names = sig[:, perm], [names[p] for p in perm]
    out = reorder_leads(shuffled, shuffled_names)
    assert out[0].tolist() == list(range(12))
    upper = reorder_leads(sig, [n.upper() for n in LEAD_ORDER])
    assert np.array_equal(upper, sig)


def test_missing_lead_raises():
    names = LEAD_ORDER[:11] + ["X"]
    with pytest.raises(ValueError):
        reorder_leads(np.zeros((10, 12)), names)


def test_normalizer_fit_only_on_given_data(tmp_path):
    train = _signal(10, 100, seed=0) * 2 + 5
    other = _signal(10, 100, seed=1) * 100 - 50
    norm = PerLeadNormalizer().fit(train[None])
    assert np.allclose(norm.mean, train.mean(axis=0), atol=1e-5)
    assert np.allclose(norm.std, train.std(axis=0), atol=1e-5)
    z = norm.transform(train)
    assert np.allclose(z.mean(axis=0), 0, atol=1e-4)
    norm.transform(other)                                   # transforming must not refit
    assert np.allclose(norm.mean, train.mean(axis=0), atol=1e-5)
    path = tmp_path / "norm.json"
    norm.save(path)
    loaded = PerLeadNormalizer.load(path)
    assert np.allclose(loaded.transform(other), norm.transform(other))


@pytest.mark.parametrize("seconds", [10, 30])
def test_sliding_windows_cover_every_sample(seconds):
    fs, window = 100, 250
    n = seconds * fs
    sig = np.arange(n, dtype=np.float32)[:, None].repeat(12, axis=1)
    windows, starts = sliding_windows(sig, window, overlap=0.5)
    covered = np.zeros(n, dtype=bool)
    for s in starts:
        covered[s:s + window] = True
    assert covered.all()
    assert starts[-1] + window == n
    assert windows.shape == (len(starts), window, 12)
    assert np.array_equal(windows[-1][:, 0], np.arange(n - window, n))


def test_window_starts_tail_not_on_step():
    starts = window_starts(1030, 250, 0.5)
    assert starts[-1] == 1030 - 250


def test_short_signal_is_padded():
    windows, starts = sliding_windows(np.ones((100, 12), np.float32), 250)
    assert windows.shape == (1, 250, 12) and starts == [0]
    assert random_crop(np.ones((100, 12)), 250, np.random.default_rng(0)).shape == (250, 12)


def test_training_and_inference_share_preprocessing():
    raw = _signal(10, 500, seed=3)
    norm = PerLeadNormalizer().fit(preprocess(raw, 500, 100)[None])
    windows, starts = inference_windows(raw, 500, 100, 2.5, normalizer=norm)
    crop = training_example(raw, 500, 100, 2.5, np.random.default_rng(7), normalizer=norm)
    full = norm.transform(preprocess(raw, 500, 100))
    start = int(np.random.default_rng(7).integers(0, len(full) - 250 + 1))
    assert np.array_equal(crop, full[start:start + 250])
    assert np.array_equal(windows[1], full[starts[1]:starts[1] + 250])
