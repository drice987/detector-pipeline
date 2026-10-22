import numpy as np
import pytest

from pipeline.stats import EMAStats, WelfordRollingStats


def test_welford_single_scan_copies_inputs_and_returns_zero_sem():
    stats = WelfordRollingStats()
    raw_1d = np.array([10.0, 20.0, 30.0])
    raw_2d = np.ones((2, 3), dtype=float)
    norm_1d = raw_1d / 60.0
    norm_2d = raw_2d / 60.0

    stats.update(raw_1d, raw_2d, norm_1d, norm_2d, intensity=60.0)
    raw_1d[:] = 999.0

    sem_raw, sem_norm = stats.get_sem()

    assert stats.n_scans == 1
    assert stats.mean_intensity == 60.0
    np.testing.assert_allclose(stats.mean_1d_raw, [10.0, 20.0, 30.0])
    np.testing.assert_array_equal(sem_raw, np.zeros(3))
    np.testing.assert_array_equal(sem_norm, np.zeros(3))


def test_welford_matches_batch_numpy_statistics():
    rng = np.random.default_rng(42)
    scans_2d = rng.normal(loc=100.0, scale=5.0, size=(10, 4, 8))
    scans_1d = scans_2d.sum(axis=1)
    totals = scans_1d.sum(axis=1)
    norm_1d = scans_1d / totals[:, None]
    norm_2d = scans_2d / totals[:, None, None]

    stats = WelfordRollingStats()
    for s1d, s2d, n1d, n2d, tot in zip(scans_1d, scans_2d, norm_1d, norm_2d, totals):
        stats.update(s1d, s2d, n1d, n2d, intensity=float(tot))

    sem_raw, sem_norm = stats.get_sem()
    n = len(scans_2d)

    assert stats.n_scans == n
    assert stats.mean_intensity == pytest.approx(totals.mean())
    np.testing.assert_allclose(stats.mean_1d_raw, scans_1d.mean(axis=0))
    np.testing.assert_allclose(stats.mean_2d_raw, scans_2d.mean(axis=0))
    np.testing.assert_allclose(stats.mean_1d_norm, norm_1d.mean(axis=0))
    np.testing.assert_allclose(stats.mean_2d_norm, norm_2d.mean(axis=0))
    np.testing.assert_allclose(sem_raw, scans_1d.std(axis=0, ddof=1) / np.sqrt(n))
    np.testing.assert_allclose(sem_norm, norm_1d.std(axis=0, ddof=1) / np.sqrt(n))


def test_ema_asymmetric_alpha_weights():
    stats = EMAStats(alpha=0.2)
    zeros_1d, zeros_2d = np.zeros(4), np.zeros((2, 4))
    tens_1d, tens_2d = np.full(4, 10.0), np.full((2, 4), 10.0)

    stats.update(zeros_1d, zeros_2d, zeros_1d, zeros_2d, intensity=0.0)
    stats.update(tens_1d, tens_2d, tens_1d, tens_2d, intensity=40.0)

    sem_raw, _ = stats.get_sem()

    assert stats.n_scans == 2
    assert stats.mean_intensity == pytest.approx(8.0)
    np.testing.assert_allclose(stats.mean_1d_raw, np.full(4, 2.0))
    np.testing.assert_allclose(sem_raw, np.full(4, np.sqrt(0.8 * 0.2 * 100.0)))
