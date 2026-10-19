import numpy as np
import pytest
from scipy.stats import genpareto

from hai_zd.evt import DSPOT, SPOT, fit_gpd, pot_threshold


def test_pot_threshold_matches_exponential_and_gpd_formulas():
    exponential = pot_threshold(t=2.0, q=0.001, n=1000, n_peaks=100, gamma=0.0, sigma=0.5)
    assert exponential == pytest.approx(2.0 - 0.5 * np.log(0.01))
    gamma, sigma = 0.25, 0.5
    shaped = pot_threshold(2.0, 0.001, 1000, 100, gamma, sigma)
    expected = 2.0 + sigma * np.expm1(-gamma * np.log(0.01)) / gamma
    assert shaped == pytest.approx(expected)
    assert pot_threshold(2.0, 0.1, 1000, 100, gamma, sigma) == 2.0


def test_grimshaw_recovers_synthetic_gpd_parameters():
    excesses = genpareto.rvs(c=0.2, scale=2.0, size=5000, random_state=42)
    gamma, sigma = fit_gpd(excesses)
    assert gamma == pytest.approx(0.2, abs=0.12)
    assert sigma == pytest.approx(2.0, abs=0.35)


def test_spot_alarm_does_not_update_fallback_threshold():
    spot = SPOT(np.arange(100, dtype=float), q=0.001, min_peaks=20)
    result = spot.run(np.array([97.0, 99.0, 100.0]))
    np.testing.assert_array_equal(result.alarms, [False, True, True])
    np.testing.assert_array_equal(result.thresholds, [98.0, 98.0, 98.0])


def test_dspot_uses_only_past_non_alarm_values():
    initial = np.zeros(100)
    model = DSPOT(initial, q=0.001, depth=10, min_peaks=20)
    assert model.n == 90
    drift = model.run(np.array([-1.0, 0.0]))
    np.testing.assert_array_equal(drift.alarms, [False, True])
    np.testing.assert_array_equal(drift.thresholds, [0.0, -0.1])
    locked = DSPOT(initial, q=0.001, depth=10, min_peaks=20).run(np.array([1.0, 1.0]))
    np.testing.assert_array_equal(locked.alarms, [True, True])
    np.testing.assert_array_equal(locked.thresholds, [0.0, 0.0])
