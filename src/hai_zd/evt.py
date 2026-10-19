"""One-sided SPOT/DSPOT anomaly thresholds (Siffer et al., KDD 2017).

Independent implementation from the published formulas: peaks-over-threshold
with a GPD tail fitted through Grimshaw's (1993) likelihood reduction, solved
by a bounded multi-start root search. No GPL-licensed reference code is copied.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

from typing import Any

import numpy as np
from numpy.typing import NDArray

Array = NDArray[Any]

__all__ = ["DSPOT", "SPOT", "ThresholdResult", "fit_gpd", "pot_threshold"]


@dataclass
class ThresholdResult:
    """Per-step decisions; thresholds are the pre-decision decision thresholds."""

    alarms: Array  # bool, shape (N,)
    thresholds: Array  # float64, shape (N,)


def _as_1d_finite(values, name: str, min_size: int = 0) -> Array:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {arr.shape}")
    if arr.size < min_size:
        raise ValueError(f"{name} needs at least {min_size} values, got {arr.size}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def _probability(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or not 0.0 < value < 1.0:
        raise ValueError(f"{name} must lie in the open interval (0, 1), got {value}")
    return value


def _positive_int(value: int, name: str) -> int:
    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value}")
    return value


def _grimshaw_equation(theta: float, y: Array) -> float:
    """Reduced score equation g(theta) whose nonzero roots are GPD stationary points.

    With theta = -gamma/sigma and a(theta) = mean(log(1 - theta*y)), the two GPD
    score equations give gamma = a(theta) and mean(y/(1 - theta*y)) =
    -a(theta)/(theta*(a(theta) + 1)); g is the difference of both sides.
    """
    s = 1.0 - theta * y
    a = float(np.log(s).mean())
    denom = theta * (a + 1.0)
    if denom == 0.0:
        return math.nan
    return float(np.mean(y / s) + a / denom)


def _bisect_root(lo: float, hi: float, f_lo: float, f_hi: float, y: Array) -> float | None:
    """Bounded bisection root search of the Grimshaw equation on [lo, hi]."""
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if mid == lo or mid == hi:
            break
        f_mid = _grimshaw_equation(mid, y)
        if not math.isfinite(f_mid):
            return None  # gamma = -1 pole inside the bracket; discard it
        if f_mid == 0.0 or abs(hi - lo) <= 1e-12 * max(1.0, abs(lo), abs(hi)):
            return mid
        if (f_lo < 0.0) == (f_mid < 0.0):
            lo, f_lo = mid, f_mid
        else:
            hi, f_hi = mid, f_mid
    return 0.5 * (lo + hi)


def fit_gpd(excesses: Array) -> tuple[float, float]:
    """Grimshaw MLE of the GPD shape gamma and scale sigma for positive excesses."""
    y = _as_1d_finite(excesses, "excesses", min_size=2)
    if not np.all(y > 0.0):
        raise ValueError("excesses must be strictly positive")
    n = y.size
    mean = float(y.mean())
    y_min = float(y.min())
    y_max = float(y.max())

    # Exponential (gamma = 0) branch: MLE sigma = mean, log-likelihood -n*(ln(mean) + 1).
    best_ll = -n * (math.log(mean) + 1.0)
    best = (0.0, mean)

    # theta = -gamma/sigma; support requires theta < 1/y_max. The gamma > 0 side is
    # unbounded, so a generous magnitude 1/y_min closes the search interval.
    upper = 1.0 / y_max
    magnitudes = np.geomspace(upper * 1e-6, 1.0 / y_min, num=100)
    grid = np.concatenate((-magnitudes[::-1], np.geomspace(upper * 1e-6, upper * (1.0 - 1e-9), num=100)))
    values = [_grimshaw_equation(float(theta), y) for theta in grid]

    roots: list[float] = []
    for i in range(grid.size - 1):
        f_lo, f_hi = values[i], values[i + 1]
        if not (math.isfinite(f_lo) and math.isfinite(f_hi)):
            continue
        if f_lo == 0.0:
            roots.append(float(grid[i]))
        elif (f_lo < 0.0) != (f_hi < 0.0):
            root = _bisect_root(float(grid[i]), float(grid[i + 1]), f_lo, f_hi, y)
            if root is not None:
                roots.append(root)

    for theta in roots:
        if abs(theta) < 1e-12:
            continue  # trivial zero root; covered by the exponential branch
        log_s = np.log1p(-theta * y)
        gamma = float(log_s.mean())
        sigma = -gamma / theta
        if not (math.isfinite(gamma) and math.isfinite(sigma)):
            continue
        if gamma <= -1.0 or sigma <= 0.0:
            continue  # gamma <= -1 makes the likelihood unbounded; not a valid MLE
        ll = -n * math.log(sigma) - (1.0 + 1.0 / gamma) * float(log_s.sum())
        if math.isfinite(ll) and ll > best_ll:
            best = (gamma, sigma)
            best_ll = ll
    return best


def pot_threshold(t: float, q: float, n: int, n_peaks: int, gamma: float, sigma: float) -> float:
    """POT quantile z_q above base threshold t; falls back to t when not estimable."""
    t = float(t)
    gamma = float(gamma)
    sigma = float(sigma)
    q = _probability(q, "q")
    if not math.isfinite(t):
        raise ValueError(f"t must be finite, got {t}")
    if not math.isfinite(gamma) or not math.isfinite(sigma):
        raise ValueError(f"gamma and sigma must be finite, got {gamma} and {sigma}")
    n = int(n)
    n_peaks = int(n_peaks)
    if n < 1:
        raise ValueError(f"n must be positive, got {n}")
    if n_peaks < 0:
        raise ValueError(f"n_peaks must be non-negative, got {n_peaks}")
    if n_peaks == 0 or sigma <= 0.0:
        return t
    r = q * n / n_peaks
    if r >= 1.0:
        return t  # requested quantile is not in the tail
    log_r = math.log(r)
    if gamma == 0.0:
        return t - sigma * log_r
    z = t + sigma * math.expm1(-gamma * log_r) / gamma
    if gamma < 0.0:
        z = min(z, t - sigma / gamma)  # never beyond the GPD upper endpoint
    return z


class SPOT:
    """One-sided SPOT: static base threshold t, adaptive GPD excess threshold z_q.

    Alarms never update the model; values in (t, z_q] are stored as peaks and
    update both counts; values at or below t only update the observation count.
    The GPD is refitted every `update_every` new peaks once `min_peaks` exist.
    """

    def __init__(
        self,
        initial_scores: Array,
        q: float,
        level: float = 0.98,
        min_peaks: int = 20,
        update_every: int = 25,
    ) -> None:
        self.q = _probability(q, "q")
        self.level = _probability(level, "level")
        self.min_peaks = _positive_int(min_peaks, "min_peaks")
        self.update_every = _positive_int(update_every, "update_every")
        initial = _as_1d_finite(initial_scores, "initial_scores", min_size=2)
        self.n = int(initial.size)
        ordered = np.sort(initial)
        self.t = float(ordered[int(self.level * initial.size)])
        peaks = initial[initial > self.t] - self.t
        self.peaks: list[float] = peaks.tolist()
        self.nt = int(peaks.size)
        self.gamma = 0.0
        self.sigma = float(peaks.mean()) if self.nt else 0.0
        self._since_fit = 0
        if self.nt >= max(self.min_peaks, 2):
            self._fit()

    def _fit(self) -> None:
        self.gamma, self.sigma = fit_gpd(np.asarray(self.peaks, dtype=np.float64))
        self._since_fit = 0

    @property
    def threshold(self) -> float:
        """Current decision threshold z_q on the model (decision) scale."""
        if self.nt < self.min_peaks:
            return self.t
        return pot_threshold(self.t, self.q, self.n, self.nt, self.gamma, self.sigma)

    def _observe(self, value: float) -> None:
        """Update counts and model with a non-alarm value on the decision scale."""
        if value > self.t:
            self.peaks.append(value - self.t)
            self.nt += 1
            self._since_fit += 1
            fit_start = max(self.min_peaks, 2)
            if self.nt >= fit_start and (self.nt == fit_start or self._since_fit >= self.update_every):
                self._fit()
        self.n += 1

    def _step(self, value: float) -> tuple[bool, float]:
        z = self.threshold
        alarm = value > z
        if not alarm:
            self._observe(value)
        return alarm, z

    def run(self, scores: Array) -> ThresholdResult:
        """Stream scores through the detector; returns pre-decision thresholds."""
        scores = _as_1d_finite(scores, "scores")
        alarms = np.empty(scores.size, dtype=bool)
        thresholds = np.empty(scores.size, dtype=np.float64)
        for i, value in enumerate(scores):
            alarms[i], thresholds[i] = self._step(float(value))
        return ThresholdResult(alarms=alarms, thresholds=thresholds)


class DSPOT(SPOT):
    """DSPOT: SPOT on residuals against the causal mean of recent raw values.

    The decision for x uses residual x - M, where M averages the last `depth`
    non-alarm raw values. Alarms update neither the window nor the model; peaks
    and normal values update both. Reported thresholds are on the raw scale M+z.
    """

    def __init__(
        self,
        initial_scores: Array,
        q: float,
        level: float = 0.98,
        min_peaks: int = 20,
        update_every: int = 25,
        depth: int = 360,
    ) -> None:
        self.depth = _positive_int(depth, "depth")
        initial = _as_1d_finite(initial_scores, "initial_scores", min_size=2)
        if initial.size <= self.depth:
            raise ValueError("initial_scores must contain more samples than depth")
        # Paper Algorithm 3: residuals start only after one complete causal window.
        cumulative = np.concatenate(([0.0], np.cumsum(initial)))
        means = (cumulative[self.depth : initial.size] - cumulative[: initial.size - self.depth]) / self.depth
        residuals = initial[self.depth :] - means
        super().__init__(residuals, q, level, min_peaks, update_every)
        window = initial[-self.depth :]
        self._window: deque[float] = deque(float(v) for v in window)
        self._window_sum = float(window.sum())

    @property
    def threshold(self) -> float:
        """Current decision threshold on the raw scale: window mean plus z_q."""
        return self._window_sum / len(self._window) + super().threshold

    def _step(self, value: float) -> tuple[bool, float]:
        mean = self._window_sum / len(self._window)
        residual = value - mean
        z = super().threshold
        alarm = residual > z
        if not alarm:
            if len(self._window) >= self.depth:
                self._window_sum -= self._window.popleft()
            self._window.append(value)
            self._window_sum += value
            self._observe(residual)
        return alarm, mean + z
