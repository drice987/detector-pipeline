import numpy as np


class WelfordRollingStats:
    """Tracks rolling means and variance using Welford's algorithm."""

    def __init__(self) -> None:
        self.n_scans = 0
        self.mean_intensity = 0.0

        self.mean_1d_raw: np.ndarray | None = None
        self.m2_1d_raw: np.ndarray | None = None
        self.mean_2d_raw: np.ndarray | None = None

        self.mean_1d_norm: np.ndarray | None = None
        self.m2_1d_norm: np.ndarray | None = None
        self.mean_2d_norm: np.ndarray | None = None

    def update(
        self,
        raw_1d: np.ndarray,
        raw_2d: np.ndarray,
        norm_1d: np.ndarray,
        norm_2d: np.ndarray,
        intensity: float,
    ) -> None:
        """Updates rolling statistics for both absolute and normalized tracks."""
        if self.n_scans == 0:
            self.mean_1d_raw = np.copy(raw_1d)
            self.m2_1d_raw = np.zeros_like(raw_1d)
            self.mean_2d_raw = np.copy(raw_2d)

            self.mean_1d_norm = np.copy(norm_1d)
            self.m2_1d_norm = np.zeros_like(norm_1d)
            self.mean_2d_norm = np.copy(norm_2d)

            self.mean_intensity = intensity
            self.n_scans = 1
        else:
            self.n_scans += 1
            self.mean_intensity += (intensity - self.mean_intensity) / self.n_scans

            # Raw Math
            delta_raw = raw_1d - self.mean_1d_raw
            self.mean_1d_raw += delta_raw / self.n_scans
            self.m2_1d_raw += delta_raw * (raw_1d - self.mean_1d_raw)
            self.mean_2d_raw += (raw_2d - self.mean_2d_raw) / self.n_scans

            # Normalized Math
            delta_norm = norm_1d - self.mean_1d_norm
            self.mean_1d_norm += delta_norm / self.n_scans
            self.m2_1d_norm += delta_norm * (norm_1d - self.mean_1d_norm)
            self.mean_2d_norm += (norm_2d - self.mean_2d_norm) / self.n_scans

    def get_sem(self) -> tuple[np.ndarray, np.ndarray]:
        """Returns the Standard Error of the Mean for (raw, normalized) 1D arrays."""
        if self.n_scans > 1:
            var_raw = self.m2_1d_raw / (self.n_scans - 1)
            sem_raw = np.sqrt(var_raw / self.n_scans)

            var_norm = self.m2_1d_norm / (self.n_scans - 1)
            sem_norm = np.sqrt(var_norm / self.n_scans)
            return sem_raw, sem_norm

        return np.zeros_like(self.mean_1d_raw), np.zeros_like(self.mean_1d_norm)


class EMAStats:
    """Tracks Exponential Moving Average (EMA) to prioritize recent scans."""

    def __init__(self, alpha: float = 0.1) -> None:
        self.alpha = alpha
        self.n_scans = 0
        self.mean_intensity = 0.0

        self.mean_1d_raw: np.ndarray | None = None
        self.var_1d_raw: np.ndarray | None = None
        self.mean_2d_raw: np.ndarray | None = None

        self.mean_1d_norm: np.ndarray | None = None
        self.var_1d_norm: np.ndarray | None = None
        self.mean_2d_norm: np.ndarray | None = None

    def update(
        self,
        raw_1d: np.ndarray,
        raw_2d: np.ndarray,
        norm_1d: np.ndarray,
        norm_2d: np.ndarray,
        intensity: float,
    ) -> None:
        if self.n_scans == 0:
            self.mean_1d_raw, self.var_1d_raw = np.copy(raw_1d), np.zeros_like(raw_1d)
            self.mean_2d_raw = np.copy(raw_2d)
            self.mean_1d_norm, self.var_1d_norm = (
                np.copy(norm_1d),
                np.zeros_like(norm_1d),
            )
            self.mean_2d_norm = np.copy(norm_2d)
            self.mean_intensity = intensity
            self.n_scans = 1
        else:
            self.n_scans += 1
            self.mean_intensity += self.alpha * (intensity - self.mean_intensity)

            diff_raw = raw_1d - self.mean_1d_raw
            self.mean_1d_raw += self.alpha * diff_raw
            self.var_1d_raw = (1 - self.alpha) * (
                self.var_1d_raw + self.alpha * diff_raw**2
            )
            self.mean_2d_raw += self.alpha * (raw_2d - self.mean_2d_raw)

            diff_norm = norm_1d - self.mean_1d_norm
            self.mean_1d_norm += self.alpha * diff_norm
            self.var_1d_norm = (1 - self.alpha) * (
                self.var_1d_norm + self.alpha * diff_norm**2
            )
            self.mean_2d_norm += self.alpha * (norm_2d - self.mean_2d_norm)

    def get_sem(self) -> tuple[np.ndarray, np.ndarray]:
        if self.n_scans > 1:
            return np.sqrt(self.var_1d_raw), np.sqrt(self.var_1d_norm)
        return np.zeros_like(self.mean_1d_raw), np.zeros_like(self.mean_1d_norm)
