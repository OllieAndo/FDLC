import numpy as np

_A = np.exp(2j * np.pi / 3)
_A_INV = np.array([[1, 1, 1], [1, _A, _A**2], [1, _A**2, _A]]) / 3


def phasor(x: np.ndarray, fs: float, f0: float = 50.0) -> np.ndarray:
    """Fundamental-frequency phasor (peak magnitude) of each signal over the last full cycle.

    :param x: Samples with time on the last axis, e.g. shape (3, n) for phases a, b, c.
    :param fs: Sampling frequency in Hz.
    :param f0: Fundamental frequency in Hz.
    """
    n = round(fs / f0)
    window = x[..., -n:]
    t = np.arange(n) / fs
    return 2.0 / n * np.sum(window * np.exp(-2j * np.pi * f0 * t), axis=-1)


def sequence_components(abc: np.ndarray) -> np.ndarray:
    """Zero, positive and negative sequence phasors from phase phasors (a, b, c) on axis 0."""
    return np.tensordot(_A_INV, abc, axes=1)
