import numpy as np

from fdlc.features import phasor, sequence_components


def _abc(magnitudes, angles_deg, fs=10_000.0, f0=50.0, cycles=2):
    t = np.arange(int(fs / f0 * cycles)) / fs
    return np.array([m * np.cos(2 * np.pi * f0 * t + np.deg2rad(a)) for m, a in zip(magnitudes, angles_deg)])


def test_balanced_is_pure_positive_sequence():
    abc = phasor(_abc([1, 1, 1], [0, -120, 120]), fs=10_000.0)
    i0, i1, i2 = np.abs(sequence_components(abc))
    assert np.isclose(i1, 1.0, atol=1e-6)
    assert i0 < 1e-6 and i2 < 1e-6


def test_single_phase_has_equal_sequence_components():
    abc = phasor(_abc([1, 0, 0], [0, 0, 0]), fs=10_000.0)
    seq = np.abs(sequence_components(abc))
    assert np.allclose(seq, 1 / 3, atol=1e-6)
