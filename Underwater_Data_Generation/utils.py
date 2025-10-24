import numpy as np

def rng_from_seed(seed: int | None):
    return np.random.default_rng(seed if seed is not None else np.random.SeedSequence().entropy)

def tukey_mask(f, f_lo, f_hi, roll=0.1):
    """
    Smooth band mask in frequency domain using a cosine ramp.
    f: frequency vector (Hz)
    f_lo, f_hi: passband edges
    roll: fraction of transition width
    """
    mask = np.zeros_like(f, dtype=float)
    if f_hi <= f_lo:
        return mask
    width = max((f_hi - f_lo) * roll, 1e-6)
    # rise
    idx1 = (f >= (f_lo - width)) & (f < f_lo)
    mask[idx1] = 0.5 * (1 + np.cos(np.pi * (f_lo - f[idx1]) / width))
    # flat
    idx2 = (f >= f_lo) & (f <= f_hi)
    mask[idx2] = 1.0
    # fall
    idx3 = (f > f_hi) & (f <= (f_hi + width))
    mask[idx3] = 0.5 * (1 + np.cos(np.pi * (f[idx3] - f_hi) / width))
    return mask

def thorp_absorption_dB_per_m(f_hz: np.ndarray) -> np.ndarray:
    """
    Thorp-like absorption approximation (dB/m) vs frequency.
    Roughly valid from a few hundred Hz to tens of kHz.
    """
    f = np.maximum(f_hz, 1e-6) / 1000.0  # kHz
    a_db_per_km = 0.11 * (f**2 / (1 + f**2)) + 44 * (f**2 / (4100 + f**2)) + 2.75e-4 * f**2 + 0.003
    return a_db_per_km / 1000.0  # dB/m

def db_to_lin(db):
    return 10 ** (db / 20.0)

def lin_to_db(lin):
    lin = np.maximum(lin, 1e-20)
    return 20 * np.log10(lin)
