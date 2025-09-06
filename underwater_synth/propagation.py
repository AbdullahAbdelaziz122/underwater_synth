import numpy as np
from .utils import thorp_absorption_dB_per_m

def fast_propagate(x, fs, range_m, geometry="spherical", n_paths=3, src_depth=50.0, rx_depth=50.0, sea_state=2, rng=None):
    """
    Frequency-dependent TL (spreading + Thorp absorption) and simple multipath (few delayed, attenuated copies).
    """
    if rng is None:
        rng = np.random.default_rng()

    n = len(x)
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, 1/fs)

    A = 20.0 if geometry == "spherical" else 10.0
    TL_spread = A * np.log10(max(range_m, 1.0))
    TL_abs = thorp_absorption_dB_per_m(f) * range_m
    TL_total = TL_spread + TL_abs
    H_mag = 10 ** (-TL_total / 20.0)

    H = H_mag.astype(np.complex64)
    c = 1500.0  # m/s
    for i in range(n_paths):
        extra = rng.uniform(0.02, 0.2) * range_m / c
        refl = rng.uniform(0.2, 0.9) * (-1 if i == 0 else 1)
        loss_db = rng.uniform(0.5, 3.0) * (1 + 0.2*sea_state)
        H += (refl * 10 ** (-loss_db/20.0)) * np.exp(-2j*np.pi*f*extra) * H_mag

    Y = X * H
    y = np.fft.irfft(Y, n)
    y /= (np.max(np.abs(y)) + 1e-9)
    return y.astype(np.float32)
