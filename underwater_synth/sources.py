from dataclasses import dataclass
import numpy as np
from .utils import tukey_mask

@dataclass
class ClassParams:
    blades_lo: int; blades_hi: int
    rpm_lo: float; rpm_hi: float
    broad: tuple; hub: tuple; tip: tuple
    alpha_broad: tuple; alpha_hub: tuple
    harmonics: int
    tip_rate_hz: tuple; tip_dur_ms: tuple
    rpm_drift_pct: tuple; rpm_drift_period_s: tuple

def colored_noise(alpha, fs, band, n, rng):

    """
    Generates band-limited noise with a 1/f^α spectral shape.
    """
    x = rng.standard_normal(n)
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, 1/fs)
    shape = 1.0 / np.maximum(f, 1e-6) ** (alpha/2.0)
    mask = tukey_mask(f, band[0], band[1], roll=0.1)
    Y = X * shape * mask
    y = np.fft.irfft(Y, n=n)
    y /= (np.std(y) + 1e-12)
    return y

def bursty_band_noise(fs, band, n, events, dur_range_s, rng):

    """ Creates intermittent, band-limited noise for cavitation bursts. """

    y = np.zeros(n, dtype=float)
    f = np.fft.rfftfreq(n, 1/fs)
    env = np.zeros(n, dtype=float)
    for _ in range(events):
        dur = rng.uniform(*dur_range_s)
        L = int(max(1, dur*fs))
        start = rng.integers(0, max(1, n - L))
        window = 0.5 - 0.5*np.cos(2*np.pi*np.arange(L)/max(L-1,1))
        env[start:start+L] += window
    env = np.clip(env, 0.0, 1.0)
    x = rng.standard_normal(n)
    X = np.fft.rfft(x)
    mask = tukey_mask(f, band[0], band[1], roll=0.08)
    Y = X * mask
    z = np.fft.irfft(Y, n=n)
    z /= (np.std(z) + 1e-12)
    return z * env

def rpm_profile(T, fs, rpm_lo, rpm_hi, drift_pct_rng, drift_period_rng, rng):

    """Models time-varying propeller rotation speed with drift."""

    n = int(T*fs)
    rpm0 = rng.uniform(rpm_lo, rpm_hi)
    pct = rng.uniform(*drift_pct_rng)/100.0
    period = rng.uniform(*drift_period_rng)
    t = np.arange(n)/fs
    rpm_t = rpm0 * (1 + pct*np.sin(2*np.pi*t/period))
    return rpm_t

def bpf_tones(blades, rpm_t, harmonics, fs, amp_rng, rng):

    """Generates tonal components from propeller blade rates."""

    n = len(rpm_t)
    f_bpf = blades * rpm_t / 60.0
    phase = 2*np.pi*np.cumsum(f_bpf)/fs
    y = np.zeros(n, dtype=float)
    base = rng.uniform(*amp_rng)
    for k in range(1, harmonics+1):
        y += (base/k) * np.sin(k*phase + rng.uniform(0, 2*np.pi))
    return y

def synthesize(class_name, T, fs, params: ClassParams, rng):

    """Combines all components into a final signal with metadata."""

    n = int(T*fs)
    blades = rng.integers(params.blades_lo, params.blades_hi+1)
    rpm_t  = rpm_profile(T, fs, params.rpm_lo, params.rpm_hi,
                        params.rpm_drift_pct, params.rpm_drift_period_s, rng)
    tones  = bpf_tones(blades, rpm_t, params.harmonics, fs, amp_rng=(0.005, 0.05), rng=rng)
    broad  = colored_noise(rng.uniform(*params.alpha_broad), fs, params.broad, n, rng)
    hub    = colored_noise(rng.uniform(*params.alpha_hub), fs, params.hub, n, rng) * rng.uniform(0.1, 0.3)
    rate   = rng.uniform(*params.tip_rate_hz)
    events = max(0, rng.poisson(rate * T))
    tip    = bursty_band_noise(fs, params.tip, n, events,
                              (params.tip_dur_ms[0]/1000.0, params.tip_dur_ms[1]/1000.0), rng) * rng.uniform(0.1, 0.4)
    s = 0.7*broad + 0.2*tip + 0.1*hub + tones
    s /= (np.max(np.abs(s)) + 1e-9)
    meta = {
        "class": class_name,
        "blades": int(blades),
        "RPM_mean": float(np.mean(rpm_t)),
        "BPF_mean_Hz": float(np.mean(blades * rpm_t / 60.0))
    }
    return s.astype(np.float32), meta
