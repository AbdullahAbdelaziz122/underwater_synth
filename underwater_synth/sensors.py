import numpy as np
from scipy.signal import butter, sosfiltfilt

def bandlimit(x, fs, lo, hi, order=6):
    lo = max(1.0, lo)
    hi = min(0.49*fs, hi)
    sos = butter(order, [lo, hi], btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, x)

def add_ambient_and_sensor_noise(x, fs, class_name, sea_state, snr_db, rng):
    """
    Add ambient-shaped noise + white sensor noise to reach approximate SNR (in-band).
    """
    n = len(x)
    if class_name == "submarine":
        band = (20.0, 800.0)
    else:
        band = (300.0, 8000.0)
    amb = rng.standard_normal(n).astype(np.float32)
    amb = bandlimit(amb, fs, band[0], band[1])
    amb *= (0.15 + 0.1*sea_state)
    sens = rng.standard_normal(n).astype(np.float32) * 0.01
    noise = amb + sens
    sig_rms = np.sqrt(np.mean(x**2) + 1e-12)
    noise_rms = np.sqrt(np.mean(noise**2) + 1e-12)
    target_noise_rms = sig_rms / (10 ** (snr_db/20.0))
    scale = target_noise_rms / (noise_rms + 1e-12)
    noise *= scale
    y = x + noise
    y /= (max(1.0, np.max(np.abs(y)) * 1.05))
    return y.astype(np.float32)
