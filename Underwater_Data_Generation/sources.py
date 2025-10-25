from dataclasses import dataclass
import numpy as np
from utils import tukey_mask

@dataclass
class ClassParams:
    blades_lo: int; blades_hi: int
    rpm_lo: float; rpm_hi: float
    broad: tuple; hub: tuple; tip: tuple
    alpha_broad: tuple; alpha_hub: tuple
    harmonics: int
    tip_rate_base_hz: tuple  # Base cavitation rate at threshold RPM
    tip_rpm_threshold: float  # RPM above which cavitation increases
    tip_rate_exponent: float  # How quickly cavitation increases with RPM
    tip_dur_ms: tuple
    rpm_drift_pct: tuple
    rpm_drift_period_s: tuple
    rpm_walk_std: float  # Random walk standard deviation
    transient_prob: float  # Probability of speed change event

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
    """Creates intermittent, band-limited noise for cavitation bursts."""
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

def rpm_profile(T, fs, rpm_lo, rpm_hi, drift_pct_rng, drift_period_rng, walk_std, transient_prob, rng):
    """
    Models time-varying propeller rotation speed with:
    - Slow sinusoidal drift
    - Random walk fluctuations
    - Occasional transient speed changes
    """
    n = int(T*fs)
    t = np.arange(n)/fs
    
    # Base RPM with sinusoidal drift
    rpm0 = rng.uniform(rpm_lo, rpm_hi)
    pct = rng.uniform(*drift_pct_rng)/100.0
    period = rng.uniform(*drift_period_rng)
    rpm_t = rpm0 * (1 + pct*np.sin(2*np.pi*t/period))
    
    # Add random walk (models load variations, control imperfections)
    walk = np.cumsum(rng.normal(0, walk_std, n))
    walk = walk - np.mean(walk)  # Zero mean
    rpm_t += walk
    
    # Add transient events (speed changes)
    n_transients = rng.binomial(int(T/10), transient_prob)  # Check every ~10s
    for _ in range(n_transients):
        t_start = rng.integers(0, max(1, n - int(3*fs)))
        delta_rpm = rng.uniform(-0.15, 0.15) * rpm0  # ±15% change
        transition_samples = int(rng.uniform(2, 8) * fs)  # 2-8 second transition
        ramp = np.linspace(0, 1, transition_samples)
        t_end = min(t_start + transition_samples, n)
        rpm_t[t_start:t_end] += delta_rpm * ramp[:t_end-t_start]
        if t_end < n:
            rpm_t[t_end:] += delta_rpm
    
    # Ensure RPM stays in reasonable bounds
    rpm_t = np.clip(rpm_t, rpm_lo*0.8, rpm_hi*1.2)
    
    return rpm_t

def bpf_tones(blades, rpm_t, harmonics, fs, amp_rng, rng):
    """
    Generates tonal components from propeller blade rates.
    Uses 1/k^1.5 decay (more realistic than 1/k for marine propellers).
    """
    n = len(rpm_t)
    f_bpf = blades * rpm_t / 60.0
    phase = 2*np.pi*np.cumsum(f_bpf)/fs
    y = np.zeros(n, dtype=float)
    base = rng.uniform(*amp_rng)
    for k in range(1, harmonics+1):
        # 1/k^1.5 decay is more realistic for marine propellers
        y += (base/(k**1.5)) * np.sin(k*phase + rng.uniform(0, 2*np.pi))
    return y

def rpm_dependent_cavitation_rate(rpm_t, rpm_threshold, base_rate, exponent):
    """
    Cavitation rate increases dramatically above threshold RPM.
    Physics: Cavitation number σ ~ 1/V^2, where V ~ RPM
    """
    rpm_ratio = np.maximum(rpm_t / rpm_threshold, 1.0)
    rate_multiplier = rpm_ratio ** exponent
    return base_rate * rate_multiplier

def speed_dependent_mixing(rpm_mean, rpm_lo, rpm_hi, class_name):
    """
    Adjust component mixing weights based on operating speed.
    At low speeds: More broadband flow noise
    At high speeds: More tonal and cavitation components
    """
    rpm_normalized = (rpm_mean - rpm_lo) / max(rpm_hi - rpm_lo, 1.0)
    rpm_normalized = np.clip(rpm_normalized, 0.0, 1.0)
    
    if class_name == "submarine":
        # Submarines: Stealth design, tonals suppressed at low speed
        w_broad = 0.75 - 0.15 * rpm_normalized  # 0.75 -> 0.60
        w_tip = 0.10 + 0.20 * (rpm_normalized**2)  # 0.10 -> 0.30 (quadratic, cavitation onset)
        w_hub = 0.10
        w_tonal = 0.05 + 0.10 * rpm_normalized  # 0.05 -> 0.15
    else:  # torpedo
        # Torpedoes: Higher speed, tonals more prominent
        w_broad = 0.50 - 0.10 * rpm_normalized  # 0.50 -> 0.40
        w_tip = 0.20 + 0.25 * rpm_normalized  # 0.20 -> 0.45
        w_hub = 0.15
        w_tonal = 0.15 + 0.20 * rpm_normalized  # 0.15 -> 0.35
    
    # Normalize to ensure sum ~ 1.0
    total = w_broad + w_tip + w_hub + w_tonal
    return w_broad/total, w_tip/total, w_hub/total, w_tonal/total

def apply_doppler_shift(s, fs, doppler_factor):
    """
    Apply Doppler shift to signal.
    doppler_factor = (c + v_r) / (c + v_s)
    where v_r = receiver velocity toward source (negative if away)
          v_s = source velocity away from receiver (negative if toward)
    For stationary receiver: doppler_factor ≈ 1 + v_s/c
    """
    if abs(doppler_factor - 1.0) < 1e-6:
        return s
    
    # Resample to simulate Doppler
    n_orig = len(s)
    n_new = int(n_orig / doppler_factor)
    t_orig = np.arange(n_orig) / fs
    t_new = np.arange(n_new) / (fs / doppler_factor)
    
    # Simple linear interpolation
    s_doppler = np.interp(t_new, t_orig, s)
    
    # Pad or truncate to original length for consistency
    if len(s_doppler) < n_orig:
        s_doppler = np.pad(s_doppler, (0, n_orig - len(s_doppler)), 'constant')
    else:
        s_doppler = s_doppler[:n_orig]
    
    return s_doppler

def synthesize(class_name, T, fs, params: ClassParams, rng, doppler_velocity_ms=None):
    """
    Combines all components into a final SOURCE signal with metadata.
    This signal is ready for Bellhop propagation (no channel effects applied here).
    
    doppler_velocity_ms: Source velocity in m/s (positive = away from receiver)
    """
    n = int(T*fs)
    blades = rng.integers(params.blades_lo, params.blades_hi+1)
    
    # Generate RPM profile with enhanced realism
    rpm_t = rpm_profile(T, fs, params.rpm_lo, params.rpm_hi,
                        params.rpm_drift_pct, params.rpm_drift_period_s,
                        params.rpm_walk_std, params.transient_prob, rng)
    
    rpm_mean = float(np.mean(rpm_t))
    
    # Generate tonal components
    tones = bpf_tones(blades, rpm_t, params.harmonics, fs, amp_rng=(0.005, 0.05), rng=rng)
    
    # Generate broadband and hub noise
    broad = colored_noise(rng.uniform(*params.alpha_broad), fs, params.broad, n, rng)
    hub = colored_noise(rng.uniform(*params.alpha_hub), fs, params.hub, n, rng)
    
    # RPM-dependent cavitation
    base_rate = rng.uniform(*params.tip_rate_base_hz)
    mean_rate = rpm_dependent_cavitation_rate(
        np.array([rpm_mean]), 
        params.tip_rpm_threshold, 
        base_rate, 
        params.tip_rate_exponent
    )[0]
    events = max(0, rng.poisson(mean_rate * T))
    tip = bursty_band_noise(fs, params.tip, n, events,
                           (params.tip_dur_ms[0]/1000.0, params.tip_dur_ms[1]/1000.0), rng)
    
    # Speed-dependent mixing
    w_broad, w_tip, w_hub, w_tonal = speed_dependent_mixing(
        rpm_mean, params.rpm_lo, params.rpm_hi, class_name
    )
    
    # Combine components with dynamic weights
    s = w_broad*broad + w_tip*tip + w_hub*hub + w_tonal*tones
    s /= (np.max(np.abs(s)) + 1e-9)
    
    # Apply Doppler shift if velocity specified
    doppler_factor = 1.0
    if doppler_velocity_ms is not None:
        c_water = 1500.0  # m/s
        doppler_factor = 1.0 + doppler_velocity_ms / c_water
        s = apply_doppler_shift(s, fs, doppler_factor)
    
    # Metadata
    meta = {
        "class": class_name,
        "blades": int(blades),
        "RPM_mean": float(rpm_mean),
        "RPM_std": float(np.std(rpm_t)),
        "BPF_mean_Hz": float(np.mean(blades * rpm_t / 60.0)),
        "cavitation_events": int(events),
        "cavitation_rate_hz": float(mean_rate),
        "mixing_broad": float(w_broad),
        "mixing_tonal": float(w_tonal),
        "mixing_tip": float(w_tip),
        "mixing_hub": float(w_hub),
        "doppler_factor": float(doppler_factor),
        "source_velocity_ms": float(doppler_velocity_ms) if doppler_velocity_ms else 0.0
    }
    
    return s.astype(np.float32), meta
