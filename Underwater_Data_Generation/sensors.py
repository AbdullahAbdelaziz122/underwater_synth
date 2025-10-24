import numpy as np
from scipy.signal import butter, sosfiltfilt

def bandlimit(x, fs, lo, hi, order=6):
    """Apply bandpass filter."""
    lo = max(1.0, lo)
    hi = min(0.49*fs, hi)
    sos = butter(order, [lo, hi], btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, x)

def wenz_ambient_noise(f_hz, sea_state, shipping_level=5):
    """
    Generate ambient noise spectrum based on Wenz curves.
    
    Parameters:
    - f_hz: frequency array in Hz
    - sea_state: 0-6 (Beaufort scale approximation)
    - shipping_level: 0-7 (0=no shipping, 7=heavy shipping)
    
    Returns noise level in dB re 1 µPa²/Hz
    """
    f_khz = np.maximum(f_hz / 1000.0, 0.001)
    
    # Turbulence (very low freq, <10 Hz) - not modeled for typical sonar bands
    
    # Shipping noise (10-300 Hz): ~ 76 - 60*log10(f_kHz) + shipping_factor
    shipping_factor = (7 - shipping_level) * 5  # 0 to 35 dB reduction
    N_ship = 76 - 60*np.log10(f_khz) - shipping_factor
    
    # Wind-dependent noise (>300 Hz): dominant at high frequencies
    # Wenz: NL(f, wind) ≈ 50 + 7.5*wind^0.5 + 20*log10(f) - 40*log10(f+0.4)
    wind_speed_knots = sea_state * 5  # Rough conversion
    N_wind = 50 + 7.5*np.sqrt(wind_speed_knots) + 20*np.log10(f_khz) - 40*np.log10(f_khz + 0.4)
    
    # Thermal noise (high frequencies, >50 kHz) - usually negligible
    N_thermal = -15 + 20*np.log10(f_khz)
    
    # Combine using energy summation (10^(NL/10))
    N_total_linear = 10**(N_ship/10) + 10**(N_wind/10) + 10**(N_thermal/10)
    N_total_dB = 10 * np.log10(N_total_linear)
    
    return N_total_dB

def generate_ambient_noise(n, fs, sea_state, shipping_level, rng):
    """
    Generate realistic ambient noise with Wenz spectrum.
    """
    # Generate white noise
    noise = rng.standard_normal(n).astype(np.float32)
    
    # Apply Wenz spectrum shaping
    N = np.fft.rfft(noise)
    f = np.fft.rfftfreq(n, 1/fs)
    
    # Get Wenz spectrum (dB re 1 µPa²/Hz)
    spectrum_dB = wenz_ambient_noise(f, sea_state, shipping_level)
    
    # Convert to linear magnitude (relative scaling)
    # Normalize to reference frequency (e.g., 100 Hz)
    f_ref_idx = np.argmin(np.abs(f - 100.0))
    spectrum_dB_norm = spectrum_dB - spectrum_dB[f_ref_idx]
    spectrum_linear = 10 ** (spectrum_dB_norm / 20.0)
    
    # Apply shaping
    N_shaped = N * spectrum_linear
    noise_shaped = np.fft.irfft(N_shaped, n)
    
    # Normalize
    noise_shaped /= (np.std(noise_shaped) + 1e-12)
    
    return noise_shaped.astype(np.float32)

def add_ambient_and_sensor_noise(x, fs, class_name, sea_state, snr_db, shipping_level=5, rng=None):
    """
    Add realistic ambient noise (Wenz curves) + white sensor noise to reach target SNR.
    SNR is measured in the signal's primary frequency band.
    
    NOTE: This is for Stage 1 validation only. In Stage 2, Bellhop will handle 
    ambient noise more accurately with proper spatial modeling.
    """
    n = len(x)
    
    # Define signal band for SNR calculation
    if class_name == "submarine":
        band = (20.0, 800.0)
    else:
        band = (300.0, 8000.0)
    
    # Generate ambient noise with Wenz spectrum
    amb = generate_ambient_noise(n, fs, sea_state, shipping_level, rng)
    
    # Bandlimit ambient to relevant frequencies (slight filtering for realism)
    amb = bandlimit(amb, fs, max(1, band[0]*0.5), min(fs*0.49, band[1]*1.5))
    
    # Add white sensor self-noise (thermal, electronic)
    sens = rng.standard_normal(n).astype(np.float32) * 0.01
    
    # Total noise
    noise = amb + sens
    
    # Calculate RMS in signal band for accurate SNR
    sos = butter(6, [band[0], band[1]], btype="band", fs=fs, output="sos")
    x_filt = sosfiltfilt(sos, x)
    noise_filt = sosfiltfilt(sos, noise)
    
    sig_rms = np.sqrt(np.mean(x_filt**2) + 1e-12)
    noise_rms = np.sqrt(np.mean(noise_filt**2) + 1e-12)
    
    # Scale noise to achieve target SNR
    target_noise_rms = sig_rms / (10 ** (snr_db/20.0))
    scale = target_noise_rms / (noise_rms + 1e-12)
    noise *= scale
    
    # Add noise to signal
    y = x + noise
    
    # Normalize to prevent clipping
    y /= (max(1.0, np.max(np.abs(y)) * 1.05))
    
    return y.astype(np.float32)

def sea_state_surface_loss(sea_state):
    """
    Calculate additional surface reflection loss due to sea state.
    Returns loss in dB per surface bounce.
    
    Physics: Rough surface scattering increases with wave height.
    Loss ≈ 0.5 * SS^2 dB per bounce (empirical)
    """
    return 0.5 * sea_state**2
