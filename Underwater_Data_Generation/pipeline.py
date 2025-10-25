from dataclasses import dataclass
import json, os
from pathlib import Path
import numpy as np
from scipy.io import wavfile
from scipy.signal import stft
from utils import rng_from_seed
from sources import ClassParams, synthesize
from sensors import add_ambient_and_sensor_noise, bandlimit

def class_params_from_cfg(name, cfg):
    """Extract class parameters from config."""
    c = cfg["classes"][name]
    return ClassParams(
        blades_lo=int(c["blades"][0]), 
        blades_hi=int(c["blades"][1]),
        rpm_lo=float(c["rpm"][0]), 
        rpm_hi=float(c["rpm"][1]),
        broad=tuple(c["bands"]["broad_hz"]), 
        hub=tuple(c["bands"]["hub_hz"]), 
        tip=tuple(c["bands"]["tip_hz"]),
        alpha_broad=tuple(c["alpha_broad"]), 
        alpha_hub=tuple(c["alpha_hub"]),
        harmonics=int(c["bpf_harmonics"]),
        tip_rate_base_hz=tuple(c["tip_bursts"]["rate_base_hz"]),
        tip_rpm_threshold=float(c["tip_bursts"]["rpm_threshold"]),
        tip_rate_exponent=float(c["tip_bursts"]["rate_exponent"]),
        tip_dur_ms=tuple(c["tip_bursts"]["dur_ms"]),
        rpm_drift_pct=tuple(c["rpm_drift_pct"]), 
        rpm_drift_period_s=tuple(c["rpm_drift_period_s"]),
        rpm_walk_std=float(c["rpm_walk_std"]),
        transient_prob=float(c["transient_prob"]),
    )

def render_sample(class_name, cfg, out_dir, idx, rng, bellhop_mode=False):
    """
    Render a single sample.
    
    bellhop_mode: If True, outputs clean source signal only (no propagation/noise).
                  If False, applies simplified propagation and noise for Stage 1 validation.
    """
    # Sampling rate
    fs = cfg["fs"]["submarine"] if class_name=="submarine" else cfg["fs"]["torpedo"]
    
    # Duration
    dur_rng = cfg["duration_s"]["submarine"] if class_name=="submarine" else cfg["duration_s"]["torpedo"]
    T = rng.uniform(*dur_rng)
    
    # Doppler shift (if target is moving)
    doppler_vel = None
    if cfg.get("doppler", {}).get("enabled", False):
        vel_range = cfg["doppler"]["velocity_ms"]
        doppler_vel = rng.uniform(*vel_range)
    
    # Synthesize source signal
    params = class_params_from_cfg(class_name, cfg)
    s, meta_src = synthesize(class_name, T, fs, params, rng, doppler_velocity_ms=doppler_vel)
    
    # Environmental parameters (for metadata, used later in Bellhop)
    prop = cfg["propagation"]
    r = rng.uniform(*prop["ranges_m"])
    sd = rng.uniform(*prop["source_depth_m"])
    rd = rng.uniform(*prop["rx_depth_m"])
    ss = int(rng.integers(prop["sea_state"][0], prop["sea_state"][1]+1))
    water_depth = rng.uniform(*prop.get("water_depth_m", [100, 1000]))
    bottom_type = rng.choice(prop.get("bottom_types", ["sand", "mud", "rock"]))
    
    if bellhop_mode:
        # Output clean source signal only
        y = s
        actual_snr = None
    else:
        # Stage 1 validation: apply basic bandlimiting and noise
        # (In Stage 2, Bellhop will handle propagation properly)
        if class_name == "submarine":
            y = bandlimit(s, fs, 10, 1000)
        else:
            y = bandlimit(s, fs, 50, 10000)
        
        snr = float(rng.choice(cfg["sweeps"]["snr_dB"]))
        shipping = cfg.get("ambient", {}).get("shipping_level", 5)
        y = add_ambient_and_sensor_noise(y, fs, class_name, ss, snr, shipping, rng)
        actual_snr = snr
    
    # Save audio
    audio_dir = Path(out_dir)/"audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{idx:06d}_{class_name}.wav"
    wavfile.write(audio_dir/fname, fs, (np.clip(y, -1, 1) * 32767).astype(np.int16))
    
    # Generate spectrogram
    spec_dir = Path(out_dir)/"spec"
    spec_dir.mkdir(parents=True, exist_ok=True)
    f, t, S = stft(y, fs=fs, window="hann", 
                   nperseg=1024 if fs>=16000 else 512, 
                   noverlap=None, detrend=False, 
                   return_onesided=True, boundary=None, padded=False)
    
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use("Agg")
        SdB = 20*np.log10(np.abs(S)+1e-12)
        plt.figure(figsize=(8,4))
        plt.pcolormesh(t, f, SdB, shading="nearest", cmap="viridis")
        plt.colorbar(label="dB")
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (Hz)")
        plt.title(f"{class_name} - RPM: {meta_src['RPM_mean']:.1f}")
        plt.tight_layout()
        plt.savefig(spec_dir/f"{fname.replace('.wav','.png')}", dpi=120)
        plt.close()
    except Exception as e:
        print(f"Warning: Could not generate spectrogram: {e}")
    
    # Metadata
    md = {
        "class": class_name,
        "fs": int(fs),
        "duration_s": float(T),
        "range_m": float(r),
        "source_depth_m": float(sd),
        "rx_depth_m": float(rd),
        "water_depth_m": float(water_depth),
        "bottom_type": bottom_type,
        "sea_state": int(ss),
        "bellhop_mode": bellhop_mode,
    }
    
    if actual_snr is not None:
        md["snr_dB"] = float(actual_snr)
    
    # Add source metadata
    md.update(meta_src)
    
    return fname, md

def generate(cfg, out_dir, n, class_filter=None, seed=1337, bellhop_mode=False):
    """
    Generate dataset.
    
    Parameters:
    - cfg: Configuration dictionary
    - out_dir: Output directory
    - n: Number of samples to generate
    - class_filter: Generate only this class (None for alternating)
    - seed: Random seed
    - bellhop_mode: If True, output clean source signals for Bellhop Stage 2
    """
    rng = rng_from_seed(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    metas = []
    for i in range(n):
        cls = class_filter if class_filter in ("submarine","torpedo") else ("submarine" if (i%2==0) else "torpedo")
        fname, md = render_sample(cls, cfg, out_dir, i, rng, bellhop_mode=bellhop_mode)
        metas.append({"file": str(Path("audio")/fname), **md})
    
    # Save metadata
    with open(Path(out_dir)/"metadata.jsonl","w") as f:
        for m in metas:
            f.write(json.dumps(m)+"\n")
    
    # Save config for reference
    import yaml
    with open(Path(out_dir)/"config_used.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    
    return metas
