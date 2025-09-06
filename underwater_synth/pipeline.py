from dataclasses import dataclass
import json, os
from pathlib import Path
import numpy as np
from scipy.io import wavfile
from scipy.signal import stft
from .utils import rng_from_seed
from .sources import ClassParams, synthesize
from .propagation import fast_propagate
from .sensors import add_ambient_and_sensor_noise, bandlimit

def class_params_from_cfg(name, cfg):
    c = cfg["classes"][name]
    return ClassParams(
        blades_lo=int(c["blades"][0]), blades_hi=int(c["blades"][1]),
        rpm_lo=float(c["rpm"][0]), rpm_hi=float(c["rpm"][1]),
        broad=tuple(c["bands"]["broad_hz"]), hub=tuple(c["bands"]["hub_hz"]), tip=tuple(c["bands"]["tip_hz"]),
        alpha_broad=tuple(c["alpha_broad"]), alpha_hub=tuple(c["alpha_hub"]),
        harmonics=int(c["bpf_harmonics"]),
        tip_rate_hz=tuple(c["tip_bursts"]["rate_hz"]), tip_dur_ms=tuple(c["tip_bursts"]["dur_ms"]),
        rpm_drift_pct=tuple(c["rpm_drift_pct"]), rpm_drift_period_s=tuple(c["rpm_drift_period_s"]),
    )

def render_sample(class_name, cfg, out_dir, idx, rng):
    fs = cfg["fs"]["submarine"] if class_name=="submarine" else cfg["fs"]["torpedo"]
    dur_rng = cfg["duration_s"]["submarine"] if class_name=="submarine" else cfg["duration_s"]["torpedo"]
    T = rng.uniform(*dur_rng)
    params = class_params_from_cfg(class_name, cfg)
    s, meta_src = synthesize(class_name, T, fs, params, rng)
    prop = cfg["propagation"]
    r = rng.uniform(*prop["ranges_m"])
    sd = rng.uniform(*prop["source_depth_m"])
    rd = rng.uniform(*prop["rx_depth_m"])
    ss = int(rng.integers(prop["sea_state"][0], prop["sea_state"][1]+1))
    y = fast_propagate(s, fs, r, geometry=prop.get("geometry","spherical"),
                       n_paths=3, src_depth=sd, rx_depth=rd, sea_state=ss, rng=rng)
    if class_name == "submarine":
        y = bandlimit(y, fs, 10, 1000)
    else:
        y = bandlimit(y, fs, 50, 10000)
    snr = float(rng.choice(cfg["sweeps"]["snr_dB"]))
    y = add_ambient_and_sensor_noise(y, fs, class_name, ss, snr, rng)
    audio_dir = Path(out_dir)/"audio"; audio_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{idx:06d}_{class_name}.wav"
    wavfile.write(audio_dir/fname, fs, (np.clip(y, -1, 1) * 32767).astype(np.int16))
    spec_dir = Path(out_dir)/"spec"; spec_dir.mkdir(parents=True, exist_ok=True)
    f, t, S = stft(y, fs=fs, window="hann", nperseg=1024 if fs>=16000 else 512, noverlap=None, detrend=False, return_onesided=True, boundary=None, padded=False)
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use("Agg")
        SdB = 20*np.log10(np.abs(S)+1e-12)
        plt.figure(figsize=(6,3))
        plt.pcolormesh(t, f, SdB, shading="nearest")
        plt.xlabel("Time (s)"); plt.ylabel("Freq (Hz)"); plt.title(f"Spec - {class_name}")
        plt.tight_layout()
        plt.savefig(spec_dir/f"{fname.replace('.wav','.png')}", dpi=120)
        plt.close()
    except Exception:
        pass
    md = {
        "class": class_name,
        "fs": fs,
        "duration_s": float(T),
        "range_m": float(r),
        "source_depth_m": float(sd),
        "rx_depth_m": float(rd),
        "sea_state": int(ss),
        "snr_dB": float(snr),
    }
    md.update(meta_src)
    return fname, md

def generate(cfg, out_dir, n, class_filter=None, seed=1337):
    rng = rng_from_seed(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metas = []
    for i in range(n):
        cls = class_filter if class_filter in ("submarine","torpedo") else ("submarine" if (i%2==0) else "torpedo")
        fname, md = render_sample(cls, cfg, out_dir, i, rng)
        metas.append({"file": str(Path("audio")/fname), **md})
    with open(Path(out_dir)/"metadata.jsonl","w") as f:
        for m in metas:
            f.write(json.dumps(m)+"\n")
    return metas
