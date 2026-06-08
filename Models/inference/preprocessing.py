"""
preprocessing.py (Librosa Backend)
----------------------------------
Audio loading and spectrogram preprocessing bypassing torchaudio.
Uses Librosa to generate the exact same mel-spectrogram tensors.
"""

import logging
import torch
import librosa
import numpy as np

from config import BINARY_CONFIG, FAMILY_CONFIG

logger = logging.getLogger(__name__)

class AudioPreprocessor:
    """
    Stateless audio preprocessor using Librosa.
    Returns normalized mel-spectrogram tensors ready for the CNN stages.
    """

    def __init__(self, device: torch.device) -> None:
        self.device = device
        logger.info("Initialized Librosa AudioPreprocessor (Torchaudio Bypassed)")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def preprocess_binary(self, audio_path: str) -> torch.Tensor:
        """Pipeline for Stage 1 — 128-mel spectrogram"""
        waveform = self._load_and_prepare(audio_path, BINARY_CONFIG)
        
        # Generate Mel Spectrogram
        S = librosa.feature.melspectrogram(
            y=waveform,
            sr=BINARY_CONFIG.sample_rate,
            n_fft=BINARY_CONFIG.n_fft,
            hop_length=BINARY_CONFIG.hop_length,
            n_mels=BINARY_CONFIG.n_mels,
            power=2.0
        )
        
        # Librosa returns shape (n_mels, time). Convert to tensor and add batch/channel dims.
        spec = torch.tensor(S, dtype=torch.float32, device=self.device)
        spec = spec.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, 128, 157)
        logger.debug("Binary spectrogram shape: %s", spec.shape)
        return spec

    def preprocess_threat(self, audio_path: str) -> torch.Tensor:
        """Pipeline for Stage 2 — Identical to Stage 1"""
        return self.preprocess_binary(audio_path)

    def preprocess_family(self, audio_path: str) -> torch.Tensor:
        """Pipeline for Stage 3 — 64-mel spectrogram with AmplitudeToDB"""
        waveform = self._load_and_prepare(audio_path, FAMILY_CONFIG)
        
        S = librosa.feature.melspectrogram(
            y=waveform,
            sr=FAMILY_CONFIG.sample_rate,
            n_fft=FAMILY_CONFIG.n_fft,
            hop_length=FAMILY_CONFIG.hop_length,
            n_mels=FAMILY_CONFIG.n_mels,
            power=2.0
        )
        
        # Convert power to dB scale
        S_db = librosa.power_to_db(S, top_db=FAMILY_CONFIG.top_db)
        
        spec = torch.tensor(S_db, dtype=torch.float32, device=self.device)
        spec = spec.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, 64, 157)
        logger.debug("Family spectrogram shape: %s", spec.shape)
        return spec

    # ------------------------------------------------------------------
    # Shared private utilities
    # ------------------------------------------------------------------

    def _load_and_prepare(self, audio_path: str, cfg) -> np.ndarray:
        """
        Loads a .wav file, resamples, converts to mono, and pads/trims 
        to the exact sample length in a single pass.
        """
        # Librosa handles reading, resampling, and mono-mixing automatically
        try:
            waveform, _ = librosa.load(
                audio_path, 
                sr=cfg.sample_rate, 
                mono=True
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to load audio file '{audio_path}': {exc}")

        # Pad (with zeros) or trim to exact target length
        waveform_fixed = librosa.util.fix_length(waveform, size=cfg.num_samples)
        
        return waveform_fixed