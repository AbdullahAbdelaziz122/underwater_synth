"""
preprocessing.py
----------------
Audio loading and spectrogram preprocessing for the Hierarchical
Underwater Acoustic Classification Pipeline.

Two distinct mel-spectrogram pipelines are implemented:

  Pipeline A — 128-mel  : used by Stage 1 (Binary) and Stage 2 (Threat)
  Pipeline B —  64-mel  : used by Stage 3 (Family), includes AmplitudeToDB

Both pipelines share the same audio loading, resampling, mono conversion,
and pad/trim logic, which live in private helper methods.
"""

import logging
from pathlib import Path
from typing import Tuple

import torch
import torchaudio
import torchaudio.transforms as T

# soundfile is used as a robust fallback for loading WAV files on platforms
# where torchaudio's backend may have limited codec support (e.g. Jetson Nano
# with older FFmpeg, or newer torchaudio versions requiring torchcodec).
try:
    import soundfile as _soundfile
    _HAS_SOUNDFILE = True
except ImportError:
    _HAS_SOUNDFILE = False

from config import BINARY_CONFIG, FAMILY_CONFIG, BinaryConfig, FamilyConfig

logger = logging.getLogger(__name__)


class AudioPreprocessor:
    """
    Stateless audio preprocessor.

    Reads a .wav file and returns a normalised mel-spectrogram tensor
    ready to be fed into the appropriate CNN stage.

    Usage
    -----
    >>> preprocessor = AudioPreprocessor(device=torch.device("cpu"))
    >>> spec_128 = preprocessor.preprocess_binary("audio.wav")   # Stage 1 / 2
    >>> spec_64  = preprocessor.preprocess_family("audio.wav")   # Stage 3
    """

    def __init__(self, device: torch.device) -> None:
        self.device = device

        # ------------------------------------------------------------------
        # Pipeline A — 128-mel (Stage 1 & Stage 2)
        # ------------------------------------------------------------------
        self._mel_128 = T.MelSpectrogram(
            sample_rate=BINARY_CONFIG.sample_rate,
            n_fft=BINARY_CONFIG.n_fft,
            hop_length=BINARY_CONFIG.hop_length,
            n_mels=BINARY_CONFIG.n_mels,
        ).to(device)

        # ------------------------------------------------------------------
        # Pipeline B — 64-mel + AmplitudeToDB (Stage 3)
        # ------------------------------------------------------------------
        self._mel_64 = T.MelSpectrogram(
            sample_rate=FAMILY_CONFIG.sample_rate,
            n_fft=FAMILY_CONFIG.n_fft,
            hop_length=FAMILY_CONFIG.hop_length,
            n_mels=FAMILY_CONFIG.n_mels,
        ).to(device)

        self._amplitude_to_db = T.AmplitudeToDB(
            top_db=FAMILY_CONFIG.top_db
        ).to(device)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def preprocess_binary(self, audio_path: str) -> torch.Tensor:
        """
        Preprocess an audio file for Stage 1 — Binary Threat Detector.

        Pipeline:
            Load WAV → Resample to 16 kHz → Mono → Pad/Trim to 5 s
            → MelSpectrogram(n_mels=128) → shape (1, 1, 128, 157)

        Args:
            audio_path: Path to the input .wav file.

        Returns:
            Tensor of shape (1, 1, 128, 157) on self.device.
        """
        waveform = self._load_and_prepare(audio_path, BINARY_CONFIG)
        spec = self._mel_128(waveform)             # (1, 128, time)
        spec = spec.unsqueeze(0)                   # (1, 1, 128, time) — add batch
        logger.debug("Binary spectrogram shape: %s", spec.shape)
        return spec

    def preprocess_threat(self, audio_path: str) -> torch.Tensor:
        """
        Preprocess an audio file for Stage 2 — Threat Classifier.

        Identical pipeline to preprocess_binary; provided as a separate
        method for clarity and future flexibility.

        Returns:
            Tensor of shape (1, 1, 128, 157) on self.device.
        """
        return self.preprocess_binary(audio_path)

    def preprocess_family(self, audio_path: str) -> torch.Tensor:
        """
        Preprocess an audio file for Stage 3 — Family Classifier.

        Pipeline:
            Load WAV → Resample to 16 kHz → Mono → Pad/Trim to 5 s
            → MelSpectrogram(n_mels=64) → AmplitudeToDB(top_db=80)
            → shape (1, 1, 64, 157)

        IMPORTANT: This is a completely separate pipeline from Stage 1/2.
        The same audio file must be processed independently through both
        pipelines when needed.

        Args:
            audio_path: Path to the input .wav file.

        Returns:
            Tensor of shape (1, 1, 64, 157) on self.device.
        """
        waveform = self._load_and_prepare(audio_path, FAMILY_CONFIG)
        spec = self._mel_64(waveform)              # (1, 64, time)
        spec = self._amplitude_to_db(spec)         # convert power → dB scale
        spec = spec.unsqueeze(0)                   # (1, 1, 64, time)
        logger.debug("Family spectrogram shape: %s", spec.shape)
        return spec

    # ------------------------------------------------------------------
    # Shared private utilities
    # ------------------------------------------------------------------

    def _load_and_prepare(
        self,
        audio_path: str,
        cfg,
    ) -> torch.Tensor:
        """
        Load a .wav file and apply resampling, mono conversion, and
        fixed-length windowing.

        Args:
            audio_path: Path to the .wav file.
            cfg       : A BinaryConfig or FamilyConfig instance.

        Returns:
            Waveform tensor of shape (1, num_samples) on self.device.
        """
        waveform, original_sr = self.load_audio(audio_path)
        waveform = self.resample(waveform, original_sr, cfg.sample_rate)
        waveform = self.convert_mono(waveform)
        waveform = self.pad_or_trim(waveform, cfg.num_samples)
        return waveform.to(self.device)

    def load_audio(self, audio_path: str) -> Tuple[torch.Tensor, int]:
        """
        Load a .wav file from disk.

        Args:
            audio_path: Path to the audio file.

        Returns:
            (waveform, sample_rate)
            waveform shape: (channels, samples)

        Raises:
            FileNotFoundError: if the file does not exist.
            RuntimeError     : if torchaudio cannot decode the file.
        """
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        try:
            waveform, sample_rate = torchaudio.load(str(path))
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load audio file '{audio_path}': {exc}"
            ) from exc

        logger.debug(
            "Loaded '%s' — sr=%d, shape=%s", path.name, sample_rate, waveform.shape
        )
        return waveform, sample_rate

    def resample(
        self,
        waveform: torch.Tensor,
        original_sr: int,
        target_sr: int,
    ) -> torch.Tensor:
        """
        Resample waveform to target_sr if necessary.

        Args:
            waveform   : Tensor of shape (channels, samples).
            original_sr: Source sample rate.
            target_sr  : Desired sample rate.

        Returns:
            Resampled waveform tensor.
        """
        if original_sr == target_sr:
            return waveform

        resampler = T.Resample(
            orig_freq=original_sr,
            new_freq=target_sr,
        ).to(waveform.device)

        resampled = resampler(waveform)
        logger.debug("Resampled %d Hz → %d Hz", original_sr, target_sr)
        return resampled

    def convert_mono(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Convert a multi-channel waveform to mono by averaging channels.

        Args:
            waveform: Tensor of shape (channels, samples).

        Returns:
            Tensor of shape (1, samples).
        """
        if waveform.shape[0] == 1:
            return waveform

        mono = waveform.mean(dim=0, keepdim=True)
        logger.debug("Converted %d channels → mono", waveform.shape[0])
        return mono

    def pad_or_trim(
        self,
        waveform: torch.Tensor,
        target_length: int,
    ) -> torch.Tensor:
        """
        Pad (with zeros) or trim the waveform to exactly target_length samples.

        Args:
            waveform      : Tensor of shape (1, samples).
            target_length : Desired number of samples.

        Returns:
            Tensor of shape (1, target_length).
        """
        current_length = waveform.shape[-1]

        if current_length == target_length:
            return waveform

        if current_length > target_length:
            # Trim from the right
            trimmed = waveform[..., :target_length]
            logger.debug(
                "Trimmed waveform: %d → %d samples", current_length, target_length
            )
            return trimmed

        # Pad with zeros on the right
        pad_amount = target_length - current_length
        padded = torch.nn.functional.pad(waveform, (0, pad_amount))
        logger.debug(
            "Padded waveform: %d → %d samples (added %d zeros)",
            current_length,
            target_length,
            pad_amount,
        )
        return padded
