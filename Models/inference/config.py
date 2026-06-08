"""
config.py
---------
Central configuration file for the Hierarchical Underwater Acoustic
Classification Pipeline.

All preprocessing parameters, model paths, and class mappings live here.
No hardcoded values should appear in any other module.
"""

import os
from dataclasses import dataclass, field
from typing import List

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

BINARY_WEIGHTS_PATH  = os.path.join(BASE_DIR, "Binary_model_weights.pth")
THREAT_WEIGHTS_PATH  = os.path.join(BASE_DIR, "Threat_model_weights.pth")
FAMILY_WEIGHTS_PATH  = os.path.join(BASE_DIR, "best_family_classifier.pth")

# ---------------------------------------------------------------------------
# Stage 1 — Binary Threat Detector
# ---------------------------------------------------------------------------

@dataclass
class BinaryConfig:
    """Preprocessing and inference configuration for Stage 1."""

    # Audio
    sample_rate:  int   = 16000
    duration_ms:  int   = 5000
    channels:     int   = 1

    # Spectrogram
    n_mels:       int   = 128
    n_fft:        int   = 1024
    hop_length:   int   = 512

    # Architecture
    linear_in:    int   = 19456   # 64 channels × 19 × 16 spatial
    hidden:       int   = 128
    output_size:  int   = 1

    # Inference
    threshold:    float = 0.5

    # Labels
    labels: List[str] = field(default_factory=lambda: ["Non-Threat", "Threat"])

    @property
    def num_samples(self) -> int:
        """Total samples for the fixed-length audio window."""
        return int(self.sample_rate * self.duration_ms / 1000)


# ---------------------------------------------------------------------------
# Stage 2 — Threat Classifier
# ---------------------------------------------------------------------------

@dataclass
class ThreatConfig:
    """
    Preprocessing and inference configuration for Stage 2.
    Identical spectrogram pipeline to Stage 1 — different label mapping.
    """

    # Audio (same as Binary)
    sample_rate:  int   = 16000
    duration_ms:  int   = 5000
    channels:     int   = 1

    # Spectrogram (same as Binary)
    n_mels:       int   = 128
    n_fft:        int   = 1024
    hop_length:   int   = 512

    # Architecture (same SimpleCNN)
    linear_in:    int   = 19456
    hidden:       int   = 128
    output_size:  int   = 1

    # Inference
    threshold:    float = 0.5

    # Labels
    # sigmoid < threshold → class 0 (Submarine)
    # sigmoid ≥ threshold → class 1 (Torpedo)
    labels: List[str] = field(default_factory=lambda: ["Submarine", "Torpedo"])

    @property
    def num_samples(self) -> int:
        return int(self.sample_rate * self.duration_ms / 1000)


# ---------------------------------------------------------------------------
# Stage 3 — Family Classifier
# ---------------------------------------------------------------------------

@dataclass
class FamilyConfig:
    """
    Preprocessing and inference configuration for Stage 3.
    Uses a DIFFERENT spectrogram pipeline (64-mel, with AmplitudeToDB).
    """

    # Audio
    sample_rate:  int   = 16000
    duration_ms:  int   = 5000
    channels:     int   = 1

    # Spectrogram — DIFFERENT from Stage 1/2
    n_mels:       int   = 64
    n_fft:        int   = 1024
    hop_length:   int   = 512
    top_db:       float = 80.0

    # Architecture
    num_classes:  int   = 8

    @property
    def num_samples(self) -> int:
        return int(self.sample_rate * self.duration_ms / 1000)


# ---------------------------------------------------------------------------
# Family class mapping
# ---------------------------------------------------------------------------

FAMILY_CLASSES: List[str] = [
    "Background",
    "Beluga",
    "Dolphin",
    "Narwhal",
    "Seal",
    "Vessel",
    "Walrus",
    "Whale",
]

# Convenience: instantiated config singletons used throughout the project
BINARY_CONFIG = BinaryConfig()
THREAT_CONFIG = ThreatConfig()
FAMILY_CONFIG = FamilyConfig()
