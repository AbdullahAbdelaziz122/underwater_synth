"""
inference.py
------------
Hierarchical classification engine for underwater acoustic signals.

Decision flow:

    Audio
      ↓
    Stage 1 — Binary Threat Detector
      ↓
    ┌──────── Threat? ─────────┐
    │ YES                      │ NO
    ↓                          ↓
    Stage 2                   Stage 3
    Threat Classifier         Family Classifier
    (Submarine / Torpedo)     (8 marine classes)
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn.functional as F

from config import (
    BINARY_CONFIG,
    THREAT_CONFIG,
    FAMILY_CONFIG,
    FAMILY_CLASSES,
    BINARY_WEIGHTS_PATH,
    THREAT_WEIGHTS_PATH,
    FAMILY_WEIGHTS_PATH,
)
from models import build_binary_model, build_threat_model, build_family_model
from preprocessing import AudioPreprocessor

logger = logging.getLogger(__name__)


class HierarchicalClassifier:
    """
    Three-stage hierarchical underwater acoustic classifier.

    Attributes:
        device       : torch.device (CPU or CUDA) used for all inference.
        preprocessor : AudioPreprocessor instance.
        binary_model : Stage 1 SimpleCNN — Threat / Non-Threat.
        threat_model : Stage 2 SimpleCNN — Submarine / Torpedo.
        family_model : Stage 3 FamilyClassifierCNN — 8 marine classes.

    Usage
    -----
    >>> classifier = HierarchicalClassifier()
    >>> result = classifier.predict("sample.wav")
    >>> print(result["final_class"])
    """

    def __init__(
        self,
        binary_weights: str  = BINARY_WEIGHTS_PATH,
        threat_weights: str  = THREAT_WEIGHTS_PATH,
        family_weights: str  = FAMILY_WEIGHTS_PATH,
        device: Optional[torch.device] = None,
    ) -> None:
        """
        Initialise the classifier but do NOT load models yet.
        Call load_models() before calling predict().

        Args:
            binary_weights: Path to Binary_model_weights.pth
            threat_weights: Path to Threat_model_weights.pth
            family_weights: Path to best_family_classifier.pth
            device        : Target device. Auto-detected if None.
        """
        self.binary_weights = binary_weights
        self.threat_weights = threat_weights
        self.family_weights = family_weights

        # Auto-select device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        logger.info("Using device: %s", self.device)

        self.preprocessor: Optional[AudioPreprocessor] = None
        self.binary_model = None
        self.threat_model = None
        self.family_model = None

        self._models_loaded: bool = False

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_models(self) -> Dict[str, Dict[str, Any]]:
        """
        Load and verify all three model weight files.

        Returns:
            A dict mapping model name → loading report (keys, param counts, etc.)

        Raises:
            FileNotFoundError: if any .pth file is missing.
        """
        logger.info("=" * 60)
        logger.info("Loading models onto: %s", self.device)
        logger.info("=" * 60)

        reports: Dict[str, Dict[str, Any]] = {}

        # Stage 1 — Binary Threat Detector
        logger.info("[Stage 1] Loading Binary Threat Detector...")
        self.binary_model, reports["binary"] = build_binary_model(
            self.binary_weights, self.device
        )

        # Stage 2 — Threat Classifier
        logger.info("[Stage 2] Loading Threat Classifier...")
        self.threat_model, reports["threat"] = build_threat_model(
            self.threat_weights, self.device
        )

        # Stage 3 — Family Classifier
        logger.info("[Stage 3] Loading Family Classifier...")
        self.family_model, reports["family"] = build_family_model(
            self.family_weights, self.device, num_classes=FAMILY_CONFIG.num_classes
        )

        # Instantiate preprocessor
        self.preprocessor = AudioPreprocessor(device=self.device)

        self._models_loaded = True

        # Summary
        for name, report in reports.items():
            status = "✓ OK" if report["success"] else "✗ ISSUES DETECTED"
            logger.info(
                "[%s] %s — %s total params",
                name,
                status,
                f"{report['total_params']:,}",
            )
            if not report["success"]:
                logger.warning(
                    "  Missing keys   : %s", report["missing_keys"]
                )
                logger.warning(
                    "  Unexpected keys: %s", report["unexpected_keys"]
                )

        logger.info("All models loaded successfully.")
        return reports

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, audio_path: str) -> Dict[str, Any]:
        """
        Run the full hierarchical classification pipeline on one audio file.

        Args:
            audio_path: Path to a .wav file.

        Returns:
            Structured result dictionary. See module docstring for schema.

        Raises:
            RuntimeError     : if load_models() has not been called.
            FileNotFoundError: if audio_path does not exist.
        """
        if not self._models_loaded:
            raise RuntimeError(
                "Models have not been loaded. Call load_models() first."
            )

        audio_path = str(audio_path)
        logger.info("Processing: %s", Path(audio_path).name)

        result: Dict[str, Any] = {
            "audio_file": Path(audio_path).name,
        }

        # ------------------------------------------------------------------
        # Stage 1 — Binary Threat Detection
        # ------------------------------------------------------------------
        stage1_pred, stage1_conf = self._run_binary(audio_path)
        result["stage1"] = {
            "prediction": stage1_pred,
            "confidence": round(stage1_conf, 4),
        }
        logger.info(
            "[Stage 1] %s (confidence: %.2f%%)",
            stage1_pred,
            stage1_conf * 100,
        )

        # ------------------------------------------------------------------
        # Stage 2 or Stage 3 depending on Stage 1 outcome
        # ------------------------------------------------------------------
        if stage1_pred == "Threat":
            # Route to Stage 2
            stage2_pred, stage2_conf = self._run_threat(audio_path)
            result["stage2"] = {
                "prediction": stage2_pred,
                "confidence": round(stage2_conf, 4),
            }
            result["final_class"] = stage2_pred
            logger.info(
                "[Stage 2] %s (confidence: %.2f%%)",
                stage2_pred,
                stage2_conf * 100,
            )
        else:
            # Route to Stage 3
            stage3_pred, stage3_conf = self._run_family(audio_path)
            result["stage3"] = {
                "prediction": stage3_pred,
                "confidence": round(stage3_conf, 4),
            }
            result["final_class"] = stage3_pred
            logger.info(
                "[Stage 3] %s (confidence: %.2f%%)",
                stage3_pred,
                stage3_conf * 100,
            )

        logger.info("Final classification: %s", result["final_class"])
        return result

    # ------------------------------------------------------------------
    # Private stage runners
    # ------------------------------------------------------------------

    def _run_binary(self, audio_path: str):
        """
        Run Stage 1 — Binary Threat Detector.

        Returns:
            (label, confidence) where label ∈ {"Threat", "Non-Threat"}
            and confidence is the sigmoid probability of the winning class.
        """
        spectrogram = self.preprocessor.preprocess_binary(audio_path)

        with torch.no_grad():
            logits = self.binary_model(spectrogram)           # (1, 1)
            prob   = torch.sigmoid(logits).squeeze().item()   # scalar in [0, 1]

        # prob ≥ threshold → Threat (class 1)
        # prob <  threshold → Non-Threat (class 0)
        if prob >= BINARY_CONFIG.threshold:
            label      = BINARY_CONFIG.labels[1]  # "Threat"
            confidence = prob
        else:
            label      = BINARY_CONFIG.labels[0]  # "Non-Threat"
            confidence = 1.0 - prob

        return label, confidence

    def _run_threat(self, audio_path: str):
        """
        Run Stage 2 — Threat Classifier.

        Returns:
            (label, confidence) where label ∈ {"Submarine", "Torpedo"}
        """
        spectrogram = self.preprocessor.preprocess_threat(audio_path)

        with torch.no_grad():
            logits = self.threat_model(spectrogram)
            prob   = torch.sigmoid(logits).squeeze().item()

        if prob >= THREAT_CONFIG.threshold:
            label      = THREAT_CONFIG.labels[1]  # "Torpedo"
            confidence = prob
        else:
            label      = THREAT_CONFIG.labels[0]  # "Submarine"
            confidence = 1.0 - prob

        return label, confidence

    def _run_family(self, audio_path: str):
        """
        Run Stage 3 — Family Classifier.

        Uses the SEPARATE 64-mel preprocessing pipeline.

        Returns:
            (label, confidence) where label is one of FAMILY_CLASSES
            and confidence is the softmax probability of the top class.
        """
        # IMPORTANT: use the family-specific 64-mel pipeline, NOT the
        # 128-mel pipeline used by Stage 1/2.
        spectrogram = self.preprocessor.preprocess_family(audio_path)

        with torch.no_grad():
            logits      = self.family_model(spectrogram)          # (1, 8)
            probs       = F.softmax(logits, dim=1)                # (1, 8)
            confidence, class_idx = probs.max(dim=1)

        label      = FAMILY_CLASSES[class_idx.item()]
        confidence = confidence.item()

        return label, confidence
