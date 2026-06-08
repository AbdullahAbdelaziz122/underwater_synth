"""
models.py
---------
CNN architecture definitions and weight-loading utilities for the
Hierarchical Underwater Acoustic Classification Pipeline.

Three models are defined:
  - SimpleCNN          : used by Stage 1 (Binary) and Stage 2 (Threat)
  - FamilyClassifierCNN: used by Stage 3 (Family)

All weights are loaded via load_state_dict() — the .pth files contain
state_dicts only, not full serialised models.
"""

import logging
from pathlib import Path
from typing import Tuple, Dict, Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Architecture: SimpleCNN  (Stage 1 & Stage 2)
# ---------------------------------------------------------------------------

class SimpleCNN(nn.Module):
    """
    Lightweight CNN for binary classification of mel-spectrograms.

    Expected input shape : (batch, 1, 128, 157)
    Output shape         : (batch, 1)  — raw logit; apply sigmoid for probability

    Used by:
        Stage 1 — Binary Threat Detector  (Threat / Non-Threat)
        Stage 2 — Threat Classifier       (Submarine / Torpedo)
    """

    def __init__(self) -> None:
        super().__init__()

        # Three conv blocks, each followed by ReLU + MaxPool2d(2)
        # Input : (B, 1,  128, 157)
        # After block 1: (B, 16,  64,  78)
        # After block 2: (B, 32,  32,  39)
        # After block 3: (B, 64,  16,  19)  → flattened = 64 × 16 × 19 = 19 456
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.flatten = nn.Flatten()

        self.fc = nn.Sequential(
            nn.Linear(19456, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: mel-spectrogram tensor of shape (B, 1, 128, H)
        Returns:
            logit tensor of shape (B, 1)
        """
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


# ---------------------------------------------------------------------------
# Architecture: FamilyClassifierCNN  (Stage 3)
# ---------------------------------------------------------------------------

class FamilyClassifierCNN(nn.Module):
    """
    Deeper CNN with BatchNorm and AdaptiveAvgPool for multi-class
    family-level classification of underwater acoustic signals.

    Expected input shape : (batch, 1, 64, H)   — 64-mel spectrogram
    Output shape         : (batch, num_classes) — raw logits; apply softmax

    Used by:
        Stage 3 — Family Classifier  (8 classes)
    """

    def __init__(self, num_classes: int = 8) -> None:
        super().__init__()

        # Feature extraction backbone
        # Six conv layers grouped into three double-conv blocks,
        # each followed by MaxPool2d(2) + Dropout2d
        self.features = nn.Sequential(
            # Block 1 — 1 → 32 channels
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),

            # Block 2 — 32 → 64 channels
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),

            # Block 3 — 64 → 128 channels
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),
        )

        # Collapse spatial dims to (B, 128, 1, 1) regardless of input resolution
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: mel-spectrogram tensor of shape (B, 1, 64, H)
        Returns:
            logit tensor of shape (B, num_classes)
        """
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x


# ---------------------------------------------------------------------------
# Weight loading & verification
# ---------------------------------------------------------------------------

def verify_model_loading(
    model: nn.Module,
    weight_path: str,
    device: torch.device,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Load a state_dict checkpoint into `model`, verify compatibility,
    and return a diagnostic report.

    Args:
        model      : Instantiated (but un-loaded) PyTorch model.
        weight_path: Path to the .pth file containing the state_dict.
        device     : torch.device to map the weights onto.

    Returns:
        model  : The model with weights loaded, set to eval mode.
        report : Dictionary with loading diagnostics.

    Raises:
        FileNotFoundError : if weight_path does not exist.
        RuntimeError      : if the checkpoint cannot be parsed.
    """
    path = Path(weight_path)
    if not path.exists():
        raise FileNotFoundError(f"Weight file not found: {weight_path}")

    logger.info("Loading weights from: %s", weight_path)

    # Load checkpoint — weights_only=True avoids arbitrary code execution
    try:
        checkpoint = torch.load(
            weight_path,
            map_location=device,
            weights_only=True,
        )
    except Exception:
        # Fallback for older PyTorch versions that don't support weights_only
        checkpoint = torch.load(weight_path, map_location=device)

    # Some checkpoints wrap the state_dict in a dict with a 'state_dict' key
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and not any(
        isinstance(v, torch.Tensor) for v in checkpoint.values()
    ):
        # Checkpoint has no tensors at top level — something unexpected
        raise RuntimeError(
            f"Checkpoint at {weight_path} does not appear to contain a state_dict."
        )
    else:
        state_dict = checkpoint

    # Load with strict=False to capture missing/unexpected keys
    incompatible = model.load_state_dict(state_dict, strict=False)

    missing_keys    = incompatible.missing_keys
    unexpected_keys = incompatible.unexpected_keys

    # Count parameters
    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    report: Dict[str, Any] = {
        "weight_file"      : str(path.resolve()),
        "missing_keys"     : missing_keys,
        "unexpected_keys"  : unexpected_keys,
        "total_params"     : total_params,
        "trainable_params" : trainable_params,
        "success"          : len(missing_keys) == 0 and len(unexpected_keys) == 0,
    }

    # Log the diagnostics
    if report["success"]:
        logger.info(
            "✓ Weights loaded successfully. Parameters: %s", f"{total_params:,}"
        )
    else:
        if missing_keys:
            logger.warning("Missing keys  (%d): %s", len(missing_keys), missing_keys)
        if unexpected_keys:
            logger.warning(
                "Unexpected keys (%d): %s", len(unexpected_keys), unexpected_keys
            )

    model.eval()
    return model, report


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_binary_model(weight_path: str, device: torch.device) -> Tuple[nn.Module, Dict]:
    """Construct SimpleCNN and load Binary Threat Detector weights."""
    model = SimpleCNN().to(device)
    return verify_model_loading(model, weight_path, device)


def build_threat_model(weight_path: str, device: torch.device) -> Tuple[nn.Module, Dict]:
    """Construct SimpleCNN and load Threat Classifier weights."""
    model = SimpleCNN().to(device)
    return verify_model_loading(model, weight_path, device)


def build_family_model(
    weight_path: str,
    device: torch.device,
    num_classes: int = 8,
) -> Tuple[nn.Module, Dict]:
    """Construct FamilyClassifierCNN and load Family Classifier weights."""
    model = FamilyClassifierCNN(num_classes=num_classes).to(device)
    return verify_model_loading(model, weight_path, device)
