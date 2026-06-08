"""
main.py
-------
Command-line entry point for the Hierarchical Underwater Acoustic
Classification Pipeline.

Usage
-----
    python main.py <path_to_audio.wav>

    python main.py sample.wav
    python main.py /data/recordings/torpedo_test.wav

Output
------
    ==================================================
    HIERARCHICAL CLASSIFICATION RESULT
    ==================================================

    Audio File  : sample.wav

    Stage 1:
      Prediction : Threat
      Confidence : 99.84%

    Stage 2:
      Prediction : Torpedo
      Confidence : 99.12%

    Final Classification:
      Torpedo

    ==================================================
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any

from inference import HierarchicalClassifier

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def _configure_logging(verbose: bool = False) -> None:
    """Configure logging to stdout with optional verbose level."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="[%(levelname)s] %(name)s — %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

SEPARATOR = "=" * 52

def _format_result(result: Dict[str, Any]) -> str:
    """
    Format a classification result dictionary into a human-readable string.

    Args:
        result: Dictionary returned by HierarchicalClassifier.predict()

    Returns:
        Multi-line formatted string.
    """
    lines = [
        "",
        SEPARATOR,
        "   HIERARCHICAL CLASSIFICATION RESULT",
        SEPARATOR,
        "",
        f"  Audio File  : {result['audio_file']}",
        "",
    ]

    # Stage 1 — always present
    s1 = result["stage1"]
    lines += [
        "  Stage 1 — Binary Threat Detection:",
        f"    Prediction : {s1['prediction']}",
        f"    Confidence : {s1['confidence'] * 100:.2f}%",
        "",
    ]

    # Stage 2 or Stage 3 depending on routing
    if "stage2" in result:
        s2 = result["stage2"]
        lines += [
            "  Stage 2 — Threat Classification:",
            f"    Prediction : {s2['prediction']}",
            f"    Confidence : {s2['confidence'] * 100:.2f}%",
            "",
        ]
    elif "stage3" in result:
        s3 = result["stage3"]
        lines += [
            "  Stage 3 — Family Classification:",
            f"    Prediction : {s3['prediction']}",
            f"    Confidence : {s3['confidence'] * 100:.2f}%",
            "",
        ]

    lines += [
        "  ─" * 26,
        f"  Final Classification : {result['final_class']}",
        "",
        SEPARATOR,
        "",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description=(
            "Hierarchical Underwater Acoustic Classification Pipeline\n"
            "Classifies a .wav file through a three-stage CNN decision tree."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "audio",
        type=str,
        help="Path to the input .wav file.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Output the result as JSON instead of formatted text.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=False,
        help="Enable verbose/debug logging.",
    )
    parser.add_argument(
        "--binary-weights",
        type=str,
        default=None,
        help="Override path to Binary_model_weights.pth.",
    )
    parser.add_argument(
        "--threat-weights",
        type=str,
        default=None,
        help="Override path to Threat_model_weights.pth.",
    )
    parser.add_argument(
        "--family-weights",
        type=str,
        default=None,
        help="Override path to best_family_classifier.pth.",
    )

    return parser


def main() -> int:
    """
    Entry point.

    Returns:
        0 on success, non-zero on failure.
    """
    parser  = _build_parser()
    args    = parser.parse_args()

    _configure_logging(verbose=args.verbose)
    logger = logging.getLogger(__name__)

    # Validate audio path
    audio_path = Path(args.audio)
    if not audio_path.exists():
        logger.error("Audio file not found: %s", audio_path)
        return 1

    # Build classifier — allow weight path overrides from CLI
    kwargs: Dict[str, Any] = {}
    if args.binary_weights:
        kwargs["binary_weights"] = args.binary_weights
    if args.threat_weights:
        kwargs["threat_weights"] = args.threat_weights
    if args.family_weights:
        kwargs["family_weights"] = args.family_weights

    try:
        classifier = HierarchicalClassifier(**kwargs)
        classifier.load_models()
    except FileNotFoundError as exc:
        logger.error("Model weight file missing: %s", exc)
        logger.error(
            "Ensure Binary_model_weights.pth, Threat_model_weights.pth, "
            "and best_family_classifier.pth are in the project directory."
        )
        return 2
    except Exception as exc:
        logger.error("Failed to load models: %s", exc)
        return 2

    # Run inference
    try:
        result = classifier.predict(str(audio_path))
    except Exception as exc:
        logger.error("Inference failed: %s", exc)
        return 3

    # Print output
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(_format_result(result))

    return 0


if __name__ == "__main__":
    sys.exit(main())
