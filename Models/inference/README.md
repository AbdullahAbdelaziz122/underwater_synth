# Hierarchical Underwater Acoustic Classification Pipeline

A production-ready three-stage CNN inference system for classifying
underwater acoustic signals.

---

## Decision Tree

```
                         Input Audio
                               │
                               ▼
               Stage 1: Binary Threat Detector
                               │
              ┌────────────────┴────────────────┐
              │                                 │
          Threat                           Non-Threat
              │                                 │
              ▼                                 ▼
 Stage 2: Threat Classifier        Stage 3: Family Classifier
  ┌──────────┴───────┐          ┌──────────────┴──────────────┐
  │                  │          │          │         │         │
Submarine        Torpedo   Background  Dolphin    Whale    Beluga
                           Vessel      Seal       Walrus   Narwhal
```

---

## Project Structure

```
project/
│
├── config.py           ← All parameters and paths
├── models.py           ← CNN architectures + weight loading
├── preprocessing.py    ← Audio → mel-spectrogram pipelines
├── inference.py        ← HierarchicalClassifier engine
├── main.py             ← CLI entry point
│
├── architecture.mmd    ← Mermaid architecture diagram
├── requirements.txt
│
├── Binary_model_weights.pth      ← Place here
├── Threat_model_weights.pth      ← Place here
└── best_family_classifier.pth    ← Place here
```

---

## Setup

### Jetson Nano (JetPack 4.6)

PyTorch and torchaudio must be installed using the NVIDIA ARM64 wheels:

```bash
# 1. Install PyTorch 1.9.0
sudo pip3 install torch-1.9.0-cp36-cp36m-linux_aarch64.whl

# 2. Install torchaudio 0.9.0 from source
sudo apt-get install libsox-dev libsox-fmt-all
pip3 install torchaudio==0.9.0

# 3. Install remaining dependencies
pip3 install -r requirements.txt
```

### Standard (x86 / cloud)

```bash
pip install -r requirements.txt
```

---

## Usage

```bash
python main.py sample.wav
```

### JSON output

```bash
python main.py sample.wav --json
```

### Verbose logging

```bash
python main.py sample.wav --verbose
```

### Override weight paths

```bash
python main.py sample.wav \
  --binary-weights /path/to/Binary_model_weights.pth \
  --threat-weights /path/to/Threat_model_weights.pth \
  --family-weights /path/to/best_family_classifier.pth
```

---

## Example Output

```
====================================================
   HIERARCHICAL CLASSIFICATION RESULT
====================================================

  Audio File  : sample.wav

  Stage 1 — Binary Threat Detection:
    Prediction : Threat
    Confidence : 99.84%

  Stage 2 — Threat Classification:
    Prediction : Torpedo
    Confidence : 99.12%

  ────────────────────────────────────────────────
  Final Classification : Torpedo

====================================================
```

---

## Preprocessing Pipelines

| Stage   | n_mels | n_fft | hop_length | AmplitudeToDB |
|---------|--------|-------|------------|---------------|
| 1 & 2   | 128    | 1024  | 512        | No            |
| 3       | 64     | 1024  | 512        | Yes (80 dB)   |

All stages share: 16 kHz sample rate, 5-second window, mono.

---

## Models

| Stage | Architecture       | Output        | Loss         |
|-------|--------------------|---------------|--------------|
| 1     | SimpleCNN          | 1 logit       | BCEWithLogits|
| 2     | SimpleCNN          | 1 logit       | BCEWithLogits|
| 3     | FamilyClassifierCNN| 8 logits      | CrossEntropy |
