import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import soundfile as sf


WEIGHT_FILE = "./best_family_classifier.pth"
AUDIO_FILE = "./inferencedata/walrus.wav"

FAMILY_CLASSES = [
    "Background",
    "Beluga",
    "Dolphin",
    "Narwhal",
    "Seal",
    "Vessel",
    "Walrus",
    "Whale"
]


class FamilyClassifierCNN(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

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

            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x


def preprocess_audio(audio_path: str):
    if not os.path.exists(audio_path):
        raise FileNotFoundError(audio_path)

    print(f"Loading audio: {audio_path}")

    audio, sr = sf.read(audio_path)

    print("Original SR:", sr)
    print("Original Shape:", audio.shape)

    waveform = torch.tensor(
        audio,
        dtype=torch.float32
    )

    # Stereo -> Mono
    if waveform.ndim > 1:
        waveform = waveform.mean(dim=1)

    waveform = waveform.unsqueeze(0)

    # Resample
    if sr != 16000:
        print(f"Resampling {sr} -> 16000")

        waveform = torchaudio.transforms.Resample(
            orig_freq=sr,
            new_freq=16000
        )(waveform)

    target_samples = 16000 * 5

    # Pad / Trim
    if waveform.shape[1] < target_samples:
        waveform = F.pad(
            waveform,
            (0, target_samples - waveform.shape[1])
        )
    else:
        waveform = waveform[:, :target_samples]

    print("Waveform Shape:", waveform.shape)

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )

    mel = mel_transform(waveform)

    mel = torchaudio.transforms.AmplitudeToDB(
        top_db=80
    )(mel)

    print("Mel Shape:", mel.shape)

    mel = mel.unsqueeze(0)

    print("Final Input Shape:", mel.shape)

    return mel


def load_weights(model, path):
    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=True
    )

    print("\n===== MODEL VERIFICATION =====")

    print("Checkpoint Type:", type(checkpoint))

    if isinstance(checkpoint, dict):
        print("First Keys:")
        print(list(checkpoint.keys())[:15])

        if "model_state_dict" in checkpoint:
            checkpoint = checkpoint["model_state_dict"]

    missing, unexpected = model.load_state_dict(
        checkpoint,
        strict=False
    )

    print("Missing Keys:", missing)
    print("Unexpected Keys:", unexpected)

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    print("Parameters:", f"{total_params:,}")

    print("==============================\n")


def main():
    model = FamilyClassifierCNN(
        num_classes=len(FAMILY_CLASSES)
    )

    load_weights(
        model,
        WEIGHT_FILE
    )

    model.eval()

    x = preprocess_audio(
        AUDIO_FILE
    )

    with torch.no_grad():
        logits = model(x)

        probs = torch.softmax(
            logits,
            dim=1
        )[0]

    print("\n========== CLASS PROBABILITIES ==========\n")

    for cls, prob in zip(
            FAMILY_CLASSES,
            probs
    ):
        print(
            f"{cls:<12} : {prob.item():.6f}"
        )

    best_idx = torch.argmax(
        probs
    ).item()

    print("\n========== RESULT ==========")

    print(
        "Prediction:",
        FAMILY_CLASSES[best_idx]
    )

    print(
        "Confidence:",
        probs[best_idx].item()
    )

    print("============================")


if __name__ == "__main__":
    main()