import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import soundfile as sf


WEIGHT_FILE = "./Binary_model_weights.pth"
AUDIO_FILE = "./inferencedata/000000_submarine.wav"


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.flatten = nn.Flatten()

        self.fc = nn.Sequential(
            nn.Linear(19456, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


def preprocess_audio(audio_path: str) -> torch.Tensor:
    """
    Load WAV file using SoundFile and create the
    exact mel spectrogram expected by the model.
    """

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    print(f"Loading audio: {audio_path}")

    # Load audio
    audio, sr = sf.read(audio_path)

    print(f"Original sample rate: {sr}")
    print(f"Original shape: {audio.shape}")

    # Convert to tensor
    waveform = torch.tensor(audio, dtype=torch.float32)

    # Stereo -> Mono
    if waveform.ndim > 1:
        waveform = waveform.mean(dim=1)

    # Shape => (1, samples)
    waveform = waveform.unsqueeze(0)

    # Resample if necessary
    if sr != 16000:
        print(f"Resampling {sr} -> 16000")

        resampler = torchaudio.transforms.Resample(
            orig_freq=sr,
            new_freq=16000
        )

        waveform = resampler(waveform)

    target_samples = 16000 * 5

    # Pad / Trim to exactly 5 sec
    if waveform.shape[1] < target_samples:
        pad_amount = target_samples - waveform.shape[1]

        waveform = F.pad(
            waveform,
            (0, pad_amount)
        )

        print(f"Padded audio by {pad_amount} samples")

    else:
        waveform = waveform[:, :target_samples]

    print("Waveform shape:", waveform.shape)

    # Create mel spectrogram
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000,
        n_mels=128,
        n_fft=1024,
        hop_length=512
    )

    mel = mel_transform(waveform)

    print("Mel shape before batch:", mel.shape)

    # Add batch dimension
    mel = mel.unsqueeze(0)

    print("Final input shape:", mel.shape)

    return mel


def load_weights(model: nn.Module, path: str):
    """
    Load state dict and verify compatibility.
    """

    if not os.path.exists(path):
        raise FileNotFoundError(f"Weight file not found: {path}")

    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=True
    )

    print("\n===== MODEL VERIFICATION =====")
    print("Checkpoint type:", type(checkpoint))

    if isinstance(checkpoint, dict):
        print("First keys:")
        print(list(checkpoint.keys())[:10])

        if "model_state_dict" in checkpoint:
            checkpoint = checkpoint["model_state_dict"]

    missing_keys, unexpected_keys = model.load_state_dict(
        checkpoint,
        strict=False
    )

    print("\nMissing keys:", missing_keys)
    print("Unexpected keys:", unexpected_keys)

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    print(f"Parameters: {total_params:,}")
    print("==============================\n")


def main():
    model = SimpleCNN()

    load_weights(
        model,
        WEIGHT_FILE
    )

    model.eval()

    x = preprocess_audio(AUDIO_FILE)

    with torch.no_grad():
        logits = model(x)

        probability = torch.sigmoid(
            logits
        ).item()

    prediction = (
        "Threat"
        if probability > 0.5
        else "Non-Threat"
    )

    print("\n========== RESULT ==========")
    print("Prediction :", prediction)
    print("Probability:", probability)
    print("============================")


if __name__ == "__main__":
    main()