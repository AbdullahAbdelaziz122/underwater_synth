import torch
import torch.nn as nn
import torch.onnx

# 1. Define the architecture (copied from your script)
class FamilyClassifierCNN(nn.Module):
    def __init__(self, num_classes):
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

# 2. Wrapper to bake in the Softmax activation for TensorRT
class Stage3ExportWrapper(nn.Module):
    def __init__(self, base_model):
        super(Stage3ExportWrapper, self).__init__()
        self.base_model = base_model

    def forward(self, x):
        logits = self.base_model(x)
        # Apply Softmax across the classes (dim=1) to output probabilities
        return torch.softmax(logits, dim=1)

def export_model():
    # --- CRITICAL CONFIGURATION ---
    # Change this to match the len(le_family.classes_) from your training output!
    NUM_CLASSES = 8  
    WEIGHTS_PATH = "Models/non-threat classifier/best_family_classifier.pth"
    ONNX_PATH = "Models/non-threat classifier/stage3_family_classifier.onnx"
    # ------------------------------

    print("Initializing model...")
    base_model = FamilyClassifierCNN(num_classes=NUM_CLASSES)
    
    print(f"Loading weights from {WEIGHTS_PATH}...")
    base_model.load_state_dict(torch.load(WEIGHTS_PATH, map_location='cpu'))
    
    wrapped_model = Stage3ExportWrapper(base_model)
    wrapped_model.eval()

    # Shape derived from torchaudio.transforms in your script
    CHANNELS = 1
    HEIGHT = 64  # n_mels
    WIDTH = 157  # Based on duration=5000, sr=16000, hop_length=512
    
    print(f"Creating dummy input tensor of shape (1, {CHANNELS}, {HEIGHT}, {WIDTH})...")
    dummy_input = torch.randn(1, CHANNELS, HEIGHT, WIDTH)

    print(f"Exporting ONNX graph to {ONNX_PATH}...")
    torch.onnx.export(
        wrapped_model, 
        dummy_input, 
        ONNX_PATH,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=['input_spectrogram'],
        output_names=['family_probabilities'] # Tensor containing array of probabilities
    )
    print("Success! Ready for Jetson Nano.")

if __name__ == "__main__":
    export_model()