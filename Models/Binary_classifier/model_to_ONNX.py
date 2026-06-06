import torch
import torch.nn as nn
import torch.onnx

# 1. Define your architecture (copied exactly from your code)
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

# 2. Wrapper to bake in the Sigmoid activation for TensorRT
class Stage1ExportWrapper(nn.Module):
    def __init__(self, base_model):
        super(Stage1ExportWrapper, self).__init__()
        self.base_model = base_model

    def forward(self, x):
        logits = self.base_model(x)
        return torch.sigmoid(logits)

def export_model():
    print("Initializing model...")
    base_model = SimpleCNN()
    
    # LOAD WEIGHTS HERE (Change the filename if yours is different)
    weights_path = "Models/Binary_classifier/Binary_model_weights.pth" 
    print(f"Loading weights from {weights_path}...")
    base_model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    
    wrapped_model = Stage1ExportWrapper(base_model)
    wrapped_model.eval()

    # DUMMY INPUT TENSOR
    # If your shape is different than 128x152, change H and W here.
    # As long as (H/8) * (W/8) == 304, it will work.
    CHANNELS = 1
    HEIGHT = 128
    WIDTH = 152
    
    print(f"Creating dummy input tensor of shape (1, {CHANNELS}, {HEIGHT}, {WIDTH})...")
    dummy_input = torch.randn(1, CHANNELS, HEIGHT, WIDTH)

    onnx_path = "stage1_threat_detector.onnx"
    print(f"Exporting ONNX graph to {onnx_path}...")
    
    torch.onnx.export(
        wrapped_model, 
        dummy_input, 
        onnx_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=['input_spectrogram'],
        output_names=['threat_probability']
    )
    print("Success! Ready for Jetson Nano.")

if __name__ == "__main__":
    export_model()