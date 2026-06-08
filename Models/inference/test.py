import torch
import torch.nn as nn
import time

# ==========================================
# 1. OPTIONAL: ARCHITECTURE PLACEHOLDERS
# ==========================================
# If your .pth files only contain weights (state_dict), define or import your 
# classes here. If they contain the full model object, you can skip this.
class MockStage1And2Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Linear(16, 1) # Binary for Stage 1 / Multi-class for Stage 2
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

# ==========================================
# 2. HIERARCHICAL PIPELINE RUNNER
# ==========================================
class JetsonPipelinePyTorch:
    def __init__(self, stage1_path, stage2_path, stage3_path):
        print("Initializing PyTorch Pipeline on CUDA...")
        
        # Determine if GPU acceleration is ready
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Target Device: {self.device}")
        
        # Load models. 
        # Note: If your .pth files are just weights, instantiate the class first:
        # self.stage1 = YourClass().to(self.device)
        # self.stage1.load_state_dict(torch.load(stage1_path, map_location=self.device))
        
        print("Loading model binaries into GPU memory...")
        try:
            # Assuming full model serialization (.pth containing architecture + weights)
            self.stage1 = torch.load(stage1_path, map_location=self.device)
            self.stage2 = torch.load(stage2_path, map_location=self.device)
            self.stage3 = torch.load(stage3_path, map_location=self.device)
        except AttributeError:
            print("💡 Notice: Loading as state_dicts instead. Using placeholder architectures...")
            # Fallback wrapper for state_dict testing
            self.stage1 = MockStage1And2Net().to(self.device)
            self.stage2 = MockStage1And2Net().to(self.device)
            self.stage2.classifier = nn.Linear(16, 3) # 3 threat classes
            self.stage3 = MockStage1And2Net().to(self.device)
            self.stage3.classifier = nn.Linear(16, 8) # 8 family classes
            
            # If your files are state_dicts, uncomment these lines:
            # self.stage1.load_state_dict(torch.load(stage1_path, map_location=self.device))
            # self.stage2.load_state_dict(torch.load(stage2_path, map_location=self.device))
            # self.stage3.load_state_dict(torch.load(stage3_path, map_location=self.device))

        # Set all networks strictly to evaluation mode
        self.stage1.eval()
        self.stage2.eval()
        self.stage3.eval()

        # Class lookup tables matching your deployment spec
        self.threat_classes = ['Threat_A', 'Threat_B', 'Threat_C']
        self.family_classes = ['Background', 'Beluga', 'Dolphin', 'Narwhal', 'Seal', 'Vessel', 'Walrus', 'Whale']

    def process_frame(self, master_spectrogram):
        """
        Executes hierarchical routing without generating gradients or tracking backward states.
        """
        # Ensure input tensor is pushed to GPU
        master_spectrogram = master_spectrogram.to(self.device)
        
        print("\n--- Processing Acoustic Input Frame ---")
        
        # Synchronize CUDA timelines before measuring start time
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.perf_counter()

        with torch.no_grad():
            # ==========================================
            # STAGE 1: BINARY THREAT DETECTION
            # Input slice: 128x152
            # ==========================================
            s1_input = master_spectrogram[:, :, :, :152]
            s1_output = self.stage1(s1_input)
            
            # Apply sigmoid if the raw model outputs logits
            threat_prob = torch.sigmoid(s1_output).item()
            print(f"Stage 1 Threat Probability: {threat_prob:.4f}")

            # ==========================================
            # HIERARCHICAL ROUTING LOGIC
            # ==========================================
            THRESHOLD = 0.85
            
            if threat_prob >= THRESHOLD:
                print("🚨 >> ROUTING TO STAGE 2: THREAT CLASSIFICATION")
                s2_input = master_spectrogram[:, :, :, :152]
                s2_output = self.stage2(s2_input)
                
                class_idx = torch.argmax(s2_output, dim=1).item()
                print(f"⚠️ TARGET IDENTIFIED: {self.threat_classes[class_idx]}")
                
            else:
                print("🌊 >> ROUTING TO STAGE 3: NON-THREAT (FAMILY) CLASSIFICATION")
                # Input slice: 64x157
                s3_input = master_spectrogram[:, :, :64, :]
                s3_output = self.stage3(s3_input)
                
                class_idx = torch.argmax(s3_output, dim=1).item()
                print(f"Family Identified: {self.family_classes[class_idx]}")

        # Synchronize CUDA timelines before calculating execution latency
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        print(f"Total GPU Pipeline Latency: {(end_time - start_time) * 1000:.2f} ms")

if __name__ == "__main__":
    # Instantiating pipeline referencing your localized storage files
    pipeline = JetsonPipelinePyTorch(
        stage1_path="./stage1_binary_detector.pth",
        stage2_path="./stage2_threat_detector.pth",
        stage3_path="./stage3_family_classifier.pth"
    )

    # Generate dummy tensor matching your master input shape matrix (1, 1, 128, 157)
    mock_input = torch.randn(1, 1, 128, 157, dtype=torch.float32)
    
    # Run a warm-up pass (initializes CUDA contexts)
    print("\nRunning warm-up cycle...")
    pipeline.process_frame(mock_input)
    
    # Run the timed validation pass
    print("\nRunning timed execution cycle...")
    pipeline.process_frame(mock_input)
