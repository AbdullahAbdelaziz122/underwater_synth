import onnxruntime as ort
import numpy as np
import time

class HierarchicalPipelineTest:
    def __init__(self, stage1_path, stage2_path, stage3_path):
        print("Loading ONNX Execution Sessions...")
        # providers=['CPUExecutionProvider'] ensures it runs universally on your Mac
        self.stage1_sess = ort.InferenceSession(stage1_path, providers=['CPUExecutionProvider'])
        # self.stage2_sess = ort.InferenceSession(stage2_path, providers=['CPUExecutionProvider'])
        self.stage3_sess = ort.InferenceSession(stage3_path, providers=['CPUExecutionProvider'])
        
        # Get the exact input node names assigned during export
        self.s1_input_name = self.stage1_sess.get_inputs()[0].name
        # self.s2_input_name = self.stage2_sess.get_inputs()[0].name
        self.s3_input_name = self.stage3_sess.get_inputs()[0].name

        # For the mock output mappings
        self.family_classes = ['Background', 'Beluga', 'Dolphin', 'Narwhal', 'Seal', 'Vessel', 'Walrus', 'Whale']
        # self.threat_classes = ['Submarine', 'Torpedo', 'Diver'] # Example

    def run_inference(self, master_spectrogram):
        """
        Executes the hierarchical routing logic.
        master_spectrogram shape should be (1, 1, 128, 157)
        """
        print("\n--- Starting Inference Cycle ---")
        start_time = time.perf_counter()

        # ==========================================
        # STAGE 1: THREAT DETECTION
        # ==========================================
        # Slice master to 128x152 (crop the last 5 time frames)
        s1_input = master_spectrogram[:, :, :, :152]
        
        s1_outputs = self.stage1_sess.run(None, {self.s1_input_name: s1_input})
        threat_prob = s1_outputs[0][0][0] # Extract the single float value
        
        print(f"Stage 1 Threat Probability: {threat_prob:.4f}")

        # ==========================================
        # GATEKEEPER ROUTING LOGIC
        # ==========================================
        THRESHOLD = 0.85
        
        if threat_prob >= THRESHOLD:
            print(">> ROUTING TO STAGE 2: THREAT CLASSIFICATION")
            # STAGE 2 MOCKUP
            # s2_input = master_spectrogram[:, :, :, :152] # Adjust slice to Stage 2's shape
            # s2_outputs = self.stage2_sess.run(None, {self.s2_input_name: s2_input})
            # class_idx = np.argmax(s2_outputs[0])
            # print(f"Threat Identified: {self.threat_classes[class_idx]}")
            
        else:
            print(">> ROUTING TO STAGE 3: NON-THREAT (FAMILY) CLASSIFICATION")
            # STAGE 3 EXECUTION
            # Slice master to 64x157 (crop the top 64 frequency bins)
            s3_input = master_spectrogram[:, :, :64, :]
            s3_outputs = self.stage3_sess.run(None, {self.s3_input_name: s3_input})
            
            # Extract the probabilities and find the highest one
            probabilities = s3_outputs[0][0]
            class_idx = np.argmax(probabilities)
            confidence = probabilities[class_idx]
            
            print(f"Family Identified: {self.family_classes[class_idx]} (Confidence: {confidence:.2f})")

        # ==========================================
        # PERFORMANCE PROFILING
        # ==========================================
        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000
        print(f"Total Pipeline Latency: {latency_ms:.2f} ms")

if __name__ == "__main__":
    # Initialize the pipeline
    # Update these paths to where your ONNX files are saved
    pipeline = HierarchicalPipelineTest(
        stage1_path="Hardware/inference/stage1_binary_detector.onnx",
        stage2_path="Hardware/inference/stage2_threat_detector.onnx", 
        stage3_path="Hardware/inference/stage3_family_classifier.onnx"
    )

    # Simulate the STFT output from the Jetson's CPU
    # Array format: Float32 (Standard for ONNX/TensorRT)
    print("\nGenerating dummy master spectrogram (1, 1, 128, 157)...")
    mock_master_spectrogram = np.random.randn(1, 1, 128, 157).astype(np.float32)

    # Run the test
    pipeline.run_inference(mock_master_spectrogram)