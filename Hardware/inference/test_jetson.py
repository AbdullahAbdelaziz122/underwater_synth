import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # Automatically handles CUDA context initialization
import numpy as np
import time

class JetsonTRTHierarchicalPipeline:
    def __init__(self, stage1_engine_path, stage3_engine_path):
        print("Initializing Jetson Hardware Inference Engine...")
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        
        # Load and deserialize the engine files compiled on the Nano
        self.stage1_engine = self.load_engine(stage1_engine_path)
        self.stage3_engine = self.load_engine(stage3_engine_path)
        
        # Create execution contexts
        self.s1_context = self.stage1_engine.create_execution_context()
        self.s3_context = self.stage3_engine.create_execution_context()
        
        # Allocate CUDA memory buffers for Stage 1
        self.s1_inputs, self.s1_outputs, self.s1_bindings, self.s1_stream = self.allocate_buffers(self.stage1_engine)
        
        # Allocate CUDA memory buffers for Stage 3
        self.s3_inputs, self.s3_outputs, self.s3_bindings, self.s3_stream = self.allocate_buffers(self.stage3_engine)

        self.family_classes = ['Background', 'Beluga', 'Dolphin', 'Narwhal', 'Seal', 'Vessel', 'Walrus', 'Whale']

    def load_engine(self, path):
        with open(path, "rb") as f:
            return self.runtime.deserialize_cuda_engine(f.read())

    def allocate_buffers(self, engine):
        """Allocates pinned host memory and device VRAM memory for data transfers"""
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()
        
        for binding in engine:
            size = trt.volume(engine.get_binding_shape(binding)) * 1 # 1 batch size
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            
            # Allocate Page-Locked (Pinned) Host Memory - critical for ultra-low latency DMA transfers
            host_mem = cuda.pagelocked_empty(size, dtype)
            # Allocate GPU VRAM Memory
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            bindings.append(int(device_mem))
            
            if engine.binding_is_input(binding):
                inputs.append({"host": host_mem, "device": device_mem})
            else:
                outputs.append({"host": host_mem, "device": device_mem})
                
        return inputs, outputs, bindings, stream

    def infer_stage1(self, sliced_spectrogram):
        # 1. Copy input data from normal numpy array to the pinned Host memory buffer
        np.copyto(self.s1_inputs[0]["host"], sliced_spectrogram.ravel())
        
        # 2. Transfer data from Host (CPU) memory to Device (GPU) VRAM asynchronously
        cuda.memcpy_htod_async(self.s1_inputs[0]["device"], self.s1_inputs[0]["host"], self.s1_stream)
        
        # 3. Execute inference context on GPU
        self.s1_context.execute_async_v2(bindings=self.s1_bindings, stream_handle=self.s1_stream.handle)
        
        # 4. Transfer outputs from GPU VRAM back to Host CPU memory
        cuda.memcpy_dtoh_async(self.s1_outputs[0]["host"], self.s1_outputs[0]["device"], self.s1_stream)
        
        # Synchronize stream to wait for async operations to finish
        self.s1_stream.synchronize()
        
        return self.s1_outputs[0]["host"][0]

    def infer_stage3(self, sliced_spectrogram):
        np.copyto(self.s3_inputs[0]["host"], sliced_spectrogram.ravel())
        cuda.memcpy_htod_async(self.s3_inputs[0]["device"], self.s3_inputs[0]["host"], self.s3_stream)
        self.s3_context.execute_async_v2(bindings=self.s3_bindings, stream_handle=self.s3_stream.handle)
        cuda.memcpy_dtoh_async(self.s3_outputs[0]["host"], self.s3_outputs[0]["device"], self.s3_stream)
        self.s3_stream.synchronize()
        return self.s3_outputs[0]["host"]

    def run_pipeline(self, master_spectrogram):
        print("\n--- Jetson TRT Inference Cycle ---")
        start_time = time.perf_counter()

        # Zero-Copy NumPy Slicing matching the Mac script layout
        s1_input = master_spectrogram[:, :, :, :152]
        
        # Execute Stage 1
        threat_prob = self.infer_stage1(s1_input)
        print(f"Stage 1 Threat Probability: {threat_prob:.4f}")

        THRESHOLD = 0.85
        if threat_prob >= THRESHOLD:
            print(">> ROUTING TO STAGE 2")
            # Stage 2 code will match Stage 3 pattern exactly when ready
        else:
            print(">> ROUTING TO STAGE 3: NON-THREAT")
            s3_input = master_spectrogram[:, :, :64, :]
            probabilities = self.infer_stage3(s3_input)
            
            class_idx = np.argmax(probabilities)
            confidence = probabilities[class_idx]
            print(f"Family Identified: {self.family_classes[class_idx]} (Confidence: {confidence:.2f})")

        end_time = time.perf_counter()
        print(f"Total Jetson Pipeline Latency: {(end_time - start_time) * 1000:.2f} ms")

if __name__ == "__main__":
    # Ensure you replace these paths with the actual .engine or .trt files compiled via trtexec on the Nano
    pipeline = JetsonTRTHierarchicalPipeline(
        stage1_engine_path="Hardware/inference/stage1_binary_detector.engine",
        stage3_engine_path="Hardware/inference/stage3_family_classifier.engine"
    )

    # Generate a dummy master tensor to verify pipeline consistency
    mock_master_spectrogram = np.random.randn(1, 1, 128, 157).astype(np.float32)
    pipeline.run_pipeline(mock_master_spectrogram)