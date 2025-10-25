import arlpy.uwapm as pm
import numpy as np
import pandas as pd
import os


# ---------ENVIRONMENT SETUP---------
ssp_path = "../assets/final_experiment/ssp.csv"  # depth (m), sound_speed (m/s)
profile = pd.read_csv(ssp_path)
ssp = profile.values.tolist()

# Environment parameters

TX_DEPTH = [200, 250, 300, 350, 400]                      # transmitter depth (m)
FREQ = 500                                                # propagation frequency (Hz)
RANGE_MIN, RANGE_MAX = 2000, 8000                         # meters
RANGE_STEPS = 1000                                        # receiver horizontal positions
DEPTH_MIN, DEPTH_MAX = 0, 400                             # receiver depths
DEPTH_STEPS = 100                                         # number of hydrophones (receiver depths)

# Create random wavy surface
surface = np.array([[r, 0.5+0.5*np.sin(2*np.pi*0.005*r)] for r in np.linspace(0,8500,1001)])

# Bellhop parameters
fs = 16000                                                          # Sampling rate of source signals
source_dir = "./sources"                                            # Folder of clean source .wav files (Stage 1)
output_dir = "./bellhop_out"                                        # Where to save propagated outputs
impulse_response_out = "./bellhop_envs"                             # Where to save impulse response 

os.makedirs(output_dir, exist_ok=True)
os.makedirs(impulse_response_out, exist_ok=True)


# Loop over transmitter depths, receiver depths, and receiver ranges
for tx_depth in TX_DEPTH:
    for rx_depth in range(DEPTH_MIN, DEPTH_MAX, DEPTH_STEPS):
        for rx_range in range(RANGE_MIN, RANGE_MAX, RANGE_STEPS):
            # ---------BELLHOP SIMULATION---------
            env = pm.create_env2d(
                depth = 500,
                soundspeed = ssp,
                bottom_soundspeed = 1600,
                bottom_density=1800,                        # kg/m³
                bottom_absorption=0.8,                      # attenuation (dB/λ)
                rx_depth=rx_depth,
                rx_range=rx_range,
                frequency=FREQ
            )
            
            # Add Tx_depth
            env['tx_depth'] = tx_depth
            
            # Add surface
            env['surface'] = surface
            
            # Run Bellhop
            # Compute Arrivals
            arrivals = pm.compute_arrivals(env)
            
            # Convert Arrivals to Impulse Response
            ir = pm.arrivals_to_impulse_response(arrivals, fs)
            # Save Impulse Response
            ir_filename = f"ir_tx{tx_depth}_rx{rx_depth}_{rx_range}"
            ir_path = os.path.join(impulse_response_out, ir_filename)
            np.save(ir_path, ir)
            print(f"Saved Impulse Response: {ir_path}.npy")


print("Bellhop environment simulations completed.")