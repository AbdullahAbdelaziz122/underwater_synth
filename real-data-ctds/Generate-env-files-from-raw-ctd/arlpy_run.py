"""
A module to run Bellhop computations (TL, rays, arrivals, IR)
on a pre-built ARLPY environment.
"""

import arlpy.uwapm as pm
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict, Any
from . import arlpy_env
# Import the environment creation function from the other module
try:
    import arlpy_env
except ImportError:
    print("Warning: Could not import arlpy_env.py. Assumes it's in the same directory.")
    # This will fail in the main block, but allows importing the function itself
    pass


def run_bellhop_computations(
    env: Dict[str, Any],
    tloss_mode: str = 'incoherent',
    ir_sample_rate: int = 48000,
    do_plots: bool = False,
    save_ir_path: Optional[str] = None,
    save_tloss_path: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Runs Bellhop computations on a given ARLPY environment.

    Args:
        env (Dict[str, Any]): The pre-built ARLPY environment dictionary.
        tloss_mode (str): Transmission loss mode ('incoherent', 'coherent', etc.).
        ir_sample_rate (int): Sample rate (Hz) for the impulse response.
        do_plots (bool): If True, generate and display plots for each computation.
        save_ir_path (Optional[str]): If provided, save the IR to this .npy file.
        save_tloss_path (Optional[str]): If provided, save the TL magnitude to this .csv file.

    Returns:
        Optional[Dict[str, Any]]: A dictionary with results:
        {'tloss', 'rays', 'arrivals', 'ir'}, or None on failure.
    """
    
    results = {}

    try:
        # --- 1) Compute transmission loss
        print(f"Computing transmission loss (mode: {tloss_mode})...")
        tloss = pm.compute_transmission_loss(env, mode=tloss_mode)
        results['tloss'] = tloss
        
        if do_plots:
            print("Plotting transmission loss...")
            pm.plot_transmission_loss(tloss, env=env, width=1000)
            plt.title("Transmission Loss")
            plt.show()

        # --- 2) Compute eigenrays
        print("Computing eigenrays...")
        rays = pm.compute_eigenrays(env)
        results['rays'] = rays
        
        if do_plots:
            print("Plotting eigenrays...")
            pm.plot_rays(rays, env=env, width=1000)
            plt.show()

        # --- 3) Compute arrivals
        print("Computing arrivals...")
        arrivals = pm.compute_arrivals(env)
        results['arrivals'] = arrivals
        
        if do_plots:
            print("Plotting arrivals...")
            pm.plot_arrivals(arrivals)
            plt.show()
        
        # --- 4) Compute impulse response
        print(f"Generating impulse response (fs={ir_sample_rate} Hz)...")
        ir = pm.arrivals_to_impulse_response(arrivals, fs=ir_sample_rate)
        results['ir'] = ir
        print(f"Impulse response length: {len(ir)} samples")

    except Exception as e:
        print(f"Error during Bellhop computation: {e}")
        return None

    # --- 5) Save files
    if save_ir_path:
        try:
            np.save(save_ir_path, ir)
            print(f"Impulse response saved to: {save_ir_path}")
        except Exception as e:
            print(f"Error saving IR to {save_ir_path}: {e}")

    if save_tloss_path:
        try:
            tloss_mag = np.abs(tloss)**2  # Save as magnitude squared (power)
            np.savetxt(save_tloss_path, tloss_mag, delimiter=',')
            print(f"Transmission loss magnitude saved to: {save_tloss_path}")
        except Exception as e:
            print(f"Error saving TL to {save_tloss_path}: {e}")

    return results





if __name__ == "__main__":
    
    print("--- Running Full Simulation Pipeline (Env + Run) ---")
    
    # --- 1) Define Simulation Parameters
    SSP_FILE = "/Users/abdullahabdelaizz/Myfiles/Underwater/bellhop_output/assets/extended_ssp_0_500m_v2.csv"
    ENV_DEPTH = 500.0
    FREQ = 1500.0
    RECEIVER_DEPTHS = np.linspace(5, 400, 10)
    RECEIVER_RANGES = np.linspace(0, 2000, 200)
    
    IR_OUT_FILE = "./assets/ir_salish_env_PIPELINE.npy"
    TLOSS_OUT_FILE = "./assets/tloss_salish_env_PIPELINE.csv"

    # --- 2) Create Environment (using Module 1)
    print("\n--- STAGE 1: CREATING ENVIRONMENT ---")
    env = arlpy_env.create_arlpy_environment(
        ssp_csv_path=SSP_FILE,
        target_depth=ENV_DEPTH,
        frequency=FREQ,
        rx_depths=RECEIVER_DEPTHS,
        rx_ranges=RECEIVER_RANGES
    )

    if not env:
        print("Pipeline failed: Could not create environment.")
    else:
        # Optionally plot the environment
        # arlpy_env.plot_environment(env)
        
        # --- 3) Run Computations (using Module 2)
        print("\n--- STAGE 2: RUNNING COMPUTATIONS ---")
        sim_results = run_bellhop_computations(
            env=env,
            tloss_mode='incoherent',
            ir_sample_rate=48000,
            do_plots=True,  # Set to True to see all plots
            save_ir_path=IR_OUT_FILE,
            save_tloss_path=TLOSS_OUT_FILE
        )
        
        if sim_results:
            print("\n--- Simulation Successful ---")
            print(f"Transmission Loss shape: {sim_results['tloss'].shape}")
            print(f"IR length: {len(sim_results['ir'])} samples")
        else:
            print("\n--- Simulation Failed ---")