"""
A module to create and visualize ARLPY 2D environments from SSP files.
"""

import arlpy.uwapm as pm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Dict, Any, List

def create_arlpy_environment(
    ssp_csv_path: str,
    target_depth: float,
    frequency: float,
    rx_depths: List[float] | np.ndarray,
    rx_ranges: List[float] | np.ndarray,
    bottom_soundspeed: float = 1600.0,
    bottom_density: float = 1800.0,
    bottom_absorption: float = 0.8,
) -> Optional[Dict[str, Any]]:
    """
    Creates a 2D ARLPY environment from an extended SSP file.

    Args:
        ssp_csv_path (str): Path to the SSP CSV file ('depth', 'sound_speed').
        target_depth (float): Total water depth for the environment (meters).
        frequency (float): Source frequency (Hz).
        rx_depths (List[float] | np.ndarray): Array of receiver depths (meters).
        rx_ranges (List[float] | np.ndarray): Array of receiver ranges (meters).
        bottom_soundspeed (float): Bottom sound speed (m/s).
        bottom_density (float): Bottom density (kg/m³).
        bottom_absorption (float): Bottom absorption (dB/wavelength).

    Returns:
        Optional[Dict[str, Any]]: The ARLPY 'env' dictionary, or None on failure.
    
    Note:
        Don't forget to add 'tx_depth' as it's `0` in default
    """
    
    # --- 1) Load your extended SSP file
    try:
        profile = pd.read_csv(ssp_csv_path)
        if 'depth' not in profile.columns or 'sound_speed' not in profile.columns:
            print(f"Error: SSP file must have 'depth' and 'sound_speed' columns.")
            return None
    except FileNotFoundError:
        print(f"Error: SSP file not found at {ssp_csv_path}")
        return None
    except Exception as e:
        print(f"Error loading {ssp_csv_path}: {e}")
        return None
    
    print(f"Loaded SSP from {ssp_csv_path}")

    # --- 2) Create ARLPY environment
    try:
        env = pm.create_env2d(
            depth=target_depth,
            soundspeed=profile[['depth', 'sound_speed']].values.tolist(),
            bottom_soundspeed=bottom_soundspeed,
            bottom_density=bottom_density,
            bottom_absorption=bottom_absorption,
            rx_depth=rx_depths,
            rx_range=rx_ranges,
            frequency=frequency
        )
    except Exception as e:
        print(f"Error during pm.create_env2d: {e}")
        return None

    # --- 3) Check environment
    print("Checking environment...")
    pm.check_env2d(env)
    print("Environment created successfully.")
    
    return env

def plot_environment(env: Dict[str, Any]):
    """Helper function to plot the environment and SSP."""
    print("Plotting environment...")
    pm.plot_env(env)
    plt.title("Environment Layout")
    plt.show()
    
    print("Plotting Sound Speed Profile...")
    pm.plot_ssp(env)
    plt.title("Sound Speed Profile")
    plt.show()




if __name__ == "__main__":
    
    print("--- Testing Environment Creation Module ---")
    
    # --- Parameters
    SSP_FILE = "../assets/extended_ssp_0_500m_v2.csv"
    ENV_DEPTH = 500.0
    FREQ = 1500.0
    RECEIVER_DEPTHS = np.linspace(5, 400, 10)
    RECEIVER_RANGES = np.linspace(0, 2000, 200)

    # --- 1) Create Environment
    my_env = create_arlpy_environment(
        ssp_csv_path=SSP_FILE,
        target_depth=ENV_DEPTH,
        frequency=FREQ,
        rx_depths=RECEIVER_DEPTHS,
        rx_ranges=RECEIVER_RANGES
    )

    # --- 2) Plot Environment
    if my_env:
        plot_environment(my_env)
        pm.print_env(my_env)
    else:
        print("Failed to create environment.")