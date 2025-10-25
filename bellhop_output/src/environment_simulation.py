"""
Bellhop Environment Simulation Module
=====================================

This module provides a function to generate underwater acoustic propagation environments 
using the Bellhop acoustic propagation model (via the `arlpy.uwapm` library).

The script reads a sound speed profile (SSP), defines transmitter and receiver geometries,
creates synthetic surface waves, and computes impulse responses between transmitters and receivers.
These impulse responses represent the underwater acoustic channel characteristics under 
the given environmental conditions.

Author: Abdullah Abdelaziz
Created: 2025
Dependencies:
    - arlpy (https://pypi.org/project/arlpy/)
    - numpy
    - pandas
    - os
"""

import arlpy.uwapm as pm
import numpy as np
import pandas as pd
import os


def bellhop_envs(
    ssp_path: str,
    output_dir: str,
    impulse_response_out: str,
    TX_DEPTH: list,
    FREQ: float,
    RANGE_MIN: int,
    RANGE_MAX: int,
    RANGE_STEPS: int,
    DEPTH_MIN: int,
    DEPTH_MAX: int,
    DEPTH_STEPS: int,
    fs: int,
    surface: np.ndarray,
    source_dir: str,
    depth: int = 500,
    bottom_soundspeed: float = 1600,
    bottom_density: float = 1800,
    bottom_absorption: float = 0.8
):
    """
    Generate and simulate underwater acoustic environments using Bellhop.

    This function sets up a range of Bellhop 2D acoustic propagation simulations
    across varying transmitter depths, receiver depths, and ranges. It computes
    the channel impulse responses for each configuration and saves them as `.npy` files.

    Parameters
    ----------
    ssp_path : str
        Path to the sound speed profile CSV file. The file should contain two columns:
        depth (m) and sound speed (m/s).
    output_dir : str
        Directory to store propagated outputs and intermediate results.
    impulse_response_out : str
        Directory to save generated impulse responses.
    TX_DEPTH : list
        List of transmitter depths in meters (e.g., [200, 250, 300]).
    FREQ : float
        Acoustic propagation frequency in Hz.
    RANGE_MIN : int
        Minimum receiver range in meters.
    RANGE_MAX : int
        Maximum receiver range in meters.
    RANGE_STEPS : int
        Step size (in meters) between consecutive receiver ranges.
    DEPTH_MIN : int
        Minimum receiver depth in meters.
    DEPTH_MAX : int
        Maximum receiver depth in meters.
    DEPTH_STEPS : int
        Step size (in meters) between consecutive receiver depths.
    fs : int
        Sampling rate of source signals (Hz).
    surface : np.ndarray
        2D numpy array defining the surface profile, shaped as [[range, height], ...].
    source_dir : str
        Path to the folder containing clean source `.wav` files (from Stage 1).
    depth : int, optional
        Total water column depth in meters. Default is 500.
    bottom_soundspeed : float, optional
        Speed of sound in the seabed (m/s). Default is 1600.
    bottom_density : float, optional
        Seabed density in kg/m³. Default is 1800.
    bottom_absorption : float, optional
        Seabed absorption coefficient (dB/λ). Default is 0.8.

    Returns
    -------
    None
        The function saves impulse response `.npy` files to the specified output directory.

    Notes
    -----
    - The Bellhop model computes acoustic propagation paths (ray arrivals) through
      the defined water column and seabed layers.
    - Each simulation generates an impulse response representing the acoustic
      channel transfer function between transmitter and receiver.
    - Results are stored in the specified `impulse_response_out` directory.
    """

    # --------- Load Sound Speed Profile (SSP) ---------
    profile = pd.read_csv(ssp_path)
    ssp = profile.values.tolist()

    # Ensure output directories exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(impulse_response_out, exist_ok=True)

    # --------- Main Simulation Loops ---------
    for tx_depth in TX_DEPTH:
        for rx_depth in range(DEPTH_MIN, DEPTH_MAX, DEPTH_STEPS):
            for rx_range in range(RANGE_MIN, RANGE_MAX, RANGE_STEPS):

                # Create Bellhop 2D environment configuration
                env = pm.create_env2d(
                    depth=depth,
                    soundspeed=ssp,
                    bottom_soundspeed=bottom_soundspeed,
                    bottom_density=bottom_density,
                    bottom_absorption=bottom_absorption,
                    rx_depth=rx_depth,
                    rx_range=rx_range,
                    frequency=FREQ
                )

                # Add custom fields
                env['tx_depth'] = tx_depth
                env['surface'] = surface

                # --------- Compute Bellhop Results ---------
                arrivals = pm.compute_arrivals(env)
                ir = pm.arrivals_to_impulse_response(arrivals, fs)

                # Save the generated impulse response
                ir_filename = f"ir_tx{tx_depth}_rx{rx_depth}_{rx_range}.npy"
                ir_path = os.path.join(impulse_response_out, ir_filename)
                np.save(ir_path, ir)

                print(f"Saved Impulse Response: {ir_path}")

    print("Bellhop environment simulations completed successfully.")


if __name__ == "__main__":
    """
    Example Usage
    -------------
    This block demonstrates how to call the `bellhop_envs()` function
    using a predefined environment setup and a given sound speed profile (SSP).

    The generated impulse responses will be saved under `./bellhop_envs`.
    """

    # Load SSP (depth in meters, sound_speed in m/s)
    ssp_path = "../ssp_data/ssp.csv"
    profile = pd.read_csv(ssp_path)
    ssp = profile.values.tolist()

    # Define environment parameters
    TX_DEPTH = [200, 250, 300, 350, 400]
    FREQ = 500  # Hz
    RANGE_MIN, RANGE_MAX = 2000, 8000
    RANGE_STEPS = 1000
    DEPTH_MIN, DEPTH_MAX = 0, 400
    DEPTH_STEPS = 100

    # Define synthetic wavy surface
    surface = np.array([[r, 0.5 + 0.5 * np.sin(2 * np.pi * 0.005 * r)] for r in np.linspace(0, 8500, 1001)])

    # Define directories
    fs = 16000
    source_dir = "../sources"
    output_dir = "../bellhop_out"
    impulse_response_out = "../bellhop_envs"

    # Run simulation
    bellhop_envs(
        ssp_path=ssp_path,
        output_dir=output_dir,
        impulse_response_out=impulse_response_out,
        TX_DEPTH=TX_DEPTH,
        FREQ=FREQ,
        RANGE_MIN=RANGE_MIN,
        RANGE_MAX=RANGE_MAX,
        RANGE_STEPS=RANGE_STEPS,
        DEPTH_MIN=DEPTH_MIN,
        DEPTH_MAX=DEPTH_MAX,
        DEPTH_STEPS=DEPTH_STEPS,
        fs=fs,
        surface=surface,
        source_dir=source_dir
    )
