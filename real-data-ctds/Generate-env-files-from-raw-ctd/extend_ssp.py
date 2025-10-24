"""
A utility function to load a CTD profile from a CSV, calculate sound speed, 
and extend it to a target depth using blended linear extrapolation.

Original script logic provided by the user, refactored into a function.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional

# --- Optional Library Imports ---
# These are checked at the top level so the function knows what's available.
try:
    import gsw
    HAS_GSW = True
except ImportError:
    HAS_GSW = False

try:
    import xarray as xr
    HAS_XARRAY = True
except ImportError:
    HAS_XARRAY = False


def process_and_extend_ssp(
    ctd_csv_path: str,
    target_max_depth: int = 500,
    bin_size: int = 5,
    blend_window: int = 30,
    extrap_gradient: float = 0.008,
    output_csv_path: Optional[str] = None,
    plot: bool = False
) -> Optional[pd.DataFrame]:
    """
    Loads a CTD profile, calculates sound speed, and extends it to a target depth.

    Args:
        ctd_csv_path (str): Path to the input CSV file.
            Required columns: 'p1' (pressure, dbar), 't1' (temp, C), 'sal' (PSU).
            Optional column: 'sv' (sound velocity, m/s).
        target_max_depth (int): The final target depth (in meters) for the profile.
        bin_size (int): The depth resolution (in meters) for binning and the final grid.
        blend_window (int): The depth (in meters) over which to blend the
            measured data with the extrapolated data.
        extrap_gradient (float): The sound speed gradient (m/s per meter) to use
            for extrapolation below the measured data.
        output_csv_path (Optional[str]): If provided, the final profile will be
            saved to this file path.
        plot (bool): If True, a plot of the final profile will be displayed.

    Returns:
        Optional[pd.DataFrame]: A DataFrame with 'depth' and 'sound_speed'
        columns, or None if an error occurs.
    """

    # 1) Load a subset profile
    try:
        df = pd.read_csv(ctd_csv_path)
    except FileNotFoundError:
        print(f"Error: Input file not found at {ctd_csv_path}")
        return None
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None

    # If file contains many timestamps, select a single timestamp
    if 'date' in df.columns:
        first_date = df['date'].iloc[0]
        df = df[df['date'] == first_date].copy()

    # 2) Clean & Sort
    required_cols = ['p1', 't1', 'sal']
    if not all(col in df.columns for col in required_cols):
        print(f"Error: Input CSV must contain columns: {required_cols}")
        return None
    
    # Use 'sv' if present, else it will be calculated
    cols_to_use = required_cols.copy()
    if 'sv' in df.columns:
        cols_to_use.append('sv')

    df = df[cols_to_use].dropna(subset=['p1']).sort_values('p1')
    df = df[(df['p1'] >= 0) & (df['p1'] <= target_max_depth)]
    df = df.drop_duplicates(subset='p1', keep='first')

    if df.empty:
        print("Error: No valid data found after cleaning.")
        return None

    meas_max = df['p1'].max()
    print(f"Measured max depth: {meas_max:.1f} m")

    # 3) Compute sound speed
    if HAS_GSW:
        s = df['sal'].values
        t = df['t1'].values
        p = df['p1'].values
        ss = gsw.sound_speed(s, t, p)  # approx if SA~=SP
        df['sv_calc'] = ss
        print("Computed sound speed using gsw.")
    else:
        # Mackenzie 1981 approximation
        T = df['t1'].values; S = df['sal'].values; D = df['p1'].values
        c = (1448.96 + 4.591*T - 5.304e-2*T**2 + 2.374e-4*T**3 +
             1.340*(S-35) + 1.630e-2*D + 1.675e-7*D**2 -
             1.025e-2*T*(S-35) - 7.139e-13*T*D**3)
        df['sv_calc'] = c
        print("Computed sound speed using Mackenzie formula (fallback).")

    # 4) Use measured 'sv' if available but prefer computed for consistency
    if 'sv' in df.columns and df['sv'].notna().sum() > 0:
        err = np.nanmean(np.abs(df['sv'].values - df['sv_calc'].values))
        print(f"Mean abs diff measured vs computed sv: {err:.3f} m/s. Using computed.")
        df['sv_use'] = df['sv_calc']
    else:
        df['sv_use'] = df['sv_calc']

    # 5) Prepare the measured profile (binned by bin_size)
    df['depth_bin'] = (df['p1'] / bin_size).round() * bin_size
    profile = df.groupby('depth_bin')['sv_use'].mean().reset_index().rename(
        columns={'depth_bin': 'depth', 'sv_use': 'sound_speed'}
    )
    profile = profile.sort_values('depth').reset_index(drop=True)

    # 6) Check if measured top is already >= target_max_depth
    if profile['depth'].max() >= target_max_depth:
        print("Profile already reaches target depth. No extension needed.")
        profile_extended = profile[profile['depth'] <= target_max_depth].copy()
    else:
        # 7) Climatology check (non-functional in original, skipped)
        # The original script's xarray block was a placeholder.
        # We proceed directly to the extrapolation fallback.

        # 8) Fallback: build a blended extrapolation
        print("Extending profile with blended extrapolation...")
        last_depth = profile['depth'].max()
        last_speed = profile['sound_speed'].iloc[-1]

        # Create depths from last_measured + bin_size to target_max_depth
        extra_depths = np.arange(last_depth + bin_size, target_max_depth + 1, bin_size)
        if extra_depths.size == 0 and last_depth < target_max_depth:
             # Ensure we at least reach the target depth
             extra_depths = np.array([target_max_depth])
             
        extra_speeds = last_speed + extrap_gradient * (extra_depths - last_depth)
        extra_df = pd.DataFrame({'depth': extra_depths, 'sound_speed': extra_speeds})

        # Smoothly blend in a transition window (1m resolution for blending)
        blend = blend_window
        trans_depths = np.arange(last_depth, last_depth + blend + 1, 1)

        # Interpolate measured and extrapolated sections to 1m grid for blending
        meas_interp = np.interp(trans_depths, profile['depth'], profile['sound_speed'])
        
        extra_full_depths = np.concatenate(([last_depth], extra_depths))
        extra_full_speeds = np.concatenate(([last_speed], extra_speeds))
        extra_interp = np.interp(trans_depths, extra_full_depths, extra_full_speeds)

        # Create linear weight from 0 (at last_depth) to 1 (at last_depth + blend)
        w = np.clip((trans_depths - last_depth) / blend, 0, 1)
        blended_interp = (1 - w) * meas_interp + w * extra_interp
        blended_df = pd.DataFrame({'depth': trans_depths, 'sound_speed': blended_interp})

        # Combine: measured up to blend, blended transition, then rest of extra_df
        beyond = extra_df[extra_df['depth'] > last_depth + blend]
        profile_extended = pd.concat([
            profile[profile['depth'] < last_depth],  # measured up to before blend
            blended_df,                              # the blended 1m section
            beyond                                   # the pure extrapolated section
        ], ignore_index=True)

        # Clean up any duplicates from concatenation
        profile_extended = profile_extended.drop_duplicates(subset='depth', keep='first')


    # 9) Downsample the complete extended profile to the final grid
    grid_depths = np.arange(0, target_max_depth + 1, bin_size)
    interp_sounds = np.interp(grid_depths, profile_extended['depth'], profile_extended['sound_speed'])
    final_profile = pd.DataFrame({'depth': grid_depths, 'sound_speed': interp_sounds})
    final_profile = final_profile.round({'depth': 2, 'sound_speed': 3}) # Clean up precision

    # 10) Quick plot if requested
    if plot:
        plt.figure(figsize=(5, 8))
        plt.plot(final_profile['sound_speed'], final_profile['depth'], 'b-o', 
                 markersize=3, label=f'Final Profile ({bin_size}m grid)')
        plt.plot(profile['sound_speed'], profile['depth'], 'r.-', 
                 markersize=4, alpha=0.6, label=f'Original ({bin_size}m bin)')
        
        # Highlight the blended/extrapolated part
        if 'last_depth' in locals():
             plt.axhspan(last_depth, target_max_depth, color='gray', alpha=0.15, 
                         label='Extrap./Blended')

        plt.gca().invert_yaxis()
        plt.xlabel("Sound speed (m/s)")
        plt.ylabel("Depth (m)")
        plt.title(f"Extended SSP 0-{target_max_depth} m")
        plt.legend()
        plt.grid(True)
        plt.show()

    # 11) Save profile to CSV if requested
    if output_csv_path:
        try:
            final_profile.to_csv(output_csv_path, index=False)
            print(f"Saved extended profile to: {output_csv_path}")
        except Exception as e:
            print(f"Error saving file to {output_csv_path}: {e}")

    return final_profile


# --- Example Usage ---
# This block will only run if you execute this script directly (e.g., `python extend_ssp.py`)
# It will not run when you `import extend_ssp` in another file.
if __name__ == "__main__":

    # --- Parameters (edit for testing) ---
    INPUT_FILE = "../assets/subset_batch_1.csv"
    TARGET_DEPTH = 500
    BIN = 5
    OUTPUT_FILE = f"./assets/extended_ssp_0_{TARGET_DEPTH}m_FUNCTION_TEST.csv"
    # -------------------------------------

    print("Running SSP extension as a standalone script...")
    
    # Call the function
    ssp_df = process_and_extend_ssp(
        ctd_csv_path=INPUT_FILE,
        target_max_depth=TARGET_DEPTH,
        bin_size=BIN,
        blend_window=30,
        extrap_gradient=0.008,
        output_csv_path=OUTPUT_FILE,
        plot=True  # Set to True to see the plot
    )

    if ssp_df is not None:
        print("\n--- Final Profile (Head) ---")
        print(ssp_df.head())
        print("\n--- Final Profile (Tail) ---")
        print(ssp_df.tail())
    else:
        print("SSP generation failed.")