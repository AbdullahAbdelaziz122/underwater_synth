import pandas as pd
import glob
import os
import shutil
import math

def convert_feather_to_csv_batches(
    input_directory: str,
    output_directory: str,
    batch_size: int,
    file_pattern: str = '*.feather',
    csv_prefix: str = 'batch'
) -> None:
    """
    Reads all Feather files from an input directory, combines them into
    batches of a specified size, and saves each batch as a separate CSV file
    in the output directory.

    Args:
        input_directory (str):
            The path to the directory containing the feather files.
            Example: './my_data'
            
        output_directory (str):
            The path to the directory where the CSV files will be saved.
            The directory will be created if it doesn't exist.
            Example: './csv_output'
            
        batch_size (int):
            The number of feather files to combine into a single CSV file.
            Example: 100
            
        file_pattern (str, optional):
            The glob pattern to match files within the directory.
            Defaults to '*.feather'.
            Example: 'sales_*.feather'
            
        csv_prefix (str, optional):
            The prefix for the output CSV files. Files will be named
            like '{csv_prefix}_001.csv', '{csv_prefix}_002.csv', etc.
            Defaults to 'batch'.

    Returns:
        None:
            This function does not return a value. It saves files to disk
            and prints status messages.
    """
    
    # --- 1. Define the full search path ---
    search_path = os.path.join(input_directory, file_pattern)
    
    # --- 2. Find all the files ---
    # Sort to ensure consistent batching every time
    all_feather_files = sorted(glob.glob(search_path)) 
    
    if not all_feather_files:
        print(f"Warning: No files found matching '{search_path}'. No CSV files created.")
        return
        
    print(f"Found {len(all_feather_files)} files matching '{search_path}'.")
    
    # --- 3. Ensure output directory exists ---
    os.makedirs(output_directory, exist_ok=True)
    
    # --- 4. Process files in batches ---
    num_files = len(all_feather_files)
    # Use math.ceil to round up, ensuring the last batch is processed
    num_batches = math.ceil(num_files / batch_size) 
    
    # Determine padding for file numbers (e.g., 001, 002... or 01, 02...)
    # Use at least 3 digits, or more if there are over 999 batches
    padding = max(3, len(str(num_batches))) 

    print(f"Processing {num_files} files into {num_batches} batch(es) of size {batch_size}...")

    for i in range(num_batches):
        batch_number = i + 1
        start_index = i * batch_size
        end_index = start_index + batch_size
        
        # Get the slice of files for this batch
        batch_files = all_feather_files[start_index:end_index]
        
        print(f"--- Processing Batch {batch_number}/{num_batches} ({len(batch_files)} files) ---")
        
        # --- 5. Load each file in the batch into a list of DataFrames ---
        list_of_dfs = []
        for file in batch_files:
            try:
                df = pd.read_feather(file)
                list_of_dfs.append(df)
            except Exception as e:
                print(f"  Error reading {file}: {e}. Skipping this file.")
    
        # --- 6. Concatenate all DataFrames in the batch ---
        if not list_of_dfs:
            print(f"  Batch {batch_number} had no readable files. Skipping CSV creation.")
            continue
            
        try:
            # Combine all DataFrames for this batch
            batch_df = pd.concat(list_of_dfs, ignore_index=True)
        except pd.errors.InvalidIndexError as e:
            # This can happen if DFs have incompatible columns/indexes
            print(f"  Error concatenating files in batch {batch_number}: {e}. Skipping batch.")
            print(f"  Files in this batch: {batch_files}")
            continue

        # --- 7. Save the batch DataFrame to a CSV file ---
        # Format batch number with leading zeros (e.g., 001, 002)
        batch_filename = f"{csv_prefix}_{str(batch_number).zfill(padding)}.csv"
        output_path = os.path.join(output_directory, batch_filename)
        
        try:
            # Save to CSV, index=False is common to avoid an extra column
            batch_df.to_csv(output_path, index=False)
            print(f"  Successfully created {output_path} (Shape: {batch_df.shape})")
        except Exception as e:
            print(f"  Error writing to {output_path}: {e}")

    print(f"\n--- All processing complete. {num_batches} CSV batch file(s) created in {output_directory} ---")

# --- This block runs only when the script is executed directly ---
if __name__ == "__main__":
    
    print("--- Running demonstration for convert_feather_to_csv_batches ---")
    
    # --- A. Set up a temporary test environment ---
    temp_in_dir = 'temp_feather_test_in'
    temp_out_dir = 'temp_feather_test_out'
    os.makedirs(temp_in_dir, exist_ok=True)
    os.makedirs(temp_out_dir, exist_ok=True)
    
    print(f"Creating temporary directories: {temp_in_dir}, {temp_out_dir}")

    # Create dummy DataFrames
    df1 = pd.DataFrame({'a': [1, 2], 'b': ['x', 'y']})
    df2 = pd.DataFrame({'a': [3, 4], 'b': ['z', 'w']})
    df3 = pd.DataFrame({'a': [5, 6], 'b': ['a', 'b']})
    df4 = pd.DataFrame({'a': [7, 8], 'b': ['c', 'd']})
    df_other = pd.DataFrame({'c': [99, 100]})
    
    # Save them as feather files
    df1.to_feather(os.path.join(temp_in_dir, 'data_01.feather'))
    df2.to_feather(os.path.join(temp_in_dir, 'data_02.feather'))
    df3.to_feather(os.path.join(temp_in_dir, 'data_03.feather'))
    df4.to_feather(os.path.join(temp_in_dir, 'data_04.feather'))
    df_other.to_feather(os.path.join(temp_in_dir, 'other_data.arrow')) # Different extension
    
    print("Created 5 dummy files (4 .feather, 1 .arrow)")

    # --- B. Test 1: Batch size of 3 ---
    # This should create 2 files: batch_001.csv (3 feathers) and batch_002.csv (1 feather)
    print("\n--- Test 1: Running with batch_size = 3 ---")
    convert_feather_to_csv_batches(
        input_directory=temp_in_dir,
        output_directory=temp_out_dir,
        batch_size=3,
        csv_prefix='batch_test'
    )
    
    print("\n--- Files created in output directory (Test 1): ---")
    # List files to show they were created
    created_files = os.listdir(temp_out_dir)
    print(created_files)
    
    # --- C. Read back and check contents ---
    try:
        csv_path1 = os.path.join(temp_out_dir, 'batch_test_001.csv')
        read_csv1 = pd.read_csv(csv_path1)
        print(f"\nContent of {csv_path1} (Shape: {read_csv1.shape}):")
        print(read_csv1)
    except FileNotFoundError:
        print(f"\nFile {csv_path1} not found.")

    try:
        csv_path2 = os.path.join(temp_out_dir, 'batch_test_002.csv')
        read_csv2 = pd.read_csv(csv_path2)
        print(f"\nContent of {csv_path2} (Shape: {read_csv2.shape}):")
        print(read_csv2)
    except FileNotFoundError:
        print(f"\nFile {csv_path2} not found.")

    # --- D. Clean up the temporary directories ---
    print("\n--- Cleaning up temporary files ---")
    try:
        shutil.rmtree(temp_in_dir)
        shutil.rmtree(temp_out_dir)
        print(f"Successfully removed {temp_in_dir} and {temp_out_dir}")
    except Exception as e:
        print(f"Error cleaning up directories: {e}")
        
    print("\n--- Demonstration complete ---")

