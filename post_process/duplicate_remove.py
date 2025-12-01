import sys
import os
import glob
import pandas as pd
import warnings
from pathlib import Path
from tqdm import tqdm
import multiprocessing as mp
from ccdc.crystal import PackingSimilarity
from ccdc.io import CrystalReader
import shutil
import argparse

warnings.filterwarnings("ignore", category=DeprecationWarning)

#########################
# Global Settings
#########################

# The number of matching molecules required in a packing shell to consider two crystals similar.
molecule_shell_size = 15

# Initialize the packing similarity engine from the CCDC API.
similarity_engine = PackingSimilarity()

# Configure the similarity engine settings.
similarity_engine.settings.allow_molecular_differences = False
similarity_engine.settings.distance_tolerance = 0.2
similarity_engine.settings.angle_tolerance = 20
similarity_engine.settings.packing_shell_size = molecule_shell_size

# Settings to make the comparison less strict regarding hydrogen atoms and bond counts.
similarity_engine.settings.ignore_hydrogen_positions = True
similarity_engine.settings.ignore_bond_counts = True
similarity_engine.settings.ignore_hydrogen_counts = True

# Pre-filtering thresholds to avoid expensive comparisons for vastly different structures.
ENERGY_DIFF = 5.0
DENSITY_DIFF = 0.2


def compare_sim_pack_pair(args):
    """
    Compares two crystal structures to determine if they are identical based on packing similarity.
    Accepts a tuple (candidate_fp, ufp) containing file paths to the two structures.
    Returns True if the structures are considered the same, False otherwise.
    """
    cand_fp, ufp = args
    try:
        c1 = CrystalReader(cand_fp)[0]
        c2 = CrystalReader(ufp)[0]
        h = similarity_engine.compare(c1, c2)
        # Structures are deemed identical if the number of matched molecules meets the global shell size.
        return h.nmatched_molecules >= molecule_shell_size
    except Exception:
        # Return False if any error occurs during file reading or comparison.
        return False

def find_top_unique(struct_list, target_count, n_workers):
    """
    Identifies a target number of unique structures from a list, sorted by energy.
    It iterates through structures from lowest to highest energy and uses parallel processing
    to compare each candidate against the list of already confirmed unique structures.

    Args:
        struct_list (list): A list of tuples, where each tuple is (density, energy, filepath).
        target_count (int): The desired number of unique structures to find.
        n_workers (int): The number of worker processes for parallel comparison.

    Returns:
        tuple: A tuple containing:
            - A list of file paths for the unique structures found.
            - A dictionary mapping each unique structure's path to a list of its duplicate paths.
    """
    unique_list = []
    
    # A dictionary to store each unique structure and its corresponding duplicates.
    # Format: {unique_fp: [dup_fp1, dup_fp2, ...]}
    duplicate_map = {}
    
    # Initialize the multiprocessing pool.
    pool = mp.Pool(processes=n_workers)
    try:
        # Iterate through the main structure list, which is pre-sorted by energy.
        for dens, ene, cand_fp in struct_list:
            # Stop if the target count of unique structures has been reached.
            if len(unique_list) >= target_count:
                break

            # Create comparison tasks only for unique structures within the density and energy thresholds.
            tasks = [(cand_fp, ufp) for dens2, ene2, ufp in unique_list if abs(dens2 - dens) < DENSITY_DIFF and abs(ene2 - ene) < ENERGY_DIFF]
            is_dup = False
            
            if tasks:
                # Use tqdm for a progress bar during parallel comparison.
                results_iterator = tqdm(pool.imap(compare_sim_pack_pair, tasks),
                                        total=len(tasks),
                                        desc=f"Comparing {os.path.basename(cand_fp)}",
                                        leave=False)
                # Convert iterator to a list to process results.
                results_list = list(results_iterator)
                
                # Pair the boolean results with the original tasks to identify which comparison was successful.
                for result, task in zip(results_list, tasks):
                    if result:
                        is_dup = True
                        # The second element of the task tuple is the path of the matched unique structure.
                        matched_unique_fp = task[1]
                        
                        # Record the current candidate as a duplicate of the matched unique structure.
                        duplicate_map[matched_unique_fp].append(cand_fp)
                        
                        # Once a duplicate is found, no more comparisons are needed for this candidate.
                        break

            if not is_dup:
                # If the candidate is not a duplicate of any existing unique structure, add it to the list.
                unique_list.append((dens, ene, cand_fp))
                
                # Create a new entry in the duplicate map for this new unique structure.
                duplicate_map[cand_fp] = []
                
                # Print real-time progress.
                print(f">>> Unique count: {len(unique_list)}  (added {os.path.basename(cand_fp)})")

    finally:
        # Ensure the multiprocessing pool is properly closed.
        pool.close()
        pool.join()
        
    # Return the list of unique file paths and the map of duplicates.
    return [fp for _, _, fp in unique_list], duplicate_map


def process_folder(folder_path):
    """
    Main workflow function to process a given folder. It reads structural data,
    finds unique structures, and organizes the results into folders and a CSV file.
    """
    base_path = folder_path

    # Define output directories for unique structures and their duplicates.
    t_folder = os.path.join(base_path, f"unique_{TARGET_UNIQUE_COUNT}")
    d_folder = os.path.join(base_path, f"unique_{TARGET_UNIQUE_COUNT}_duplicates")
    
    # Load the summary data from the CSV file.
    df = pd.read_csv(os.path.join(base_path, "results_scheduler.csv"))
    cif_folder = os.path.join(base_path, "cif_result_final")
    #########################

    # Create a map from a simplified filename to its full file path for quick lookup.
    file_paths = glob.glob(os.path.join(cif_folder, "*.cif"))
    file_map = {}
    for fp in file_paths:
        base = Path(fp).stem
        # Standardize the name by removing "_opt" suffix if it exists.
        if base.endswith("_opt"):
            key = base[:-4]
        else:
            key = base
        file_map[key] = fp

    # Build the list of candidate structures, filtering by the energy threshold.
    struct_list = []
    for _, row in df.iterrows():
        name = row["file"]
        dens = row["stage2_density"]
        ene  = row["relative_energy"]
        if ene > ENERGY_THRESHOLD:
            continue
        fp = file_map.get(name)
        if fp:
            struct_list.append((dens, ene, fp))
        else:
            print(f"Warning: no CIF for {name}")

    # Sort the list of structures by energy in ascending order.
    struct_list.sort(key=lambda x: x[1])

    # Call the main function to find unique structures and the duplicate map.
    unique_paths, duplicate_map = find_top_unique(
        struct_list,
        target_count=TARGET_UNIQUE_COUNT,
        n_workers=WORKER_COUNT
    )

    # Prepare output directories, removing them first if they already exist.
    if os.path.exists(t_folder):
        shutil.rmtree(t_folder)
    os.makedirs(t_folder)

    if os.path.exists(d_folder):
        shutil.rmtree(d_folder)
    os.makedirs(d_folder)
    
    print(f"\nFound {len(unique_paths)} unique structures "
          f"(energy <= {ENERGY_THRESHOLD}, target {TARGET_UNIQUE_COUNT}).")
    
    # Copy the unique structure files to the target folder.
    for p in unique_paths:
        shutil.copy(p, t_folder)

    # Helper function to get a clean filename without the "_opt" suffix.
    get_clean_name = lambda p: Path(p).stem.replace('_opt', '')

    # Create a new DataFrame containing only the data for the unique structures.
    unique_names = [get_clean_name(p) for p in unique_paths]
    unique_df = df[df['file'].isin(unique_names)].copy()
    unique_df['duplicates'] = '' # Add a new column to store duplicate names.

    # Populate the 'duplicates' column and copy duplicate files to their folder.
    for unique_fp, duplicates_list in duplicate_map.items():
        unique_name = get_clean_name(unique_fp)
        if unique_name in unique_df['file'].values:
            du_name = [get_clean_name(p) for p in duplicates_list]
            # Add the list of duplicate names as a comma-separated string.
            unique_df.loc[unique_df['file'] == unique_name, 'duplicates'] = ', '.join(du_name)
        # Copy duplicate files to the duplicates folder.
        for p in duplicates_list:
            shutil.copy(p, d_folder)

    # Save the final DataFrame with unique structures and their duplicates to a new CSV file.
    unique_csv_path = os.path.join(base_path, 'unique_structures.csv')
    unique_df.to_csv(unique_csv_path, index=False)

if __name__ == '__main__':
    # Required for freezing the application when creating executables with multiprocessing.
    mp.freeze_support()
    
    # Set up command-line argument parsing.
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default="./", help='Path to process')
    parser.add_argument('--energy', type=float, default=30, help='Energy threshold for filtering structures')
    parser.add_argument('--count', type=int, default=200, help='Target number of unique structures to find')
    parser.add_argument('--workers', type=int, default=80, help='Max worker number limit')
    args = parser.parse_args()
    
    # Set global variables from command-line arguments.
    target_folder = args.path
    ENERGY_THRESHOLD = args.energy
    TARGET_UNIQUE_COUNT = args.count
    WORKER_COUNT = args.workers

    # Start the processing if the specified folder exists.
    if os.path.exists(target_folder):
        print(f"Processing folder: {target_folder}")
        process_folder(target_folder)