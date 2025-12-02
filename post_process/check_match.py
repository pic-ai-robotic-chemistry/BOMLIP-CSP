import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from ccdc.crystal import PackingSimilarity
from ccdc.io import CrystalReader
import glob
import os
import sys
import random
import pandas as pd
from multiprocessing import Pool, cpu_count, TimeoutError as mpTimeoutError # import TimeoutError
import argparse

# --- Global Configuration ---
REPORT_TARGET = 15
LARGE_CONFORMER_DIFF = True

# --- Worker Process Initializer ---
def init_worker(ref_path, engine_settings):
    """
    Initializes a worker process.
    This function loads the reference structure and creates the similarity engine
    once per process, storing them in global variables for that process.
    """
    # print(f"Worker process {os.getpid()} initializing...")
    # sys.stdout.flush()
    global worker_ref_crystal
    global worker_similarity_engine
    worker_ref_crystal = CrystalReader(ref_path)[0]
    worker_similarity_engine = PackingSimilarity()
    worker_similarity_engine.settings.allow_molecular_differences = engine_settings['allow_molecular_differences']
    worker_similarity_engine.settings.distance_tolerance = engine_settings['distance_tolerance']
    worker_similarity_engine.settings.angle_tolerance = engine_settings['angle_tolerance']
    worker_similarity_engine.settings.packing_shell_size = engine_settings['packing_shell_size']
    worker_similarity_engine.settings.ignore_hydrogen_positions = engine_settings['ignore_hydrogen_positions']
    worker_similarity_engine.settings.ignore_bond_counts = engine_settings['ignore_bond_counts']
    worker_similarity_engine.settings.ignore_hydrogen_counts = engine_settings['ignore_hydrogen_counts']

# --- Single Task Processing Function ---
def process_single_cif(csp_file_path):
    """
    Compares a single candidate structure against the reference structure
    loaded in the worker's global scope.
    Returns a tuple indicating the result type ('matched' or 'failed') and the file path.
    """
    global worker_ref_crystal
    global worker_similarity_engine
    try:
        try_structure = CrystalReader(csp_file_path)[0]
        h = worker_similarity_engine.compare(try_structure, worker_ref_crystal)
        if h.nmatched_molecules >= REPORT_TARGET:
            print(f"MATCH: {os.path.basename(csp_file_path)} | Matched Molecules: {h.nmatched_molecules}, RMSD: {h.rmsd:.3f}")
            sys.stdout.flush()
            return ("matched", csp_file_path)
    except Exception as e:
        if not LARGE_CONFORMER_DIFF:
            print(f"FAIL: {os.path.basename(csp_file_path)} | Reason: {e}")
            sys.stdout.flush()
        return ("failed", csp_file_path)
    return None

# --- Main Execution Block ---
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default="./", help='Path to process')
    parser.add_argument('--ref_path', type=str, default="../refs", help='Path to find reference structrues')
    parser.add_argument('--workers', type=int, default=80, help='Max worker number limit')
    parser.add_argument('--timeout', type=int, default=20, help='Timeout for each task in seconds')

    args = parser.parse_args()
    base_path = args.path
    refs_dir = args.ref_path
    PROCESS_NUM = min(args.workers, cpu_count())  # Use the specified number of workers or the max available
    TIMEOUT_SECONDS = args.timeout  # Set the timeout for each task
    all_results = []
    
    print(f"Starting checking match using up to {PROCESS_NUM} processes with a {TIMEOUT_SECONDS}s timeout per task...")

    folders_to_process = []
    csp_dir = os.path.join(base_path, "cif_result_final")
    if os.path.exists(csp_dir) and os.path.exists(refs_dir):
        folders_to_process.append((csp_dir, refs_dir))

    for csp_dir, refs_dir in folders_to_process:
        for ref_filename in os.listdir(refs_dir):
            if not ref_filename.endswith(".cif"):
                continue

            ref_full_path = os.path.join(refs_dir, ref_filename)
            print(f"\n--- Processing Reference File: {ref_full_path} ---")
            
            csp_files = glob.glob(os.path.join(csp_dir, '*.cif'))
            random.shuffle(csp_files)

            if not csp_files:
                print("No candidate .cif files found, skipping.")
                continue

            engine_settings = {
                'allow_molecular_differences': False,
                'distance_tolerance': 0.2,
                'angle_tolerance': 20,
                'packing_shell_size': 15,
                'ignore_hydrogen_positions': True,
                'ignore_bond_counts': True,
                'ignore_hydrogen_counts': True
            }
            
            with Pool(processes=PROCESS_NUM, initializer=init_worker, initargs=(ref_full_path, engine_settings)) as pool:
                
                async_results = []
                for f in csp_files:
                    res = pool.apply_async(process_single_cif, args=(f,))
                    async_results.append(res)

                results_list = []
                for i, res_obj in enumerate(async_results):
                    try:
                        result = res_obj.get(timeout=TIMEOUT_SECONDS)
                        results_list.append(result)
                    except mpTimeoutError:
                        timed_out_file = csp_files[i]
                        print(f"TIMEOUT: {timed_out_file} | Task exceeded {TIMEOUT_SECONDS}s limit.")
                        sys.stdout.flush()
                        results_list.append(("failed", timed_out_file))
            
            matched_structures = []
            failed_structures = []
            for res in results_list:
                if res:
                    status, path = res
                    if status == "matched":
                        matched_structures.append(os.path.basename(path))
                    elif status == "failed":
                        failed_structures.append(os.path.basename(path))

            all_results.append({
                "ref_name": ref_filename,
                "matched_count": len(matched_structures),
                "matched_structures": ";".join(matched_structures),
                "failed_count": len(failed_structures),
                "failed_structures": ";".join(failed_structures)
            })
            print(f"--- Finished {ref_filename}. Matched: {len(matched_structures)}, Failed: {len(failed_structures)} ---")

    if all_results:
        df = pd.DataFrame(all_results)
        output_filename = "match_results.csv"
        df.to_csv(output_filename, index=False)
        print(f"\nAll processing finished. Results saved to {output_filename}")
    else:
        print("\nNo valid data processed. No output file generated.")