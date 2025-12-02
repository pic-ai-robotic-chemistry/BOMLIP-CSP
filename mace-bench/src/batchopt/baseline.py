"""
Copyright (c) 2025 {Chengxi Zhao, Zhaojia Ma, Dingrui Fan}

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from ase.io import read
import logging
from joblib import Parallel, delayed
from ase.optimize import LBFGS as ASE_LBFGS
from ase.optimize import QuasiNewton as ASE_QuasiNewton
from ase.optimize import BFGS as ASE_BFGS
import time
import csv
import os
try:
    from mace.calculators import mace_off
except ImportError:
    logging.warning("Failed to import MACE modules")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def ensure_directory(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        logging.info(f"Created directory: {directory}")

def baseline_task(file, device, max_steps, filter1=None, filter2=None, skip_second_stage=False, scalar_pressure=0.0006, first_optimizer="LBFGS", second_optimizer="LBFGS"):
    """
    Runs the baseline optimization using LBFGS from ase.optimize.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = device.split(":")[-1]
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info(f"Starting baseline optimization for file {file} on device {device}.")


    start_time = time.perf_counter()

    crystal = read(file)
    # calc = mace_off(model="small", device=device)
    calc = mace_off(model="small", device="cuda")
    crystal.calc = calc

    first_optimizer_class ={
        "LBFGS": ASE_LBFGS,
        "QuasiNewton": ASE_QuasiNewton,
        "BFGS": ASE_BFGS
    }.get(first_optimizer, ASE_LBFGS)

    # First optimization stage
    if filter1 == "UnitCellFilter":
        from ase.filters import UnitCellFilter
        atoms_with_filter = UnitCellFilter(crystal, scalar_pressure=scalar_pressure)
        first_optimizer_instance = first_optimizer_class(atoms_with_filter)
    elif filter1 == "FrechetCellFilter":
        from ase.filters import FrechetCellFilter
        atoms_with_filter = FrechetCellFilter(crystal, scalar_pressure=scalar_pressure)
        first_optimizer_instance = first_optimizer_class(atoms_with_filter)
    else:
        first_optimizer_instance = first_optimizer_class(crystal)
    
    start_time1 = time.perf_counter()
    first_optimizer_instance.run(fmax=0.01, steps=max_steps)
    end_time1 = time.perf_counter()
    
    # Save intermediate result
    output_dir_press = "./cif_result_press"
    output_file_press = os.path.join(output_dir_press, os.path.basename(file).replace(".cif", "_press.cif"))
    crystal.write(output_file_press)

    elapsed_time1 = end_time1 - start_time1
    steps1 = first_optimizer_instance.nsteps

    if skip_second_stage:
        
        ret_result = {
            "file": file,
            "stage1_time": elapsed_time1,
            "stage1_steps": steps1,
            "stage2_time": 0.0,
            "stage2_steps": 0,
            "total_time": elapsed_time1,
            "total_steps": steps1
        }
    else:
        # Second optimization stage
        crystal = read(output_file_press)
        crystal.calc = calc

        second_optimizer_class = {
            "LBFGS": ASE_LBFGS,
            "QuasiNewton": ASE_QuasiNewton,
            "BFGS": ASE_BFGS
        }.get(second_optimizer, ASE_LBFGS)
        
        if filter2 == "UnitCellFilter":
            from ase.filters import UnitCellFilter
            atoms_with_filter2 = UnitCellFilter(crystal)
            second_optimizer_instance = second_optimizer_class(atoms_with_filter2)
        elif filter2 == "FrechetCellFilter":
            from ase.filters import FrechetCellFilter
            atoms_with_filter2 = FrechetCellFilter(crystal)
            second_optimizer_instance = second_optimizer_class(atoms_with_filter2)
        else:
            second_optimizer_instance = second_optimizer_class(crystal)
        
        start_time2 = time.perf_counter()
        second_optimizer_instance.run(fmax=0.01, steps=max_steps)
        end_time2 = time.perf_counter()

        # Save final result
        output_dir_final = "./cif_result_final"
        output_file_final = os.path.join(output_dir_final, os.path.basename(file).replace(".cif", "_opt.cif"))
        crystal.write(output_file_final)

        # Collect metrics
        elapsed_time2 = end_time2 - start_time2
        total_time = elapsed_time1 + elapsed_time2
        steps2 = second_optimizer_instance.nsteps

        ret_result = {
            "file": file,
            "stage1_time": elapsed_time1,
            "stage1_steps": steps1,
            "stage2_time": elapsed_time2,
            "stage2_steps": steps2,
            "total_time": total_time,
            "total_steps": steps1 + steps2
        }

    logging.info(f"Baseline optimization completed for file {file}.")
    return ret_result

def run_baseline(files, num_workers, devices, max_steps, 
                 filter1=None, filter2=None, skip_second_stage=False, scalar_pressure=0.0006,
                 optimizer1=None, optimizer2=None):
    """
    Runs the baseline optimization using LBFGS from ase.optimize.
    """
    logging.info(f"Starting baseline optimization with {num_workers} workers.")

    start_time = time.perf_counter()
    results = Parallel(n_jobs=num_workers)(
        delayed(baseline_task)(file, devices[i % len(devices)], max_steps, filter1, filter2, skip_second_stage, scalar_pressure, optimizer1, optimizer2)
        for i, file in enumerate(files)
    )
    end_time = time.perf_counter()

    csv_file = "results_baseline.csv"
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=["file", "stage1_time", "stage1_steps", "stage2_time", "stage2_steps", "total_time", "total_steps"])
        writer.writeheader()
        for result in results:
            writer.writerow(result)

    logging.info(f"Baseline optimization completed in {end_time - start_time:.2f} seconds.")
    final_elapsed_time = end_time - start_time
    summary_csv_file = "summary_baseline.csv"
    with open(summary_csv_file, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=["elapsed_time", "num_workers", "batch_size"])
        writer.writeheader()
        writer.writerow({
            "elapsed_time": final_elapsed_time,
            "num_workers": num_workers,
            "batch_size": 1
        })

    logging.info(f"Summary results written to {summary_csv_file}.")