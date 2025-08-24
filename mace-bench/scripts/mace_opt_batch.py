"""
Copyright (c) 2025 Ma Zhaojia

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import os
import argparse

parser = argparse.ArgumentParser(description="Run batch optimization on molecular crystals.")
parser.add_argument("--target_folder", type=str, required=True, help="Target folder containing crystal files")
parser.add_argument("--num_workers", type=int, default=4, help="Number of workers to distribute the files to")
parser.add_argument("--n_gpus", type=int, default=1, help="Number of GPUs to use for the optimization")
parser.add_argument("--gpu_offset", type=int, default=0, help="Offset for GPU numbering")
parser.add_argument("--batch_size", type=int, default=4, help="Number of files to process in a single batch")
parser.add_argument("--run_baseline", type=bool, default=False, help="Run baseline optimization using LBFGS from ase.optimize")
parser.add_argument("--max_steps", type=int, default=100, help="Number of max steps to run the optimization (default: 100)")
parser.add_argument("--filter1", type=str, default=None, 
                    choices=[None, "UnitCellFilter"],
                    help="Type of cell filter to use in first optimization")
parser.add_argument("--filter2", type=str, default=None,
                    choices=[None, "UnitCellFilter"],
                    help="Type of cell filter to use in second optimization")
parser.add_argument("--optimizer1", type=str, default="LBFGS",
                    choices=["LBFGS", "QuasiNewton", "BFGS", "BFGSLineSearch", "BFGSFusedLS"],
                    help="First optimizer to use (default: LBFGS)")
parser.add_argument("--optimizer2", type=str, default="LBFGS",
                    choices=["LBFGS", "QuasiNewton", "BFGS", "BFGSLineSearch", "BFGSFusedLS"],
                    help="Second optimizer to use (default: LBFGS)")
parser.add_argument("--skip_second_stage", type=bool, default=False, help="Skip the second optimization stage")
parser.add_argument("--scalar_pressure", type=float, default=0.0006,
                    help="Scalar pressure for cell optimization (default: 0.0006)")
parser.add_argument("--compile_mode", type=str, default=None, 
                    choices=[None, "default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"],
                    help="Compile mode for MACE calculator")
parser.add_argument("--profile", type=str, default="False", 
                    help="Enable profiling. Set to 'True' for basic profiling or provide a JSON string with profiler config options for wait, warmup, active, and repeat")
parser.add_argument("--num_threads", type=int, default=16, help="Number of cpu threads per process to use while running the optimization")
parser.add_argument("--bind_cores", type=str, default=None,  
                    help=("Specify a comma-separated list of core ranges (e.g., '0-15,16-31,...') for each worker. The number of ranges must equal --num_workers."))
parser.add_argument("--cueq", type=bool, default=False, help="Whether to use cuEquivariance Library (default: False)")
parser.add_argument("--molecule_single", type=int, default=64, help="Number of atoms per molecule (default: 64)")
parser.add_argument("--output_path", type=str, default="./", help="Absolute path for output files")
parser.add_argument("--model", type=str, default="mace", choices=["mace", "chgnet", "sevennet"], help="Model to use for optimization")
parser.add_argument("--use_ordered_files", type=bool, default=False, 
                    help="Whether to sort files by atomic number in descending order before optimization")
args = parser.parse_args()

os.environ['OMP_NUM_THREADS'] = str(args.num_threads)
os.environ['MKL_NUM_THREADS'] = str(args.num_threads)

import pathlib
import logging
from batchopt import Scheduler, ensure_directory, run_baseline, count_atoms_cif
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(process)d - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S',
    force=True
)

if __name__ == '__main__':
    target_folder = pathlib.Path(args.target_folder)
    files = [str(file) for file in target_folder.glob("*.cif")]
    devices = [f"cuda:{i}" for i in range(args.gpu_offset, args.gpu_offset + args.n_gpus)]

    logging.info("Starting batch optimization.")
    logging.info(f"Use devices: {devices}")
    logging.info(f"files: {files}")
    
    output_path = args.output_path
    if not os.path.isabs(output_path):
        output_path = os.path.abspath(output_path)
    logging.info(f"Output path: {output_path}")
    
    for output_dir in ["cif_result_press", "cif_result_final", "json_result_press", "json_result_final", "worker_results", "log"]:
        dir_path = os.path.join(output_path, output_dir)
        ensure_directory(dir_path)

    import time 
    start_time = time.perf_counter()

    use_ordered_files = args.use_ordered_files
    if use_ordered_files:
        logging.info(f"Use ordered files.")
        if files[0].endswith("cif"):
            files = sorted(files, key=count_atoms_cif, reverse=True)
        else:
            logging.error(f"No support for the file type in {target_folder}.")
    end_time = time.perf_counter()
    logging.info(f"atomic sorting time: {end_time - start_time:.4f} seconds.")
    
    if args.run_baseline:
        run_baseline(files, args.num_workers, devices, args.max_steps, 
                     args.filter1, args.filter2, args.skip_second_stage, 
                     args.scalar_pressure, args.optimizer1, args.optimizer2,
                     output_path=output_path)
    else:
        scheduler = Scheduler(files=files, num_workers=args.num_workers, devices=devices,
                            batch_size=args.batch_size, max_steps=args.max_steps,
                            filter1=args.filter1, filter2=args.filter2,
                            skip_second_stage=args.skip_second_stage,
                            scalar_pressure=args.scalar_pressure, optimizer1=args.optimizer1, optimizer2=args.optimizer2,
                            compile_mode=args.compile_mode, profile=args.profile,
                            num_threads=args.num_threads, bind_cores=args.bind_cores, 
                            cueq=args.cueq, molecule_single=args.molecule_single,
                            output_path=output_path, model=args.model)
        scheduler.run()

    logging.info("Batch optimization completed.")
