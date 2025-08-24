"""
Copyright (c) 2025 {Chengxi Zhao, Zhaojia Ma, Dingrui Fan}

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from ase.io import read

# from ase.optimize import ASE_LBFGS
import torch
from torch.multiprocessing import Process, set_start_method
from batchopt.atoms_to_graphs import AtomsToGraphs
from batchopt.utils import data_list_collater
from batchopt.relaxation.optimizers import (
    BFGS,
    BFGSFusedLS,
)
from batchopt.relaxation import OptimizableBatch, OptimizableUnitCellBatch
import logging
import time
import csv
from multiprocessing import Queue
import os
import psutil
import multiprocessing
import json
import subprocess

try:
    from chgnet.model.dynamics import CHGNetCalculator
except ImportError:
    logging.warning("Failed to import CHGNet modules")

try:
    from sevenn.calculator import SevenNetCalculator, SevenNetD3Calculator
except ImportError:
    logging.warning("Failed to import SevenNet modules")

try:
    from fairchem.core import pretrained_mlip, FAIRChemCalculator
except ImportError:
    logging.warning("Failed to import FAIRChem modules")

try:
    from mace.calculators import mace_off
except ImportError:
    logging.warning("Failed to import MACE modules")

import threading
from .utils import count_atoms_cif
from collections import deque


class Scheduler:
    """
    Scheduler distributes relaxation tasks to workers.
    """

    def __init__(
        self,
        files,
        num_workers,
        devices,
        batch_size,
        max_steps,
        filter1,
        filter2,
        optimizer1,
        optimizer2,
        skip_second_stage,
        scalar_pressure,
        compile_mode,
        profile,
        num_threads,
        bind_cores,
        cueq,
        molecule_single,
        output_path,
        model,
    ):

        self.files = files
        self.num_workers = num_workers
        self.devices = devices
        self.batch_size = batch_size
        self.max_steps = max_steps
        self.filter1 = filter1
        self.filter2 = filter2
        self.optimizer1 = optimizer1
        self.optimizer2 = optimizer2
        self.skip_second_stage = skip_second_stage
        self.scalar_pressure = scalar_pressure
        self.compile_mode = compile_mode
        self.profile = profile
        self.num_threads = num_threads
        self.cueq = cueq
        self.molecule_single = molecule_single
        self.output_path = (
            output_path
            if os.path.isabs(output_path)
            else os.path.abspath(output_path)
        )
        self.model = model

        try:
            set_start_method("spawn")
        except RuntimeError:
            logging.warning(
                "set_start_method('spawn') failed, trying 'forkserver' instead."
            )

        if bind_cores is not None:
            self.cpu_mask = self._parse_bind_cores(bind_cores)
        else:
            self.cpu_mask = None

    def _parse_bind_cores(self, bind_cores):
        # Expect custom_bind_str to be like "0-15,16-31,..."
        ranges = bind_cores.split(",")
        if len(ranges) != self.num_workers:
            return None
        binding = []
        for r in ranges:
            try:
                start_str, end_str = r.split("-")
                start = int(start_str)
                end = int(end_str)
            except ValueError:
                logging.error("Custom binding format should be 'start-end'.")
                return None

            binding.append(set(range(start, end + 1)))
        return binding

    def _get_physical_logical_core_mapping(self):
        """Get the mapping between logical cores and their physical core IDs."""
        try:
            # This information is available in Linux systems
            mapping = {}
            logical_cores = psutil.cpu_count(logical=True)

            for i in range(logical_cores):
                try:
                    # Read core_id from /sys/devices/system/cpu/cpu{i}/topology/core_id
                    with open(
                        f"/sys/devices/system/cpu/cpu{i}/topology/core_id"
                    ) as f:
                        core_id = int(f.read().strip())
                    # Read physical_package_id (socket) for more complete information
                    with open(
                        f"/sys/devices/system/cpu/cpu{i}/topology/physical_package_id"
                    ) as f:
                        package_id = int(f.read().strip())
                    mapping[i] = (package_id, core_id)
                except (FileNotFoundError, ValueError, IOError):
                    mapping[i] = None
            return mapping
        except Exception as e:
            logging.error(f"Failed to get core mapping: {e}")
            return {}

    def _get_physical_core_mask(self):
        # Get the number of physical and logical cores
        physical_cores = psutil.cpu_count(logical=False)
        logical_cores = psutil.cpu_count(logical=True)

        if physical_cores is None or physical_cores < 1:
            # Fallback to multiprocessing if psutil fails
            logical_cores = multiprocessing.cpu_count()
            physical_cores = logical_cores // 2
            if physical_cores < 1:
                physical_cores = 1
            print(f"Using estimated physical cores: {physical_cores}")

        # Get the mapping between logical and physical cores
        core_mapping = self._get_physical_logical_core_mapping()

        # Create a CPU mask that includes all physical cores (first core of each physical core)
        physical_core_mask = set()
        if core_mapping:
            # Group by physical core ID
            cores_by_physical = {}
            for logical_id, physical_info in core_mapping.items():
                if physical_info is not None:
                    package_id, core_id = physical_info
                    key = (package_id, core_id)
                    if key not in cores_by_physical:
                        cores_by_physical[key] = []
                    cores_by_physical[key].append(logical_id)

            # Select one logical core from each physical core
            for physical_cores_list in cores_by_physical.values():
                physical_core_mask.add(
                    physical_cores_list[0]
                )  # First logical core of each physical core
        else:
            # If mapping fails, use a simple assumption (may not be accurate on all systems)
            threads_per_core = logical_cores // physical_cores
            physical_core_mask = set(range(0, logical_cores, threads_per_core))

        return physical_core_mask

    def worker_task(
        self, files, device, batch_size, result_queue, physical_cores
    ):
        if physical_cores is not None:
            try:
                # Bind the current process to physical cores
                pid = os.getpid()
                os.sched_setaffinity(pid, physical_cores)
                logging.info(f"bind to physical_core_ids: {physical_cores}")

                # Verify the affinity was set correctly
                current_affinity = os.sched_getaffinity(pid)
                logging.info(
                    f"Process bound to {len(current_affinity)} cores: {sorted(current_affinity)}"
                )

            except AttributeError:
                logging.error(
                    "sched_setaffinity not supported on this platform"
                )
            except Exception as e:
                logging.error(f"Failed to bind to physical cores: {e}")

        # pass the number of processes on each worker
        nproc = self.num_workers // len(self.devices)

        worker = Worker(
            files,
            device,
            batch_size,
            self.max_steps,
            self.filter1,
            self.filter2,
            self.optimizer1,
            self.optimizer2,
            self.skip_second_stage,
            self.scalar_pressure,
            self.compile_mode,
            self.profile,
            self.cueq,
            self.molecule_single,
            self.output_path,
            self.model,
            nproc,
        )
        # results = worker.run()
        results = worker.continuous_run()
        result_queue.put(results)

    def _terminate_processes(self, processes):
        """Helper method to terminate all processes."""
        for i, p in processes:
            if p.is_alive():
                logging.info(f"Terminating process {p.pid}")
                p.terminate()
                p.join(timeout=3)  # Wait for up to 3 seconds
                if p.is_alive():
                    logging.warning(
                        f"Process {p.pid} did not terminate, killing it"
                    )
                    p.kill()
                    p.join()

    # create a thread to conduct "nvidia-smi"
    @staticmethod
    def _monitor_memory(interval=2, gpu_index=1):
        try:
            while True:
                result = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=memory.used,memory.total",
                        "--format=csv,nounits,noheader",
                    ]
                ).decode("utf-8")

                lines = result.strip().split("\n")
                used, total = map(int, lines[gpu_index].split(","))
                logging.info(
                    f"[nvidia-smi] Memory-Usage on GPU {gpu_index}: {used}MiB / {total}MiB"
                )

                time.sleep(interval)
        except KeyboardInterrupt:
            logging.info("Monitor interrupted.")

        except Exception as e:
            logging.error(f"Unexpected error when monitor memory: {str(e)}")

    def run(self):
        logging.info(f"Starting Scheduler with {self.num_workers} workers.")
        processes = []
        result_queue = Queue()
        start_time = time.perf_counter()

        if self.cpu_mask is not None:
            physical_cores_per_worker = self.cpu_mask
            logging.info(
                f"Use customed cores binding. Physical cores per worker: {physical_cores_per_worker}"
            )
        else:
            # all_physical_cores = self._get_physical_core_mask()
            # num_per_worker = len(all_physical_cores) // self.num_workers
            # physical_cores_per_worker = [
            #     list(all_physical_cores)[i:i + num_per_worker] for i in range(0, len(all_physical_cores), num_per_worker)
            # ]
            # logging.info(f"Physical cores per worker: {physical_cores_per_worker}")
            physical_cores_per_worker = [None] * self.num_workers

        try:
            # Start all worker processes
            for i in range(self.num_workers):
                files_for_worker = self.files[i :: self.num_workers]
                device = self.devices[i % len(self.devices)]
                logging.info(
                    f"Starting worker {i} with {len(files_for_worker)} files on device {device}."
                )
                p = Process(
                    target=self.worker_task,
                    args=(
                        files_for_worker,
                        device,
                        self.batch_size,
                        result_queue,
                        physical_cores_per_worker[i],
                    ),
                )
                p.start()
                processes.append((i, p))

            # monitor gpu memory usage to figure out what makes the differences of footprint among batches
            # in each iteration.
            use_memory_monitor = False
            if use_memory_monitor:
                monitor_proc = Process(
                    target=Scheduler._monitor_memory, args=()
                )
                monitor_proc.start()

            # Monitor processes and collect results
            csv_paths = []
            completed_processes = 0
            while completed_processes < self.num_workers:
                for i, p in processes:
                    if not p.is_alive() and p.exitcode != 0:
                        if p.exitcode == -11 or p.exitcode == 1:
                            # Restart the process if exit code is -11 or -1
                            logging.warning(
                                f"Worker process {p.pid} exited with code {p.exitcode}. Restarting worker {i}."
                            )
                            files_for_worker = self.files[i :: self.num_workers]
                            device = self.devices[i % len(self.devices)]
                            new_process = Process(
                                target=self.worker_task,
                                args=(
                                    files_for_worker,
                                    device,
                                    self.batch_size,
                                    result_queue,
                                    physical_cores_per_worker[i],
                                ),
                            )
                            new_process.start()
                            processes[i] = (
                                i,
                                new_process,
                            )  # Replace the old process with the new one
                        else:
                            # Raise an error for other exit codes
                            raise RuntimeError(
                                f"Worker process {p.pid} failed with exit code {p.exitcode}"
                            )

                # Try to get result from queue with timeout
                try:
                    result = result_queue.get(timeout=10)
                    csv_paths.append(result)
                    completed_processes += 1
                except Exception as e:
                    continue

            # terminate monitor
            if use_memory_monitor:
                monitor_proc.terminate()
                monitor_proc.join()

            # Process results and create final CSV
            merged_results = []
            for csv_path in csv_paths:
                try:
                    with open(csv_path, mode="r") as f:
                        reader = csv.DictReader(f)
                        merged_results.extend(list(reader))
                except Exception as e:
                    logging.error(f"Error processing {csv_path}: {str(e)}")

        except Exception as e:
            # Log the error and elapsed time
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
            logging.error(
                f"Error occurred after running for {elapsed_time:.2f} seconds: {str(e)}"
            )

            # Create error log file
            error_log = f"scheduler_error_{int(time.time())}.log"
            with open(error_log, "w") as f:
                f.write(f"Error occurred after {elapsed_time:.2f} seconds\n")
                f.write(f"Error message: {str(e)}\n")
                f.write(f"Number of workers: {self.num_workers}\n")
                f.write(f"Batch size: {self.batch_size}\n")

            # Terminate all processes
            self._terminate_processes(processes)
            raise  # Re-raise the exception after cleanup

        finally:
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time

            # Write final results if we have any
            if "merged_results" in locals() and merged_results:
                csv_file = os.path.join(
                    self.output_path, "results_scheduler.csv"
                )
                with open(csv_file, mode="w", newline="") as file:
                    writer = csv.DictWriter(
                        file,
                        fieldnames=[
                            "file",
                            "stage1_steps",
                            "stage1_time",
                            "stage1_energy",
                            "stage1_density",
                            "stage2_steps",
                            "stage2_time",
                            "stage2_energy",
                            "stage2_density",
                            "total_steps",
                            "total_time",
                        ],
                    )
                    writer.writeheader()
                    for row in merged_results:
                        try:
                            processed_row = {
                                "file": row["file"],
                                "stage1_steps": int(row["stage1_steps"]),
                                "stage1_time": float(row["stage1_time"]),
                                "stage1_energy": float(row["stage1_energy"]),
                                "stage1_density": float(row["stage1_density"]),
                                "stage2_steps": int(row["stage2_steps"]),
                                "stage2_time": float(row["stage2_time"]),
                                "stage2_energy": float(row["stage2_energy"]),
                                "stage2_density": float(row["stage2_density"]),
                                "total_steps": int(row["total_steps"]),
                                "total_time": float(row["total_time"]),
                            }
                            writer.writerow(processed_row)
                        except (KeyError, ValueError) as e:
                            logging.error(
                                f"Invalid data format in row {row}: {str(e)}"
                            )

            # Write summary
            summary_csv_file = os.path.join(
                self.output_path, "summary_scheduler.csv"
            )
            with open(summary_csv_file, mode="w", newline="") as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=["elapsed_time", "num_workers", "batch_size"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "elapsed_time": elapsed_time,
                        "num_workers": self.num_workers,
                        "batch_size": self.batch_size,
                    }
                )

            logging.info(f"Scheduler completed in {elapsed_time:.2f} seconds.")

    def run_debug(self):
        logging.info("Starting Scheduler in debug mode (sequential execution).")

        def worker_task(files, device, batch_size):
            worker = Worker(
                files, device, batch_size, self.max_steps, self.filter1
            )
            worker.run()

        for i in range(self.num_workers):
            files_for_worker = self.files[i :: self.num_workers]
            device = self.devices[i % len(self.devices)]
            logging.info(
                f"Running worker {i} with {len(files_for_worker)} files on device {device}."
            )
            worker_task(files_for_worker, device, self.batch_size)

        logging.info("All workers have completed their tasks in debug mode.")


class Worker:
    """
    Worker is single process that runs a batch of optimization tasks.
    """

    def __init__(
        self,
        files,
        device,
        batch_size,
        max_steps,
        filter1,
        filter2,
        optimizer1,
        optimizer2,
        skip_second_stage,
        scalar_pressure,
        compile_mode,
        profile,
        cueq,
        molecule_single,
        output_path,
        model,
        nproc,
    ):
        self.files = files
        self.device = device
        self.batch_size = batch_size
        self.max_steps = max_steps
        self.filter1 = filter1
        self.filter2 = filter2
        self.optimizer1 = optimizer1
        self.optimizer2 = optimizer2
        self.skip_second_stage = skip_second_stage  # Store skip_second_stage
        self.scalar_pressure = scalar_pressure
        self.compile_mode = compile_mode
        self.profile = profile
        self.cueq = cueq
        self.molecule_single = molecule_single
        self.output_path = (
            output_path
            if os.path.isabs(output_path)
            else os.path.abspath(output_path)
        )
        self.model = model
        self.nproc = nproc

        # Parse profiler options if provided
        self.use_profiler = False
        self.profiler_schedule_config = {
            "wait": 48,
            "warmup": 1,
            "active": 1,
            "repeat": 1,
        }
        self.profiler_log_dir = None

        if self.profile and self.profile != "False":
            self.use_profiler = True
            # Create directory for profiler output
            self.profiler_log_dir = os.path.join(self.output_path, "log")
            os.makedirs(self.profiler_log_dir, exist_ok=True)
            if self.profile != "True":
                try:
                    # Try to parse profile as a JSON string with schedule config
                    profile_config = json.loads(self.profile)
                    if isinstance(profile_config, dict):
                        for key in ["wait", "warmup", "active", "repeat"]:
                            if key in profile_config and isinstance(
                                profile_config[key], int
                            ):
                                self.profiler_schedule_config[key] = (
                                    profile_config[key]
                                )
                except json.JSONDecodeError:
                    logging.warning(
                        f"Could not parse profile config: {self.profile}, using defaults"
                    )

        # For monitor thread
        self.stop_event = threading.Event()

    def run(self):
        logging.info(
            f"Worker started on device {self.device} with {len(self.files)} files."
        )
        a2g = AtomsToGraphs(r_edges=False, r_pbc=True)
        # model = torch.load("/home/mazhaojia/.cache/mace/MACE-OFF23_small.model", map_location=self.device)
        # z_table = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])
        calculator = mace_off(model="small", device=self.device)

        results = []

        for batch_files in self._batch_files(self.files, self.batch_size):
            logging.info(f"Processing batch with {len(batch_files)} files.")
            start_time = time.perf_counter()

            atoms_list = []
            for file in batch_files:
                atoms = read(file)
                atoms_list.append(atoms)
            gbatch = data_list_collater(
                [a2g.convert(atoms) for atoms in atoms_list]
            )

            gbatch = gbatch.to(self.device)
            if self.filter1 == "UnitCellFilter":
                from batchopt.relaxation import OptimizableUnitCellBatch

                obatch = OptimizableUnitCellBatch(
                    gbatch,
                    trainer=calculator,
                    numpy=False,
                    scalar_pressure=self.scalar_pressure,
                )
            else:
                obatch = OptimizableBatch(
                    gbatch, trainer=calculator, numpy=False
                )

            # First optimization stage
            if self.optimizer1 == "LBFGS":
                batch_optimizer1 = LBFGS(
                    obatch, damping=1.0, alpha=70.0, maxstep=0.2
                )
            elif self.optimizer1 == "BFGS":
                batch_optimizer1 = BFGS(obatch, alpha=70.0, maxstep=0.2)
            elif self.optimizer1 == "BFGSLineSearch":
                batch_optimizer1 = BFGSLineSearch(obatch, device=self.device)
            elif self.optimizer1 == "BFGSFusedLS":
                batch_optimizer1 = BFGSFusedLS(obatch, device=self.device)
            else:
                raise ValueError(f"Unknown optimizer: {self.optimizer1}")

            start_time1 = time.perf_counter()
            batch_optimizer1.run(0.01, self.max_steps)
            end_time1 = time.perf_counter()
            elapsed_time1 = end_time1 - start_time1

            # Save intermediate results
            atoms_list = obatch.get_atoms_list()
            for atoms, file_path in zip(atoms_list, batch_files):
                file_name = file_path.split("/")[-1]
                output_file = os.path.join(
                    self.output_path,
                    "cif_result_press",
                    file_name.replace(".cif", "_press.cif"),
                )
                atoms.write(output_file)

            # Capture maximum force after first optimization stage
            max_force1 = obatch.get_max_forces(apply_constraint=True)

            steps1 = batch_optimizer1.nsteps

            if self.skip_second_stage:
                # If skipping second stage, set metrics to zero
                for file, force in zip(batch_files, max_force1):
                    results.append(
                        {
                            "file": file,
                            "stage1_time": elapsed_time1,
                            "stage1_steps": steps1,
                            "stage2_time": 0.0,
                            "stage2_steps": 0,
                            "total_time": elapsed_time1,
                            "total_steps": steps1,
                            "force1": force.item(),
                            "force2": 0.0,
                        }
                    )
                continue

            # Only proceed with second stage if not skipping
            # Reload intermediate structures for second stage
            atoms_list = []
            for file_path in batch_files:
                file_name = file_path.split("/")[-1]
                press_file = os.path.join(
                    self.output_path,
                    "cif_result_press",
                    file_name.replace(".cif", "_press.cif"),
                )
                atoms = read(press_file)
                atoms_list.append(atoms)

            # Rebuild batch from optimized structures
            gbatch = data_list_collater(
                [a2g.convert(atoms) for atoms in atoms_list]
            )
            gbatch = gbatch.to(self.device)

            # Second optimization stage
            if self.filter2 == "UnitCellFilter":
                obatch2 = OptimizableUnitCellBatch(
                    gbatch, trainer=calculator, numpy=False, scalar_pressure=0.0
                )
            else:
                obatch2 = OptimizableBatch(
                    gbatch, trainer=calculator, numpy=False
                )

            if self.optimizer2 == "LBFGS":
                batch_optimizer2 = LBFGS(
                    obatch2, damping=1.0, alpha=70.0, maxstep=0.2
                )
            elif self.optimizer2 == "BFGS":
                batch_optimizer2 = BFGS(obatch2, alpha=70.0, maxstep=0.2)
            elif self.optimizer2 == "BFGSLineSearch":
                batch_optimizer2 = BFGSLineSearch(obatch2, device=self.device)
            elif self.optimizer2 == "BFGSFusedLS":
                batch_optimizer2 = BFGSFusedLS(obatch2, device=self.device)
            else:
                raise ValueError(f"Unknown optimizer: {self.optimizer2}")
            start_time2 = time.perf_counter()
            batch_optimizer2.run(0.01, self.max_steps)
            end_time2 = time.perf_counter()
            elapsed_time2 = end_time2 - start_time2

            # Save final results
            atoms_list = obatch2.get_atoms_list()
            for atoms, file_path in zip(atoms_list, batch_files):
                file_name = file_path.split("/")[-1]
                output_file = os.path.join(
                    self.output_path,
                    "cif_result_final",
                    file_name.replace(".cif", "_opt.cif"),
                )
                atoms.write(output_file)

            # Capture maximum force after second optimization stage
            max_force2 = obatch2.get_max_forces(apply_constraint=True)

            steps2 = batch_optimizer2.nsteps

            for file, f1, f2 in zip(batch_files, max_force1, max_force2):
                results.append(
                    {
                        "file": file,
                        "stage1_time": elapsed_time1,
                        "stage1_steps": steps1,
                        "stage2_time": elapsed_time2,
                        "stage2_steps": steps2,
                        "total_time": elapsed_time1 + elapsed_time2,
                        "total_steps": steps1 + steps2,
                        "force1": f1.item(),
                        "force2": f2.item(),
                    }
                )

        return results

    def _batch_files(self, files, batch_size):
        for i in range(0, len(files), batch_size):
            yield files[i : i + batch_size]

    @staticmethod
    def _torch_memory_monitor(interval=2, device=None, stop_event=None):
        try:
            # explicitly CUDA initialization
            torch.cuda._lazy_init()
            while not stop_event.is_set():
                allocated = torch.cuda.memory_allocated(device=device)
                reserved = torch.cuda.memory_reserved(device=device)
                logging.info(
                    f"[torch] Allocated Memory: {allocated / 1024**2:.2f} MiB"
                )
                logging.info(
                    f"[torch] Reserved Memory: {reserved / 1024**2:.2f} MiB"
                )
                time.sleep(interval)
        except Exception as e:
            logging.error(f"Unexpected error when monitor memory: {str(e)}")

    def continuous_run(self):
        """
        Execute a continuous run of the batching optimization process.
        """
        logging.info("Starting continuous_run with two rounds of optimization.")

        # torch memory monitor api
        use_torch_memory_monitor = False
        if use_torch_memory_monitor:
            memory_monitor = threading.Thread(
                target=Worker._torch_memory_monitor,
                args=(2, self.device, self.stop_event),
            )
            memory_monitor.start()

        # First round of optimization
        try:
            logging.info("Starting first round of optimization.")
            results_round1, new_atoms_files = self.continuous_batching(
                atoms_path=self.files,
                result_path_prefix=os.path.join(
                    self.output_path, "cif_result_press/"
                ),
                fmax=0.01,
                maxstep=self.max_steps,
                use_filter=self.filter1,
                optimizer=self.optimizer1,
                scalar_pressure=self.scalar_pressure,
                dtype=torch.float64,
            )
            logging.info(
                f"Completed first round of optimization. Results: {len(results_round1)}"
            )
        except KeyboardInterrupt as e:
            if use_torch_memory_monitor:
                self.stop_event.set()
                memory_monitor.join()
            logging.error(f"Error during first round of optimization: {e}")
            raise
        except Exception as e:
            logging.error(f"Error during first round of optimization: {e}")
            raise

        if self.skip_second_stage:
            logging.info("Skipping second round of optimization.")
            return results_round1

        # Second round of optimization without pressure
        try:
            logging.info("Starting second round of optimization.")
            results_round2, _ = self.continuous_batching(
                atoms_path=new_atoms_files,
                result_path_prefix=os.path.join(
                    self.output_path, "cif_result_final/"
                ),
                fmax=0.01,
                maxstep=self.max_steps,
                # maxstep=3000,
                use_filter=self.filter2,
                optimizer=self.optimizer2,
                scalar_pressure=0.0,
                dtype=torch.float64,
            )
            logging.info(
                f"Completed second round of optimization. Results: {len(results_round2)}"
            )
        except KeyboardInterrupt as e:
            if use_torch_memory_monitor:
                self.stop_event.set()
                memory_monitor.join()
            logging.error(f"Error during second round of optimization: {e}")
            raise
        except Exception as e:
            logging.error(f"Error during second round of optimization: {e}")
            raise

        if use_torch_memory_monitor:
            self.stop_event.set()
            memory_monitor.join()

        return self._save_results_to_csv(results_round1, results_round2)

    def _save_results_to_csv(self, results_round1, results_round2):
        """Helper method to save results to CSV file and return the path."""
        combined_results = []
        results_map = {}

        # Process first round results
        for result in results_round1:
            file_name = result["file"]
            results_map[file_name] = {
                "file": file_name,
                "stage1_steps": result["steps"],
                "stage1_time": result["runtime"],
                "stage1_energy": result["energy"],
                "stage1_density": result["density"],
                "stage2_steps": 0,
                "stage2_time": 0.0,
                "stage2_energy": 0.0,
                "stage2_density": 0,
                "total_steps": result["steps"],
                "total_time": result["runtime"],
            }

        # Process second round results
        for result in results_round2:
            file_name = result["file"]
            if file_name in results_map:
                results_map[file_name].update(
                    {
                        "stage2_steps": result["steps"],
                        "stage2_time": result["runtime"],
                        "stage2_energy": result["energy"],
                        "stage2_density": result["density"],
                        "total_steps": results_map[file_name]["stage1_steps"]
                        + result["steps"],
                        "total_time": results_map[file_name]["stage1_time"]
                        + result["runtime"],
                    }
                )
            else:
                results_map[file_name] = {
                    "file": file_name,
                    "stage1_steps": 0,
                    "stage1_time": 0.0,
                    "stage1_energy": 0.0,
                    "stage1_density": 0,
                    "stage2_steps": result["steps"],
                    "stage2_time": result["runtime"],
                    "stage2_energy": result["energy"],
                    "stage2_density": result["density"],
                    "total_steps": result["steps"],
                    "total_time": result["runtime"],
                }

        # Convert map to list
        combined_results = list(results_map.values())

        logging.info(
            f"Combined results from both rounds. Total results: {len(combined_results)}"
        )

        worker_id = os.getpid()
        timestamp = int(time.time())
        csv_filename = f"worker_{worker_id}_{timestamp}.csv"
        csv_path = os.path.join(
            self.output_path, "worker_results", csv_filename
        )
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)

        with open(csv_path, mode="w", newline="") as csvfile:
            fieldnames = [
                "file",
                "stage1_steps",
                "stage1_time",
                "stage1_energy",
                "stage1_density",
                "stage2_steps",
                "stage2_time",
                "stage2_energy",
                "stage2_density",
                "total_steps",
                "total_time",
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for result in combined_results:
                writer.writerow(result)

        return csv_path

    def _get_density(self, crystal):
        # 计算总质量，ASE 中的 get_masses 方法返回一个数组，包含了所有原子的质量
        total_mass = sum(crystal.get_masses())  # 转换为克

        # 获取体积，ASE 的 get_volume 方法返回晶胞的体积，单位是 Å^3
        # 1 Å^3 = 1e-24 cm^3
        volume = crystal.get_volume()  # 转换为立方厘米

        # 计算密度，质量除以体积
        density = (
            total_mass / (volume * 10**-24) / (6.022140857 * 10**23)
        )  # 单位是 g/cm^3
        return density

    @staticmethod
    def select_factor(history: deque):
        # TODO: when history is mix of different size, the smaller `values` should be selected.
        boundaries = [0, 50, 100, 200, 400, 800]
        values = [0.4, 0.8, 0.9, 0.6, 0.5, 0.4]
        factor_result = []
        for graph_size in history:
            for i in range(len(boundaries) - 1):
                if boundaries[i] <= graph_size < boundaries[i + 1]:
                    factor_result.append(values[i])
                    break
        if len(factor_result) == 0:
            return 0.4
        else:
            return min(factor_result)

    def continuous_batching(
        self,
        atoms_path,
        result_path_prefix,
        fmax,
        maxstep,
        use_filter,
        optimizer,
        scalar_pressure,
        dtype=torch.float64,
    ):
        """
        Performs continuous batched optimization of atomic structures.

        This method implements a continuous batching strategy for optimizing multiple atomic structures,
        where converged structures are replaced with new ones to maintain batch efficiency.

        Parameters
        ----------
        atoms_path : list
            List of file paths to atomic structure files to be optimized
        result_path_prefix : str
            Prefix for output file paths where optimized structures will be saved
        fmax : float, optional
            Maximum force criterion for convergence, by default 0.01
        maxstep : int, optional
            Maximum number of optimization steps per batch, by default 3000
        use_filter : str, optional
            Filter to be used for optimization, by default "UnitCellFilter"
        optimizer : str, optional
            Optimizer to be used for optimization, by default "LBFGS"
        scalar_pressure : float, optional
            Scalar pressure to be applied, by default 0.0

        Returns
        -------
        None
            The optimized structures are saved to disk

        Notes
        -----
        The method:
        - Processes structures in batches of predefined size
        - Uses MACE neural network potential for energy/force calculations
        - Employs LBFGS optimization with unit cell relaxation
        - Dynamically replaces converged structures with new ones in the batch
        - Tracks convergence and optimization steps for each structure
        """
        # Load saved structures
        result = []
        optimized_atoms_paths = []

        json_dir = result_path_prefix.replace("cif", "json")

        remove_list = []
        # TODO: Why we read all CIF here?
        for pre_cif in atoms_path:
            cif_path = os.path.join(result_path_prefix, pre_cif.split("/")[-1])
            json_path = os.path.join(
                json_dir, pre_cif.split("/")[-1].replace(".cif", ".json")
            )
            if (
                os.path.exists(cif_path)
                and os.path.exists(json_path)
                and os.path.getsize(cif_path) > 0
                and os.path.getsize(json_path) > 0
            ):
                with open(json_path, "r") as f:
                    result_data = json.load(f)
                    result.append(result_data)
                    optimized_atoms_paths.append(cif_path)
                    remove_list.append(pre_cif)
                    logging.info(f"File {cif_path} already exists, loaded.")
            # else:
            #     try:
            #         read(pre_cif)
            #     except Exception as e:
            #         logging.info(f"Failed to read {pre_cif}: {e}")
            #         remove_list.append(pre_cif)
        for i in remove_list:
            atoms_path.remove(i)

        if self.batch_size > 0:
            # Initialize variables
            room_in_batch = self.batch_size
            indices_to_process = 0
            cur_batch_path = atoms_path[
                indices_to_process : indices_to_process + room_in_batch
            ]
            if len(cur_batch_path) == 0:
                logging.info("No structures to process.")
                return result, optimized_atoms_paths
            room_in_batch -= len(cur_batch_path)
            indices_to_process += len(cur_batch_path)
            cur_atoms_list = [read(path) for path in cur_batch_path]
            a2g = AtomsToGraphs(r_edges=False, r_pbc=True)
            gbatch = data_list_collater(
                [a2g.convert(read(path)) for path in cur_batch_path]
            )
        else:
            # Set Maximum Number of atoms per batch
            history = deque(maxlen=10)
            history.append(1000)
            max_bnatoms = 24080
            safe_factor = self.select_factor(history)

            indices_to_process = 0
            bnatoms = 0
            cur_batch_path = []
            graphs_list = []
            a2g = AtomsToGraphs(r_edges=False, r_pbc=True)

            while indices_to_process < len(atoms_path):
                graph_natoms = count_atoms_cif(atoms_path[indices_to_process])
                if (
                    bnatoms + graph_natoms
                    > max_bnatoms * safe_factor // self.nproc
                ):
                    break
                graph = a2g.convert(read(atoms_path[indices_to_process]))
                bnatoms += graph_natoms
                cur_batch_path.append(atoms_path[indices_to_process])
                graphs_list.append(graph)
                indices_to_process += 1
                history.append(graph_natoms)
            safe_factor = self.select_factor(history)
            if len(graphs_list) == 0:
                logging.info("No structures to process.")
                return result, optimized_atoms_paths
            gbatch = data_list_collater(graphs_list)
            logging.info(f"current batch size: {len(cur_batch_path)}")

            total_natoms = sum([graph.natoms for graph in graphs_list])
            logging.info(f"total_natoms: {total_natoms}")

        gbatch = gbatch.to(self.device)
        batch_optimizer = None

        # Initial calculator
        if self.model == "mace":
            if dtype == torch.float32:
                calculator = mace_off(
                    model="small",
                    device=self.device,
                    enable_cueq=self.cueq,
                    default_dtype="float32",
                )
            else:
                calculator = mace_off(
                    model="small", device=self.device, enable_cueq=self.cueq
                )
        elif self.model == "chgnet":
            calculator = CHGNetCalculator(
                use_device=self.device, enable_cueq=self.cueq
            )
        elif self.model == "sevennet":
            # calculator = SevenNetCalculator(device=self.device, enable_cueq=self.cueq)
            calculator = SevenNetD3Calculator(
                device=self.device,
                enable_cueq=self.cueq,
                batch_size=self.batch_size,
            )
            # calculator = SevenNetCalculator('7net-mf-ompa', modal='mpa', device=self.device)
        # calculator = MACECalculator(model_paths="/home/mazhaojia/.cache/mace/MACE-OFF23_small.model", device=self.device, compile_mode=self.compile_mode)
        if use_filter == "UnitCellFilter":
            obatch = OptimizableUnitCellBatch(
                gbatch,
                trainer=calculator,
                numpy=False,
                scalar_pressure=scalar_pressure,
            )
        else:
            obatch = OptimizableBatch(gbatch, trainer=calculator, numpy=False)

        orig_cells = obatch.orig_cells.clone()

        converged_atoms_count = 0
        converge_indices = []
        all_indices = []
        cur_batch_steps = [0] * len(cur_batch_path)
        cur_batch_times = [time.perf_counter()] * len(
            cur_batch_path
        )  # Track start times

        while converged_atoms_count < len(atoms_path):
            # Update batch
            if len(all_indices) > 0:
                if self.batch_size > 0:
                    room_in_batch += len(all_indices)
                    new_batch_path = atoms_path[
                        indices_to_process : indices_to_process + room_in_batch
                    ]
                    logging.info(f"new_batch_path: {new_batch_path}")
                    room_in_batch -= len(new_batch_path)
                    indices_to_process += len(new_batch_path)

                    optimized_atoms_new = []
                    cur_batch_path_new = []
                    cur_batch_steps_new = []
                    cur_batch_times_new = []
                    orig_cells_new = torch.zeros(
                        [self.batch_size - room_in_batch, 3, 3],
                        device=self.device,
                    )
                    cell_offset = 0

                    restart_indices = []
                    old_batch_indices = obatch.batch_indices
                    for i in range(len(optimized_atoms)):
                        if i in all_indices:
                            continue
                        else:
                            restart_indices.append(i)
                        optimized_atoms_new.append(optimized_atoms[i])
                        cur_batch_path_new.append(cur_batch_path[i])
                        cur_batch_steps_new.append(cur_batch_steps[i])
                        cur_batch_times_new.append(cur_batch_times[i])

                        orig_cells_new[cell_offset] = orig_cells[i]
                        cell_offset += 1

                    for new_path in new_batch_path:
                        optimized_atoms_new.append(read(new_path))
                        cur_batch_path_new.append(new_path)
                        cur_batch_steps_new.append(0)
                        cur_batch_times_new.append(time.perf_counter())

                    # Update the batch with new structures
                    optimized_atoms = optimized_atoms_new
                    cur_batch_path = cur_batch_path_new
                    cur_batch_steps = cur_batch_steps_new
                    cur_batch_times = cur_batch_times_new
                else:
                    bnatoms = 0
                    optimized_atoms_new = []
                    cur_batch_path_new = []
                    cur_batch_steps_new = []
                    cur_batch_times_new = []

                    restart_indices = []
                    old_batch_indices = obatch.batch_indices
                    for i in range(len(optimized_atoms)):
                        if i in all_indices:
                            continue
                        restart_indices.append(i)
                        optimized_atoms_new.append(optimized_atoms[i])
                        cur_batch_path_new.append(cur_batch_path[i])
                        cur_batch_steps_new.append(cur_batch_steps[i])
                        cur_batch_times_new.append(cur_batch_times[i])
                        bnatoms += a2g.convert(read(cur_batch_path[i])).natoms

                    while indices_to_process < len(atoms_path):
                        new_path = atoms_path[indices_to_process]
                        graph_natoms = count_atoms_cif(new_path)
                        if (
                            bnatoms + graph_natoms
                            > max_bnatoms * safe_factor // self.nproc
                        ):
                            break
                        bnatoms += graph_natoms
                        optimized_atoms_new.append(read(new_path))
                        cur_batch_path_new.append(new_path)
                        cur_batch_steps_new.append(0)
                        cur_batch_times_new.append(time.perf_counter())
                        indices_to_process += 1
                        history.append(graph_natoms)
                    safe_factor = self.select_factor(history)

                    orig_cells_new = torch.zeros(
                        [len(optimized_atoms_new), 3, 3], device=self.device
                    )
                    cell_offset = 0
                    for i in range(len(optimized_atoms)):
                        if i in all_indices:
                            continue
                        orig_cells_new[cell_offset] = orig_cells[i]
                        cell_offset += 1

                    # Update the batch with new structures
                    optimized_atoms = optimized_atoms_new
                    cur_batch_path = cur_batch_path_new
                    cur_batch_steps = cur_batch_steps_new
                    cur_batch_times = cur_batch_times_new

                    logging.info(f"current batch size: {len(optimized_atoms)}")

                graphs_list = [a2g.convert(atoms) for atoms in optimized_atoms]
                total_natoms = sum([graph.natoms for graph in graphs_list])
                logging.info(f"total_natoms: {total_natoms}")
                logging.info(f"cur_batch_path to processing: {cur_batch_path}")
                gbatch = data_list_collater(graphs_list)
                gbatch = gbatch.to(self.device)
                if self.model == "sevennet":
                    # calculator = SevenNetCalculator('7net-mf-ompa', modal='mpa', device=self.device)
                    calculator = SevenNetD3Calculator(
                        device=self.device,
                        enable_cueq=self.cueq,
                        batch_size=self.batch_size,
                    )
                if use_filter == "UnitCellFilter":
                    obatch = OptimizableUnitCellBatch(
                        gbatch,
                        trainer=calculator,
                        numpy=False,
                        scalar_pressure=scalar_pressure,
                    )
                else:
                    obatch = OptimizableBatch(
                        gbatch, trainer=calculator, numpy=False
                    )
                for i in range(cell_offset):
                    obatch.orig_cells[i] = orig_cells_new[i]
                orig_cells = obatch.orig_cells.clone()

            # Optimize the current batch
            if optimizer == "LBFGS":
                batch_optimizer = LBFGS(
                    obatch,
                    damping=1.0,
                    alpha=70.0,
                    maxstep=0.2,
                    early_stop=True,
                )
            elif optimizer == "BFGS":
                if len(all_indices) > 0:
                    logging.info(f"Restarting with indices: {restart_indices}")
                    batch_optimizer.optimizable = obatch
                else:
                    batch_optimizer = BFGS(
                        obatch, alpha=70.0, maxstep=0.2, early_stop=True
                    )
            elif optimizer == "BFGSLineSearch":
                batch_optimizer = BFGSLineSearch(
                    obatch,
                    device=self.device,
                    early_stop=True,
                    use_profiler=self.use_profiler,
                    profiler_log_dir=self.profiler_log_dir,
                    profiler_schedule_config=self.profiler_schedule_config,
                )
            elif optimizer == "BFGSFusedLS":
                if len(all_indices) > 0:
                    logging.info(f"Restarting with indices: {restart_indices}")
                    batch_optimizer.optimizable = obatch
                else:
                    batch_optimizer = BFGSFusedLS(
                        obatch,
                        device=self.device,
                        early_stop=True,
                        use_profiler=self.use_profiler,
                        profiler_log_dir=self.profiler_log_dir,
                        profiler_schedule_config=self.profiler_schedule_config,
                    )
            else:
                raise ValueError(f"Unknown optimizer: {optimizer}")

            # 动态计算剩余可用步数（基于当前批次最大已执行步数）
            current_max_steps = max(cur_batch_steps) if cur_batch_steps else 0
            remaining_steps = max(
                maxstep - current_max_steps, 1
            )  # 保证至少运行1步

            # 执行优化并获取收敛的索引
            if (optimizer == "BFGSFusedLS" or optimizer == "BFGS") and len(
                all_indices
            ) > 0:
                converge_indices = batch_optimizer.run(
                    fmax,
                    remaining_steps,
                    is_restart_earlystop=True,
                    restart_indices=restart_indices,
                    old_batch_indices=old_batch_indices,
                )
            else:
                converge_indices = batch_optimizer.run(fmax, remaining_steps)

            # Print energies of all structures
            # logging.info(f"Final energies of all structures: {batch_optimizer.energies}")
            energies_list = (
                batch_optimizer.optimizable.get_potential_energies().tolist()
            )
            logging.info(f"Final energies of all structures: {energies_list}")

            # 更新所有结构的累计步数
            cur_batch_steps = [
                steps + batch_optimizer.nsteps for steps in cur_batch_steps
            ]

            # 找出超过最大步数的结构索引
            over_maxstep_indices = [
                i
                for i, steps in enumerate(cur_batch_steps)
                if steps >= maxstep - 1
            ]

            # 合并收敛和超限的索引（去重）
            all_indices = list(set(converge_indices + over_maxstep_indices))

            # Get optimized atoms
            optimized_atoms = obatch.get_atoms_list()
            converged_atoms_count += len(all_indices)

            end_time = time.perf_counter()
            # 处理所有需要退出的结构（包括收敛和超限）
            for idx in all_indices:
                runtime = end_time - cur_batch_times[idx]

                energy_per_mol = (
                    energies_list[idx]
                    / (
                        len(optimized_atoms[idx].get_atomic_numbers())
                        / self.molecule_single
                    )
                    * 96.485
                )
                density = self._get_density(optimized_atoms[idx])

                # Save results
                result_data = {
                    "file": cur_batch_path[idx].split("/")[-1].split(".")[0],
                    "steps": cur_batch_steps[idx],
                    "runtime": runtime,
                    "energy": energy_per_mol,
                    "density": density,
                }
                result.append(result_data)

                # Save optimized structure
                # converged_atoms_path = os.path.join(result_path_prefix, cur_batch_path[idx].split('/')[-1].replace('.cif', '.traj'))
                converged_atoms_path = os.path.join(
                    result_path_prefix, cur_batch_path[idx].split("/")[-1]
                )
                optimized_atoms[idx].write(converged_atoms_path)
                optimized_atoms_paths.append(converged_atoms_path)

                # write a json file to store reslt_data
                os.makedirs(json_dir, exist_ok=True)
                # json_path = os.path.join(json_dir, cur_batch_path[idx].split('/')[-1]+'.json')
                json_path = os.path.join(
                    json_dir,
                    cur_batch_path[idx].split("/")[-1].replace(".cif", ".json"),
                )
                with open(json_path, "w") as f:
                    json.dump(result_data, f)

            logging.info(f"cur_batch_path: {cur_batch_path}")
            logging.info(f"cur_batch_steps: {cur_batch_steps}")
            logging.info(f"all_indices: {all_indices}")
            logging.info(f"length of optimized_atoms: {len(optimized_atoms)}")

        return result, optimized_atoms_paths
