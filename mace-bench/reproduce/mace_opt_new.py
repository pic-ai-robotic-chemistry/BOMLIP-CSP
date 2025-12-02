"""
Copyright (c) 2025 Ma Zhaojia

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import sys
# sys.path.append('/home/jiangj1group/zcxzcx1/volatile/mace')
from mace.calculators import mace_off, mace_mp
from ase.io import read, write
from ase.optimize import BFGS,LBFGS,FIRE,GPMin,MDMin, QuasiNewton
from ase.filters import UnitCellFilter, ExpCellFilter, FrechetCellFilter
import re
import io
from contextlib import redirect_stdout
import os
import pandas as pd
from joblib import Parallel, delayed
import json
import torch
import numpy as np
import random
import argparse
import time
import pathlib
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', force=True)
#####################################################################
os.environ['PYTHONHASHSEED'] = '1'
torch.manual_seed(1)
np.random.seed(1)
random.seed(1)
torch.cuda.manual_seed(1)
torch.cuda.manual_seed_all(1)
#####################################################################
# n_jobs=32
# # n_jobs=2
# path = './'
# molecule_single = 64
# target_folder = "/data_raw/"
#####################################################################

def calculate_density(crystal):
    # 计算总质量，ASE 中的 get_masses 方法返回一个数组，包含了所有原子的质量
    total_mass = sum(crystal.get_masses())  # 转换为克

    # 获取体积，ASE 的 get_volume 方法返回晶胞的体积，单位是 Å^3
    # 1 Å^3 = 1e-24 cm^3
    volume = crystal.get_volume()  # 转换为立方厘米

    # 计算密度，质量除以体积
    density = total_mass / (volume*10**-24)/(6.022140857*10**23)  # 单位是 g/cm^3
    return density

def run_calculation_one(path,file,target_folder,molecule_single,idx):
    # os.environ['OMP_NUM_THREADS'] = '1'
    # os.environ['MKL_NUM_THREADS'] = '1'
    # os.environ['OPENBLAS_NUM_THREADS'] = '1'
    if reproduce:
        print("Reproducing deterministic results.")
        torch.use_deterministic_algorithms(True)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        np.set_printoptions(precision=17, suppress=False)
        torch.set_printoptions(precision=17, sci_mode=False, linewidth=200)
    if multithread and (not reproduce):
        print("Using OMP and MKL multithreads will introduce non-deterministic results.")
    else: 
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'
        os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ["CUDA_VISIBLE_DEVICES"]=str((idx%n_gpus)+gpu_offset)

    with io.StringIO() as buf, redirect_stdout(buf):
        crystal = read(path+target_folder+file)
        if molecule_single < 0:
            molecule_single = int(file.split('_')[-1].split('.')[0])
        molecule_count = len(crystal.get_atomic_numbers())/molecule_single
        calc = mace_off(model=model_path,dispersion=True, device='cuda', enable_cueq=cueq)
        crystal.calc = calc
        if filter1 == "UnitCellFilter":
            sf = UnitCellFilter(crystal,scalar_pressure=0.0006)
        elif filter1 == "FrechetCellFilter":
            sf = FrechetCellFilter(crystal,scalar_pressure=0.0006)
        else:
            raise ValueError(f"Unrecognized filter type '{filter1}'. "
                            "Supported types are 'UnitCellFilter' and 'FrechetCellFilter'.")
        if optimizer_type1 == "BFGS":
            if use_cuda_eigh:
                optimizer = BFGS(sf, use_cuda_eigh=True)
            else:
                optimizer = BFGS(sf)
        elif optimizer_type1 == "LBFGS":
            optimizer = LBFGS(sf)
        elif optimizer_type1 == "QuasiNewton":
            optimizer = QuasiNewton(sf)
        else:
            raise ValueError(f"Unrecognized optimizer type '{optimizer_type1}'. "
                            "Supported types are 'BFGS' and 'LBFGS'.")

        if use_nsys or use_torch_profiler : # warmup for profiling
            optimizer.run(fmax=0.01,steps=100)
        if use_torch_profiler:
            profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA
                ],
                # schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
                on_trace_ready=torch.profiler.tensorboard_trace_handler('./log'),
                with_stack=True
            )
            profiler.start()

        start_time1 = time.time()
        optimizer.run(fmax=0.01,steps=max_steps)
        end_time1 = time.time()

        if use_torch_profiler:
            profiler.stop()

        crystal.write(path+'cif_result_press/'+file[:-4]+"_press.cif")
        output_1 = buf.getvalue()
        # step_used_1 = float(re.split("\\s+", output_1.split('\n')[-2])[1][:])
        step_used_1 = optimizer.nsteps
        if use_nsys or use_torch_profiler : 
            step_used_1 = step_used_1 - 100
        total_time1 = end_time1 - start_time1
        avg_time1 = total_time1 / step_used_1 if step_used_1 != 0 else 0
        
        crystal = read(path+'cif_result_press/'+file[:-4]+"_press.cif")
        crystal.calc = calc
        if filter2 == "UnitCellFilter":
            sf = UnitCellFilter(crystal)
        elif filter2 == "FrechetCellFilter":
            sf = FrechetCellFilter(crystal)
        else:
            raise ValueError(f"Unrecognized filter type '{filter2}'. "
                            "Supported types are 'UnitCellFilter' and 'FrechetCellFilter'.")
        if optimizer_type2 == "BFGS":
            if use_cuda_eigh:
                optimizer = BFGS(sf, use_cuda_eigh=True)
            else:
                optimizer = BFGS(sf)
        elif optimizer_type2 == "LBFGS":
            optimizer = LBFGS(sf)
        elif optimizer_type2 == "QuasiNewton":
            optimizer = QuasiNewton(sf)
        else:
            raise ValueError(f"Unrecognized optimizer type '{optimizer_type2}'. "
                            "Supported types are 'BFGS' and 'LBFGS'.")
        if use_torch_profiler:
            profiler = torch.profiler.profile(
                activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA
                ],
                # schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
                on_trace_ready=torch.profiler.tensorboard_trace_handler('./log'),
                with_stack=True
            )
            profiler.start()

        start_time2 = time.time()
        optimizer.run(fmax=0.01,steps=max_steps)
        end_time2 = time.time()

        if use_torch_profiler:
            profiler.stop()

        density = calculate_density(crystal)
        crystal.write(path+'cif_result_final/'+file[:-4]+"_opt.cif")
        output_2 = buf.getvalue()
        energy = float(re.split("\\s+", output_2.split('\n')[-2])[3][:])
        # step_used_2 = float(re.split("\\s+", output_2.split('\n')[-2])[1][:])
        step_used_2 = optimizer.nsteps
        energy_per_mol = energy / molecule_count * 96.485
        total_time2 = end_time2 - start_time2
        avg_time2 = total_time2 / step_used_2 if step_used_2 != 0 else 0

        new_row = {
            'name': file[:-4], 'density': density, 'energy_kj': energy_per_mol, 
            'step_used_1': step_used_1, 'step_used_2': step_used_2,
            'total_time1_s': total_time1, 'avg_time1_s': avg_time1,
            'total_time2_s': total_time2, 'avg_time2_s': avg_time2
        }

    print(f'output_2: {output_2}')
    with open(path+'json_result/'+file[:-4]+".json", 'w') as json_file:
        json.dump(new_row, json_file, indent=4)                                                                   
    return new_row


def already_have_calculation_one(path, file, target_folder, molecule_single, idx):
    logging.info(f"reading on structure {file}")
    print(f"reading on structure {file}")
    with open(path + 'json_result/' + file[:-4] + ".json", 'r') as file:
        old_row = json.load(file)
    return old_row

def run():
    df = pd.DataFrame(columns=['name', 'density', 'energy_kj', 'step_used_1', 'step_used_2', 'total_time1_s', 'avg_time1_s', 'total_time2_s', 'avg_time2_s'])
    for root, dirs, files in os.walk(path + target_folder):
        old_row = Parallel(n_jobs=n_jobs)(
            delayed(already_have_calculation_one)(path, file, target_folder, molecule_single, idx) for idx, file in
            enumerate(files) if os.path.exists(path + 'json_result/' + file[:-4] + ".json"))

        filtered_files = [file for file in files if not os.path.exists(path + 'json_result/' + file[:-4] + ".json")]
        new_row = Parallel(n_jobs=n_jobs)(
            delayed(run_calculation_one)(path, file, target_folder, molecule_single, idx) for idx, file in
            enumerate(filtered_files))
        # show the length of new_row
        print(f'new_row length: {len(new_row)}')
        print(f'root: {root}\ndirs: {dirs}\nfiles: {files}')
        for row in new_row:
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True, axis=0)
        for row in old_row:
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True, axis=0)
        df.to_csv(path + '/result.csv')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run parallel calculations on molecular crystals.")
    parser.add_argument("--n_jobs", type=int, default=32, help="Number of parallel jobs to run (default: 32)")
    parser.add_argument("--target_folder", type=str, required=True, help="Path to the target folder containing input files")
    parser.add_argument("--path", type=str, default='./', help="Base path for the project (default: './')")
    parser.add_argument("--molecule_single", type=int, default=-1, help="Number of atoms per molecule (default: 64)")
    parser.add_argument("--n_gpus", type=int, default=2, help="Number of GPUs to use (default: 2)")
    parser.add_argument("--cueq", action='store_true', help="Whether to use cuEquivariance Library (default: False)")
    parser.add_argument("--max_steps", type=int, default=3000, help="Number of max steps to run the optimization (default: 3000)")
    parser.add_argument("--use_torch_profiler", action='store_true', help="Whether to use torch profiler (default: False)")
    parser.add_argument("--use_nsys", action='store_true', help="Whether to use nsys profiler (default: False)")
    parser.add_argument("--model", type=str, default="small", help="Model to use for the calculation (default: 'small')")
    parser.add_argument("--optimizer", type=str, default="BFGS", help="Optimizer to use for the calculation (default: 'BFGS')")
    parser.add_argument("--use_cuda_eigh", action='store_true', help="Whether to use CUDA for eigh (default: False)")
    parser.add_argument("--gpu_offset", type=int, default=0, help="GPU offset to use for the calculation (default: 0)")
    parser.add_argument("--multithread", action='store_true', help="Whether to use multithread (default: False)")
    parser.add_argument("--reproduce", action='store_true', help="Whether to reproduce deterministic results (default: False)")
    parser.add_argument("--filter1", type=str, default="UnitCellFilter", help="1st filter to use for the calculation (default: 'UnitCellFilter')")
    parser.add_argument("--filter2", type=str, default="UnitCellFilter", help="2nd filter to use for the calculation (default: 'UnitCellFilter')")
    parser.add_argument("--optimizer1", type=str, default="BFGS", help="1st optimizer to use for the calculation (default: 'BFGS')")
    parser.add_argument("--optimizer2", type=str, default="BFGS", help="2nd optimizer to use for the calculation (default: 'BFGS')")

    args = parser.parse_args()

    n_jobs = args.n_jobs
    target_folder = args.target_folder
    path = args.path
    molecule_single = args.molecule_single
    n_gpus = args.n_gpus
    cueq = args.cueq
    max_steps = args.max_steps
    use_torch_profiler = args.use_torch_profiler
    use_nsys = args.use_nsys
    model_path = args.model
    optimizer_type = args.optimizer
    use_cuda_eigh = args.use_cuda_eigh
    gpu_offset = args.gpu_offset
    multithread = args.multithread
    reproduce = args.reproduce
    filter1 = args.filter1
    filter2 = args.filter2
    optimizer_type1 = args.optimizer1
    optimizer_type2 = args.optimizer2


    try:
        os.mkdir("./cif_result_press")
        os.mkdir("./cif_result_final")
    except:
        pass
    try:
        os.mkdir("./json_result")
    except:
        pass
        
    start_time_all = time.time()


    iter = 0
    while iter < 100:
        iter += 1
        try:
            run()
            break
        except Exception as e:
            print(f"Error occurred: {e}")
            print("Retrying...")
            time.sleep(10)

    end_time_all = time.time()
    total_time_all = end_time_all - start_time_all
    print('dataset,total_time_all_s,attempts')
    print(f"{pathlib.Path(target_folder).name},{total_time_all},{iter}")
    with open(path + 'timing.csv', 'w') as f:
        f.write('dataset,total_time_all_s,attempts\n')
        f.write(f"{pathlib.Path(target_folder).name},{total_time_all},{iter}\n")