from basic_function import format_parser
from basic_function import packaged_function
from basic_function import conformer_search
import time
import argparse
import os
import itertools


if __name__ == '__main__':

    time_start = time.time()

    # initiate configuration
    ##############################################################################################
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default="./", help='Path to process')
    parser.add_argument('--smiles', type=str, required=True, help='SMILES string of the molecules, split by . if multiple molecules are used')
    parser.add_argument('--generate_conformers', type=int, default=20, help='Number of conformers to generate. When it is <=0, only load existing conformers to generate structures')
    parser.add_argument('--use_conformers', type=int, default=4, help='Number of conformers used to generate structure. When it is <=0, no structure generation would be done')
    parser.add_argument('--molecule_num_in_cell', type=str, nargs='+', default=['1'], help='number of molecules in a unit cell, split by comma for multiple molecules, and split by space for multiple packings')
    parser.add_argument('--num_generation', type=int, nargs='+', default=[100], help='number of structures to generate, split by space for multiple packings')
    parser.add_argument('--space_group_list', type=str, nargs='+', default=["2,14"], help='Space group list for structure generation, spilt by comma to add mutiple groups, split by space for multiple packings')
    parser.add_argument('--add_name', type=str, nargs='+', default=["CRYSTAL"], help='Add name for the generated structures, split by space for multiple packings')
    parser.add_argument('--max_workers', type=int, default=8, help='Maximum number of workers for parallel processing')
    parser.add_argument('--mode', type=str, default=8, choices=['all', 'conformer_only', 'structure_only'], help='choose the jobs to do')
    args = parser.parse_args()

    target_folder = args.path
    smiles_list = args.smiles.split('.')
    generate_conformers = args.generate_conformers
    use_conformers = args.use_conformers
    molecule_num_in_cell = [list(map(int, num.split(','))) for num in args.molecule_num_in_cell]
    num_generation = args.num_generation
    space_group_list = [list(map(int, group.split(','))) for group in args.space_group_list]
    add_name = args.add_name
    max_workers = args.max_workers
    mode = args.mode

    num_molecules = len(smiles_list)
    num_packings = max(len(molecule_num_in_cell), len(space_group_list))

    for i in range(len(molecule_num_in_cell)):
        if len(molecule_num_in_cell[i]) < num_molecules:
            molecule_num_in_cell[i].extend([1] * (num_molecules - len(molecule_num_in_cell[i])))
        elif len(molecule_num_in_cell[i]) > num_molecules:
            molecule_num_in_cell[i] = molecule_num_in_cell[i][:num_molecules]

    while len(molecule_num_in_cell) < num_packings:
        molecule_num_in_cell.append(molecule_num_in_cell[-1])

    while len(space_group_list) < num_packings:
        space_group_list.append(space_group_list[-1])

    while len(add_name) < num_packings:
        add_name.append(add_name[-1])

    while len(num_generation) < num_packings:
        num_generation.append(num_generation[-1])

    
    # step1: conformer search
    ##############################################################################################
    molecule_data = []
    for i in range(num_molecules):
        molecule_folder = os.path.join(target_folder, f"molecule_{i+1}")
        molecule_data.append([])
        if generate_conformers > 0 and mode != "structure_only":
            conformer_search.conformer_search(smiles_list[i], molecule_folder, num_conformers=generate_conformers, max_attempts=10000, rms_thresh=0.1)
            with open(os.path.join(molecule_folder, "info.txt"), "w") as smiles_file:
                smiles_file.write(f"SMILES: {smiles_list[i]}")
        file_num = len(os.listdir(os.path.join(molecule_folder, "conformers")))
        cnt = 0
        for j in range(file_num):
            if cnt >= use_conformers:
                break
            temp_path = os.path.join(molecule_folder, "conformers", f"conformer_{j}.xyz")
            if not os.path.exists(temp_path):
                break
            molecule_data[i].append(format_parser.read_xyz_file(temp_path))
            cnt += 1
            
        if len(molecule_data[i]) <= 0:
            print(f"No conformer loaded for molecule_{i+1}. Check configurations!")
            break

    idx_data = [list(range(len(item))) for item in molecule_data]
    combinations = list(itertools.product(*idx_data))


    # step2: structure generation
    ##############################################################################################
    if mode != "conformer_only":
        for i in range(num_packings):
            for combination in combinations:
                molecule_list = []
                for j in range(num_molecules):
                    for cnt in range(molecule_num_in_cell[i][j]):
                        molecule_list.append(molecule_data[j][combination[j]])
                c_name = "".join(map(str, combination))
                packaged_function.CSP_generater_parallel(molecule_list, target_folder, need_structure=num_generation[i], space_group_list=space_group_list[i],add_name=f"{add_name[i]}_C{c_name}", max_workers=max_workers,start_seed=1)

    time_end=time.time()
    print('time cost',time_end-time_start,'s')
