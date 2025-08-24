#!/bin/bash
TOP_DIR=$(pwd)
TAR_DIR="${TOP_DIR}/test"

mkdir -p "${TAR_DIR}"
cd ${TAR_DIR}

# generate structures
python "${TOP_DIR}/main.py" --path ${TAR_DIR} --smiles "OC(=O)c1cc(O)c(O)c(O)c1.O" \
    --molecule_num_in_cell 1,1 --space_group_list 13,14 --add_name KONTIQ --max_workers 16\
     --num_generation 100 --generate_conformers 20 --use_conformers 4 > generate.log 2>&1

# opt structures using mace, --batch_size 0 means auto batch size only for mace
mkdir -p "${TAR_DIR}/mace_opt"
cd "${TAR_DIR}/mace_opt"
python "${TOP_DIR}/mace-bench/scripts/mace_opt_batch.py" --target_folder "${TAR_DIR}/structures" \
    --molecule_single 21 --gpu_offset 0 --n_gpus 8 --num_workers 80 --batch_size 0 \
    --max_steps 3000 --filter1 UnitCellFilter --filter2 UnitCellFilter \
    --optimizer1 BFGSFusedLS --optimizer2 BFGS --num_threads 1 --cueq true \
    --use_ordered_files true --model mace > opt.log 2>&1

# opt structures using 7net
# mkdir -p "${TAR_DIR}/7net_opt"
# cd "${TAR_DIR}/7net_opt"
# python "${TOP_DIR}/mace-bench/scripts/mace_opt_batch.py" --target_folder "${TAR_DIR}/structures" \
#     --molecule_single 21 --gpu_offset 0 --n_gpus 8 --num_workers 48 --batch_size 2 \
#     --max_steps 3000 --filter1 UnitCellFilter --filter2 UnitCellFilter \
#     --optimizer1 BFGSFusedLS --optimizer2 BFGS --num_threads 2 --cueq true \
#     --use_ordered_files true --model sevennet > opt.log 2>&1

# Postprocess the opt structures
python "${TOP_DIR}/post_process/clean_table.py"
## Make sure you have installed csd-python-api in current env before execuing following commands
# conda activate ccdc
# python "${TOP_DIR}/post_process/check_match.py" --workers 80 --timeout 20 --ref_path "${TAR_DIR}/refs"
# python "${TOP_DIR}/post_process/duplicate_remove.py" --workers 80
