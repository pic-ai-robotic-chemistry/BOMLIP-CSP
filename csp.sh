
TOP_DIR=$(pwd)
TAR_DIR="${TOP_DIR}/test"

mkdir -p "${TAR_DIR}"
cd ${TAR_DIR}

# conformer search and structure generation
# change --mode to conformer_only or structure_only to seperate the process. 
python "${TOP_DIR}/main.py" --path ${TAR_DIR} --smiles "C1CC2=COC=C12" \
    --molecule_num_in_cell 1 --space_group_list 14,61 --add_name XULDUD --max_workers 16 \
     --num_generation 100 --generate_conformers 10 --use_conformers 4 --mode all > generate.log 2>&1

# python "${TOP_DIR}/main.py" --path ${TAR_DIR} --smiles "C1CC2=COC=C12" \
#      --num_generation 100 --generate_conformers 10 --mode conformer_only > generate_conformer.log 2>&1

# python "${TOP_DIR}/main.py" --path ${TAR_DIR} --molecule_num_in_cell 1 \
#      --space_group_list 14,61 --add_name XULDUD --max_workers 16 --num_generation 100 \
#      --use_conformers 4 --mode structure_only > generate_structure.log 2>&1

# opt structures using mace, --batch_size 0 means auto batch size only for mace
mkdir -p "${TAR_DIR}/mace_opt"
cd "${TAR_DIR}/mace_opt"
python "${TOP_DIR}/mace-bench/scripts/opt_batch.py" --target_folder "${TAR_DIR}/structures" \
    --molecule_single 13 --gpu_offset 0 --n_gpus 8 --num_workers 80 --batch_size 0 \
    --max_steps 3000 --filter1 UnitCellFilter --filter2 UnitCellFilter \
    --optimizer1 BFGSFusedLS --optimizer2 BFGS --num_threads 1 --cueq true \
    --use_ordered_files true --model mace > opt.log 2>&1

# opt structures using 7net
# mkdir -p "${TAR_DIR}/7net_opt"
# cd "${TAR_DIR}/7net_opt"
# python "${TOP_DIR}/mace-bench/scripts/opt_batch.py" --target_folder "${TAR_DIR}/structures" \
#     --molecule_single 13 --gpu_offset 0 --n_gpus 8 --num_workers 48 --batch_size 2 \
#     --max_steps 3000 --filter1 UnitCellFilter --filter2 UnitCellFilter \
#     --optimizer1 BFGSFusedLS --optimizer2 BFGS --num_threads 2 --cueq true \
#     --use_ordered_files true --model sevennet > opt.log 2>&1

# Postprocess the opt structures
python "${TOP_DIR}/post_process/clean_table.py"
## Make sure you have installed csd-python-api in current env before execuing following commands
# conda activate ccdc
# python "${TOP_DIR}/post_process/check_match.py" --workers 80 --timeout 20 --ref_path "${TAR_DIR}/refs"
# python "${TOP_DIR}/post_process/duplicate_remove.py" --workers 80
