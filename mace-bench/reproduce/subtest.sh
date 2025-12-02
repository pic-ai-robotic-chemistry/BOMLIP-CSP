

top_dir=$(pwd)

natoms_nw_bs=(
    "92 48 25"
    "184 40 12"
    "368 40 5"
)

for config in "${natoms_nw_bs[@]}"; do
    read natoms nw bs <<< "$config"

    dir="$top_dir/subtest_BATCH_${natoms}_g4_j${nw}_bs${bs}_cueq_cupbc"
    mkdir -p "$dir"
    cd "$dir" || continue

    pwd
    python ../../scripts/opt_batch.py \
        --target_folder "../../data/perf_v2_sorted/perf_v2_${natoms}" \
        --molecule_single 46 --gpu_offset 0 --n_gpus 4 --num_workers ${nw} --batch_size ${bs} \
        --max_steps 6000 --filter1 UnitCellFilter --filter2 UnitCellFilter \
        --optimizer1 BFGSFusedLS --optimizer2 BFGS --num_threads 2 \
        --use_ordered_files true --cueq true > opt.log 2>&1
done