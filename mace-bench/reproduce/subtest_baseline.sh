#!/bin/bash

top_dir=$(pwd)

natoms_nw_bs=(
    "92 64"
    "184 64"
    "368 64"
)

for config in "${natoms_nw_bs[@]}"; do
    read natoms nw <<< "$config"

    dir="$top_dir/subtest_BASE_${natoms}_g4_j${nw}"
    mkdir -p "$dir"
    cd "$dir" || continue

    pwd

    python ../mace_opt_new.py --n_jobs ${nw} --molecule_single 46 \
        --target_folder ../../data/perf_v2_sorted/perf_v2_${natoms}/ --model small --n_gpus 4 \
        --gpu_offset 0 --optimizer1 QuasiNewton --filter1 UnitCellFilter \
        --filter2 UnitCellFilter --max_steps 3000 > opt.log 2>&1
done