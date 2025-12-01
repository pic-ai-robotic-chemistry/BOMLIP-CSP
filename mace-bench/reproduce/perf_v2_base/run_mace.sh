
python ../mace_opt_new.py --n_jobs 64 --molecule_single 46 \
    --target_folder ../../data/perf_v2/ --model small --n_gpus 4 --gpu_offset 0 \
    --optimizer1 QuasiNewton --filter1 UnitCellFilter --filter2 UnitCellFilter