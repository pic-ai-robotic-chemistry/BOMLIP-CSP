
rm -r *_result_*

python ../../scripts/opt_batch.py --target_folder "../../data/perf_v2" --molecule_single 46 --gpu_offset 0 --n_gpus 4 --num_workers 40 --batch_size 0 \
    --max_steps 6000 --filter1 UnitCellFilter --filter2 UnitCellFilter --optimizer1 BFGSFusedLS --optimizer2 BFGS --num_threads 2 --cueq true --use_ordered_files true