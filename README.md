# BOMLIP-CSP

An open-source Python framework that integrates machine learning interatomic 
potentials (MLIPs) with a tailored batched optimization strategy, enabling rapid, 
unbiased structure prediction across the full density range

## Perform the complete CSP process

```sh
git clone https://github.com/pic-ai-robotic-chemistry/BOMLIP-CSP.git --recursive && cd BOMLIP-CSP

conda create -n BOMLIP_CSP python=3.10 -y && conda activate BOMLIP_CSP
cd BOMLIP-CSP
top_dir=$(pwd)
cd $top_dir/mace-bench
./reproduce/init_mace.sh && source util/env.sh
sudo ./util/mps_start.sh

cd $top_dir
./csp.sh

sudo ./util/mps_clean.sh
```
## Reproduce mace batch opt speedup.

```sh
#!/bin/bash

git clone https://github.com/pic-ai-robotic-chemistry/BOMLIP-CSP.git --recursive && cd BOMLIP-CSP
conda create -n BOMLIP_CSP python=3.10 -y && conda activate BOMLIP_CSP
cd BOMLIP-CSP/mace-bench

# initialize mace env.
./reproduce/init_mace.sh && source util/env.sh
sudo ./util/mps_start.sh
cd reproduce

# run baseline sub-test
./subtest_baseline.sh

# run baseline mixed test
cd perf_v2_base
./run_mace.sh

# run BOMLIP_CSP sub-test
cd ../
./subtest.sh

# run BOMLIP_CSP mixed test
cd perf_v2_batch
./opt.sh

# clean mps
./util/mps_clean.sh

```

## If you want to configure the 7net environment.

```sh
#!/bin/bash
conda create -n 7net-cueq python=3.10 -y && conda activate 7net-cueq
./reproduce/init_7net.sh && source util/env.sh

# Use a fixed batch size for structural optimization
python ../../scripts/mace_opt_batch.py --target_folder "../../data/perf_v2" \ 
    --molecule_single 46 --gpu_offset 0 --n_gpus 4 --num_workers 4 \
    --batch_size 2 --max_steps 3000 --filter1 UnitCellFilter \
    --filter2 UnitCellFilter --optimizer1 BFGSFusedLS --optimizer2 BFGS \
    --num_threads 2 --cueq true --use_ordered_files true --model sevennet
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Third-party Dependencies

This project includes dependencies with various licenses:
- **MACE**: MIT License (compatible)
- **FairChem**: MIT License (compatible)
- **SevenNet**: GPL v3 License (Note: GPL is a copyleft license)

### License Compatibility Notice

**Important**: This project can run completely without relying on SevenNet. 
This project includes SevenNet as an optional dependency, which is licensed under GPL v3.
If you use SevenNet functionality, you should be aware of the GPL licensing requirements.
For commercial use or to avoid GPL restrictions, consider using only the MACE calculator 
functionality.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{BOMLIP_CSP,
  author = {Chengxi Zhao, Zhaojia Ma, Dingrui Fan},
  title = {BOMLIP_CSP: Integrating machine learning interatomic potentials with batched optimization for crystal structure prediction},
  year = {2025},
  url = {https://github.com/pic-ai-robotic-chemistry/BOMLIP-CSP}
}
```