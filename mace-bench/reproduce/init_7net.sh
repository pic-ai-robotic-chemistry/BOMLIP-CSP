

pip install torch_scatter==2.1.2+pt24cu121 -f https://pytorch-geometric.com/whl/torch-2.4.0+cu121.html
pip install torch_sparse==0.6.18+pt24cu121 -f https://pytorch-geometric.com/whl/torch-2.4.0+cu121.html 
pip install torch_spline_conv==1.2.2+pt24cu121 -f https://pytorch-geometric.com/whl/torch-2.4.0+cu121.html
pip install -r requirements.txt
pip install -e 3rdparty/SevenNet
pip install -e .
pip install ase==3.23.0
pip install ninja
pip install rdkit==2024.3.5