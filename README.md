# Dir-GSN

**Recommended setup installations**:
```
conda create -n dir_gsn python=3.10
conda activate dir_gsn 
conda install pytorch==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-2.1.0+cu121.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-2.1.0+cu121.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-2.1.0+cu121.html
pip install torch-geometric
conda install -c conda-forge graph-tool
pip install ogb
pip install tqdm
pip install wandb

```
