
------

# Topological Feature Reconstruction (TFR)

Official code for Paper "Learning from Graph: Mitigating Label Noise on Graph through Topological Feature Reconstruction".
We implemented TFR using the framework provided by [NoisyGL](https://proceedings.neurips.cc/paper_files/paper/2024/hash/436ffa18e7e17be336fd884f8ebb5748-Abstract-Datasets_and_Benchmarks_Track.html). You can find implementations and config file of most baseline methods in [their repository](https://github.com/eaglelab-zju/NoisyGL).

## Installation
**Note:** These codes built upon [PyTorch](https://pytorch.org/), [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html), [PyTorch Sparse](https://github.com/rusty1s/pytorch_sparse) and [PyTorch Cluster](https://github.com/rusty1s/pytorch_cluster). 
Please install them from the above links for running NoisyGL. Also, please make sure that you have installed the following dependencies.

## Required Dependencies:
- Python 3.11+
- torch>=2.1.0
- pyg>=2.5.0
- torch_sparse>=0.6.18
- torch_cluster>=1.6.2
- pandas
- scipy
- scikit-learn
- ruamel 
- ruamel.yaml
- nni
- matplotlib
- numpy
- xlsxwriter

## Quick Start
###  Run comprehensive benchmark.
``` bash
python total_exp.py --runs 10 --methods tfr_gcn --datasets dblp --noise_type clean uniform --noise_rate 0.1 0.2 --device cuda:0 --seed 3000
```
By running the command above, "TFR+GCN" will be tested 
on dblp dataset under different types and rates of label noise.
Each experiment will run 10 times and the total results will be saved at ./log and named by the current timestamp.
You can customize the combination of method, data, noise type, and noise rate by changing the corresponding arguments.

###  Run single experiment.
``` bash
python single_exp.py --method tfr_gcn --data dblp --noise_type uniform --noise_rate 0.1 --device cuda:0 --seed 3000
```
This command runs a single experiment in debug mode and is usually used for debugging. 
By running this, detailed experiment information will be printed on the terminal, which can be used to locate the problem.

**Method available** :
`tfr_gcn`, `tfr_gin`, `tfr_sage` 

**Dataset available** :
 `dblp`, `blogcatalog`, `flickr`,  `roman-empire`, `ogbn-arxiv`, 

**noise type** ： 
`clean`, `pair`, `uniform`, `instance_dependent`




