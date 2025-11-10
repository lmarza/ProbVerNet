# ϵ-ProVe

This is the Python implementation of ϵ-ProVe, a tool for the *AllDNN-Verification Problem* (i.e., the problem of computing the set of all the areas that do not result in a violation for a given DNN and a safety property). ϵ-ProVe is an approximation approach that provides  provable (probabilistic) guarantees on the returned areas. 

## Abstract
Identifying safe areas is a key point to guarantee trust for systems that are based on Deep Neural Networks (DNNs). To this end, we introduce the AllDNN-Verification problem: given a safety property and a DNN, enumerate the set of all the regions of the property input domain which are safe, i.e., where the property does hold. Due to the #P-hardness of the problem, we propose an efficient approximation method called ε-ProVe. Our approach exploits a controllable underestimation of the output reachable sets obtained via statistical prediction of tolerance limits, and can provide a tight —with provable probabilistic guarantees— lower estimate of the safe areas. Our empirical evaluation on different standard benchmarks shows the scalability and effectiveness of our method, offering valuable insights for this new type of verification of DNNs.

## Installation and Setup

eProVe is tested on Python 3.8+ and PyTorch 2.x. It can be installed
easily into a conda environment. If you don't have conda, you can install
[miniconda](https://docs.conda.io/en/latest/miniconda.html).

```bash
# Remove the old environment, if necessary.
conda deactivate; conda env remove -n eProVe
# Create a new conda environment
conda create -n eProVe python=3.11
# Activate the environment
conda activate eProVe
# Install PyTorch and TQDM
pip3 install torch torchvision torchaudio
pip3 install tqdm
```

*NB: if you want to install a specific version of PyTorch (e.g., GPU, Windows, or Mac), please refer to the official installation page ([here](https://pytorch.org)).*

## Reproduce the results of the paper
To run the default experiments and reproduce the results described in the paper, run the following command:
```bash
conda activate eProVe
python3 main_paper.py
```

The results will be stored in a csv file named *'full_results.csv'*.

## Custom properties
To run a custom property on a custom neural network, please modify the main file at *'main.py'*, copy your torch model inside the folder *'models/'* and run the command:
```bash
conda activate eProVe
python3 main.py
```

### Parameters
eProVe will use the default parameters for the analysis; you can change them when creating the eProVe object: 
```python
from scripts.eProVe import eProVe
prove = eProVe(network, domain, point_cloud=3500, *args)
```

List of the available parameters (with the default parameters):
```python
point_cloud=3500
max_depth=18 
```

### Heuristics
eProVe will use the default heuristics for the analysis; you can change them when creating the eProVe object: 
```python
from scripts.eProVe import eProVe
prove = eProVe( network, domain, split_node_heu="distr", *args  )
```

List of the available heuristics (with the default parameters):
```python
split_node_heu = "distr"
split_pos_heu = "distr"
```

Following a list of the valid options, please refer to the main paper [1] for a complete description of the approaches:
```python
#####
## split_node_heu: to decide which node should be split for the iterative refinement loop.
#####
split_node_heu = "rand" # i.e., random choice
split_node_heu = "size" # i.e., always select the node with the largest interval size
split_node_heu = "distr" # i.e., select the node that maximizes the unsafe portion of the area (please refer to the main paper for details [1])

#####
## split_pos_heu: to decide where the selected node should be split.
#####
split_pos_heu = "median" # i.e., select the median value based on the point cloud collected
split_pos_heu = "mean" # i.e., always split the interval into two equal portions
split_pos_heu = "distr" # i.e., select the position in the interval that maximizes the unsafe portion of the area (please refer to the main paper for details [1])
```

### Models
All the neural networks to be verified must be in PyTorch standard format. The models must have a single output, a positive output means that the neural network is safe. Please refer to the main paper for a detailed explanation of how a general safety property can be converted into the correct format.


## Contributors
*  **Luca Marzari** - luca.marzari@univr.it
*  **Davide Corsi**
