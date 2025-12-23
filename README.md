
# PT-LiRPA

We present **P**robabilistically **T**ightened **Li**near **R**elaxation-based **P**erturbation **A**nalysis ($\texttt{PT-LiRPA}$), a novel framework that combines over-approximation techniques from LiRPA-based approaches with a sampling-based method to compute tight intermediate reachable sets. In detail, we show that with negligible computational overhead, $\texttt{PT-LiRPA}$ exploiting the estimated reachable sets, significantly tightens the lower and upper linear bounds of a neural network's output,  reducing the computational cost of formal verification tools while providing probabilistic guarantees on verification soundness. Extensive experiments on standard formal verification benchmarks, including the International Verification of Neural Networks Competition, show that our $\texttt{PT-LiRPA}$-based verifier improves robustness certificates, i.e., the certified lower bound of $\varepsilon$ perturbation tolerated by the models, by up to 3.31X and 2.26X compared to related work. Importantly, our probabilistic approach results in a valuable solution for challenging competition entries where state-of-the-art formal verification methods fail, allowing us to provide answers with high confidence (i.e., at least 99%). 

## Getting Started

$\texttt{PT-LiRPA}$ is strongly based and directly integrated in $\alpha,\beta$-CROWN (https://github.com/Verified-Intelligence/alpha-beta-CROWN). To use our probabilistic version of the tool, we provide a modified version of Zhang et al. code directly in this repo.


**Installation**: Clone the original repository (https://github.com/lmarza/ProbVerNet) and follow the setup instructions provided  there. $\texttt{PT-LiRPA}$ is tested on Python 3.10 and CUDA 11.8. It can be installed easily into a conda environment. If you don't have anaconda, you can install it from here [miniconda](https://docs.conda.io/en/latest/miniconda.html). The code reported in this repo works with the **prob-ver** conda environment created in the main folder. 

All the data are collected on a cluster running Rocky Linux 9.34 equipped with Nvidia RTX A6000 (48 GiB) and a CPU AMD Epyc 7313 (16 cores). 


## PT-LiRPA for max tolerated $\varepsilon$-pertubation computation

For the comparison with CROWN, PROVEN and Randomized smoothing, you can follow these instructions.

```bash
conda activate prob-ver
cd PT-LiRPA/alpha-beta-CROWN/complete_verifier
```

Inside ```exp_configs/max_eps_perturbation``` we have already provided the configuration.yaml both for mnist and cifar, to select the model names and seeds so that the paper results can be reproduced. In detail, you have to specify the model name at line 7 and the vnnlib_path of the based property to verify. 
```python
name_model in ["mnist_2layer_relu_1024", "mnist_3layer_relu_1024", "mnist_2layer_tanh_1024", "mnist_3layer_tanh_1024", "mnist_4layer_sigmoid_1024"]
```

and update the corresponding ```idxs_images``` field at line 16.
To run the experiment with CROWN on models trained in dataset_name 

```bash
python abcrown_max_perturbation.py --config exp_configs/max_eps_perturbation/mnist.yaml 
```

To run the experiment with CROWN+PT-LiRPA on models trained in dataset_name.
```bash
python abcrown_max_perturbation.py --config exp_configs/max_eps_perturbation/mnist.yaml --use_pt_lirpa=True	
```



## PT-LiRPA in the VNN-COMP

For the VNN-COMP experiments, you have to first unzipped all the models in the ```PT-LiRPA/vnncomp2022_benchmarks``` and ```PT-LiRPA/vnncomp2023_benchmarks```.  Then:

```bash
conda activate prob-ver
cd PT-LiRPA/alpha-beta-CROWN/complete_verifier
```
And run one of the following:

```bash
python abcrown.py --config exp_configs/vnncomp23/acasxu.yaml
```
To run the experiment with $\alpha,\beta$-CROWN on the property $\phi_2$ of the ACAS XU challenge
```bash
python abcrown.py --config exp_configs/vnncomp23/acasxu.yaml --use_pt_lirpa True --use_pt_attack True
```
for $\alpha,\beta$-CROWN enhanced with $\texttt{PT-LiRPA}$.

Similarly for the other benchmarks. In the ```PT-LiRPA/job``` folder we have also provided the bash scripts to run the verification on slurm-based cluster.


### ⚠️ Troubleshooting

As previously mentioned, we use an Nvidia RTX A6000 (48 GiB) in our experiments. Any potential memory errors from using different hardware can be addressed by changing the number of samples used in $\texttt{PT-LiRPA}$ to compute intermediate reachable sets in ```alpha-beta-CROWN/complete_verifier/utils_cifar.py``` and other utils files.

```python
def IBP_batch_wilks_cifar_w_error(model, lower, upper, batch_size=350_000, beta=0.85, p=0.01, device='cpu')
```
```python
def  IBP_batch_wilks_cifar(model, lower, upper, batch_size=350_000, device='cpu')
```
```python
def pt_lirpa_attack(model_ori, data_min, data_max, sample_size=100_000, idx_prop_to_verify=1, device='cpu')
```

Specifically the `batch_size` and `sample_size` parameters.
