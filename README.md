# ProbVerNet 🚀
A unified deep neural networks probabilistic verification framework. ProbVerNet provides a suite of tools to both probabilistically verify the neural network's robustness to adversarial pertubation and approximately compute the preimage bounds of neural networks, i.e., to enumerate all the input regions where a specified property holds. The framework is currently under construction. 


## Available Algorithms ✅


- [ ] PT-LiRPA [A]: Probabilistically Tightened Linear Relaxation-based Perturbation Analysis, a novel probabilistic framework that combines over-approximation techniques from LiRPA-based approaches with a sampling-based method to compute tight intermediate reachable sets, significantly tightening the lower and upper linear bounds of a neural network's output and reducing the computational cost of formal verification tools while providing probabilistic guarantees on verification soundness. (coming soon).
- [ ] CountingProVe [B,C]: Approximate count method with probabilistic guarantees on the interval of violation rate present in the property's domain (coming soon).
- [x] ϵ-ProVe [C]: Efficient approximate enumeration strategy with tight probabilistic guarantees for enumerating all the (un)safe regions of the property's domain for a given safety property. 
- [x] RF-ProVe[D]: Compact probabilistic enumeration of preimage bounds of neural networks with guarantees on the coverage and bounded error of the solution returned.

## Installation 

```bash
git clone https://github.com/lmarza/ProbVerNet.git
cd ProbVerNet
conda env create -f environment.yml
conda activate prob-ver
```

### Troubleshooting ⚠️
  - If installation fails due to a specific package (e.g., `pplpy`):
  - Try editing the `environment.yml` to specify a compatible version.
  - Alternatively, install all other packages first, then install the problematic one manually.



## Definition of the properties
Properties can be defined with 3 different formulations:

#### Reachable sets
Following the definition of rechable set [Liu et al.], given an precondition on the input X and a desired reachable set Y, the property is verified if for each *x* in X, it follows that N(x) is in Y *(i.e., if the input belongs to the interval X, the output must belong to the interval Y)*.
```python
property = {
	"X" : [[0.1, 0.34531], [0.7, 1.1]],
	"Y" : [[0.0, 0.2], [0.0, 0.2]]
}
```

#### Decision
Following the definition of ACAS XU [Julian et al.], given an input property P and an output node A corresponding to an action, the property is verified if, for each *x* in P, it follows that the action A will never be selected *(i.e., if the input belongs to the interval P, the output of node A is never the one with the highest value)*.
```python
property = {
	"X" : [[0.1, 0.34531], [0.7, 1.1]],
	"Y" : 1
}
```

#### Positive 
Following the definition of α,β-CROWN [Wang et al.], given an input property P, the output of the network is non-negative *(i.e., if the input belongs to the interval P, the output of each node is greater or equals zero)*
```python
property = {
  "X" : [[0.1, 0.34531], [0.7, 1.1]],	
  "Y" : y > 0
}
```

ProbVerNet framework adopts a positive property encoding, where a multi-output node model can be seamlessly converted into a single-output node to verify the desired property (as shown below). We are working towards providing a more intuitive and straightforward way to define safety properties for the enum methods of ProbVerNet ($\varepsilon$-ProVe and RF-ProVe), with upcoming support for the VNN-Lib format as for PT-LiRPA. 

> Example of how to convert a 2-output node model, to verify the property $\mathcal{Y}: y_0 \geq y_1$
```python
# Wrap the model to override forward
class BinaryDecisionWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        return torch.where(out[:, 0] >= out[:, 1], torch.tensor(1), torch.tensor(-1))
```

## Results of the analysis 🔍
The verification analysis via ProbVerNet yields different possible outcomes depending on the selected verification type. 


**PT-LiRPA verififier:**
- maximum tolerated $\varepsilon$ input perturbation by the model under analysis.
- *SAT* (robust) if the property is respected, UNSAT (not-robust) otherwise.

**Enumeration verifiers ($\varepsilon$-ProVe and RF-ProVe):**
- preimage bounds encoded with convex polytopes (for now, only hyperrectangles)
- number of preimage bounds enumerated where the property is respected.
- Estimated coverage with respect to the target one.
- % error on the returned solution.


## Example of running the enumeration verification with our tools.

```bash
cd RF-ProVe
python rf_prove.py --config_path "configs_rf_prove/cartpole.yaml" # for RF-ProVe
```

```bash
cd eProVe
python e_prove.py --config_path "configs_e_prove/cartpole.yaml" # for ε-ProVe
```



## Reference 📚
If you use our probabilistic framework in your work, please kindly cite our papers:


**[A]** [Probabilistically Tightened Linear Relaxation-based Perturbation Analysis for Neural Network Verification](https://arxiv.org/pdf/2507.05405).  Marzari L., Cicalese F. and Farinelli A. Under review JAIR 2025
```
@article{marzari2025probabilistically,
  title={Probabilistically Tightened Linear Relaxation-based Perturbation Analysis for Neural Network Verification},
  author={Marzari, Luca and Cicalese, Ferdinando and Farinelli, Alessandro},
  journal={arXiv preprint arXiv:2507.05405},
  year={2025}
}

```
    
**[B]** [The \#DNN-Verification Problem: Counting Unsafe Inputs for Deep Neural Networks](https://dl.acm.org/doi/abs/10.24963/ijcai.2023/25).  Marzari L., Corsi D., Cicalese F. and Farinelli A. In IJCAI 2023
```
@inproceedings{marzari2023dnn,
  title={The \#DNN-Verification Problem: Counting Unsafe Inputs for Deep Neural Networks},
  author={Marzari, Luca and Corsi, Davide and Cicalese, Ferdinando and Farinelli, Alessandro},
  booktitle={Proceedings of the Thirty-Second International Joint Conference on Artificial Intelligence},
  pages={217--224},
  year={2023}
}
```

**[C]** [Scaling #DNN-Verification Tools with Efficient Bound Propagation and Parallel Computing](https://arxiv.org/pdf/2312.05890).  Marzari L., Roncolato G. and Farinelli A. In AIRO 2023
```
@incollection{marzari2023scaling,
  title={Scaling \#DNN-Verification Tools with Efficient Bound Propagation and Parallel Computing},
  author={Marzari, Luca and Roncolato, Gabriele and Farinelli, Alessandro},
  booktitle={AIRO 2023 Artificial Intelligence and Robotics 2023},
  year={2023}
}
```

**[D]** [Enumerating safe regions in deep neural networks with provable probabilistic guarantees](https://ojs.aaai.org/index.php/AAAI/article/view/30134).  Marzari L., Corsi D., Marchesini E., Farinelli A. and Cicalese F. In AAAI 2024
```
@inproceedings{marzari2024enumerating,
  title={Enumerating safe regions in deep neural networks with provable probabilistic guarantees},
  author={Marzari, Luca and Corsi, Davide and Marchesini, Enrico and Farinelli, Alessandro and Cicalese, Ferdinando},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={19},
  pages={21387--21394},
  year={2024}
}
```

**[E]** [On the Probabilistic Learnability of Compact Neural Network Preimage Bounds](https://ojs.aaai.org/index.php/AAAI/article/view/30134).  Marzari L., Bicego M., Cicalese F. and Farinelli A. In AAAI 2026
```
@inproceedings{marzari2026enum,
  title={On the Probabilistic Learnability of Compact Neural Network Preimage Bounds},
  author={Marzari, Luca and Bicego, Manuele and Cicalese, Ferdinando and Farinelli, Alessandro},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={},
  number={},
  pages={},
  year={2026}
}
```
