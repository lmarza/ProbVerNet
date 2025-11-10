# Reproducing Results for *"On the Probabilistic Learnability of Compact Neural Network Preimages Bounds"* AAAI 2026

This repository includes the code to reproduce results for the following methods:
- **PREMAP** (Zhang et al. 2025)
- **ε-ProVe** (Marzari et al. 2024)
- **RF-ProVe**

It consists of two main folders:
- `RF-ProVe/`
- `PreimageApproxForNNs/`

> ℹ️ For the **Exact method** from Matoba and Fleuret (2020), please refer to the original publication and implementation.

---

## 1. Installation

To install all required dependencies, create a virtual environment using your preferred tool (e.g., Conda, `venv`, etc.).

### Example using pip:

cd PreimageApproxForNNs
> pip install -r requirements.txt

### Troubleshooting

- If installation fails due to a specific package (e.g., `pplpy`):
  - Try editing the `requirements.txt` to specify a compatible version.
  - Alternatively, install all other packages first, then install the problematic one manually.

- For more information, refer to the original PREMAP repository:  
  👉 https://github.com/Zhang-Xiyue/PreimageApproxForNNs

---

## 2. Reproducing PREMAP Results

Once your environment is set up:

> conda activate premap  #or your chosen environment

> pip install scikit-learn

> cd src

Then, open `preimage_main.py` and **uncomment** the desired property at lines 472–479:

```python
# Cartpole
vnnlib_all[0][0][0][3] = [-2.0, 0.0]

# LunarLander
#vnnlib_all[0][0][0][3] = [-4.0, 0.0]

# DubinsRejoin
#vnnlib_all[0][0][0][5] = [-0.3, 0.3]
```

Now run the script with the corresponding configuration file. Example for Cartpole:

> python preimage_main.py --config preimg_configs/cartpole.yaml

and similarly for the others.

---

## 3. Reproducing ε-ProVe and RF-ProVe Results

With the same environment active, navigate to the `RF-ProVe` directory:

cd ../../RF-ProVe

### Cartpole
```python
> python rf_prove.py --config_path "configs_rf_prove/cartpole.yaml" # for RF-ProVE

> python e_prove.py --config_path "configs_e_prove/cartpole.yaml" # for ε-ProVE
```

### LunarLander
```python
> python rf_prove.py --config_path "configs_rf_prove/lunarlander.yaml"

> python e_prove.py --config_path "configs_e_prove/lunarlander.yaml"
```
### DubinsRejoin
```python
> python rf_prove.py --config_path "configs_rf_prove/dubinsrejoin.yaml"

> python e_prove.py --config_path "configs_e_prove/dubinsrejoin.yaml"
```
### VCAS
```python
> python rf_prove_vcas.py --config_path "configs_rf_prove/vcas.yaml"

> python e_prove_vcas.py --config_path "configs_e_prove/vcas.yaml"
```
