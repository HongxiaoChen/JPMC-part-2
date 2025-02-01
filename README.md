# PM-HMC with NUTS and HNN

This repository extends the work presented in *"Pseudo-marginal Hamiltonian Monte Carlo"* by **Johan Alenlöv, Arnaud Doucet, and Fredrik Lindsten**. Specifically, it integrates **No-U-Turn Sampling (NUTS)** and **Hamiltonian Neural Networks (HNN)** into the pseudo-marginal Hamiltonian Monte Carlo (PM-HMC) framework.

---

## Environment

The code is written in **Python 3.8** and requires TensorFlow 2.1x, including necessary libraries and dependencies.

---

## Repository Structure

The main code files are stored in the **`codes/`** directory. Below is a detailed description of the key files and their functionality:

### Core Files
- **`generate_samples.py`**: Generates observed data for the model.
- **`log_likelihood_auto.py`**: Computes the log-likelihood and its gradient using auto-differentiation.
- **`log_likelihood_stable.py`**: Computes the log-likelihood and explicitly expresses the gradient.

### Training
- **`hnn_architectures.py`**: Defines the architecture of Hamiltonian Neural Networks (HNNs).
- **`collect_hamiltonian_trajectories.py`**: Collects Hamiltonian trajectories to generate training data for the HNN.
- **`train_hnn_hamiltonian.py`**: Trains the Hamiltonian Neural Networks.

### Sampling
- **`pm_hmc_steps.py`**: Implements PM-HMC sampling steps, including:
  - Strang splitting for solving Hamiltonian dynamics.
  - Hamiltonian computation.
  - Metropolis-Hastings acceptance criterion.
- **`run_pm_hmc.py`**: Executes PM-HMC sampling.
- **`nuts_hnn_olm_complex.py`**: Contains the main structure for NUTS sampling, inspired by the PM-HMC framework.
- **`run_pm_hmc_nuts_complex.py`**: Executes NUTS sampling.

### Parameters
- **`params.py`**: Stores all the parameters required for the project.

---

## Other Folders

- **`codes/logs/`**: Contains logs generated during code execution.
- **`codes/figures/`**: Stores figures generated during the experiments.
- **`codes/ckpt/`**: Contains pre-trained weights required for the models.

---

## How to Run

1. Set up the environment using Python 3.8 and install the required libraries.
2. Navigate to the **`codes/`** folder to access the core scripts and modules.
3. Use **`generate_samples.py`** to generate observed data.
4. Train the Hamiltonian Neural Networks using the pipeline:
   - **`collect_hamiltonian_trajectories.py`** to gather training data.
   - **`train_hnn_hamiltonian.py`** to train the HNN.
5. Run PM-HMC sampling using **`run_pm_hmc.py`** or NUTS sampling using **`run_pm_hmc_nuts_complex.py`**.
6. Check logs and figures in the respective folders for results.

---

## Notes

This repository integrates methodologies for advanced Bayesian inference and Hamiltonian dynamics modeling. For any issues or questions, feel free to raise an issue or contribute to the repository.
