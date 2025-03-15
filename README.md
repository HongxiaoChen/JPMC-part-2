# PM-HMC with NUTS and HNN

This repository extends the work presented in *"Pseudo-marginal Hamiltonian Monte Carlo"* by **Johan Alenlöv, Arnaud Doucet, and Fredrik Lindsten**. Specifically, it integrates **No-U-Turn Sampling (NUTS)** and **Hamiltonian Neural Networks (HNN)** into the pseudo-marginal Hamiltonian Monte Carlo (PM-HMC) framework.

Codes and reports for part 1 of this task are in [HNN HMC](https://github.com/HongxiaoChen/JPMC-part-1-modified).

---

## Update on 16th March, 2025

This update is beyond the deadline on 13th March and only serves for making this repository more complete.

To review the actual submission on 13th March, you can click 'Commits' and then click 'browse repository at this point' (with the sign '<>' at the right side) to choose the version uploaded on 13th March.

In this update, new replication files are added:
- Run HNN-HMC sampling with TFP transition kernels using **`run_tfp_hnn_hmc_original.py`**

New Transition Kernels are put in the folder **`codes/tfp_modified_kernels`**, which are
- **`hnn_hmc_original.py`** that performs HNN-HMC .
- **`hnn_hmc_integrator_original.py`** that performs one-step leapfrog described in Dhulipala (2023).

---

## How to Run

1. Set up the environment using Python 3.11 and install TensorFlow 2.15.
2. Navigate to the **`codes/`** folder to access the core scripts and modules.
3. Train the Hamiltonian Neural Networks using
   - **`train_hnn_hamiltonian.py`** to train the HNN.
4. Use the following files to replicate results:
   - Run PM-HMC sampling with TFP transition kernels using **`run_tfp_pm_hmc.py`**
   - Run PM-NUTS sampling with TFP transition kernels using **`run_tfp_pm_nuts.py`** 
   - Run customed PM-HMC sampling using **`run_pm_hmc.py`** 
   - Run customed PM-NUTS sampling using **`run_pm_hmc_nuts_complex.py`**
   - Run customed HNN-HMC sampling using **`run_hnn_hmc.py`**
6. Check logs and figures in the respective folders for results.

---

## Running Tests

To run tests, use the command from the root directory:

```bash
python -m unittest discover -v
```


---

## Repository Structure

The main code files are stored in the **`codes/`** directory. Below is a detailed description of the key files and their functionality:

### Transition Kernels
Transition Kernels are put in the folder **`codes/tfp_modified_kernels`**, which are
- **`pm_hmc.py`** that performs PM-HMC .
- **`pm_hmc_integrator.py`** that performs strang splitting.
- **`NUTS.py`** that performs PM-NUTS.
  
### Log-probability files
- **`log_likelihood_auto.py`**: Computes the log-likelihood and its gradient using auto-differentiation.
- **`log_likelihood_stable.py`**: Computes the log-likelihood and explicitly expresses the gradient.

### Training
- **`hnn_architectures.py`**: Defines the architecture of Hamiltonian Neural Networks (HNNs).
- **`collect_hamiltonian_trajectories.py`**: Collects Hamiltonian trajectories to generate training data for the HNN.
- **`train_hnn_hamiltonian.py`**: Trains the Hamiltonian Neural Networks.

### Sampling
- **`run_tfp_pm_hmc.py`**: Executes PM-HMC sampling using modified TFP transition Kernels
- **`run_pm_hmc.py`**: Executes PM-HMC sampling, which produces identical results w.r.t PM-HMC sampling using TFP.
- **`run_hnn_hmc.py`**: Executes HNN-HMC sampling.
- **`run_tfp_pm_nuts.py`**: Executes PM-NUTS sampling using modified TFP transition Kernels
- **`run_pm_hmc_nuts_complex.py`**: Executes NUTS sampling, with extra stopping conditions and extra interface for HNN.

### Utilities
- **`pm_hmc_steps.py`**: Contains functions for performing PM-HMC sampling steps, including:
  - Strang splitting for solving Hamiltonian dynamics.
  - Hamiltonian computation.
  - Metropolis-Hastings acceptance criterion.
- **`nuts_hnn_olm_complex.py`**: Contains the main structure for NUTS sampling, inspired by the PM-HMC framework.
- **`generate_samples.py`**: Generates observed data for the model.

### Parameters
- **`params.py`**: Stores all the parameters required for the project.

---

## Other Folders

- **`codes/logs/`**: Contains logs generated during code execution.
- **`codes/figures/`**: Stores figures generated during the experiments.
- **`codes/ckpt/`**: Contains pre-trained weights required for the models.

---

## Test Overview

The repository includes comprehensive test files to ensure the correctness, consistency, and stability of the implemented algorithms:

### 1. **test_generate_samples.py**

#### Purpose:
This file tests the functionality of Monte Carlo Hamiltonian steps and sample generation.

#### Key Tests:
- **`generate_samples` Function**: Ensures that the generated samples (`θ`, `y`, `Z`) have the correct shapes based on the provided parameters.

- **`initialize_theta` Function**:
  - Ensures the function generates initial parameters (`θ`) with the correct dimensions and consistent values when a fixed seed is used.

---

### 2. **test_log_likelihood.py**

#### Purpose:
This file tests the computation of log-likelihood and its gradients, focusing on correctness and consistency.

#### Key Tests:
- **Component Functions**:
  - Validates the shapes, types, and values of outputs for functions such as `generate_X`, `compute_logg`, `compute_logf_components`, and `compute_logq`.
- **Normalized Weights**:
  - Ensures that computed weights are normalized and sum to 1.
- **Log Prior and Posterior**:
  - Verifies the correctness of computed log prior and posterior values.
- **Automatic vs. Manual Implementation**:
  - Compares the results of `compute_log_likelihood_and_gradients_auto` (automatic differentiation) and `compute_manual` (manual computation) for consistency.
- **Gradient Consistency**:
  - Confirms that the gradients computed via TensorFlow's `GradientTape` match those from the custom implementation.

---

### 3. **test_pm_hmc_steps.py**

#### Purpose:
This file tests the implementation of Preconditioned Monte Carlo Hamiltonian (PM-HMC) steps.

#### Key Tests:
- **Hamiltonian Steps**:
  - Ensures the correctness of input/output shapes for `full_step_A`, `full_step_B`, and `leapfrog_step`.
- **Energy Conservation**:
  - Validates that the Hamiltonian remains conserved during integration.
- **Reversibility**:
  - Confirms that forward and backward integration return to the original state.
- **Metropolis-Hastings**:
  - Tests the correctness of the accept/reject step and ensures state consistency.
- **Full PM-HMC Iteration**:
  - Runs complete iterations of PM-HMC and verifies output shapes and finite values.

---

### 4. **test_nuts.py**

#### Purpose:
This file tests the No-U-Turn Sampler (NUTS) implementation.

#### Key Tests:
- **Single Leapfrog Update**:
  - Checks the correctness of shapes and values for single update steps.
- **Tree Building**:
  - Tests the base case (`j=0`) of the `build_tree` function to ensure correctness in sampling.
- **NUTS Sampling**:
  - Validates the complete NUTS sampling process, including shapes and acceptance rates.
- **Determinism**:
  - Ensures deterministic behavior when a fixed random seed is used.

---

### 5. **test_hnn.py**

#### Purpose:
This file tests the architecture and functionality of the Hamiltonian Neural Network (HNN).

#### Key Tests:
- **Model Initialization**:
  - Verifies that the HNN can be initialized with different activation functions.
- **Forward Pass**:
  - Tests the forward pass in both training and inference modes, ensuring correct output shapes and finite values.
- **Gradient Computation**:
  - Validates the gradients of the Hamiltonian with respect to `θ` and `ρ`.
- **Batch Consistency**:
  - Confirms model behavior remains consistent across different batch sizes.
- **Dropout Behavior**:
  - Tests dropout functionality in training and inference modes to ensure variability during training and consistency during inference.

---

### 6. **test_hnn_training_data.py**

#### Purpose:
This file tests the collection of training data for the HNN using Hamiltonian trajectories.

#### Key Tests:
- **Single Trajectory Collection**:
  - Verifies the correctness of shapes and values for collected trajectory data.
- **Chain Samples Collection**:
  - Tests the collection of samples along a Markov chain.
- **Full Training Data Collection**:
  - Validates the entire process of generating training datasets, ensuring correct shapes and inclusion of all required components.

---

### 7. **test_hnn_hmc.py**

#### Purpose:
This file tests the integration of HNN with Hamiltonian Monte Carlo (HMC) sampling.

#### Key Tests:
- **Directory Setup**:
  - Ensures the required directories for logging and figures are correctly created.
- **Parameter Trace Plotting**:
  - Tests the generation of parameter trace plots during sampling.
- **Leapfrog Integration**:
  - Validates the correctness of the leapfrog integration process.
- **Metropolis Step**:
  - Tests the accept/reject step for HMC sampling.
- **Full HNN-HMC Sampling**:
  - Runs the entire HNN-HMC sampling process and verifies the outputs, including posterior means and acceptance rates.
- **Hamiltonian Conservation**:
  - Tests energy conservation during HMC sampling with varying step sizes.

---

### 8. **test_tfp_pm_hmc.py** and **test_tfp_pm_nuts.py**

#### Purpose:
These two files test the sampling using Tensorflow Probability transition kernels.

Core functions are tested to ensure the sampling can perform.

---
