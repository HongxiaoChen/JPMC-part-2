import tensorflow as tf
import time
from datetime import datetime
import logging
from pathlib import Path
import os
import matplotlib.pyplot as plt
import numpy as np
from hnn_architectures import HNN
from params import MCMC_PARAMS, HNN_WEIGHTS_PATH, initialize_theta, DATA_PARAMS
from generate_samples import generate_samples
from tensorflow_probability.python.mcmc import sample_chain
from tfp_modified_kernels.hnn_hmc_original import HNNHamiltonianMonteCarlo


def setup_logger():
    """Set up a logger for training process"""
    log_dir = Path('log')
    log_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'hnn_hmc_tfp_{timestamp}.log'

    logger = logging.getLogger('HNN_HMC_TFP')
    logger.setLevel(logging.INFO)

    if logger.handlers:
        logger.handlers.clear()

    fh = logging.FileHandler(log_file)
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(message)s')

    for handler in [fh, ch]:
        handler.setLevel(logging.INFO)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def ensure_directories():
    """Ensure necessary directories exist"""
    directories = ['log', 'figures']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")


def plot_parameter_traces(samples, h, L, rho_size):
    """
    Plot parameter trace plots with sampling parameters information.

    Args:
        samples: numpy array of shape [M-burn_in, 13] containing samples after burn-in
        h: step size
        L: number of leapfrog steps
        rho_size: rho size parameter
    """
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Left plot: trajectories of mu1 and mu2
    ax1.plot(samples[:, 8], color='blue', alpha=0.7, label='μ₁')
    ax1.plot(samples[:, 9], color='red', alpha=0.7, label='μ₂')
    ax1.axhline(y=np.mean(samples[:, 8]), color='blue', linestyle='--', alpha=0.5)
    ax1.axhline(y=np.mean(samples[:, 9]), color='red', linestyle='--', alpha=0.5)
    ax1.set_title('μ₁ and μ₂')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Value')
    ax1.set_ylim(-2, 6)
    ax1.legend()

    # Right plot: trajectories of 1/lambda1 and 1/lambda2
    inv_lambda1 = 1.0 / np.exp(samples[:, 10])
    inv_lambda2 = 1.0 / np.exp(samples[:, 11])

    mean_inv_lambda1 = np.mean(inv_lambda1)
    mean_inv_lambda2 = np.mean(inv_lambda2)

    ax2.plot(inv_lambda1, color='blue', alpha=0.7, label='1/λ₁')
    ax2.plot(inv_lambda2, color='red', alpha=0.7, label='1/λ₂')
    ax2.axhline(y=mean_inv_lambda1, color='blue', linestyle='--', alpha=0.5)
    ax2.axhline(y=mean_inv_lambda2, color='red', linestyle='--', alpha=0.5)
    ax2.set_title('1/λ₁ and 1/λ₂')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Value')
    ax2.set_ylim(0, 10)
    ax2.legend()

    # Set title with parameter information
    plt.suptitle(
        'HNN-HMC (TFP) - '
        'Traces for parameters μ₁ and μ₂ (left) and 1/λ₁ and 1/λ₂ (right)\n'
        f'HNN-HMC sampler with h = {h}, L = {L}\n'
        f'Total steps per iteration = {h * L:.4f}'
    )

    # Save figure
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.savefig(f'figures/parameter_traces_tfp_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()


def run_hnn_hmc_tfp(seed=42):
    """sample using TFP-HNN-HMC
    
    Args:
        seed: random seed, default is 42
    """
    # set up logger
    logger = setup_logger()
    ensure_directories()
    
    # use the fixed random seed
    logger.info(f"Using fixed random seed: {seed}")
    tf.random.set_seed(seed)
    np.random.seed(seed)

    # generate true data for comparison
    T = DATA_PARAMS['T']
    n = DATA_PARAMS['N_SUBJECTS']
    num_features = DATA_PARAMS['NUM_FEATURES']
    Y, X, Z, beta_true = generate_samples(T, n, num_features)

    # load model and weights
    model = HNN(activation='sin')
    model.load_weights(HNN_WEIGHTS_PATH)

    # get parameters
    M = MCMC_PARAMS['M']
    h = MCMC_PARAMS['H']
    L = MCMC_PARAMS['L']
    burn_in = MCMC_PARAMS['BURN_IN']
    rho_size = MCMC_PARAMS['RHO_SIZE']

    current_theta = initialize_theta()

    # create HNN-HMC kernel
    hnn_hmc_kernel = HNNHamiltonianMonteCarlo(
        hnn_model=model,
        step_size=h,
        num_leapfrog_steps=L,
        rho_size=rho_size
    )

    # record start time
    start_time = time.time()

    logger.info("Starting HNN-HMC-TFP sampling")
    logger.info(f"Parameters: M={M}, h={h}, L={L}, burn_in={burn_in}, rho_size={rho_size}")

    # use TFP's sample_chain to execute sampling
    @tf.function
    def run_chain():
        samples = sample_chain(
            num_results=M,
            current_state=current_theta,
            kernel=hnn_hmc_kernel,
            num_burnin_steps=burn_in,
            trace_fn=None,
            return_final_kernel_results=False,
            seed=seed
        )
        return samples

    # execute sampling
    samples = run_chain()
    
    # convert to NumPy array for calculation
    samples_np = samples.numpy()
    samples_after_burnin = samples_np

    # calculate the posterior mean
    posterior_mean_beta = np.mean(samples_after_burnin[:, :8], axis=0)
    posterior_mean_mu = np.mean(samples_after_burnin[:, 8:10], axis=0)

    # for lambda, calculate the inverse mean
    inv_lambda1 = 1.0 / np.exp(samples_after_burnin[:, 10])
    inv_lambda2 = 1.0 / np.exp(samples_after_burnin[:, 11])
    posterior_mean_lambda = np.array([
        1.0 / np.mean(inv_lambda1),
        1.0 / np.mean(inv_lambda2)
    ])

    posterior_mean_w = 1.0 - tf.sigmoid(np.mean(samples_after_burnin[:, 12])).numpy()

    # final statistics
    total_time = time.time() - start_time

    # record results
    logger.info("\nSampling completed:")
    logger.info(f"Total samples: {M} (including {burn_in} burn-in samples)")
    logger.info(f"Samples after burn-in: {M - burn_in}")
    logger.info(f"Total time: {total_time / 60:.1f}min")

    logger.info("\nParameter estimates:")
    logger.info(f"Beta (true): {beta_true.numpy().tolist()}")
    logger.info(f"Beta (estimated): {posterior_mean_beta.tolist()}")
    logger.info(f"mu1: {posterior_mean_mu[0]}")
    logger.info(f"mu2: {posterior_mean_mu[1]}")
    logger.info(f"lambda1: {posterior_mean_lambda[0]}")
    logger.info(f"lambda2: {posterior_mean_lambda[1]}")
    logger.info(f"w1: {posterior_mean_w}")

    # produce plots
    logger.info("\nGenerating parameter trace plots...")
    plot_parameter_traces(samples_after_burnin, h, L, rho_size)
    logger.info("Parameter trace plots have been saved to the figures directory.")


    return (samples_after_burnin, 
            posterior_mean_beta, posterior_mean_mu,
            posterior_mean_lambda, posterior_mean_w, logger)


if __name__ == "__main__":

    (samples, 
     posterior_mean_beta, posterior_mean_mu,
     posterior_mean_lambda, posterior_mean_w, logger) = run_hnn_hmc_tfp()

    print("\nHNN-HMC-TFP sampling completed:")
    print("\nParameter estimates:")
    print("Beta:", posterior_mean_beta)
    print("mu1:", posterior_mean_mu[0])
    print("mu2:", posterior_mean_mu[1])
    print("lambda1:", posterior_mean_lambda[0])
    print("lambda2:", posterior_mean_lambda[1])
    print("w1:", posterior_mean_w) 